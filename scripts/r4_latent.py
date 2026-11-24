"""R4: latent-grid dynamics on the frozen sharp autoencoder (ivs.latentdyn; design + pre-registered gates in plan.md "R4").
Stages (each is one process; the Kaggle entry scripts/kaggle_r4.py runs them):
  ae   train the S32_ssim autoencoder (heldout_clarity arm, held-out clarity scored as in R3c) and SAVE it         -> <out>/ae/ae.pt, results.json, sheets
  dyn  train the latent dynamics on real windows + synthetic touch interventions for a wall-clock budget, then eval  -> <out>/<variant>/dyn.pt, eval.json, sheets
  eval evaluate an existing dyn.pt                                                                                 -> <out>/<variant>/eval.json, sheets
Variants: free = the model's own previous sample is its next input; proj = that sample is first re-projected through Enc(Dec(.)) (the analogue of the
UI re-perceiving its own render).
Run 2 (plan.md R4 run 2, after run 1 drifted to white beyond its 24-tick horizon): long = proj trained free-running up to 96 ticks (curriculum over half the budget, truncated BPTT every 24 steps, context length drawn from 4/8/12/16 per batch); anchor = long + the last real context latent fed to the dynamics at every tick as a reference.
Eval (all on held-out recordings, one 16-frame real context per val recording, zero action): dream --ticks ticks at temperatures 0 / 0.5 / 1, scored with the
SAME function as the L1 baseline (score_dreams = make_birth._stats at 96 px gray: eye_frac / eye_moving / still_frac / first100 / max_len), dream sharpness
vs the AE's own sharpness on real frames, touch response vs copy-last (shift_x, shift_y, gain, light_x), one-step image MSE vs copy-last and the AE floor.
CPU-aware (batch / sizes by flags; the Kaggle run only uses bigger numbers). Usage (tiny local check): python scripts/r4_latent.py dyn --cache DIR ... --minutes 1
"""
import argparse
import json
import math
import queue
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ivs import eyemetric as E                                       # noqa: E402
from ivs.actions import InterventionSampler                          # noqa: E402
from ivs.cachedata import CacheBank                                  # noqa: E402
from ivs.latentdyn import (LatentDynamics, dream, dyn_loss, load_ae, load_dyn, one_step_mse, save_dyn)   # noqa: E402
from ivs.pilot import sharpness                                      # noqa: E402
import make_birth as MB                                              # noqa: E402

SOURCE_WEIGHTS = {"video": 0.5, "openeds": 0.5}
AE_ARM = "S32_ssim"
CTX, FREE_MAX, FREE_MIN = 8, 24, 4                                   # run 1: window = CTX real frames + up to FREE_MAX free-running ones
CTX_CHOICES = (4, 8, 12, 16)                                         # run 2: context length drawn per batch (run 1 trained on 8 only and was evaluated on 16: error at tick 24 doubled)
# variant -> (reproject own sample through Enc(Dec(.)), anchor input, free-run horizon, curriculum ramp share, truncated-BPTT chunk). free/proj = run 1; long/anchor = run 2 (plan.md R4 run 2)
VARIANTS = {"free": (False, False, 24, 0.35, 0), "proj": (True, False, 24, 0.35, 0), "long": (True, False, 96, 0.5, 24), "anchor": (True, True, 96, 0.5, 24)}
EVAL_CTX, SEG_SIZE = 16, 96
TEMPS = (0.0, 0.5, 1.0)
PRIMARY_T = 0.5
SHEET_TICKS = (1, 5, 15, 30, 60, 100, 200, 299)
TOUCH = {"shift_x": (0, 1.0), "shift_y": (1, 1.0), "gain": (2, 1.0), "light_x": (3, -1.0)}   # state dim, sign of the 3-tick push (action +-1 = one sampler step)
TOUCH_TICKS, TOUCH_READ = 3, (3, 8, 16)
S0 = (0.0, 0.0, 0.0, 0.4, -0.4)                                      # static start state of the touch tests (light on, in a corner)


def pick_device(name):
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else ("cpu" if name == "auto" else name))


def banks(cache, device="cpu", size=128, splits=("train", "val")):
    return {s: CacheBank(cache, image_size=size, split=s, val_frac=0.15, seed=0, tint_mode="gray", source_weights=SOURCE_WEIGHTS if s == "train" else None,
                         device=device, sources=["video", "openeds"]) for s in splits}


# ----------------------------------------------------------------------------------------------- scoring (shared with scripts/r4_baseline.py)
def to_seg_frames(fr):
    """(T,n,3,H,W) float -> (T,n,3,96,96) so every model is scored by the segmenter at the size it was trained for."""
    if fr.shape[-1] == SEG_SIZE:
        return fr
    T, n = fr.shape[:2]
    return F.interpolate(fr.reshape(T * n, *fr.shape[2:]), size=(SEG_SIZE, SEG_SIZE), mode="area").reshape(T, n, 3, SEG_SIZE, SEG_SIZE)


def score_dreams(fr, seg):
    """Per-world statistics of a dream (same function the L1 baseline used): returns make_birth._stats dict (flags (T,n), eye_frac, max_len, n_long, still_frac, eye_moving, first100)."""
    return MB._stats(to_seg_frames(fr.cpu()), seg)


def mean_stats(st, idx=None):
    keys = ("eye_frac", "eye_moving", "still_frac", "first100", "max_len", "n_long")
    idx = list(range(len(st["eye_frac"]))) if idx is None else idx
    return {k: round(float(np.mean([st[k][i] for i in idx])), 3) for k in keys}


# ----------------------------------------------------------------------------------------------- training data
class Prefetch:
    """Background thread that builds (images, actions) training windows so the GPU never waits for CPU indexing."""

    def __init__(self, bank, batch, seed, length_fn, depth=3, ctx_choices=(CTX,)):
        self.bank, self.batch, self.length_fn, self.ctx_choices = bank, batch, length_fn, tuple(ctx_choices)
        self.rng = np.random.RandomState(seed)
        self.q, self.stop = queue.Queue(depth), False
        self.gen = torch.Generator().manual_seed(seed)
        self.isamp = InterventionSampler()
        self.t = threading.Thread(target=self.run, daemon=True)
        self.t.start()

    def run(self):
        while not self.stop:
            free, ctx = self.length_fn(), int(self.rng.choice(self.ctx_choices))
            x = self.bank.sample(self.batch, ctx + free, generator=self.gen, train=True)["images"]
            x, a = self.isamp(x, generator=self.gen)
            while not self.stop:
                try:
                    self.q.put((x, a, free, ctx), timeout=0.5)
                    break
                except queue.Full:
                    pass

    def get(self):
        return self.q.get()

    def close(self):
        self.stop = True


def lr_at(frac, lr, warm=0.02, floor=0.1):
    """Linear warm-up over the first `warm` of the budget, then cosine down to floor*lr (frac = share of the wall-clock budget used)."""
    cos = floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, frac)))
    return lr * (frac / warm if frac < warm else 1.0) * cos


def free_at(frac, ramp=0.35, free_max=FREE_MAX):
    """Free-running length of the curriculum at time fraction `frac` (FREE_MIN -> free_max over the first `ramp` of the budget)."""
    return int(round(FREE_MIN + (free_max - FREE_MIN) * min(1.0, frac / ramp)))


def train_dyn(ae, bank, out, variant, minutes, batch, lr, device, seed, hid, layers, log_every=100, max_steps=None, free_max=None):
    reproject, anchor, v_free, ramp, detach_every = VARIANTS[variant]
    free_max = free_max or v_free
    ctx_choices = (CTX,) if v_free == FREE_MAX else CTX_CHOICES
    torch.manual_seed(seed)
    z = ae.encode(bank.sample(2, 2, train=False)["images"].to(device))
    dyn = LatentDynamics(z_ch=z.shape[2], grid=z.shape[-1], hid=hid, layers=layers, anchor=anchor).to(device)
    opt = torch.optim.Adam(dyn.parameters(), lr=lr)
    n_par = sum(p.numel() for p in dyn.parameters())
    print(f"[{variant}] dynamics params {n_par:,} | latent {tuple(z.shape[2:])} = {ae.latent_numbers} numbers | batch {batch} | budget {minutes} min | reproject={reproject} anchor={anchor} free<= {free_max} ctx {ctx_choices} detach_every {detach_every}", flush=True)
    t0 = time.time()
    budget = minutes * 60.0
    state = dict(frac=0.0)
    pf = Prefetch(bank, batch, seed + 7, lambda: free_at(state["frac"], ramp, free_max), ctx_choices=ctx_choices)
    gen = torch.Generator(device="cpu").manual_seed(seed + 3)
    curve, step, run = [], 0, {}
    ckpt_t = t0
    while True:
        frac = (time.time() - t0) / budget
        if frac >= 1.0 or (max_steps and step >= max_steps):
            break
        state["frac"] = frac
        for g in opt.param_groups:
            g["lr"] = lr_at(frac, lr)
        x, a, free, ctx = pf.get()
        x, a = x.to(device), a.to(device)
        loss, parts = dyn_loss(dyn, ae, x, a, ctx, free, pix_every=max(3, free // 8), reproject=reproject, generator=None, detach_every=detach_every)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(dyn.parameters(), 1.0))
        if not math.isfinite(gn):
            print(f"[{variant}] step {step}: non-finite gradient, step skipped", flush=True)
            opt.zero_grad(set_to_none=True)
            step += 1
            continue
        opt.step()
        step += 1
        for k, v in dict(loss=float(loss.detach()), gn=gn, **parts).items():
            run[k] = run.get(k, 0.0) + v
        if step % log_every == 0:
            row = {k: v / log_every for k, v in run.items()}
            row.update(step=step, free=free, ctx=ctx, lr=lr_at(frac, lr), min=round((time.time() - t0) / 60, 1))
            curve.append(row)
            print(f"[{variant}] " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
            run = {}
        if time.time() - ckpt_t > 600:
            save_dyn(Path(out) / "dyn.pt", dyn, meta=dict(variant=variant, step=step, minutes=(time.time() - t0) / 60))
            ckpt_t = time.time()
    pf.close()
    save_dyn(Path(out) / "dyn.pt", dyn, meta=dict(variant=variant, step=step, minutes=(time.time() - t0) / 60, batch=batch, lr=lr, hid=hid, layers=layers, free_max=free_max, curve=curve))
    print(f"[{variant}] trained {step} steps in {(time.time() - t0) / 60:.1f} min", flush=True)
    return dyn, dict(steps=step, minutes=round((time.time() - t0) / 60, 1), params=n_par, curve=curve)


# ----------------------------------------------------------------------------------------------- evaluation
def touch_test(dyn, ae, ctx_real, reproject, device):
    """Constant-state context (touch state S0 applied to the real context), then a 3-tick push of one state dimension with action +-1.
    Truth = the same real last frame under the new state (ivs.actions.apply_state); baselines: copy-last (what a model that ignores touch shows)
    and the dream itself. Reported per kind and read tick: mse(dream, truth) / mse(copy_last, truth) (< 1 = the touch moved the picture the right way)."""
    isamp = InterventionSampler()
    b, tc = ctx_real.shape[:2]
    ticks = max(TOUCH_READ)
    s0 = torch.tensor(S0).view(1, 1, 5).expand(b, 1, 5)
    ctx_i = isamp.apply_state(ctx_real, s0.expand(b, tc, 5))
    base = ctx_real[:, -1:]
    out = {}
    step_unit = isamp.step * isamp.limit / isamp.limit[0]
    for name, (dim, sign) in TOUCH.items():
        act = torch.zeros(b, ticks, 5)
        act[:, :TOUCH_TICKS, dim] = sign
        states = s0.expand(b, ticks, 5).clone()
        for t in range(ticks):
            states[:, t] = (states[:, t - 1] if t else s0[:, 0]) + act[:, t] * step_unit
        states = torch.max(torch.min(states, isamp.limit), -isamp.limit)
        truth = isamp.apply_state(base.expand(b, ticks, *base.shape[2:]), states)
        copy = ctx_i[:, -1:].expand(-1, ticks, -1, -1, -1)
        fr = dream(dyn, ae, ctx_i.to(device), ticks, temperature=0.0, reproject=reproject, actions=act.to(device)).cpu().transpose(0, 1)    # B,ticks,3,H,W
        row = {}
        for r in TOUCH_READ:
            m_d = float(((fr[:, r - 1] - truth[:, r - 1]) ** 2).mean())
            m_c = float(((copy[:, r - 1] - truth[:, r - 1]) ** 2).mean())
            row[str(r)] = dict(dream=m_d, copy_last=m_c, ratio=m_d / max(m_c, 1e-9))
        out[name] = row
        out[name]["_frames"] = (ctx_i[:, -1], truth, fr)
    return out


def tile_row(imgs, flags, size):
    return np.concatenate([MB.tile(im, f, "", size) for im, f in zip(imgs, flags)], 1)


def eval_variant(dyn, ae, val_bank, seg, out, variant, ticks, reproject, device, n_worlds=64, seed=0, one_step_n=64):
    out = Path(out)
    t0 = time.time()
    cands = MB.candidates(val_bank, EVAL_CTX, n_worlds, seed)
    ctx = torch.stack([val_bank.window_at(c["seq"], c["offset"], EVAL_CTX) for c in cands])           # n,16,3,H,W  (evaluation view)
    src = [c["source"] for c in cands]
    res = dict(variant=variant, reproject=reproject, n_worlds=len(cands), ticks=ticks, context=EVAL_CTX, dream={}, seconds=0)
    frames, stats = {}, {}
    with torch.no_grad():
        recon = ae.decode(ae.encode(ctx[:, -1].to(device))).clamp(0, 1).cpu()
    ref_sharp = sharpness(recon)
    real_sharp = sharpness(ctx[:, -1])
    res["sharpness_ref"] = dict(ae_recon_of_real=ref_sharp, real=real_sharp)
    contrast = lambda x: float(x.std(dim=(1, 2, 3)).mean())                       # a flat grey picture has ~0 contrast but can still fool the (lenient) eye segmenter
    ref_contrast = contrast(ctx[:, -1])
    for T in TEMPS:
        fr = dream(dyn, ae, ctx.to(device), ticks, temperature=T, reproject=reproject, seed=seed + 11).cpu()      # (ticks,n,3,H,W)
        st = score_dreams(fr, seg)
        stats[T] = st
        if T == PRIMARY_T:
            frames[T] = fr                                                          # only the primary dream is kept (3 x 64 x 300 x 3 x 128^2 floats killed the first run: exit -9, out of memory)
        sharp = {str(t): sharpness(fr[t]) for t in (29, 99, ticks - 1) if t < ticks}
        res["dream"][str(T)] = dict(mean=mean_stats(st), video=mean_stats(st, [i for i, s in enumerate(src) if s == "video"]),
                                    openeds=mean_stats(st, [i for i, s in enumerate(src) if s == "openeds"]), sharpness=sharp,
                                    sharp_ratio={k: v / ref_sharp for k, v in sharp.items()},
                                    contrast_ratio={str(t): contrast(fr[t]) / ref_contrast for t in (29, 99, ticks - 1) if t < ticks})
        print(f"[{variant}] dream T={T}: {res['dream'][str(T)]['mean']} sharpness/ref {({k: round(v, 2) for k, v in res['dream'][str(T)]['sharp_ratio'].items()})} contrast/real {({k: round(v, 2) for k, v in res['dream'][str(T)]['contrast_ratio'].items()})}", flush=True)
    # one-step vs copy-last on random held-out windows (teacher forced, passive)
    gen = torch.Generator().manual_seed(seed + 5)
    win = val_bank.sample(one_step_n, EVAL_CTX + 1, generator=gen, train=False)["images"]
    res["one_step"] = one_step_mse(dyn, ae, win.to(device), CTX)
    print(f"[{variant}] one-step mse {res['one_step']}", flush=True)
    touch = touch_test(dyn, ae, ctx[:16], reproject, device)
    res["touch"] = {k: {r: v[r] for r in v if r != "_frames"} for k, v in touch.items()}
    print(f"[{variant}] touch ratio@8 " + " ".join(f"{k}={v['8']['ratio']:.2f}" for k, v in touch.items()), flush=True)
    # sheets: dream sheet (rows = worlds, cols = real last context frame then dream ticks; green = segmenter flags an eye) and a touch sheet
    sel = sorted(set(list(range(0, len(cands), max(1, len(cands) // 8)))[:8]))
    size = ctx.shape[-1]
    for T in (PRIMARY_T,):
        fr, st = frames[T], stats[T]
        rows = [np.concatenate([MB.tile(ctx[i, -1], False, "real", size)] + [MB.tile(fr[t, i], bool(st["flags"][t, i]), f"t{t}", size) for t in SHEET_TICKS if t < ticks], 1) for i in sel]
        cv2.imwrite(str(out / f"dream_sheet_T{T}.png"), cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR))
    rows = []
    for name, v in touch.items():
        last, truth, fr = v["_frames"]
        for i in range(0, min(4, len(last))):
            rows.append(np.concatenate([MB.tile(last[i], False, f"{name} ctx", size)] + [MB.tile(truth[i, r - 1], False, f"truth t{r}", size) for r in TOUCH_READ[:2]]
                                       + [MB.tile(fr[i, r - 1], False, f"dream t{r}", size) for r in TOUCH_READ[:2]], 1))
    cv2.imwrite(str(out / "touch_sheet.png"), cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR))
    np.savez_compressed(out / "dream_frames_T0.5.npz", frames=(frames[PRIMARY_T][::2, sel].clamp(0, 1) * 255).round().to(torch.uint8).numpy(), ticks=np.arange(0, ticks, 2), worlds=np.array([cands[i]["group"] for i in sel]))
    res["worlds"] = [dict(group=c["group"], source=c["source"], seq=c["seq"], offset=c["offset"]) for c in cands]
    res["per_world"] = {k: [round(float(v), 3) for v in stats[PRIMARY_T][k]] for k in ("eye_frac", "eye_moving", "still_frac", "first100", "max_len")}
    res["seconds"] = round(time.time() - t0)
    (out / "eval.json").write_text(json.dumps(res, indent=1))
    return res


# ----------------------------------------------------------------------------------------------- stages
def stage_ae(a):
    import heldout_clarity as H
    out = Path(a.out) / "ae"
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device(a.device)
    ctx = H.prepare_height(a.cache, a.pack, a.size)
    res = H.run_height(a.cache, a.pack, a.size, out, a.ae_steps, a.ae_batch, a.ae_lr, device, a.seed, max(1, a.ae_steps // 4), arm=AE_ARM, ctx=ctx, save_to=out / "ae.pt")
    (out / "results.json").write_text(json.dumps({H.result_key(AE_ARM, a.size): res}, indent=1))
    return res


def stage_dyn(a):
    device = pick_device(a.device)
    out = Path(a.out) / a.variant
    out.mkdir(parents=True, exist_ok=True)
    ae = load_ae(a.ae).to(device)
    bk = banks(a.cache, size=ae.image_size)
    dyn, tr = train_dyn(ae, bk["train"], out, a.variant, a.minutes, a.batch, a.lr, device, a.seed, a.hid, a.layers, max_steps=a.max_steps, free_max=a.free_max)
    seg = E.load_segmenter(a.segmenter)
    res = eval_variant(dyn.eval(), ae, bk["val"], seg, out, a.variant, a.ticks, VARIANTS[a.variant][0], device, n_worlds=a.worlds)
    res["train"] = {k: v for k, v in tr.items() if k != "curve"}
    (out / "eval.json").write_text(json.dumps(res, indent=1))
    (out / "curve.json").write_text(json.dumps(tr["curve"]))
    return res


def stage_eval(a):
    device = pick_device(a.device)
    out = Path(a.out) / a.variant
    out.mkdir(parents=True, exist_ok=True)
    ae = load_ae(a.ae).to(device)
    dyn, _ = load_dyn(a.dyn)
    dyn = dyn.to(device)
    bk = banks(a.cache, size=ae.image_size, splits=("val",))
    return eval_variant(dyn, ae, bk["val"], E.load_segmenter(a.segmenter), out, a.variant, a.ticks, VARIANTS[a.variant][0], device, n_worlds=a.worlds)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("ae", "dyn", "eval"))
    ap.add_argument("--cache", nargs="+", required=True, help="CacheBank directory/ies at the AE size (ae stage takes ONE directory)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ae", default=None, help="ae.pt (dyn/eval); default <out>/ae/ae.pt")
    ap.add_argument("--dyn", default=None, help="dyn.pt (eval only)")
    ap.add_argument("--pack", default=str(HERE.parent / "data" / "heldout_pack"), help="held-out clarity pack (ae stage)")
    ap.add_argument("--segmenter", default=str(HERE / "pilot_results" / "eyemetric" / "segmenter.pt"))
    ap.add_argument("--variant", default="free", choices=tuple(VARIANTS))
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ae-steps", type=int, default=10000)
    ap.add_argument("--ae-batch", type=int, default=128)
    ap.add_argument("--ae-lr", type=float, default=1e-3)
    ap.add_argument("--minutes", type=float, default=105.0)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hid", type=int, default=96)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--free-max", type=int, default=None, help="override the variant's free-running horizon")
    ap.add_argument("--ticks", type=int, default=300)
    ap.add_argument("--worlds", type=int, default=64)
    a = ap.parse_args(argv)
    a.ae = a.ae or str(Path(a.out) / "ae" / "ae.pt")
    if a.stage == "ae":
        a.cache = a.cache[0]
        return stage_ae(a)
    a.cache = a.cache if len(a.cache) > 1 else a.cache[0]
    return stage_dyn(a) if a.stage == "dyn" else stage_eval(a)


if __name__ == "__main__":
    main()
