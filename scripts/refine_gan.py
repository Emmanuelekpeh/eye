"""R5 stage 1 (plan.md "R5"): does a discriminator make the SOFT render of a base model sharper and still faithful?
Bases: l1 = the L1 RSSM (96 px) posterior render of a real frame (what the UI re-encodes and shows); ae = the S32_ssim autoencoder reconstruction (128 px).
Arms: reg = residual U-Net trained with L1 + SSIM only (control), gan = same + hinge adversarial loss from a conditional PatchGAN. Eval = the R3 held-out clarity pack
(heldout_clarity.build_items / score_items / gate_of: same metrics, same gate) for [base, reg, gan], the pre-registered verdicts S1-G1..G3 (G4 = the sheets, looked at by a person),
and (stage `dream`, CPU) L1 dream renders at several ticks with and without the refiner.
  python scripts/refine_gan.py train --base l1 --ckpt <L1 model.pt> --cache <96px CacheBank dir> --pack <pack dir> --out <dir> --device auto
  python scripts/refine_gan.py train --base ae --ckpt <ae.pt> --cache <128px CacheBank dir> --pack <pack dir> --out <dir>
  python scripts/refine_gan.py dream --ckpt <L1 model.pt> --refiners <dir with l1_reg.pt l1_gan.pt> --out <dir>
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import heldout_clarity as H                                         # noqa: E402
from ivs.cachedata import CacheBank                                 # noqa: E402
from ivs.refiner import PatchDisc, Refiner, d_hinge, g_loss         # noqa: E402

SIZES = {"l1": 96, "ae": 128}
ARMS = ("reg", "gan")
# pre-registered thresholds (plan.md R5)
G1_KEEP, G2_FID, G2_PSNR, G3_KEEP = 0.08, 0.02, 1.5, 0.03
DECISIVE = ("openeds", "video_gray")


# ----------------------------------------------------------------------------------------------- bases
class Base:
    """A frozen model that turns a real frame into its soft render. single(x): (B,3,H,W) -> (B,3,H,W) in [0,1]; windowed(x): (B,T,3,H,W) -> (B,T,3,H,W) (L1: posterior recon along a window)."""

    def __init__(self, kind, ckpt, device):
        self.kind, self.size, self.device = kind, SIZES[kind], torch.device(device)
        if kind == "l1":
            from ivs.config import ModelConfig
            from ivs.model import EyeWorldModel
            ck = torch.load(ckpt, map_location="cpu", weights_only=False)
            self.model = EyeWorldModel(ModelConfig(**ck["config"]))
            self.model.load_state_dict(ck["model"])
            self.model.eval().to(self.device)
            self.action_dim = self.model.cfg.action_dim
        else:
            from ivs.latentdyn import load_ae
            self.ae = load_ae(ckpt).to(self.device)

    @torch.no_grad()
    def single(self, x):
        x = x.to(self.device)
        if self.kind == "ae":
            return self.ae.decode(self.ae.encode(x)).clamp(0, 1)
        out = self.model.observe(x[:, None], torch.zeros(x.shape[0], 1, self.action_dim, device=self.device))
        return out["recon"][:, 0].clamp(0, 1)

    @torch.no_grad()
    def windowed(self, x):
        x = x.to(self.device)
        if self.kind == "ae":
            return self.ae.decode(self.ae.encode(x)).clamp(0, 1)
        return self.model.observe(x, torch.zeros(x.shape[0], x.shape[1], self.action_dim, device=self.device))["recon"].clamp(0, 1)


def pair_batch(base, bank, batch, gen, device, window=8, t_min=3):
    """(render, target) for one step. l1: half single frames (the UI re-encodes one render per tick), half frames t_min..window-1 of real windows; ae: single frames.
    The target is always the real frame in the same (train) view."""
    if base.kind == "ae":
        x = bank.sample(batch, 1, generator=gen, train=True)["images"][:, 0].to(device)
        return base.single(x), x
    n1 = batch // 2
    x1 = bank.sample(n1, 1, generator=gen, train=True)["images"][:, 0].to(device)
    xw = bank.sample(batch - n1, window, generator=gen, train=True)["images"].to(device)
    t = torch.randint(t_min, window, (xw.shape[0],), generator=gen).to(device)
    rw = base.windowed(xw)
    idx = torch.arange(xw.shape[0], device=device)
    return torch.cat([base.single(x1), rw[idx, t]]), torch.cat([x1, xw[idx, t]])


# ----------------------------------------------------------------------------------------------- training
def train_arm(base, bank, arm, steps, batch, lr, device, seed, log_every=200, probe=None, probe_every=0, init=None):
    """init = a checkpoint dict from save_refiner (resume): the generator continues from its weights, and the discriminator from its saved weights when the file has them
    (files written before the resume option hold only the generator: the discriminator then starts fresh against a trained generator). The optimiser and the lr schedule
    restart (200-step warm-up, cosine), the data stream follows `seed`."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    g = Refiner().to(device)
    d = PatchDisc().to(device) if arm == "gan" else None
    if init is not None:
        g.load_state_dict(init["state"])
        if d is not None and init.get("disc") is not None:
            d.load_state_dict(init["disc"])
        print(f"  [{base.kind}/{arm}] resumed generator{' + discriminator' if d is not None and init.get('disc') is not None else ''} from {init.get('meta')}", flush=True)
    og = torch.optim.Adam(g.parameters(), lr=lr, betas=(0.5, 0.999))
    od = torch.optim.Adam(d.parameters(), lr=lr, betas=(0.5, 0.999)) if d else None
    t0, run, curve = time.time(), {}, []
    for step in range(steps):
        cur = lr * min(1.0, (step + 1) / 200) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / steps)))
        for o in (og, od):
            if o:
                for pg in o.param_groups:
                    pg["lr"] = cur
        render, target = pair_batch(base, bank, batch, gen, device)
        out = g(render)
        if d is not None:                                                  # discriminator first (on the detached candidate), then the generator
            ld = d_hinge(d(render, target), d(render, out.detach()))
            od.zero_grad(set_to_none=True)
            ld.backward()
            od.step()
            run["d"] = run.get("d", 0.0) + float(ld.detach())
        loss, parts = g_loss(render, out, target, d=d)
        og.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(g.parameters(), 10.0)
        og.step()
        for k, v in dict(loss=loss.detach(), **parts).items():
            run[k] = run.get(k, 0.0) + float(v)
        if (step + 1) % log_every == 0 or step + 1 == steps:
            n = log_every if (step + 1) % log_every == 0 else (step + 1) % log_every
            row = {k: round(v / n, 4) for k, v in run.items()}
            row.update(step=step + 1, sec=round(time.time() - t0, 1), identity_l1=round(float((render - target).abs().mean()), 4))
            if probe and probe_every and ((step + 1) % probe_every == 0 or step + 1 == steps):
                row.update(probe(g))
            curve.append(row)
            print(f"  [{base.kind}/{arm}] " + " ".join(f"{k} {v}" for k, v in row.items()), flush=True)
            run = {}
    g.eval()
    g.__dict__["disc"] = d                                                 # kept for save_refiner (None for the reg arm); written to __dict__ so it is NOT registered as a submodule of g
    return g, curve, round(time.time() - t0, 1)


def save_refiner(path, g, meta):
    d = getattr(g, "disc", None)
    torch.save(dict(state=g.state_dict(), disc=None if d is None else d.state_dict(), meta=meta), path)


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def load_refiner(path, device="cpu"):
    ck = load_checkpoint(path)
    g = Refiner()
    g.load_state_dict(ck["state"])
    return g.eval().to(device)


# ----------------------------------------------------------------------------------------------- evaluation
@torch.no_grad()
def render_recs(base, refiner, items, device, chunk=48, seed=0):
    """uint8 (h,h,3) reconstructions of the held-out items: refiner(base render) (refiner None = the base render itself). The base noise is seeded per chunk, so arms see the same renders."""
    out = []
    for i in range(0, len(items), chunk):
        x = torch.from_numpy(np.stack([it["view"] for it in items[i:i + chunk]])).permute(0, 3, 1, 2).float().div(255)
        torch.manual_seed(seed + i)
        r = base.single(x)
        y = r if refiner is None else refiner(r.to(device))
        out.extend((y.clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy() * 255).round().astype(np.uint8))
    return out


def verdicts(scores):
    """scores = {arm: {slice: {edge_keep, edge_fidelity, psnr, ...}}} with arms 'base', 'reg', 'gan' -> pre-registered S1-G1..G3 (G4 = the sheets, a person looks)."""
    b, r, g = scores["base"], scores["reg"], scores["gan"]
    gate = scores.get("_gate", {}).get("gan", {})
    g1 = all(g[s]["edge_keep"] - b[s]["edge_keep"] >= G1_KEEP for s in DECISIVE) and all(bool(gate.get(s)) and all(gate[s].values()) for s in DECISIVE)
    g2 = all(g[s]["edge_fidelity"] >= b[s]["edge_fidelity"] - G2_FID and g[s]["psnr"] >= b[s]["psnr"] - G2_PSNR for s in g)
    g3 = all(g[s]["edge_keep"] - r[s]["edge_keep"] >= G3_KEEP and g[s]["edge_fidelity"] >= r[s]["edge_fidelity"] - G2_FID for s in DECISIVE)
    return dict(G1_clarity=bool(g1), G2_faithful=bool(g2), G3_attribution=bool(g3), G4="LOOK at the sheets", gain_keep_vs_base={a: {s: round(scores[a][s]["edge_keep"] - b[s]["edge_keep"], 3) for s in scores[a]} for a in ("reg", "gan")},
                gain_keep_gan_vs_reg={s: round(g[s]["edge_keep"] - r[s]["edge_keep"], 3) for s in g})


def stack(rows):
    w = max(r.shape[1] for r in rows)
    return np.concatenate([np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0)), constant_values=0) for r in rows], 0)


def evaluate(base, refiners, pack, out, device):
    h = base.size
    items = H.build_items(pack, h)
    ceil = H.score_items(items, [it["view"] for it in items])
    recs = {"base": render_recs(base, None, items, device)}
    for arm, g in refiners.items():
        recs[arm] = render_recs(base, g, items, device)
    scores, gates, res = {}, {}, {}
    for arm, r in recs.items():
        sc = H.score_items(items, r)
        scores[arm] = sc
        gates[arm] = {s: H.gate_of(sc[s], ceil[s]) for s in sc}
        res[arm] = {s: dict(metrics=sc[s], gate=gates[arm][s], passed=bool(all(gates[arm][s].values()))) for s in sc}
        print(f"[{base.kind}] {arm}: " + " | ".join(f"{s} keep {sc[s]['edge_keep']:.3f} fid {sc[s]['edge_fidelity']:.3f} psnr {sc[s]['psnr']:.1f} {'PASS' if res[arm][s]['passed'] else 'FAIL'}" for s in sc), flush=True)
    out = Path(out)
    for s, ids in H.show_pairs(items, H.sheet_ids_of(pack)).items():
        if len(ids) == 2:
            rows = [H.label_row(H.sheet_row(items, recs[a], ids), f"{base.kind} {a}: native | ceiling | {a}") for a in recs]
            cv2.imwrite(str(out / f"refine_sheet_{base.kind}_{s}.png"), stack(rows))
    out_scores = dict(scores, _gate=gates)
    ver = verdicts(out_scores) if all(a in scores for a in ("base", "reg", "gan")) else {}
    return dict(ceiling=ceil, arms=res, verdict=ver, eval_frames=len(items))


def train_main(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if a.device == "auto" and torch.cuda.is_available() else ("cpu" if a.device == "auto" else a.device))
    base = Base(a.base, a.ckpt, device)
    bank = CacheBank(a.cache, image_size=base.size, split="train", val_frac=H.VAL_FRAC, seed=H.SEED, tint_mode="gray", source_weights=H.SOURCE_WEIGHTS, device="cpu")
    print(f"device={device} base={a.base} size={base.size} train bank {bank.info()['frames']} frames / {bank.info()['groups']} groups steps={a.steps} batch={a.batch}", flush=True)
    items = H.build_items(a.pack, base.size)
    probe_items = H.probe_subset(items)

    def probe(g):
        sc = H.score_items(probe_items, render_recs(base, g.eval(), probe_items, device))
        g.train()
        return {f"{s}/edge_keep": round(v["edge_keep"], 3) for s, v in sc.items()}
    refiners, train = {}, {}
    for arm in a.arms:
        init, prev_steps = None, 0
        if a.resume:
            src = Path(a.resume) / f"{a.base}_{arm}.pt"
            if not src.exists():
                raise SystemExit(f"--resume {a.resume}: {src.name} not found")
            init = load_checkpoint(src)
            prev_steps = int(init.get("meta", {}).get("steps_total", init.get("meta", {}).get("steps", 0)))
        g, curve, sec = train_arm(base, bank, arm, a.steps, a.batch, a.lr, device, a.seed + (1000 if init else 0), probe=probe, probe_every=max(1, a.steps // 3), init=init)
        refiners[arm] = g
        save_refiner(out / f"{a.base}_{arm}.pt", g, dict(base=a.base, arm=arm, steps=a.steps, steps_total=prev_steps + a.steps, batch=a.batch, seed=a.seed, resumed=bool(init)))
        train[arm] = dict(seconds=sec, curve=curve, params=sum(p.numel() for p in g.parameters()))
    res = evaluate(base, refiners, a.pack, out, device)
    res["train"], res["base"], res["steps"], res["batch"], res["resumed_from"] = train, a.base, a.steps, a.batch, a.resume
    (out / f"results_{a.base}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["verdict"], indent=1), flush=True)
    return res


def dream_main(a):
    """CPU: L1 dream renders (UI loop from a real held-out context) with and without the refiner. Rows = worlds; columns = ticks; each cell = [render | reg | gan]."""
    sys.path.insert(0, str(HERE / "scripts"))
    import make_birth as MB
    from ivs import eyemetric as E
    from ivs.birth import perceive
    base = Base("l1", a.ckpt, "cpu")
    refs = {arm: load_refiner(Path(a.refiners) / f"l1_{arm}.pt") for arm in ARMS if (Path(a.refiners) / f"l1_{arm}.pt").exists()}
    bank = CacheBank(a.cache, image_size=96, split="val", val_frac=H.VAL_FRAC, seed=H.SEED, tint_mode="gray", sources=["video", "openeds"])
    cands = MB.candidates(bank, 16, a.worlds * 4, 0)
    cands = cands[::max(1, len(cands) // a.worlds)][:a.worlds]
    wins = torch.stack([bank.window_at(c["seq"], c["offset"], 16) for c in cands])
    p = perceive(base.model, wins, seed=0)
    fr = E.ui_dream_frames(base.model, ticks=max(a.ticks) + 1, n=len(cands), noise_seed=21, init_state=(p["h"], p["z"]))      # (T,n,3,H,W)
    rows = []
    with torch.no_grad():
        for i, c in enumerate(cands):
            cells = [MB.tile(wins[i, -1], False, f"real {c['source'][:2]}", 96)]
            for t in a.ticks:
                x = fr[t, i:i + 1]
                tiles = [MB.tile(x[0], False, f"t{t}", 96)] + [MB.tile(refs[arm](x)[0].clamp(0, 1), False, arm, 96) for arm in refs]
                cells.append(np.concatenate(tiles, 0))
            cells[0] = np.pad(cells[0], ((0, cells[1].shape[0] - cells[0].shape[0]), (0, 0), (0, 0)))       # the real frame sits at the top of its column
            rows.append(np.concatenate(cells, 1))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(Path(a.out) / "refine_dream_sheet.png"), cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR))
    print("wrote", Path(a.out) / "refine_dream_sheet.png")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="stage", required=True)
    t = sub.add_parser("train")
    t.add_argument("--base", choices=tuple(SIZES), required=True)
    t.add_argument("--ckpt", required=True)
    t.add_argument("--cache", required=True)
    t.add_argument("--pack", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    t.add_argument("--steps", type=int, default=6000)
    t.add_argument("--batch", type=int, default=24)
    t.add_argument("--lr", type=float, default=2e-4)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default="auto")
    t.add_argument("--resume", default=None, help="directory holding <base>_<arm>.pt from an earlier run: continue those refiners (new data seed, restarted lr schedule)")
    d = sub.add_parser("dream")
    d.add_argument("--ckpt", required=True)
    d.add_argument("--cache", nargs="+", default=[str(HERE.parent / "data" / "cache" / "videos"), str(HERE.parent / "data" / "cache" / "openeds")])
    d.add_argument("--refiners", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--worlds", type=int, default=8)
    d.add_argument("--ticks", type=int, nargs="+", default=[1, 10, 30, 60, 100])
    a = ap.parse_args(argv)
    return train_main(a) if a.stage == "train" else dream_main(a)


if __name__ == "__main__":
    main()
