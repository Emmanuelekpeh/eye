"""PD-NCA trainer: the persistent denoising NCA (backlog P0j). Same EyeNCA network (ivs/nca.py) with the PD flags, trained the Growing-NCA way.

  python -m ivs.pnca --cache ../data/cache/videos ../data/cache/openeds --source-weights video=0.5,openeds=0.5 \
      --out pnca_out --runs P0_core P1_gate --steps 3000 --device auto

What is different from ivs/ncapilot.py (E0): the hidden channels are NEVER rebuilt. A POOL of persistent states (image x, hidden h, clean target y,
level j) is trained on: each iteration takes `batch` of them (25% fresh seeds at sigma_max + the highest-loss pooled state re-seeded), runs `k_ticks`
ticks with the gradient flowing through the carried hidden state, then writes the detached states back, so a state lives for many iterations.
  denoise ticks (level j < levels): the input is y + sigma_j * eps (the sample's own fixed noise), the hidden state is carried to the next level
  living ticks  (j >= levels)     : re-noise the own render to sigma_live in {.02,.05,.1}, denoise with the carried hidden, feed the result back
                                    (homeostasis); with probability p_damage a living state is first scatter-damaged and must come back to y.
Denoise ticks use the TRUE forward path of each pooled sample: the input at level j is y + sigma_j * eps with that sample's own stored eps (so the input
carries information about y and D(x) can be trained to E[y|x], exactly the EDM objective; an input produced by the model's own Euler path from noise would
carry none), and the hidden state is what persists across levels. Living ticks start from the model's own last render. Loss weights: denoise ticks
w = EDM weight (unit weight on the network output at every sigma), living ticks `live_weight` (a plain MSE: the EDM weight at sigma_live = .02 is ~2500 and would
make the model correct its own render with one tiny step). Loss axes, each
switchable per run: deep (loss on every tick, else the last tick only), l1l2 (+ L1), edge (Sobel + Laplacian-pyramid L1), coarse (12x12 pooled
MSE), overflow (hidden outside [-1,1]), drift (|x_next - x| on living ticks). Evaluation always uses the EMA weights.
Outputs per run (same layout as ncapilot): metrics.json, train_log.jsonl, model.pt, sheet_dreams.png (persistent dreams), sheet_living.png
(rows = dreams, columns = living tick 0/10/30/100), sheet_repair.png. The full gates are scripts/eval_nca.py (it reads run['persist']).
"""
import argparse
import json
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import nca, ncapilot
from .actions import scatter_batch
from .pilot import save_sheet

SIGMA_LIVE = (0.02, 0.05, 0.1)
LIVING_TICKS = (0, 10, 30, 100)
SCATTER = dict(max_discs=3, radius=(0.15, 0.25), strength=(0.7, 1.0))


@dataclass
class PRun:
    name: str = "P0_core"
    # --- architecture (NCAConfig)
    state: int = 32
    hidden: int = 128
    n_inner: int = 8
    dilations: tuple = (1, 2, 4)
    fire_rate: float = 1.0
    gate: bool = False
    fourier: bool = False
    fourier_hidden: int = 64
    lattn: bool = False
    heads: int = 2
    attn_dk: int = 4
    attn_dv: int = 4
    film: bool = False
    # --- architecture family: 'nca' = EyeNCA (fields above); 'vitca' = ivs/vitca.py EyeViTCA (fields below). plain=True = E0-style training of the same
    #     network (one denoise call per sample, hidden state rebuilt each call, no pool): the control that answers "does this architecture draw eyes at all".
    arch: str = "nca"
    patch: int = 8
    dim: int = 256
    blocks: int = 3
    loops: int = 4
    vheads: int = 8
    mlp_mult: int = 4
    plain: bool = False
    # --- optimisation
    lr: float = 1e-3
    lr_floor: float = 0.02
    warmup: int = 200
    batch: int = 16
    grad_clip: float = 1.0
    weight_decay: float = 0.0
    ema: float = 0.999
    checkpoint: bool = False               # recompute inner steps in the backward pass (memory for time)
    train_seed: int = 0
    # --- pool training
    persist: bool = True                   # marker read by scripts/eval_nca.py: sample / repair with the hidden state carried
    pool_size: int = 1024
    fresh_frac: float = 0.25
    k_ticks: int = 3
    levels: int = 12                       # noise levels of the pool trajectory
    sigma_max: float = 5.0
    sigma_min: float = 0.01
    p_damage: float = 0.5                  # chance a living state is scatter-damaged before its tick (regeneration)
    live_weight: float = 20.0              # loss weight of a living tick (plain MSE-type loss); denoise ticks use the EDM weight (~5 at sigma 1, ~15 at .3)
    # --- loss axes
    deep: bool = True
    l1l2: bool = False
    edge: float = 0.0
    coarse: float = 0.0
    overflow: float = 0.0
    drift: float = 0.0


PRESETS = {
    "P0_core": dict(),
    "P1_gate": dict(gate=True),
    "P1_fourier": dict(fourier=True),
    "P1_lattn": dict(lattn=True),
    "P1_film": dict(film=True),
    "P1_l1l2": dict(l1l2=True),
    "P1_edge": dict(edge=0.5),
    "P1_coarse": dict(coarse=1.0),
    "P1_overflow": dict(overflow=1.0),
    "P1_drift": dict(drift=1.0),
    "P1_nodeep": dict(deep=False),
    "P1_noregen": dict(p_damage=0.0),
    "P2_arch": dict(gate=True, fourier=True, lattn=True, film=True),
    "P2_losses": dict(l1l2=True, edge=0.5, coarse=1.0, overflow=1.0, drift=1.0),
    # ---- ViTCA (ivs/vitca.py): recurrent global-attention tokens with a persistent bounded state. V_plain = E0-style control (no pool, no persistence).
    "V_plain": dict(arch="vitca", plain=True, persist=False, batch=64, lr=5e-4, weight_decay=0.01),
    "V_pers": dict(arch="vitca", lr=5e-4, weight_decay=0.01),
    "V2_pers": dict(arch="vitca", blocks=2, lr=5e-4, weight_decay=0.01),
    "P3_all": dict(gate=True, fourier=True, lattn=True, film=True, l1l2=True, edge=0.5, coarse=1.0, overflow=1.0, drift=1.0),
}


def build_run(name):
    return replace(PRun(name=name), **PRESETS[name])


def model_config(rc, size):
    if rc.arch == "vitca":
        from . import vitca
        return vitca.ViTCAConfig(image_size=size, patch=rc.patch, dim=rc.dim, blocks=rc.blocks, loops=rc.loops, vheads=rc.vheads, mlp_mult=rc.mlp_mult)
    return nca.NCAConfig(image_size=size, state=rc.state, hidden=rc.hidden, n_inner=rc.n_inner, dilations=tuple(rc.dilations), fire_rate=rc.fire_rate,
                         gate=rc.gate, fourier=rc.fourier, fourier_hidden=rc.fourier_hidden, lattn=rc.lattn, heads=rc.heads,
                         attn_dk=rc.attn_dk, attn_dv=rc.attn_dv, film=rc.film)


# ----------------------------------------------------------------------------------------------- losses
def _grad(x):
    return x[..., :, 1:] - x[..., :, :-1], x[..., 1:, :] - x[..., :-1, :]


def edge_pyramid_loss(d, y, levels=3):
    """Per-sample L1 between the finite-difference gradients of d and y plus L1 between their Laplacian-pyramid bands (sharpness, no blur)."""
    dx, dy = _grad(d)
    tx, ty = _grad(y)
    loss = (dx - tx).abs().mean((1, 2, 3)) + (dy - ty).abs().mean((1, 2, 3))
    for _ in range(levels):
        d2, y2 = F.avg_pool2d(d, 2), F.avg_pool2d(y, 2)
        bd, by = d - F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False), y - F.interpolate(y2, scale_factor=2, mode="bilinear", align_corners=False)
        loss = loss + (bd - by).abs().mean((1, 2, 3))
        d, y = d2, y2
    return loss


def coarse_loss(d, y, cells=12):
    k = max(1, d.shape[-1] // cells)
    return ((F.avg_pool2d(d, k) - F.avg_pool2d(y, k)) ** 2).mean((1, 2, 3))


def tick_terms(rc, d, y, weight):
    """Per-sample loss of one tick: weight * (MSE [+ L1] + edge * edge_pyramid + coarse * coarse_mse). d, y in model range; weight (B,).
    Returns (weighted terms, plain per-sample MSE)."""
    err = d - y
    mse = (err ** 2).mean((1, 2, 3))
    base = mse
    if rc.l1l2:
        base = base + err.abs().mean((1, 2, 3))
    if rc.edge > 0:
        base = base + rc.edge * edge_pyramid_loss(d, y)
    if rc.coarse > 0:
        base = base + rc.coarse * coarse_loss(d, y)
    return weight * base, mse


def overflow_loss(h):
    return (h - h.clamp(-1.0, 1.0)).abs().mean()


# ----------------------------------------------------------------------------------------------- pool
class Pool:
    """Persistent states. y (clean target), eps (this sample's fixed noise), x (the model's own last render: what a LIVING state is), h (hidden channels),
    j (level; >= levels means living), loss (last per-sample loss)."""
    def __init__(self, size, cfg, device):
        s, c = cfg.image_size, cfg.channels
        self.size = size
        self.y = torch.zeros(size, c, s, s, device=device)
        self.eps = torch.zeros(size, c, s, s, device=device)
        self.x = torch.zeros(size, c, s, s, device=device)
        self.h = torch.zeros(size, *cfg.hidden_shape(), device=device)
        self.j = torch.zeros(size, dtype=torch.long, device=device)
        self.loss = torch.zeros(size, device=device)

    def seed_all(self, bank, rc, gen, device):
        for i in range(0, self.size, 64):
            n = min(64, self.size - i)
            y = nca.to_model(bank.sample(n, 1, generator=gen, train=True)["images"][:, 0].to(device))
            self.y[i:i + n] = y
            self.eps[i:i + n] = nca._randn(y.shape, gen, device)


def pool_batch(pool, rc, bank, gen, device):
    """Pick the batch: (B - n_fresh) pooled states (the highest-loss one is re-seeded) + n_fresh seeds. Returns (y, eps, x, h, j) and the pool slots to write back."""
    B, P = rc.batch, pool.size
    n_fresh = max(1, round(B * rc.fresh_frac))
    n_pool = B - n_fresh
    perm = torch.randperm(P, generator=gen)
    cand = perm[:n_pool].to(device)
    w = int(pool.loss[cand].argmax())
    keep = torch.cat([cand[:w], cand[w + 1:]])
    slots = torch.cat([keep, cand[w:w + 1], perm[n_pool:n_pool + n_fresh].to(device)])        # kept, then the (n_fresh + 1) re-seeded slots
    n_new = len(slots) - len(keep)
    y_new = nca.to_model(bank.sample(n_new, 1, generator=gen, train=True)["images"][:, 0].to(device))
    eps_new = nca._randn(y_new.shape, gen, device)
    y = torch.cat([pool.y[keep], y_new])
    eps = torch.cat([pool.eps[keep], eps_new])
    x = torch.cat([pool.x[keep], torch.zeros_like(y_new)])
    h = torch.cat([pool.h[keep], torch.zeros(n_new, *pool.h.shape[1:], device=device)])
    j = torch.cat([pool.j[keep], torch.zeros(n_new, dtype=torch.long, device=device)])
    return y, eps, x, h, j, slots


def run_ticks(model, rc, y, eps, x, h, j, sig, gen, device):
    """k_ticks ticks on a batch of persistent states with gradient through the carried hidden state. Returns (loss, per-sample last terms, x, h, j, stats)
    where x is each state's own last render (the state a living entry continues from)."""
    sd, L = model.cfg.sigma_data, rc.levels
    sig = sig.to(device)
    live_choices = torch.tensor(SIGMA_LIVE, device=device)
    living = j >= L
    if rc.p_damage > 0 and living.any():                                             # regeneration: damage some living states, the target stays y
        hit = living & (torch.rand(len(j), generator=gen).to(device) < rc.p_damage)
        if hit.any():
            x = x.clone()
            x[hit] = nca.to_model(scatter_batch(nca.to_unit(x[hit]), generator=gen, **SCATTER))
    totals, mse_den, mse_live, drift, n_den, n_live = [], 0.0, 0.0, [], 0, 0
    for t in range(rc.k_ticks):
        living = j >= L
        lv = living.view(-1, 1, 1, 1)
        sigma = torch.where(living, live_choices[torch.randint(len(SIGMA_LIVE), (len(j),), generator=gen).to(device)], sig[j.clamp(max=L - 1)])
        s4 = sigma.view(-1, 1, 1, 1)
        x_in = torch.where(lv, x + s4 * nca._randn(x.shape, gen, device), y + s4 * eps)       # living: the own render re-noised; denoise: the true forward path
        d, h = model.denoise(x_in, sigma, hidden=h, return_hidden=True, checkpoint_steps=rc.checkpoint)
        weight = torch.where(living, torch.full_like(sigma, rc.live_weight), nca.edm_weight(sigma, sd))
        terms, mse = tick_terms(rc, d, y, weight)
        mse = mse.detach()                                                                       # statistics only
        totals.append(terms)
        if t == rc.k_ticks - 1:
            last_terms = terms.detach()
        x_next = d.clamp(-1, 1)
        if living.any():
            drift.append((x_next - x).abs().mean((1, 2, 3))[living])
            mse_live += float(mse[living].sum()); n_live += int(living.sum())
        if (~living).any():
            mse_den += float(mse[~living].sum()); n_den += int((~living).sum())
        x, j = x_next, j + (~living).long()
    loss = torch.stack(totals).mean(0) if rc.deep else totals[-1]
    loss = loss.mean()
    if rc.overflow > 0:
        loss = loss + rc.overflow * overflow_loss(h)
    if rc.drift > 0 and drift:
        loss = loss + rc.drift * torch.cat(drift).mean()
    stats = dict(mse_den=mse_den / max(n_den, 1) if n_den else None, mse_live=mse_live / max(n_live, 1) if n_live else None,
                 live_frac=float((j >= L).float().mean()), h_abs=float(h.detach().abs().mean()))
    return loss, last_terms, x.detach(), h.detach(), j, stats


# ----------------------------------------------------------------------------------------------- training
def train_run(rc, train_bank, device, total_steps, seed, log_path, log_every=100, ckpt_every=0):
    seed = seed + rc.train_seed
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    cfg = model_config(rc, train_bank.size)
    model = nca.net_from_config(asdict(cfg)).to(device)
    ema = nca.make_ema(model)
    opt = torch.optim.AdamW(model.parameters(), lr=rc.lr, weight_decay=rc.weight_decay)
    gen = torch.Generator().manual_seed(seed)
    sig = nca.sigma_schedule(rc.levels, rc.sigma_min, rc.sigma_max)
    pool = None if rc.plain else Pool(rc.pool_size, cfg, device)
    if pool is not None:
        pool.seed_all(train_bank, rc, gen, device)
    log = open(log_path, "w")
    t0, hist, acc = time.time(), [], {}
    model.train()
    for step in range(total_steps):
        for g in opt.param_groups:
            g["lr"] = ncapilot.lr_at(rc, step, total_steps)
        opt.zero_grad(set_to_none=True)
        if rc.plain:            # E0-style control: fresh frames, ONE denoise call at a lognormal sigma, hidden state rebuilt (zeros) every call
            y = nca.to_model(train_bank.sample(rc.batch, 1, generator=gen, train=True)["images"][:, 0].to(device))
            loss, per, _ = nca.edm_loss(model, y, generator=gen, checkpoint_steps=rc.checkpoint)
            st = dict(mse_den=float(per.mean()))
        else:
            y, eps, x, h, j, slots = pool_batch(pool, rc, train_bank, gen, device)
            loss, terms, xn, hn, jn, st = run_ticks(model, rc, y, eps, x, h, j, sig, gen, device)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"[{rc.name}] loss {float(loss)} at step {step + 1}; see {log_path}")
        loss.backward()
        gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), rc.grad_clip))
        opt.step()
        nca.ema_update(ema, model, min(rc.ema, (1 + step) / (10 + step)))
        if pool is not None:
            pool.x[slots], pool.h[slots], pool.y[slots], pool.eps[slots], pool.j[slots], pool.loss[slots] = xn, hn, y, eps, jn, terms
        for k, v in (("loss", float(loss)), ("gnorm", gnorm), ("gclip", float(gnorm > rc.grad_clip)), *st.items()):
            if v is not None:
                acc.setdefault(k, []).append(v)
        if (step + 1) % log_every == 0 or step == 0 or step + 1 == total_steps:
            row = dict(step=step + 1, lr=round(opt.param_groups[0]["lr"], 6), sec=round(time.time() - t0, 1))
            row.update({k + "_avg": round(float(np.mean(v)), 5) for k, v in acc.items() if k not in ("gnorm", "gclip")})
            row.update(gnorm=round(float(np.mean(acc["gnorm"])), 3), gnorm_max=round(float(np.max(acc["gnorm"])), 3), gclip=round(float(np.mean(acc["gclip"])), 3))
            acc = {}
            hist.append(row)
            log.write(json.dumps(row) + "\n"); log.flush()
            print(f"[{rc.name}] " + " ".join(f"{k}={v}" for k, v in row.items()), flush=True)
        if ckpt_every and (step + 1) % ckpt_every == 0 and step + 1 < total_steps:
            ncapilot.save_model(Path(log_path).parent / "model_partial.pt", model, ema, cfg, rc, step=step + 1)
    log.close()
    return model, ema, cfg, hist, time.time() - t0


# ----------------------------------------------------------------------------------------------- quick evaluation + sheets
@torch.no_grad()
def persistence_probe(ema, device, n=32, seed=21, ticks=(0, 10, 30, 100), steps=18, sigma=0.05, persist=True):
    """Dream n frames with the hidden state carried, then run the living loop. Cheap numbers (no segmenter): per tick the mean absolute change from
    tick 0, pixel std (contrast) and Sobel energy; the full P1/P2 gates are scripts/eval_nca.py. Returns (metrics, frames at each tick)."""
    ema.eval()
    gen = torch.Generator().manual_seed(seed)
    if persist:
        dream, h = nca.sample(ema, n, steps=steps, generator=gen, device=device, persist=True)
    else:                                   # plain (E0-style) run: no carried state, so no living ticks: the dream only
        dream, h, ticks = nca.sample(ema, n, steps=steps, generator=gen, device=device), None, (0,)
    img, snaps = dream, {0: dream}
    t_prev = 0
    for t in ticks[1:]:
        img, h = nca.live(ema, img, h, t - t_prev, sigma=sigma, generator=gen)
        snaps[t], t_prev = img, t
    from . import eyemetric as E
    sob = lambda f: float(E.grad_mag(E.gray(f.cpu())).mean())               # eyemetric's Sobel kernel lives on the CPU
    out = {str(t): dict(change_from_dream=float((f - dream).abs().mean()), std=float(f.flatten(1).std(1).mean()), sobel=sob(f),
                        finite=bool(torch.isfinite(f).all())) for t, f in snaps.items()}
    return out, snaps


@torch.no_grad()
def sheets(ema, val, out_dir, device, probe_frames, seed=21, repair_sigma=0.6, persist=True):
    ema.eval()
    out_dir = Path(out_dir)
    n = len(probe_frames[0])
    save_sheet([[probe_frames[0][i].cpu() for i in range(r * 4, min(r * 4 + 4, n))] for r in range(min(4, -(-n // 4)))], out_dir / "sheet_dreams.png")
    ts = sorted(probe_frames)
    if len(ts) > 1:
        save_sheet([[probe_frames[t][i].cpu() for t in ts] for i in range(min(8, n))], out_dir / "sheet_living.png")
    gen = torch.Generator().manual_seed(seed)
    clean = val.sample(8, 1, generator=gen, train=False)["images"][:, 0].to(device)
    dmg = scatter_batch(clean, generator=gen, **SCATTER, p_disc=1.0)
    rep = nca.repair_blind(ema, dmg, repair_sigma, steps=12, generator=gen, persist=persist)
    rep = rep[0] if persist else rep
    save_sheet([[clean[i].cpu(), dmg[i].cpu(), rep[i].cpu()] for i in range(8)], out_dir / "sheet_repair.png")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, nargs="+")
    ap.add_argument("--source-weights", default=None)
    ap.add_argument("--out", default="pnca_out")
    ap.add_argument("--runs", nargs="+", default=["P0_core"])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--pool-size", type=int, default=None)
    ap.add_argument("--image-size", type=int, default=96)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tint-mode", default="gray", choices=["gray", "keep"])
    ap.add_argument("--max-videos", type=int, default=None)
    ap.add_argument("--max-frames-per-video", type=int, default=None)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--eval-frames", type=int, default=128)
    ap.add_argument("--probe-n", type=int, default=32)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--ckpt-every", type=int, default=0)
    ap.add_argument("--checkpoint", action="store_true")
    ap.add_argument("--k-ticks", type=int, default=None)
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args(argv)
    device = torch.device("cuda" if a.device == "auto" and torch.cuda.is_available() else ("cpu" if a.device == "auto" else a.device))
    torch.set_num_threads(max(1, a.threads))
    out_root = Path(a.out); out_root.mkdir(parents=True, exist_ok=True)
    train, val, val_by_source, dyn_sources, _ = ncapilot.make_banks(a, device)
    print(f"device={device} sources={dyn_sources} train={train.info()} val={val.info()}", flush=True)
    summary = {}
    for name in a.runs:
        run_dir = out_root / name
        if a.skip_existing and (run_dir / "metrics.json").exists():
            summary[name] = json.loads((run_dir / "metrics.json").read_text()); continue
        run_dir.mkdir(exist_ok=True)
        rc = build_run(name)
        rc = replace(rc, **{k: v for k, v in (("batch", a.batch), ("pool_size", a.pool_size), ("k_ticks", a.k_ticks)) if v},
                     **({"checkpoint": True} if a.checkpoint else {}))
        model, ema, cfg, hist, secs = train_run(rc, train, device, a.steps, a.seed, run_dir / "train_log.jsonl", a.log_every, ckpt_every=a.ckpt_every)
        ncapilot.save_model(run_dir / "model.pt", model, ema, cfg, rc)                     # saved BEFORE the probe: a failing probe never costs a trained model
        t1 = time.time()
        probe, frames = persistence_probe(ema, device, n=a.probe_n, persist=rc.persist)
        metrics = dict(run=asdict(rc), config=asdict(cfg), train_seconds=round(secs, 1), steps_per_sec=round(a.steps / secs, 2), params=nca.count_params(model),
                       heldout_ema=ncapilot.eval_denoise(ema, val, device, n=a.eval_frames),
                       heldout_ema_by_source={s: ncapilot.eval_denoise(ema, b, device, n=a.eval_frames) for s, b in val_by_source.items()},
                       persistence=probe)
        sheets(ema, val_by_source.get("openeds", val), run_dir, device, frames, persist=rc.persist)
        metrics["eval_seconds"] = round(time.time() - t1, 1)
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=1, default=float))
        summary[name] = metrics
        print(f"[{name}] done: train {secs:.0f}s, eval {metrics['eval_seconds']}s", flush=True)
    (out_root / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    return summary


if __name__ == "__main__":
    main()
