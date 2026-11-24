"""EyeNCA trainer + quick evaluation on the video / S_ caches (stage E0: single frames, no time). CPU-aware, GPU-fast.
Same CLI and output layout as ivs/pilot.py (so scripts/kaggle_pilot.py runs it with ENTRY = "ncapilot"). Design + gates: backlog.md P0g.

  python -m ivs.ncapilot --cache ../data/cache/videos ../data/cache/openeds --source-weights video=0.5,openeds=0.5 \
      --out nca_out --runs E0_N50 E0_N135 --steps 20000 --device auto

Data: temporal sources only (video, openeds), one frame per sample (T=1), the same per-source group val split as the RSSM runs; colour
appearance sources (mrl) are NOT used in E0. Training = EDM denoising loss; every evaluation and sample uses the EMA weights.
Every run writes metrics.json, train_log.jsonl, model.pt ({'model': EMA, 'raw': live weights, 'config', 'run'}) and PNG sheets that must
be LOOKED AT (sheet_dreams.png, sheet_denoise.png, sheet_repair.png): held-out MSE cannot tell a sharp eye from a blur.
The real comparison against the RSSM (repair, dream metrics, cost) is scripts/eval_nca.py.
"""
import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch

from . import nca
from .actions import scatter_batch
from .cachedata import CacheBank, source_kinds
from .pilot import save_sheet

SIGMAS = (0.05, 0.15, 0.5, 1.0, 2.5)        # held-out denoising levels (model range, data std ~0.5)


@dataclass
class NCARun:
    name: str = "E0_N50"
    state: int = 32
    hidden: int = 128
    n_inner: int = 8
    dilations: tuple = (1, 2, 4)
    fire_rate: float = 1.0
    lr: float = 1e-3
    lr_floor: float = 0.02                 # cosine ends at lr * lr_floor
    warmup: int = 200
    batch: int = 16
    grad_clip: float = 1.0
    weight_decay: float = 0.0
    ema: float = 0.999
    p_mean: float = -1.2                   # EDM: sigma ~ lognormal(p_mean, p_std)
    p_std: float = 1.2
    checkpoint: bool = False               # recompute inner steps in backward (memory for time); needed for big batches on CPU RAM
    train_seed: int = 0


PRESETS = {
    "E0_N50": dict(state=32, hidden=128),
    "E0_N135": dict(state=48, hidden=232),
    "E0_N50_fire": dict(state=32, hidden=128, fire_rate=0.5),
}


def build_run(name):
    return replace(NCARun(name=name), **PRESETS[name])


def lr_at(rc, step, total):
    warm = min(1.0, (step + 1) / max(1, rc.warmup))
    return rc.lr * warm * (rc.lr_floor + (1.0 - rc.lr_floor) * 0.5 * (1 + math.cos(math.pi * step / max(1, total))))


def make_batch(bank, rc, gen, device):
    """rc.batch real frames (train augmentation: random crop + h-flip), model range [-1, 1], (B,3,H,W)."""
    return nca.to_model(bank.sample(rc.batch, 1, generator=gen, train=True)["images"][:, 0].to(device))


def model_config(rc, size):
    return nca.NCAConfig(image_size=size, state=rc.state, hidden=rc.hidden, n_inner=rc.n_inner, dilations=tuple(rc.dilations),
                         fire_rate=rc.fire_rate)


def load_init(model, ema, cfg, path):
    """Continue from a model.pt written by this module. The architecture must match exactly; the optimiser starts fresh."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    want, have = asdict(cfg), asdict(nca.NCAConfig(**{**ck["config"], "dilations": tuple(ck["config"]["dilations"])}))
    diff = {k: (have[k], want[k]) for k in want if have[k] != want[k]}
    if diff:
        raise ValueError(f"--init-from {path}: model config differs from this run's (checkpoint, run): {diff}")
    model.load_state_dict(ck["raw"]); ema.load_state_dict(ck["model"])


def train_run(rc, train_bank, device, total_steps, seed, log_path, log_every=100, ckpt_every=0, init_from=None):
    seed = seed + rc.train_seed
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    cfg = model_config(rc, train_bank.size)
    model = nca.EyeNCA(cfg).to(device)
    ema = nca.make_ema(model)
    if init_from:
        load_init(model, ema, cfg, init_from)
    opt = torch.optim.AdamW(model.parameters(), lr=rc.lr, weight_decay=rc.weight_decay)
    gen = torch.Generator().manual_seed(seed)                    # batches + noise (CPU generator, moved to the device)
    log = open(log_path, "w")
    t0, hist, acc, n_acc = time.time(), [], {}, 0
    model.train()
    for step in range(total_steps):
        x = make_batch(train_bank, rc, gen, device)
        for g in opt.param_groups:
            g["lr"] = lr_at(rc, step, total_steps)
        opt.zero_grad(set_to_none=True)
        loss, per, sigma = nca.edm_loss(model, x, generator=gen, p_mean=rc.p_mean, p_std=rc.p_std, checkpoint_steps=rc.checkpoint)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"[{rc.name}] loss {float(loss)} at step {step + 1} (sigma {sigma.tolist()}); see {log_path}")
        loss.backward()
        gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), rc.grad_clip))      # norm BEFORE clipping
        opt.step()
        nca.ema_update(ema, model, min(rc.ema, (1 + step) / (10 + step)))
        lo, hi = sigma < 0.2, sigma > 1.0
        for k, v in (("loss", float(loss)), ("gnorm", gnorm), ("gclip", float(gnorm > rc.grad_clip)),
                     ("mse_lo", float(per[lo].mean()) if lo.any() else None), ("mse_mid", float(per[~lo & ~hi].mean()) if (~lo & ~hi).any() else None),
                     ("mse_hi", float(per[hi].mean()) if hi.any() else None)):
            if v is not None:
                acc.setdefault(k, []).append(v)
        n_acc += 1
        if (step + 1) % log_every == 0 or step == 0 or step + 1 == total_steps:
            row = dict(step=step + 1, T=1, lr=round(opt.param_groups[0]["lr"], 6), sec=round(time.time() - t0, 1))
            row.update({k + "_avg": round(float(np.mean(v)), 5) for k, v in acc.items() if k not in ("gnorm", "gclip")})
            row.update(gnorm=round(float(np.mean(acc["gnorm"])), 3), gnorm_max=round(float(np.max(acc["gnorm"])), 3),
                       gclip=round(float(np.mean(acc["gclip"])), 3))
            acc, n_acc = {}, 0
            hist.append(row)
            log.write(json.dumps(row) + "\n"); log.flush()
            print(f"[{rc.name}] " + " ".join(f"{k}={v}" for k, v in row.items()), flush=True)
        if ckpt_every and (step + 1) % ckpt_every == 0 and step + 1 < total_steps:
            save_model(Path(log_path).parent / "model_partial.pt", model, ema, cfg, rc, step=step + 1)
    log.close()
    return model, ema, cfg, hist, time.time() - t0


def save_model(path, model, ema, cfg, rc, **extra):
    torch.save({"model": {k: v.cpu() for k, v in ema.state_dict().items()}, "raw": {k: v.cpu() for k, v in model.state_dict().items()},
                "config": asdict(cfg), "run": asdict(rc), **extra}, path)


# ----------------------------------------------------------------------------------------------- evaluation
@torch.no_grad()
def eval_denoise(net, bank, device, n=128, chunk=32, seed=11, sigmas=SIGMAS):
    """Held-out denoising error at fixed noise levels on the same real frames/noise for every call. mse in [0,1] pixel units;
    noisy_mse is the error of doing nothing (sigma^2/4)."""
    net.eval()
    gen = torch.Generator().manual_seed(seed)
    x = nca.to_model(bank.sample(n, 1, generator=gen, train=False)["images"][:, 0]).to(device)
    out = {}
    for s in sigmas:
        noise = nca._randn(x.shape, gen, device)
        d = torch.cat([net.denoise(x[i:i + chunk] + s * noise[i:i + chunk], s) for i in range(0, n, chunk)])
        out[str(s)] = dict(mse=float(((d - x) ** 2).mean() / 4), noisy_mse=s * s / 4.0, ratio_to_noisy=float(((d - x) ** 2).mean() / s ** 2))
    return out


@torch.no_grad()
def sheets(ema, val, out_dir, device, seed=21, n_dream=16, steps=18, repair_sigma=0.6):
    """sheet_dreams.png (4x4 samples from pure noise), sheet_denoise.png (clean | noisy sigma .5 | denoised), sheet_repair.png
    (clean | scatter-damaged | blind repair at sigma_start repair_sigma). Returns timing (seconds per image on this device)."""
    ema.eval()
    gen = torch.Generator().manual_seed(seed)
    t0 = time.time()
    dreams = nca.sample(ema, n_dream, steps=steps, generator=gen, device=device)
    dream_sec = (time.time() - t0) / n_dream
    save_sheet([[dreams[i] for i in range(r * 4, min(r * 4 + 4, n_dream))] for r in range(-(-n_dream // 4))], Path(out_dir) / "sheet_dreams.png")
    clean = val.sample(8, 1, generator=gen, train=False)["images"][:, 0].to(device)
    xm = nca.to_model(clean)
    noisy = xm + 0.5 * nca._randn(xm.shape, gen, device)
    den = nca.to_unit(ema.denoise(noisy, 0.5))
    save_sheet([[clean[i], nca.to_unit(noisy[i]), den[i]] for i in range(8)], Path(out_dir) / "sheet_denoise.png")
    dmg = scatter_batch(clean, generator=gen, max_discs=3, radius=(0.15, 0.25), strength=(0.7, 1.0))
    rep = nca.repair_blind(ema, dmg, repair_sigma, steps=12, generator=gen)
    save_sheet([[clean[i], dmg[i], rep[i]] for i in range(8)], Path(out_dir) / "sheet_repair.png")
    return dict(dream_sec_per_image=round(dream_sec, 3), dream_steps=steps, dream_denoise_calls=2 * steps - 1)


# ----------------------------------------------------------------------------------------------- driver
def make_banks(a, device):
    """(train bank, val bank, {source: val bank}, temporal sources, appearance sources) from the CLI namespace `a` (--cache, --source-weights,
    --image-size, --val-frac, --seed, --tint-mode, --max-videos, --max-frames-per-video). Temporal sources only (E0: single frames)."""
    weights = ({k: float(v) for k, v in (p.split("=") for p in a.source_weights.split(","))} if a.source_weights else {})
    kw = dict(image_size=a.image_size, val_frac=a.val_frac, seed=a.seed, tint_mode=a.tint_mode, max_videos=a.max_videos,
              max_frames_per_video=a.max_frames_per_video, device=device)
    dyn_sources, img_sources = source_kinds(a.cache)
    if not dyn_sources:
        raise SystemExit("no temporal source in --cache")

    def bank(split, srcs, hflip=True):
        w = {s: v for s, v in weights.items() if s in srcs}
        return CacheBank(a.cache, split=split, sources=list(srcs), hflip=hflip, source_weights=w or None, **kw)

    train, val = bank("train", dyn_sources), bank("val", dyn_sources, hflip=False)
    return train, val, {s: bank("val", [s], hflip=False) for s in dyn_sources}, dyn_sources, img_sources


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, nargs="+", help="one or more cache dirs (each with manifest.json)")
    ap.add_argument("--source-weights", default=None, help="share of samples per source, e.g. video=0.5,openeds=0.5")
    ap.add_argument("--out", default="nca_out")
    ap.add_argument("--runs", nargs="+", default=list(PRESETS))
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--image-size", type=int, default=96)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tint-mode", default="gray", choices=["gray", "keep"])
    ap.add_argument("--max-videos", type=int, default=None, help="smoke tests only")
    ap.add_argument("--max-frames-per-video", type=int, default=None, help="smoke tests only")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--eval-frames", type=int, default=128)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--ckpt-every", type=int, default=0, help="save model_partial.pt every N steps (long runs)")
    ap.add_argument("--checkpoint", action="store_true", help="gradient checkpointing over the inner steps (less memory, ~1.3x time)")
    ap.add_argument("--init-from", default=None, help="model.pt to continue from (architecture must match)")
    a = ap.parse_args(argv)
    device = torch.device("cuda" if a.device == "auto" and torch.cuda.is_available() else ("cpu" if a.device == "auto" else a.device))
    torch.set_num_threads(max(1, min(6, torch.get_num_threads())))
    out_root = Path(a.out); out_root.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    train, val, val_by_source, dyn_sources, img_sources = make_banks(a, device)
    print(f"device={device} sources={dyn_sources} (appearance sources not used: {img_sources}) train={train.info()} val={val.info()} "
          f"load={time.time() - t0:.0f}s", flush=True)
    (out_root / "data_info.json").write_text(json.dumps(dict(train=train.info(), val=val.info(), args=vars(a)), indent=1))
    summary = {}
    for name in a.runs:
        run_dir = out_root / name
        if a.skip_existing and (run_dir / "metrics.json").exists():
            summary[name] = json.loads((run_dir / "metrics.json").read_text()); continue
        run_dir.mkdir(exist_ok=True)
        rc = build_run(name)
        rc = replace(rc, **({"batch": a.batch} if a.batch else {}), **({"checkpoint": True} if a.checkpoint else {}))
        model, ema, cfg, hist, secs = train_run(rc, train, device, a.steps, a.seed, run_dir / "train_log.jsonl", a.log_every,
                                              ckpt_every=a.ckpt_every, init_from=a.init_from)
        t1 = time.time()
        metrics = dict(run=asdict(rc), config=asdict(cfg), train_seconds=round(secs, 1), steps_per_sec=round(a.steps / secs, 2),
                       params=nca.count_params(model),
                       heldout_ema=eval_denoise(ema, val, device, n=a.eval_frames),
                       heldout_raw=eval_denoise(model, val, device, n=a.eval_frames),
                       heldout_ema_by_source={s: eval_denoise(ema, b, device, n=a.eval_frames) for s, b in val_by_source.items()},
                       train_fit_ema=eval_denoise(ema, train, device, n=a.eval_frames),
                       sheets=sheets(ema, val_by_source.get("openeds", val), run_dir, device))
        metrics["eval_seconds"] = round(time.time() - t1, 1)
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=1, default=float))
        save_model(run_dir / "model.pt", model, ema, cfg, rc)
        summary[name] = metrics
        print(f"[{name}] done: train {secs:.0f}s, eval {metrics['eval_seconds']}s", flush=True)
    (out_root / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    return summary


if __name__ == "__main__":
    main()
