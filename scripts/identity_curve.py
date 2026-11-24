"""Identity curve: at which dream tick does a model stop resembling the eye it started from? (plan.md "DISTILL: identity", phase 0; measurement only, CPU)
Every model dreams (the UI loop, ivs.eyemetric.ui_dream_frames, zero action) from the SAME real 16-frame held-out contexts (CacheBank split='val', one window per
held-out recording = 42 worlds, scripts/make_birth.candidates; --seeds replicates draw other windows, the same protocol as s2_eval.py) and each dream frame is compared with the
context's last real frame:
  retrieval  top-1 accuracy of luma-thumbnail correlation: is the dream at tick t more similar to its OWN context than to the 41 other worlds' contexts? (chance 1/42; the
             thumbnail is 24x24 mean-removed luma, blur-tolerant, chroma-blind, independent of any encoder, so a model that turns a grey eye into a colour prototype is judged on structure)
  mrr        mean reciprocal rank of the own context
  own_vs_other  mse(dream_t, own context) / mean mse(dream_t, the other contexts), luma (< 1 resembles its own start, ~1 no memory); drift = mse to the own context
Reference rows: 'recon' = the model's own render of the context (tick 0, should retrieve ~1), 'real_-8' = the real frame 8 steps before the last (what ordinary change looks like).
  python scripts/identity_curve.py --models L1=<model.pt> adv=<model.pt> ... --out pilot_results/identity_curve [--seeds 0 1 2] [--ticks 300]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
import make_birth as MB                                          # noqa: E402
from ivs import eyemetric as E                                   # noqa: E402
from ivs.birth import perceive                                   # noqa: E402
from ivs.cachedata import CacheBank                              # noqa: E402
from ivs.config import ModelConfig                               # noqa: E402
from ivs.model import EyeWorldModel                              # noqa: E402

TICKS = (1, 2, 5, 10, 15, 20, 30, 45, 60, 100, 200, 299)
THUMB = 24


def luma(x):
    """(..., 3, H, W) -> (..., H, W)"""
    return 0.299 * x[..., 0, :, :] + 0.587 * x[..., 1, :, :] + 0.114 * x[..., 2, :, :]


def thumb(x):
    """(n,3,H,W) -> (n, THUMB*THUMB) mean-removed unit-norm luma thumbnails."""
    t = F.adaptive_avg_pool2d(luma(x)[:, None], THUMB).flatten(1)
    t = t - t.mean(1, keepdim=True)
    return t / t.norm(dim=1, keepdim=True).clamp(min=1e-8)


def identity_metrics(dream, context):
    """dream, context: (n,3,H,W) float; row i of `dream` should resemble row i of `context`. Returns top1, mrr, own_vs_other, drift (floats)."""
    n = len(dream)
    sim = thumb(dream) @ thumb(context).T                                           # (n dream, n context)
    own = sim.diag()
    rank = (sim > own[:, None]).sum(1) + 1                                          # 1 = the own context is the best match
    ld, lc = luma(dream).flatten(1), luma(context).flatten(1)
    d = ((ld[:, None] - lc[None]) ** 2).mean(-1)                                    # (n dream, n context) mse
    mse_own = d.diag()
    other = (d.sum(1) - mse_own) / (n - 1)
    return dict(top1=float((rank == 1).float().mean()), mrr=float((1.0 / rank.float()).mean()),
                own_vs_other=float((mse_own / other.clamp(min=1e-12)).mean()), drift=float(mse_own.mean()))


def curve(frames, context, ticks=TICKS):
    """frames (T,n,3,H,W) -> {tick: identity_metrics} for the ticks that exist (tick 1 = frames[0])."""
    return {str(t): identity_metrics(frames[t - 1], context) for t in ticks if t <= frames.shape[0]}


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = EyeWorldModel(ModelConfig(**ck["config"]))
    model.load_state_dict(ck["model"])
    return model.eval()


def worlds(cache, seeds, context=16, n_max=64):
    """list of (K,T,3,H,W) context windows, one set per seed (one window per held-out recording)."""
    val = CacheBank(cache, image_size=96, split="val", val_frac=0.15, seed=0, tint_mode="gray", sources=["video", "openeds"])
    out = []
    for s in seeds:
        cands = MB.candidates(val, context, n_max, s)
        out.append(torch.stack([val.window_at(c["seq"], c["offset"], context) for c in cands]))
    return out


@torch.no_grad()
def model_curve(model, wins_list, ticks, seed=0):
    """Mean over the world sets of the identity curve of one model, plus the per-set curves and the reference rows."""
    per = []
    for k, wins in enumerate(wins_list):
        p = perceive(model, wins, seed=seed + k)
        fr = E.ui_dream_frames(model, ticks=max(ticks), n=len(wins), noise_seed=seed + 1000 + k, init_state=(p["h"], p["z"]))
        row = curve(fr, wins[:, -1], ticks)
        row["recon"] = identity_metrics(p["recon"].clamp(0, 1).expand(-1, 3, -1, -1), wins[:, -1])
        row["real_-8"] = identity_metrics(wins[:, -8], wins[:, -1])
        per.append(row)
    keys = per[0].keys()
    mean = {k: {m: float(np.mean([r[k][m] for r in per])) for m in per[0][k]} for k in keys}
    return mean, per


def chart(results, ticks, path):
    """retrieval top-1 vs tick, one line per model (simple cv2 plot, no extra dependency)."""
    import cv2
    w, h, m = 900, 520, 60
    img = np.full((h, w, 3), 255, np.uint8)
    xs = {t: m + int((w - 2 * m) * i / (len(ticks) - 1)) for i, t in enumerate(ticks)}
    for y in np.linspace(0, 1, 6):
        yy = h - m - int((h - 2 * m) * y)
        cv2.line(img, (m, yy), (w - m, yy), (225, 225, 225), 1)
        cv2.putText(img, f"{y:.1f}", (12, yy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    for t in ticks:
        cv2.putText(img, str(t), (xs[t] - 10, h - m + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    rng = np.random.RandomState(3)
    for i, (name, r) in enumerate(results.items()):
        col = tuple(int(c) for c in rng.randint(30, 200, 3))
        pts = [(xs[t], h - m - int((h - 2 * m) * r["mean"][str(t)]["top1"])) for t in ticks if str(t) in r["mean"]]
        for a, b in zip(pts, pts[1:]):
            cv2.line(img, a, b, col, 2)
        cv2.putText(img, name[:22], (w - m - 190, m + 14 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
    cv2.putText(img, "own-context retrieval (top-1) vs dream tick; chance ~0.02", (m, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.imwrite(str(path), img)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="name=path ...")
    ap.add_argument("--cache", nargs="+", default=[str(HERE.parent / "data" / "cache" / "videos"), str(HERE.parent / "data" / "cache" / "openeds")])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--ticks", type=int, default=300)
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ticks = tuple(t for t in TICKS if t <= a.ticks)
    wins_list = worlds(a.cache, a.seeds)
    print(f"{len(wins_list)} world sets x {len(wins_list[0])} worlds, ticks {ticks}", flush=True)
    res = {}
    f = out / "identity_curve.json"
    if f.exists():
        res = json.loads(f.read_text())
    for spec in a.models:
        name, _, path = spec.partition("=")
        t0 = time.time()
        mean, per = model_curve(load(path), wins_list, ticks)
        res[name] = dict(path=path, mean=mean, per_set=per)
        print(f"{name:14s} top1 " + " ".join(f"t{t}:{mean[str(t)]['top1']:.2f}" for t in ticks) + f" | own/other t30 {mean['30']['own_vs_other']:.2f} t100 {mean.get('100', {}).get('own_vs_other', float('nan')):.2f} [{time.time() - t0:.0f}s]", flush=True)
        f.write_text(json.dumps(res, indent=1))
    ref = next(iter(res.values()))["mean"]
    print(f"reference: recon top1 {ref['recon']['top1']:.2f}, real_-8 top1 {ref['real_-8']['top1']:.2f}, chance {1 / len(wins_list[0]):.3f}")
    chart(res, ticks, out / "identity_curve.png")
    return res


if __name__ == "__main__":
    main()
