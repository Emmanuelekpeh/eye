"""Do the dynamics add anything over 'nothing changes'? Fair persistence baselines at the model's OWN fidelity.

Usage: python scripts/diag_dynamics.py --wm L1=<model.pt> [--wm G0=<model.pt>] --out pilot_results/diag_dynamics.json

Real held-out S_ windows: 8 context frames (zero action), then horizon H real future frames. For each k in --ks, MSE (and pupil-centroid
error from the eye segmenter, in px) of the prediction of REAL frame k steps after the context, by
  real_copy     the real last context frame held still (not available to a model: it needs a perfect sharp copy of the input)
  recon_copy    the model's own reconstruction of the last context frame held still  = persistence AT THE MODEL'S FIDELITY (the fair baseline)
  teacher       the model observing the real frame k (posterior reconstruction): the fidelity floor of this model at that frame
  free_sample   open-loop imagination, z sampled from the prior each step (what the UI / dreams do), mean over --seeds seeds
  free_mean     open-loop imagination with z = prior mean (no sampling noise)
  gt_motion     (pupil only) how far the real pupil centroid moved between the last context frame and frame k, i.e. the error of 'pupil stays'
A model's dynamics only matter where free_* beats recon_copy; the gap to `teacher` is what the dynamics could still win.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ivs import eyemetric as E  # noqa: E402
from ivs.cachedata import CacheBank  # noqa: E402
from ivs.config import ModelConfig  # noqa: E402
from ivs.model import EyeWorldModel  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CTX = 8


def pupil_centroids(cls):
    """(N,H,W) class ids -> (N,2) pupil centroid in px (nan when fewer than MIN_PUPIL_PX pupil pixels)."""
    p = (cls == 3)
    n = p.sum((1, 2)).float()
    h, w = cls.shape[1:]
    ys, xs = torch.meshgrid(torch.arange(h).float(), torch.arange(w).float(), indexing="ij")
    cx = (p * xs).sum((1, 2)) / n.clamp(min=1)
    cy = (p * ys).sum((1, 2)) / n.clamp(min=1)
    c = torch.stack([cx, cy], 1)
    c[n < E.MIN_PUPIL_PX] = float("nan")
    return c


def mse(a, b):
    return float(((a - b) ** 2).mean())


def centroid_err(seg, pred_rgb, gt_mask):
    """Median / mean px distance between the pupil centroid of a predicted frame and the ground-truth mask's (windows with both)."""
    pc = pupil_centroids(E.segment(seg, E.gray(pred_rgb)))
    gc = pupil_centroids(gt_mask)
    d = (pc - gc).norm(dim=1)
    ok = torch.isfinite(d)
    return (float(d[ok].median()) if ok.any() else None), float(ok.float().mean())


@torch.no_grad()
def run_model(model, cfg, x, masks_k, ks, seg, seeds):
    n, t = x.shape[:2]
    xin = x if cfg.channels == 3 else x[:, :, :1]
    three = lambda r: r if r.shape[1] == 3 else r.expand(-1, 3, -1, -1)  # noqa: E731
    zero = lambda m: torch.zeros(n, m, cfg.action_dim)  # noqa: E731
    torch.manual_seed(0)
    full = model.observe(xin, zero(t))                                  # teacher-forced over context + future
    ctx = model.observe(xin[:, :CTX], zero(CTX))
    h0, z0 = ctx["h"][:, -1], ctx["z"][:, -1]
    recon_copy = three(model.decode(h0, z0))
    horizon = max(ks)
    free = {s: {} for s in ("sample", "mean")}
    for mode in ("sample", "mean"):
        for seed in (seeds if mode == "sample" else [0]):
            torch.manual_seed(100 + seed)
            h, z = h0, z0
            for k in range(1, horizon + 1):
                h = model.rssm.gru(torch.cat([z, torch.zeros(n, cfg.action_dim)], -1), h)
                mean, raw = model.rssm.prior(h).chunk(2, -1)
                std = torch.nn.functional.softplus(raw) + cfg.min_std
                z = mean + std * torch.randn_like(std) if mode == "sample" else mean
                if k in ks:
                    free[mode].setdefault(k, []).append(three(model.decode(h, z)))
    out = {}
    for k in ks:
        real_k = x[:, CTX - 1 + k]
        row = dict(
            real_copy=mse(x[:, CTX - 1], real_k), recon_copy=mse(recon_copy, real_k),
            teacher=mse(three(full["recon"][:, CTX - 1 + k]), real_k),
            free_sample=float(np.mean([mse(f, real_k) for f in free["sample"][k]])), free_mean=mse(free["mean"][k][0], real_k))
        gt = masks_k[k]
        for name, fr in (("recon_copy", recon_copy), ("teacher", three(full["recon"][:, CTX - 1 + k])), ("free_sample", free["sample"][k][0]),
                         ("free_mean", free["mean"][k][0])):
            c, found = centroid_err(seg, fr, gt)
            row[f"pupil_px_{name}"] = c
            row[f"pupil_found_{name}"] = round(found, 2)
        out[k] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm", nargs="*", action="extend", default=[], help="NAME=path/to/model.pt")
    ap.add_argument("--cache", default=str(ROOT / "data" / "cache" / "openeds"))
    ap.add_argument("--masks", default=str(ROOT / "data" / "cache" / "openeds_masks"))
    ap.add_argument("--segmenter", default=str(Path(__file__).resolve().parents[1] / "pilot_results" / "eyemetric" / "segmenter.pt"))
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--ks", type=int, nargs="*", default=[1, 2, 4, 8, 16])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    seg = E.load_segmenter(a.segmenter)
    val = CacheBank([a.cache], image_size=96, hflip=False, split="val", sources=["openeds"])
    masks = E.aligned_masks(val, a.masks)
    horizon = max(a.ks)
    x, _m, pick = E.eval_windows(val, masks, a.n, CTX + horizon, seed=7)
    masks_k = {k: torch.from_numpy(masks[pick - horizon + k]) for k in range(0, horizon + 1)}
    gt0 = pupil_centroids(masks_k[0])
    res = {"n_windows": len(x), "gt_motion_px": {}, "models": {}}
    for k in a.ks:
        d = (pupil_centroids(masks_k[k]) - gt0).norm(dim=1)
        res["gt_motion_px"][k] = float(d[torch.isfinite(d)].median())
    print("real pupil motion since the last context frame (median px): " + " ".join(f"k={k}:{v:.2f}" for k, v in res["gt_motion_px"].items()))
    # headroom check on GROUND TRUTH only: is passive pupil motion predictable at all? constant velocity from the last 3 context frames
    c_prev = pupil_centroids(torch.from_numpy(masks[pick - horizon - 2]))
    vel = (gt0 - c_prev) / 2.0
    res["gt_velocity_extrapolation_px"] = {}
    for k in a.ks:
        gk = pupil_centroids(masks_k[k])
        e_pers, e_vel = (gk - gt0).norm(dim=1), (gk - (gt0 + vel * k)).norm(dim=1)
        ok = torch.isfinite(e_pers) & torch.isfinite(e_vel)
        res["gt_velocity_extrapolation_px"][k] = dict(persistence=float(e_pers[ok].median()), const_velocity=float(e_vel[ok].median()))
    print("ground-truth pupil error (median px), persistence vs constant-velocity extrapolation: " +
          " ".join(f"k={k}: {v['persistence']:.2f} vs {v['const_velocity']:.2f}" for k, v in res["gt_velocity_extrapolation_px"].items()))
    for item in a.wm:
        name, path = item.split("=", 1)
        ck = torch.load(path, map_location="cpu")
        cfg = ModelConfig(**ck["config"])
        model = EyeWorldModel(cfg)
        model.load_state_dict(ck["model"])
        model.eval()
        r = run_model(model, cfg, x, masks_k, a.ks, seg, a.seeds)
        res["models"][name] = r
        print(f"\n[{name}] MSE vs the real frame k steps after the context   (lower is better; fair baseline = recon_copy)")
        print("   k   real_copy  recon_copy   teacher  free_sample  free_mean | pupil px err: recon_copy teacher free_sample free_mean")
        for k, row in r.items():
            f = lambda v: "  n/a" if v is None else f"{v:5.2f}"  # noqa: E731
            print(f"  {k:2d}   {row['real_copy']:.5f}   {row['recon_copy']:.5f}   {row['teacher']:.5f}    {row['free_sample']:.5f}   {row['free_mean']:.5f} |"
                  f"   {f(row['pupil_px_recon_copy'])}      {f(row['pupil_px_teacher'])}   {f(row['pupil_px_free_sample'])}      {f(row['pupil_px_free_mean'])}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
