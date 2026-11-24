"""What is different INSIDE the models: which units are alive, what information they carry, and in which situations they fire.

Usage: python scripts/diag_activations.py --wm R1=<model.pt> --wm L1=<model.pt> ... --out pilot_results/diag_activations [--n 1000]

Every model sees the SAME real held-out S_ windows (8 frames, zero action, fixed noise seed); everything is read at the LAST frame.
Ground-truth situation variables come from that frame's real segmentation mask / pixels:
  gaze_x, gaze_y (pupil centroid), pupil_area, lid_open (non-background area = how open the eyelids are), bright (mean luma), blink (no pupil).
Per model and per representation (enc1..enc4 = spatial-mean activation per conv channel, embed = encoder output, h = GRU state, z = posterior mean):
  alive      fraction of units whose std across windows is > 2 % of the layer's mean std (the rest never change with the input)
  eff_dim    participation ratio of the unit covariance, (sum lambda)^2 / sum lambda^2 = how many independent directions are used
  probe_r2   5-fold ridge regression R2 of each situation variable (lambda from {.1, 1, 10, 100} x n_units by CV; equal protocol for all models)
  tuned      units with |Cohen d| >= 1 between the lowest and highest tercile of a variable (gaze/pupil/openness/brightness) or blink vs not;
             count + top-3 unit ids. Unit ids are NOT comparable between models (independently trained), only the counts are.
Latent use: per z unit KL(posterior || prior) at the last frame -> total nats, active units (> 0.1 nat), posterior std.
Writes diag_activations.json and diag_activations.png (probe R2 heat map, alive units, active z units).
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
VARS = ["gaze_x", "gaze_y", "pupil_area", "lid_open", "bright"]


def situation_labels(x_last, masks):
    """x_last (N,3,H,W) real last frames, masks (N,H,W) uint8 (class ids 0 bg, 1 sclera, 2 iris, 3 pupil). -> dict of (N,) arrays + blink flag."""
    n, h, w = masks.shape
    ys, xs = np.mgrid[0:h, 0:w]
    pupil = masks == 3
    area = pupil.sum((1, 2)).astype(np.float64)
    cx = np.where(area > 0, (pupil * xs).sum((1, 2)) / np.maximum(area, 1), np.nan)
    cy = np.where(area > 0, (pupil * ys).sum((1, 2)) / np.maximum(area, 1), np.nan)
    lid = ((masks >= 1) & (masks <= 3)).sum((1, 2)).astype(np.float64)
    bright = E.gray(x_last).mean((1, 2)).numpy().astype(np.float64)
    return dict(gaze_x=cx, gaze_y=cy, pupil_area=area, lid_open=lid, bright=bright), area < E.MIN_PUPIL_PX


def participation_ratio(a):
    a = a - a.mean(0, keepdims=True)
    s = np.linalg.svd(a, compute_uv=False) ** 2
    return float(s.sum() ** 2 / max((s ** 2).sum(), 1e-12))


def ridge_cv_r2(a, y, folds=5, seed=0):
    """Mean held-out R2 of ridge regression (features standardised); lambda picked by CV from a fixed grid."""
    ok = np.isfinite(y)
    a, y = a[ok], y[ok]
    if len(y) < 4 * folds or y.std() < 1e-9:
        return None
    a = (a - a.mean(0)) / (a.std(0) + 1e-6)
    idx = np.random.RandomState(seed).permutation(len(y))
    parts = np.array_split(idx, folds)
    best = -1e9
    for lam in (0.1, 1.0, 10.0, 100.0):
        r2s = []
        for k in range(folds):
            te = parts[k]
            tr = np.concatenate([parts[j] for j in range(folds) if j != k])
            m = y[tr].mean()
            wgt = np.linalg.solve(a[tr].T @ a[tr] + lam * a.shape[1] * np.eye(a.shape[1]), a[tr].T @ (y[tr] - m))
            pred = a[te] @ wgt + m
            r2s.append(1 - ((y[te] - pred) ** 2).sum() / max(((y[te] - y[te].mean()) ** 2).sum(), 1e-12))
        best = max(best, float(np.mean(r2s)))
    return best


def cohen_d(a, low, high):
    """Per-unit Cohen d between two boolean-selected groups of windows."""
    g1, g2 = a[low], a[high]
    if len(g1) < 5 or len(g2) < 5:
        return None
    sd = np.sqrt((g1.var(0) + g2.var(0)) / 2) + 1e-6
    return (g2.mean(0) - g1.mean(0)) / sd


def tuned_report(a, lab, blink):
    out = {}
    for v in VARS:
        y = lab[v]
        ok = np.isfinite(y) & ~blink if v.startswith("gaze") else np.isfinite(y)
        if ok.sum() < 30:
            continue
        lo, hi = np.percentile(y[ok], [33.3, 66.7])
        d = cohen_d(a, ok & (y <= lo), ok & (y >= hi))
        if d is not None:
            out[v] = dict(n_tuned=int((np.abs(d) >= 1).sum()), max_abs_d=round(float(np.abs(d).max()), 2),
                          top=[int(i) for i in np.argsort(-np.abs(d))[:3]])
    d = cohen_d(a, ~blink, blink)
    if d is not None:
        out["blink"] = dict(n_tuned=int((np.abs(d) >= 1).sum()), max_abs_d=round(float(np.abs(d).max()), 2), top=[int(i) for i in np.argsort(-np.abs(d))[:3]],
                            n_blink=int(blink.sum()))
    return out


def unit_stats(a):
    sd = a.std(0)
    return dict(n_units=int(a.shape[1]), alive=round(float((sd > 0.02 * sd.mean()).mean()), 3), eff_dim=round(participation_ratio(a), 1))


@torch.no_grad()
def collect(model, cfg, x):
    """Activations at the last frame for every window: dict name -> (N, units) numpy; plus per-unit KL and posterior std."""
    xin = x if cfg.channels == 3 else x[:, :, :1]
    b, t = xin.shape[:2]
    acts = {}
    hooks = []
    layer_idx = {"enc1": 1, "enc2": 4, "enc3": 7, "enc4": 10}         # the SiLU after each conv stage of ConvEncoder.net

    def mk(name):
        def f(_m, _i, o):
            acts[name] = o.reshape(b, t, *o.shape[1:])[:, -1].mean((2, 3)).numpy()      # spatial mean per channel, last frame
        return f
    for name, i in layer_idx.items():
        hooks.append(model.encoder.net[i].register_forward_hook(mk(name)))
    torch.manual_seed(0)
    o = model.observe(xin, torch.zeros(b, t, cfg.action_dim))
    for hk in hooks:
        hk.remove()
    pm, ps = o["posterior"][-1]
    qm, qs = o["prior"][-1]
    kl = (torch.log(qs / ps) + (ps ** 2 + (pm - qm) ** 2) / (2 * qs ** 2) - 0.5).mean(0).numpy()      # KL(post || prior) per unit, mean over windows
    embed = model.encoder(xin[:, -1])
    acts.update(embed=embed.numpy(), h=o["h"][:, -1].numpy(), z=pm.numpy())
    return acts, kl, float(ps.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm", nargs="*", action="extend", default=[], help="NAME=path/to/model.pt (order = order in the figure)")
    ap.add_argument("--cache", default=str(ROOT / "data" / "cache" / "openeds"))
    ap.add_argument("--masks", default=str(ROOT / "data" / "cache" / "openeds_masks"))
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "pilot_results" / "diag_activations"))
    a = ap.parse_args()

    val = CacheBank([a.cache], image_size=96, hflip=False, split="val", sources=["openeds"])
    masks = E.aligned_masks(val, a.masks)
    x, m_last, pick = E.eval_windows(val, masks, a.n, 8, seed=11)
    lab, blink = situation_labels(x[:, -1], m_last.numpy())
    print(f"{len(x)} windows | blink (no pupil) {int(blink.sum())} | pupil area median {np.nanmedian(lab['pupil_area']):.0f}px | "
          f"gaze_x {np.nanpercentile(lab['gaze_x'], 5):.0f}..{np.nanpercentile(lab['gaze_x'], 95):.0f}px", flush=True)
    res = {}
    for item in a.wm:
        name, path = item.split("=", 1)
        ck = torch.load(path, map_location="cpu")
        cfg = ModelConfig(**ck["config"])
        model = EyeWorldModel(cfg)
        model.load_state_dict(ck["model"])
        model.eval()
        acts, kl, post_std = collect(model, cfg, x)
        acts["h+z"] = np.concatenate([acts["h"], acts["z"]], 1)
        r = dict(model=path, steps=ck.get("step"), kl_total_nats=round(float(kl.sum()), 2), active_z_units=int((kl > 0.1).sum()), z_units=int(len(kl)),
                 post_std_mean=round(post_std, 3), kl_per_unit_top=[round(float(v), 2) for v in np.sort(kl)[::-1][:5]], layers={})
        for layer, arr in acts.items():
            lr = dict(unit_stats(arr) if layer != "h+z" else {})
            lr["probe_r2"] = {v: (None if (s := ridge_cv_r2(arr, lab[v])) is None else round(s, 3)) for v in VARS}
            if layer != "h+z":
                lr["tuned"] = tuned_report(arr, lab, blink)
            r["layers"][layer] = lr
        res[name] = r
        z, hz = r["layers"]["z"], r["layers"]["h+z"]
        print(f"[{name}] KL {r['kl_total_nats']} nats, active z {r['active_z_units']}/{r['z_units']} | z alive {z['alive']} eff_dim {z['eff_dim']} | "
              f"probe R2 from h+z: " + " ".join(f"{v}={hz['probe_r2'][v]}" for v in VARS), flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(dict(n_windows=len(x), n_blink=int(blink.sum()), models=res), open(out / "diag_activations.json", "w"), indent=1)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    names = list(res)
    fig, ax = plt.subplots(1, 3, figsize=(17, 0.55 * len(names) + 3.2), gridspec_kw=dict(width_ratios=[1.3, 1, 1]))
    grid = np.array([[res[n]["layers"]["h+z"]["probe_r2"][v] or 0 for v in VARS] for n in names])
    im = ax[0].imshow(grid, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax[0].set_xticks(range(len(VARS)), VARS, rotation=30); ax[0].set_yticks(range(len(names)), names)
    for i in range(len(names)):
        for j in range(len(VARS)):
            ax[0].text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", color="w" if grid[i, j] < .6 else "k")
    ax[0].set_title("held-out R2 of the situation, read linearly from (h, z)"); fig.colorbar(im, ax=ax[0])
    ax[1].barh(names, [res[n]["active_z_units"] for n in names], color="tab:orange"); ax[1].set_title("z units carrying information (KL > 0.1 nat)")
    ax[1].set_xlim(0, max(res[n]["z_units"] for n in names))
    ax[2].barh(names, [res[n]["kl_total_nats"] for n in names], color="tab:green"); ax[2].set_title("total KL(post||prior) at a frame (nats)")
    for a_ in ax[1:]:
        a_.invert_yaxis()
    fig.tight_layout(); fig.savefig(out / "diag_activations.png", dpi=110)


if __name__ == "__main__":
    main()
