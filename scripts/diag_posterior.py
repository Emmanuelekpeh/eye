"""Posterior-collapse check: does the posterior carry the frame's information into the latent, and does the decoder use it?

Usage: python scripts/diag_posterior.py --wm L1=<model.pt> --wm R1=<model.pt> ... [--n 256] [--out pilot_results/diag_posterior.json]
Real held-out windows (8 frames, zero action, fixed seed), read at the LAST frame, per source (openeds S_ frames and the video cache separately,
because the failure mode of interest appears on heterogeneous / mixed data). Per model and source:
  kl_nats / active_units      KL(post||prior) summed over the latent (the training objective's number) and units above 0.1 nat; collapse = ~0
  mean_shift_over_prior_std   mean over units of |mu_post - mu_prior| / sigma_prior : 'the posterior mean barely differs from the prior' if ~0
  std_ratio                   mean sigma_post / sigma_prior : ~1 = posterior is as noisy as the prior (no information)
  recon_*                     teacher-forced last-frame MSE with the model's own sampled posterior latent (full) and with the decoder input changed:
                              z_prior (z sampled from the prior given h), z_shuffled / h_shuffled (taken from another window), z_zero, h_zero;
                              mean_image = predicting the train-set mean picture. decoder_uses_z = recon_z_shuffled / recon_full (>>1 = the decoder
                              depends on THIS window's z; ~1 = it ignores it), same for h.
  motion_corr                 Pearson r over windows between ||mu_post(t) - mu_post(t-1)|| and mean |frame(t) - frame(t-1)| (is movement in the latent?)
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


def windows(n, t=8, seed=11):
    """{source: (N,t,3,H,W)} real held-out windows (same ones for every model)."""
    out = {}
    val = CacheBank([str(ROOT / "data" / "cache" / "openeds")], image_size=96, hflip=False, split="val", sources=["openeds"])
    out["openeds"] = E.eval_windows(val, E.aligned_masks(val, str(ROOT / "data" / "cache" / "openeds_masks")), n, t, seed=seed)[0]
    vid = CacheBank([str(ROOT / "data" / "cache" / "videos")], image_size=96, hflip=False, split="val", sources=["video"])
    out["video"] = vid.sample(n, t, generator=torch.Generator().manual_seed(seed), train=False)["images"]
    return out


def mse(a, b):
    return float(((a - b) ** 2).mean())


@torch.no_grad()
def check(model, cfg, x, mean_img):
    xin = x if cfg.channels == 3 else x[:, :, :1]
    mi = mean_img if cfg.channels == 3 else mean_img[:1]
    b, t = xin.shape[:2]
    torch.manual_seed(0)
    o = model.observe(xin, torch.zeros(b, t, cfg.action_dim))
    (pm, ps), (qm, qs) = o["posterior"][-1], o["prior"][-1]
    kl = (torch.log(qs / ps) + (ps ** 2 + (pm - qm) ** 2) / (2 * qs ** 2) - 0.5).mean(0)
    h, z, target = o["h"][:, -1], o["z"][:, -1], xin[:, -1]
    perm = torch.roll(torch.arange(b), 1)
    torch.manual_seed(1)
    z_prior = qm + qs * torch.randn_like(qs)
    dec = lambda hh, zz: mse(model.decode(hh, zz), target)
    full = dec(h, z)
    rec = dict(full=full, z_prior=dec(h, z_prior), z_shuffled=dec(h, z[perm]), h_shuffled=dec(h[perm], z), z_zero=dec(h, torch.zeros_like(z)),
               h_zero=dec(torch.zeros_like(h), z), mean_image=mse(mi[None].expand_as(target), target))
    dz = (o["posterior"][-1][0] - o["posterior"][-2][0]).norm(dim=-1).numpy()
    dx = (xin[:, -1] - xin[:, -2]).abs().mean((1, 2, 3)).numpy()
    return dict(kl_nats=round(float(kl.sum()), 2), active_units=int((kl > 0.1).sum()), z_units=int(kl.numel()),
                mean_shift_over_prior_std=round(float(((pm - qm).abs() / qs).mean()), 3), std_ratio=round(float((ps / qs).mean()), 3),
                post_std=round(float(ps.mean()), 3), prior_std=round(float(qs.mean()), 3),
                **{f"recon_{k}": round(v, 5) for k, v in rec.items()},
                decoder_uses_z=round(rec["z_shuffled"] / full, 2), decoder_uses_h=round(rec["h_shuffled"] / full, 2),
                recon_vs_mean_image=round(rec["mean_image"] / full, 2),
                motion_corr=round(float(np.corrcoef(dz, dx)[0, 1]), 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm", nargs="*", action="extend", default=[], help="NAME=path/to/model.pt")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "pilot_results" / "diag_posterior.json"))
    a = ap.parse_args()
    wins = windows(a.n)
    train_mean = CacheBank([str(ROOT / "data" / "cache" / "openeds"), str(ROOT / "data" / "cache" / "videos")], image_size=96, hflip=False,
                           split="train").sample(512, 1, generator=torch.Generator().manual_seed(0), train=False)["images"][:, 0].mean(0)
    res = {}
    for item in a.wm:
        name, path = item.split("=", 1)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ModelConfig(**ck["config"])
        model = EyeWorldModel(cfg)
        model.load_state_dict(ck["model"])
        model.eval()
        res[name] = {src: check(model, cfg, x, train_mean) for src, x in wins.items()}
        for src, r in res[name].items():
            print(f"[{name:10s} {src:7s}] KL {r['kl_nats']:7.1f} nats, active {r['active_units']:3d}/{r['z_units']:3d} | shift/σp {r['mean_shift_over_prior_std']:.2f} "
                  f"σq/σp {r['std_ratio']:.2f} | recon {r['recon_full']:.4f} (mean-image x{r['recon_vs_mean_image']}) "
                  f"z-shuffled x{r['decoder_uses_z']} h-shuffled x{r['decoder_uses_h']} z=0 x{r['recon_z_zero'] / r['recon_full']:.1f} | motion r {r['motion_corr']}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
