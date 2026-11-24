"""Same held-out video frames through (a) the world-model checkpoint at every window position t and (b) the plain-AE probe
checkpoints, on CPU. Separates 'RSSM path is worse' from 'evaluation differs'.

  python scripts/diag_wm_vs_ae.py --cache ../data/cache/videos --wm F0=<model.pt> [--wm S1=<model.pt> ...] --ae AE128=<model.pt> ...
Writes diag_wm_vs_ae.json (+ sheet.png: real | AE | WM t=0 | WM t=7) to --out. Small per-video sample: this box has ~3.4 GB RAM.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ivs.aeprobe import PlainAE  # noqa: E402
from ivs.cachedata import CacheBank  # noqa: E402
from ivs.config import ModelConfig  # noqa: E402
from ivs.model import EyeWorldModel  # noqa: E402
from ivs.pilot import mse, save_sheet  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, nargs="+")
    ap.add_argument("--wm", nargs="*", default=[], help="NAME=path/to/model.pt (world model)")
    ap.add_argument("--ae", nargs="*", default=[], help="NAME=path/to/AE_model.pt (plain AE probe)")
    ap.add_argument("--out", default="diag_wm_vs_ae_out")
    ap.add_argument("--frames-per-video", type=int, default=150)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--t", type=int, default=8)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    val = CacheBank(a.cache, split="val", hflip=False, max_frames_per_video=a.frames_per_video, sources=["video"])
    g = torch.Generator().manual_seed(7)
    x = val.sample(a.n, a.t, generator=g, train=False)["images"]            # B,T,3,H,W
    res, tiles = {"n_windows": a.n, "T": a.t, "mean_image_mse": float(((x - x.mean((0, 1), keepdim=True)) ** 2).mean())}, {}
    for name, p in [s.split("=", 1) for s in a.ae]:
        ck = torch.load(p, map_location="cpu")
        spec = ck["spec"]
        m = PlainAE(val.size, spec["code"], spec.get("conv_mult", 1.0), vae=spec.get("vae", 0.0)); m.load_state_dict(ck["model"]); m.eval()
        with torch.no_grad():
            y = torch.stack([m(x[:, i]) for i in range(a.t)], 1)
        res[name] = dict(kind="plain_ae", mse_all=mse(y, x), mse_by_t=[mse(y[:, i], x[:, i]) for i in range(a.t)])
        tiles[name] = y
    for name, p in [s.split("=", 1) for s in a.wm]:
        ck = torch.load(p, map_location="cpu")
        cfg = ModelConfig(**ck["config"]); m = EyeWorldModel(cfg); m.load_state_dict(ck["model"]); m.eval()
        with torch.no_grad():
            torch.manual_seed(0)
            o = m.observe(x, torch.zeros(a.n, a.t, 5))
            y = o["recon"]
            z_only = torch.stack([m.decode(torch.zeros_like(o["h"][:, i]), o["z"][:, i]) for i in range(a.t)], 1)
            single = m.observe(x[:, :1], torch.zeros(a.n, 1, 5))["recon"]
            pm = torch.stack([p[0] for p in o["posterior"]], 1)                 # posterior mean B,T,Z
            ps = torch.stack([p[1] for p in o["posterior"]], 1)                 # posterior std
            mean_z = torch.stack([m.decode(o["h"][:, i], pm[:, i]) for i in range(a.t)], 1)
            from ivs.learning import gaussian_kl
            kl = float(torch.stack([gaussian_kl(o["posterior"][i], o["prior"][i]) for i in range(a.t)], 1).mean())
        res[name] = dict(kind="world_model", mse_all=mse(y, x), mse_by_t=[mse(y[:, i], x[:, i]) for i in range(a.t)],
                         mse_single_step_t0=mse(single, x[:, :1]), mse_z_only_h_zeroed=mse(z_only, x), params=sum(q.numel() for q in m.parameters()),
                         stoch=cfg.stoch_dim, deter=cfg.deter_dim, mse_mean_z_no_noise=mse(mean_z, x), kl_nats=kl,
                         post_std_mean=float(ps.mean()), post_std_min=float(ps.min()), post_mean_abs=float(pm.abs().mean()),
                         active_dims=int(((pm.flatten(0, 1).std(0)) > 0.1).sum()))
        tiles[name] = y
    json.dump(res, open(out / "diag_wm_vs_ae.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
    order = list(tiles)
    rows = [[x[i, -1]] + [tiles[k][i, -1] for k in order] for i in range(8)]
    save_sheet(rows, out / "sheet.png")
    (out / "sheet_columns.txt").write_text("real | " + " | ".join(order) + "   (window position t = last)")


if __name__ == "__main__":
    main()
