"""Record what a free-running dream does step by step (video + dense labelled contact sheets).

Usage: python scripts/record_rollout.py --wm L1=<model.pt> [--wm G4=<model.pt>] --start noise|real --seeds 4 --steps 300 --every 10 --out pilot_results/rollouts
       [--temperature 1.0] [--clamp-z]   (clamp-z: cap each sample's z rms at the rms of the real observed states, see handoff LATEST 13)

--start noise : the UI's start (zero state, 4 seed steps, zero action) like eyemetric.dream_frames.
--start real  : observe a real held-out S_ window (8 frames, zero action), then free-run from its last state.
Per model/start writes
  rollout_<tag>.mp4            every step, the seeds side by side (2 columns), step number burned in
  contact_<tag>_s<k>.png       seed k: frames every --every steps, 8 per row, labelled t<step>
and prints |z| rms / frame std / clipped fraction every 30 steps (mean over the seeds).
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ivs import eyemetric as E  # noqa: E402
from ivs.cachedata import CacheBank  # noqa: E402
from ivs.config import ModelConfig  # noqa: E402
from ivs.model import EyeWorldModel  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TILE = 128


def tile(img, label):
    a = (img.detach().float().clamp(0, 1).cpu().expand(3, -1, -1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    a = cv2.resize(a, (TILE, TILE), interpolation=cv2.INTER_CUBIC)
    cv2.putText(a, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
    return a


@torch.no_grad()
def rollout(model, cfg, start, n, steps, temperature, z_cap, x_real):
    """Frames for t = 0..steps (list of (n,3,H,W)) and the mean z rms per step. z_cap: None or the rms every sample's z is limited to."""
    zero = torch.zeros(n, cfg.action_dim)
    torch.manual_seed(0)
    if start == "real":
        xin = x_real if cfg.channels == 3 else x_real[:, :, :1]
        o = model.observe(xin, torch.zeros(n, xin.shape[1], cfg.action_dim))
        h, z = o["h"][:, -1], o["z"][:, -1]
    else:
        h, z = model.rssm.initial(n, "cpu")
        for _ in range(4):
            h, z, _ = model.rssm.imagine_step(h, z, zero)
    frames, zr = [], []
    torch.manual_seed(1000)
    for _ in range(steps + 1):
        fr = model.decode(h, z)
        frames.append(fr if fr.shape[1] == 3 else fr.expand(-1, 3, -1, -1))
        zr.append(float(z.pow(2).mean().sqrt()))
        h, z, _ = model.rssm.imagine_step(h, z, zero, temperature)
        if z_cap is not None:
            z = z * (z_cap / z.pow(2).mean(1, keepdim=True).sqrt()).clamp(max=1.0)
    return frames, zr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm", nargs="*", action="extend", default=[], help="NAME=path/to/model.pt")
    ap.add_argument("--start", choices=["noise", "real"], default="noise")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--clamp-z", action="store_true")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--cache", default=str(ROOT / "data" / "cache" / "openeds"))
    ap.add_argument("--masks", default=str(ROOT / "data" / "cache" / "openeds_masks"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    val = CacheBank([a.cache], image_size=96, hflip=False, split="val", sources=["openeds"])
    masks = E.aligned_masks(val, a.masks)
    x_all, _, _ = E.eval_windows(val, masks, 64, 8, seed=5)
    for item in a.wm:
        name, path = item.split("=", 1)
        ck = torch.load(path, map_location="cpu")
        cfg = ModelConfig(**ck["config"])
        model = EyeWorldModel(cfg)
        model.load_state_dict(ck["model"])
        model.eval()
        with torch.no_grad():                                          # real-state energy (the clamp level) from 64 real windows
            torch.manual_seed(0)
            xin = x_all if cfg.channels == 3 else x_all[:, :, :1]
            z_real = float(model.observe(xin, torch.zeros(len(xin), 8, cfg.action_dim))["z"].pow(2).mean().sqrt())
        frames, zr = rollout(model, cfg, a.start, a.seeds, a.steps, a.temperature, z_real if a.clamp_z else None, x_all[:a.seeds])
        tag = f"{name}_{a.start}" + ("_clamp" if a.clamp_z else "")
        print(f"[{tag}] real-state |z| rms {z_real:.2f}; every 30 steps:", flush=True)
        for t in range(0, a.steps + 1, 30):
            g = E.gray(frames[t])
            print(f"   t={t:3d} z {zr[t]:.2f} frame_std {float(g.flatten(1).std(1).mean()):.3f} clipped {float(((g < .02) | (g > .98)).float().mean()):.3f}")
        cols = 2 if a.seeds > 1 else 1
        rows_n = int(np.ceil(a.seeds / cols))
        vw = cv2.VideoWriter(str(out / f"rollout_{tag}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (cols * TILE, rows_n * TILE))
        for t in range(a.steps + 1):
            tiles = [tile(frames[t][i], f"t{t}") for i in range(a.seeds)]
            tiles += [np.zeros_like(tiles[0])] * (rows_n * cols - a.seeds)
            vw.write(cv2.cvtColor(np.concatenate([np.concatenate(tiles[r * cols:(r + 1) * cols], 1) for r in range(rows_n)], 0), cv2.COLOR_RGB2BGR))
        vw.release()
        ts = list(range(0, a.steps + 1, a.every))
        for i in range(a.seeds):
            tl = [tile(frames[t][i], f"t{t}") for t in ts]
            tl += [np.zeros_like(tl[0])] * (-len(tl) % 8)
            sheet = np.concatenate([np.concatenate(tl[r * 8:(r + 1) * 8], 1) for r in range(len(tl) // 8)], 0)
            cv2.imwrite(str(out / f"contact_{tag}_s{i}.png"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
        print(f"   wrote rollout_{tag}.mp4 and {a.seeds} contact sheets", flush=True)


if __name__ == "__main__":
    main()
