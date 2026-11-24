"""Tiny synthetic moving-pupil dataset for PIPELINE TESTS ONLY (not real eyes).
Layout: <out>/subject_XX/session_YY/frame_NNNN.png  (subject folder = split group)."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument("--out", default="data/synthetic")
p.add_argument("--subjects", type=int, default=5)
p.add_argument("--sessions", type=int, default=2)
p.add_argument("--frames", type=int, default=32)
p.add_argument("--size", type=int, default=64)
a = p.parse_args()
rng = np.random.default_rng(0)
yy, xx = np.mgrid[0:a.size, 0:a.size] / (a.size - 1) * 2 - 1
for s in range(a.subjects):
    iris_r, bg = rng.uniform(0.45, 0.65), rng.uniform(0.55, 0.85)
    for k in range(a.sessions):
        d = Path(a.out) / f"subject_{s:02d}" / f"session_{k:02d}"
        d.mkdir(parents=True, exist_ok=True)
        ph = rng.uniform(0, 6.28)
        for t in range(a.frames):
            cx, cy = 0.25 * np.sin(0.3 * t + ph), 0.15 * np.cos(0.21 * t + ph)
            pupil_r = 0.18 + 0.05 * np.sin(0.15 * t + ph)
            r = np.hypot(xx - cx, yy - cy)
            img = np.full_like(r, bg)
            img[r < iris_r] = 0.45
            img[r < pupil_r] = 0.05
            Image.fromarray((img * 255).astype(np.uint8)).save(d / f"frame_{t:04d}.png")
print("wrote", a.out)
