"""R4 baseline: how long does the CURRENT model (L1_v6_long, 96 px RSSM) keep an eye when it dreams from a real held-out context?
Same procedure as make_birth.py's honesty check but on EVERY candidate (no selection): one window of --context frames per held-out recording
(CacheBank split='val', the pilot split), perceive it with zero action, dream --ticks UI ticks (ivs.eyemetric.ui_dream_frames), score with the eye
segmenter exactly like make_birth._stats (eye_frac / eye_moving / still_frac / first100 / max_len / n_long). The R4 dreams are scored with the SAME
function (r4_latent.score_dreams) so the numbers are comparable. CPU only. Writes <out>.json (+ a sheet of 8 dreams).
  python scripts/r4_baseline.py --out pilot_results/r4/baseline_L1.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
from ivs import eyemetric as E                                   # noqa: E402
from ivs.birth import perceive                                   # noqa: E402
from ivs.cachedata import CacheBank                              # noqa: E402
from ivs.config import ModelConfig                               # noqa: E402
from ivs.model import EyeWorldModel                              # noqa: E402
import make_birth as MB                                          # noqa: E402

KEYS = ("eye_frac", "eye_moving", "still_frac", "first100", "max_len", "n_long")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(HERE / "pilot_results" / "kaggle_long3_30k" / "pilot" / "L1_v6_long" / "model.pt"))
    ap.add_argument("--cache", nargs="+", default=[str(HERE.parent / "data" / "cache" / "videos"), str(HERE.parent / "data" / "cache" / "openeds")])
    ap.add_argument("--segmenter", default=str(HERE / "pilot_results" / "eyemetric" / "segmenter.pt"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--context", type=int, default=16)
    ap.add_argument("--ticks", type=int, default=300)
    ap.add_argument("--candidates", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists; nothing is overwritten")
    t0 = time.time()
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ck["config"])
    model = EyeWorldModel(cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    seg = E.load_segmenter(a.segmenter)
    bank = CacheBank(a.cache, image_size=cfg.image_size, split="val", val_frac=0.15, seed=0, tint_mode="gray", sources=["video", "openeds"])
    cands = MB.candidates(bank, a.context, a.candidates, a.seed)
    wins = torch.stack([bank.window_at(c["seq"], c["offset"], a.context) for c in cands])
    p = perceive(model, wins, seed=a.seed)
    fr, st = MB.dwell(model, seg, (p["h"], p["z"]), a.ticks, noise_seed=a.seed + 1000)
    row = {k: round(float(np.mean(st[k])), 3) for k in KEYS}
    by_src = {s: {k: round(float(np.mean([st[k][i] for i, c in enumerate(cands) if c["source"] == s])), 3) for k in KEYS} for s in sorted({c["source"] for c in cands})}
    res = dict(model=a.checkpoint, context=a.context, ticks=a.ticks, n=len(cands), mean=row, by_source=by_src, groups=[c["group"] for c in cands],
               per_world={k: [round(float(v), 3) for v in st[k]] for k in KEYS}, seconds=round(time.time() - t0))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(dict(n=res["n"], mean=row, by_source=by_src, seconds=res["seconds"]), indent=1), flush=True)


if __name__ == "__main__":
    main()
