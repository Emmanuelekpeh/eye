"""How long does the UI's DREAM stay an eye? Per world model: run the UI's dream loop (ivs.eyemetric.ui_dream_frames = World.tick(dream=True), no
touch, no gaze command; parity-tested against ivs.ui) for --ticks ticks in --worlds independent worlds, segment every frame, flag 'nested eye'
(pupil inside iris), and report dwell statistics:
  eye_frac      share of ticks that look like an eye          eye_frac_first100  same, first 100 ticks (does it START as an eye?)
  episodes      eye runs after dropping 1-tick flicker        median_len / max_len / n_long (runs >= 20 ticks)       colour_frac  chroma > .03
  still_frac    share of ticks where the picture is (nearly) frozen: mean |window-mean(t) - window-mean(t-20)| < STILL (10-tick windows average out the
                prior-noise flicker that every dream has). eye_moving = eye-like AND not still.
CAUTION (viewed): the eye flag alone is fooled by dead dreams: O0 freezes after tick ~100 into one blue/orange blob that the segmenter calls
pupil-in-iris 100% of the time; R4_consistency (30k) and R5 freeze into grey smears also flagged. Their window change is .014-.024; L1 (wandering) is .08-.11.
A stably-held REAL eye is also slow, so still_frac/eye_moving are a conservative cross-check, not a verdict: read the sheets.
Usage: python scripts/eval_ui_dream.py [--wm NAME=path/model.pt ...] [--ticks 1200] [--worlds 4] [--out pilot_results/ui_dream_eval.json]
Without --wm every pilot_results/**/model.pt that is an EyeWorldModel checkpoint is scored. CPU only. The detector is lenient (it counts a blob the
segmenter calls pupil-in-iris), so --sheets writes a contact sheet per model (world 0, a tile every 20 ticks, green frame = flagged): LOOK at them.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from ivs import eyemetric as E                      # noqa: E402
from ivs.config import ModelConfig                  # noqa: E402
from ivs.model import EyeWorldModel                 # noqa: E402

STILL = 0.03        # calibrated on sheets that were looked at (see docstring)


def window_change(fr, win=10, lag=20):
    """(T,n,3,H,W) -> (T,n) change of the 10-tick mean picture vs 20 ticks earlier (0 for the first win+lag ticks = counted as moving)."""
    wm = fr.unfold(0, win, 1).mean(-1)                                       # (T-win+1,n,3,H,W): mean of ticks t-win+1..t
    d = (wm[lag:] - wm[:-lag]).abs().mean((2, 3, 4))                         # (T-win+1-lag, n)
    return torch.cat([torch.full((len(fr) - len(d), fr.shape[1]), float("inf")), d]).numpy()


def discover(root):
    """name = path of the checkpoint folder relative to root (kaggle_gaze1_8000/pilot/G4_cmd_geom_prior)."""
    return {p.parent.relative_to(root).as_posix(): p for p in sorted(Path(root).rglob("model.pt"))}


def score(model, seg, ticks, worlds, noise_seed):
    fr = E.ui_dream_frames(model, ticks=ticks, n=worlds, noise_seed=noise_seed)            # (T,n,3,H,W)
    T, n = fr.shape[:2]
    flat = fr.reshape(T * n, *fr.shape[2:])
    cls = torch.cat([E.segment(seg, E.gray(flat[i:i + 512])) for i in range(0, len(flat), 512)])
    flags = E.nested_flags(cls).reshape(T, n).numpy()
    chroma = ((flat[:, 0] - flat[:, 1]).abs().mean((1, 2)) + (flat[:, 2] - flat[:, 1]).abs().mean((1, 2))).reshape(T, n).numpy()
    still = window_change(fr) < STILL
    per = [E.eye_episodes(flags[:, k]) for k in range(n)]
    first = [float(flags[:100, k].mean()) for k in range(n)]
    agg = {k: float(np.mean([p[k] for p in per])) for k in per[0]}
    agg.update(eye_frac_first100=float(np.mean(first)), still_frac=float(still.mean()), eye_moving=float((flags & ~still).mean()), colour_frac=float((chroma > .03).mean()),
               eye_frac_per_world=[round(p["eye_frac"], 3) for p in per])
    return agg, fr[:, 0], flags[:, 0]


def sheet(frames, flags, path, every=20, cols=10, tile=100):
    tl = []
    for t in range(0, len(frames), every):
        img = (frames[t].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img = np.ascontiguousarray(cv2.resize(img, (tile, tile), interpolation=cv2.INTER_CUBIC))
        if flags[t]:
            cv2.rectangle(img, (0, 0), (tile - 1, tile - 1), (0, 255, 0), 2)
        cv2.putText(img, str(t), (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
        tl.append(img)
    tl += [np.zeros_like(tl[0])] * (-len(tl) % cols)
    cv2.imwrite(str(path), cv2.cvtColor(np.concatenate([np.concatenate(tl[r * cols:(r + 1) * cols], 1) for r in range(len(tl) // cols)], 0), cv2.COLOR_RGB2BGR))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm", nargs="*", action="extend", default=[], help="NAME=path/to/model.pt (default: every checkpoint under --root)")
    ap.add_argument("--root", default=str(HERE / "pilot_results"))
    ap.add_argument("--segmenter", default=str(HERE / "pilot_results" / "eyemetric" / "segmenter.pt"))
    ap.add_argument("--ticks", type=int, default=1200)
    ap.add_argument("--worlds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--sheets", default=None, help="directory for one contact sheet per model (world 0)")
    ap.add_argument("--out", default=str(HERE / "pilot_results" / "ui_dream_eval.json"))
    a = ap.parse_args()
    models = dict(w.split("=", 1) for w in a.wm) if a.wm else discover(a.root)
    seg = E.load_segmenter(a.segmenter)
    res = {}
    if a.sheets:
        Path(a.sheets).mkdir(parents=True, exist_ok=True)
    for name, path in models.items():
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if "config" not in ck or not any(k.startswith("rssm.") for k in ck["model"]):
            print(f"skip {name}: not an EyeWorldModel checkpoint")
            continue
        m = EyeWorldModel(ModelConfig(**ck["config"]))
        m.load_state_dict(ck["model"])
        t0 = time.time()
        res[name], frames, flags = score(m, seg, a.ticks, a.worlds, a.seed)
        res[name].update(path=str(path), image_size=m.cfg.image_size, channels=m.cfg.channels)
        r = res[name]
        print(f"{name:42s} eye {r['eye_frac']:.2f} (first100 {r['eye_frac_first100']:.2f}) episodes {r['n_episodes']:5.1f} median {r['median_len']:4.1f} "
              f"max {r['max_len']:6.1f} long {r['n_long']:4.1f} colour {r['colour_frac']:.2f} still {r['still_frac']:.2f} eye_moving {r['eye_moving']:.2f}  per-world {r['eye_frac_per_world']}  [{time.time() - t0:.0f}s]", flush=True)
        if a.sheets:
            sheet(frames, flags, Path(a.sheets) / f"{name.replace('/', '__')}.png")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(ticks=a.ticks, worlds=a.worlds, seed=a.seed, models=res), indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
