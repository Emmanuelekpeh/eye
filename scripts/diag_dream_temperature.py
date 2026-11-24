"""Do dreams survive when the prior's sampling noise is turned down? (inference-time only, no retraining)

Usage: python scripts/diag_dream_temperature.py --wm L1=<model.pt> [--wm G4=<model.pt>] [--temps 1 .6 .3 0] --out pilot_results/diag_dream_temp

From the UI's noise start (eyemetric.dream_frames), the free-run uses z = prior_mean + T * prior_std * eps with T from --temps.
Per model and temperature, at t = 10/30/100/300: eye structure of the dreamed frames (segmenter: pupil present, nested eye), plus GT-free image
statistics (std inside the frame, fraction of clipped pixels, and the mean change between frames 10 steps apart = does it move at all or freeze).
Writes diag_dream_temp.json and sheet_<model>.png (blocks of 3 seeds per temperature; columns t = 0,10,20,30,100,300).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ivs import eyemetric as E  # noqa: E402
from ivs.config import ModelConfig  # noqa: E402
from ivs.model import EyeWorldModel  # noqa: E402
from ivs.pilot import save_sheet  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm", nargs="*", action="extend", default=[], help="NAME=path/to/model.pt")
    ap.add_argument("--segmenter", default=str(Path(__file__).resolve().parents[1] / "pilot_results" / "eyemetric" / "segmenter.pt"))
    ap.add_argument("--temps", type=float, nargs="*", default=[1.0, 0.6, 0.3, 0.0])
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    seg = E.load_segmenter(a.segmenter)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    times = (0, 10, 20, 30, 100, 300)
    res = {}
    for item in a.wm:
        name, path = item.split("=", 1)
        ck = torch.load(path, map_location="cpu")
        model = EyeWorldModel(ModelConfig(**ck["config"]))
        model.load_state_dict(ck["model"])
        model.eval()
        res[name], rows = {}, []
        for temp in a.temps:
            fr = E.dream_frames(model, times=times, n=a.n, temperature=temp)
            stats = {}
            for t in (10, 30, 100, 300):
                g = E.gray(fr[t])
                st = E.structure_stats(E.segment(seg, g))
                stats[t] = dict(nested_eye=round(st["nested_eye"], 3), pupil_present=round(st["pupil_present"], 3),
                                pupil_area=st["pupil_area_median"], iris_area=st["iris_area_median"],
                                frame_std=round(float(g.flatten(1).std(1).mean()), 3),
                                clipped=round(float(((g < 0.02) | (g > 0.98)).float().mean()), 3))
            stats["change_10_to_30"] = round(float((E.gray(fr[30]) - E.gray(fr[10])).abs().mean()), 4)
            stats["change_30_to_100"] = round(float((E.gray(fr[100]) - E.gray(fr[30])).abs().mean()), 4)
            res[name][str(temp)] = stats
            print(f"[{name}] T={temp}: nested t10/30/100/300 = " + " ".join(f"{stats[t]['nested_eye']:.2f}" for t in (10, 30, 100, 300)) +
                  " | frame_std " + " ".join(f"{stats[t]['frame_std']:.3f}" for t in (10, 30, 100, 300)) +
                  " | clipped@300 " + f"{stats[300]['clipped']:.3f} | change 10->30 {stats['change_10_to_30']} 30->100 {stats['change_30_to_100']}", flush=True)
            for i in range(3):
                rows.append([fr[t][i] for t in times])
        save_sheet(rows, out / f"sheet_{name}.png")
    json.dump(dict(temps=a.temps, times=times, models=res), open(out / "diag_dream_temp.json", "w"), indent=1)


if __name__ == "__main__":
    main()
