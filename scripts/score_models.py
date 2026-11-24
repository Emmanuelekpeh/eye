"""Which trained world models can at least produce an eye? (feeds the UI model dropdown filter; plan.md "MODEL FILTER")

For every checkpoint ivs.modelzoo lists: start `--worlds` UI worlds from NOISE (what the page does by default), run the UI's own dream loop for
`--ticks` ticks (ivs.eyemetric.ui_dream_frames: predict -> render -> perceive the render, parity-tested against ivs.ui.World), score every shown
frame with the eye segmenter (nested pupil-in-iris, scripts/make_birth._stats; LENIENT, so the sheets decide) and store per model:
  eye_frac (share of ticks flagged eye), first100 (share of the first 100 ticks), best100 (a world's best 100-tick window share, max over the
  worlds' eye_frac of any 100-tick run), max_len, any_eye (share of worlds with >= 20 consecutive eye ticks), still_frac, chroma.
Models whose image size is not 96 are resized to 96 for the segmenter (it was trained at 96 px). Results go to pilot_results/model_scores.json
(resumable: a model is skipped when its file size matches the stored record; --force rescores), plus a strip PNG per model (3 worlds x ticks
tick 100 and last, 4 worlds, 96 px) in pilot_results/model_scores/. `--sheets` builds contact sheets (sorted by eye_frac) from the strips so they can be LOOKED at.
CPU only. Usage: python scripts/score_models.py [--worlds 10 --ticks 200] ; python scripts/score_models.py --sheets
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
from ivs import eyemetric as E                      # noqa: E402
from ivs import modelzoo                            # noqa: E402
from ivs.config import ModelConfig                  # noqa: E402
from ivs.model import EyeWorldModel                 # noqa: E402
from make_birth import _stats                       # noqa: E402

SCORES = HERE / "pilot_results" / "model_scores.json"
STRIPS = HERE / "pilot_results" / "model_scores"
SEG = HERE / "pilot_results" / "eyemetric" / "segmenter.pt"
MARKS = (1, 30, 100)


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = EyeWorldModel(ModelConfig(**ck["config"]))
    m.load_state_dict(ck["model"])
    return m.eval()


def to96(fr):
    """(T,n,3,H,W) -> 96 px (the segmenter's size)."""
    if fr.shape[-1] == 96:
        return fr
    T, n = fr.shape[:2]
    return F.interpolate(fr.flatten(0, 1), size=(96, 96), mode="bilinear", align_corners=False).reshape(T, n, 3, 96, 96)


def best_window(flags, win=100):
    """max over worlds of the share of eye ticks in any `win`-tick window (flags (T,n) bool)."""
    T = flags.shape[0]
    if T <= win:
        return float(flags.mean(0).max())
    c = np.concatenate([np.zeros((1, flags.shape[1])), np.cumsum(flags, 0)])
    return float(((c[win:] - c[:-win]) / win).max())


def score(model, seg, worlds, ticks, seed):
    fr = to96(E.ui_dream_frames(model, ticks=ticks, n=worlds, noise_seed=seed))
    st = _stats(fr, seg)
    chroma = float((fr.max(2).values - fr.min(2).values).mean())
    return fr, dict(eye_frac=float(st["eye_frac"].mean()), first100=float(st["first100"].mean()), best100=best_window(st["flags"]),
                    max_len=float(st["max_len"].mean()), any_eye=float((st["max_len"] >= 20).mean()), still_frac=float(st["still_frac"].mean()),
                    chroma=chroma)


def strip(fr, worlds=4, tile=96):
    """One row: the first `worlds` worlds at tick 100 then at the last tick, at 96 px (what the UI shows), so a sheet can be read by eye."""
    T = fr.shape[0]
    marks = [min(100, T - 1), T - 1]
    n = min(worlds, fr.shape[1])
    tiles = [cv2.resize((fr[t, r].permute(1, 2, 0).numpy() * 255).astype(np.uint8), (tile, tile), interpolation=cv2.INTER_AREA) for t in marks for r in range(n)]
    return np.concatenate(tiles, 1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=10)
    ap.add_argument("--ticks", type=int, default=200)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--only", default=None, help="regex on the model id")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default=str(SCORES))
    ap.add_argument("--strips", default=str(STRIPS))
    ap.add_argument("--sheets", action="store_true", help="build contact sheets from the stored strips instead of scoring")
    ap.add_argument("--per-sheet", type=int, default=15)
    a = ap.parse_args(argv)
    out, strips = Path(a.out), Path(a.strips)
    res = json.loads(out.read_text()) if out.exists() else {}
    if a.sheets:
        return sheets(res, strips, a.per_sheet)
    seg = E.load_segmenter(str(SEG))
    strips.mkdir(parents=True, exist_ok=True)
    zoo = [e for e in modelzoo.discover(HERE) if not a.only or re.search(a.only, e["id"])]
    print(f"{len(zoo)} models, {a.worlds} worlds x {a.ticks} ticks from noise (seed {a.seed})", flush=True)
    t0 = time.time()
    for i, e in enumerate(zoo):
        old = res.get(e["id"])
        if old and old.get("bytes") == e["bytes"] and old.get("worlds") == a.worlds and old.get("ticks") == a.ticks and not a.force:
            continue
        t1 = time.time()
        fr, s = score(load(HERE / e["id"]), seg, a.worlds, a.ticks, a.seed)
        res[e["id"]] = dict(**s, bytes=e["bytes"], worlds=a.worlds, ticks=a.ticks, seed=a.seed, label=e["label"], group=e["group"], size=e["size"],
                            seconds=round(time.time() - t1, 1))
        cv2.imwrite(str(strips / (re.sub(r"[^A-Za-z0-9_.-]", "_", e["id"]) + ".png")), cv2.cvtColor(strip(fr), cv2.COLOR_RGB2BGR))
        out.write_text(json.dumps(res, indent=1))
        print(f"[{i + 1}/{len(zoo)}] {e['id']}: eye_frac {s['eye_frac']:.2f} first100 {s['first100']:.2f} best100 {s['best100']:.2f} max_len {s['max_len']:.0f} "
              f"any_eye {s['any_eye']:.2f} still {s['still_frac']:.2f} chroma {s['chroma']:.3f}  [{res[e['id']]['seconds']}s, total {time.time() - t0:.0f}s]", flush=True)
    return res


def sheets(res, strips, per_sheet):
    rows = sorted(res.items(), key=lambda kv: -kv[1]["eye_frac"])
    paths = []
    for s in range(0, len(rows), per_sheet):
        tiles = []
        for mid, r in rows[s:s + per_sheet]:
            img = cv2.cvtColor(cv2.imread(str(strips / (re.sub(r"[^A-Za-z0-9_.-]", "_", mid) + ".png"))), cv2.COLOR_BGR2RGB)
            lab = np.zeros((img.shape[0], 250, 3), np.uint8)
            for k, txt in enumerate([r["label"][:30], r["group"][:30], f"eye {r['eye_frac']:.2f} f100 {r['first100']:.2f}",
                                     f"any {r['any_eye']:.2f} still {r['still_frac']:.2f} {r['size']}px"]):
                cv2.putText(lab, txt, (4, 24 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0) if k == 0 else (200, 200, 200), 1, cv2.LINE_AA)
            tiles.append(np.concatenate([lab, img], 1))
        sheet = np.concatenate(tiles, 0)
        p = strips / f"sheet_{s // per_sheet + 1}.png"
        cv2.imwrite(str(p), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
        paths.append(p)
    print("wrote", *map(str, paths), sep="\n  ")
    return paths


if __name__ == "__main__":
    main()
