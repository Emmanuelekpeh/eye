"""Build the 96px cache for SINGLE-IMAGE (appearance) sources (read-only on raw data; writes only under --out).

These sources have no temporal order, so every image is stored as its own length-1 sequence (seq_bounds = 0,1,2,...).
`ivs.cachedata.CacheBank` can then never build a T>1 window across unrelated images; they are only reachable through
T=1 sampling (the appearance objective in `ivs.pilot`: train encoder/decoder to reproduce real images, no dynamics).

mrl: MRL Eye dataset crops (gray IR, 61-171 px square). Filename s<subject>_<frame>_<gender>_<glasses>_<eyestate>_<reflections>_
     <lighting>_<sensor>.png; eye state follows the folder (awake=1 / sleepy=0). Files are in set/awake, set/sleepy and loose
     set/s*.png. Consecutive frames of a subject are near-duplicates, so at most --cap images are kept per (subject, eye state),
     evenly spaced over the frame numbers. One npz per subject; group = subject (a person never straddles train/val).
     Images are square: stored 96x96 and padded by REFLECTION to the loader's store width (meta pad_mode).
"""
import argparse, json, re, sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

HEIGHT = 96
MRL_NAME = re.compile(r"^(s\d+)_(\d+)_(\d)_(\d)_(\d)_(\d)_(\d)_(\d+)\.png$")


def gather_mrl(root):
    root = Path(root)
    files = [p for d in ("awake", "sleepy") for p in (root / d).glob("*.png")] + [p for p in root.glob("s*.png")]
    by = defaultdict(list)
    unparsed = []
    for p in files:
        m = MRL_NAME.match(p.name)
        if not m:
            unparsed.append(p.name)
            continue
        by[(m[1], m[5])].append((int(m[2]), p))
    return by, unparsed


def evenly(items, cap):
    items = sorted(items, key=lambda t: t[0])
    if len(items) <= cap:
        return items
    return [items[i] for i in np.unique(np.linspace(0, len(items) - 1, cap).round().astype(int))]


def to96(g):
    h, w = g.shape
    interp = cv2.INTER_AREA if max(h, w) > HEIGHT else cv2.INTER_CUBIC
    return cv2.resize(g, (HEIGHT, HEIGHT), interpolation=interp)


def build_mrl(root, out, cap, force):
    by, unparsed = gather_mrl(root)
    if unparsed:
        print(f"WARNING: {len(unparsed)} MRL-looking files did not match the naming pattern and were skipped: {unparsed[:3]}")
    subjects = sorted({s for s, _ in by})
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    metas = []
    for subj in subjects:
        dst = out / f"mrl_{subj}.npz"
        if dst.exists() and not force:
            with np.load(dst) as z:
                metas.append(json.loads(str(z["meta"])))
            print(f"{subj}: exists", flush=True)
            continue
        picked = []
        for state in ("0", "1"):
            picked += [(state, n, p) for n, p in evenly(by.get((subj, state), []), cap)]
        total_raw = sum(len(by.get((subj, s), [])) for s in ("0", "1"))
        frames, states, nums, sizes = [], [], [], []
        for state, n, p in picked:
            g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if g is None:
                print(f"WARNING: unreadable {p}")
                continue
            sizes.append(g.shape[0])
            frames.append(to96(g)); states.append(int(state)); nums.append(n)
        arr = np.repeat(np.stack(frames)[..., None], 3, axis=3)
        meta = dict(name=f"mrl_{subj}.set", group=f"mrl_{subj}", source="mrl", kind="appearance", pad_mode="reflect",
                    fps=0.0, stride=1, eff_fps=0.0, native=[int(np.median(sizes))] * 2, total_raw=total_raw,
                    crop_box=[0, 0, 0, 0], chroma=0.0, hue_spread=0.0, cuts=0, bad_frames=0, decoded=len(arr), clean=len(arr),
                    cap=cap, kept=len(arr), sequences=len(arr), width=HEIGHT, height=HEIGHT,
                    eye_open=int(sum(states)), eye_closed=int(len(states) - sum(states)))
        np.savez_compressed(dst, frames=arr, seq_bounds=np.arange(len(arr) + 1, dtype=np.int64), meta=json.dumps(meta),
                            frame_numbers=np.array(nums, dtype=np.int64), eye_state=np.array(states, dtype=np.uint8))
        meta["mb"] = round(dst.stat().st_size / 1e6, 2)
        metas.append(meta)
        print(f"{subj}: kept {len(arr)}/{total_raw} (open {meta['eye_open']}, closed {meta['eye_closed']}), {meta['mb']} MB", flush=True)
    return metas


def main():
    here = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["mrl"], required=True)
    ap.add_argument("--root", default=str(here.parent / "set"))
    ap.add_argument("--out", default=None, help="default: data/cache/<source>")
    ap.add_argument("--cap", type=int, default=300, help="max images per (subject, eye state)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    out = a.out or str(here.parent / "data" / "cache" / a.source)
    metas = build_mrl(a.root, out, a.cap, a.force)
    (Path(out) / "manifest.json").write_text(json.dumps(metas, indent=1))
    print(f"manifest: {len(metas)} groups, {sum(m['kept'] for m in metas)} images -> {out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
