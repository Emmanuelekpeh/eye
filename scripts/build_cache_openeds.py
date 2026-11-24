"""Build the 96px cache for the OpenEDS-style `S_*` sequences (read-only on raw data; writes only under --out).

Raw: set/S_<k>/<n>.png (400x640 gray IR) + <n>.npy segmentation masks (0 bg, 1 sclera, 2 iris, 3 pupil) + a few unpaired
label_<n>.npy (ignored). Frame numbers are contiguous in every folder and motion per frame matches the 20 fps video cache
(median lag-1 difference 0.004 vs 0.006), so every frame is kept (stride 1).

Crop: ONE fixed crop per sequence (so the eye's own motion is preserved), 4:3 like the video cache, centred on the median
iris/pupil centroid and sized as `--scale` x the median iris diameter (clamped to [--min-h, 400]); the window is shifted to stay
inside the image. Output is stored like the video cache so `ivs.cachedata.CacheBank` reads it unchanged:
  S_<k>.npz: frames (N,96,128,3) uint8 (gray replicated to 3 channels), seq_bounds, meta JSON (group = S_<k>, chroma 0, ...)
  manifest.json
Group = folder (neighbouring folders show different people; whether one person appears in two folders is NOT verifiable).
"""
import argparse, json, re, sys, time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

HEIGHT, WIDTH = 96, 128


def nat(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def iris_geometry(folder, nums, step=4):
    """Median iris+pupil centroid (x, y) and equivalent diameter over every `step`-th frame that has a mask with enough iris."""
    cx, cy, dia = [], [], []
    for n in nums[::step]:
        f = folder / f"{n}.npy"
        if not f.exists():
            continue
        m = np.load(f)
        iris = (m == 2) | (m == 3)
        a = int(iris.sum())
        if a < 400:                                             # closed eye / partial: no usable iris
            continue
        ys, xs = np.nonzero(iris)
        cx.append(xs.mean()); cy.append(ys.mean()); dia.append(2.0 * np.sqrt(a / np.pi))
    if len(cx) < 3:
        return None
    return float(np.median(cx)), float(np.median(cy)), float(np.median(dia)), len(cx)


def crop_box(geom, scale, min_h, img_h, img_w):
    cx, cy, dia, _ = geom
    ch = int(round(min(img_h, max(min_h, scale * dia))))
    cw = int(round(ch * WIDTH / HEIGHT))
    cw = min(cw, img_w); ch = int(round(cw * HEIGHT / WIDTH))
    x0 = int(round(min(max(cx - cw / 2, 0), img_w - cw)))
    y0 = int(round(min(max(cy - ch / 2, 0), img_h - ch)))
    return x0, y0, x0 + cw, y0 + ch


def read_sequences(folder, nums, box, img_h, img_w, cut_thr, with_masks=False, height=HEIGHT, width=WIDTH):
    """Crop+resize every frame of one folder, drop bad frames / split at cuts. Shared by the frame cache and the mask cache so
    both select exactly the same frames. Returns (frames, masks_or_None, seqs, bad, cut_after) or an error string.
    masks: (n,HEIGHT,WIDTH) uint8 nearest-resized with the same crop (255 where a frame has no mask file)."""
    x0, y0, x1, y1 = box
    frames, masks, small = [], [], []
    for n in nums:
        g = cv2.imread(str(folder / f"{n}.png"), cv2.IMREAD_GRAYSCALE)
        if g is None or g.shape != (img_h, img_w):
            return f"skipped: unreadable/odd frame {n}"
        r = cv2.resize(g[y0:y1, x0:x1], (width, height), interpolation=cv2.INTER_AREA)
        frames.append(r)
        small.append(cv2.resize(r, (48, 48), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0)
        if with_masks:
            f = folder / f"{n}.npy"
            m = np.load(f) if f.exists() else None
            masks.append(np.full((height, width), 255, np.uint8) if m is None or m.shape != (img_h, img_w)
                         else cv2.resize(m[y0:y1, x0:x1], (width, height), interpolation=cv2.INTER_NEAREST))
    sm = np.stack(small)
    bad = (sm.mean(axis=(1, 2)) < 0.03) | (sm.std(axis=(1, 2)) < 0.02)
    d = np.abs(np.diff(sm, axis=0)).mean(axis=(1, 2))
    cut_after = d > cut_thr
    seqs, start, n_fr = [], 0, len(frames)
    for i in range(n_fr):
        if bad[i]:
            if i > start:
                seqs.append((start, i))
            start = i + 1
        elif i == n_fr - 1 or cut_after[i]:
            seqs.append((start, i + 1)); start = i + 1
    seqs = [(a, b) for a, b in seqs if b - a >= 16]
    if not seqs:
        return "skipped: no clean sequence"
    return frames, (masks if with_masks else None), seqs, bad, cut_after


def process(job):
    folder, out_dir, scale, min_h, cut_thr, force = job[:6]
    height = job[6] if len(job) > 6 else HEIGHT           # the crop box is chosen at the native size (4:3), so every height sees the same pixels
    width = int(round(height * WIDTH / HEIGHT))
    folder = Path(folder)
    dst = Path(out_dir) / (folder.name + ".npz")
    if dst.exists() and not force:
        return folder.name, "exists", None
    nums = sorted(int(p.stem) for p in folder.glob("*.png") if p.stem.isdigit())
    if len(nums) < 16 or nums != list(range(nums[0], nums[0] + len(nums))):
        return folder.name, f"skipped: {len(nums)} frames, contiguous={nums == list(range(nums[0], nums[0] + len(nums))) if nums else False}", None
    geom = iris_geometry(folder, nums)
    if geom is None:
        return folder.name, "skipped: no usable iris masks", None
    first = cv2.imread(str(folder / f"{nums[0]}.png"), cv2.IMREAD_GRAYSCALE)
    img_h, img_w = first.shape
    x0, y0, x1, y1 = crop_box(geom, scale, min_h, img_h, img_w)
    got = read_sequences(folder, nums, (x0, y0, x1, y1), img_h, img_w, cut_thr, height=height, width=width)
    if isinstance(got, str):
        return folder.name, got, None
    frames, _, seqs, bad, cut_after = got
    arr = np.concatenate([np.stack(frames[a:b]) for a, b in seqs])
    arr = np.repeat(arr[..., None], 3, axis=3)
    bounds = np.cumsum([0] + [b - a for a, b in seqs]).astype(np.int64)
    meta = dict(name=folder.name + ".png-seq", group=folder.name, source="openeds", fps=20.0, stride=1, eff_fps=20.0,
                native=[img_w, img_h], total_raw=len(nums), crop_box=[x0, y0, x1, y1], chroma=0.0, hue_spread=0.0,
                cuts=int(cut_after.sum()), bad_frames=int(bad.sum()), decoded=len(nums), clean=int(len(arr)), cap=0,
                kept=int(len(arr)), sequences=len(seqs), width=width, height=height,
                iris=dict(cx=round(geom[0], 1), cy=round(geom[1], 1), diameter=round(geom[2], 1), frames_used=geom[3]))
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, frames=arr, seq_bounds=bounds, meta=json.dumps(meta))
    meta["mb"] = round(dst.stat().st_size / 1e6, 2)
    return folder.name, "ok", meta


def process_masks(job):
    """Mask cache for one existing frame-cache entry: S_<k>.npz with masks (N,96,128) uint8 in the SAME frame order as the
    cached frames (0 bg, 1 sclera, 2 iris, 3 pupil, 255 unknown). Uses the manifest's crop_box and asserts every selected frame
    equals the cached one, so a mask can never be paired with the wrong frame."""
    meta, cache_dir, masks_dir, root, cut_thr = job
    folder = Path(root) / meta["group"]
    nums = sorted(int(p.stem) for p in folder.glob("*.png") if p.stem.isdigit())
    img_w, img_h = meta["native"]
    got = read_sequences(folder, nums, tuple(meta["crop_box"]), img_h, img_w, cut_thr, with_masks=True)
    if isinstance(got, str):
        return meta["group"], got, None
    frames, masks, seqs, _, _ = got
    with np.load(Path(cache_dir) / (meta["group"] + ".npz")) as z:
        cached, bounds = z["frames"][..., 0], z["seq_bounds"]
    arr = np.concatenate([np.stack(frames[a:b]) for a, b in seqs])
    mk = np.concatenate([np.stack(masks[a:b]) for a, b in seqs])
    if len(arr) != len(cached) or not np.array_equal(arr, cached) or list(np.diff(bounds)) != [b - a for a, b in seqs]:
        return meta["group"], "MISMATCH: recomputed frames differ from the cache (not written)", None
    np.savez_compressed(Path(masks_dir) / (meta["group"] + ".npz"), masks=mk, seq_bounds=bounds)
    cnt = {int(v): int((mk == v).sum()) for v in np.unique(mk)}
    return meta["group"], "ok", dict(frames=len(mk), pixels=cnt)


def main():
    here = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(here.parent / "set"))
    ap.add_argument("--out", default=str(here.parent / "data" / "cache" / "openeds"))
    ap.add_argument("--scale", type=float, default=2.5, help="crop height = scale x median iris diameter")
    ap.add_argument("--min-h", type=int, default=160)
    ap.add_argument("--height", type=int, default=HEIGHT, help="stored frame height (width = 4/3 of it); frames only, the mask cache stays at 96")
    ap.add_argument("--cut-thr", type=float, default=0.30)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--stats-only", action="store_true", help="print iris geometry statistics and exit")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--masks", action="store_true",
                    help="instead of frames, write the aligned mask cache for the EXISTING frame cache in --out into --masks-out")
    ap.add_argument("--masks-out", default=str(here.parent / "data" / "cache" / "openeds_masks"))
    a = ap.parse_args()
    root = Path(a.root)
    if a.masks:
        metas = json.loads((Path(a.out) / "manifest.json").read_text())
        if a.only:
            metas = [m for m in metas if m["group"] in set(a.only)]
        Path(a.masks_out).mkdir(parents=True, exist_ok=True)
        jobs = [(m, a.out, a.masks_out, str(root), a.cut_thr) for m in metas]
        t0, bad = time.time(), 0
        with Pool(a.workers) as pool:
            for name, status, info in pool.imap_unordered(process_masks, jobs):
                bad += status != "ok"
                print(f"[{time.time() - t0:5.0f}s] {name}: {status} {info or ''}", flush=True)
        print(f"{len(jobs) - bad}/{len(jobs)} mask files in {a.masks_out}", flush=True)
        return 1 if bad else 0
    folders = sorted([p for p in root.iterdir() if p.is_dir() and re.fullmatch(r"S_\d+", p.name)], key=lambda p: nat(p.name))
    if a.only:
        folders = [p for p in folders if p.name in set(a.only)]
    if a.stats_only:
        dias, small = [], 0
        for p in folders:
            nums = sorted(int(q.stem) for q in p.glob("*.png") if q.stem.isdigit())
            g = iris_geometry(p, nums, step=8)
            if g is None:
                small += 1; continue
            dias.append(g[2])
        dias = np.array(dias)
        print(f"{len(folders)} folders, {small} without usable iris; iris diameter px: min {dias.min():.0f} p10 {np.percentile(dias, 10):.0f} "
              f"median {np.median(dias):.0f} p90 {np.percentile(dias, 90):.0f} max {dias.max():.0f}")
        return
    if a.masks is False and a.height != HEIGHT and Path(a.out).resolve() == (here.parent / "data" / "cache" / "openeds").resolve():
        raise SystemExit(f"--height {a.height} would overwrite the 96 px cache in {a.out}: pass a different --out")
    jobs = [(str(p), a.out, a.scale, a.min_h, a.cut_thr, a.force, a.height) for p in folders]
    print(f"{len(jobs)} folders -> {a.out}", flush=True)
    t0 = time.time()
    with Pool(a.workers) as pool:
        for name, status, meta in pool.imap_unordered(process, jobs):
            line = f"[{time.time() - t0:5.0f}s] {name}: {status}"
            if meta:
                line += f" kept {meta['kept']}/{meta['decoded']}, seqs {meta['sequences']}, cuts {meta['cuts']}, bad {meta['bad_frames']}, crop {meta['crop_box']}, {meta['mb']} MB"
            print(line, flush=True)
    metas = []
    for f in sorted(Path(a.out).glob("*.npz"), key=lambda p: nat(p.name)):
        with np.load(f) as z:
            metas.append(json.loads(str(z["meta"])))
    (Path(a.out) / "manifest.json").write_text(json.dumps(metas, indent=1))
    print(f"manifest: {len(metas)} sequences, {sum(m['kept'] for m in metas)} frames", flush=True)


if __name__ == "__main__":
    sys.exit(main())

