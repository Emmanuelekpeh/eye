"""Build the 96px training cache for the VIDEO source (read-only on raw data; writes only under --out).

One compressed .npz per video so the build is resumable and each file can be uploaded on its own:
  frames      (N, 96, W, 3) uint8 RGB, aspect ratio kept (W = 96 * crop_w / crop_h). Crop to 96x96 at train time.
  seq_bounds  (S+1,) int64: sequence i = frames[seq_bounds[i]:seq_bounds[i+1]]. Windows must never straddle one.
  meta        JSON string: name, group (= file stem; every file is its own subject), fps, stride, eff_fps, native size,
              letterbox crop box, colour statistics, cuts found, frames dropped, cap, kept.

Cleaning: letterbox bars cropped (rows/cols that never exceed black in sampled frames), scene cuts and black/blank
frames split sequences, sequences shorter than --min-seq are dropped. If a video keeps more than --cap frames, evenly
spaced cut-free chunks across the WHOLE recording are kept (not only the first minute).
Colour: stored natively. `chroma` = mean max-channel difference (0-255), `hue_spread` = std of chromaticity over bright
pixels. A tinted monochrome camera has high chroma but tiny hue_spread; real colour has both high.
"""
import argparse, json, re, sys, time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

VID = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
HEIGHT = 96


def nat(p):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", Path(p).name)]


def safe(stem):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")


def sample_frames(path, n=24):
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for i in np.linspace(0, max(0, total - 2), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            out.append(f)
    cap.release()
    return out


def find_crop(frames, bar_level=24):
    """Tight box (x0, y0, x1, y1) inside black letterbox/pillarbox bars; full frame if nothing clearly bar-like."""
    h, w = frames[0].shape[:2]
    g = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames])
    row_max = g.max(axis=(0, 2)); col_max = g.max(axis=(0, 1))
    ys = np.where(row_max >= bar_level)[0]; xs = np.where(col_max >= bar_level)[0]
    if len(ys) == 0 or len(xs) == 0:
        return (0, 0, w, h)
    x0, x1, y0, y1 = int(xs[0]), int(xs[-1]) + 1, int(ys[0]), int(ys[-1]) + 1
    if (x1 - x0) < 0.5 * w or (y1 - y0) < 0.5 * h:      # implausible: do not crop
        return (0, 0, w, h)
    return (x0, y0, x1, y1)


def colour_stats(frames, box):
    x0, y0, x1, y1 = box
    chroma, spread = [], []
    for f in frames:
        c = cv2.resize(f[y0:y1, x0:x1], (24, 24), interpolation=cv2.INTER_AREA).astype(np.float32)
        chroma.append(float(max(np.abs(c[..., 0] - c[..., 1]).mean(), np.abs(c[..., 1] - c[..., 2]).mean(), np.abs(c[..., 0] - c[..., 2]).mean())))
        s = c.sum(-1); m = s > 120
        if m.sum() > 20:
            ch = c[m] / s[m][:, None]
            spread.append(float(ch[:, :2].std(axis=0).mean()))
    return float(np.mean(chroma)), float(np.mean(spread)) if spread else 0.0


def pick_chunks(seq_lens, cap, chunk):
    """seq_lens: list of (start, end). Returns list of (start, end) with total <= ~cap, evenly spaced over the video."""
    if sum(e - s for s, e in seq_lens) <= cap:        # under the cap: keep every clean sequence whole
        return list(seq_lens)
    pieces = []
    for s, e in seq_lens:
        if e - s > chunk:
            for a in range(s, e, chunk):
                b = min(e, a + chunk)
                if b - a >= max(32, chunk // 3):
                    pieces.append((a, b))
        else:
            pieces.append((s, e))
    total = sum(b - a for a, b in pieces)
    if total <= cap:
        return pieces
    need = max(1, int(round(cap / (total / len(pieces)))))
    idx = np.unique(np.linspace(0, len(pieces) - 1, need).round().astype(int))
    return [pieces[i] for i in idx]


def window_spans(total, windows, raw_len):
    """(start, raw frames) of `windows` equal-length spans centred in equal stretches of the video; [(0, None)] (= read it all) when the spans
    would cover most of a short video anyway, so no frame is ever used twice."""
    if windows <= 0 or windows * raw_len >= 0.9 * total:
        return [(0, None)]
    return [(min(max(0, int(round((k + 0.5) * total / windows - raw_len / 2))), total - raw_len), raw_len) for k in range(windows)]


def process(job):
    path, out_dir, cap, target_fps, cut_thr, min_seq, chunk, force = job[:8]
    height = job[8] if len(job) > 8 else HEIGHT          # stored frame height; 96 = the original cache
    windows, window_len = (job[9], job[10]) if len(job) > 10 else (0, 0)    # windows > 0: decode only that many spread-out spans of window_len frames
    path = Path(path)
    dst = Path(out_dir) / (safe(path.stem) + ".npz")
    if dst.exists() and not force:
        return path.name, "exists", None
    t0 = time.time()
    cap_ = cv2.VideoCapture(str(path))
    if not cap_.isOpened():
        return path.name, "unreadable", None
    fps = cap_.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap_.get(cv2.CAP_PROP_FRAME_COUNT))
    w0, h0 = int(cap_.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_.release()
    samples = sample_frames(path)
    if not samples:
        return path.name, "no frames", None
    box = find_crop(samples)
    chroma, spread = colour_stats(samples, box)
    x0, y0, x1, y1 = box
    cw, ch_ = x1 - x0, y1 - y0
    W = int(round(height * cw / ch_))
    stride = max(1, int(round(fps / target_fps)))
    spans = window_spans(total, windows, window_len * stride)

    cap_ = cv2.VideoCapture(str(path))
    frames, small, ends = [], [], []
    for start, length in spans:
        if start:
            cap_.set(cv2.CAP_PROP_POS_FRAMES, start)
        idx = 0
        while length is None or idx < length:
            if not cap_.grab():
                break
            if idx % stride == 0:
                ok, f = cap_.retrieve()
                if not ok:
                    break
                f = f[y0:y1, x0:x1]
                rgb = cv2.cvtColor(cv2.resize(f, (W, height), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
                frames.append(rgb)
                small.append(cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (48, 48), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0)
            idx += 1
        ends.append(len(frames))
    cap_.release()
    n = len(frames)
    if n < min_seq:
        return path.name, f"too short ({n})", None

    sm = np.stack(small)
    bad = (sm.mean(axis=(1, 2)) < 0.03) | (sm.std(axis=(1, 2)) < 0.02)       # black / blank frames
    d = np.abs(np.diff(sm, axis=0)).mean(axis=(1, 2))
    cut_after = d > cut_thr                                                 # boundary between frame i and i+1
    for e in ends[:-1]:                                                     # two spans are never one continuous shot
        if 0 < e < n:
            cut_after[e - 1] = True
    seqs, start = [], 0
    for i in range(n):
        if bad[i]:                                   # a bad frame closes the running sequence and is dropped
            if i > start:
                seqs.append((start, i))
            start = i + 1
        elif i == n - 1 or cut_after[i]:             # a cut between i and i+1 closes it after frame i
            seqs.append((start, i + 1))
            start = i + 1
    seqs = [(a, b) for a, b in seqs if b - a >= min_seq]
    kept_before_cap = sum(b - a for a, b in seqs)
    if not seqs:
        return path.name, "no clean sequence", None
    chosen = pick_chunks(seqs, cap, chunk)
    arr = np.concatenate([np.stack(frames[a:b]) for a, b in chosen])
    bounds = np.cumsum([0] + [b - a for a, b in chosen]).astype(np.int64)
    # every file is its own subject (user), except re-downloads of the same YouTube video (verified identical frames):
    # yt_<id> and youtube_eye_<id> share group yt_<id> so a duplicate can never straddle train/val.
    m = re.fullmatch(r"(?:yt|youtube_eye)_(.+)", path.stem)
    meta = dict(name=path.name, group=("yt_" + m.group(1)) if m else path.stem, source="video", fps=round(float(fps), 3), stride=stride,
                eff_fps=round(float(fps) / stride, 2), native=[w0, h0], total_raw=total, crop_box=[x0, y0, x1, y1],
                chroma=round(chroma, 2), hue_spread=round(spread, 4), cuts=int(cut_after.sum()), bad_frames=int(bad.sum()),
                decoded=n, clean=kept_before_cap, cap=cap, kept=int(len(arr)), sequences=len(chosen), width=W, height=height)
    if windows:
        meta.update(windows=windows, window_len=window_len, spans=len(spans))
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, frames=arr, seq_bounds=bounds, meta=json.dumps(meta))
    meta["seconds"] = round(time.time() - t0, 1)
    meta["mb"] = round(dst.stat().st_size / 1e6, 1)
    return path.name, "ok", meta


def main():
    here = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(here.parent / "set"))
    ap.add_argument("--out", default=str(here.parent / "data" / "cache" / "videos"))
    ap.add_argument("--cap", type=int, default=3000, help="max frames kept per video")
    ap.add_argument("--target-fps", type=float, default=20.0)
    ap.add_argument("--cut-thr", type=float, default=0.30, help="mean |diff| of 48px gray frames above which a cut is declared "
                    "(checked by eye on yt_KVgQV: 0.15-0.30 is continuous hand-held motion, >0.30 are shots/exposure flashes)")
    ap.add_argument("--min-seq", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=300, help="chunk length when a video is over the cap")
    ap.add_argument("--height", type=int, default=HEIGHT, help="stored frame height (default 96 = the original cache); use a different --out")
    ap.add_argument("--windows", type=int, default=0, help="0 = decode the whole video (original behaviour); N = decode only N spread-out spans of --window-len "
                    "frames (bounded memory at large --height); spans are separate sequences")
    ap.add_argument("--window-len", type=int, default=90, help="frames (after the fps stride) per span with --windows")
    ap.add_argument("--groups-from", default=None, help="manifest.json whose group names override the computed ones (same train/val split as that cache)")
    ap.add_argument("--only", nargs="*", default=None, help="video file names to process (default all)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    root = Path(a.root)
    vids = sorted([p for p in root.iterdir() if p.is_file() and p.suffix.lower() in VID], key=nat)
    if a.only:
        vids = [p for p in vids if p.name in set(a.only)]
    vids.sort(key=lambda p: -p.stat().st_size)                       # biggest first: balances the pool
    if a.height != HEIGHT and Path(a.out).resolve() == (here.parent / "data" / "cache" / "videos").resolve():
        raise SystemExit(f"--height {a.height} would write into the 96 px cache {a.out}: pass a different --out")
    jobs = [(str(p), a.out, a.cap, a.target_fps, a.cut_thr, a.min_seq, a.chunk, a.force, a.height, a.windows, a.window_len) for p in vids]
    print(f"{len(jobs)} videos -> {a.out} (cap {a.cap}, workers {a.workers})", flush=True)
    t0 = time.time()
    with Pool(a.workers) as pool:
        for name, status, meta in pool.imap_unordered(process, jobs):
            line = f"[{time.time() - t0:6.0f}s] {name}: {status}"
            if meta:
                line += f" kept {meta['kept']}/{meta['clean']} clean of {meta['decoded']} decoded, {meta['sequences']} seqs, cuts {meta['cuts']}, bad {meta['bad_frames']}, {meta['mb']} MB, chroma {meta['chroma']}/spread {meta['hue_spread']}"
            print(line, flush=True)
    metas = []
    for f in sorted(Path(a.out).glob("*.npz")):
        with np.load(f) as z:
            metas.append(json.loads(str(z["meta"])))
    if a.groups_from:                                   # keep the group names (= the train/val split) of an existing manifest, e.g. the 96 px cache
        ref = {m["name"]: m["group"] for m in json.loads(Path(a.groups_from).read_text())}
        changed = 0
        for m in metas:
            if m["name"] in ref and ref[m["name"]] != m["group"]:
                print(f"group of {m['name']}: {m['group']} -> {ref[m['name']]} (from {a.groups_from})", flush=True)
                m["group"] = ref[m["name"]]
                changed += 1
        print(f"groups taken from {a.groups_from}: {changed} changed", flush=True)
    (Path(a.out) / "manifest.json").write_text(json.dumps(metas, indent=1))
    print(f"manifest: {len(metas)} videos, {sum(m['kept'] for m in metas)} frames", flush=True)


if __name__ == "__main__":
    sys.exit(main())
