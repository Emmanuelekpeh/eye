"""Clarity test, stage 1: how much detail does the RESOLUTION alone throw away? No training, CPU, read-only on raw data.

Real frames are read at NATIVE resolution (same crop box the caches used), shrunk to H px high exactly like the cache builders do (INTER_AREA),
then scaled back to native size (INTER_CUBIC) and compared with the native frame. Whatever is lost here no model trained at H px can bring
back, so this is the CEILING of the 96 px cache versus 128 / 192 / 256. Sources:
  openeds  gray IR close-ups with segmentation masks -> the ground-truth eye metrics of ivs.eyemetric.gt_metrics at native size
           (pupil contrast ratio, pupil-edge and limbus gradient ratios; 1.0 = nothing lost) + iris diameter in px at each H
  videos   LPW-style tint/gray and YouTube colour clips (no masks) -> PSNR, Laplacian-variance ratio (ivs.pilot.sharpness) and the share of
           edge strength kept in the 10% strongest native gradient pixels
Writes <out>/ceiling.json and <out>/sheet_<kind>.png (rows = frames; columns = the most detailed 160 px native patch as native | 96 | 128 | 192
| 256, all shown at the same zoom). LOOK at the sheets.
  python scripts/clarity_ceiling.py --out pilot_results/clarity_ceiling
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from ivs.eyemetric import gt_metrics, grad_mag   # noqa: E402
from ivs.pilot import sharpness                  # noqa: E402

HEIGHTS = (96, 128, 192, 256)
PATCH = 160
ROOT = HERE.parent
VID_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def kind_of(m):
    """gray / tint / colour exactly as ivs.cachedata.colour_class."""
    if m["chroma"] < 1.0:
        return "gray"
    return "tint" if m["hue_spread"] < 0.012 else "colour"


def down_up(img, h):
    """img (H0,W0[,3]) uint8 -> same size after a round trip through h px height (cache-style INTER_AREA down, cubic up).
    A height at or above the native one adds nothing, so the frame comes back unchanged."""
    h0, w0 = img.shape[:2]
    if h >= h0:
        return img.copy()
    w = max(1, int(round(w0 * h / h0)))
    small = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w0, h0), interpolation=cv2.INTER_CUBIC)


def to_gray(img):
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


_SOBEL = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])


def grad_xy(g):
    """Sobel gradient vector (gx, gy) of a gray batch (N,H,W)."""
    k = _SOBEL.to(g.device)
    return F.conv2d(g[:, None], k.view(1, 1, 3, 3), padding=1)[:, 0], F.conv2d(g[:, None], k.t().contiguous().view(1, 1, 3, 3), padding=1)[:, 0]


def frame_metrics(real, rec):
    """real, rec: uint8 gray (H,W). psnr (dB); sharp_ratio (Laplacian variance rec/real, noise-dominated, kept for comparability with ivs.pilot);
    edge_keep = gradient MAGNITUDE kept where the native frame has its 10% strongest gradients (noise, ringing and over-contrast inflate it);
    edge_fidelity = 1 - sum|grad_rec - grad_real|^2 / sum|grad_real|^2 on the same pixels = the share of the native edge energy the
    reconstruction explains: 1 = identical edges, blur lowers it, and noise, misplaced edges or exaggerated contrast lower it too (it can go
    negative), so this is the one the overfit gate uses."""
    r = torch.from_numpy(real).float()[None] / 255.0
    c = torch.from_numpy(rec).float()[None] / 255.0
    mse = float(((r - c) ** 2).mean())
    gr, gc = grad_mag(r), grad_mag(c)
    top = gr >= torch.quantile(gr.flatten(), 0.9)
    rx, ry = grad_xy(r)
    cx, cy = grad_xy(c)
    fid = 1.0 - float((((rx - cx) ** 2 + (ry - cy) ** 2) * top).sum() / ((rx ** 2 + ry ** 2) * top).sum().clamp(min=1e-12))
    return dict(psnr=10 * np.log10(1.0 / max(mse, 1e-10)),
                sharp_ratio=sharpness(c[None]) / max(sharpness(r[None]), 1e-12),
                edge_keep=float((gc * top).sum() / (gr * top).sum().clamp(min=1e-9)),
                edge_fidelity=fid)


def detail_patch(gray, size=PATCH):
    """(y, x) top-left of the size x size native window with the most gradient energy (where fine detail actually is)."""
    g = grad_mag(torch.from_numpy(gray).float()[None] / 255.0)[None]
    size = min(size, gray.shape[0], gray.shape[1])
    e = F.avg_pool2d(g, size, stride=8)[0, 0]
    iy, ix = np.unravel_index(int(e.argmax()), tuple(e.shape))
    return iy * 8, ix * 8, size


def evenly(seq, k):
    if len(seq) <= k:
        return list(seq)
    return [seq[int(round(i))] for i in np.linspace(0, len(seq) - 1, k)]


def read_video_frames(path, box, n):
    """n native BGR frames (cropped to box) at evenly spaced positions in 5%..95% of the clip."""
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    x0, y0, x1, y1 = box
    out = []
    for i in np.linspace(0.05 * total, 0.95 * total, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok and f.std() > 4:
            out.append(f[y0:y1, x0:x1])
    cap.release()
    return out


def read_openeds(folder, box, n):
    """n native (gray, mask) pairs cropped to box, evenly over the frames that have a mask file; frames with no visible pupil are skipped."""
    x0, y0, x1, y1 = box
    nums = sorted(int(p.stem) for p in folder.glob("*.png") if p.stem.isdigit() and (folder / f"{p.stem}.npy").exists())
    out = []
    for k in evenly(nums, n * 3):
        g = cv2.imread(str(folder / f"{k}.png"), cv2.IMREAD_GRAYSCALE)
        m = np.load(folder / f"{k}.npy")
        if g is None or m.shape != g.shape:
            continue
        g, m = g[y0:y1, x0:x1], m[y0:y1, x0:x1]
        if int((m == 3).sum()) >= 60 and int((m == 2).sum()) >= 60:
            out.append((g, m))
        if len(out) == n:
            break
    return out


def mean_dict(rows):
    keys = sorted({k for r in rows for k, v in r.items() if isinstance(v, (int, float)) and v is not None and np.isfinite(v)})
    return {k: float(np.mean([r[k] for r in rows if r.get(k) is not None and np.isfinite(r[k])])) for k in keys}


def sheet_row(native_bgr_or_gray, gray, heights):
    y, x, s = detail_patch(gray)
    tiles = []
    for h in (None,) + tuple(heights):
        im = native_bgr_or_gray if h is None else down_up(native_bgr_or_gray, h)
        t = im[y:y + s, x:x + s]
        t = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR) if t.ndim == 2 else t
        t = cv2.resize(t, (240, 240), interpolation=cv2.INTER_NEAREST)
        cv2.putText(t, "native" if h is None else f"{h}px", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        tiles.append(t)
    return np.concatenate(tiles, 1)


def run(out, root=ROOT, heights=HEIGHTS, openeds_seqs=10, openeds_frames=4, videos_per_kind=(("gray", 4), ("tint", 6), ("colour", 6)),
        video_frames=6, sheet_rows=3):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    cache = root / "data" / "cache"
    res = {"heights": list(heights), "patch_native_px": PATCH}

    # ------------------------------------------------------------------ openeds (masks -> ground-truth eye metrics)
    ms = json.loads((cache / "openeds" / "manifest.json").read_text())
    res["openeds_iris_px"] = {str(h): float(np.median([m["iris"]["diameter"] * h / (m["crop_box"][3] - m["crop_box"][1]) for m in ms])) for h in heights}
    chosen = evenly(sorted(ms, key=lambda m: m["name"]), openeds_seqs)
    per_h = {h: [] for h in heights}
    sheet = []
    for m in chosen:
        pairs = read_openeds(root / "set" / m["group"], tuple(m["crop_box"]), openeds_frames)
        if not pairs:
            continue
        real = torch.from_numpy(np.stack([p[0] for p in pairs])).float() / 255.0
        mask = torch.from_numpy(np.stack([p[1] for p in pairs])).to(torch.uint8)
        for h in heights:
            rec = torch.from_numpy(np.stack([down_up(p[0], h) for p in pairs])).float() / 255.0
            row = gt_metrics(real, rec, mask)
            row.update(mean_dict([frame_metrics(p[0], down_up(p[0], h)) for p in pairs]))
            per_h[h].append(row)
        if len(sheet) < sheet_rows:
            sheet.append(sheet_row(pairs[len(pairs) // 2][0], pairs[len(pairs) // 2][0], heights))
    res["openeds"] = {str(h): mean_dict(per_h[h]) for h in heights}
    res["openeds"]["sequences"] = len(chosen)
    if sheet:
        cv2.imwrite(str(out / "sheet_openeds.png"), np.concatenate(sheet, 0))

    # ------------------------------------------------------------------ videos (no masks)
    vm = [m for m in json.loads((cache / "videos" / "manifest.json").read_text()) if not m["name"].startswith("youtube_eye_")]
    for kind, k in videos_per_kind:
        pool = sorted([m for m in vm if kind_of(m) == kind], key=lambda m: m["name"])
        per_h, sheet, used = {h: [] for h in heights}, [], 0
        for m in evenly(pool, k):
            path = root / "set" / m["name"]
            frames = read_video_frames(path, tuple(m["crop_box"]), video_frames) if path.exists() and path.suffix.lower() in VID_EXT else []
            if not frames:
                continue
            used += 1
            for f in frames:
                g = to_gray(f)
                for h in heights:
                    per_h[h].append(frame_metrics(g, to_gray(down_up(f, h))))
            if len(sheet) < sheet_rows:
                sheet.append(sheet_row(frames[len(frames) // 2], to_gray(frames[len(frames) // 2]), heights))
        res[kind] = {str(h): mean_dict(per_h[h]) for h in heights}
        res[kind]["videos"] = used
        if sheet:
            cv2.imwrite(str(out / f"sheet_{kind}.png"), np.concatenate(sheet, 0))
    (out / "ceiling.json").write_text(json.dumps(res, indent=1))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "pilot_results" / "clarity_ceiling"))
    ap.add_argument("--heights", type=int, nargs="+", default=list(HEIGHTS))
    ap.add_argument("--openeds-seqs", type=int, default=10)
    ap.add_argument("--video-frames", type=int, default=6)
    a = ap.parse_args(argv)
    res = run(a.out, heights=tuple(a.heights), openeds_seqs=a.openeds_seqs, video_frames=a.video_frames)
    print("iris diameter in px at each height (median over OpenEDS sequences):", res["openeds_iris_px"])
    for src in ("openeds", "gray", "tint", "colour"):
        print(f"== {src}")
        for h in a.heights:
            r = res[src][str(h)]
            print(f"  {h:>3}px  " + "  ".join(f"{k}={r[k]:.3f}" for k in ("psnr", "sharp_ratio", "edge_keep", "pupil_contrast_ratio_median", "pupil_edge_grad_ratio", "limbus_grad_ratio") if k in r))
    return res


if __name__ == "__main__":
    main()
