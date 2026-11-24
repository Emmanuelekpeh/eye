"""Real eye geometry and a cleaned gaze command, both measured from the S_ segmentation masks (0 bg, 1 sclera, 2 iris, 3 pupil).

Geometry per frame (G=7, pixels of the STORED 96x128 frame, x right / y down, pixel-index coordinates):
    0 pupil_cx  1 pupil_cy  2 pupil_r  3 iris_cx  4 iris_cy  5 iris_r  6 open_r
  pupil = class 3, iris = classes 2+3, eye opening = classes 1+2+3; r = sqrt(area / pi). Each dimension has its own validity:
  pupil dims need >= MIN_PUPIL_PX pupil pixels, iris dims need >= MIN_IRIS_PX iris pixels, open_r is valid on every known frame
  (a closed eye is a real target: open_r -> 0).
Command per frame: the pupil centre, cleaned: gaps (blinks) filled from the nearest visible frame, median filter (k=5) against mask
jitter, then HELD until the filtered centre is more than HOLD_PX away from the held value (then it jumps): an absolute target
position without sub-pixel jitter. Measured on all S_ data (27k frames): the pupil moves a median 1.8 px per 0.25 s and 11 px per second, so
the eye is in near-continuous motion and a 1.5 px hold changes the command on ~23% of frames (p90 tracking error 1.3 px); raw per-frame
deltas are mostly jitter (43% < 0.25 px, lag-1 autocorrelation of the size change -0.35), which is why the command is a target and not a delta.
Everything stays in this module's pixel units; ivs.cachedata maps it through the same crop / flip / scale as the pixels.
"""
from pathlib import Path

import numpy as np

MIN_PUPIL_PX = 12          # pupil pixels (96 px frames) below which the pupil counts as absent / eye closed
MIN_IRIS_PX = 60           # iris+pupil pixels below which the iris counts as absent
GEOM_NAMES = ("pupil_cx", "pupil_cy", "pupil_r", "iris_cx", "iris_cy", "iris_r", "open_r")
GEOM_DIM = len(GEOM_NAMES)
POS_PAIRS = ((0, 1), (3, 4))   # (x,y) dims that are positions (pupil, iris centre): they move with crop / flip / synthetic shifts
RADIUS_DIMS = (2, 5, 6)
MEDIAN_K = 5
HOLD_PX = 1.5


def frame_geometry(masks):
    """masks (N,H,W) uint8 -> (geom (N,7) float32, valid (N,7) bool); invalid entries are 0 in geom."""
    m = np.asarray(masks)
    n, h, w = m.shape
    ys = np.arange(h, dtype=np.float32)[None, :, None]
    xs = np.arange(w, dtype=np.float32)[None, None, :]
    known = (m != 255).any((1, 2))
    geom = np.zeros((n, GEOM_DIM), np.float32)
    valid = np.zeros((n, GEOM_DIM), bool)

    def centroid(sel):
        a = sel.sum((1, 2)).astype(np.float32)
        safe = np.maximum(a, 1.0)
        return (sel * xs).sum((1, 2)) / safe, (sel * ys).sum((1, 2)) / safe, a

    cx, cy, a = centroid(m == 3)
    ok = a >= MIN_PUPIL_PX
    geom[:, 0], geom[:, 1], geom[:, 2] = np.where(ok, cx, 0), np.where(ok, cy, 0), np.where(ok, np.sqrt(a / np.pi), 0)
    valid[:, 0] = valid[:, 1] = valid[:, 2] = ok
    cx, cy, a = centroid((m == 2) | (m == 3))
    ok = a >= MIN_IRIS_PX
    geom[:, 3], geom[:, 4], geom[:, 5] = np.where(ok, cx, 0), np.where(ok, cy, 0), np.where(ok, np.sqrt(a / np.pi), 0)
    valid[:, 3] = valid[:, 4] = valid[:, 5] = ok
    a = ((m >= 1) & (m <= 3)).sum((1, 2)).astype(np.float32)
    geom[:, 6] = np.where(known, np.sqrt(a / np.pi), 0)
    valid[:, 6] = known
    return geom, valid


def _median_filter(v, k=MEDIAN_K):
    pad = k // 2
    p = np.pad(v, (pad, pad), mode="edge")
    return np.median(np.stack([p[i:i + len(v)] for i in range(k)]), axis=0)


def clean_command(cx, cy, visible, hold_px=HOLD_PX, k=MEDIAN_K):
    """One clean sequence: cx, cy (n,) pixel centres, visible (n,) bool -> (cmd (n,2) float32, has (n,) bool).
    has is False for the whole sequence when no frame shows the pupil."""
    n = len(cx)
    if not visible.any():
        return np.zeros((n, 2), np.float32), np.zeros(n, bool)
    idx = np.where(visible, np.arange(n), -1)
    idx = np.maximum.accumulate(idx)                                   # last visible frame at or before i
    first = int(np.argmax(visible))
    idx = np.where(idx < 0, first, idx)                                # leading blink frames take the first visible frame
    fx, fy = _median_filter(cx[idx].astype(np.float64), k), _median_filter(cy[idx].astype(np.float64), k)
    out = np.zeros((n, 2), np.float32)
    held = np.array([fx[0], fy[0]])
    for i in range(n):
        if np.hypot(fx[i] - held[0], fy[i] - held[1]) > hold_px:
            held = np.array([fx[i], fy[i]])
        out[i] = held
    return out, np.ones(n, bool)


def group_labels(masks, seq_bounds):
    """One S_ group: masks (N,H,W) in cached-frame order, seq_bounds (nseq+1,). Returns dict(geom, valid, cmd, has_cmd)."""
    geom, valid = frame_geometry(masks)
    cmd = np.zeros((len(masks), 2), np.float32)
    has = np.zeros(len(masks), bool)
    for a, b in zip(seq_bounds[:-1], seq_bounds[1:]):
        vis = valid[a:b, 0]
        c, hs = clean_command(geom[a:b, 0], geom[a:b, 1], vis)
        cmd[a:b], has[a:b] = c, hs
    return dict(geom=geom, valid=valid, cmd=cmd, has_cmd=has)


def load_group_labels(labels_dir, group, n=None):
    """Labels for the cache group `group` from `<labels_dir>/<group>.npz` (written by build_cache_openeds.py --masks), or None
    when the group has no mask file. n truncates to the first n frames (CacheBank max_frames_per_video), which keeps the
    sequence structure because the cache truncates at a sequence boundary."""
    f = Path(labels_dir) / (group + ".npz")
    if not f.exists():
        return None
    with np.load(f) as z:
        masks, bounds = z["masks"], z["seq_bounds"]
    if n is not None and n < len(masks):
        cut = int(np.searchsorted(bounds, n, side="right")) - 1
        bounds = bounds[:max(1, cut) + 1]
        masks = masks[:bounds[-1]]
    return group_labels(masks, bounds)
