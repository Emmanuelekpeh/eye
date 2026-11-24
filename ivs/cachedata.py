"""Loader for the 96px video cache written by scripts/build_cache.py (one npz per video, see its docstring).

Everything lives in ONE uint8 tensor (optionally on the GPU), so a batch is an index gather + crop: no DataLoader.
  * windows never straddle a sequence boundary (cuts / gaps / chunks are boundaries);
  * sampling is balanced over subject groups (group = one video file, except YouTube re-downloads which share a group),
    then sequences inside a group in proportion to how many windows they hold;
  * train/val is split by GROUP and stratified by colour class (gray / tint / colour) so val always has each kind;
  * frames are stored 96 high and centre-cropped to 112 wide; train takes a random 96-wide crop (+ optional h-flip),
    val takes the centre crop;
  * colour classes come from the manifest: chroma < 1 -> gray, hue_spread < 0.012 -> tint (monochrome camera cast,
    not eye colour), else colour. tint_mode='gray' turns gray AND tint videos into luma so the model is not taught a
    camera's lavender cast as eye colour; tint_mode='keep' leaves them as recorded.
"""
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch

from .gaze import GEOM_DIM, POS_PAIRS, RADIUS_DIMS, load_group_labels

STORE_W = 112          # stored width at 96 px height (scaled with image_size)
CLASSES = ("gray", "tint", "colour")


def colour_class(meta):
    if meta["chroma"] < 1.0:
        return "gray"
    return "tint" if meta["hue_spread"] < 0.012 else "colour"


def split_groups_stratified(metas, val_frac, seed):
    """Deterministic group-level split, stratified by colour class. Returns the set of validation group names.
    Splits the FULL manifest so val membership never changes when a subset of videos is loaded."""
    by_class = {c: sorted({m["group"] for m in metas if colour_class(m) == c}) for c in CLASSES}
    rng = random.Random(seed)
    val = set()
    for c, groups in by_class.items():
        if len(groups) < 2 or val_frac <= 0:
            continue
        rng.shuffle(groups)
        val |= set(groups[:min(len(groups) - 1, max(1, round(val_frac * len(groups))))])
    return val


def source_kinds(cache_dirs):
    """(dynamics_sources, appearance_sources) over one or more cache dirs, from the manifests (kind='appearance' marks
    single-image sources stored as length-1 sequences; everything else is a temporal source)."""
    dyn, app = set(), set()
    for d in ([cache_dirs] if isinstance(cache_dirs, (str, Path)) else cache_dirs):
        for m in json.loads((Path(d) / "manifest.json").read_text()):
            (app if m.get("kind") == "appearance" else dyn).add(m.get("source", "video"))
    return sorted(dyn), sorted(app)


def train_only_sources(cache_dirs):
    """Sources whose every manifest entry is marked "train_only" (no validation groups exist for them)."""
    flags = {}
    for d in ([cache_dirs] if isinstance(cache_dirs, (str, Path)) else cache_dirs):
        for m in json.loads((Path(d) / "manifest.json").read_text()):
            s = m.get("source", "video")
            flags[s] = flags.get(s, True) and bool(m.get("train_only"))
    return sorted(s for s, f in flags.items() if f)


def subset_groups(metas, frac, seed):
    """Keep ceil(frac * n) groups per colour class (stratified). One fixed permutation per class, so a smaller frac is
    always a subset of a larger one (nested data-scaling curve)."""
    by_class = {}
    for m in metas:
        by_class.setdefault(colour_class(m), {}).setdefault(m["group"], []).append(m)
    keep_groups = set()
    for cls in sorted(by_class):
        groups = sorted(by_class[cls])
        order = np.random.RandomState(seed + 17).permutation(len(groups))
        keep_groups.update(groups[i] for i in order[:max(1, int(math.ceil(frac * len(groups))))])
    return [m for m in metas if m["group"] in keep_groups]


class CacheBank:
    def __init__(self, cache_dir, image_size=96, split="train", val_frac=0.15, seed=0, tint_mode="gray",
                 max_videos=None, max_frames_per_video=None, device="cpu", hflip=True, group_frac=1.0,
                 source_weights=None, sources=None, labels_dir=None):
        """cache_dir: one directory or a list of them (each with its own manifest.json). source_weights: optional
        {source: weight} (sources = manifest 'source' field, e.g. video / openeds); the weight is the share of samples
        a source gets, groups uniform inside it. None = every group equally likely (a source with more groups dominates).
        sources: keep only these sources AFTER the split was made (per-source evaluation sets).
        labels_dir: aligned mask cache (build_cache_openeds.py --masks). Frames of groups with a mask file then also come with
        eye geometry + a cleaned gaze command (ivs.gaze), mapped through the SAME scale / crop / flip as the pixels; other
        frames get validity 0. sample() returns geom, geom_valid, cmd, has_cmd, off, flip in that case."""
        assert split in ("train", "val", "all") and tint_mode in ("gray", "keep")
        assert 0.0 < group_frac <= 1.0
        self.size, self.split, self.device, self.hflip = image_size, split, torch.device(device), hflip
        self.source_weights = dict(source_weights) if source_weights else None
        scale = image_size / 96.0
        self.store_w = max(image_size, int(round(STORE_W * scale)))
        dirs = [Path(d) for d in ([cache_dir] if isinstance(cache_dir, (str, Path)) else cache_dir)]
        metas = []
        for d in dirs:
            for m in json.loads((d / "manifest.json").read_text()):
                m["_dir"] = str(d)
                m.setdefault("source", "video")
                metas.append(m)
        # train/val split is made INSIDE each source, so adding a source never changes another source's val groups
        val_groups = set()
        for src in sorted({m["source"] for m in metas}):
            # manifest entries with "train_only": true (synthetic clips, e.g. the dream pool) never enter the val split and never change another group's membership
            val_groups |= split_groups_stratified([m for m in metas if m["source"] == src and not m.get("train_only")], val_frac, seed)
        keep = [m for m in metas if split == "all" or ((m["group"] in val_groups) == (split == "val"))]
        if sources:
            keep = [m for m in keep if m["source"] in set(sources)]
        keep.sort(key=lambda m: m["name"])
        if group_frac < 1.0:                                   # data-scaling experiments: fewer SUBJECTS, same val set
            keep = subset_groups(keep, group_frac, seed)
        if max_videos:
            keep = keep[:max_videos]
        frames, seq_start, seq_len, seq_group, seq_class, self.videos = [], [], [], [], [], []
        lab_geom, lab_valid, lab_cmd, lab_has = [], [], [], []
        group_ids, group_source, total = {}, {}, 0
        for m in keep:
            with np.load(Path(m["_dir"]) / (_safe(Path(m["name"]).stem) + ".npz")) as z:
                fr, bounds = z["frames"], z["seq_bounds"]
            cls = colour_class(m)
            raw_h, raw_w = fr.shape[1], fr.shape[2]
            fr = self._prepare(fr, scale, cls, tint_mode, m.get("pad_mode", "edge"))
            if max_frames_per_video:
                cut = int(np.searchsorted(bounds, max_frames_per_video, side="right")) - 1
                cut = max(1, cut)
                bounds = bounds[:cut + 1]
                fr = fr[:bounds[-1]]
            if labels_dir is not None:
                lab_geom_i, lab_valid_i, lab_cmd_i, lab_has_i = self._labels_for(labels_dir, m, len(fr), max_frames_per_video,
                                                                               raw_h, raw_w, scale)
                lab_geom.append(lab_geom_i); lab_valid.append(lab_valid_i); lab_cmd.append(lab_cmd_i); lab_has.append(lab_has_i)
            gid = group_ids.setdefault(m["group"], len(group_ids))
            for a, b in zip(bounds[:-1], bounds[1:]):
                seq_start.append(total + int(a)); seq_len.append(int(b - a)); seq_group.append(gid)
                seq_class.append(CLASSES.index(cls if not (tint_mode == "gray" and cls == "tint") else "gray"))
            frames.append(fr)
            total += len(fr)
            self.videos.append(dict(name=m["name"], group=m["group"], cls=cls, frames=len(fr), source=m["source"]))
            group_source[gid] = m["source"]
        if not frames:
            raise RuntimeError(f"no videos for split={split} in {[str(d) for d in dirs]}")
        if self.source_weights:
            unknown = set(self.source_weights) - set(group_source.values())
            if unknown:
                raise ValueError(f"source_weights names sources not present in this split: {sorted(unknown)}")
        self.group_source = group_source
        self.frames = torch.from_numpy(np.concatenate(frames)).to(self.device)        # N,H,store_w,3 uint8
        self.has_labels = labels_dir is not None
        if self.has_labels:                                      # stored-frame pixel units (see _labels_for)
            self.geom = torch.from_numpy(np.concatenate(lab_geom)).to(self.device)           # N,7
            self.geom_valid = torch.from_numpy(np.concatenate(lab_valid)).to(self.device)    # N,7 bool
            self.cmd = torch.from_numpy(np.concatenate(lab_cmd)).to(self.device)             # N,2
            self.has_cmd = torch.from_numpy(np.concatenate(lab_has)).to(self.device)         # N bool
            assert len(self.geom) == len(self.frames)
        self.seq_start = np.array(seq_start); self.seq_len = np.array(seq_len)
        self.seq_group = np.array(seq_group); self.seq_class = np.array(seq_class)
        self.n_groups = len(group_ids)
        self._probs = {}

    def _prepare(self, fr, scale, cls, tint_mode, pad_mode="edge"):
        n, h, w, _ = fr.shape
        if h != self.size:
            fr = np.stack([cv2.resize(f, (max(1, int(round(w * scale))), self.size), interpolation=cv2.INTER_AREA) for f in fr])
            w = fr.shape[2]
        if w >= self.store_w:
            x0 = (w - self.store_w) // 2
            fr = fr[:, :, x0:x0 + self.store_w]
        else:
            pad = self.store_w - w
            fr = np.pad(fr, ((0, 0), (0, 0), (pad // 2, pad - pad // 2), (0, 0)), mode=pad_mode)   # edge | reflect
        if tint_mode == "gray" and cls in ("gray", "tint"):
            g = (fr[..., 0] * 0.299 + fr[..., 1] * 0.587 + fr[..., 2] * 0.114).round().clip(0, 255).astype(np.uint8)
            fr = np.repeat(g[..., None], 3, axis=3)
        return np.ascontiguousarray(fr)

    def _labels_for(self, labels_dir, m, n, max_frames, raw_h, raw_w, scale):
        """geom (n,7), valid (n,7), cmd (n,2), has (n,) of one video in STORED-frame pixel units (self.size high, store_w wide):
        the same resize / centre-crop (or edge-pad) mapping as _prepare, so labels land on the pixels they describe."""
        lab = load_group_labels(labels_dir, m["group"], max_frames)
        if lab is None:
            return (np.zeros((n, GEOM_DIM), np.float32), np.zeros((n, GEOM_DIM), bool), np.zeros((n, 2), np.float32), np.zeros(n, bool))
        if len(lab["geom"]) != n:
            raise ValueError(f"{m['group']}: {len(lab['geom'])} labels for {n} frames (mask cache out of date? rebuild with --masks)")
        resized = raw_h != self.size
        new_w = max(1, int(round(raw_w * scale))) if resized else raw_w
        fx, fy = new_w / raw_w, self.size / raw_h
        x0 = (new_w - self.store_w) // 2 if new_w >= self.store_w else -((self.store_w - new_w) // 2)    # columns cropped (+) / padded (-)
        mx = lambda x: (x + 0.5) * fx - 0.5 - x0
        my = lambda y: (y + 0.5) * fy - 0.5
        g, v = lab["geom"].copy(), lab["valid"]
        for xi, yi in POS_PAIRS:
            g[:, xi], g[:, yi] = mx(g[:, xi]), my(g[:, yi])
        for ri in RADIUS_DIMS:
            g[:, ri] *= (fx + fy) / 2
        c = np.stack([mx(lab["cmd"][:, 0]), my(lab["cmd"][:, 1])], 1).astype(np.float32)
        has = lab["has_cmd"]
        return np.where(v, g, 0).astype(np.float32), v, np.where(has[:, None], c, 0).astype(np.float32), has

    def _label_batch(self, idx, off, flip):
        """Labels of the sampled windows after the same crop offset / h-flip as the pixels, normalised so the image spans
        [-1,1] (positions) and radii are fractions of half the image size. Invalid entries are 0 (mask with *_valid / has_cmd)."""
        size = self.size
        half, mid = size / 2.0, (size - 1) / 2.0
        g, v, c, hc = self.geom[idx], self.geom_valid[idx], self.cmd[idx], self.has_cmd[idx]            # B,T,...
        offf, fl = off.view(-1, 1).float(), flip.view(-1, 1)
        px = lambda x: (torch.where(fl, (size - 1) - (x - offf), x - offf) - mid) / half
        py = lambda y: (y - mid) / half
        gn = torch.zeros_like(g)
        for xi, yi in POS_PAIRS:
            gn[..., xi], gn[..., yi] = px(g[..., xi]), py(g[..., yi])
        for ri in RADIUS_DIMS:
            gn[..., ri] = g[..., ri] / half
        cn = torch.stack([px(c[..., 0]), py(c[..., 1])], -1)
        return dict(geom=torch.where(v, gn, torch.zeros_like(gn)), geom_valid=v,
                    cmd=torch.where(hc[..., None], cn, torch.zeros_like(cn)), has_cmd=hc, off=off, flip=flip, index=idx)

    @torch.no_grad()
    def labels_at(self, index):
        """Labels (geom, geom_valid, cmd, has_cmd; image units) of the frames at `index` (any int tensor/array shape (...)) with the
        EVALUATION view: centre crop, no flip (what bank.sample(train=False) shows). Returns tensors shaped (index.shape, ...)."""
        idx = torch.as_tensor(np.asarray(index), dtype=torch.long, device=self.device)
        flat = idx.reshape(-1, 1)
        n = flat.shape[0]
        g = self._label_batch(flat, torch.full((n,), (self.store_w - self.size) // 2, dtype=torch.long, device=self.device),
                              torch.zeros(n, dtype=torch.bool, device=self.device))
        return {k: g[k][:, 0].reshape(*idx.shape, *g[k].shape[2:]) for k in ("geom", "geom_valid", "cmd", "has_cmd")}

    @torch.no_grad()
    def label_stats(self):
        """(mean, std) (G,) of the VALID geometry entries over the whole bank at the centre crop, in image units: the
        standardisation of the geometry head (std floored at .01)."""
        g = self.labels_at(np.arange(len(self)))
        x, v = g["geom"], g["geom_valid"].float()
        cnt = v.sum(0).clamp(min=1)
        mean = (x * v).sum(0) / cnt
        std = (((x - mean) ** 2 * v).sum(0) / cnt).sqrt().clamp(min=0.01)
        return mean.cpu(), std.cpu()

    # ------------------------------------------------------------------ info
    def __len__(self):
        return int(self.frames.shape[0])

    def info(self):
        cls = {CLASSES[c]: int(self.seq_len[self.seq_class == c].sum()) for c in range(len(CLASSES))}
        by_source = {}
        for v in self.videos:
            by_source[v["source"]] = by_source.get(v["source"], 0) + v["frames"]
        return dict(split=self.split, videos=len(self.videos), groups=self.n_groups, frames=len(self),
                    sequences=len(self.seq_len), max_seq=int(self.seq_len.max()), frames_by_class=cls,
                    frames_by_source=by_source, source_weights=self.source_weights)

    def max_window(self):
        return int(self.seq_len.max())

    def group_names(self):
        """{group id (as in seq_group): group name}, ids numbered in manifest order exactly as __init__ assigned them."""
        ids = {}
        for v in self.videos:
            ids.setdefault(v["group"], len(ids))
        return {g: name for name, g in ids.items()}

    @torch.no_grad()
    def window_at(self, seq, offset, t):
        """(t,3,H,W) float in [0,1]: frames [offset, offset+t) of sequence `seq` in the EVALUATION view (centre crop, no flip)."""
        if offset < 0 or offset + t > int(self.seq_len[seq]):
            raise ValueError(f"window [{offset}, {offset + t}) does not fit sequence {seq} of {int(self.seq_len[seq])} frames")
        s, x0 = int(self.seq_start[seq]) + offset, (self.store_w - self.size) // 2
        return self.frames[s:s + t, :, x0:x0 + self.size].permute(0, 3, 1, 2).float() / 255.0

    # ------------------------------------------------------------------ sampling
    def _probabilities(self, t):
        if t in self._probs:
            return self._probs[t]
        windows = np.maximum(0, self.seq_len - t + 1).astype(np.float64)
        p = np.zeros(len(windows))
        eligible_groups = [g for g in range(self.n_groups) if windows[self.seq_group == g].sum() > 0]
        if not eligible_groups:
            raise RuntimeError(f"no sequence is long enough for T={t} (longest {self.seq_len.max()})")
        if self.source_weights:                                  # share per source, uniform over its eligible groups
            per_src = {}
            for g in eligible_groups:
                per_src.setdefault(self.group_source[g], []).append(g)
            total_w = sum(self.source_weights.get(s, 0.0) for s in per_src)
            if total_w <= 0:
                raise RuntimeError(f"no eligible source has positive weight for T={t}: {self.source_weights}")
            gp = {g: self.source_weights.get(s, 0.0) / total_w / len(gs) for s, gs in per_src.items() for g in gs}
        else:
            gp = {g: 1.0 / len(eligible_groups) for g in eligible_groups}
        for g in eligible_groups:
            m = self.seq_group == g
            p[m] = windows[m] / windows[m].sum() * gp[g]
        self._probs[t] = (torch.from_numpy(p), windows)
        return self._probs[t]

    @torch.no_grad()
    def sample(self, batch, t, generator=None, train=True):
        """Returns dict: images float (B,T,3,H,W) in [0,1], is_colour (B,) bool (sequence belongs to the colour class)."""
        probs, windows = self._probabilities(t)
        gen = generator
        seq = torch.multinomial(probs, batch, replacement=True, generator=gen).numpy()
        start = self.seq_start[seq] + np.floor(torch.rand(batch, generator=gen).numpy() * windows[seq]).astype(np.int64)
        idx = torch.from_numpy(start[:, None] + np.arange(t)[None, :]).to(self.device)           # B,T
        if train:
            off = torch.randint(0, self.store_w - self.size + 1, (batch,), generator=gen).to(self.device)
        else:
            off = torch.full((batch,), (self.store_w - self.size) // 2, dtype=torch.long, device=self.device)
        cols = off[:, None] + torch.arange(self.size, device=self.device)[None, :]               # B,size
        fr = self.frames[idx]                                                                    # B,T,H,store_w,3
        bi = torch.arange(batch, device=self.device)[:, None, None, None]
        ti = torch.arange(t, device=self.device)[None, :, None, None]
        hi = torch.arange(self.size, device=self.device)[None, None, :, None]
        fr = fr[bi, ti, hi, cols[:, None, None, :]]                                              # B,T,H,size,3
        flip = torch.zeros(batch, dtype=torch.bool, device=self.device)
        if train and self.hflip:
            flip = (torch.rand(batch, generator=gen) < 0.5).to(self.device)
            fr = torch.where(flip.view(batch, 1, 1, 1, 1), fr.flip(3), fr)
        x = fr.permute(0, 1, 4, 2, 3).float() / 255.0
        out = dict(images=x, is_colour=torch.from_numpy(self.seq_class[seq] == CLASSES.index("colour")).to(self.device))
        if self.has_labels:
            out.update(self._label_batch(idx, off, flip))
        return out


def _safe(stem):
    import re
    return re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")
