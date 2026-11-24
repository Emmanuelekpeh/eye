"""Sequence loading with group-level (subject/session) train/val splits.

Group key = first path component under the data root (so organise real data as
data/eyes/<subject>/<session>/frame_*.png or data/eyes/<subject>/<video>.mp4).
A loose file at the root is its own group. Whole groups go to train OR val,
so adjacent frames of one recording never straddle the split.
"""
from pathlib import Path
import random
from typing import List, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _read_image(path: Path, size: int) -> np.ndarray:
    image = Image.open(path).convert("L").resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.uint8)[None, ...]          # 1,H,W uint8


def _read_video(path: Path, size: int, max_frames: int) -> List[np.ndarray]:
    import cv2  # lazy: image-only datasets must not require OpenCV
    cap = cv2.VideoCapture(str(path))
    frames, index = [], 0
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if index % 5 == 0:  # high-FPS eye video is redundant at full rate
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
            frames.append(frame.astype(np.uint8)[None, ...])
        index += 1
    cap.release()
    return frames


def _group_key(root: Path, path: Path) -> str:
    rel = path.relative_to(root)
    return rel.parts[0] if len(rel.parts) > 1 else rel.stem


def _discover(root: Path, include_stills: bool):
    sources = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            sources.append((_group_key(root, p), "video", p))
    for folder in sorted(p for p in root.rglob("*") if p.is_dir()):
        paths = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        if paths:
            sources.append((_group_key(root, paths[0]), "folder", paths))
    if include_stills:
        for p in sorted(root.iterdir()):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                sources.append((p.stem, "still", p))
    return sources


def split_groups(keys, val_frac: float, seed: int):
    """Deterministic group-level split. Returns the set of validation keys."""
    keys = sorted(set(keys))
    if len(keys) < 2 or val_frac <= 0:
        return set()
    rng = random.Random(seed)
    rng.shuffle(keys)
    n_val = min(len(keys) - 1, max(1, round(val_frac * len(keys))))   # always leave >=1 train group
    return set(keys[:n_val])


class EyeSequenceDataset(Dataset):
    """Ordered image sequences. Real data is passive: actions are zeros.

    split: 'train' | 'val' | 'all'. Stills are excluded unless include_stills=True
    (a repeated still teaches 'nothing ever changes', which poisons dynamics).
    Sequences shorter than sequence_length are skipped, not padded.
    """
    def __init__(self, root: str, sequence_length: int = 8, image_size: int = 64,
                 max_frames_per_video: int = 600, split: str = "all", val_frac: float = 0.2,
                 seed: int = 0, include_stills: bool = False):
        assert split in ("train", "val", "all")
        self.root = Path(root)
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.split = split
        self.keys: List[str] = []
        self.sequences: List[np.ndarray] = []   # each T,1,H,W uint8
        sources = _discover(self.root, include_stills) if self.root.exists() else []
        val_keys = split_groups([s[0] for s in sources], val_frac, seed)
        for key, kind, src in sources:
            is_val = key in val_keys
            if (split == "train" and is_val) or (split == "val" and not is_val):
                continue
            if kind == "video":
                frames = _read_video(src, image_size, max_frames_per_video)
            elif kind == "folder":
                frames = [_read_image(p, image_size) for p in src[:max_frames_per_video]]
            else:
                frames = [_read_image(src, image_size)] * sequence_length
            if len(frames) >= sequence_length:
                self.sequences.append(np.stack(frames, 0))
                self.keys.append(key)
        if not self.sequences:
            raise RuntimeError(f"No usable sequences (split={split}) under {self.root}")
        stride = max(1, sequence_length // 2)
        self.windows = [(i, s) for i, seq in enumerate(self.sequences)
                        for s in range(0, len(seq) - sequence_length + 1, stride)]

    def mean_image(self) -> torch.Tensor:
        total = np.zeros(self.sequences[0].shape[1:], dtype=np.float64)
        count = 0
        for seq in self.sequences:
            total += seq.sum(axis=0, dtype=np.float64)
            count += len(seq)
        return torch.from_numpy((total / count / 255.0).astype(np.float32))  # C,H,W

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        seq_idx, start = self.windows[index]
        
        # Check if this sequence is actually a static still image (all frames identical)
        # We assume it's a still if the sequence length is exactly what we padded it to
        # and we can check if the first and last frames are identical.
        is_still = False
        if len(self.sequences[seq_idx]) == self.sequence_length:
            first_frame = self.sequences[seq_idx][0]
            last_frame = self.sequences[seq_idx][-1]
            if np.array_equal(first_frame, last_frame):
                is_still = True

        if is_still:
            # If it's a still image, we just return the clip normally.
            # We will handle the "skip RSSM" logic in the training loop by passing a flag.
            clip = self.sequences[seq_idx][start:start + self.sequence_length]
            images = torch.from_numpy(clip).float() / 255.0
            actions = torch.zeros(self.sequence_length, 5, dtype=torch.float32)
            # We return a special flag in the actions tensor (e.g. set the first action value to -999)
            # to signal to the training loop that this is a static image and should only train the CNN.
            actions[:, 0] = -999.0
            return images, actions

        # 80% normal, 10% chimera (2 eyes), 10% multi-chimera (T eyes)
        mode = random.random()
        
        if mode < 0.8 or len(self.sequences) < 2:
            clip = self.sequences[seq_idx][start:start + self.sequence_length]
        elif mode < 0.9:
            # Chimera: Transition from one eye to another halfway through
            seq2_idx = random.choice([i for i in range(len(self.sequences)) if i != seq_idx])
            split_point = self.sequence_length // 2
            clip1 = self.sequences[seq_idx][start:start + split_point]
            
            seq2_len = len(self.sequences[seq2_idx])
            start2 = random.randint(0, max(0, seq2_len - (self.sequence_length - split_point)))
            clip2 = self.sequences[seq2_idx][start2:start2 + (self.sequence_length - split_point)]
            
            clip = np.concatenate([clip1, clip2], axis=0)
        else:
            # Multi-chimera: Rapidly shift between random eyes
            frames = []
            for _ in range(self.sequence_length):
                s_idx = random.randint(0, len(self.sequences) - 1)
                s_len = len(self.sequences[s_idx])
                f_idx = random.randint(0, s_len - 1)
                frames.append(self.sequences[s_idx][f_idx:f_idx+1])
            clip = np.concatenate(frames, axis=0)

        images = torch.from_numpy(clip).float() / 255.0           # T,C,H,W
        actions = torch.zeros(self.sequence_length, 5, dtype=torch.float32)  # passive observation
        return images, actions
