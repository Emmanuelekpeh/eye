"""Birth memory: the model's OWN latent state after it has perceived a real eye sequence once, offline.

The UI used to start from noise (the prior sampled from a zero state), which only sometimes settles into an eye. A birth
memory replaces that start: scripts/make_birth.py lets the trained model observe a short real window with its own
encoder/RSSM and keeps only the resulting latent (h, z) plus where it came from. At run time the UI loads that file, decodes
the state with the model's decoder and carries on; no dataset frame is read or shown. A file is only valid for the exact
checkpoint it was made with (sha256 of the checkpoint file), because a latent state means nothing to another set of weights.
"""
import hashlib
from pathlib import Path

import torch

VERSION = 1


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def to_model_channels(x, channels):
    """(..., C, H, W) real frames -> the model's channel count (RGB -> BT.601 luma for a gray model)."""
    if x.shape[-3] == channels:
        return x
    if channels == 1 and x.shape[-3] == 3:
        w = torch.tensor([0.299, 0.587, 0.114], dtype=x.dtype, device=x.device).view(3, 1, 1)
        return (x * w).sum(-3, keepdim=True)
    raise ValueError(f"cannot feed {x.shape[-3]}-channel frames to a {channels}-channel model")


@torch.no_grad()
def perceive(model, windows, seed=0):
    """The model watches real windows (K,T,C,H,W in [0,1]) with zero action. Returns h (K,deter), z (K,stoch) at the last frame
    and recon (K,C,H,W), the model's own render of that state."""
    model.eval()
    x = to_model_channels(windows, model.cfg.channels)
    torch.manual_seed(int(seed) & 0x7FFFFFFF)
    out = model.observe(x, torch.zeros(x.shape[0], x.shape[1], model.cfg.action_dim))
    return dict(h=out["h"][:, -1], z=out["z"][:, -1], recon=out["recon"][:, -1])


def save_birth(path, h, z, checkpoint, meta=None):
    """h (K,deter), z (K,stoch). `meta` (plain python: lists of strings/numbers) says where each state came from."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(version=VERSION, h=h.detach().cpu().float(), z=z.detach().cpu().float(),
                    checkpoint_name=Path(checkpoint).name, checkpoint_sha256=file_sha256(checkpoint),
                    meta=meta or {}), path)


def load_birth(path, checkpoint, cfg):
    """Load a birth memory for `checkpoint` (a path) whose ModelConfig is `cfg`; refuses a file made for other weights."""
    b = torch.load(path, map_location="cpu", weights_only=True)
    if b.get("version") != VERSION:
        raise ValueError(f"{path}: birth memory version {b.get('version')}, expected {VERSION}")
    if b["checkpoint_sha256"] != file_sha256(checkpoint):
        raise ValueError(f"{path} was made for {b['checkpoint_name']} (sha256 {b['checkpoint_sha256'][:12]}), "
                         f"not for {checkpoint}: a latent state only means something to the weights that produced it")
    if b["h"].shape[1] != cfg.deter_dim or b["z"].shape[1] != cfg.stoch_dim or len(b["h"]) != len(b["z"]) or len(b["h"]) < 1:
        raise ValueError(f"{path}: state shapes {tuple(b['h'].shape)} / {tuple(b['z'].shape)} do not fit the model "
                         f"(deter {cfg.deter_dim}, stoch {cfg.stoch_dim})")
    return b
