"""Which trained world models can the UI load? (python -m ivs.ui model switcher)

Scans `<root>/pilot_results/**` for the checkpoint files the pilot / stage-2 scripts write (`model.pt`, `model_s1.pt`, `model_s2.pt`;
`model_partial.pt` is mid-run insurance and is skipped) and `<root>/checkpoints/latest.pt`. A file is listed only when it really is an
`ivs.model.EyeWorldModel` checkpoint: it has `config` (all keys known to `ModelConfig`) and `model` whose tensors carry the encoder / rssm /
decoder prefixes. NCA / ViT / autoencoder / refiner checkpoints fail that test and are not listed. Nothing is loaded into memory beyond the
file header (mmap), nothing is ever deleted or moved. The UI only accepts ids from this list, never a path from the browser.
"""
import dataclasses
import json
from pathlib import Path

import torch

from .config import ModelConfig

FILES = ("model.pt", "model_s1.pt", "model_s2.pt")
PREFIXES = ("encoder.", "decoder.", "rssm.")
TIERS = ("eye", "some", "no")     # looked-at verdict from noise starts: eye in >= 6 of 8 tiles / 3-5 of 8 / blobs (pilot_results/model_curation.json)
SCORE_KEYS = ("eye_frac", "first100", "best100", "max_len", "any_eye", "still_frac", "chroma")   # scripts/score_models.py -> pilot_results/model_scores.json
_cache = {}          # (path, mtime_ns, size) -> entry dict or None


def _read(path):
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except Exception:                                   # older pickles are not zip files: no mmap
        ck = torch.load(path, map_location="cpu", weights_only=False)
    return ck


def inspect(path):
    """Header facts of one checkpoint, or None when it is not an EyeWorldModel checkpoint."""
    path = Path(path)
    st = path.stat()
    key = (str(path), st.st_mtime_ns, st.st_size)
    if key in _cache:
        return _cache[key]
    out = None
    try:
        ck = _read(path)
        cfg, sd = ck.get("config"), ck.get("model")
        known = {f.name for f in dataclasses.fields(ModelConfig)}
        if isinstance(cfg, dict) and isinstance(sd, dict) and set(cfg) <= known and all(any(k.startswith(p) for k in sd) for p in PREFIXES):
            c = ModelConfig(**cfg)
            out = dict(image_size=c.image_size, channels=c.channels, action_dim=c.action_dim, geom_dim=c.geom_dim,
                       params=int(sum(v.numel() for v in sd.values())), step=ck.get("step"))
    except Exception:
        out = None
    _cache[key] = out
    return out


def _label(rel):
    parts = Path(rel).parts
    if parts[0] == "checkpoints":
        return "latest.pt (first 64 px model)", "checkpoints"
    group = next((p[len("kaggle_"):] if p.startswith("kaggle_") else p for p in parts[1:] if p.startswith("kaggle_")), parts[1])
    run = next((p for p in reversed(parts[:-1]) if p not in ("pilot", "s2") and not p.startswith(("pilot_part", "kaggle_"))), parts[-2])
    stem = Path(parts[-1]).stem
    suffix = "" if stem == "model" else f" ({stem[len('model_'):]})"
    return f"{run}{suffix}", group


def discover(root):
    """[{id (posix path relative to root), label, group, size, channels, params, gaze, bytes}], sorted by group then label."""
    root = Path(root)
    found = []
    latest = root / "checkpoints" / "latest.pt"
    if latest.exists():
        found.append(latest)
    pr = root / "pilot_results"
    if pr.exists():
        found += [p for p in pr.rglob("*.pt") if p.name in FILES]
    scores = {}
    sf = pr / "model_scores.json"
    if sf.exists():
        try:
            scores = json.loads(sf.read_text())
        except ValueError:
            scores = {}
    tiers = {}
    cf = pr / "model_curation.json"      # looked-at verdict per model (eye / some / no), see its "note"; only valid for the file size it was made for
    if cf.exists():
        try:
            tiers = json.loads(cf.read_text()).get("models", {})
        except ValueError:
            tiers = {}
    out = []
    for p in sorted(set(found)):
        info = inspect(p)
        if info is None:
            continue
        rel = p.relative_to(root).as_posix()
        label, group = _label(rel)
        tr = tiers.get(rel)
        tier = tr["tier"] if tr and tr.get("bytes") == p.stat().st_size and tr.get("tier") in TIERS else None
        sc = scores.get(rel)
        if sc and sc.get("bytes") != p.stat().st_size:      # the file changed since it was scored
            sc = None
        out.append(dict(id=rel, label=label, group=group, size=info["image_size"], channels=info["channels"], params=info["params"],
                        gaze=info["action_dim"] >= 8, geom=info["geom_dim"] > 0, bytes=p.stat().st_size,
                        tier=tier, score={k: sc[k] for k in SCORE_KEYS} if sc else None))
    out.sort(key=lambda e: (TIERS.index(e["tier"]) if e["tier"] else len(TIERS), e["group"], e["label"]))
    return out


def resolve(root, model_id):
    """The checkpoint path for an id from discover(); ValueError for anything else (no path from the browser is ever opened)."""
    for e in discover(root):
        if e["id"] == model_id:
            return Path(root) / e["id"]
    raise ValueError(f"unknown model id {model_id!r}")
