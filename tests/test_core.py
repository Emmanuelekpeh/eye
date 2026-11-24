"""Run: python tests/test_core.py   (or pytest)"""
from pathlib import Path
import sys, tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from PIL import Image
from ivs.data import EyeSequenceDataset, split_groups
from ivs.learning import PlasticityController
from ivs.actions import InterventionSampler


def _make(root, subjects=4, frames=8):
    for s in range(subjects):
        d = Path(root) / f"subject_{s}" / "session_0"
        d.mkdir(parents=True)
        for t in range(frames):
            Image.fromarray(np.full((16, 16), 10 * t + s, np.uint8)).save(d / f"f{t:03d}.png")


def test_split_is_by_group_and_disjoint():
    with tempfile.TemporaryDirectory() as r:
        _make(r)
        kw = dict(sequence_length=4, image_size=16, val_frac=0.5, seed=1)
        tr, va = EyeSequenceDataset(r, split="train", **kw), EyeSequenceDataset(r, split="val", **kw)
        assert set(tr.keys).isdisjoint(va.keys) and len(tr.keys) + len(va.keys) == 4


def test_stills_excluded_by_default():
    with tempfile.TemporaryDirectory() as r:
        _make(r, subjects=2)
        Image.fromarray(np.zeros((16, 16), np.uint8)).save(Path(r) / "still.png")
        ds = EyeSequenceDataset(r, sequence_length=4, image_size=16)
        assert "still" not in ds.keys


def test_passive_actions_are_zero():
    with tempfile.TemporaryDirectory() as r:
        _make(r, subjects=2)
        _, a = EyeSequenceDataset(r, sequence_length=4, image_size=16)[0]
        assert a.abs().sum() == 0


def test_sampler_shapes_range_and_first_action_zero():
    x = torch.rand(6, 5, 1, 32, 32)
    xi, a = InterventionSampler(p=1.0)(x, generator=torch.Generator().manual_seed(0))
    assert xi.shape == x.shape and a.shape == (6, 5, 5)
    assert xi.min() >= 0 and xi.max() <= 1 and a[:, 0].abs().max() == 0 and a[:, 1:].abs().sum() > 0
    xp, ap = InterventionSampler(p=0.0)(x)
    assert torch.equal(xp, x) and ap.abs().sum() == 0


def test_controller_shrinks_on_retention_failure_and_is_bounded():
    c = PlasticityController()
    for _ in range(30):
        c.update(0.1, retention_ok=False)
    assert all(0.1 <= s < 1.0 for s in c.scales.values())
    assert c.scales["encoder"] <= c.scales["rssm"]


def test_split_groups_never_all_val():
    assert split_groups(["a"], 0.5, 0) == set()
    v = split_groups(["a", "b", "c"], 0.9, 0)
    assert 1 <= len(v) <= 3


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
