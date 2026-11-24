"""Run: python -m pytest -p no:xonsh tests/test_dream_pool.py   (real cache, real L1 checkpoint, real segmenter: the pool is built from actual dreams)."""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import pytest
import torch

import make_dream_pool as DP
from ivs.cachedata import CacheBank, source_kinds, train_only_sources

CACHE = [str(ROOT.parent / "data" / "cache" / "videos"), str(ROOT.parent / "data" / "cache" / "openeds")]
L1 = ROOT / "pilot_results" / "kaggle_long3_30k" / "pilot" / "L1_v6_long" / "model.pt"
SEG = ROOT / "pilot_results" / "eyemetric" / "segmenter.pt"
_pool = {}


def test_runs_and_smoothing_and_mask():
    assert DP.runs_of(np.array([0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1], bool), 3) == [(1, 4), (8, 12)]
    assert DP.runs_of(np.zeros(5, bool), 1) == [] and DP.runs_of(np.ones(5, bool), 5) == [(0, 5)]
    f = np.ones((12, 2), bool)
    f[5, 0] = False                                                       # one-tick flicker is filtered, a 3-tick gap is not
    f[2:5, 1] = False
    s = DP.smooth_flags(f)
    assert s[5, 0] and not s[3, 1]
    still = np.zeros((12, 2), bool)
    sharp = np.ones((12, 2))
    m = DP.keep_mask(f, still, sharp, 0.5, min_tick=4)
    assert not m[:3].any() and m[3:, 0].all()                              # tick 4 = row 3 is the first allowed row
    assert not DP.keep_mask(f, np.ones_like(still), sharp, 0.5, 1).any()   # frozen frames never kept
    assert not DP.keep_mask(f, still, sharp * 0.1, 0.5, 1).any()           # soft frames never kept


def test_frame_sharpness_orders_blur():
    x = torch.rand(4, 3, 96, 96)
    blur = torch.nn.functional.avg_pool2d(torch.nn.functional.pad(x, (2, 2, 2, 2), mode="replicate"), 5, 1)
    assert (DP.frame_sharpness(x) > 10 * DP.frame_sharpness(blur)).all()


@pytest.mark.skipif(not (L1.exists() and SEG.exists()), reason="needs L1 + segmenter")
def test_pool_from_real_dreams_is_train_only_and_cachebank_compatible():
    out = Path(tempfile.mkdtemp()) / "pool"
    info = DP.build({"L1": str(L1)}, CACHE, out, contexts=8, ticks=60, context_len=16, min_tick=16, min_run=8, sharp_ratio=0.0, seed=0, segmenter=str(SEG), chunk=8, log=lambda s: None)
    _pool["out"] = out
    assert info["total_frames"] > 0 and info["val_groups_used"] == 0
    man = json.loads((out / "manifest.json").read_text())
    assert man and all(m["source"] == "dream" and m["train_only"] and m["teacher"] == "L1" for m in man)
    assert train_only_sources([out]) == ["dream"] and train_only_sources(CACHE) == []
    # contexts are TRAIN recordings only
    val = CacheBank(CACHE, split="val", sources=["video", "openeds"], tint_mode="gray")
    val_groups = {v["group"] for v in val.videos}
    assert not ({m["context_group"] for m in man} & val_groups)
    # the dream source adds train groups and changes nothing about the val split
    caches = CACHE + [str(out)]
    dyn, app = source_kinds(caches)
    assert "dream" in dyn
    tr_with = CacheBank(caches, split="train", tint_mode="gray", sources=["dream"], source_weights={"dream": 1.0})
    assert len(tr_with) == info["total_frames"] and tr_with.n_groups == len(man)
    val_with = CacheBank(caches, split="val", tint_mode="gray", sources=["video", "openeds"])
    assert {v["group"] for v in val_with.videos} == val_groups
    with pytest.raises(RuntimeError):
        CacheBank(caches, split="val", tint_mode="gray", sources=["dream"])        # no validation groups exist for the dream source
    b = tr_with.sample(4, 8, generator=torch.Generator().manual_seed(0))
    assert b["images"].shape == (4, 8, 3, 96, 96) and float(b["images"].std()) > 0.02


def test_curate_moves_clips_and_keeps_the_rest():
    out = Path(tempfile.mkdtemp()) / "pool"
    out.mkdir()
    man = []
    for t in ("A", "B"):
        m = dict(name=f"dream_{t}.dream", group=f"dream_{t}", source="dream", teacher=t, train_only=True)
        np.savez_compressed(out / f"dream_{t}.npz", frames=np.zeros((4, 96, 96, 3), np.uint8), seq_bounds=np.array([0, 4]))
        man.append(m)
    (out / "manifest.json").write_text(json.dumps(man))
    (out / "pool.json").write_text(json.dumps(dict(teachers=dict(A=dict(kept=4), B=dict(kept=4)), clips=2, total_frames=8)))
    assert DP.curate(out, {"A"}, log=lambda s: None) == 4
    assert [m["teacher"] for m in json.loads((out / "manifest.json").read_text())] == ["B"]
    assert (out / "rejected" / "dream_A.npz").exists() and not (out / "dream_A.npz").exists() and (out / "dream_B.npz").exists()
    assert json.loads((out / "pool.json").read_text())["dropped_teachers"] == ["A"]


@pytest.mark.skipif(not (L1.exists() and SEG.exists()), reason="needs L1 + segmenter")
def test_grey_contexts_give_luma_frames_and_windows_stay_inside_runs():
    if "out" not in _pool:
        pytest.skip("pool test did not run")
    out = _pool["out"]
    man = json.loads((out / "manifest.json").read_text())
    for m in man:
        with np.load(out / (m["group"] + ".npz")) as z:
            fr, b = z["frames"], z["seq_bounds"]
        assert fr.shape[1:] == (96, 96, 3) and fr.dtype == np.uint8 and b[0] == 0 and b[-1] == len(fr) and (np.diff(b) >= 8).all()
        if m["context_class"] != "colour":
            assert m["chroma"] == 0.0 and (fr[..., 0] == fr[..., 1]).all() and (fr[..., 1] == fr[..., 2]).all()
