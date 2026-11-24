"""Run: python -m pytest -p no:xonsh tests/test_identity_curve.py   (real held-out context windows from the real cache; real L1 checkpoint for the dream path)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import pytest
import torch
import torch.nn.functional as F

import identity_curve as IC

CACHE = [str(ROOT.parent / "data" / "cache" / "videos"), str(ROOT.parent / "data" / "cache" / "openeds")]
L1 = ROOT / "pilot_results" / "kaggle_long3_30k" / "pilot" / "L1_v6_long" / "model.pt"
_c = {}


def wins():
    if "w" not in _c:
        _c["w"] = IC.worlds(CACHE, [0])[0]
    return _c["w"]


def test_real_frames_retrieve_themselves_and_shifted_ones_do_not():
    w = wins()
    assert w.shape[0] >= 30 and w.shape[2] == 3
    last = w[:, -1]
    m = IC.identity_metrics(last, last)
    assert m["top1"] == 1.0 and m["mrr"] == 1.0 and m["drift"] == 0.0 and m["own_vs_other"] < 1e-6
    n = len(last)
    roll = IC.identity_metrics(last.roll(1, 0), last)                          # every dream is another world's picture
    prev = IC.identity_metrics(w[:, -8], last)                                 # the same eye 7 steps earlier: ordinary change keeps identity
    assert roll["top1"] == 0.0 and roll["mrr"] < 0.5
    assert roll["own_vs_other"] > 2 * prev["own_vs_other"]                     # raw mse ratio is the weaker metric (real eyes look alike); retrieval is primary
    assert prev["top1"] > 0.6 and prev["own_vs_other"] < 0.5                   # measured 0.79: blinks / saccades within 7 ticks cost some retrieval, so even real frames are not 1.0


def test_metric_is_blur_tolerant_but_chroma_and_contrast_blind():
    last = wins()[:, -1]
    blur = F.avg_pool2d(F.pad(last, (3, 3, 3, 3), mode="replicate"), 7, 1)
    assert IC.identity_metrics(blur, last)["top1"] > 0.9                       # a soft copy is still its own eye
    tinted = last * torch.tensor([1.0, 0.8, 0.6]).view(1, 3, 1, 1)
    assert IC.identity_metrics(tinted, last)["top1"] > 0.9                     # colour change alone does not change identity
    prototype = last.mean(0, keepdim=True).expand_as(last)                     # every world collapsed onto one picture
    assert IC.identity_metrics(prototype, last)["top1"] < 0.2


def test_thumb_is_unit_norm_and_curve_uses_one_based_ticks():
    t = IC.thumb(wins()[:, -1])
    assert torch.allclose(t.norm(dim=1), torch.ones(len(t)), atol=1e-5) and abs(float(t.mean())) < 1e-5
    fr = torch.stack([wins()[:, -1], wins()[:, -1].roll(1, 0)])                # tick 1 = frames[0] (own), tick 2 = frames[1] (rolled)
    c = IC.curve(fr, wins()[:, -1], ticks=(1, 2, 5))
    assert set(c) == {"1", "2"} and c["1"]["top1"] == 1.0 and c["2"]["top1"] == 0.0


@pytest.mark.skipif(not L1.exists(), reason="needs the L1 checkpoint")
def test_real_model_dream_has_reference_rows_and_decays_from_its_context():
    model = IC.load(L1)
    w = wins()[:12]
    mean, per = IC.model_curve(model, [w], ticks=(1, 5, 30), seed=0)
    assert mean["recon"]["top1"] > 0.8                                         # the model's own render of the context is that eye
    assert mean["real_-8"]["top1"] > 0.5
    assert mean["1"]["top1"] >= mean["30"]["top1"] - 0.3 and len(per) == 1
    assert all(np.isfinite(v) for k in ("1", "5", "30") for v in mean[k].values())
