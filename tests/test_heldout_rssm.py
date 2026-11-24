"""Run: python -m pytest -p no:xonsh tests/test_heldout_rssm.py   (real L1 checkpoint + the real R3 held-out pack data/heldout_pack; tiny real-cache pilot run for the trainer switch)."""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import numpy as np
import pytest
import torch

import heldout_clarity as H
import heldout_rssm as HR
from ivs import pilot
from ivs.config import ModelConfig
from ivs.model import EyeWorldModel

L1 = ROOT / "pilot_results" / "kaggle_long3_30k" / "pilot" / "L1_v6_long" / "model.pt"
PACK = ROOT.parent / "data" / "heldout_pack"
R5 = ROOT / "pilot_results" / "kaggle_r5" / "r5" / "results_l1.json"
_c = {}


def items():
    if "i" not in _c:
        _c["i"] = H.build_items(PACK, 96)
    return _c["i"]


def test_prior_stopgrad_changes_gradients_only_and_defaults_off():
    assert EyeWorldModel(ModelConfig(image_size=32, deter_dim=16, stoch_dim=8, feature_dim=32, hidden_dim=16)).rssm.prior_stopgrad is False
    cfg = ModelConfig(image_size=32, channels=3, deter_dim=16, stoch_dim=8, feature_dim=32, hidden_dim=16)
    x, a = torch.rand(2, 4, 3, 32, 32), torch.zeros(2, 4, 5)
    outs = {}
    for flag in (False, True):
        torch.manual_seed(0)
        m = EyeWorldModel(cfg)
        m.rssm.prior_stopgrad = flag
        outs[flag] = m.observe(x, a)
    for k in ("h", "z", "recon"):
        assert torch.equal(outs[False][k], outs[True][k])
    assert all(torch.equal(p[0], q[0]) and torch.equal(p[1], q[1]) for p, q in zip(outs[False]["prior"], outs[True]["prior"]))
    assert "prior_stopgrad" not in EyeWorldModel(cfg).state_dict()


def test_finetune_presets_are_l1_plus_only_their_knob():
    base = pilot.PRESETS["L1_v6_long"]
    rest = lambda n, drop: {k: v for k, v in pilot.PRESETS[n].items() if k not in drop}
    keep = {k: v for k, v in base.items() if k != "stages"}
    for n in ("P0_ft_control", "P0_ft_control_s1", "P1_ft_stopgrad"):
        assert pilot.PRESETS[n]["stages"] == ((8, 1.0),)
    assert rest("P0_ft_control", ("stages",)) == keep
    assert rest("P0_ft_control_s1", ("stages", "train_seed")) == keep and pilot.PRESETS["P0_ft_control_s1"]["train_seed"] == 1
    assert rest("P1_ft_stopgrad", ("stages", "prior_stopgrad")) == keep and pilot.PRESETS["P1_ft_stopgrad"]["prior_stopgrad"] is True
    assert pilot.RunConfig().prior_stopgrad is False and pilot.build_run("L1_v6_long", 10).prior_stopgrad is False


def test_train_run_applies_the_switch_same_first_loss_different_weights():
    import test_pilot as TP
    with tempfile.TemporaryDirectory() as root:
        dv, ds = TP._mini_multisource(root, n_video=6, n_s=10)
        small = {"image_weight": 0.0, "stages": ((8, 1.0),), "warmup": 2, "stoch_dim": 8, "deter_dim": 16, "feature_dim": 32, "hidden_dim": 16}
        pilot.PRESETS["TFT0"] = {**pilot.PRESETS["P0_ft_control"], **small}
        pilot.PRESETS["TFT1"] = {**pilot.PRESETS["P1_ft_stopgrad"], **small}
        try:
            common = ["--cache", str(dv), str(ds), "--source-weights", "video=0.5,openeds=0.5", "--steps", "12", "--batch", "4", "--image-size", "64", "--device", "cpu",
                      "--max-frames-per-video", "200", "--val-frac", "0.34", "--eval-windows", "64", "--log-every", "1", "--out", str(Path(root) / "o"), "--runs", "TFT0", "TFT1"]
            pilot.main(common)
        finally:
            del pilot.PRESETS["TFT0"], pilot.PRESETS["TFT1"]
        log = lambda n: [json.loads(l) for l in open(Path(root) / "o" / n / "train_log.jsonl")]
        l0, l1 = log("TFT0"), log("TFT1")
        assert l0[0]["loss"] == l1[0]["loss"], "forward values are identical, so the first step's loss must match"
        assert l0[-1]["loss"] != l1[-1]["loss"], "the gradients differ, so training must diverge"
        w0 = torch.load(Path(root) / "o" / "TFT0" / "model.pt", map_location="cpu", weights_only=False)
        w1 = torch.load(Path(root) / "o" / "TFT1" / "model.pt", map_location="cpu", weights_only=False)
        assert w0["run"]["prior_stopgrad"] is False and w1["run"]["prior_stopgrad"] is True
        assert not torch.equal(w0["model"]["rssm.gru.weight_hh"], w1["model"]["rssm.gru.weight_hh"])


@pytest.mark.skipif(not (L1.exists() and PACK.exists() and R5.exists()), reason="needs the L1 checkpoint, data/heldout_pack and the R5 results")
def test_l1_heldout_scores_reproduce_the_recorded_r5_base_numbers():
    it = items()
    assert len(it) == 344 or len(it) > 300
    ceil = H.score_items(it, [i["view"] for i in it])
    m = HR.load_model(L1)
    recs = HR.recon_views(m, it)
    assert len(recs) == len(it) and recs[0].shape == (96, 96, 3) and recs[0].dtype == np.uint8
    assert all(np.array_equal(a, b) for a, b in zip(recs[:5], HR.recon_views(m, it[:5])))          # seeded: reproducible
    res = HR.score_model(m, it, ceil)
    rec5 = json.loads(R5.read_text())["arms"]["base"]
    for s in ("openeds", "video_gray", "video_colour"):
        for k in ("edge_keep", "edge_fidelity"):
            assert abs(res[s]["scores"][k] - rec5[s]["metrics"][k]) < 0.01, (s, k, res[s]["scores"][k], rec5[s]["metrics"][k])
    assert not any(r["passed"] for r in res.values())                                                 # L1 fails the held-out gate (recorded)
