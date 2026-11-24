"""Run: python -m pytest -p no:xonsh tests/test_overfit_rssm.py   (needs the real pack: python scripts/overfit_clarity.py pack --out ../overfit_pack; real S_105 frames, tiny step counts)."""
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

import overfit_rssm as R

PACK = ROOT.parent / "overfit_pack"
pytestmark = pytest.mark.skipif(not (PACK / "openeds.npz").exists(), reason="build the pack first (scripts/overfit_clarity.py pack)")
_c = {}


def clip():
    if "c" not in _c:
        _c["c"] = R.load_clip(PACK)
    return _c["c"]


def test_eval_windows_end_at_the_eval_frame():
    c = clip()
    n = len(c["train"])
    for t in (8, 4, 1):
        starts, pos = R.eval_windows(n, c["eval_idx"], t)
        assert (starts >= 0).all() and (starts + t <= n).all()
        assert (pos >= 0).all() and (pos < t).all()
        assert (starts + pos == c["eval_idx"]).all()
        late = c["eval_idx"] >= t - 1
        assert (pos[late] == t - 1).all()                              # window ends exactly on the eval frame wherever the clip allows


def test_deterministic_arm_uses_posterior_mean_and_stochastic_arm_samples():
    torch.manual_seed(0)
    x = torch.rand(2, 3, 3, 96, 96)
    a = torch.zeros(2, 3, 5)
    det, _ = R.build_model("C_det")
    o1, o2 = det.observe(x, a), det.observe(x, a)
    assert torch.equal(o1["recon"], o2["recon"])
    assert torch.equal(o1["z"][:, -1], o1["posterior"][-1][0])
    sto, _ = R.build_model("A_l1")
    s1, s2 = sto.observe(x, a), sto.observe(x, a)
    assert not torch.equal(s1["recon"], s2["recon"])
    assert not torch.equal(s1["z"][:, -1], s1["posterior"][-1][0])


def test_stopgrad_arm_blocks_prior_loss_gradient_into_the_gru_but_trains_the_prior_head():
    from ivs.learning import gaussian_kl
    x = torch.rand(2, 4, 3, 96, 96)
    a = torch.zeros(2, 4, 5)
    grads = {}
    for arm in ("A_l1", "G_stopgrad"):
        torch.manual_seed(2)
        m, _ = R.build_model(arm)
        o = m.observe(x, a)
        kl = torch.stack([gaussian_kl((q[0].detach(), q[1].detach()), pr) for q, pr in zip(o["posterior"], o["prior"])]).mean()      # = kl_dyn
        m.zero_grad()
        kl.backward()
        grads[arm] = (m.rssm.gru.weight_hh.grad, m.rssm.prior[0].weight.grad, m.encoder.net[0].weight.grad)
    gru, prior, enc = grads["G_stopgrad"]
    assert (gru is None or float(gru.abs().sum()) == 0.0) and (enc is None or float(enc.abs().sum()) == 0.0)
    assert float(prior.abs().sum()) > 0
    assert float(grads["A_l1"][0].abs().sum()) > 0                      # the L1 recipe does push the prior loss into the GRU


def test_arms_follow_l1_architecture_and_differ_only_where_stated():
    n = {a: sum(p.numel() for p in R.build_model(a)[0].parameters()) for a in R.ARMS}
    assert n["A_l1"] == n["B_nokl"] == n["C_det"]
    assert 4.0e6 < n["A_l1"] < 4.3e6                                   # L1 is 4.14 M
    assert n["D_wide"] > n["A_l1"]
    assert R.ARMS["B_nokl"]["kl_weight"] == 0 and R.ARMS["C_det"]["kl_weight"] == 0 and R.ARMS["D_wide"]["stoch_dim"] == 512
    _, p = R.build_model("A_l1")
    assert (p["kl_balance"], p["free_nats"], p["lr"], p["feature_dim"], p["deter_dim"], p["stoch_dim"]) == (0.99, 1.0, 1e-3, 512, 256, 128)


def test_nokl_loss_is_reconstruction_only_and_l1_adds_kl():
    c = clip()
    x = (torch.from_numpy(c["train"][:8]).permute(0, 3, 1, 2).float() / 255.0)[None].repeat(2, 1, 1, 1, 1)
    a = torch.zeros(2, 8, 5)
    torch.manual_seed(1)
    mb, pb = R.build_model("B_nokl")
    tb, partsb = R.train_step(mb, pb, x, a)
    assert abs(float(tb) - partsb["recon"]) < 1e-3 * max(1.0, partsb["recon"])
    torch.manual_seed(1)
    ma, pa = R.build_model("A_l1")
    ta, partsa = R.train_step(ma, pa, x, a)
    assert float(ta) > partsa["recon"] + 0.5                           # free-nats clamp: both KL terms contribute at least 1 nat x their weight (.99 + .01)


def test_reconstruct_eval_shapes_and_seed_reproducibility():
    c = clip()
    x_all = torch.from_numpy(c["train"]).permute(0, 3, 1, 2).float() / 255.0
    m, _ = R.build_model("A_l1")
    r1 = R.reconstruct_eval(m, x_all, c["eval_idx"], 8, 2, seed=5)
    r2 = R.reconstruct_eval(m, x_all, c["eval_idx"], 8, 2, seed=5)
    assert len(r1["sample"]) == 2 and r1["sample"][0].shape == (len(c["eval_idx"]), 96, 96, 3) and r1["sample"][0].dtype == np.uint8
    assert all(np.array_equal(a, b) for a, b in zip(r1["sample"], r2["sample"]))
    assert not np.array_equal(r1["sample"][0], r1["sample"][1])        # different noise draws differ
    assert m.training                                                  # restores the training flag


def test_gap_arms_override_window_batch_and_steps():
    c = clip()
    with tempfile.TemporaryDirectory() as d:
        r = R.run_arm("H_det_t1", c, d, steps=4, batch=2, t=8, seed=0, probe_every=2, draws=1, threads=4)
        assert (r["t"], r["batch"], r["steps"]) == (1, 64, 4)
        assert R.ARMS["J_det_long"]["steps_x"] == 2 and R.ARMS["I_det_wide"]["stoch_dim"] == 512
        assert np.isfinite(r["sample"]["scores"]["edge_fidelity"])


def test_training_reduces_loss_and_run_arm_writes_everything():
    c = clip()
    with tempfile.TemporaryDirectory() as d:
        res = R.run_arm("A_l1", c, d, steps=40, batch=4, t=8, seed=0, probe_every=20, draws=1, threads=4)
        assert res["curve"][-1]["recon"] < res["curve"][0]["recon"] * 0.9, res["curve"]
        assert res["params"] > 4.0e6 and res["steps"] == 40
        for name in ("arm_A_l1.json", "rec_A_l1.npz", "model_A_l1.pt"):
            assert (Path(d) / name).exists()
        saved = json.loads((Path(d) / "arm_A_l1.json").read_text())
        assert set(saved["sample"]["gate"]) == {"edge_keep", "edge_fidelity", "pupil_edge"}
        assert np.isfinite(saved["sample"]["scores"]["edge_fidelity"]) and np.isfinite(saved["mean"]["scores"]["psnr"])
        out = R.report(PACK, d)
        assert (Path(d) / "sheet_openeds_rssm.png").stat().st_size > 5000 and "A_l1" in out
