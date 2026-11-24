"""Run: python -m pytest -p no:xonsh tests/test_lab.py   (real checkpoints/latest.pt + the real eye segmenter; planted-signal checks use synthetic arrays only to test the estimators)."""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from ivs import eyemetric as E
from ivs import lab
from ivs.learning import gaussian_kl

CKPT = ROOT / "checkpoints" / "latest.pt"
SEG = ROOT / "pilot_results" / "eyemetric" / "segmenter.pt"
_cache = {}


def model():
    if "m" not in _cache:
        _cache["m"] = lab.load(CKPT)
    return _cache["m"]


def seg():
    if "s" not in _cache:
        _cache["s"] = E.load_segmenter(str(SEG))
    return _cache["s"]


def test_dream_is_bit_identical_to_the_ui_dream_loop():
    m = model()
    ref = E.ui_dream_frames(m, ticks=10, n=3, noise_seed=21)
    got = lab.dream(m, ticks=10, n=3, noise_seed=21)
    assert torch.equal(lab.as_float(got["frames"]), ref)
    assert got["kl"].shape == (10, 3, m.cfg.stoch_dim) and (got["kl"] > -1e-5).all()


def test_per_unit_kl_sums_to_the_trainers_kl():
    m = model()
    h, z = lab.seed_state(m, 2, 5)
    a = torch.randn(2, m.cfg.action_dim)
    x = torch.rand(2, m.cfg.channels, m.cfg.image_size, m.cfg.image_size)
    R = m.rssm
    torch.manual_seed(1)
    h2, z2, kl, qm = lab._observe(R, h, z, a, m.encoder(x))
    hh = R.gru(torch.cat([z, a], -1), h)
    prior, post = R._stats(R.prior(hh)), R._stats(R.posterior(torch.cat([hh, m.encoder(x)], -1)))
    assert torch.allclose(kl.sum(-1), gaussian_kl(post, prior), atol=1e-5) and torch.allclose(qm, post[0]) and torch.equal(h2, hh)


def test_actions_are_the_training_walk_and_labels_are_their_integral():
    a, s = lab.action_walk(4, 30, 3, action_dim=8)
    assert a.shape == (4, 30, 8) and (a[..., 5:] == 0).all() and (a[:, 0] == 0).all()
    sam = lab.InterventionSampler(p=1.0)
    states, a5 = sam._walk(4, 30, torch.Generator().manual_seed(3))
    assert torch.allclose(s, states - states[:, :1], atol=1e-5)          # cumulative action == hidden-state change since tick 0
    assert (s[..., :2].abs() <= 0.4 + 1e-6).all()


def test_noop_edit_changes_nothing_and_clamp_forces_the_units():
    m = model()
    D = m.cfg.deter_dim + m.cfg.stoch_dim
    base = lab.dream(m, 12, 3, 4)
    same = lab.dream(m, 12, 3, 4, intervention=lab.Intervention(m.cfg.deter_dim, add=np.zeros(D)))
    assert torch.equal(base["frames"], same["frames"])
    idx = [1, 5, m.cfg.deter_dim + 2]
    cl = lab.dream(m, 12, 3, 4, intervention=lab.Intervention(m.cfg.deter_dim, t0=4, clamp_idx=idx, clamp_val=[0.3, -0.2, 0.9]))
    st = torch.cat([cl["ph"], cl["pz"]], -1)
    assert torch.allclose(st[4:, :, idx], torch.tensor([0.3, -0.2, 0.9]).expand(8, 3, 3))
    assert torch.equal(base["frames"][:4], cl["frames"][:4]) and not torch.equal(base["frames"], cl["frames"])
    pst = torch.cat([cl["h"], cl["z"]], -1)                              # the carried state is clamped too
    assert torch.allclose(pst[4:, :, idx], torch.tensor([0.3, -0.2, 0.9]).expand(8, 3, 3))


def test_patching_all_units_makes_the_recipient_render_the_donor():
    m = model()
    D = m.cfg.deter_dim + m.cfg.stoch_dim
    rec = lab.dream(m, 14, 4, 6)
    src = {}
    for stage in ("pred", "post"):
        s = torch.tensor(lab._states(rec, stage), dtype=torch.float32)
        s[:, 0] = s[:, 1]
        src[stage] = s
    out = lab.dream(m, 14, 4, 6, intervention=lab.Intervention(m.cfg.deter_dim, t0=5, patch=(list(range(D)), src)))
    assert torch.equal(out["frames"][:5], rec["frames"][:5])
    assert (out["frames"][5:, 0].int() - rec["frames"][5:, 1].int()).abs().max() <= 1
    assert torch.equal(out["frames"][5:, 2], rec["frames"][5:, 2])         # untouched rows are untouched


def test_ridge_probe_recovers_a_planted_signal_and_not_noise_and_splits_by_world():
    rng = np.random.RandomState(0)
    n, T, D = 12, 40, 30
    X = rng.randn(n * T, D)
    w = rng.randn(D)
    y = X @ w + 0.3 * rng.randn(n * T)
    g = np.repeat(np.arange(n), T)
    r2, alpha, rows = lab.cv_r2(X, y, g, (0.1, 1, 10, 100), 4)
    assert r2 > 0.9 and rows == n * T
    assert lab.cv_r2(X, rng.permutation(y), g, (0.1, 1, 10, 100), 4)[0] < 0.1
    d, gain, sd = lab.fit_direction(X, y, 1.0)
    assert abs(np.dot(d, w / np.linalg.norm(w))) > 0.98 and abs(gain - np.linalg.norm(w)) / np.linalg.norm(w) < 0.1 and sd > 0
    # world-level leakage: a label that is constant inside each world but unrelated to X must NOT be predictable from held-out worlds
    yw = np.repeat(rng.randn(n), T)
    assert lab.cv_r2(X + np.repeat(rng.randn(n, D), T, 0) * 3, yw, g, (0.1, 1, 10, 100), 4)[0] < 0.2
    assert np.isnan(lab.cv_r2(X, np.full_like(y, np.nan), g, (1.0,), 4)[0]) and np.isnan(lab.cv_r2(X, np.ones_like(y), g, (1.0,), 4)[0])


def test_gram_ridge_equals_the_direct_closed_form():
    rng = np.random.RandomState(2)
    A, B = rng.randn(90, 12), rng.randn(15, 12)
    y = A @ rng.randn(12) + 0.1 * rng.randn(90)
    mu, sd = A.mean(0), A.std(0) + 1e-8
    As, Bs = (A - mu) / sd, (B - mu) / sd
    for al, got in zip((0.5, 20.0), lab._ridge_preds(A, y, B, (0.5, 20.0))):
        w = np.linalg.solve(As.T @ As + al * np.eye(12), As.T @ (y - y.mean()))
        assert np.allclose(got, Bs @ w + y.mean(), atol=1e-8)


def test_shuffle_keeps_marginals_and_breaks_alignment():
    Y = np.random.RandomState(0).randn(60, 5, 3).cumsum(0)
    S = lab._shuffled(Y, np.random.RandomState(1))
    assert np.allclose(np.sort(Y, 0), np.sort(S, 0)) and not np.allclose(Y, S)


def test_cka_identity_rotation_scale_and_independence():
    rng = np.random.RandomState(0)
    X = rng.randn(200, 20)
    Q, _ = np.linalg.qr(rng.randn(20, 20))
    assert abs(lab.linear_cka(X, X) - 1) < 1e-9 and abs(lab.linear_cka(X, 3.7 * X @ Q) - 1) < 1e-9
    assert lab.linear_cka(X, rng.randn(200, 20)) < 0.25


def test_jacobian_spectral_radius_matches_finite_differences():
    m = model()
    rec = lab.dream(m, 6, 2, 3)
    t, w = 3, 1
    x = torch.cat([rec["z"][t, w], torch.zeros(m.cfg.action_dim)])[None]
    h = rec["h"][t, w]
    gru = m.rssm.gru
    eps, J = 1e-3, np.zeros((len(h), len(h)))
    with torch.no_grad():
        for i in range(len(h)):
            e = torch.zeros(len(h)); e[i] = eps
            J[:, i] = ((gru(x, (rec["h"][t, w] + e)[None]) - gru(x, (rec["h"][t, w] - e)[None]))[0] / (2 * eps)).numpy()
    want = float(np.abs(np.linalg.eigvals(J)).max())
    got = lab.jacobian_spectrum(m, rec, (t,), worlds=2)
    assert got["ticks"] == [t] and abs(got["radius_mean"] - np.mean([want, _radius(m, rec, t, 0)])) < 0.02 * max(1, want)


def _radius(m, rec, t, w):
    x = torch.cat([rec["z"][t, w], torch.zeros(m.cfg.action_dim)])[None]
    J = torch.autograd.functional.jacobian(lambda hh: m.rssm.gru(x, hh[None])[0], rec["h"][t, w]).numpy()
    return float(np.abs(np.linalg.eigvals(J)).max())


def test_measure_and_capacity_on_a_real_dream():
    m = model()
    rec = lab.dream(m, 24, 4, 21)
    me = lab.measure(rec["frames"], seg())
    assert set(me) == set(lab.M_NAMES) and all(v.shape == (24, 4) for v in me.values())
    assert set(np.unique(me["eye"])) <= {0.0, 1.0} and 0 <= np.nanmin(me["brightness"]) and np.nanmax(me["brightness"]) <= 1
    assert np.nanmax(np.abs(me["pupil_x"])) <= 1.0
    cap = lab.capacity(rec, m.cfg.deter_dim, burn=4)
    assert 0 <= cap["active_kl"] <= cap["z_dim"] and len(cap["kl_unit"]) == cap["z_dim"] and cap["pr"]["hz"] >= 1
    assert cap["kl_total"] >= 0 and len(cap["h_std"]) == m.cfg.deter_dim


def test_requests_for_live_edits_are_validated():
    deter, stoch = 4, 3
    assert lab.interventions_from_request({}, None, deter, stoch) is None and lab.interventions_from_request({"clear": True}, None, deter, stoch) is None
    z = lab.interventions_from_request({"lesion": {"space": "z", "idx": [0, 2], "value": "zero"}}, None, deter, stoch)
    assert z.clamp_idx.tolist() == [4, 6] and float(z.clamp_val.abs().sum()) == 0
    for bad in ({"lesion": {"space": "z", "idx": [3], "value": "zero"}}, {"lesion": {"idx": [0], "value": "mean"}},
                {"lesion": {"group": "z_active"}}, {"steer": {"target": "pupil_x"}}, {"lesion": {"idx": [0], "value": "zero"}, "steer": {"target": "nope"}}):
        try:
            lab.interventions_from_request(bad, None, deter, stoch)
        except ValueError:
            continue
        raise AssertionError(bad)
    res = {"reference": {"mean": list(range(7))}, "probes": {"pupil_x": {"dir": [1, 0, 0, 0, 0, 0, 0], "proj_sd": 2.0}},
           "capacity": {"kl_unit": [0.5, 0.0, 0.02]}, "lab": {"group_size": 2, "seed": 1}, "lesion": {"groups": {"z_active": {}, "z_inactive": {}}}}
    iv = lab.interventions_from_request({"lesion": {"group": "z_active", "value": "mean"}, "steer": {"target": "pupil_x", "alpha": 1.5}}, res, deter, stoch)
    assert iv.clamp_idx.tolist() == [4, 6] and iv.clamp_val.tolist() == [4.0, 6.0] and abs(float(iv.add[0]) - 3.0) < 1e-6 and iv.stages == ("pred", "post")
    st = lab.interventions_from_request({"steer": {"target": "pupil_x", "alpha": -1}}, res, deter, stoch)
    assert st.stages == ("pred",) and abs(float(st.add[0]) + 2.0) < 1e-6
    assert lab._group_units("z_inactive", res, deter, stoch) == [5]


def test_analyse_quick_runs_every_section_and_caches():
    m = model()
    lc = lab.LabConfig.quick()
    res, imgs = lab.analyse(CKPT, seg(), lc)
    for k in ("capacity", "probes", "steer", "lesion", "patch", "dynamics", "reference", "nat_sd"):
        assert k in res, k
    assert set(res["probes"]) == set(lab.TARGETS)
    p = res["probes"]["shift_x"]
    assert p["r2_shuffle"] is None or p["r2_shuffle"] < 0.3                # the label-shuffled control must not find structure
    assert "r2_random_init" in p
    for k in ("z_active", "h_all", "z_all"):
        assert k in res["lesion"]["groups"] or k == "z_active"
    assert res["lesion"]["groups"]["h_all"]["pix"] > 0 and res["lesion"]["groups"]["z_all"]["pix"] > 0
    assert res["dynamics"]["noisy"]["marks"][0] == 0 and "lambda_per_tick" in res["dynamics"]["perturbation"]
    assert {"filmstrip", "encoder_am", "dream"} <= set(imgs) and all(i.ndim == 3 and i.shape[2] == 3 and i.dtype == np.uint8 for i in imgs.values())
    json.dumps(res)                                                         # NaN/inf were cleaned
    with tempfile.TemporaryDirectory() as tmp:
        d = lab.save_results("checkpoints/latest.pt", 123, res, imgs, root=tmp)
        assert (d / "results.json").exists() and (d / "filmstrip.png").exists()
        assert lab.load_results("checkpoints/latest.pt", 123, root=tmp)["probes"]["shift_x"]["alpha"] == p["alpha"]
        assert lab.load_results("checkpoints/latest.pt", 124, root=tmp) is None and lab.load_results("nope", root=tmp) is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
