"""Tests for ivs/latentdyn.py + scripts/r4_latent.py on REAL data (mini cache copied from data/cache_hr/h128f) and the real eye segmenter. CPU.
Run: python tests\\test_latentdyn.py (from ivs2)."""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
import r4_latent as R                  # noqa: E402
from ivs import eyemetric as E        # noqa: E402
from ivs.actions import InterventionSampler   # noqa: E402
from ivs.aeprobe import PlainAE       # noqa: E402
from ivs.latentdyn import (FrozenAE, LatentDynamics, dream, dyn_loss, load_ae, load_dyn, nll, one_step_mse, save_ae, save_dyn, unroll)   # noqa: E402
from ivs.learning import ssim_loss    # noqa: E402

ROOT = HERE.parent
SRC = ROOT / "data" / "cache_hr" / "h128f"
_TMP = tempfile.TemporaryDirectory()
TMP = Path(_TMP.name)
_STATE = {}


def mini_cache():
    """4 smallest video files + 6 OpenEDS sequences of the real whole-video 128 px cache (enough groups for a train and a val split)."""
    if "dir" in _STATE:
        return _STATE["dir"]
    metas = json.loads((SRC / "manifest.json").read_text())
    import heldout_clarity as H
    size = lambda m: (SRC / (H.safe(Path(m["name"]).stem) + ".npz")).stat().st_size
    vids = sorted([m for m in metas if m.get("source", "video") == "video"], key=size)[:8]
    ope = sorted([m for m in metas if m.get("source") == "openeds"], key=lambda m: m["name"])[:10]
    out = TMP / "mini"
    out.mkdir()
    for m in vids + ope:
        f = H.safe(Path(m["name"]).stem) + ".npz"
        shutil.copy2(SRC / f, out / f)
    (out / "manifest.json").write_text(json.dumps(vids + ope))
    _STATE["dir"] = out
    return out


def banks():
    if "banks" not in _STATE:
        _STATE["banks"] = R.banks(mini_cache(), size=128)
    return _STATE["banks"]


def tiny_ae(steps=0):
    """A real PlainAE S32 (spatial 32 at 128 px -> 8x8 grid); optionally trained a few steps on real frames so reconstructions are not noise."""
    torch.manual_seed(0)
    ae = PlainAE(128, 512, spatial=32)
    if steps:
        opt = torch.optim.Adam(ae.parameters(), 1e-3)
        bank, g = banks()["train"], torch.Generator().manual_seed(0)
        for _ in range(steps):
            x = bank.sample(8, 1, generator=g)["images"][:, 0]
            rec = ae(x)
            loss = (rec - x).abs().mean() + 0.1 * ssim_loss(rec, x)
            opt.zero_grad(); loss.backward(); opt.step()
    return FrozenAE(ae)


def test_dynamics_start_as_copy_with_small_sigma_and_have_the_stated_shapes():
    dyn = LatentDynamics(32, 8, 5, hid=24, act_ch=8, layers=2)
    z, a = torch.randn(3, 32, 8, 8), torch.randn(3, 5)
    h, mu, sd = dyn.step(dyn.initial(3), z, a)
    assert mu.shape == sd.shape == z.shape and [t.shape for t in h] == [(3, 24, 8, 8)] * 2
    assert torch.allclose(mu, z) and torch.allclose(sd, torch.full_like(sd, 0.12), atol=1e-4), "fresh head must copy the last latent with sigma 0.1 + floor"
    assert LatentDynamics(**dyn.config()).state_dict().keys() == dyn.state_dict().keys()


def test_unroll_feeds_real_latents_during_context_then_its_own_sample_and_indexes_frames():
    torch.manual_seed(1)
    dyn = LatentDynamics(32, 8, 5, hid=16, act_ch=4, layers=1)
    for p in dyn.head[-1].parameters():                   # make the head non-trivial so inputs matter
        torch.nn.init.normal_(p, std=0.05)
    e, a = torch.randn(2, 6, 32, 8, 8), torch.randn(2, 6, 5)
    mu, sd, s = unroll(dyn, e, a, ctx=3, free=3, temperature=0.0)
    assert mu.shape == (2, 5, 32, 8, 8)                   # frames 1..5
    h, prev = dyn.initial(2), None
    for t in range(1, 6):
        z_in = e[:, t - 1] if t <= 3 else prev
        h, m, _ = dyn.step(h, z_in, a[:, t])
        assert torch.allclose(m, mu[:, t - 1], atol=1e-6), f"frame {t}"
        prev = m
    mu2, _, _ = unroll(dyn, e, a, ctx=5, free=1, temperature=0.0)
    assert torch.allclose(mu2[:, :3], mu[:, :3], atol=1e-6), "teacher-forced frames do not depend on the free length"
    # sampling noise: temperature 1 differs, and the generator makes it reproducible
    g1, g2 = torch.Generator().manual_seed(4), torch.Generator().manual_seed(4)
    _, _, s1 = unroll(dyn, e, a, 3, 3, temperature=1.0, generator=g1)
    _, _, s2 = unroll(dyn, e, a, 3, 3, temperature=1.0, generator=g2)
    assert torch.equal(s1, s2) and not torch.equal(s1, mu)


def test_loss_trains_only_the_dynamics_and_is_finite_for_both_variants():
    ae = tiny_ae()
    bank, g = banks()["train"], torch.Generator().manual_seed(2)
    x = bank.sample(2, R.CTX + 6, generator=g)["images"]
    x, a = InterventionSampler()(x, generator=g)
    for reproject in (False, True):
        dyn = LatentDynamics(32, 8, 5, hid=16, act_ch=4, layers=2)
        loss, parts = dyn_loss(dyn, ae, x, a, R.CTX, 6, reproject=reproject)
        loss.backward()
        assert torch.isfinite(loss) and set(parts) == {"nll", "l1", "ssim"}
        gn = [p.grad for p in dyn.parameters()]
        assert all(t is not None and torch.isfinite(t).all() for t in gn) and sum(float(t.abs().sum()) for t in gn) > 0
        assert all(p.grad is None for p in ae.parameters()), "the autoencoder must stay frozen"


def test_training_lowers_the_nll_on_real_windows_and_beats_nothing_it_cannot():
    ae = tiny_ae(steps=60)
    bank = banks()["train"]
    torch.manual_seed(0)
    dyn = LatentDynamics(32, 8, 5, hid=32, act_ch=8, layers=2)
    opt = torch.optim.Adam(dyn.parameters(), 2e-3)
    g, isamp, first, last = torch.Generator().manual_seed(3), InterventionSampler(), [], []
    for i in range(40):
        x, a = isamp(bank.sample(4, R.CTX + 4, generator=g)["images"], generator=g)
        loss, parts = dyn_loss(dyn, ae, x, a, R.CTX, 4)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(dyn.parameters(), 1.0); opt.step()
        (first if i < 5 else last if i >= 35 else []).append(parts["nll"])
    assert np.mean(last) < np.mean(first), (first, last)
    win = banks()["val"].sample(6, 13, train=False)["images"]
    r = one_step_mse(dyn.eval(), ae, win, 8)
    assert set(r) == {"model", "copy_last", "ae_floor"} and all(np.isfinite(list(r.values()))) and r["ae_floor"] > 0


def test_dream_is_seeded_quantised_causal_and_the_projection_changes_it():
    ae, dyn = tiny_ae(steps=30), LatentDynamics(32, 8, 5, hid=16, act_ch=4, layers=1)
    for p in dyn.head[-1].parameters():
        torch.nn.init.normal_(p, std=0.05)
    ctx = banks()["val"].sample(3, 10, train=False)["images"]
    a = dream(dyn, ae, ctx, 6, temperature=0.5, reproject=False, seed=1)
    b = dream(dyn, ae, ctx, 6, temperature=0.5, reproject=False, seed=1)
    c = dream(dyn, ae, ctx, 6, temperature=0.5, reproject=False, seed=2)
    p = dream(dyn, ae, ctx, 6, temperature=0.5, reproject=True, seed=1)
    assert a.shape == (6, 3, 3, 128, 128) and torch.equal(a, b) and not torch.equal(a, c) and not torch.equal(a, p)
    assert torch.allclose(a * 255, (a * 255).round()), "frames must be 8-bit quantised like the page"
    assert torch.equal(a[:3], dream(dyn, ae, ctx, 3, temperature=0.5, reproject=False, seed=1)), "tick k must not depend on later ticks"
    t0 = dream(dyn, ae, ctx, 4, temperature=0.0)
    assert torch.equal(t0, dream(dyn, ae, ctx, 4, temperature=0.0, seed=9)), "temperature 0 is deterministic"
    act = torch.zeros(3, 4, 5); act[:, 1:, 0] = 1.0
    assert not torch.equal(t0, dream(dyn, ae, ctx, 4, temperature=0.0, actions=act)), "actions must reach the dynamics"


def test_touch_test_truth_moves_the_right_way_and_reports_every_kind_and_tick():
    ae, dyn = tiny_ae(steps=30), LatentDynamics(32, 8, 5, hid=16, act_ch=4, layers=1)
    ctx = banks()["val"].sample(4, R.EVAL_CTX, train=False)["images"]
    out = R.touch_test(dyn, ae, ctx, False, torch.device("cpu"))
    assert set(out) == set(R.TOUCH)
    for name, v in out.items():
        last, truth, fr = v["_frames"]
        assert truth.shape == fr.shape == (4, 16, 3, 128, 128)
        for r in R.TOUCH_READ:
            row = v[str(r)]
            assert row["copy_last"] > 0 and np.isfinite(row["ratio"]), (name, r)
        assert float((truth[:, 7] - truth[:, 3]).abs().mean()) < 1e-6, "the push stops after 3 ticks, so the held state must stay constant"
        assert float((truth[:, 3] - last).abs().mean()) > 1e-3, f"{name}: the pushed state must change the picture"
    # a model that ignores touch (fresh dynamics copy the last latent) cannot beat copy-last on the shift / gain / light truths by much
    assert all(out[k]["8"]["ratio"] > 0.5 for k in out)


def test_schedules_and_scoring_function_on_real_frames():
    assert R.free_at(0.0) == R.FREE_MIN and R.free_at(0.35) == R.FREE_MAX and R.free_at(1.0) == R.FREE_MAX
    assert R.lr_at(0.0, 1e-3) == 0.0 and abs(R.lr_at(0.02, 1e-3) - 1e-3 * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * 0.02)))) < 1e-9
    assert abs(R.lr_at(1.0, 1e-3) - 1e-4) < 1e-9
    seg = E.load_segmenter(str(HERE / "pilot_results" / "eyemetric" / "segmenter.pt"))
    win = banks()["val"].window_at(0, 0, 40)[:, None]                    # real 40-frame window, (40,1,3,128,128)
    st = R.score_dreams(win, seg)
    assert st["flags"].shape == (40, 1) and set(R.mean_stats(st)) == {"eye_frac", "eye_moving", "still_frac", "first100", "max_len", "n_long"}
    print("   real window scored by the segmenter:", R.mean_stats(st))
    assert R.to_seg_frames(win).shape == (40, 1, 3, 96, 96)


def test_save_load_roundtrip_and_prefetch_windows():
    ae = tiny_ae(steps=5)
    save_ae(TMP / "ae.pt", ae.ae)
    ae2 = load_ae(TMP / "ae.pt")
    x = banks()["val"].sample(2, 2, train=False)["images"][:, 0]
    assert torch.allclose(ae.encode(x), ae2.encode(x)) and torch.allclose(ae.decode(ae.encode(x)), ae2.decode(ae2.encode(x)))
    dyn = LatentDynamics(32, 8, 5, hid=16, act_ch=4, layers=2)
    save_dyn(TMP / "dyn.pt", dyn, meta=dict(x=1))
    d2, meta = load_dyn(TMP / "dyn.pt")
    z = torch.randn(1, 32, 8, 8)
    assert meta["x"] == 1 and torch.allclose(dyn.step(dyn.initial(1), z, torch.zeros(1, 5))[1], d2.step(d2.initial(1), z, torch.zeros(1, 5))[1])
    pf = R.Prefetch(banks()["train"], 3, 0, lambda: 6)
    try:
        x, a, free, ctx = pf.get()
        assert ctx == R.CTX and x.shape == (3, R.CTX + 6, 3, 128, 128) and a.shape == (3, R.CTX + 6, 5) and free == 6 and 0 <= float(x.min()) and float(x.max()) <= 1
    finally:
        pf.close()


def test_train_and_eval_stage_end_to_end_on_the_mini_cache():
    """The Kaggle path (train_dyn + eval_variant) with tiny sizes: writes dyn.pt, eval.json, both sheets, frames npz; every promised metric is present."""
    ae = tiny_ae(steps=30)
    out = TMP / "run"
    out.mkdir()
    bk = banks()
    dyn, tr = R.train_dyn(ae, bk["train"], out, "proj", 0.4, 2, 1e-3, torch.device("cpu"), 0, 16, 1, log_every=2, max_steps=4)
    seg = E.load_segmenter(str(HERE / "pilot_results" / "eyemetric" / "segmenter.pt"))
    res = R.eval_variant(dyn.eval(), ae, bk["val"], seg, out, "proj", 40, True, torch.device("cpu"), n_worlds=4, one_step_n=4)
    for f in ("dyn.pt", "eval.json", "dream_sheet_T0.5.png", "touch_sheet.png", "dream_frames_T0.5.npz"):
        assert (out / f).exists(), f
    assert set(res["dream"]) == {"0.0", "0.5", "1.0"} and set(res["touch"]) == set(R.TOUCH) and set(res["one_step"]) == {"model", "copy_last", "ae_floor"}
    assert set(res["dream"]["0.5"]["mean"]) == {"eye_frac", "eye_moving", "still_frac", "first100", "max_len", "n_long"} and 1 <= res["n_worlds"] <= 4
    assert tr["steps"] == 4 and json.loads((out / "eval.json").read_text())["variant"] == "proj"


def test_anchor_detach_variants_and_random_context_lengths():
    torch.manual_seed(3)
    dyn = LatentDynamics(32, 8, 5, hid=16, act_ch=4, layers=1, anchor=True)
    assert LatentDynamics(**dyn.config()).anchor and dyn.cells[0].gates.in_channels == 32 + 4 + 32 + 16
    for p in dyn.head[-1].parameters():
        torch.nn.init.normal_(p, std=0.05)
    e, a = torch.randn(2, 12, 32, 8, 8), torch.randn(2, 12, 5)
    mu, _, _ = unroll(dyn, e, a, ctx=4, free=8, temperature=0.0)
    h, prev = dyn.initial(2), None
    for t in range(1, 12):                                    # reference loop: anchor = real latent of frame min(t, ctx)-1
        z_in = e[:, t - 1] if t <= 4 else prev
        h, m, _ = dyn.step(h, z_in, a[:, t], anchor=e[:, min(t, 4) - 1])
        assert torch.allclose(m, mu[:, t - 1], atol=1e-6), t
        prev = m
    e2 = e.clone(); e2[:, 0] += 1.0                           # the anchor is the LAST context frame: changing frame 0 must not change the anchor but changes the context path
    mu_d, _, _ = unroll(dyn, e, a, ctx=4, free=8, temperature=0.0, detach_every=3)
    assert torch.allclose(mu_d, mu, atol=1e-6), "detaching changes gradients only, never the forward values"
    loss = nll(*unroll(dyn, e, a, 4, 8, detach_every=3)[:2], e[:, 1:])
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in dyn.parameters() if p.grad is not None)
    # dream with an anchor runs and the anchor matters
    ae = tiny_ae(steps=5)
    ctx = banks()["val"].sample(2, 8, train=False)["images"]
    d1 = dream(dyn, ae, ctx, 4, temperature=0.0)
    assert d1.shape == (4, 2, 3, 128, 128) and torch.isfinite(d1).all()
    assert set(R.VARIANTS) == {"free", "proj", "long", "anchor"} and R.VARIANTS["anchor"][1] and R.VARIANTS["long"][2] == 96
    assert R.free_at(0.5, 0.5, 96) == 96 and R.free_at(0.0, 0.5, 96) == R.FREE_MIN
    pf = R.Prefetch(banks()["train"], 2, 5, lambda: 5, ctx_choices=R.CTX_CHOICES)
    try:
        seen = {pf.get()[3] for _ in range(12)}
    finally:
        pf.close()
    assert seen <= set(R.CTX_CHOICES) and len(seen) > 1, seen


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print("PASS", name)
            except Exception as e:                       # noqa: BLE001
                import traceback
                failed += 1
                print("FAIL", name, repr(e)); traceback.print_exc()
    sys.exit(1 if failed else 0)
