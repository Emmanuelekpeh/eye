"""Run: python tests/test_nca.py   (REAL frames from ../data/cache/videos, small subsets, CPU). Tests ivs/nca.py (EyeNCA, EDM, samplers)."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from ivs import nca
from ivs.cachedata import CacheBank

CACHE = Path(__file__).resolve().parents[2] / "data" / "cache" / "videos"
SIZE = 64
TINY = dict(image_size=SIZE, state=12, hidden=24, n_inner=3)
_bank = None


def real(n=4, seed=0):
    """n real video frames at SIZE px in [0,1] (B,3,H,W)."""
    global _bank
    if _bank is None:
        _bank = CacheBank(CACHE, split="all", device="cpu", image_size=SIZE, max_videos=3, max_frames_per_video=200)
    return _bank.sample(n, 1, generator=torch.Generator().manual_seed(seed), train=False)["images"][:, 0]


def tiny(**kw):
    torch.manual_seed(0)
    return nca.EyeNCA(nca.NCAConfig(**{**TINY, **kw}))


def nudge(m, steps=3, lr=0.02):
    """A few SGD steps on a real batch so the zero-initialised update layer becomes non-zero."""
    x = nca.to_model(real(4, 1))
    opt = torch.optim.SGD(m.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss, _, _ = nca.edm_loss(m, x, generator=torch.Generator().manual_seed(2))
        loss.backward(); opt.step()
    return m


def test_param_counts_hit_the_two_size_targets():
    assert 47_000 <= nca.count_params(nca.build("N50")) <= 53_000
    assert 128_000 <= nca.count_params(nca.build("N135")) <= 142_000


def test_edm_coefficients_match_the_closed_form():
    sig = torch.tensor([0.002, 0.05, 0.5, 1.0, 5.0, 40.0], dtype=torch.float64)
    sd = 0.5
    c_skip, c_out, c_in, c_noise = nca.edm_coeffs(sig, sd)
    w = nca.edm_weight(sig, sd)
    assert torch.allclose(c_skip, sd ** 2 / (sig ** 2 + sd ** 2))
    assert torch.allclose(c_out, sig * sd / (sig ** 2 + sd ** 2).sqrt())
    assert torch.allclose(c_in ** 2 * (sig ** 2 + sd ** 2), torch.ones_like(sig))        # unit-variance network input
    assert torch.allclose(w * c_out ** 2, torch.ones_like(sig))                          # unit effective loss weight on F's output
    assert torch.allclose(c_noise, sig.log() / 4)
    assert float(c_skip[0]) > 0.999 and float(c_skip[-1]) < 1e-3                         # sigma->0: identity, sigma->inf: network output


def test_sobel_and_laplacian_filters_are_per_channel_and_scaled():
    m = tiny(dilations=(1, 2))
    a = 0.3
    ramp = torch.arange(SIZE).float().view(1, 1, 1, SIZE).expand(1, 1, SIZE, SIZE) * a
    state = torch.zeros(1, m.cfg.state, SIZE, SIZE)
    state[:, 0] = ramp[:, 0]                                                              # a horizontal ramp in channel 0 only
    feats = m.perceive(state)
    assert len(feats) == 3 and feats[0].shape == state.shape and feats[1].shape == (1, 3 * m.cfg.state, SIZE, SIZE)
    for f, d in ((feats[1], 1), (feats[2], 2)):
        inner = (slice(None), slice(None), slice(8, SIZE - 8), slice(8, SIZE - 8))
        f = f[inner]
        assert torch.allclose(f[:, 0], torch.full_like(f[:, 0], a * d), atol=1e-5), "Sobel-x of a ramp = slope * dilation"
        assert float(f[:, 1].abs().max()) < 1e-5 and float(f[:, 2].abs().max()) < 1e-5     # Sobel-y and Laplacian of a ramp are 0
        assert float(f[:, 3:].abs().max()) < 1e-6                                         # other channels untouched (depthwise)


def test_untrained_net_leaves_the_state_alone():
    m = tiny()
    x = nca.to_model(real(2))
    state = m.run_state(x, torch.zeros(2))
    assert torch.equal(state[:, :3], x) and float(state[:, 3:].abs().max()) == 0.0        # zero-init update layer = identity


def test_denoise_shapes_range_and_sigma_to_zero_identity():
    m = tiny().eval()
    x = nca.to_model(real(3))
    with torch.no_grad():
        d = m.denoise(x, 0.7)
        assert d.shape == x.shape and torch.isfinite(d).all()
        near = m.denoise(x, 1e-4)
        assert float((near - x).abs().max()) < 1e-3, "D(x, sigma->0) must return x"
        per_sample = m.denoise(x, torch.tensor([0.1, 0.5, 2.0]))
        assert per_sample.shape == x.shape


def test_loss_falls_on_real_frames_and_grads_reach_every_parameter():
    m = tiny()
    x = nca.to_model(real(8, 3))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    sig = torch.full((8,), 0.6)
    losses = []
    for i in range(40):
        opt.zero_grad()
        loss, per, _ = nca.edm_loss(m, x, sigma=sig, generator=torch.Generator().manual_seed(i))
        loss.backward()
        if i == 5:                                         # after the zero-init layer has moved, every parameter must get a gradient
            for n, p in m.named_parameters():
                assert p.grad is not None and float(p.grad.abs().sum()) > 0, f"no gradient reaches {n}"
        opt.step(); losses.append(float(loss))
    assert math.isfinite(losses[-1]) and sum(losses[-5:]) / 5 < 0.8 * sum(losses[:5]) / 5, f"loss did not fall: {losses[:3]} -> {losses[-3:]}"


def test_checkpointing_gives_the_same_gradients():
    m = nudge(tiny())
    x = nca.to_model(real(2, 4))
    sig = torch.tensor([0.4, 1.5])
    grads = []
    for ck in (False, True):
        m.zero_grad()
        loss, _, _ = nca.edm_loss(m, x, sigma=sig, generator=torch.Generator().manual_seed(9), checkpoint_steps=ck)
        loss.backward()
        grads.append([p.grad.clone() for p in m.parameters()])
    for a, b in zip(*grads):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-4)


def test_sigma_schedule_is_descending_and_ends_at_zero():
    s = nca.sigma_schedule(12, 0.01, 5.0)
    assert len(s) == 13 and abs(float(s[0]) - 5.0) < 1e-5 and abs(float(s[-2]) - 0.01) < 1e-6 and float(s[-1]) == 0.0
    assert bool((s[:-1] > s[1:]).all())


def test_sampler_is_deterministic_given_the_seed():
    m = nudge(tiny()).eval()
    g = lambda s: torch.Generator().manual_seed(s)
    a, b, c = nca.sample(m, 3, steps=4, generator=g(1)), nca.sample(m, 3, steps=4, generator=g(1)), nca.sample(m, 3, steps=4, generator=g(2))
    assert a.shape == (3, 3, SIZE, SIZE) and 0 <= float(a.min()) and float(a.max()) <= 1
    assert torch.equal(a, b) and not torch.equal(a, c)


def test_blind_repair_at_sigma_zero_returns_the_input_and_changes_at_large_sigma():
    m = nudge(tiny()).eval()
    x = real(3, 5)
    near = nca.repair_blind(m, x, sigma_start=1e-3, steps=4, generator=torch.Generator().manual_seed(0))
    assert float((near - x).abs().max()) < 1e-2
    far = nca.repair_blind(m, x, sigma_start=2.0, steps=4, generator=torch.Generator().manual_seed(0))
    assert float((far - x).abs().mean()) > 10 * float((near - x).abs().mean())


def test_masked_repair_keeps_known_pixels_exactly_and_rewrites_the_hole():
    m = nudge(tiny()).eval()
    x = real(3, 6)
    mask = torch.ones(3, 1, SIZE, SIZE)
    mask[:, :, 20:44, 20:44] = 0
    for repeats in (1, 2):
        out = nca.repair_masked(m, x, mask, steps=4, repeats=repeats, generator=torch.Generator().manual_seed(0))
        assert float(((out - x).abs() * mask).max()) < 1e-6, "known pixels must be returned exactly"
        assert float(((out - x).abs() * (1 - mask)).max()) > 1e-3, "the hole must be filled by the model, not copied"


def test_cond_planes_are_used_after_training_and_shapes_validated():
    m = tiny(cond_dim=2)
    x = nca.to_model(real(2, 7))
    c1, c2 = torch.zeros(2, 2, SIZE, SIZE), torch.ones(2, 2, SIZE, SIZE)
    xin = x
    opt = torch.optim.SGD(m.parameters(), lr=0.02)
    for i in range(3):
        opt.zero_grad()
        d = m.denoise(xin + 0.5 * torch.randn_like(x), 0.5, cond=c2)
        ((d - x) ** 2).mean().backward(); opt.step()
    with torch.no_grad():
        assert float((m.denoise(x, 0.5, cond=c1) - m.denoise(x, 0.5, cond=c2)).abs().max()) > 0, "cond planes must change the output"
    try:
        m.denoise(x, 0.5)                                   # a cond model called without cond must fail loudly, not silently
        raise AssertionError("missing cond must raise")
    except RuntimeError:
        pass


def test_fire_rate_is_asynchronous_but_default_is_deterministic():
    x = nca.to_model(real(2, 8))
    m = nudge(tiny()).eval()
    with torch.no_grad():
        assert torch.equal(m.denoise(x, 0.5), m.denoise(x, 0.5))
    m2 = nudge(tiny(fire_rate=0.5)).eval()
    with torch.no_grad():
        torch.manual_seed(0); a = m2.denoise(x, 0.5)
        torch.manual_seed(1); b = m2.denoise(x, 0.5)
    assert not torch.equal(a, b)


def test_ema_update_endpoints():
    m = nudge(tiny())
    e = nca.make_ema(m)
    assert all(not p.requires_grad for p in e.parameters())
    assert all(torch.equal(a, b) for a, b in zip(e.parameters(), m.parameters()))       # the EMA starts as a copy
    for p in m.parameters():
        p.data.add_(1.0)                                                                 # the live model moves away
    before = [p.clone() for p in e.parameters()]
    nca.ema_update(e, m, 1.0)
    assert all(torch.equal(a, b) for a, b in zip(before, e.parameters())), "decay 1 keeps the EMA weights"
    nca.ema_update(e, m, 0.5)
    assert all(torch.allclose(b + 0.5, p, atol=1e-6) for b, p in zip(before, e.parameters()))   # halfway to the live weights
    nca.ema_update(e, m, 0.0)
    assert all(torch.equal(a, b) for a, b in zip(m.parameters(), e.parameters()))


def _tiny_preset():
    from ivs import ncapilot
    ncapilot.PRESETS["T_tiny"] = dict(state=12, hidden=24, n_inner=2, warmup=2)
    return ncapilot


def test_ncapilot_trains_saves_continues_and_evaluates_on_real_frames():
    import tempfile
    import json
    ncapilot = _tiny_preset()
    try:
        bank = CacheBank(CACHE, split="train", device="cpu", image_size=SIZE, max_videos=3, max_frames_per_video=120)
        val = CacheBank(CACHE, split="val", device="cpu", image_size=SIZE, max_videos=3, max_frames_per_video=120, hflip=False)
        rc = ncapilot.replace(ncapilot.build_run("T_tiny"), batch=4)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            model, ema, cfg, hist, secs = ncapilot.train_run(rc, bank, torch.device("cpu"), 6, 0, d / "log.jsonl", log_every=2)
            rows = [json.loads(l) for l in (d / "log.jsonl").read_text().splitlines()]
            assert [r["step"] for r in rows] == [1, 2, 4, 6] and all(r["T"] == 1 and "loss_avg" in r for r in rows)
            assert not all(torch.equal(a, b) for a, b in zip(model.parameters(), ema.parameters())), "the EMA must lag the live weights"
            ncapilot.save_model(d / "model.pt", model, ema, cfg, rc)
            ck = torch.load(d / "model.pt", map_location="cpu", weights_only=False)
            assert set(ck) >= {"model", "raw", "config", "run"} and all(v.device.type == "cpu" for v in ck["model"].values())
            m2, e2 = nca.EyeNCA(cfg), nca.make_ema(nca.EyeNCA(cfg))
            ncapilot.load_init(m2, e2, cfg, d / "model.pt")
            assert all(torch.equal(a, b) for a, b in zip(m2.parameters(), model.parameters()))
            assert all(torch.equal(a, b) for a, b in zip(e2.parameters(), ema.parameters()))
            bad = nca.NCAConfig(**{**TINY, "state": 16})
            try:
                ncapilot.load_init(nca.EyeNCA(bad), nca.make_ema(nca.EyeNCA(bad)), bad, d / "model.pt")
                raise AssertionError("an architecture mismatch must be refused")
            except ValueError as e:
                assert "state" in str(e)
            ev = ncapilot.eval_denoise(ema, val, torch.device("cpu"), n=8, chunk=4)
            assert set(ev) == {str(s) for s in ncapilot.SIGMAS} and all(math.isfinite(v["mse"]) and v["ratio_to_noisy"] > 0 for v in ev.values())
            assert ev == ncapilot.eval_denoise(ema, val, torch.device("cpu"), n=8, chunk=4), "evaluation must be deterministic"
            info = ncapilot.sheets(ema, val, d, torch.device("cpu"), n_dream=4, steps=3)
            assert all((d / f).stat().st_size > 1000 for f in ("sheet_dreams.png", "sheet_denoise.png", "sheet_repair.png")) and info["dream_denoise_calls"] == 5
    finally:
        ncapilot.PRESETS.pop("T_tiny", None)


def test_ncapilot_cli_end_to_end_and_lr_schedule():
    import tempfile
    import json
    ncapilot = _tiny_preset()
    try:
        rc = ncapilot.build_run("E0_N50")
        up = [ncapilot.lr_at(rc, s, 10 ** 6) for s in (0, 50, 100, 199)]
        assert up[0] < up[1] < up[2] < up[3] <= rc.lr and abs(up[0] - rc.lr / rc.warmup) < 1e-9 * rc.lr, "linear warm-up"
        down = [ncapilot.lr_at(rc, s, 1000) for s in (200, 500, 999)]
        assert down[0] > down[1] > down[2] >= rc.lr * rc.lr_floor * 0.999, "cosine decay ends at lr * lr_floor"
        with tempfile.TemporaryDirectory() as d:
            s = ncapilot.main(["--cache", str(CACHE), "--runs", "T_tiny", "--steps", "3", "--batch", "2", "--image-size", str(SIZE), "--max-videos", "3",
                               "--max-frames-per-video", "60", "--eval-frames", "8", "--out", d, "--device", "cpu", "--log-every", "1"])
            m = json.loads((Path(d) / "T_tiny" / "metrics.json").read_text())
            assert m["params"] == nca.count_params(nca.EyeNCA(nca.NCAConfig(**{**TINY, "n_inner": 2}))) and (Path(d) / "T_tiny" / "model.pt").exists()
            assert (Path(d) / "summary.json").exists() and (Path(d) / "data_info.json").exists() and s["T_tiny"]["steps_per_sec"] > 0
            ncapilot.main(["--cache", str(CACHE), "--runs", "T_tiny", "--steps", "3", "--out", d, "--skip-existing", "--max-videos", "3"])   # resumes without retraining
    finally:
        ncapilot.PRESETS.pop("T_tiny", None)


# ----------------------------------------------------------------------------------------------- PD-NCA (backlog P0j)
FLAGS = {"gate": dict(gate=True), "fourier": dict(fourier=True, fourier_hidden=16), "lattn": dict(lattn=True), "film": dict(film=True),
         "all": dict(gate=True, fourier=True, fourier_hidden=16, lattn=True, film=True)}


def test_pdnca_flags_off_is_the_e0_network():
    m = tiny()
    assert {n for n, _ in m.named_parameters()} == {"local.weight", "local.bias", "glob.weight", "pos.weight", "update.weight", "update.bias",
                                                    "readout.weight", "readout.bias"}
    assert nca.count_params(nca.build("N50")) == 50179 and nca.count_params(nca.build("N135")) == 135451
    m = nudge(m).eval()
    x = nca.to_model(real(2, 9))
    with torch.no_grad():
        zeros = x.new_zeros(2, m.cfg.state - 3, SIZE, SIZE)
        assert torch.equal(m.denoise(x, 0.6), m.denoise(x, 0.6, hidden=zeros))                # hidden=None == zeros
        d, h = m.denoise(x, 0.6, return_hidden=True)
        assert torch.equal(d, m.denoise(x, 0.6)) and h.shape == zeros.shape
    old = {k: v for k, v in nca.NCAConfig().__dict__.items()}                                  # an E0 checkpoint config (no new keys) still loads
    e0 = {k: old[k] for k in ("image_size", "channels", "state", "hidden", "n_inner", "dilations", "cond_dim", "sigma_data", "fire_rate")}
    assert nca.EyeNCA(nca.NCAConfig(**e0)).cfg.gate is False


def test_hidden_state_is_carried_across_calls_and_through_the_sampler():
    m = nudge(tiny(), steps=6).eval()
    x = nca.to_model(real(2, 10))
    with torch.no_grad():
        d0, h0 = m.denoise(x, 0.6, return_hidden=True)
        assert float(h0.abs().max()) > 0, "the nudged net must write into its hidden channels"
        d1 = m.denoise(x, 0.6, hidden=h0)
        assert not torch.equal(d0, d1), "a call that starts from a carried hidden state must differ from a rebuilt one"
    g = lambda: torch.Generator().manual_seed(3)
    plain = nca.sample(m, 2, steps=4, generator=g())
    img, hid = nca.sample(m, 2, steps=4, generator=g(), persist=True)
    img2, hid2 = nca.sample(m, 2, steps=4, generator=g(), persist=True)
    assert hid.shape == (2, m.cfg.state - 3, SIZE, SIZE) and torch.equal(img, img2) and torch.equal(hid, hid2)
    assert not torch.equal(img, plain), "carrying the hidden state changes the trajectory"
    rb, _ = nca.repair_blind(m, real(2, 11), 0.6, steps=3, generator=g(), persist=True)
    assert rb.shape == (2, 3, SIZE, SIZE) and 0 <= float(rb.min()) and float(rb.max()) <= 1


def test_every_new_module_starts_silent_trains_and_gets_gradients():
    for name, kw in FLAGS.items():
        m = tiny(**kw)
        x = nca.to_model(real(8, 3))
        state = m.run_state(x[:2], torch.zeros(2))
        assert torch.equal(state[:, :3], x[:2]) and float(state[:, 3:].abs().max()) == 0.0, f"{name}: untrained net must leave the state alone"
        opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
        sig = torch.full((8,), 0.6)
        losses = []
        for i in range(40):
            opt.zero_grad()
            loss, per, _ = nca.edm_loss(m, x, sigma=sig, generator=torch.Generator().manual_seed(i))
            loss.backward()
            if i == 12:
                for n, p in m.named_parameters():
                    assert p.grad is not None and float(p.grad.abs().sum()) > 0, f"{name}: no gradient reaches {n}"
            opt.step(); losses.append(float(loss))
        assert math.isfinite(losses[-1]) and sum(losses[-5:]) / 5 < 0.85 * sum(losses[:5]) / 5, f"{name}: loss did not fall {losses[:3]} -> {losses[-3:]}"


def test_fourier_pathway_is_global_and_attention_is_local_and_convex():
    m = tiny(fourier=True, fourier_hidden=16, lattn=True)
    torch.manual_seed(1)
    for p in (m.fw2.weight, m.fw2.bias, m.rel):
        p.data.normal_(0, 0.5)
    s = torch.randn(1, m.cfg.state, SIZE, SIZE)
    s2 = s.clone(); s2[:, :, 2, 2] += 3.0
    with torch.no_grad():
        far = float((m._fourier(s2) - m._fourier(s))[:, :, 50:, 50:].abs().max())
        assert far > 1e-4, "a perturbation in one corner must reach the opposite corner through the frequency pathway"
        a, a2 = m._attn(s), m._attn(s2)
        assert float((a2 - a)[:, :, 20:, 20:].abs().max()) == 0.0, "attention reach is local: 4 px (max dilation) + 1, nothing beyond"
        assert float((a2 - a)[:, :, 2:4, 2:4].abs().max()) > 0, "and it does see its neighbours"
        v = m.wv(s).view(1, m.cfg.heads, m.cfg.attn_dv, SIZE, SIZE)
        o = a.view(1, m.cfg.heads, m.cfg.attn_dv, SIZE, SIZE)
        assert bool((o <= v.amax((3, 4), keepdim=True) + 1e-5).all()) and bool((o >= v.amin((3, 4), keepdim=True) - 1e-5).all()), "convex combination of values"


def test_gate_keeps_hidden_channels_bounded():
    m = nudge(tiny(gate=True), steps=8).eval()
    x = nca.to_model(real(3, 12))
    with torch.no_grad():
        h = None
        for s in (2.0, 0.5, 0.1, 0.02):
            _, h = m.denoise(x + s * torch.randn_like(x), s, hidden=h, return_hidden=True)
            assert float(h.abs().max()) <= 1.0 + 1e-6, "tanh-candidate convex update cannot leave [-1, 1]"
    assert float(h.abs().max()) > 0


def test_checkpointing_matches_for_every_new_module_and_live_ticks_work():
    m = nudge(tiny(**FLAGS["all"]), steps=6)
    x = nca.to_model(real(2, 4))
    sig = torch.tensor([0.4, 1.5])
    grads = []
    for ck in (False, True):
        m.zero_grad()
        loss, _, _ = nca.edm_loss(m, x, sigma=sig, generator=torch.Generator().manual_seed(9), checkpoint_steps=ck)
        loss.backward()
        grads.append([p.grad.clone() for p in m.parameters()])
    for a, b in zip(*grads):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-4)
    m.eval()
    x01 = real(2, 13)
    g = lambda: torch.Generator().manual_seed(5)
    h0 = torch.zeros(2, m.cfg.state - 3, SIZE, SIZE)
    img, h, snaps = nca.live(m, x01, h0, 6, sigma=0.05, generator=g(), every=2)
    img2, h2 = nca.live(m, x01, h0, 6, sigma=0.05, generator=g())
    assert [t for t, _ in snaps] == [0, 2, 4, 6] and torch.equal(snaps[-1][1], img) and torch.equal(img, img2) and torch.equal(h, h2)
    assert torch.equal(snaps[0][1], nca.to_unit(nca.to_model(x01))) and 0 <= float(img.min()) and float(img.max()) <= 1
    assert float((img - x01).abs().mean()) < 0.2, "sigma_live is small: a few ticks stay near the input"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print("ok", fn.__name__)
    print("all nca tests passed")
