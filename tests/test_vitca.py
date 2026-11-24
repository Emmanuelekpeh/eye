"""Run: python tests/test_vitca.py   (CPU; REAL frames from ../data/cache/videos for the training tests). Tests ivs/vitca.py (EyeViTCA) and its integration with
ivs/nca.py samplers / live mode, ivs/pnca.py (pool + plain trainers) and the checkpoint factory."""
import json
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from ivs import nca, pnca, vitca
from ivs.cachedata import CacheBank

CACHE = Path(__file__).resolve().parents[2] / "data" / "cache" / "videos"
SIZE = 96
TINY = dict(dim=32, blocks=1, loops=2, vheads=2, mlp_mult=2)         # 96 px, patch 8 -> 144 tokens
_bank = None


def bank(size=SIZE):
    global _bank
    if _bank is None:
        _bank = CacheBank(CACHE, split="train", device="cpu", image_size=size, max_videos=3, max_frames_per_video=200)
    return _bank


def real(n=4, seed=0):
    return nca.to_model(bank().sample(n, 1, generator=torch.Generator().manual_seed(seed), train=False)["images"][:, 0])


def tiny(**kw):
    torch.manual_seed(0)
    return vitca.EyeViTCA(vitca.ViTCAConfig(image_size=SIZE, **{**TINY, **kw}))


def perturbed(m, scale=0.05, seed=1):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(scale * torch.randn(p.shape, generator=g))
    return m


def test_parameter_counts_of_the_presets_are_in_the_small_range():
    n = {k: nca.count_params(vitca.EyeViTCA(vitca.ViTCAConfig(image_size=SIZE, **v))) for k, v in vitca.VITCA_PRESETS.items()}
    print("   param counts:", n)
    assert 1.5e6 < n["V2"] < n["V3"] < 5e6


def test_untrained_net_is_the_edm_skip_connection_and_state_has_the_right_shape():
    m = tiny().eval()
    x = real(3)
    sigma = torch.tensor([0.05, 0.5, 2.0])
    with torch.no_grad():
        d, h = m.denoise(x + sigma.view(-1, 1, 1, 1) * torch.randn_like(x), sigma, return_hidden=True)
        c_skip = nca.edm_coeffs(sigma, m.cfg.sigma_data)[0].view(-1, 1, 1, 1)
        xn = x + sigma.view(-1, 1, 1, 1) * torch.randn(x.shape, generator=torch.Generator().manual_seed(0))
        assert torch.allclose(m.denoise(xn, sigma), c_skip * xn, atol=1e-6)         # zero-initialised unpatchify
    assert d.shape == x.shape and h.shape == (3, 32, 12, 12) and m.cfg.hidden_shape() == (32, 12, 12)
    assert float(h.abs().max()) <= 1.0                                              # tanh-bounded state


def test_state_is_carried_bounded_and_none_means_zeros():
    m = perturbed(tiny()).eval()
    x = real(2)
    with torch.no_grad():
        a, h1 = m.denoise(x, 0.3, return_hidden=True)
        a0 = m.denoise(x, 0.3, hidden=torch.zeros_like(h1))
        b = m.denoise(x, 0.3, hidden=h1)
        _, h2 = m.denoise(x, 0.3, hidden=h1, return_hidden=True)
    assert torch.equal(a, a0) and not torch.equal(a, b) and not torch.equal(h1, h2)
    assert float(h1.abs().max()) <= 1 and float(h2.abs().max()) <= 1 and float(h1.abs().max()) > 0


def test_every_update_is_global_a_corner_pixel_reaches_the_opposite_corner_in_one_call():
    m = perturbed(tiny(), 0.1).eval()
    x = real(1)
    x2 = x.clone(); x2[:, :, :4, :4] += 1.0                                          # inside the top-left patch only
    with torch.no_grad():
        d = (m.denoise(x2, 0.5) - m.denoise(x, 0.5)).abs()
    far = float(d[:, :, -8:, -8:].mean())
    assert far > 1e-6, far                                                           # the NCA needed ~8 steps x dilation 4 (36 px) for this


def test_checkpointing_matches_and_all_params_get_gradients():
    m = perturbed(tiny(loops=3)).train()
    x = real(2)
    s = torch.tensor([0.2, 1.0])
    g = []
    for ck in (False, True):
        m.zero_grad()
        m.denoise(x, s, checkpoint_steps=ck).square().mean().backward()
        g.append(torch.cat([p.grad.flatten() for p in m.parameters()]))
    assert torch.allclose(g[0], g[1], atol=1e-6)
    m.zero_grad()
    m.denoise(x, s).square().mean().backward()
    dead = [n for n, p in m.named_parameters() if p.grad is None or float(p.grad.abs().sum()) == 0]
    assert not dead, dead


def test_checkpoint_factory_builds_both_families_and_old_nca_configs_still_load():
    v = tiny()
    assert isinstance(nca.net_from_config(asdict(v.cfg)), vitca.EyeViTCA)
    n = nca.EyeNCA(nca.NCAConfig(state=12, hidden=24, n_inner=2))
    old = asdict(n.cfg); old.pop("arch")                                               # a checkpoint saved before the marker existed
    assert isinstance(nca.net_from_config(old), nca.EyeNCA)
    w = nca.net_from_config(asdict(v.cfg)); w.load_state_dict(v.state_dict())
    x = real(2)
    with torch.no_grad():
        assert torch.equal(w.eval().denoise(x, 0.5), v.eval().denoise(x, 0.5))


def test_samplers_repair_and_living_mode_work_with_the_vit_and_carry_its_state():
    m = perturbed(tiny()).eval()
    g = lambda: torch.Generator().manual_seed(5)
    img, h = nca.sample(m, 2, steps=4, generator=g(), persist=True)
    assert img.shape == (2, 3, SIZE, SIZE) and h.shape == (2, 32, 12, 12) and torch.isfinite(img).all()
    assert torch.equal(img, nca.sample(m, 2, steps=4, generator=g(), persist=True)[0])
    plain = nca.sample(m, 2, steps=4, generator=g())
    assert not torch.equal(img, plain)                                                  # carrying the state changes the trajectory
    rep, h2 = nca.repair_blind(m, img, 0.5, steps=3, generator=g(), hidden=h, persist=True)
    assert rep.shape == img.shape and not torch.equal(h, h2)
    live, h3 = nca.live(m, img, h, 4, sigma=0.05, generator=g())
    assert live.shape == img.shape and torch.isfinite(live).all() and not torch.equal(h, h3)
    known = torch.ones(2, 1, SIZE, SIZE); known[:, :, 30:60, 30:60] = 0
    mk, _ = nca.repair_masked(m, img, known, steps=3, generator=g(), persist=True)
    assert torch.allclose(mk[known.bool().expand_as(mk)], img[known.bool().expand_as(img)], atol=1e-6)


def run_cfg(name, **kw):
    return pnca.replace(pnca.build_run(name), **{**dict(TINY, warmup=2, pool_size=32, levels=3, k_ticks=2, batch=8), **kw})


def test_pooled_persistent_training_lowers_the_loss_on_real_frames():
    rc = run_cfg("V_pers")
    with tempfile.TemporaryDirectory() as d:
        pnca.train_run(rc, bank(), torch.device("cpu"), 40, 0, Path(d) / "log.jsonl", log_every=10)
        rows = [json.loads(l) for l in open(Path(d) / "log.jsonl")]
    assert all(r["loss_avg"] == r["loss_avg"] for r in rows) and rows[-1]["loss_avg"] < rows[0]["loss_avg"], [r["loss_avg"] for r in rows]


def test_plain_control_trains_without_pool_and_lowers_the_loss():
    rc = run_cfg("V_plain", batch=16)
    assert rc.plain and not rc.persist
    with tempfile.TemporaryDirectory() as d:
        pnca.train_run(rc, bank(), torch.device("cpu"), 40, 0, Path(d) / "log.jsonl", log_every=10)
        rows = [json.loads(l) for l in open(Path(d) / "log.jsonl")]
    assert rows[-1]["loss_avg"] < rows[0]["loss_avg"] and "live_frac_avg" not in rows[-1], rows[-1]


def test_cli_end_to_end_for_both_vit_runs():
    pnca.PRESETS["T_vp"] = dict(arch="vitca", warmup=2, pool_size=32, levels=3, k_ticks=2, **TINY)
    pnca.PRESETS["T_vq"] = dict(arch="vitca", plain=True, persist=False, warmup=2, **TINY)
    try:
        with tempfile.TemporaryDirectory() as d:
            argv = ["--cache", str(CACHE), "--runs", "T_vp", "T_vq", "--steps", "4", "--batch", "4", "--image-size", str(SIZE), "--max-videos", "3",
                    "--max-frames-per-video", "60", "--eval-frames", "8", "--probe-n", "4", "--out", d, "--device", "cpu", "--log-every", "2", "--threads", "2"]
            pnca.main(argv)
            for name, persistent in (("T_vp", True), ("T_vq", False)):
                r = Path(d) / name
                m = json.loads((r / "metrics.json").read_text())
                assert m["run"]["persist"] is persistent and m["config"]["arch"] == "vitca" and m["steps_per_sec"] > 0
                assert set(m["persistence"]) == ({"0", "10", "30", "100"} if persistent else {"0"})
                assert all((r / f).stat().st_size > 500 for f in ("sheet_dreams.png", "sheet_repair.png", "model.pt"))
                assert (r / "sheet_living.png").exists() is persistent
                ck = torch.load(r / "model.pt", map_location="cpu", weights_only=False)
                net = nca.net_from_config(ck["config"]); net.load_state_dict(ck["model"])
                assert isinstance(net, vitca.EyeViTCA) and nca.count_params(net) == m["params"]
    finally:
        pnca.PRESETS.pop("T_vp", None); pnca.PRESETS.pop("T_vq", None)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name, flush=True)
    print("all vitca tests passed")
