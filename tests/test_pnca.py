"""Run: python tests/test_pnca.py   (REAL frames from ../data/cache/videos, tiny architecture, CPU). Tests ivs/pnca.py (PD-NCA pool trainer, backlog P0j)."""
import json
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F

from ivs import nca, ncapilot, pnca
from ivs.cachedata import CacheBank

CACHE = Path(__file__).resolve().parents[2] / "data" / "cache" / "videos"
SIZE = 64
TINY = dict(state=12, hidden=24, n_inner=2, warmup=2, batch=8, pool_size=48, levels=3, k_ticks=2)
CPU = torch.device("cpu")
_bank = None


def bank():
    global _bank
    if _bank is None:
        _bank = CacheBank(CACHE, split="train", device="cpu", image_size=SIZE, max_videos=3, max_frames_per_video=200)
    return _bank


def real(n=4, seed=0):
    return nca.to_model(bank().sample(n, 1, generator=torch.Generator().manual_seed(seed), train=False)["images"][:, 0])


def run(**kw):
    return pnca.replace(pnca.PRun(name="T"), **{**TINY, **kw})


def model_for(rc):
    torch.manual_seed(0)
    return nca.EyeNCA(pnca.model_config(rc, SIZE))


def test_loss_terms_on_real_frames():
    y = real(4, 1)
    assert float(pnca.edge_pyramid_loss(y, y).max()) == 0.0 and float(pnca.coarse_loss(y, y).max()) == 0.0
    blur = F.avg_pool2d(F.pad(y, (2, 2, 2, 2), mode="replicate"), 5, 1)
    noise = y + 0.1 * torch.randn_like(y)
    assert float(pnca.edge_pyramid_loss(blur, y).mean()) > 0 and float(pnca.edge_pyramid_loss(noise, y).mean()) > 0
    ripple = torch.zeros_like(y)
    ripple[..., ::2, ::2] = 0.3; ripple[..., 1::2, 1::2] = 0.3; ripple[..., ::2, 1::2] = -0.3; ripple[..., 1::2, ::2] = -0.3   # zero mean over every 2x2 block
    assert float(pnca.coarse_loss(y + ripple, y, cells=8).max()) < 1e-9 and float(pnca.edge_pyramid_loss(y + ripple, y).min()) > 0.01   # coarse is blind to fine detail, edge sees it
    assert float(pnca.coarse_loss(y + 0.3, y, cells=8).min()) > 0.05                                                                    # but sees a global shift
    assert float(pnca.overflow_loss(torch.full((2, 3, 4, 4), 0.99))) == 0.0 and abs(float(pnca.overflow_loss(torch.full((2, 3, 4, 4), 1.5))) - 0.5) < 1e-6
    rc, d = run(), real(4, 2)
    sig = torch.tensor([0.05, 0.3, 1.0, 3.0])
    w = nca.edm_weight(sig, 0.5)
    terms, mse = pnca.tick_terms(rc, d, y, w)
    assert torch.allclose(terms, w * mse) and torch.allclose(mse, ((d - y) ** 2).mean((1, 2, 3)))     # default = plain EDM
    t2, _ = pnca.tick_terms(run(l1l2=True, edge=0.5, coarse=1.0), d, y, w)
    assert bool((t2 > terms).all()), "every extra loss term adds a positive amount"


def test_pool_batch_bookkeeping_reseeds_the_worst_state():
    rc = run()
    m = model_for(rc)
    pool = pnca.Pool(rc.pool_size, m.cfg, CPU)
    gen = torch.Generator().manual_seed(1)
    pool.seed_all(bank(), rc, gen, CPU)
    assert abs(float(pool.eps.std()) - 1.0) < 0.1 and float(pool.h.abs().max()) == 0.0 and float(pool.x.abs().max()) == 0.0
    assert float(pool.y.min()) >= -1.0 and float(pool.y.max()) <= 1.0 and float(pool.y.std()) > 0.1, "targets are real frames in model range"
    pool.h += 1.0; pool.j += 2; pool.loss = torch.rand(rc.pool_size)
    pool.loss[7] = 10.0                                                                    # the worst state
    y_before = pool.y.clone()
    y, eps, x, h, j, slots = pnca.pool_batch(pool, rc, bank(), torch.Generator().manual_seed(3), CPU)
    assert y.shape == eps.shape == x.shape == (rc.batch, 3, SIZE, SIZE) and h.shape[0] == j.shape[0] == len(slots) == rc.batch
    assert len(set(slots.tolist())) == rc.batch, "write-back slots are unique"
    n_new = max(1, round(rc.batch * rc.fresh_frac)) + 1
    assert bool((j[-n_new:] == 0).all()) and float(h[-n_new:].abs().max()) == 0.0 and float(x[-n_new:].abs().max()) == 0.0, "fresh seeds: level 0, hidden zero"
    assert bool((j[:-n_new] == 2).all()) and torch.equal(h[:-n_new], torch.ones_like(h[:-n_new])), "kept states are the pooled ones"
    assert torch.equal(y[:-n_new], y_before[slots[:-n_new]]), "kept states keep their own target"
    assert not torch.equal(y[-n_new:], y_before[slots[-n_new:]]), "re-seeded slots get new real frames"
    assert 7 not in slots[:-n_new].tolist(), "the highest-loss candidate is never kept"
    assert abs(float(eps[-n_new:].std()) - 1.0) < 0.3


def _nudged(rc):
    m = nca.EyeNCA(pnca.model_config(rc, SIZE))
    opt = torch.optim.SGD(m.parameters(), lr=0.02)
    xx = real(8, 3)
    for _ in range(4):                                                                     # make the update layer non-zero
        opt.zero_grad(); l, _, _ = nca.edm_loss(m, xx, sigma=torch.full((8,), 0.5)); l.backward(); opt.step()
    return m.eval()


def test_denoise_inputs_are_the_true_forward_path_and_living_states_continue_from_their_own_render():
    rc = run(k_ticks=1, p_damage=0.0)
    m = _nudged(rc)
    sig = nca.sigma_schedule(rc.levels, rc.sigma_min, rc.sigma_max)
    y, eps = real(4, 4), torch.randn(4, 3, SIZE, SIZE)
    own = real(4, 6)                                                                       # the living states' own render (differs from y)
    h = torch.zeros(4, m.cfg.state - 3, SIZE, SIZE)
    j = torch.tensor([0, rc.levels - 1, rc.levels, rc.levels + 5])                        # first level, last denoise level, two living
    seen, orig = {}, m.denoise

    def spy(xin, sigma, **kw):
        seen["x_in"], seen["sigma"] = xin.clone(), sigma.clone()
        return orig(xin, sigma, **kw)
    m.denoise = spy
    gen = torch.Generator().manual_seed(2)
    with torch.no_grad():
        loss, terms, xn, hn, jn, st = pnca.run_ticks(m, rc, y, eps, own, h, j, sig, gen, CPU)
    for i, lvl in ((0, 0), (1, rc.levels - 1)):                                            # denoise phase: input = y + sigma_j * eps, exactly
        assert torch.allclose(seen["x_in"][i], y[i] + float(sig[lvl]) * eps[i], atol=1e-6) and abs(float(seen["sigma"][i]) - float(sig[lvl])) < 1e-6
    for i in (2, 3):                                                                       # living: own render + a little noise, sigma_live
        assert min(abs(float(seen["sigma"][i]) - s) for s in pnca.SIGMA_LIVE) < 1e-6 and 0 < float((seen["x_in"][i] - own[i]).abs().mean()) < 0.2
        assert float((seen["x_in"][i] - y[i]).abs().mean()) > float((seen["x_in"][i] - own[i]).abs().mean()), "living inputs are NOT y + noise"
    with torch.no_grad():
        d_last = orig(y[1:2] + float(sig[rc.levels - 1]) * eps[1:2], float(sig[rc.levels - 1]), hidden=h[1:2])
    assert torch.allclose(xn[1:2], d_last.clamp(-1, 1), atol=1e-5), "after the last level the state is the model's own estimate"
    assert jn.tolist() == [1, rc.levels, rc.levels, rc.levels + 5], "denoise states advance one level per tick; living states stay living (j frozen)"
    assert float(xn.abs().max()) <= 1.0 + 1e-6 and terms.shape == (4,) and torch.isfinite(terms).all() and float(hn.abs().max()) > 0 and not hn.requires_grad
    assert st["live_frac"] == 0.75 and st["mse_live"] is not None and st["mse_den"] is not None
    # the living weight is the plain live_weight, the denoise weight the EDM one
    w_live = rc.live_weight
    d, _ = orig(own[2:3] * 0 + seen["x_in"][2:3], float(seen["sigma"][2]), hidden=h[2:3], return_hidden=True)
    assert abs(float(terms[2]) - w_live * float(((d - y[2:3]) ** 2).mean())) < 1e-4
    # a full trajectory: from j = 0 it takes exactly `levels` ticks to go living
    rc3 = run(k_ticks=3, p_damage=0.0, levels=3)
    with torch.no_grad():
        *_, jn3, _ = pnca.run_ticks(m, rc3, y, eps, own * 0, h, torch.zeros(4, dtype=torch.long), nca.sigma_schedule(3, 0.01, 5.0), gen, CPU)
    assert jn3.tolist() == [3, 3, 3, 3]


def test_regeneration_damage_only_touches_living_states():
    rc = run(k_ticks=1, p_damage=1.0)
    m = model_for(rc).eval()
    sig = nca.sigma_schedule(rc.levels, rc.sigma_min, rc.sigma_max)
    y, eps = real(4, 5), torch.randn(4, 3, SIZE, SIZE)
    own = y.clone()                                                                        # perfect own renders: any difference at the input is damage
    h = torch.zeros(4, m.cfg.state - 3, SIZE, SIZE)
    j = torch.tensor([0, 1, rc.levels, rc.levels])
    seen, orig = {}, m.denoise

    def spy(xin, sigma, **kw):
        seen["x_in"], seen["sigma"] = xin.clone(), sigma.clone()
        return orig(xin, sigma, **kw)
    m.denoise = spy
    with torch.no_grad():
        pnca.run_ticks(m, rc, y, eps, own, h, j, sig, torch.Generator().manual_seed(0), CPU)
    for i, lvl in ((0, 0), (1, 1)):
        assert torch.allclose(seen["x_in"][i], y[i] + float(sig[lvl]) * eps[i], atol=1e-6), "denoise-phase states are never damaged"
    for i in (2, 3):
        moved = (seen["x_in"][i] - y[i]).abs().amax(0) > 0.5            # sigma_live <= .1 cannot move a pixel by .5: scatter did
        assert int(moved.sum()) > 20, "living states are scatter-damaged before their tick"


AXES = ["P0_core", "P1_gate", "P1_fourier", "P1_lattn", "P1_film", "P1_l1l2", "P1_edge", "P1_coarse", "P1_overflow", "P1_drift", "P1_nodeep", "P1_noregen"]


def test_every_loss_and_architecture_axis_trains_on_real_frames():
    for name in AXES:
        rc = pnca.replace(pnca.build_run(name), **TINY, fourier_hidden=16)
        with tempfile.TemporaryDirectory() as d:
            model, ema, cfg, hist, secs = pnca.train_run(rc, bank(), CPU, 6, 0, Path(d) / "log.jsonl", log_every=3)
            rows = [json.loads(l) for l in (Path(d) / "log.jsonl").read_text().splitlines()]
        assert [r["step"] for r in rows] == [1, 3, 6] and all(math.isfinite(r["loss_avg"]) and r["live_frac_avg"] >= 0 for r in rows), name
        assert not all(torch.equal(a, b) for a, b in zip(model.parameters(), ema.parameters())), f"{name}: EMA must lag"
        assert all(torch.isfinite(p).all() for p in model.parameters()), name
        assert cfg.gate == rc.gate and cfg.fourier == rc.fourier and cfg.lattn == rc.lattn and cfg.film == rc.film


def test_training_lowers_the_loss_on_real_frames():
    rc = run(sigma_max=1.0, levels=3, k_ticks=2, batch=8, pool_size=64, fresh_frac=0.75, lr=3e-3, p_damage=0.0)
    with tempfile.TemporaryDirectory() as d:
        *_, hist, _ = pnca.train_run(rc, bank(), CPU, 70, 0, Path(d) / "log.jsonl", log_every=10)
    first, last = hist[0]["loss_avg"], sum(r["loss_avg"] for r in hist[-2:]) / 2
    assert last < 0.85 * first, f"loss did not fall: {[r['loss_avg'] for r in hist]}"
    assert all(r["gclip"] <= 1.0 for r in hist)


def test_cli_end_to_end_writes_metrics_sheets_and_a_loadable_persistent_checkpoint():
    pnca.PRESETS["T_tiny"] = dict(state=12, hidden=24, n_inner=2, warmup=2, pool_size=32, levels=3, k_ticks=2, gate=True, fourier=True, fourier_hidden=16)
    try:
        with tempfile.TemporaryDirectory() as d:
            argv = ["--cache", str(CACHE), "--runs", "T_tiny", "--steps", "4", "--batch", "4", "--image-size", str(SIZE), "--max-videos", "3",
                    "--max-frames-per-video", "60", "--eval-frames", "8", "--probe-n", "4", "--out", d, "--device", "cpu", "--log-every", "2", "--threads", "2"]
            s = pnca.main(argv)
            r = Path(d) / "T_tiny"
            m = json.loads((r / "metrics.json").read_text())
            assert set(m["persistence"]) == {"0", "10", "30", "100"} and all(v["finite"] and v["std"] > 0 for v in m["persistence"].values())
            assert m["persistence"]["0"]["change_from_dream"] == 0.0 and m["run"]["persist"] is True and m["steps_per_sec"] > 0
            assert all((r / f).stat().st_size > 1000 for f in ("sheet_dreams.png", "sheet_living.png", "sheet_repair.png", "model.pt"))
            ck = torch.load(r / "model.pt", map_location="cpu", weights_only=False)
            net = nca.EyeNCA(nca.NCAConfig(**{**ck["config"], "dilations": tuple(ck["config"]["dilations"])}))
            net.load_state_dict(ck["model"])
            assert net.cfg.gate and net.cfg.fourier and ck["run"]["persist"] is True and nca.count_params(net) == m["params"]
            pnca.main(argv + ["--skip-existing"])                                          # resumes without retraining
            assert (Path(d) / "summary.json").exists() and s["T_tiny"]["params"] == m["params"]
    finally:
        pnca.PRESETS.pop("T_tiny", None)


def test_presets_build_and_the_param_costs_are_known():
    sizes = {}
    for name in pnca.PRESETS:
        rc = pnca.build_run(name)
        sizes[name] = nca.count_params(nca.net_from_config(pnca.asdict(pnca.model_config(rc, 96))))
    assert sizes["P0_core"] == 50179 and sizes["P2_losses"] == 50179, "loss axes and the core do not change the network"
    assert all(sizes[n] > 50179 for n in ("P1_gate", "P1_fourier", "P1_lattn", "P1_film", "P2_arch", "P3_all"))
    assert all(1.5e6 < sizes[n] < 5e6 for n in ("V_plain", "V_pers", "V2_pers")) and sizes["V_plain"] == sizes["V_pers"] > sizes["V2_pers"]
    print("   param counts:", sizes)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print("ok", fn.__name__)
    print("all pnca tests passed")
