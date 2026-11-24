"""Run: python tests/test_nca_ui.py   (the REAL trained E0_N50 checkpoint, CPU, few noise levels). Tests ivs/nca_ui.py against ivs/nca.py's own samplers."""
import base64
import io
import json
import sys
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from PIL import Image

from ivs import nca, nca_ui
from ivs.ui import scatter_pixels, SCATTER_RADIUS

CKPT = ROOT / "pilot_results" / "kaggle_nca1_15000steps" / "pilot" / "E0_N50" / "model.pt"
STEPS = 5
_world = None


def world():
    global _world
    if _world is None:
        _world = nca_ui.NCAWorld(SimpleNamespace(checkpoint=str(CKPT), seed=21, steps=STEPS, threads=2, cpu=True))
    return _world


def run_to_end(w):
    s = None
    while not w.sess.done:
        s = w.tick({})
    return s


def decode(data_url):
    return np.array(Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1]))))


def test_stepped_dream_equals_nca_sample_exactly():
    w = world()
    w.dream(21, STEPS)
    out = run_to_end(w)
    ref = nca.sample(w.model, 1, steps=STEPS, generator=torch.Generator().manual_seed(21))
    assert torch.equal(w.final, ref), float((w.final - ref).abs().max())
    assert out["i"] == out["n"] == STEPS - 0 and out["done"] and "final" in out
    assert decode(out["final"]).shape == (96, 96, 3)


def test_seed_reproducible_and_different_seeds_differ():
    w = world()
    w.dream(7, STEPS); run_to_end(w); a = w.final.clone()
    w.dream(7, STEPS); run_to_end(w); b = w.final.clone()
    w.dream(8, STEPS); run_to_end(w); c = w.final.clone()
    assert torch.equal(a, b) and not torch.equal(a, c)
    assert torch.isfinite(a).all() and 0 <= float(a.min()) and float(a.max()) <= 1


def test_trajectory_noise_levels_fall_and_estimate_appears_after_first_step():
    w = world()
    s0 = w.dream(21, STEPS)
    assert s0["est"] is None and s0["i"] == 0 and s0["sigma"] == float(nca.sigma_schedule(STEPS, 0.01, 5.0)[0])
    sig = []
    for _ in range(STEPS):
        s = w.tick({}); sig.append(s["sigma"]); assert s["est"] is not None
    assert all(a > b for a, b in zip(sig, sig[1:])), sig
    assert w.tick({})["i"] == STEPS                                   # ticking a finished dream is a no-op


def test_blind_and_masked_repair_match_nca_samplers():
    w = world()
    w.dream(21, STEPS); run_to_end(w)
    fin = w.final.clone()
    pts = [[0.5, 0.5, 1.0], [0.3, 0.6, 1.0]]
    req = dict(scatter=pts, sigma=0.6, steps=4, seed=21)
    # blind
    st = w.repair(req); assert st["kind"] == "repair (blind)"
    run_to_end(w)
    dmg = scatter_pixels(fin, [tuple(p) for p in pts], SCATTER_RADIUS, 22)
    assert not torch.equal(dmg, fin)                                  # the scatter really damaged something
    ref = nca.repair_blind(w.model, dmg, 0.6, steps=4, generator=torch.Generator().manual_seed(22))
    assert torch.equal(w.final, ref)
    # masked (the dream finished above was replaced by the repair; rebuild the same base image for the reference)
    w.final = fin.clone()
    st = w.repair({**req, "masked": True}); assert st["kind"] == "repair (masked)"
    run_to_end(w)
    hit = torch.nn.functional.max_pool2d((dmg != fin).any(1, keepdim=True).float(), 5, 1, 2)
    ref = nca.repair_masked(w.model, dmg, 1 - hit, steps=4, generator=torch.Generator().manual_seed(22))
    assert torch.equal(w.final, ref)
    outside = (1 - hit).bool().expand_as(fin)
    assert torch.allclose(w.final[outside], dmg[outside], atol=1e-6)  # known pixels are kept exactly


def test_evolve_continues_from_the_models_own_render_and_matches_repair_blind():
    w = world()
    w.dream(21, STEPS); run_to_end(w)
    fin = w.final.clone()
    st = w.evolve({"sigma": 0.8, "steps": 4})
    assert st["kind"] == "evolve #1" and w.seed == 21
    run_to_end(w)
    ref = nca.repair_blind(w.model, fin, 0.8, steps=4, generator=torch.Generator().manual_seed((21 * 1000003 + 1) & 0x7FFFFFFF))
    assert torch.equal(w.final, ref)
    first = w.final.clone()
    assert w.evolve({"sigma": 0.8, "steps": 4})["kind"] == "evolve #2"      # chains: the second evolve starts from the first one's output
    run_to_end(w)
    assert not torch.equal(w.final, first)
    w.dream(21, STEPS)
    assert w.evolve_n == 0                                                  # a fresh dream resets the chain
    # small sigma stays close to its own render, large sigma moves further away
    w.dream(21, STEPS); run_to_end(w); fin = w.final.clone()
    dists = []
    for sg in (0.1, 2.0):
        w.final = fin.clone(); w.evolve({"sigma": sg, "steps": 4}); run_to_end(w)
        dists.append(float((w.final - fin).abs().mean()))
    assert dists[0] < dists[1], dists


def test_evolve_before_any_finished_dream_is_a_clear_error():
    w = nca_ui.NCAWorld(SimpleNamespace(checkpoint=str(CKPT), seed=1, steps=STEPS, threads=2, cpu=True))
    try:
        w.evolve({})
    except ValueError as e:
        assert "finish" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_repair_before_any_finished_dream_is_a_clear_error():
    w = nca_ui.NCAWorld(SimpleNamespace(checkpoint=str(CKPT), seed=1, steps=STEPS, threads=2, cpu=True))
    try:
        w.repair({"scatter": []})
    except ValueError as e:
        assert "finish" in str(e)
    else:
        raise AssertionError("expected ValueError")


def persistent_world(tmp):
    from ivs import ncapilot, pnca
    rc = pnca.replace(pnca.PRun(name="T"), state=12, hidden=24, n_inner=2, gate=True, fourier=True, fourier_hidden=16)
    cfg = pnca.model_config(rc, 96)
    torch.manual_seed(0)
    m = nca.EyeNCA(cfg)
    with torch.no_grad():                    # the zero-initialised output layers keep a fresh net silent: perturb so the hidden state really evolves
        for p in m.parameters():
            p.add_(0.05 * torch.randn_like(p))
    ncapilot.save_model(Path(tmp) / "p.pt", m, m, cfg, rc)
    return nca_ui.NCAWorld(SimpleNamespace(checkpoint=str(Path(tmp) / "p.pt"), seed=3, steps=STEPS, threads=2, cpu=True))


def test_persistent_checkpoint_carries_hidden_through_dream_evolve_live_and_repair():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        w = persistent_world(tmp)
        assert w.persist and w.info()["persist"] is True
        w.dream(3, STEPS); run_to_end(w)
        ref, h_ref = nca.sample(w.model, 1, steps=STEPS, generator=torch.Generator().manual_seed(3), persist=True)
        assert torch.equal(w.final, ref) and torch.equal(w.hidden, h_ref) and float(w.hidden.abs().max()) > 0      # a dream leaves its hidden state behind
        fin, h0 = w.final.clone(), w.hidden.clone()
        # living mode == nca.live on the same generator, hidden continues
        st = w.live({"ticks": 5, "sigma": 0.05}); assert st["kind"] == "live #1" and st["n"] == 5
        run_to_end(w)
        img, h_live = nca.live(w.model, fin, h0, 5, sigma=0.05, generator=w._gen(3 * 999983 + 1))
        assert torch.equal(w.final, img) and torch.equal(w.hidden, h_live) and not torch.equal(w.hidden, h0)
        # evolve carries the hidden state: == repair_blind(hidden=...)
        fin2, h2 = w.final.clone(), w.hidden.clone()
        w.evolve({"sigma": 0.5, "steps": 3}); run_to_end(w)
        ref, h_ev = nca.repair_blind(w.model, fin2, 0.5, steps=3, generator=w._gen(3 * 1000003 + 1), hidden=h2, persist=True)
        assert torch.equal(w.final, ref) and torch.equal(w.hidden, h_ev)
        # a new dream starts from zeros again (not from the previous hidden)
        w.dream(3, STEPS); assert w.hidden is None; run_to_end(w)
        assert torch.equal(w.final, nca.sample(w.model, 1, steps=STEPS, generator=torch.Generator().manual_seed(3), persist=True)[0])
        # repair also continues from the carried hidden state
        fin3, h3 = w.final.clone(), w.hidden.clone()
        pts = [[0.5, 0.5, 1.0]]
        w.repair(dict(scatter=pts, sigma=0.6, steps=3, seed=3)); run_to_end(w)
        dmg = scatter_pixels(fin3, [tuple(p) for p in pts], SCATTER_RADIUS, 4)
        ref, _ = nca.repair_blind(w.model, dmg, 0.6, steps=3, generator=torch.Generator().manual_seed(4), hidden=h3, persist=True)
        assert torch.equal(w.final, ref)


def test_live_mode_needs_a_persistent_checkpoint_and_a_finished_dream():
    w = nca_ui.NCAWorld(SimpleNamespace(checkpoint=str(CKPT), seed=1, steps=STEPS, threads=2, cpu=True))
    assert not w.persist and w.info()["persist"] is False
    try:
        w.live({})
    except ValueError as e:
        assert "persistent" in str(e)
    else:
        raise AssertionError("expected ValueError")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        p = persistent_world(tmp)
        try:
            p.live({})
        except ValueError as e:
            assert "finish" in str(e)
        else:
            raise AssertionError("expected ValueError")


def test_http_endpoints_and_page():
    w = world()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), nca_ui.make_handler(w))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        page = urllib.request.urlopen(base + "/").read().decode()
        assert "EyeNCA dream viewer" in page and "/api/tick" in page
        info = json.loads(urllib.request.urlopen(base + "/api/info").read())
        assert info["size"] == 96 and info["params"] == 50179 and info["cond_dim"] == 0
        post = lambda p, b=None: json.loads(urllib.request.urlopen(urllib.request.Request(
            base + p, json.dumps(b or {}).encode(), {"Content-Type": "application/json"})).read())
        s = post("/api/dream", {"seed": 5, "steps": 3})
        assert s["n"] == 3 and s["seed"] == 5
        for _ in range(3):
            s = post("/api/tick")
        assert s["done"] and s["final"].startswith("data:image/png")
        r = post("/api/repair", {"scatter": [[0.5, 0.5, 1]], "steps": 3, "masked": False})
        assert r["kind"] == "repair (blind)"
        for _ in range(3):
            r = post("/api/tick")
        assert post("/api/evolve", {"sigma": 0.5, "steps": 3})["kind"] == "evolve #1"
        assert 'id="after"' in page and "/api/evolve" in page
        try:
            urllib.request.urlopen(base + "/nope")
        except urllib.error.HTTPError as e:
            assert e.code == 404
    finally:
        srv.shutdown()


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("PASS", name)
            except Exception as e:  # noqa: BLE001
                fails += 1; print("FAIL", name, type(e).__name__, str(e)[:400])
    sys.exit(1 if fails else 0)
