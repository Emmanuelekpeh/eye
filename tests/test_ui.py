"""Run: python tests/test_ui.py   (or pytest). Uses the real checkpoints/latest.pt; no dataset is read."""
from pathlib import Path
import base64, importlib.util, io, sys, tempfile
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from PIL import Image
from ivs import ui
from ivs.actions import InterventionSampler

CKPT = ROOT / "checkpoints" / "latest.pt"


def _world(tmp, **kw):
    argv = ["--checkpoint", str(CKPT), "--world", str(Path(tmp) / "w.pt"), "--fresh", "--seed", "7",
            "--learn-every", "1", "--probe-every", "2"]
    for k, v in kw.items():
        argv += [f"--{k.replace('_', '-')}", str(v)]
    return ui.World(ui.parse_args(argv))


def _decode(data_url):
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1]))))


def test_synthetic_path_is_gone():
    assert importlib.util.find_spec("ivs.synth") is None
    assert not hasattr(ui, "FrameSource") and not hasattr(ui, "render_pupil")


def test_invert_state_undoes_apply_state_away_from_borders():
    s = InterventionSampler(p=1.0)
    yy, xx = torch.meshgrid(torch.linspace(0, 1, 32), torch.linspace(0, 1, 32), indexing="ij")
    img = (0.2 + 0.15 * torch.sin(3 * xx) * torch.cos(2.5 * yy))[None, None, None]   # smooth, never saturates
    st = torch.tensor([[[0.1, -0.08, 0.1, 0.3, -0.2]]])
    back = s.invert_state(s.apply_state(img, st), st)
    err = (back - img)[..., 8:-8, 8:-8].abs().max()
    assert err < 0.03, float(err)


def test_scatter_moves_pixels_locally_and_deterministically():
    img = torch.rand(1, 1, 32, 32)
    pts = [(0.5, 0.5, 1.0)]
    a, b = ui.scatter_pixels(img, pts, 0.2, 5), ui.scatter_pixels(img, pts, 0.2, 5)
    assert torch.equal(a, b) and not torch.equal(a, ui.scatter_pixels(img, pts, 0.2, 6))
    assert torch.equal(a.flatten().sort().values, img.flatten().sort().values)          # moved, never invented
    ys, xs = torch.meshgrid((torch.arange(32) + .5) / 32, (torch.arange(32) + .5) / 32, indexing="ij")
    outside = ((xs - .5) ** 2 + (ys - .5) ** 2) > 0.2 ** 2
    assert torch.equal(a[0, 0][outside], img[0, 0][outside]) and not torch.equal(a, img)
    assert torch.equal(ui.scatter_pixels(img, [], 0.2, 5), img)
    assert torch.equal(ui.scatter_pixels(img, [(0.5, 0.5, 0.0)], 0.2, 5), img)


def test_watch_and_dream_ticks_on_real_checkpoint():
    with tempfile.TemporaryDirectory() as tmp:
        w = _world(tmp)
        assert w.trained
        r = w.tick({"target": [0, 0, 0, 0, 0]})
        assert r["metric"] == "prediction error" and abs(r["mse"]["pred"] - r["mse"]["pred_frozen"]) < 1e-9  # birth: live == frozen
        for k in ("stim", "pred", "recon", "frozen", "surprise"):
            assert _decode(r["img"][k]).shape == (64, 64)
        before = w.canvas.clone()
        r = w.tick({"target": [0.1, 0, 0.2, 0.3, 0], "scatter": [[0.5, 0.5, 1.0]]})
        assert not torch.equal(before, w.canvas) and 0 <= w.canvas.min() and w.canvas.max() <= 1
        d = w.tick({"target": [0.1, 0, 0.2, 0.3, 0], "dream": True, "scatter": [[0.5, 0.5, 1.0]]})
        assert d["dream"] and d["metric"] == "dissonance" and np.isfinite(d["kl"])


def test_intervention_off_ignores_all_touch():
    touch = {"target": [0.15, -0.1, 0.3, 0.5, 0.4], "scatter": [[0.5, 0.5, 1.0], [0.3, 0.6, 1.0]]}
    for dream in (False, True):
        with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
            a, b = _world(t1), _world(t2)                      # same seed, same weights
            for _ in range(6):
                ra = a.tick({**touch, "intervene": False, "dream": dream})
                rb = b.tick({"target": [0, 0, 0, 0, 0], "dream": dream})
                assert ra["intervene"] is False and ra["state"] == [0.0] * 5
                assert ra["img"] == rb["img"] and ra["kl"] == rb["kl"]          # bit-identical to an untouched world
            assert torch.equal(a.canvas, b.canvas)
            ra = a.tick({**touch, "dream": dream})                              # switching it back on works
            assert ra["intervene"] is True and any(abs(x) > 0 for x in ra["state"])


def test_learning_is_off_by_default_and_trains_rssm_only():
    with tempfile.TemporaryDirectory() as tmp:
        w = _world(tmp)
        enc0 = {k: v.clone() for k, v in w.model.encoder.state_dict().items()}
        dec0 = {k: v.clone() for k, v in w.model.decoder.state_dict().items()}
        rssm0 = {k: v.clone() for k, v in w.model.rssm.state_dict().items()}
        for i in range(30):                                   # learning off: nothing may move
            w.tick({"target": [0.1 * np.sin(i / 3), 0, 0, 0, 0], "scatter": [[0.4, 0.4, 0.8]]})
        assert w.learn_steps == 0
        assert all(torch.equal(v, rssm0[k]) for k, v in w.model.rssm.state_dict().items())
        for i in range(30):                                   # watching then dreaming, learning on
            w.tick({"target": [0.1 * np.sin(i / 3), 0, 0.1, 0, 0], "learn": True, "dream": i >= 15,
                    "scatter": [[0.4, 0.4, 0.8]] if i % 3 == 0 else []})
        assert w.learn_steps > 0 and w.last_loss is not None and np.isfinite(w.last_loss)
        assert all(torch.equal(v, enc0[k]) for k, v in w.model.encoder.state_dict().items())
        assert all(torch.equal(v, dec0[k]) for k, v in w.model.decoder.state_dict().items())
        assert any(not torch.equal(v, rssm0[k]) for k, v in w.model.rssm.state_dict().items())
        fz = w.frozen.rssm.state_dict()
        assert all(torch.equal(v, rssm0[k]) for k, v in fz.items())                         # frozen copy never moves
        assert np.isfinite(w.probe_ema) and w.probe_tol >= 1e-3


def test_reseed_snapshot_restore():
    with tempfile.TemporaryDirectory() as tmp:
        w = _world(tmp)
        c0 = w.canvas.clone()
        assert w.reseed({"seed": 8})["seed"] == 8 and not torch.equal(c0, w.canvas)
        assert w.reseed({"seed": 7})["ok"] and torch.equal(c0, w.canvas)                    # same seed -> same noise eye
        w.snapshot("a")
        sid = w.snapshots[-1]["id"]
        for i in range(5):
            w.tick({"target": [0.1, 0, 0, 0, 0], "scatter": [[0.5, 0.5, 1.0]]})
        w.restore(sid)
        assert torch.equal(w.canvas, c0)


G4 = ROOT / "pilot_results" / "kaggle_gaze1_8000" / "pilot" / "G4_cmd_geom_prior" / "model.pt"      # RGB, 96 px, 8-dim action, geometry head


def _g4_world(tmp, seed=7, **kw):
    argv = ["--checkpoint", str(G4), "--world", str(Path(tmp) / "w.pt"), "--fresh", "--seed", str(seed),
            "--learn-every", "1", "--probe-every", "2"]
    for k, v in kw.items():
        argv += [f"--{k.replace('_', '-')}", str(v)]
    return ui.World(ui.parse_args(argv))


def test_scatter_moves_whole_rgb_pixels():
    img = torch.rand(1, 3, 32, 32)
    a = ui.scatter_pixels(img, [(0.5, 0.5, 1.0)], 0.25, 5)
    assert torch.equal(a, ui.scatter_pixels(img, [(0.5, 0.5, 1.0)], 0.25, 5)) and not torch.equal(a, img)
    px = lambda t: sorted(map(tuple, t[0].permute(1, 2, 0).reshape(-1, 3).tolist()))      # noqa: E731
    assert px(a) == px(img), "pixels move with all their channels together and none is invented"
    ys, xs = torch.meshgrid((torch.arange(32) + .5) / 32, (torch.arange(32) + .5) / 32, indexing="ij")
    outside = ((xs - .5) ** 2 + (ys - .5) ** 2) > 0.25 ** 2
    assert torch.equal(a[0][:, outside], img[0][:, outside])


def test_rgb_gaze_model_renders_colour_and_reports_the_pupil():
    with tempfile.TemporaryDirectory() as tmp:
        w = _g4_world(tmp)
        info = w.info()
        assert info["channels"] == 3 and info["gaze_command"] and info["geometry"] and info["action_dim"] == 8 and info["size"] == 96
        r = w.tick({"target": [0, 0, 0, 0, 0]})
        for k in ("stim", "pred", "recon", "frozen"):
            assert _decode(r["img"][k]).shape == (96, 96, 3), k
        assert _decode(r["img"]["surprise"]).shape == (96, 96)
        assert r["gaze"]["command"] is None and len(r["gaze"]["pupil"]) == 3 and all(np.isfinite(r["gaze"]["pupil"]))
        assert abs(r["mse"]["pred"] - r["mse"]["pred_frozen"]) < 1e-9
        r = w.tick({"target": [0.1, 0, 0.2, 0.3, 0], "scatter": [[0.5, 0.5, 1.0]], "gaze": [0.3, -0.2]})
        assert r["gaze"]["command"] == [0.3, -0.2] and w.canvas.shape == (1, 3, 96, 96)
        d = w.tick({"target": [0.1, 0, 0.2, 0.3, 0], "dream": True, "scatter": [[0.5, 0.5, 1.0]], "gaze": [5.0, -9.0]})
        assert d["dream"] and np.isfinite(d["kl"]) and np.isfinite(d["mse"]["pred"])
        a = w._action(np.zeros(5, np.float32), [5.0, -9.0])
        assert a.shape == (1, 8) and float(a[0, 5]) == 1.0 and a[0, 6:8].tolist() == [1.0, -1.0], "command is clipped to the image"
        assert float(w._action(np.zeros(5, np.float32), None)[0, 5]) == 0.0


def test_gaze_command_steers_the_dreamed_pupil_and_old_models_ignore_it():
    ends = {}
    for name, target in (("left", [-0.35, 0.0]), ("right", [0.35, 0.0])):
        with tempfile.TemporaryDirectory() as tmp:
            w = _g4_world(tmp, seed=11)
            for _ in range(14):
                r = w.tick({"target": [0, 0, 0, 0, 0], "dream": True, "gaze": target})
            ends[name] = r["gaze"]["pupil"]
    assert ends["right"][0] - ends["left"][0] > 0.1, ends          # 0.7 commanded apart (image units); measured gain ~0.8 at step 8
    with tempfile.TemporaryDirectory() as tmp:
        old = _world(tmp)
        assert not old.gaze_ok and not old.info()["gaze_command"] and old.info()["channels"] == 1
        r = old.tick({"target": [0, 0, 0, 0, 0], "gaze": [0.3, 0.3]})
        assert r["gaze"] == {"command": None, "pupil": None}


def test_learning_runs_on_the_rgb_gaze_model_and_trains_rssm_only():
    with tempfile.TemporaryDirectory() as tmp:
        w = _g4_world(tmp)
        enc0 = {k: v.clone() for k, v in w.model.encoder.state_dict().items()}
        rssm0 = {k: v.clone() for k, v in w.model.rssm.state_dict().items()}
        for i in range(24):
            w.tick({"target": [0.1 * np.sin(i / 3), 0, 0, 0, 0], "learn": True, "dream": i >= 12, "gaze": [0.2, 0.1] if i % 2 else None,
                    "scatter": [[0.4, 0.4, 0.8]] if i % 3 == 0 else []})
        assert w.learn_steps > 0 and np.isfinite(w.last_loss) and np.isfinite(w.probe_ema)
        assert all(torch.equal(v, enc0[k]) for k, v in w.model.encoder.state_dict().items())
        assert any(not torch.equal(v, rssm0[k]) for k, v in w.model.rssm.state_dict().items())
        assert w.replay and w.replay[0][0].shape == (ui.SEQ_T, 3, 96, 96) and w.replay[0][1].shape == (ui.SEQ_T, 8)


def test_light_gain_zero_means_no_light_default_is_unchanged_and_inverts():
    s = InterventionSampler(p=1.0)
    yy, xx = torch.meshgrid(torch.linspace(0, 1, 32), torch.linspace(0, 1, 32), indexing="ij")
    img = (0.2 + 0.15 * torch.sin(3 * xx) * torch.cos(2.5 * yy))[None, None, None]
    zero = torch.zeros(1, 1, 5)
    assert torch.allclose(s.apply_state(img, zero, light=0.0), img, atol=1e-5), "zero touch and no light must leave the picture alone"
    default = s.apply_state(img, zero)
    assert torch.equal(default, s.apply_state(img, zero, light=1.0)), "light=None is the training behaviour (gain 1)"
    assert float((default - img)[0, 0, 0, 15:17, 15:17].mean()) > 0.4, "with the light on the centre is lit by ~light_strength"
    both = s.apply_state(torch.cat([img, img]), torch.zeros(2, 1, 5), light=torch.tensor([0.0, 1.0]))
    assert torch.allclose(both[0], img[0], atol=1e-5) and torch.equal(both[1], default[0]), "the gain is per sample"
    st = torch.tensor([[[0.1, -0.08, 0.1, 0.3, -0.2]]])
    back = s.invert_state(s.apply_state(img, st, light=0.0), st, light=0.0)
    assert float((back - img)[..., 8:-8, 8:-8].abs().max()) < 0.03


def test_ui_light_is_off_at_zero_touch_and_only_on_when_the_user_switches_it_on():
    with tempfile.TemporaryDirectory() as tmp:
        w = _world(tmp)
        c0 = ui.u8(w.canvas[0])
        r = w.tick({"target": [0, 0, 0, 0, 0]})
        assert r["light"] is False
        assert np.abs(_decode(r["img"]["stim"]).astype(int) - c0.astype(int)).max() <= 1, "zero touch: the stimulus is the model's own render, no glare"
        w.reseed({"seed": 7})
        r = w.tick({"target": [0, 0, 0, 0, 0], "light": True})
        glow = _decode(r["img"]["stim"]).astype(int) - c0.astype(int)
        assert r["light"] is True and glow[28:36, 28:36].mean() > 60 and abs(glow[0:4, 0:4].mean()) < 2, "light on: lit in the middle only"
        assert w.tick({"target": [0, 0, 0, 0, 0]})["light"] is True, "a tick without the key keeps the light as it was"
        assert w.tick({"target": [0, 0, 0, 0, 0], "light": False, "intervene": False})["light"] is True, "intervention off ignores the light switch"
        assert w.tick({"target": [0, 0, 0, 0, 0], "light": False})["light"] is False
        w.tick({"target": [0, 0, 0, 0, 0], "light": True})
        w.reseed({"seed": 7})
        assert w.light_on is False and w.tick({"target": [0, 0, 0, 0, 0]})["light"] is False, "a new eye starts unlit"
        assert w.set_state({"state": [0, 0, 0, 0, 0], "light": True}) == {"ok": True} and w.light_on is True
        w.reset()
        assert w.light_on is False


L1_ID = "pilot_results/kaggle_long3_30k/pilot/L1_v6_long/model.pt"
ADV_ID = "pilot_results/kaggle_r5s2/s2/adv/model.pt"
GAZE_ID = "pilot_results/kaggle_gaze1_8000/pilot/G4_cmd_geom_prior/model.pt"


def test_model_zoo_lists_only_world_models_and_marks_the_current_one():
    from ivs import modelzoo
    with tempfile.TemporaryDirectory() as tmp:
        w = _world(tmp)
        r = w.models()
    ids = {m["id"] for m in r["models"]}
    assert {"checkpoints/latest.pt", L1_ID, ADV_ID, GAZE_ID} <= ids
    assert r["current"] == "checkpoints/latest.pt" and sum(m["current"] for m in r["models"]) == 1
    assert not any("model_partial" in i or "nca" in i or "vit" in i for i in ids), "partial / NCA / ViT checkpoints are not world models"
    by = {m["id"]: m for m in r["models"]}
    assert by[L1_ID]["size"] == 96 and by[L1_ID]["channels"] == 3 and not by[L1_ID]["gaze"] and by[GAZE_ID]["gaze"] and by["checkpoints/latest.pt"]["size"] == 64
    for bad in ("../../etc/passwd", "pilot_results/kaggle_r4/r4/ae/ae.pt", "pilot_results/kaggle_long3_30k/pilot/L1_v6_long/model_partial.pt", ""):
        try:
            modelzoo.resolve(ui.Path(ui.__file__).resolve().parents[1], bad)
            raise AssertionError(bad)
        except ValueError:
            pass


def test_switching_models_swaps_size_channels_gaze_and_starts_a_fresh_world():
    with tempfile.TemporaryDirectory() as tmp:
        w = _world(tmp)
        w.tick({"target": [0.1, 0, 0.2, 0.3, 0], "scatter": [[0.5, 0.5, 1.0]]})
        w.snapshot("mine")
        assert len(w.snapshots) == 1 and w.s.any()
        assert w.load_model({"id": "checkpoints/latest.pt"})["changed"] is False
        r = w.load_model({"id": GAZE_ID})
        assert r["changed"] and r["info"]["size"] == 96 and r["info"]["channels"] == 3 and r["info"]["gaze_command"] and r["info"]["geometry"]
        assert [s["name"] for s in w.snapshots] == ["birth"] and not w.s.any() and not w.replay and w.tick_n == 0
        t = w.tick({"target": [0, 0, 0, 0, 0], "gaze": [0.2, 0.1]})
        assert _decode(t["img"]["stim"]).shape == (96, 96, 3) and t["gaze"]["command"] == [0.2, 0.1] and np.isfinite(t["kl"])
        assert w.models()["current"] == GAZE_ID
        r = w.load_model({"id": L1_ID})
        assert not r["info"]["gaze_command"] and r["info"]["size"] == 96
        assert w.tick({"target": [0, 0, 0, 0, 0], "gaze": [0.2, 0.1]})["gaze"] == {"command": None, "pupil": None}
        r = w.load_model({"id": "checkpoints/latest.pt"})
        assert r["info"]["size"] == 64 and r["info"]["channels"] == 1
        assert _decode(w.tick({"target": [0, 0, 0, 0, 0], "dream": True})["img"]["pred"]).shape == (64, 64)
        for bad in ("nope", "../../ivs/ui.py", None):
            try:
                w.load_model({"id": bad})
                raise AssertionError(bad)
            except ValueError:
                pass
        assert w.models()["current"] == "checkpoints/latest.pt" and w.tick({"target": [0, 0, 0, 0, 0]})["tick"] == 2, "a refused id leaves the world running"


def test_a_saved_world_is_only_loaded_for_the_checkpoint_it_was_saved_for():
    with tempfile.TemporaryDirectory() as tmp:
        w = _world(tmp)                                           # started on latest.pt with --world tmp/w.pt
        first = {k: v.clone() for k, v in w.model.state_dict().items()}
        w.load_model({"id": L1_ID})
        other = w._world_path(w.checkpoint)
        assert other != Path(tmp) / "w.pt" and other.parent == Path(tmp)
        with torch.no_grad():
            next(w.model.rssm.parameters()).add_(0.5)                  # the live (plastic) weights differ from the checkpoint
        saved = {k: v.clone() for k, v in w.model.state_dict().items()}
        w.save()
        assert torch.load(other, weights_only=False)["checkpoint"] == str(Path(w.checkpoint).resolve())
        w.load_model({"id": "checkpoints/latest.pt"})
        w.load_model({"id": L1_ID})                               # back: its own saved world is found and loaded
        assert w.resumed and all(torch.equal(v, saved[k]) for k, v in w.model.state_dict().items())
        torch.save({"model": first, "ctrl": w.ctrl.state_dict(), "learn_steps": 0, "tick": 0}, other)   # a world file without the checkpoint field
        w.load_model({"id": "checkpoints/latest.pt"})
        w.load_model({"id": L1_ID})
        assert not w.resumed, "a world file that does not say which checkpoint it belongs to is never loaded on a switch"


def test_http_models_route_and_load_model_route():
    import http.client, json, threading
    with tempfile.TemporaryDirectory() as tmp:
        w = _world(tmp)
        srv = ui.ThreadingHTTPServer(("127.0.0.1", 0), ui.make_handler(w))
        th = threading.Thread(target=srv.serve_forever, daemon=True)
        th.start()
        try:
            c = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=60)
            c.request("GET", "/api/models"); r = c.getresponse(); body = json.loads(r.read())
            assert r.status == 200 and body["current"] == "checkpoints/latest.pt" and len(body["models"]) >= 20
            c.request("POST", "/api/load_model", json.dumps({"id": L1_ID}), {"Content-Type": "application/json"}); r = c.getresponse(); body = json.loads(r.read())
            assert r.status == 200 and body["changed"] and body["info"]["size"] == 96
            c.request("POST", "/api/load_model", json.dumps({"id": "bogus"}), {"Content-Type": "application/json"}); r = c.getresponse(); body = json.loads(r.read())
            assert r.status == 500 and "unknown model id" in body["error"]
            c.request("GET", "/"); r = c.getresponse(); html = r.read().decode()
            assert 'id="modelSel"' in html and "/api/load_model" in html
        finally:
            srv.shutdown(); srv.server_close()


def test_lab_live_edit_changes_only_the_live_model_and_is_validated():
    with tempfile.TemporaryDirectory() as tmp:
        w = _world(tmp)
        w.lab_root = lambda: Path(tmp)
        assert w.info()["lab_active"] is False
        st = w.lab_state()
        assert st["available"] and st["results"] is None and st["job"] is None
        r0 = w.tick({"target": [0, 0, 0, 0, 0], "dream": True})
        assert r0["mse"]["recon"] == r0["mse"]["recon_frozen"]                   # birth: live == frozen
        out = w.lab_set({"lesion": {"space": "z", "idx": list(range(8)), "value": "zero"}})
        assert out["active"] and w.info()["lab_active"] and w.lab_state()["intervention"]["lesion"]["value"] == "zero"
        r1 = w.tick({"target": [0, 0, 0, 0, 0], "dream": True})
        assert r1["mse"]["recon"] != r1["mse"]["recon_frozen"]                   # the live model was edited, the frozen copy was not
        for bad in ({"steer": {"target": "pupil_x"}}, {"lesion": {"group": "z_active"}}, {"lesion": {"space": "z", "idx": [10 ** 6], "value": "zero"}}):
            try:
                w.lab_set(bad)
            except ValueError:
                continue
            raise AssertionError(bad)
        assert w.info()["lab_active"], "a rejected edit must not clear or change the active one"
        assert not w.lab_set({"clear": True})["active"] and not w.info()["lab_active"]
        w.lab_set({"lesion": {"space": "h", "idx": [0], "value": "zero"}})
        w.load_model({"id": L1_ID})
        assert not w.info()["lab_active"], "switching models removes the edit"


def test_lab_run_in_background_then_served_over_http():
    import http.client, json, threading, time
    with tempfile.TemporaryDirectory() as tmp:
        w = _world(tmp)
        w.lab_root = lambda: Path(tmp)
        out = w.lab_run({"quick": True})
        assert out["job"]["state"] == "running"
        try:
            w.lab_run({"quick": True})
            raise AssertionError("a second run must be refused while one is running")
        except ValueError as e:
            assert "in progress" in str(e)
        w.tick({"target": [0, 0, 0, 0, 0]})                                       # the live world keeps working meanwhile
        w.lab_thread.join(timeout=900)
        assert w.lab_job["state"] == "done", w.lab_job
        st = w.lab_state()
        res = st["results"]
        assert res["probes"]["brightness"]["r2"]["hz"] is not None and "filmstrip" in res["images"]
        iv = w.lab_set({"steer": {"target": "brightness", "alpha": 2.0}, "lesion": {"group": "z_all", "value": "mean"}})
        assert iv["active"]
        r = w.tick({"target": [0, 0, 0, 0, 0], "dream": True})
        assert np.isfinite(r["kl"])
        srv = ui.ThreadingHTTPServer(("127.0.0.1", 0), ui.make_handler(w))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            c = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=60)
            c.request("GET", "/api/lab"); r = c.getresponse(); body = json.loads(r.read())
            assert r.status == 200 and body["results"]["capacity"]["z_dim"] == w.cfg.stoch_dim and body["intervention"]["steer"]["target"] == "brightness"
            c.request("GET", "/api/lab/img?t=1&f=filmstrip"); r = c.getresponse(); data = r.read()
            assert r.status == 200 and r.getheader("Content-Type") == "image/png" and data[:4] == b"\x89PNG"
            for bad in ("../results", "nope", "", "filmstrip.png"):
                c.request("GET", "/api/lab/img?f=" + bad); r = c.getresponse(); r.read()
                assert r.status == 404, bad
            c.request("POST", "/api/lab/set", json.dumps({"clear": True}), {"Content-Type": "application/json"}); r = c.getresponse(); body = json.loads(r.read())
            assert r.status == 200 and not body["active"]
            c.request("POST", "/api/lab/set", json.dumps({"steer": {"target": "nope"}}), {"Content-Type": "application/json"}); r = c.getresponse(); body = json.loads(r.read())
            assert r.status == 500 and "no probe direction" in body["error"]
            c.request("GET", "/"); r = c.getresponse(); html = r.read().decode()
            assert 'id="labSec"' in html and "/api/lab/run" in html and "/api/lab/set" in html
        finally:
            srv.shutdown(); srv.server_close()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
