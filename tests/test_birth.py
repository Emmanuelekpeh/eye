"""Run: python tests/test_birth.py  (pytest on this box needs `-p no:xonsh`). Real L1 checkpoint + real caches, CPU, no fakes."""
import base64
import io
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import torch
from PIL import Image

from ivs import birth, eyemetric as E, ui
from ivs.cachedata import CacheBank, _safe
from ivs.config import ModelConfig
from ivs.model import EyeWorldModel

L1 = ROOT / "pilot_results" / "kaggle_long3_30k" / "pilot" / "L1_v6_long" / "model.pt"       # RGB 96 px, action_dim 5
G4 = ROOT / "pilot_results" / "kaggle_gaze1_8000" / "pilot" / "G4_cmd_geom_prior" / "model.pt"  # different weights, action_dim 8
CACHES = [ROOT.parent / "data" / "cache" / "videos", ROOT.parent / "data" / "cache" / "openeds"]


def _model(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = EyeWorldModel(ModelConfig(**ck["config"]))
    m.load_state_dict(ck["model"])
    return m.eval()


def _val_bank():
    return CacheBank(CACHES, image_size=96, split="val", val_frac=0.15, seed=0, tint_mode="gray", sources=["video", "openeds"])


def _two_births(tmp):
    """Two real held-out windows (one video, one openeds recording) perceived by L1; returns (model, path, h, z, wins)."""
    import make_birth
    bank, m = _val_bank(), _model(L1)
    c = make_birth.candidates(bank, 8, 40, 0)
    pick = [next(x for x in c if x["source"] == s) for s in ("video", "openeds")]
    wins = torch.stack([bank.window_at(x["seq"], x["offset"], 8) for x in pick])
    p = birth.perceive(m, wins, seed=0)
    path = Path(tmp) / "b.pt"
    birth.save_birth(path, p["h"], p["z"], L1, {"group": [x["group"] for x in pick]})
    return m, path, p["h"], p["z"], wins


def test_window_at_is_the_cached_frames_in_the_evaluation_crop():
    bank = CacheBank(CACHES[1], image_size=96, split="all", sources=["openeds"], max_videos=1)
    name = bank.videos[0]["name"]
    with np.load(CACHES[1] / (_safe(Path(name).stem) + ".npz")) as z:
        raw = z["frames"]                                                  # (N,96,128,3) uint8 gray-replicated
    w = bank.window_at(0, 3, 4)
    assert w.shape == (4, 3, 96, 96) and 0 <= float(w.min()) and float(w.max()) <= 1
    stored_x0, eval_x0 = (128 - bank.store_w) // 2, (bank.store_w - 96) // 2          # the cache is centre-cropped twice
    expect = torch.from_numpy(raw[3:7, :, stored_x0 + eval_x0: stored_x0 + eval_x0 + 96]).permute(0, 3, 1, 2).float() / 255
    assert torch.equal(w, expect)
    for bad in ((0, -1, 4), (0, int(bank.seq_len[0]) - 3, 4)):
        try:
            bank.window_at(*bad)
            raise AssertionError("an out-of-range window must be refused")
        except ValueError:
            pass
    names = bank.group_names()
    assert set(names.values()) == {v["group"] for v in bank.videos} and set(bank.seq_group.tolist()) <= set(names)


def test_perceive_keeps_only_a_latent_and_round_trips_through_a_file():
    with tempfile.TemporaryDirectory() as tmp:
        m, path, h, z, wins = _two_births(tmp)
        assert h.shape == (2, 256) and z.shape == (2, 128) and torch.isfinite(h).all() and torch.isfinite(z).all()
        b = birth.load_birth(path, L1, m.cfg)
        assert torch.equal(b["h"], h) and torch.equal(b["z"], z) and b["checkpoint_name"] == "model.pt"
        recon = m.decode(b["h"], b["z"])
        assert recon.shape == (2, 3, 96, 96) and 0 <= float(recon.min()) and float(recon.max()) <= 1
        real_last = wins[:, -1]
        assert float(((recon - real_last) ** 2).mean()) < float(((real_last.mean() - real_last) ** 2).mean()), \
            "the remembered state must render something closer to the eye it perceived than a flat grey does"
        keys = set(torch.load(path, map_location="cpu", weights_only=True))
        assert "frames" not in keys and "images" not in keys, "no pixels are stored, only the latent state"
        assert birth.to_model_channels(wins, 1).shape == (2, 8, 1, 96, 96)


def test_a_birth_memory_is_refused_for_other_weights_or_shapes():
    with tempfile.TemporaryDirectory() as tmp:
        m, path, h, z, _ = _two_births(tmp)
        try:
            birth.load_birth(path, G4, _model(G4).cfg)
            raise AssertionError("a state made for L1 must not load for G4")
        except ValueError as e:
            assert "not for" in str(e)
        bad = Path(tmp) / "bad.pt"
        birth.save_birth(bad, h[:, :10], z, L1, {})
        try:
            birth.load_birth(bad, L1, m.cfg)
            raise AssertionError("wrong state width must be refused")
        except ValueError as e:
            assert "do not fit" in str(e)


def _args(tmp, birth_path=None, seed=1, ckpt=L1):
    argv = ["--checkpoint", str(ckpt), "--world", str(Path(tmp) / "w.pt"), "--fresh", "--seed", str(seed)]
    return ui.parse_args(argv + (["--birth", str(birth_path)] if birth_path else []))


def _png(data_url):
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1]))))


def test_ui_starts_from_the_birth_state_and_its_dream_is_the_offline_dream():
    with tempfile.TemporaryDirectory() as tmp:
        m, path, h, z, _ = _two_births(tmp)
        w = ui.World(_args(tmp, path, seed=1))
        assert w.birth_index == 1 and torch.equal(w.h, h[1:2]) and torch.equal(w.z, z[1:2])
        assert torch.equal(w.hf, w.h) and torch.equal(w.zf, w.z), "the frozen control starts from the same state"
        assert torch.equal(w.canvas, m.decode(h[1:2], z[1:2]).clamp(0, 1))
        info = w.info()
        assert info["birth"] == {"index": 1, "count": 2} and "birth memory" in info["source"]
        shown = [_png(w.tick({"target": [0, 0, 0, 0, 0], "dream": True})["img"]["stim"]) for _ in range(6)]
        mine = (E.ui_dream_frames(m, ticks=6, n=1, noise_seed=1, init_state=(h[1:2], z[1:2]))[:, 0] * 255).round().byte().permute(0, 2, 3, 1).numpy()
        assert all(np.array_equal(a, b) for a, b in zip(shown, mine)), "the offline dream from a birth state must be the UI's dream"
        r = w.reseed({"seed": 4})
        assert r["seed"] == 4 and w.birth_index == 0 and torch.equal(w.h, h[0:1])
        w.reseed({"seed": 3})
        assert w.birth_index == 1 and w.info()["birth"]["index"] == 1
        r0 = w.tick({"target": [0, 0, 0, 0, 0]})
        assert abs(r0["mse"]["pred"] - r0["mse"]["pred_frozen"]) < 1e-9, "live == frozen at birth"
        for bad in (1, 2):
            try:
                E.ui_dream_frames(m, ticks=2, n=bad + 1, noise_seed=1, init_state=(h[:bad], z[:bad]))
                raise AssertionError("n must match the number of states")
            except ValueError:
                pass


def test_without_a_birth_file_the_ui_still_starts_from_noise_and_birth_needs_a_checkpoint():
    with tempfile.TemporaryDirectory() as tmp:
        w = ui.World(_args(tmp, None, seed=7))
        assert w.birth is None and w.info()["birth"] is None and w.info()["source"] == "model's own render"
        ref = _model(L1)
        h, z = w._seed_state(ref, 1, 7)
        assert torch.equal(w.h, h) and torch.equal(w.z, z), "the noise start is unchanged"
        try:
            ui.World(ui.parse_args(["--checkpoint", str(Path(tmp) / "missing.pt"), "--world", str(Path(tmp) / "w.pt"), "--fresh", "--birth", str(Path(tmp) / "x.pt")]))
            raise AssertionError("--birth without a checkpoint must be refused")
        except ValueError as e:
            assert "trained checkpoint" in str(e)


def test_make_birth_script_end_to_end_on_the_real_checkpoint():
    import make_birth
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "birth.pt"
        s = make_birth.main(["--checkpoint", str(L1), "--out", str(out), "--k", "2", "--candidates", "6", "--context", "8",
                             "--dwell-ticks", "12", "--check-ticks", "14"])
        assert out.exists() and out.with_suffix(".sheet.png").exists() and out.with_suffix(".json").exists()
        b = birth.load_birth(out, L1, _model(L1).cfg)
        assert len(b["h"]) == 2 and len(set(b["meta"]["group"])) == 2 and set(b["meta"]["source"]) <= {"video", "openeds"}
        val_groups = set(_val_bank().group_names().values())
        assert set(b["meta"]["group"]) <= val_groups, "birth memories must come from held-out recordings"
        assert {"saved_birth_fresh_noise", "all_candidates_fresh_noise", "noise_start_fresh_noise"} <= set(s)
        try:
            make_birth.main(["--checkpoint", str(L1), "--out", str(out)])
            raise AssertionError("an existing file must not be overwritten without --force")
        except SystemExit as e:
            assert "exists" in str(e)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
