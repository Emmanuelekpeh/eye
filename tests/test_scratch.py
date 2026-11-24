"""Run: python -m pytest -p no:xonsh tests/test_scratch.py   (preset / budget logic + the train_only validation rule through the real pilot CLI on real mini caches)."""
import dataclasses
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np

import scratch_chain as SC
from ivs.cachedata import CacheBank, train_only_sources
from ivs.pilot import PRESETS, build_run

CACHE = ROOT.parent / "data" / "cache"


def test_n0_scratch_is_l1_recipe_with_random_init_and_short_stages():
    n, l1 = build_run("N0_scratch", 100), build_run("L1_v6_long", 100)
    d = {k: (getattr(l1, k), getattr(n, k)) for k in dataclasses.asdict(n) if getattr(l1, k) != getattr(n, k)}
    assert set(d) == {"name", "train_sources", "stages"}, d
    assert n.train_sources is None and n.image_weight == 0.3 and n.lr == 1e-3 and n.kl_balance == 0.99
    assert sum(f for _, f in n.stages) == 1.0 and [t for t, _ in n.stages] == [8, 16]
    # 8500 steps at L1's measured T4 speeds (6.83 / 3.78 steps/s) must fit the stage-1 budget (~26 min)
    sec = 8500 * (0.75 / 6.83 + 0.25 / 3.78)
    assert 20 * 60 < sec < 28 * 60, sec


def test_stage2_budget_follows_the_deadline():
    assert SC.stage2_minutes(50, 27.0) == 20.0                    # 50 - 27 - 3 margin
    assert SC.stage2_minutes(50, 46.0) == 1.0 or SC.stage2_minutes(50, 46.0) == 0.0
    assert SC.stage2_minutes(50, 46.0) == 0.0                     # less than the 2 min floor: skipped, not squeezed
    assert SC.stage2_minutes(50, 60.0) == 0.0
    assert SC.WEIGHTS["ctl"] == "video=0.5,openeds=0.5" and abs(sum(float(p.split("=")[1]) for p in SC.WEIGHTS["mix"].split(",")) - 1.0) < 1e-9


def test_train_only_source_has_no_val_and_pilot_cli_skips_it_for_validation():
    from test_pilot import _mini_multisource
    from ivs import pilot
    with tempfile.TemporaryDirectory() as root:
        dv, ds = _mini_multisource(root)
        dream = Path(root) / "dreams"
        dream.mkdir()
        man = []
        with np.load(next(dv.glob("*.npz"))) as src:
            fr = src["frames"][:64, :, :96] if src["frames"].shape[2] >= 96 else src["frames"][:64]
        for i in range(3):                                                       # real frames re-labelled as a synthetic 'dream' source
            name = f"dream_t_{i}.dream"
            m = dict(name=name, group=f"dream_t_{i}", source="dream", pad_mode="reflect", chroma=0.0, hue_spread=0.0, train_only=True, fps=20.0)
            np.savez_compressed(dream / f"dream_t_{i}.npz", frames=np.ascontiguousarray(fr), seq_bounds=np.array([0, len(fr)]), meta=json.dumps(m))
            man.append(m)
        (dream / "manifest.json").write_text(json.dumps(man))
        mrl = Path(root) / "mrl"                                                     # the appearance source L1's image loss needs (real mrl subjects)
        mrl.mkdir()
        mm = json.loads((CACHE / "mrl" / "manifest.json").read_text())[:8]
        for m in mm:
            (mrl / f"{m['group']}.npz").write_bytes((CACHE / "mrl" / f"{m['group']}.npz").read_bytes())
        (mrl / "manifest.json").write_text(json.dumps(mm))
        assert train_only_sources([dv, ds, dream]) == ["dream"] and train_only_sources([dv, ds]) == []
        out = Path(root) / "out"
        s = pilot.main(["--cache", str(dv), str(ds), str(mrl), str(dream), "--source-weights", "video=0.4,openeds=0.4,dream=0.2", "--out", str(out), "--runs", "N0_scratch", "--steps", "4",
                        "--batch", "2", "--image-size", "64", "--device", "cpu", "--max-frames-per-video", "120", "--val-frac", "0.34", "--eval-windows", "64", "--log-every", "2"])
        di = json.loads((out / "data_info.json").read_text())
        assert di["train"]["frames_by_source"].get("dream", 0) > 0                  # the dream source is trained on ...
        assert "dream" not in di["val"]["frames_by_source"]                         # ... and never validated on
        m = s["N0_scratch"]
        assert set(m["heldout_by_source"]) == {"video", "openeds"} and m["train_data"]["frames_by_source"]["dream"] > 0
        assert (out / "N0_scratch" / "model.pt").exists()
