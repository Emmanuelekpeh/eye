"""Run: python tests/test_modelscore.py   (or pytest -p no:xonsh). Real checkpoints and the real eye segmenter; no dataset is read."""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import score_models as S
from ivs import modelzoo


def test_best_window_is_the_best_100_tick_share_of_any_world():
    f = np.zeros((300, 2), bool)
    f[50:150, 0] = True                       # world 0: a perfect 100-tick run in the middle
    f[:30, 1] = True
    assert S.best_window(f) == 1.0
    assert abs(S.best_window(f[:100]) - 0.5) < 1e-9, "T <= window falls back to the best world's share"


def test_scoring_writes_stats_strips_resumes_and_feeds_the_zoo():
    with tempfile.TemporaryDirectory() as tmp:
        out, strips = Path(tmp) / "scores.json", Path(tmp) / "strips"
        argv = ["--only", r"checkpoints/latest\.pt", "--worlds", "3", "--ticks", "40", "--out", str(out), "--strips", str(strips)]
        res = S.main(argv)
        r = res["checkpoints/latest.pt"]
        assert set(modelzoo.SCORE_KEYS) <= set(r) and 0 <= r["eye_frac"] <= 1 and 0 <= r["any_eye"] <= 1 and r["size"] == 64 and r["worlds"] == 3
        png = strips / "checkpoints_latest.pt.png"
        assert png.exists() and png.stat().st_size > 500
        before = out.read_text()
        S.main(argv)                                           # resume: nothing is rescored (same bytes, worlds, ticks)
        assert out.read_text() == before
        assert S.main(argv + ["--sheets"]) and (strips / "sheet_1.png").exists()
        # the zoo merges scores only from <root>/pilot_results/model_scores.json and only for an unchanged file
        root = Path(tmp) / "root"
        (root / "checkpoints").mkdir(parents=True)
        (root / "pilot_results").mkdir()
        (root / "checkpoints" / "latest.pt").write_bytes((ROOT / "checkpoints" / "latest.pt").read_bytes())
        assert modelzoo.discover(root)[0]["score"] is None
        rec = dict(r, bytes=(root / "checkpoints" / "latest.pt").stat().st_size)
        (root / "pilot_results" / "model_scores.json").write_text(json.dumps({"checkpoints/latest.pt": rec}))
        z = modelzoo.discover(root)[0]
        assert z["score"]["eye_frac"] == r["eye_frac"] and set(z["score"]) == set(modelzoo.SCORE_KEYS)
        (root / "pilot_results" / "model_scores.json").write_text(json.dumps({"checkpoints/latest.pt": dict(rec, bytes=1)}))
        assert modelzoo.discover(root)[0]["score"] is None, "a score for a different file size is stale and ignored"
        # looked-at verdicts (model_curation.json) are applied per id, only for an unchanged file, and rank the list
        size = (root / "checkpoints" / "latest.pt").stat().st_size
        cur = root / "pilot_results" / "model_curation.json"
        assert modelzoo.discover(root)[0]["tier"] is None
        cur.write_text(json.dumps({"models": {"checkpoints/latest.pt": {"tier": "eye", "bytes": size}}}))
        assert modelzoo.discover(root)[0]["tier"] == "eye"
        cur.write_text(json.dumps({"models": {"checkpoints/latest.pt": {"tier": "eye", "bytes": size + 1}}}))
        assert modelzoo.discover(root)[0]["tier"] is None, "verdict for a different file size is stale"
        cur.write_text(json.dumps({"models": {"checkpoints/latest.pt": {"tier": "bogus", "bytes": size}}}))
        assert modelzoo.discover(root)[0]["tier"] is None, "unknown tiers are ignored"
        cur.write_text("{not json")
        assert modelzoo.discover(root)[0]["tier"] is None, "a corrupt curation file must not break the zoo"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
