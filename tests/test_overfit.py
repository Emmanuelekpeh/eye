"""Tests for scripts/overfit_clarity.py on REAL raw data (set/ + data/cache manifests), CPU. Run: python tests\\test_overfit.py (from ivs2)."""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
import clarity_ceiling as C        # noqa: E402
import overfit_clarity as O        # noqa: E402

_TMP = tempfile.TemporaryDirectory()
PACK = Path(_TMP.name) / "pack"


def _pack():
    if not (PACK / "openeds.npz").exists():
        O.build_pack(PACK, heights=(96, 128))
    return PACK


def test_pack_holds_real_clips_cut_to_the_native_square_with_aligned_masks():
    p = _pack()
    for key, ch in (("colour", 3), ("openeds", 3)):
        z = np.load(p / f"{key}.npz")
        meta = json.loads(str(z["meta"]))
        n, s = meta["frames"], meta["square"]
        assert z["train_96"].shape == (n, 96, 96, 3) and z["train_128"].shape == (n, 128, 128, 3)
        assert z["native_eval"].shape[:3] == (len(z["eval_idx"]), s, s) and 4 <= len(z["eval_idx"]) <= O.EVAL_N
        assert n >= 100, "an overfit clip needs real length"
        if key == "openeds":
            m = z["masks_eval"]
            assert m.shape == z["native_eval"].shape and ((m == 3).sum((1, 2)) >= 60).all() and ((m == 2).sum((1, 2)) >= 60).all()
            assert z["train_96"][..., 0].std() > 5 and np.array_equal(z["train_96"][..., 0], z["train_96"][..., 1]), "gray replicated to 3 channels"
        else:
            assert z["native_eval"].ndim == 4 and abs(float(z["native_eval"][..., 0].mean()) - float(z["native_eval"][..., 2].mean())) > 1, "colour kept"


def test_stored_training_frames_are_exactly_the_stage1_ceiling_construction():
    z = np.load(_pack() / "colour.npz")
    for i in (0, len(z["eval_idx"]) // 2, -1):
        stored = z["train_96"][z["eval_idx"][i]]
        s = z["native_eval"].shape[1]
        assert np.array_equal(cv2.resize(stored, (s, s), interpolation=cv2.INTER_CUBIC), C.down_up(z["native_eval"][i], 96))


def test_training_improves_the_fit_and_writes_results_sheets_and_gate():
    out = Path(_TMP.name) / "run"
    a = argparse.Namespace(pack=str(_pack()), out=str(out), clips=["colour", "openeds"], heights=[96], steps=60, batch=16, lr=1e-3, seed=0, probe_every=30, device="cpu")
    res = O.train_main(a)
    assert (out / "results.json").exists() and (out / "sheet_colour.png").exists() and (out / "sheet_openeds.png").exists()
    for key in ("colour", "openeds"):
        r = res[key]["heights"]["96"]
        assert len(r["curve"]) == 2 and all(np.isfinite(c["loss"]) for c in r["curve"])
        assert r["gate"]["edge_fidelity"] == bool(r["ae"]["edge_fidelity"] >= O.GATE * r["ceiling"]["edge_fidelity"])
        assert r["gate"]["edge_keep"] == bool(O.GATE * r["ceiling"]["edge_keep"] <= r["ae"]["edge_keep"] <= O.KEEP_MAX)
        assert r["ae_over_ceiling_fidelity"] == r["ae"]["edge_fidelity"] / r["ceiling"]["edge_fidelity"] and r["ae_over_ceiling_keep"] == r["ae"]["edge_keep"] / r["ceiling"]["edge_keep"]
        assert r["passed"] == all(r["gate"].values()) and r["ae"]["psnr"] > 8
    assert "pupil_edge" in res["openeds"]["heights"]["96"]["gate"] and "pupil_edge" not in res["colour"]["heights"]["96"]["gate"]
    assert json.loads((out / "results.json").read_text()).keys() == {"colour", "openeds"}
    # the trained net must beat an untrained one on the same frames
    z = np.load(_pack() / "colour.npz")
    from ivs.aeprobe import PlainAE
    torch.manual_seed(0)
    untrained = PlainAE(96, 512)
    x = torch.from_numpy(z["train_96"][z["eval_idx"]]).permute(0, 3, 1, 2).float() / 255.0
    s = z["native_eval"].shape[1]
    ng = [O.gray_of(n) for n in z["native_eval"]]
    base = O.score(ng, O.up_to_native(O.reconstruct(untrained, x, torch.device("cpu")), s), None)
    assert res["colour"]["heights"]["96"]["ae"]["psnr"] > base["psnr"] + 1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
