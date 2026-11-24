"""Tests for scripts/clarity_ceiling.py on the REAL raw data (set/ + data/cache manifests). Run: python tests\\test_clarity.py (from ivs2)."""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
import clarity_ceiling as C   # noqa: E402

ROOT = C.ROOT


def _openeds_pairs(n=3):
    m = sorted(json.loads((ROOT / "data" / "cache" / "openeds" / "manifest.json").read_text()), key=lambda m: m["name"])[3]
    pairs = C.read_openeds(ROOT / "set" / m["group"], tuple(m["crop_box"]), n)
    assert len(pairs) == n, "no real OpenEDS frames with a visible pupil found"
    return m, pairs


def test_down_up_is_identity_at_or_above_native_and_loses_detail_below():
    _, pairs = _openeds_pairs()
    g = pairs[0][0]
    assert np.array_equal(C.down_up(g, g.shape[0]), g) and np.array_equal(C.down_up(g, g.shape[0] + 50), g)
    assert C.down_up(g, 96).shape == g.shape and not np.array_equal(C.down_up(g, 96), g)


def test_ceiling_improves_monotonically_with_height_on_real_frames():
    _, pairs = _openeds_pairs()
    g = pairs[0][0]
    rows = [C.frame_metrics(g, C.down_up(g, h)) for h in (64, 96, 128, 192, g.shape[0])]
    for a, b in zip(rows, rows[1:]):
        assert b["psnr"] >= a["psnr"] and b["edge_keep"] >= a["edge_keep"] - 1e-6 and b["edge_fidelity"] >= a["edge_fidelity"] - 1e-6
    assert rows[-1]["edge_keep"] > 0.999 and rows[-1]["edge_fidelity"] > 0.999 and rows[-1]["psnr"] > 90, "native height must reproduce the frame exactly"
    assert rows[1]["edge_keep"] < 0.95, "96 px must visibly lose edge strength on these frames"


def test_edge_fidelity_is_lowered_by_blur_noise_and_exaggerated_contrast_while_edge_keep_is_not():
    _, pairs = _openeds_pairs()
    g = pairs[0][0]
    assert abs(C.frame_metrics(g, g)["edge_fidelity"] - 1.0) < 1e-4
    import cv2
    blur = cv2.GaussianBlur(g, (0, 0), 3)
    noisy = np.clip(blur.astype(np.float32) + np.random.RandomState(0).normal(0, 20, g.shape), 0, 255).astype(np.uint8)
    loud = np.clip((g.astype(np.float32) - g.mean()) * 2 + g.mean(), 0, 255).astype(np.uint8)
    m = {k: C.frame_metrics(g, v) for k, v in (("blur", blur), ("noisy", noisy), ("loud", loud))}
    assert m["blur"]["edge_fidelity"] < 0.9, "blur must lower edge_fidelity"
    assert m["noisy"]["edge_keep"] > m["blur"]["edge_keep"] + 0.05, "noise inflates the magnitude metric ..."
    assert m["noisy"]["edge_fidelity"] < m["blur"]["edge_fidelity"] - 0.05, "... but lowers edge_fidelity"
    assert m["loud"]["edge_keep"] > 1.3 and m["loud"]["edge_fidelity"] < 0.8, "doubled contrast: more edge strength than native (> 1), fidelity drops"


def test_mask_metrics_are_one_at_native_and_lower_at_96():
    import torch
    from ivs.eyemetric import gt_metrics
    _, pairs = _openeds_pairs()
    real = torch.from_numpy(np.stack([p[0] for p in pairs])).float() / 255.0
    mask = torch.from_numpy(np.stack([p[1] for p in pairs])).to(torch.uint8)
    at = lambda h: gt_metrics(real, torch.from_numpy(np.stack([C.down_up(p[0], h) for p in pairs])).float() / 255.0, mask)
    full, low = at(pairs[0][0].shape[0]), at(96)
    assert abs(full["pupil_edge_grad_ratio"] - 1.0) < 1e-4 and abs(full["limbus_grad_ratio"] - 1.0) < 1e-4
    assert low["pupil_edge_grad_ratio"] < full["pupil_edge_grad_ratio"] and low["limbus_grad_ratio"] < full["limbus_grad_ratio"]


def test_detail_patch_stays_inside_the_frame():
    _, pairs = _openeds_pairs()
    g = pairs[0][0]
    y, x, s = C.detail_patch(g)
    assert 0 <= y <= g.shape[0] - s and 0 <= x <= g.shape[1] - s


def test_script_end_to_end_on_real_data():
    with tempfile.TemporaryDirectory() as tmp:
        res = C.run(tmp, heights=(96, 128), openeds_seqs=2, openeds_frames=2, videos_per_kind=(("gray", 1), ("tint", 1), ("colour", 1)),
                    video_frames=2, sheet_rows=1)
        for src in ("openeds", "gray", "tint", "colour"):
            assert res[src]["128"]["edge_keep"] > res[src]["96"]["edge_keep"], src
            assert (Path(tmp) / f"sheet_{src}.png").exists()
        assert res["openeds"]["96"]["pupil_edge_grad_ratio"] > 0
        assert json.loads((Path(tmp) / "ceiling.json").read_text())["heights"] == [96, 128]
        assert res["openeds_iris_px"]["128"] > res["openeds_iris_px"]["96"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
