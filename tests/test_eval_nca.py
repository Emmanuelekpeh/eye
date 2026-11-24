"""Run: python tests/test_eval_nca.py   (REAL held-out S_ frames + masks + segmenter from the repo, TINY untrained networks, CPU smoke/logic test).
Tests scripts/eval_nca.py with a PD-NCA (persistent) checkpoint next to an E0-style one: the persistent paths (hidden carried through sampling / blind and
masked repair, living-mode repair rows, living dream stats at ticks 0/10/30/100/300, P1 / P2 gate dicts) run and are consistent, and the E0 model's paths
and outputs are unchanged (no living rows, no P1 / P2). Numbers of an untrained net mean nothing; the test checks plumbing and bookkeeping only."""
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import torch

from ivs import nca, ncapilot, pnca

spec = importlib.util.spec_from_file_location("eval_nca", HERE / "scripts" / "eval_nca.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
DATA = HERE.parent / "data" / "cache"
TINY = dict(state=12, hidden=24, n_inner=2, warmup=2, batch=4, pool_size=16, levels=3, k_ticks=2)


def tiny_checkpoint(path, persistent):
    rc = pnca.replace(pnca.PRun(name="T"), **{**TINY, **(dict(gate=True, fourier=True, fourier_hidden=16) if persistent else {})})
    if not persistent:
        rc = pnca.replace(rc, persist=False)
    cfg = pnca.model_config(rc, 96)
    torch.manual_seed(0)
    m = nca.EyeNCA(cfg)
    ncapilot.save_model(path, m, m, cfg, rc)


def test_persist_flag_is_read_from_the_checkpoint():
    with tempfile.TemporaryDirectory() as d:
        tiny_checkpoint(Path(d) / "p.pt", True)
        tiny_checkpoint(Path(d) / "e.pt", False)
        assert ev.is_persistent(ev.load_nca(Path(d) / "p.pt", "cpu")) and not ev.is_persistent(ev.load_nca(Path(d) / "e.pt", "cpu"))


def test_load_nca_builds_a_vitca_checkpoint_and_reads_its_persist_flag():
    from ivs import vitca
    with tempfile.TemporaryDirectory() as d:
        for plain in (False, True):
            rc = pnca.replace(pnca.PRun(name="V"), arch="vitca", dim=32, blocks=1, loops=2, vheads=2, mlp_mult=2, plain=plain, persist=not plain)
            cfg = pnca.model_config(rc, 96)
            m = nca.net_from_config(pnca.asdict(cfg))
            ncapilot.save_model(Path(d) / "v.pt", m, m, cfg, rc)
            net = ev.load_nca(Path(d) / "v.pt", "cpu")
            assert isinstance(net, vitca.EyeViTCA) and ev.is_persistent(net) is (not plain)


def test_full_eval_runs_on_real_data_with_persistent_and_e0_models():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        tiny_checkpoint(d / "PD.pt", True)
        tiny_checkpoint(d / "E0.pt", False)
        out = d / "out"
        res = ev.main(["--nca", f"PD={d / 'PD.pt'}", f"E0={d / 'E0.pt'}", "--n", "6", "--select-n", "4", "--dreams", "8", "--device", "cpu",
                       "--cache-openeds", str(DATA / "openeds"), "--cache-video", str(DATA / "videos"), "--masks", str(DATA / "openeds_masks"),
                       "--segmenter", str(HERE / "pilot_results" / "eyemetric" / "segmenter.pt"), "--out", str(out)])
        for kind in ("scatter", "pupil_erased"):
            rows = res["repair"][kind]
            for s in ev.LIVE_SIGMAS:
                assert f"PD live{ev.LIVE_REPAIR_TICKS} s={s}" in rows and f"E0 live{ev.LIVE_REPAIR_TICKS} s={s}" not in rows
            assert "PD masked(oracle)" in rows and "E0 masked(oracle)" in rows
        assert set(res["living"]["PD"]) == {str(t) for t in ev.LIVE_TIMES} and "E0" not in res["living"]
        assert res["living"]["PD"]["0"]["change_from_dream"] == 0.0 and res["living"]["PD"]["100"]["change_from_dream"] > 0
        assert set(res["gates"]["PD"]) == {"P1_persistence", "P2_regen"} and "E0" not in res["gates"]
        p2 = res["gates"]["PD"]["P2_regen"]
        assert p2["damaged_iou"] == res["repair"]["scatter"]["damaged"]["seg"].get("iou_pupil") or p2["damaged_iou"] == 0
        assert isinstance(p2["pass_numbers"], bool) and isinstance(res["gates"]["PD"]["P1_persistence"]["pass_numbers"], bool)
        for f in ("eval_nca.json", "sheet_repair_scatter.png", "sheet_dreams_PD.png", "sheet_living_PD_t100.png", f"sheet_living_PD_t{ev.LIVE_TIMES[-1]}.png",
                  "sheet_dreams_E0.png"):
            assert (out / f).stat().st_size > 500, f
        cols = (out / "sheet_repair_scatter_columns.txt").read_text()
        assert f"PD live{ev.LIVE_REPAIR_TICKS} s=0.05" in cols and "E0 live" not in cols
        json.loads((out / "eval_nca.json").read_text())


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name, flush=True)
    print("all eval_nca tests passed")
