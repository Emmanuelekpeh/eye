"""Tests for scripts/s2_eval.py (R5 stage 2 evaluation) on REAL data: the L1 checkpoint, a stage-2 checkpoint written by scripts/adv_rssm.py (a few real training steps on a mini cache) and the
real held-out caches / segmenter. CPU. Run: python tests\\test_s2eval.py (from ivs2)."""
import copy
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
import s2_eval as S                               # noqa: E402
import test_advdream as TA                        # noqa: E402
import adv_rssm as AR                             # noqa: E402

_TMP = tempfile.TemporaryDirectory()
OUT = Path(_TMP.name)


def row(eye, first, maxlen=100.0, still=0.0, moving=0.6, sharp=1.0, contrast=0.9, recon=0.002, touch=0.5):
    n = 12
    rng = np.random.RandomState(int(eye * 1000))
    return dict(mean=dict(eye_frac=eye, first100=first, max_len=maxlen, still_frac=still, eye_moving=moving),
                per_world=dict(eye_frac=list(np.clip(eye + rng.randn(n) * 0.02, 0, 1)), first100=list(np.clip(first + rng.randn(n) * 0.02, 0, 1))),
                sharp_ratio={"99": sharp}, contrast_ratio={"99": contrast}, heldout=dict(passive=dict(recon=recon)),
                touch={k: {"8": dict(ratio=touch)} for k in S.TOUCH})


def test_gate_rule_follows_the_pre_registered_thresholds():
    l1, ctl = row(0.50, 0.47, 60, 0.004, 0.50, 0.5, 0.9, 0.002, 0.5), row(0.52, 0.49, 70)
    adv = row(0.72, 0.70, 110)
    v = S.gates(dict(L1=l1, ctl=ctl, adv=adv))
    assert v["G1_persistence"] and v["G2_attribution"] and v["G3_alive"] and v["G4_sharp_not_grain"] and v["G5_no_regression"], v
    assert not S.gates(dict(L1=l1, ctl=ctl, adv=row(0.64, 0.70, 110)))["G1_persistence"]
    assert not S.gates(dict(L1=l1, ctl=ctl, adv=row(0.72, 0.70, 80)))["G1_persistence"], "max_len gate"
    assert not S.gates(dict(L1=l1, ctl=row(0.68, 0.66), adv=adv))["G2_attribution"], "adv must beat ctl by +.08"
    assert not S.gates(dict(L1=l1, ctl=ctl, adv=row(0.72, 0.70, 110, still=0.2)))["G3_alive"]
    assert not S.gates(dict(L1=l1, ctl=ctl, adv=row(0.72, 0.70, 110, moving=0.4)))["G3_alive"]
    assert not S.gates(dict(L1=l1, ctl=ctl, adv=row(0.72, 0.70, 110, sharp=1.6)))["G4_sharp_not_grain"], "grain"
    assert not S.gates(dict(L1=l1, ctl=ctl, adv=row(0.72, 0.70, 110, contrast=0.5)))["G4_sharp_not_grain"]
    assert not S.gates(dict(L1=l1, ctl=ctl, adv=row(0.72, 0.70, 110, recon=0.0025)))["G5_no_regression"], "recon +25%"
    assert not S.gates(dict(L1=l1, ctl=ctl, adv=row(0.72, 0.70, 110, touch=0.8)))["G5_no_regression"], "touch ratio worse by .3"
    m, lo, hi = S.paired_bootstrap([1, 1, 1, 1], [0, 0, 0, 0])
    assert m == 1.0 and lo == 1.0 and hi == 1.0


def test_end_to_end_on_real_models_writes_sheets_and_a_complete_json():
    model, summ = AR.train_arm("adv", TA.L1, TA.mini_cache(), OUT / "train", steps=2, max_minutes=30, batch=2, lr=1e-3, d_lr=2e-4, w_adv=0.3, device=torch.device("cpu"), seed=0, log_every=1, probe_every=0, banks=TA.banks())
    ck = OUT / "train" / "adv" / "model.pt"
    res = S.main(["--models", f"L1={TA.L1}", f"ctl={ck}", f"adv={ck}", "--out", str(OUT / "eval"), "--ticks", "40", "--candidates", "6", "--sheet-worlds", "3"])
    assert (OUT / "eval" / "compare.png").exists() and all((OUT / "eval" / f"sheet_{n}.png").exists() for n in ("L1", "ctl", "adv"))
    saved = json.loads((OUT / "eval" / "s2_eval.json").read_text())
    for n in ("L1", "ctl", "adv"):
        r = saved["models"][n]
        assert set(S.KEYS) <= set(r["mean"]) and all(np.isfinite(v) for v in r["mean"].values())
        assert set(r["touch"]) == set(S.TOUCH) and set(r["touch"]["shift_x"]) == {"3", "8", "16"}
        assert r["heldout"]["passive"]["recon"] > 0 and "29" in r["sharp_ratio"]
    assert saved["models"]["adv"]["stage2"]["arm"] == "adv" and saved["models"]["L1"]["stage2"] is None
    assert set(saved["verdict"]) >= {"G1_persistence", "G2_attribution", "G3_alive", "G4_sharp_not_grain", "G5_no_regression", "G6"}
    print("   verdict", {k: v for k, v in saved["verdict"].items() if k != "detail"})


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as e:                           # noqa: BLE001
                n += 1
                import traceback
                traceback.print_exc()
                print("FAIL", name, repr(e))
    sys.exit(1 if n else 0)
