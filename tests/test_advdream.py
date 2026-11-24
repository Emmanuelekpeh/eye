"""Tests for ivs/advdream.py + scripts/adv_rssm.py (R5 stage 2) on REAL data: the L1 checkpoint and a mini cache copied from data/cache_hr/h96. CPU.
Run: python tests\\test_advdream.py (from ivs2)."""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
import adv_rssm as AR                             # noqa: E402
import heldout_clarity as H                       # noqa: E402
from ivs import advdream as A                     # noqa: E402
from ivs import eyemetric as E                    # noqa: E402
from ivs.config import ModelConfig                # noqa: E402
from ivs.model import EyeWorldModel               # noqa: E402
from ivs.pilot import build_run, make_batch, train_loss      # noqa: E402
from ivs.actions import InterventionSampler       # noqa: E402

ROOT = HERE.parent
L1 = HERE / "pilot_results" / "kaggle_long3_30k" / "pilot" / "L1_v6_long" / "model.pt"
_TMP = tempfile.TemporaryDirectory()
TMP = Path(_TMP.name)
_S = {}
CPU = torch.device("cpu")


def mini_cache():
    if "cache" not in _S:
        src = ROOT / "data" / "cache_hr" / "h96"
        metas = json.loads((src / "manifest.json").read_text())
        size = lambda m: (src / (H.safe(Path(m["name"]).stem) + ".npz")).stat().st_size
        v = sorted([m for m in metas if m.get("source", "video") == "video"], key=size)[:8]
        o = sorted([m for m in metas if m.get("source") == "openeds"], key=lambda m: m["name"])[:10]
        out = TMP / "mini"
        out.mkdir()
        for m in v + o:
            f = H.safe(Path(m["name"]).stem) + ".npz"
            shutil.copy2(src / f, out / f)
        (out / "manifest.json").write_text(json.dumps(v + o))
        _S["cache"] = out
    return _S["cache"]


def banks():
    if "banks" not in _S:
        _S["banks"] = AR.make_banks(mini_cache(), CPU, 96)
    return _S["banks"]


def fresh_model():
    model, ck = AR.load_model(L1, CPU)
    return model, ck


def test_freeze_splits_the_model_into_trainable_and_frozen_and_rejects_unknown_parameters():
    model, _ = fresh_model()
    params = A.freeze_for_stage2(model)
    learn = {n for n, p in model.named_parameters() if p.requires_grad}
    assert learn and all(n.startswith(A.TRAINABLE) for n in learn)
    assert all(not p.requires_grad for n, p in model.named_parameters() if n.startswith(A.FROZEN))
    assert sum(p.numel() for p in params) == sum(p.numel() for n, p in model.named_parameters() if n in learn)
    print(f"   trainable {sum(p.numel() for p in params):,} of {sum(p.numel() for p in model.parameters()):,}")
    model.stray = torch.nn.Linear(2, 2)
    try:
        A.freeze_for_stage2(model)
        raise AssertionError("an unclassified module must be rejected")
    except ValueError:
        pass


def test_ui_rollout_is_the_ui_dream_loop_and_gradients_reach_only_the_trainable_modules(monkeypatch=None):
    model, _ = fresh_model()
    A.freeze_for_stage2(model)
    real_randn = torch.randn_like
    torch.randn_like = lambda x, **kw: torch.zeros_like(x)                      # no noise: the loop becomes deterministic, so it can be compared with the UI loop exactly
    try:
        zero = torch.zeros(2, 6, model.cfg.action_dim)
        h, z = model.rssm.initial(2, CPU)
        ref = E.ui_dream_frames(model, ticks=6, n=2, noise_seed=3, init_state=(h, z))          # (6,2,3,H,W), 8-bit quantised
        fr, _ = A.ui_rollout(model, h, z, zero)
        diff = float((fr.transpose(0, 1) - ref).abs().max())
        assert diff <= 0.5 / 255 + 1e-6, f"ui_rollout differs from ivs.eyemetric.ui_dream_frames by {diff} (only the 8-bit rounding is allowed)"
    finally:
        torch.randn_like = real_randn
    fr, _ = A.ui_rollout(model, *model.rssm.initial(2, CPU), torch.zeros(2, 4, model.cfg.action_dim))
    assert fr.shape == (2, 4, 3, 96, 96) and fr.requires_grad
    fr.mean().backward()
    got = {n: p.grad is not None and float(p.grad.abs().sum()) > 0 for n, p in model.named_parameters() if p.requires_grad}
    assert all(got.values()), [n for n, v in got.items() if not v]
    assert all(p.grad is None for n, p in model.named_parameters() if n.startswith(A.FROZEN)), "frozen modules must not collect gradients"


def test_real_loss_equals_the_pilot_train_loss_with_zero_prediction_weight():
    model, _ = fresh_model()
    rc = build_run(AR.RUN, 10)
    assert rc.pred_weight == 0.0 and rc.geom_weight == 0 and rc.overshoot_weight == 0 and rc.cons_weight == 0
    gen = torch.Generator().manual_seed(0)
    aug = torch.Generator().manual_seed(1)
    inp, clean, actions = make_batch(banks()["train"], rc, gen, aug, InterventionSampler(p=rc.p_intervene), 16, CPU)
    torch.manual_seed(5)
    a, _ = train_loss(model, rc, inp, clean, actions)
    torch.manual_seed(5)
    b, m = AR.real_loss(model, rc, inp, clean, actions)
    assert abs(float(a) - float(b)) < 1e-3 * max(1.0, abs(float(a))), (float(a), float(b))
    print(f"   real-window loss {float(b):.1f} (recon {float(m['reconstruction']):.1f}, kl {float(m['kl']):.1f})")


def test_adaptive_weight_matches_the_manual_gradient_ratio_and_is_zero_without_a_gradient_path():
    w = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    l_real = (w * torch.tensor([1.0, 2.0])).sum()                                 # grad (1, 2): norm sqrt(5)
    l_adv = (w * torch.tensor([0.0, 10.0])).sum()                                 # grad (0, 10): norm 10
    lam = A.adaptive_weight(l_real, l_adv, w, scale=0.3)
    assert abs(float(lam) - 0.3 * 5 ** 0.5 / (10 + 1e-4)) < 1e-6 and not lam.requires_grad
    unrelated = torch.nn.Parameter(torch.ones(1)) * 2
    assert float(A.adaptive_weight(l_real, unrelated.sum(), w, 0.3)) == 0.0
    assert float(A.adaptive_weight(l_real, l_adv * 1e9, w, 0.3)) < 1e-3, "a huge adversarial gradient must shrink the weight"
    assert float(A.adaptive_weight(l_real, l_adv * 0 + 1e-12 * (w * 0).sum(), w, 0.3, max_weight=7.0)) <= 7.0


def test_dream_batch_uses_real_continuation_and_disc_logits_shapes():
    model, _ = fresh_model()
    A.freeze_for_stage2(model)
    rc = build_run(AR.RUN, 10)
    gen, aug = torch.Generator().manual_seed(0), torch.Generator().manual_seed(1)
    inp, clean, actions = make_batch(banks()["train"], rc, gen, aug, InterventionSampler(p=rc.p_intervene), AR.T, CPU)
    fake, real, cond = A.dream_from_clean_context(model, clean, actions, AR.CTX, AR.K)
    assert fake.shape == real.shape == (rc.batch, AR.K, 3, 96, 96) and cond.shape == (rc.batch, 3, 96, 96)
    assert torch.equal(real, clean[:, AR.CTX:]) and torch.equal(cond, clean[:, AR.CTX - 1])
    d = A.PatchDisc()
    lg = A.disc_logits(d, cond, fake, AR.TICKS)
    assert lg.shape == (rc.batch * len(AR.TICKS), 1, 12, 12), lg.shape
    assert torch.isfinite(lg).all()


def test_train_arm_end_to_end_for_both_arms_freezes_encoder_and_posterior_and_writes_loadable_checkpoints():
    ref, _ = fresh_model()
    ref_state = {k: v.clone() for k, v in ref.state_dict().items()}
    out = TMP / "s2"
    for arm in AR.ARMS:
        model, summ = AR.train_arm(arm, L1, mini_cache(), out, steps=4, max_minutes=30, batch=2, lr=1e-3, d_lr=2e-4, w_adv=0.3, device=CPU, seed=0, log_every=2, probe_every=2, banks=banks())
        assert summ["steps_done"] == 4 and summ["skipped"] == 0 and len(summ["probes"]) >= 3
        assert all(torch.isfinite(torch.tensor(v)) for pr in summ["probes"] for v in pr.values())
        ck = torch.load(out / arm / "model.pt", map_location="cpu", weights_only=False)
        assert ck["stage2"]["arm"] == arm and ck["stage2"]["step"] == 4 and "config" in ck and "run" in ck
        m2 = EyeWorldModel(ModelConfig(**ck["config"]))
        m2.load_state_dict(ck["model"])
        for k, v in ck["model"].items():
            if k.startswith(A.FROZEN):
                assert torch.equal(v, ref_state[k]), f"{arm}: frozen weight {k} changed"
        changed = [k for k, v in ck["model"].items() if k.startswith(A.TRAINABLE) and not torch.equal(v, ref_state[k])]
        assert changed, f"{arm}: no trainable weight moved"
        rows = [json.loads(line) for line in (out / arm / "train_log.jsonl").read_text().splitlines()]
        assert len(rows) == 2 and all(torch.isfinite(torch.tensor(r["real"])) for r in rows)
        if arm == "adv":
            assert all(k in rows[0] for k in ("d", "adv", "lam", "dream_mse")) and rows[0]["lam"] >= 0
        else:
            assert "d" not in rows[0]
        print(f"   {arm}: {summ['steps_done']} steps in {summ['minutes']} min, {len(changed)} trainable tensors moved, last row {rows[-1]}")
    assert (out / "adv" / "summary.json").exists()


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
