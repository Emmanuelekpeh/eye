"""Tests for ivs/refiner.py + scripts/refine_gan.py on REAL data: the L1 checkpoint, the saved S32 autoencoder (pilot_results/kaggle_r4), mini caches copied from
data/cache_hr/h96 and h128, and the real held-out clarity pack (data/heldout_pack). CPU. Run: python tests\\test_refiner.py (from ivs2)."""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
import heldout_clarity as H                       # noqa: E402
import refine_gan as RG                           # noqa: E402
from ivs.cachedata import CacheBank               # noqa: E402
from ivs.refiner import PatchDisc, Refiner, d_hinge, g_hinge, g_loss   # noqa: E402

ROOT = HERE.parent
L1 = HERE / "pilot_results" / "kaggle_long3_30k" / "pilot" / "L1_v6_long" / "model.pt"
AE = HERE / "pilot_results" / "kaggle_r4" / "r4" / "ae" / "ae.pt"
PACK = ROOT / "data" / "heldout_pack"
_TMP = tempfile.TemporaryDirectory()
TMP = Path(_TMP.name)
_S = {}


def mini(src_name, tag):
    src = ROOT / "data" / "cache_hr" / src_name
    metas = json.loads((src / "manifest.json").read_text())
    size = lambda m: (src / (H.safe(Path(m["name"]).stem) + ".npz")).stat().st_size
    v = sorted([m for m in metas if m.get("source", "video") == "video"], key=size)[:8]
    o = sorted([m for m in metas if m.get("source") == "openeds"], key=lambda m: m["name"])[:10]
    out = TMP / tag
    out.mkdir()
    for m in v + o:
        f = H.safe(Path(m["name"]).stem) + ".npz"
        shutil.copy2(src / f, out / f)
    (out / "manifest.json").write_text(json.dumps(v + o))
    return out


def bank(kind):
    if kind not in _S:
        d = mini("h96" if kind == "l1" else "h128f", f"mini_{kind}")
        _S[kind] = CacheBank(d, image_size=RG.SIZES[kind], split="train", val_frac=0.15, seed=0, tint_mode="gray", source_weights=H.SOURCE_WEIGHTS)
    return _S[kind]


def base(kind):
    if ("base", kind) not in _S:
        _S[("base", kind)] = RG.Base(kind, L1 if kind == "l1" else AE, "cpu")
    return _S[("base", kind)]


def test_refiner_is_identity_at_init_checks_sizes_and_trains_and_disc_shapes_and_hinge_values():
    g, d = Refiner(), PatchDisc()
    for s in (96, 128):
        x = torch.rand(2, 3, s, s)
        assert torch.equal(g(x), x), "zero-initialised output conv: untrained refiner must be the identity"
        assert d(x, x).shape == (2, 1, s // 8, s // 8)
    try:
        g(torch.rand(1, 3, 100, 100))
        raise AssertionError("must reject sizes that are not multiples of 8")
    except ValueError:
        pass
    assert abs(float(d_hinge(torch.tensor([2.0, 0.5]), torch.tensor([-2.0, 0.5]))) - (0.25 + 0.75)) < 1e-6      # mean(relu(1-r)) + mean(relu(1+f)) = .25 + .75
    assert float(g_hinge(torch.tensor([1.0, 3.0]))) == -2.0
    x = torch.rand(2, 3, 96, 96)
    loss, parts = g_loss(x, g(x) + 0.1, x, d=None)
    loss.backward()
    assert float(parts["adv"]) == 0.0 and float(parts["l1"]) > 0.09 and all(p.grad is not None for p in g.parameters())
    n_g, n_d = sum(p.numel() for p in g.parameters()), sum(p.numel() for p in d.parameters())
    print(f"   refiner {n_g:,} params, discriminator {n_d:,}")


def test_pair_batches_are_real_renders_of_real_frames_for_both_bases():
    for kind in ("l1", "ae"):
        b, bk = base(kind), bank(kind)
        gen = torch.Generator().manual_seed(0)
        render, target = RG.pair_batch(b, bk, 6, gen, torch.device("cpu"))
        s = RG.SIZES[kind]
        assert render.shape == target.shape == (6, 3, s, s) and 0 <= float(render.min()) and float(render.max()) <= 1 and 0 <= float(target.min()) and float(target.max()) <= 1
        err = float((render - target).abs().mean())
        assert 0.002 < err < 0.25, f"{kind}: render must resemble the real frame but not equal it (mean abs {err})"
        print(f"   {kind}: mean |render - target| = {err:.4f}")


def test_training_moves_the_refiner_toward_the_target_for_reg_and_gan_and_checkpoint_roundtrips():
    b, bk = base("l1"), bank("l1")
    for arm in RG.ARMS:
        g, curve, _ = RG.train_arm(b, bk, arm, 120, 8, 2e-3, torch.device("cpu"), 0, log_every=60)
        assert len(curve) == 2 and all(np.isfinite(r["loss"]) for r in curve) and ("d" in curve[0]) == (arm == "gan")
        gen = torch.Generator().manual_seed(123)
        render, target = RG.pair_batch(b, bk, 16, gen, torch.device("cpu"))
        with torch.no_grad():
            after, before = float((g(render) - target).abs().mean()), float((render - target).abs().mean())
        print(f"   {arm}: held-out batch L1 {before:.4f} -> {after:.4f}")
        RG.save_refiner(TMP / f"l1_{arm}.pt", g, dict(arm=arm))
        assert after < before or arm == "gan", (arm, before, after)   # a CPU smoke run only has to move the right way; the GAN arm may trade L1 for texture
        g2 = RG.load_refiner(TMP / f"l1_{arm}.pt")
        assert torch.allclose(g(render[:2]), g2(render[:2]), atol=1e-6)


def test_resume_continues_generator_and_discriminator_from_a_checkpoint():
    b, bk = base("l1"), bank("l1")
    cpu = torch.device("cpu")
    g, _, _ = RG.train_arm(b, bk, "gan", 4, 4, 1e-3, cpu, 0, log_every=4)
    assert "disc" not in g.state_dict() and not any(k.startswith("disc") for k in g.state_dict()), "the discriminator must not leak into the generator state"
    RG.save_refiner(TMP / "resume_gan.pt", g, dict(arm="gan", steps=4, steps_total=4))
    g_reg, _, _ = RG.train_arm(b, bk, "reg", 2, 4, 1e-3, cpu, 0, log_every=2)
    RG.save_refiner(TMP / "resume_reg.pt", g_reg, dict(arm="reg"))
    ck, ck_reg = RG.load_checkpoint(TMP / "resume_gan.pt"), RG.load_checkpoint(TMP / "resume_reg.pt")
    assert ck["disc"] is not None and ck_reg["disc"] is None
    RG.load_refiner(TMP / "resume_gan.pt")                                      # a checkpoint with a discriminator still loads as a plain refiner
    g2, _, _ = RG.train_arm(b, bk, "gan", 1, 4, 0.0, cpu, 1000, log_every=1, init=ck)       # lr 0: one step must leave every weight where the checkpoint had it
    for k, v in g.state_dict().items():
        assert torch.equal(v, g2.state_dict()[k]), k
    for k, v in ck["disc"].items():
        if not k.endswith(("weight_u", "weight_v")):                             # spectral-norm power-iteration buffers refresh on every forward by design
            assert torch.equal(v, g2.disc.state_dict()[k]), k
    old = {"state": ck["state"], "meta": {}}                                    # a run-1 file (generator only): the discriminator starts fresh, no error
    g3, _, _ = RG.train_arm(b, bk, "gan", 1, 4, 0.0, cpu, 1000, log_every=1, init=old)
    assert all(torch.equal(v, g3.state_dict()[k]) for k, v in g.state_dict().items())


def test_verdict_rule_follows_the_pre_registered_thresholds():
    def mk(keep, fid, psnr):
        return {s: dict(edge_keep=keep, edge_fidelity=fid, psnr=psnr) for s in ("openeds", "video_gray", "video_colour")}
    ok_gate = {"gan": {s: dict(edge_keep=True, edge_fidelity=True) for s in RG.DECISIVE}}
    sc = dict(base=mk(.40, .60, 30.0), reg=mk(.45, .62, 30.5), gan=mk(.50, .60, 29.5), _gate=ok_gate)
    v = RG.verdicts(sc)
    assert v["G1_clarity"] and v["G2_faithful"] and v["G3_attribution"]
    assert not RG.verdicts(dict(sc, gan=mk(.47, .60, 29.5)))["G1_clarity"], "+.07 over base is below the +.08 rule"
    assert not RG.verdicts(dict(sc, gan=mk(.50, .57, 29.5)))["G2_faithful"], "fidelity .03 below base is a hallucination"
    assert not RG.verdicts(dict(sc, gan=mk(.50, .60, 28.0)))["G2_faithful"], "1.5 dB PSNR rule"
    assert not RG.verdicts(dict(sc, reg=mk(.49, .62, 30.5)))["G3_attribution"], "gan only +.01 over reg"
    assert not RG.verdicts(dict(sc, _gate={"gan": {s: dict(edge_keep=False, edge_fidelity=True) for s in RG.DECISIVE}}))["G1_clarity"], "R3 gate must pass"


def test_evaluate_end_to_end_on_the_real_pack_writes_sheets_and_verdicts():
    b = base("l1")
    refs = {a: RG.load_refiner(TMP / f"l1_{a}.pt") for a in RG.ARMS}
    out = TMP / "eval"
    out.mkdir()
    res = RG.evaluate(b, refs, PACK, out, torch.device("cpu"))
    assert set(res["arms"]) == {"base", "reg", "gan"} and set(res["verdict"]) >= {"G1_clarity", "G2_faithful", "G3_attribution"} and res["eval_frames"] > 300
    for s in ("openeds", "video_gray", "video_colour"):
        assert (out / f"refine_sheet_l1_{s}.png").exists(), s
        assert np.isfinite(res["arms"]["base"][s]["metrics"]["edge_keep"])
    print("   base keep:", {s: round(v["metrics"]["edge_keep"], 3) for s, v in res["arms"]["base"].items()}, "| verdict", res["verdict"]["gain_keep_vs_base"])
    # same render seed => the un-refined arm is reproducible
    a = RG.render_recs(b, None, H.build_items(PACK, 96)[:8], torch.device("cpu"))
    c = RG.render_recs(b, None, H.build_items(PACK, 96)[:8], torch.device("cpu"))
    assert all(np.array_equal(x, y) for x, y in zip(a, c))


def test_dream_sheet_runs_on_cpu_with_the_real_l1_model():
    out = TMP / "dream"
    ns = type("A", (), dict(ckpt=str(L1), cache=[str(ROOT / "data" / "cache" / "videos"), str(ROOT / "data" / "cache" / "openeds")], refiners=str(TMP), out=str(out), worlds=2, ticks=[1, 5]))()
    RG.dream_main(ns)
    assert (out / "refine_dream_sheet.png").stat().st_size > 5000


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print("PASS", name)
            except Exception as e:                       # noqa: BLE001
                import traceback
                failed += 1
                print("FAIL", name, repr(e)); traceback.print_exc()
    sys.exit(1 if failed else 0)
