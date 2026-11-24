"""Run: python tests/test_pilot.py   (uses the REAL cache at ../data/cache/videos; small subsets, CPU)"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from ivs.actions import luma, scatter_batch
from ivs.cachedata import CLASSES, CacheBank, colour_class, split_groups_stratified
from ivs import pilot

CACHE = Path(__file__).resolve().parents[2] / "data" / "cache" / "videos"
METAS = json.loads((CACHE / "manifest.json").read_text())
SMALL = dict(image_size=64, max_videos=3, max_frames_per_video=200)


def test_split_is_group_disjoint_and_stratified():
    val = split_groups_stratified(METAS, 0.15, 0)
    train = {m["group"] for m in METAS} - val
    assert val and train and val.isdisjoint(train)
    for c in CLASSES:
        groups = {m["group"] for m in METAS if colour_class(m) == c}
        assert groups & val and groups & train, f"class {c} missing from val or train"
    assert split_groups_stratified(METAS, 0.15, 0) == val                       # deterministic
    dup = [m["group"] for m in METAS if m["name"].startswith("yt_Jwbu5") or m["name"].startswith("youtube_eye_Jwbu5")]
    assert len(set(dup)) == 1                                                  # re-downloads share a group


def test_loader_never_crosses_sequence_boundaries_and_matches_stored_frames():
    bank = CacheBank(CACHE, split="all", device="cpu", **SMALL)
    longest = bank.max_window()
    t = min(longest, 40)
    g = torch.Generator().manual_seed(0)
    for _ in range(20):
        probs, windows = bank._probabilities(t)
        b = bank.sample(4, t, generator=g, train=False)
        assert b["images"].shape == (4, t, 3, 64, 64) and 0 <= b["images"].min() and b["images"].max() <= 1
    # exact check: windows as long as the longest sequence must equal that sequence's stored (centre-cropped) frames
    seq = int(np.argmax(bank.seq_len)); s0, L = int(bank.seq_start[seq]), int(bank.seq_len[seq])
    off = (bank.store_w - bank.size) // 2
    want = bank.frames[s0:s0 + L, :, off:off + bank.size].permute(0, 3, 1, 2).float() / 255
    only = np.zeros(len(bank.seq_len), bool); only[seq] = True
    bank._probs[L] = (torch.from_numpy(only.astype(np.float64)), np.maximum(0, bank.seq_len - L + 1).astype(np.float64))
    got = bank.sample(1, L, generator=g, train=False)["images"][0]
    assert torch.equal(got, want)


def test_sampling_is_group_balanced_and_deterministic():
    bank = CacheBank(CACHE, split="all", device="cpu", **SMALL)
    t = 8
    p, _ = bank._probabilities(t)
    assert abs(float(p.sum()) - 1.0) < 1e-6
    masses = [float(p[torch.from_numpy(bank.seq_group == g)].sum()) for g in range(bank.n_groups)]
    assert max(masses) - min(masses) < 1e-9                                    # every group equally likely
    a = bank.sample(4, t, generator=torch.Generator().manual_seed(5))["images"]
    b = bank.sample(4, t, generator=torch.Generator().manual_seed(5))["images"]
    assert torch.equal(a, b)


def test_tint_mode_gray_removes_cast_but_keeps_real_colour():
    names = {m["name"]: colour_class(m) for m in METAS}
    pick = [n for n, c in names.items() if c == "tint"][:1] + [n for n, c in names.items() if c == "colour"][:1]
    sub = [m for m in METAS if m["name"] in pick]
    with tempfile.TemporaryDirectory() as r:
        r = Path(r)
        (r / "manifest.json").write_text(json.dumps(sub))
        for m in sub:
            src = CACHE / (pilot_safe(m["name"]) + ".npz")
            (r / src.name).write_bytes(src.read_bytes())
        for mode, tint_equal in (("gray", True), ("keep", False)):
            bank = CacheBank(r, split="all", tint_mode=mode, device="cpu", image_size=64, max_frames_per_video=60)
            tint_len = [v for v in bank.videos if v["cls"] == "tint"][0]["frames"]
            fr = bank.frames[:tint_len].float()
            is_gray = bool((fr[..., 0] == fr[..., 1]).all() and (fr[..., 1] == fr[..., 2]).all())
            assert is_gray == tint_equal, f"tint_mode={mode}: gray={is_gray}"
            col = bank.frames[tint_len:].float()
            assert not bool((col[..., 0] == col[..., 2]).all()), "real colour must survive tint_mode"


def pilot_safe(name):
    import re
    return re.sub(r"[^A-Za-z0-9_-]+", "_", Path(name).stem).strip("_")


def test_scatter_preserves_values_moves_whole_pixels_and_is_local():
    x = torch.rand(5, 3, 40, 56)
    y = scatter_batch(x, torch.Generator().manual_seed(1))
    for i in range(5):
        for c in range(3):
            assert torch.equal(x[i, c].flatten().sort().values, y[i, c].flatten().sort().values)
        a = {tuple(v.tolist()) for v in x[i].flatten(1).T}
        assert all(tuple(v.tolist()) in a for v in y[i].flatten(1).T)
    assert 0.02 < float((x != y).any(1).float().mean()) < 0.7
    assert torch.equal(y, scatter_batch(x, torch.Generator().manual_seed(1)))
    g1 = luma(x); assert torch.equal(g1[:, 0], g1[:, 1]) and torch.equal(luma(x[:, :1]), x[:, :1])


def test_colourise_objective_shows_gray_but_scores_against_real_colour():
    colour = [m for m in METAS if colour_class(m) == "colour"][:1]
    gray = [m for m in METAS if colour_class(m) == "gray"][:1]
    with tempfile.TemporaryDirectory() as r:
        r = Path(r)
        (r / "manifest.json").write_text(json.dumps(colour + gray))
        for m in colour + gray:
            src = CACHE / (pilot_safe(m["name"]) + ".npz")
            (r / src.name).write_bytes(src.read_bytes())
        bank = CacheBank(r, split="all", device="cpu", image_size=64, max_frames_per_video=150)
        rc = pilot.replace(pilot.RunConfig(), p_desat=1.0, p_intervene=0.0, p_scatter=0.0, batch=16)
        g = torch.Generator().manual_seed(0)
        sampler = pilot.InterventionSampler(p=0.0)
        inp, clean, _ = pilot.make_batch(bank, rc, g, torch.Generator().manual_seed(1), sampler, 8, "cpu")
        is_col = bank.sample(16, 8, generator=torch.Generator().manual_seed(0), train=True)["is_colour"]
        assert is_col.any() and (~is_col).any(), "need both colour and gray clips in the batch"
        for i in range(16):
            if is_col[i]:
                assert torch.equal(inp[i], luma(clean[i])) and pilot.chroma(clean[i]) > 0.02, "colour clip must be shown gray"
            else:
                assert torch.equal(inp[i], clean[i]), "gray clips must be untouched"
        model = pilot.EyeWorldModel(pilot.ModelConfig(image_size=64, channels=3))
        with tempfile.TemporaryDirectory() as o:
            res = pilot.eval_colour(model, rc, bank, Path(o))
            assert res["n_colour"] > 0 and all(np.isfinite(v) for v in res.values())
            assert res["chroma_target"] > 0.02 and res["mse_gray_input_vs_colour"] > 0
            assert (Path(o) / "sheet_colour.png").exists()


def test_pilot_runs_end_to_end_on_cpu_for_gray_and_rgb_with_every_feature():
    with tempfile.TemporaryDirectory() as out:
        summary = pilot.main(["--cache", str(CACHE), "--out", out, "--runs", "R0_gray_base", "R6_colourise", "R5_plasticity",
                              "--steps", "4", "--batch", "2", "--image-size", "64", "--device", "cpu",
                              "--max-videos", "3", "--max-frames-per-video", "300", "--eval-windows", "64", "--log-every", "2"])
        assert set(summary) == {"R0_gray_base", "R6_colourise", "R5_plasticity"}
        for name, m in summary.items():
            d = Path(out) / name
            for f in ("metrics.json", "train_log.jsonl", "model.pt", "sheet_noise_seeds.png", "sheet_scatter.png"):
                assert (d / f).exists(), f"{name}: missing {f}"
            for k in ("heldout", "freerun", "noise_seeds", "scatter", "colour"):
                assert k in m
            h = m["heldout"]["passive"]
            assert all(np.isfinite(v) for v in h.values()), h
            assert np.isfinite(m["scatter"]["mse_corrupt_vs_clean"]) and m["scatter"]["mse_corrupt_vs_clean"] > 0
        assert summary["R0_gray_base"]["colour"].get("skipped")                # grayscale model cannot be scored on colour
        ck = torch.load(Path(out) / "R6_colourise" / "model.pt", weights_only=True)   # plain tensors/primitives only
        assert ck["config"]["channels"] == 3 and ck["run"]["p_desat"] == 0.3


def test_new_model_and_loss_knobs_keep_defaults_and_work():
    from ivs.learning import edge_loss, _pixel_sum_loss
    base = pilot.EyeWorldModel(pilot.ModelConfig(image_size=96, channels=3))
    assert sum(p.numel() for p in base.parameters()) == 1477379, "defaults must reproduce the original model exactly"
    old_cfg = {"image_size": 64, "channels": 1, "feature_dim": 128, "deter_dim": 128, "stoch_dim": 32, "action_dim": 5,
               "min_std": 0.1}                                         # checkpoint written before the new fields existed
    pilot.EyeWorldModel(pilot.ModelConfig(**old_cfg))
    wide = pilot.EyeWorldModel(pilot.ModelConfig(image_size=96, channels=3, feature_dim=512, deter_dim=256, stoch_dim=64,
                                                 hidden_dim=256, conv_mult=2.0))
    assert sum(p.numel() for p in wide.parameters()) > 3 * sum(p.numel() for p in base.parameters())
    x = torch.rand(2, 4, 3, 96, 96)
    out = wide.observe(x, torch.zeros(2, 4, 5))
    assert out["recon"].shape == x.shape
    a, b = torch.rand(2, 3, 3, 16, 16), torch.rand(2, 3, 3, 16, 16)
    assert float(edge_loss(a, a)) == 0.0 and float(edge_loss(a, b)) > 0
    assert float(_pixel_sum_loss(a, b, "l1")) > float(_pixel_sum_loss(a, b, "smoothl1")) > 0


def test_every_ablation_preset_trains_on_cpu():
    names = [n for n in pilot.PRESETS if n.startswith("A")]
    assert len(names) == 6
    with tempfile.TemporaryDirectory() as out:
        summary = pilot.main(["--cache", str(CACHE), "--out", out, "--runs", *names, "--steps", "3", "--batch", "2",
                              "--image-size", "64", "--device", "cpu", "--max-videos", "3", "--max-frames-per-video", "300",
                              "--eval-windows", "64", "--log-every", "1", "--ckpt-every", "2"])
        for n in names:
            m = summary[n]
            assert np.isfinite(m["heldout"]["passive"]["recon"]) and m["run"]["name"] == n
            assert (Path(out) / n / "model_partial.pt").exists()
        assert summary["A5_wide"]["params"] > 3 * summary["A1_kl01"]["params"]


def test_group_frac_is_stratified_nested_and_keeps_val_unchanged():
    from ivs.cachedata import subset_groups
    metas = json.loads((CACHE / "manifest.json").read_text())
    full = split_groups_stratified(metas, 0.15, 0)
    train_metas = [m for m in metas if m["group"] not in full]
    g = lambda ms: {m["group"] for m in ms}
    half, quarter = subset_groups(train_metas, 0.5, 0), subset_groups(train_metas, 0.25, 0)
    assert g(quarter) <= g(half) <= g(train_metas) and len(g(quarter)) < len(g(half)) < len(g(train_metas))
    for cls in ("gray", "colour"):                                     # every class that exists keeps at least one group
        if any(colour_class(m) == cls for m in train_metas):
            assert any(colour_class(m) == cls for m in quarter), cls
    assert subset_groups(train_metas, 0.5, 0) == half                  # deterministic
    kw = dict(image_size=64, device="cpu", max_frames_per_video=60)
    a, b = CacheBank(CACHE, split="train", **kw), CacheBank(CACHE, split="train", group_frac=0.5, **kw)
    assert b.n_groups < a.n_groups and len(b) < len(a)
    v1, v2 = CacheBank(CACHE, split="val", hflip=False, **kw), CacheBank(CACHE, split="val", hflip=False, group_frac=1.0, **kw)
    assert torch.equal(v1.frames, v2.frames)


def test_fast_feature_presets_train_and_report_train_gap_on_cpu():
    names = [n for n in pilot.PRESETS if n.startswith("F")]
    assert len(names) == 6
    with tempfile.TemporaryDirectory() as out:
        summary = pilot.main(["--cache", str(CACHE), "--out", out, "--runs", *names, "--steps", "3", "--batch", "2",
                              "--image-size", "64", "--device", "cpu", "--max-videos", "40", "--max-frames-per-video", "60",
                              "--eval-windows", "64", "--log-every", "1"])
        for n in names:
            m = summary[n]
            assert np.isfinite(m["train_fit"]["passive"]["recon"]) and np.isfinite(m["heldout"]["passive"]["recon"])
        assert summary["F4_half_subjects"]["train_data"]["groups"] < summary["F0_base"]["train_data"]["groups"]
        assert summary["F5_quarter_subjects"]["train_data"]["groups"] < summary["F4_half_subjects"]["train_data"]["groups"]
        assert summary["F3_embed512"]["params"] > summary["F0_base"]["params"]


def test_recon_only_presets_really_drop_prediction_and_interventions():
    names = [n for n in pilot.PRESETS if n.startswith("W")]
    assert len(names) == 6
    with tempfile.TemporaryDirectory() as out:
        summary = pilot.main(["--cache", str(CACHE), "--out", out, "--runs", "W3_kl001_recon_only", "W2_kl001_nointv", "--steps", "3",
                              "--batch", "2", "--image-size", "64", "--device", "cpu", "--max-videos", "40",
                              "--max-frames-per-video", "60", "--eval-windows", "64", "--log-every", "1"])
        rows = [json.loads(l) for l in (Path(out) / "W3_kl001_recon_only" / "train_log.jsonl").read_text().splitlines()]
        # pred is still logged as a metric, but must not be in the loss: loss ~ recon + 0.01*KL (pred is the same size as recon)
        assert rows and all(r["pred"] > 0.2 * r["recon"] for r in rows)
        assert all(abs(r["loss"] - r["recon"]) < 0.1 * r["recon"] for r in rows), rows
        rc = pilot.build_run("W3_kl001_recon_only", 3)
        assert rc.pred_weight == 0.0 and rc.p_intervene == 0.0 and rc.kl_weight == 0.01
        assert all(np.isfinite(summary[n]["heldout"]["passive"]["recon"]) for n in summary)
    v = [n for n in pilot.PRESETS if n.startswith("V")]
    assert len(v) == 6
    for n in v:
        rc = pilot.build_run(n, 3)
        assert rc.kl_weight == 1.0 and rc.p_intervene == 0.5, n
        assert rc.pred_weight == (0.1 if n == "V5_bal99_wide_pred01" else 0.0), n
        assert rc.kl_balance == (0.8 if n == "V1_base" else 0.99), n
    assert pilot.build_run("V4_bal99_wide_lr1e3", 3).lr == 1e-3
    assert pilot.build_run("V6_bal99_wide_lr1e3_openeds", 3).train_sources == ("video", "openeds")
    with tempfile.TemporaryDirectory() as out:
        s = pilot.main(["--cache", str(CACHE), "--out", out, "--runs", "V3_bal99_wide", "--steps", "3", "--batch", "2", "--image-size", "64",
                        "--device", "cpu", "--max-videos", "40", "--max-frames-per-video", "60", "--eval-windows", "64", "--log-every", "1"])
        small = sum(p.numel() for p in pilot.EyeWorldModel(pilot.ModelConfig(image_size=64, channels=3)).parameters())
        assert np.isfinite(s["V3_bal99_wide"]["heldout"]["passive"]["recon"]) and s["V3_bal99_wide"]["params"] > 1.5 * small
    # the dyn term must not push on the posterior and the rep term must not push on the prior (what the balance knob relies on)
    from ivs.learning import world_model_loss
    mu = torch.zeros(2, 3, 4, requires_grad=True); sd = torch.ones(2, 3, 4)
    pm = torch.ones(2, 3, 4, requires_grad=True)
    imgs = torch.rand(2, 3, 1, 4, 4)
    out_ = dict(recon=imgs.clone(), prior=[(mu[:, t], sd[:, t]) for t in range(3)], posterior=[(pm[:, t], sd[:, t]) for t in range(3)])
    world_model_loss(imgs, out_, kl_weight=1.0, free_nats=0.0, balance=1.0)[0].backward()
    assert float(mu.grad.abs().sum()) > 0 and pm.grad is not None and float(pm.grad.abs().sum()) == 0.0, "balance=1: only the prior learns"
    mu.grad = None; pm.grad = None
    world_model_loss(imgs, out_, kl_weight=1.0, free_nats=0.0, balance=0.0)[0].backward()
    assert float(pm.grad.abs().sum()) > 0 and float(mu.grad.abs().sum()) == 0.0, "balance=0: only the posterior is regularised"
    from ivs.actions import InterventionSampler
    x = torch.rand(4, 8, 3, 32, 32)
    y, a = InterventionSampler(p=0.0)(x)
    assert torch.equal(x, y) and float(a.abs().max()) == 0.0, "p=0 must leave frames and actions untouched"


OPENEDS = CACHE.parent / "openeds"


def _mini_multisource(root, n_video=6, n_s=10):
    """Copies a few REAL npz files of each source into two temp cache dirs (+ manifests)."""
    vm = json.loads((CACHE / "manifest.json").read_text())
    sm = json.loads((OPENEDS / "manifest.json").read_text())
    dirs = []
    for name, metas, src_dir in (("videos", vm[:n_video], CACHE), ("openeds", sm[:n_s], OPENEDS)):
        d = Path(root) / name
        d.mkdir()
        for m in metas:
            f = src_dir / (pilot_safe(m["name"]) + ".npz")
            (d / f.name).write_bytes(f.read_bytes())
        (d / "manifest.json").write_text(json.dumps(metas))
        dirs.append(d)
    return dirs


def test_multi_source_split_weights_filter_and_pilot_cli():
    with tempfile.TemporaryDirectory() as root:
        dv, ds = _mini_multisource(root)
        kw = dict(image_size=64, device="cpu", max_frames_per_video=200, val_frac=0.34)
        only_v = CacheBank(dv, split="val", hflip=False, **kw)
        both_v = CacheBank([dv, ds], split="val", hflip=False, **kw)
        assert {v["group"] for v in only_v.videos} == {v["group"] for v in both_v.videos if v["source"] == "video"}, \
            "adding a source must not change another source's validation groups"
        tr_w = CacheBank([dv, ds], split="train", source_weights={"video": 0.5, "openeds": 0.5}, **kw)
        tr_u = CacheBank([dv, ds], split="train", **kw)
        def openeds_share(bank):
            probs, _ = bank._probabilities(8)
            return float(sum(probs[i] for i in range(len(bank.seq_group)) if bank.group_source[bank.seq_group[i]] == "openeds"))
        assert 0.49 < openeds_share(tr_w) < 0.51, openeds_share(tr_w)
        assert openeds_share(tr_u) > 0.55, "without weights the source with more groups must dominate (what weights fix)"
        assert set(tr_w.info()["frames_by_source"]) == {"video", "openeds"}
        f = CacheBank([dv, ds], split="val", hflip=False, sources=["openeds"], **kw)
        assert {v["source"] for v in f.videos} == {"openeds"} and f.info()["frames"] > 0
        try:
            CacheBank([dv, ds], split="train", source_weights={"nope": 1.0}, **kw)
            raise AssertionError("unknown source weight must raise")
        except ValueError:
            pass
        with tempfile.TemporaryDirectory() as out:
            s = pilot.main(["--cache", str(dv), str(ds), "--source-weights", "video=0.5,openeds=0.5", "--out", out,
                            "--runs", "F0_base", "--steps", "3", "--batch", "2", "--image-size", "64", "--device", "cpu",
                            "--max-frames-per-video", "200", "--eval-windows", "64", "--log-every", "1", "--val-frac", "0.34"])
            m = s["F0_base"]
            assert set(m["heldout_by_source"]) == {"video", "openeds"}
            assert all(np.isfinite(v["passive"]["recon"]) for v in m["heldout_by_source"].values())


MRL = CACHE.parent / "mrl"


def test_appearance_source_trains_images_and_never_enters_temporal_windows():
    from ivs.cachedata import source_kinds
    with tempfile.TemporaryDirectory() as root:
        dv, ds = _mini_multisource(root)
        dm = Path(root) / "mrl"; dm.mkdir()
        mm = json.loads((MRL / "manifest.json").read_text())[:6]
        for m in mm:
            (dm / (pilot_safe(m["name"]) + ".npz")).write_bytes((MRL / (pilot_safe(m["name"]) + ".npz")).read_bytes())
        (dm / "manifest.json").write_text(json.dumps(mm))
        assert source_kinds([dv, ds, dm]) == (["openeds", "video"], ["mrl"])
        kw = dict(image_size=64, device="cpu", max_frames_per_video=200, val_frac=0.34)
        every = CacheBank([dv, ds, dm], split="train", **kw)                       # nothing excluded: windows must still avoid mrl
        probs, _ = every._probabilities(8)
        assert all(probs[i] == 0 for i in range(len(every.seq_group)) if every.group_source[every.seq_group[i]] == "mrl")
        only = CacheBank([dv, ds, dm], split="train", sources=["mrl"], **kw)
        assert only.max_window() == 1 and only.sample(4, 1)["images"].shape == (4, 1, 3, 64, 64)
        try:
            only.sample(4, 2)
            raise AssertionError("a T=2 window across single images must be impossible")
        except RuntimeError:
            pass
        with tempfile.TemporaryDirectory() as out:
            names = ["D0_video", "D1_video_openeds", "D2_video_openeds_mrl03", "D3_video_openeds_mrl10"]
            s = pilot.main(["--cache", str(dv), str(ds), str(dm), "--source-weights", "video=0.5,openeds=0.5", "--out", out,
                            "--runs", *names, "--steps", "3", "--batch", "2", "--image-size", "64", "--device", "cpu",
                            "--max-frames-per-video", "200", "--eval-windows", "64", "--log-every", "1", "--val-frac", "0.34"])
            for n in names:
                assert set(s[n]["heldout_by_source"]) == {"video", "openeds"}
                im = s[n]["heldout_images_by_source"]["mrl"]
                assert set(im) == {"recon", "mean_image", "sharp_ratio", "n"}
                assert all(np.isfinite(v) for v in im.values()) and im["recon"] > 0 and im["mean_image"] > 0 and im["n"] >= 64
            assert set(s["D0_video"]["train_data"]["frames_by_source"]) == {"video"}
            assert set(s["D2_video_openeds_mrl03"]["train_data"]["frames_by_source"]) == {"video", "openeds"}
            log = [json.loads(l) for l in open(Path(out) / "D2_video_openeds_mrl03" / "train_log.jsonl")]
            assert all("img" in r and np.isfinite(r["img"]) for r in log)
            assert not any("img" in json.loads(l) for l in open(Path(out) / "D0_video" / "train_log.jsonl"))
        with tempfile.TemporaryDirectory() as out:                                # the long-run presets: staged, scatter, desat, images
            ln = ["L1_v6_long", "L2_v6_long_pred01"]
            s = pilot.main(["--cache", str(dv), str(ds), str(dm), "--source-weights", "video=0.5,openeds=0.5", "--out", out,
                            "--runs", *ln, "--steps", "5", "--batch", "2", "--image-size", "64", "--device", "cpu",
                            "--max-frames-per-video", "200", "--eval-windows", "64", "--log-every", "1", "--val-frac", "0.34",
                            "--ckpt-every", "2"])
            for n in ln:
                assert np.isfinite(s[n]["heldout_by_source"]["video"]["passive"]["recon"]) and s[n]["heldout_images_by_source"]["mrl"]["n"] >= 64
                log = [json.loads(l) for l in open(Path(out) / n / "train_log.jsonl")]
                assert {r["T"] for r in log} == {8, 16, 32} and all("img" in r for r in log), [r["T"] for r in log]
                assert (Path(out) / n / "model_partial.pt").exists()
            rc1, rc2 = pilot.build_run("L1_v6_long", 5), pilot.build_run("L2_v6_long_pred01", 5)
            import dataclasses
            diff = {k for k, v in dataclasses.asdict(rc1).items() if v != dataclasses.asdict(rc2)[k]}
            assert diff == {"name", "pred_weight"}, diff                              # single-factor design
        with tempfile.TemporaryDirectory() as out:                                # images requested but no appearance cache
            try:
                pilot.main(["--cache", str(dv), str(ds), "--out", out, "--runs", "D2_video_openeds_mrl03", "--steps", "2",
                            "--batch", "2", "--image-size", "64", "--device", "cpu", "--max-frames-per-video", "100"])
                raise AssertionError("image_weight>0 without an appearance source must fail loudly")
            except ValueError as e:
                assert "appearance" in str(e)


def test_decoder_refactor_keeps_params_and_plain_ae_probe_runs():
    from ivs import aeprobe
    from ivs.model import build_decoder
    base = pilot.EyeWorldModel(pilot.ModelConfig(image_size=96, channels=3))
    assert sum(p.numel() for p in base.parameters()) == 1477379
    keys = [k for k in base.state_dict() if k.startswith("decoder.")]
    twin = build_decoder(pilot.ModelConfig(image_size=96, channels=3), 160)
    assert [f"decoder.{k}" for k in twin.state_dict()] == keys, "decoder keys must match so old checkpoints still load"
    with tempfile.TemporaryDirectory() as out:
        s = aeprobe.main(["--cache", str(CACHE), "--out", out, "--variants", "AE32", "AE128_wide", "--steps", "6", "--batch", "4",
                          "--image-size", "64", "--device", "cpu", "--max-videos", "3", "--max-frames-per-video", "200",
                          "--eval-frames", "256"])
        for n in ("AE32", "AE128_wide"):
            r = s[n]
            assert all(np.isfinite(r[k][m]) for k in ("val", "train_fit") for m in ("mse", "sharp_ratio"))
            assert (Path(out) / f"{n}_sheet.png").exists() and (Path(out) / f"{n}_model.pt").exists()
        assert s["AE128_wide"]["params"] > s["AE32"]["params"]
        assert set(s["AE32"]["baselines"]) == {"mean_image", "pca32", "pca128", "pca512"}
        assert s["AE32"]["baselines"]["pca128"] < s["AE32"]["baselines"]["pca32"] < s["AE32"]["baselines"]["mean_image"]
    with tempfile.TemporaryDirectory() as out:                       # VAE / optimisation variants: KL logged, eval decodes mean z
        v = aeprobe.main(["--cache", str(CACHE), "--out", out, "--variants", "AE32_b32", "VAE32_b32", "--steps", "250", "--batch", "4",
                          "--image-size", "64", "--device", "cpu", "--max-videos", "3", "--max-frames-per-video", "200",
                          "--eval-frames", "256", "--no-pca"])
        rows = [json.loads(l) for l in (Path(out) / "VAE32_b32_train_log.jsonl").read_text().splitlines()]
        assert rows and all(np.isfinite(r["kl"]) and r["kl"] >= 0 for r in rows)
        assert "kl" not in json.loads((Path(out) / "AE32_b32_train_log.jsonl").read_text().splitlines()[0])
        assert all(np.isfinite(v[n]["val"]["mse"]) for n in v)
        ck = torch.load(Path(out) / "VAE32_b32_model.pt", map_location="cpu")
        assert any(k.startswith("head.") for k in ck["model"]) and ck["spec"]["vae"] == 1.0


def test_openeds_cache_matches_raw_pngs_and_masks():
    import cv2
    metas = json.loads((OPENEDS / "manifest.json").read_text())
    assert len(metas) == 186 and sum(m["kept"] for m in metas) == 27431
    m = metas[5]
    with np.load(OPENEDS / (pilot_safe(m["name"]) + ".npz")) as z:
        fr, b = z["frames"], z["seq_bounds"]
    assert fr.shape[1:] == (96, 128, 3) and fr.dtype == np.uint8 and b[-1] == len(fr) == m["kept"]
    assert (fr[..., 0] == fr[..., 1]).all() and (fr[..., 1] == fr[..., 2]).all(), "gray source must be stored gray"
    x0, y0, x1, y1 = m["crop_box"]
    raw = cv2.imread(str(CACHE.parents[2] / "set" / m["group"] / "0.png"), cv2.IMREAD_GRAYSCALE)[y0:y1, x0:x1]
    expect = cv2.resize(raw, (128, 96), interpolation=cv2.INTER_AREA)
    assert np.array_equal(fr[0, :, :, 0], expect), "first cached frame must equal the raw PNG crop"
    assert x0 <= m["iris"]["cx"] <= x1 and y0 <= m["iris"]["cy"] <= y1, "iris centre must be inside the crop"
    assert x1 - x0 == round((y1 - y0) * 128 / 96) or x1 - x0 == 640, "crop must keep the 4:3 aspect"


MASKS = CACHE.parent / "openeds_masks"


def test_eye_masks_align_with_frames_split_and_physics():
    from ivs import eyemetric as E
    bank = CacheBank(OPENEDS, split="val", hflip=False, sources=["openeds"])
    M = E.aligned_masks(bank, MASKS)
    assert M.shape == (len(bank), 96, 96) and set(np.unique(M).tolist()) == {0, 1, 2, 3}, "every val frame has a mask, classes 0-3 only"
    d = E.load_masked_gray(OPENEDS, MASKS)
    assert len(d["val"][0]) == len(bank) and len(d["train"][0]) + len(bank) == 27431
    assert np.array_equal(d["val"][0][:, :, 16:112], bank.frames[:, :, 8:104, 0].numpy()), "segmenter val frames must be the bank's val frames, in order"
    assert np.array_equal(d["val"][1][:, :, 16:112], M), "segmenter val masks must equal the bank-aligned masks"
    x, m, _ = E.eval_windows(bank, M, 64, 4, seed=1, need_pupil=True)
    assert x.shape == (64, 4, 3, 96, 96) and int((m == 3).sum((1, 2)).min()) >= E.MIN_PUPIL_PX
    real = E.gray(x[:, -1])
    pupil, ring = (m == 3), (m == 2)
    contrast = E._masked_mean(real, ring) - E._masked_mean(real, pupil)
    assert float(contrast.median()) > 0.03, "in real IR frames the pupil must differ from the iris ring (mask on the right pixels)"
    _, _, i1 = E.eval_windows(bank, M, 8, 4, seed=1)
    _, _, i2 = E.eval_windows(bank, M, 8, 4, seed=1)
    assert np.array_equal(i1, i2), "window choice must be deterministic for a seed"


def test_eye_metric_invariants_on_real_frames():
    from ivs import eyemetric as E
    bank = CacheBank(OPENEDS, split="val", hflip=False, sources=["openeds"], max_videos=6)
    M = E.aligned_masks(bank, MASKS)
    x, m, _ = E.eval_windows(bank, M, 48, 4, seed=2)
    real = E.gray(x[:, -1])
    s = E.gt_metrics(real, real, m)
    assert s["mse_known"] == 0 and abs(s["pupil_edge_grad_ratio"] - 1) < 1e-6 and abs(s["pupil_contrast_ratio_median"] - 1) < 1e-6
    b1, b2 = (E.gt_metrics(real, E.gaussian_blur(real, sg), m) for sg in (1.0, 2.0))
    assert 1 > b1["pupil_edge_grad_ratio"] > b2["pupil_edge_grad_ratio"] and b1["pupil_edge_band_mse"] < b2["pupil_edge_band_mse"], "blur must lower edge strength"
    assert b1["pupil_contrast_ratio_median"] > b2["pupil_contrast_ratio_median"]
    flat = E.gt_metrics(real, real.mean(0, keepdim=True).expand_as(real), m)
    assert flat["frac_pupil_washed_out"] > 0.8 and flat["pupil_contrast_ratio_median"] < 0.5, "a mean image has no pupil"
    sm = E.seg_metrics(m, m)
    assert sm["iou_pupil"] == 1.0 and sm["pupil_found_rate"] == 1.0 and sm["pupil_hallucinated_rate"] == 0.0
    st = E.structure_stats(m)
    assert st["nested_eye"] > 0.85 and st["pupil_present"] > 0.85, "ground-truth eyes must read as nested pupil-in-iris"
    z = E.structure_stats(torch.zeros(4, 96, 96, dtype=torch.uint8))
    assert z["pupil_present"] == 0 and z["nested_eye"] == 0 and z["pupil_area_median"] is None
    shifted = torch.roll(m, 8, dims=2)
    assert E.seg_metrics(shifted, m)["pupil_centroid_err_px_median"] > 6, "centroid error must see an 8 px shift"


def test_eye_segmenter_trains_roundtrips_and_loaders_work():
    from ivs import eyemetric as E
    from ivs.config import ModelConfig
    from ivs.model import EyeWorldModel
    from dataclasses import asdict
    sm = json.loads((OPENEDS / "manifest.json").read_text())[:3]
    G, Mk = [], []
    for mt in sm:
        with np.load(OPENEDS / (mt["group"] + ".npz")) as z:
            G.append(z["frames"][..., 0])
        with np.load(MASKS / (mt["group"] + ".npz")) as z:
            Mk.append(z["masks"])
    data = {"train": (np.concatenate(G), np.concatenate(Mk))}
    logs = []
    seg = E.train_segmenter(data, steps=30, batch=8, log=logs.append)
    assert len(logs) >= 2 and all(np.isfinite(float(l.rsplit(" ", 1)[1])) for l in logs)
    g = torch.from_numpy(data["train"][0][:5, :, 16:112]).float() / 255
    p = E.segment(seg, g)
    assert p.shape == (5, 96, 96) and p.dtype == torch.uint8 and int(p.max()) <= 3
    with tempfile.TemporaryDirectory() as root:
        E.save_segmenter(seg, Path(root) / "seg.pt")
        assert torch.equal(E.segment(E.load_segmenter(Path(root) / "seg.pt"), g), p)
        cfg = ModelConfig(image_size=96, channels=3)
        model = EyeWorldModel(cfg)
        torch.save({"model": model.state_dict(), "config": asdict(cfg)}, Path(root) / "wm.pt")
        fn = E.load_recon_fn("wm", Path(root) / "wm.pt", 96)
        x = torch.rand(2, 4, 3, 96, 96)
        r = fn(x)
        assert r.shape == (2, 3, 96, 96) and torch.equal(r, fn(x)), "wm recon must be deterministic for a window"
        fr = E.dream_frames(fn.model, times=(0, 3, 6), n=2)
        assert sorted(fr) == [0, 3, 6] and all(f.shape == (2, 3, 96, 96) and 0 <= float(f.min()) and float(f.max()) <= 1 for f in fr.values())
        cg = E.colour_mask(p[0].numpy())
        assert cg.shape == (3, 96, 96) and torch.isfinite(cg).all()


def test_openeds_mask_builder_reproduces_the_stored_mask_cache_and_refuses_misaligned_frames():
    import importlib
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    bc = importlib.import_module("build_cache_openeds")
    meta = json.loads((OPENEDS / "manifest.json").read_text())[7]
    with tempfile.TemporaryDirectory() as root:
        name, status, info = bc.process_masks((meta, str(OPENEDS), root, str(CACHE.parents[2] / "set"), 0.30))
        assert status == "ok", status
        with np.load(Path(root) / (meta["group"] + ".npz")) as a, np.load(MASKS / (meta["group"] + ".npz")) as b:
            assert np.array_equal(a["masks"], b["masks"]) and np.array_equal(a["seq_bounds"], b["seq_bounds"])
        bad = dict(meta, crop_box=[meta["crop_box"][0] + 3] + meta["crop_box"][1:])
        _, status2, _ = bc.process_masks((bad, str(OPENEDS), root, str(CACHE.parents[2] / "set"), 0.30))
        assert status2.startswith("MISMATCH"), "a shifted crop must be refused, never written"


def _all_group_masks():
    out = {}
    for m in json.loads((OPENEDS / "manifest.json").read_text()):
        with np.load(MASKS / (m["group"] + ".npz")) as z:
            out[m["group"]] = (z["masks"], z["seq_bounds"])
    return out


def test_gaze_geometry_and_command_from_real_masks():
    from ivs import gaze
    groups = _all_group_masks()
    M, b = groups["S_5"]
    g, v = gaze.frame_geometry(M)
    assert g.shape == (len(M), 7) and v.shape == g.shape
    for k in (0, 40, 100):                                   # independent centroid / area from the raw pixels
        ys, xs = np.nonzero(M[k] == 3)
        if len(xs) >= gaze.MIN_PUPIL_PX:
            assert abs(g[k, 0] - xs.mean()) < 1e-3 and abs(g[k, 1] - ys.mean()) < 1e-3 and abs(g[k, 2] - np.sqrt(len(xs) / np.pi)) < 1e-3
    vis = v[:, 0] & v[:, 3]
    assert vis.mean() > 0.9 and (g[vis, 6] > g[vis, 5]).mean() > 0.95 and (g[vis, 5] > g[vis, 2]).mean() > 0.99, "opening > iris > pupil radius"
    changes, frames, dev = 0, 0, []
    for name, (Mk, bd) in groups.items():
        lab = gaze.group_labels(Mk, bd)
        assert lab["has_cmd"].mean() > 0.9, name
        for a, e in zip(bd[:-1], bd[1:]):
            c, vis_k, hs = lab["cmd"][a:e], lab["valid"][a:e, 0], lab["has_cmd"][a:e]
            if hs.all():
                changes += int((np.abs(np.diff(c, axis=0)).sum(1) > 0).sum()); frames += e - a
                dev.append(np.hypot(*(c - lab["geom"][a:e, :2]).T)[vis_k])
    dev = np.concatenate(dev)
    assert 0.1 * frames < changes < 0.35 * frames, f"command is a held target, not a raw trace ({changes} changes in {frames} frames; measured ~0.23)"
    assert np.percentile(dev, 90) < 2.0 and np.percentile(dev, 99) < 4.0, "held command stays near the real pupil"


def test_gaze_command_logic_hold_median_and_blink():
    """Logic test on a constructed trajectory (known answer): 1-frame outlier removed by the median, hold below HOLD_PX, jump at a saccade,
    blink frames keep the command, an all-invisible sequence has no command."""
    from ivs import gaze
    n = 60
    cx = np.full(n, 10.0); cy = np.full(n, 20.0)
    cx[5] = 40.0                                              # one-frame jitter outlier
    cx[10:] += 0.4                                            # drift below HOLD_PX: must not move the command
    cx[30:] = 30.0                                            # saccade
    vis = np.ones(n, bool); vis[40:44] = False; cx[40:44] = 999.0   # blink: garbage centroid while the eye is shut
    c, has = gaze.clean_command(cx, cy, vis)
    assert has.all() and np.allclose(c[:30, 0], 10.0) and np.allclose(c[:30, 1], 20.0), c[:12]
    assert np.allclose(c[30:, 0], 30.0) and np.allclose(c[30:, 1], 20.0), c[28:34]
    c2, has2 = gaze.clean_command(cx, cy, np.zeros(n, bool))
    assert not has2.any() and not c2.any()


def _centroid_px(mask):
    ys, xs = np.nonzero(mask)
    return xs.mean(), ys.mean(), len(xs)


def test_cachebank_labels_follow_crop_flip_and_scale():
    import cv2
    groups = _all_group_masks()
    kw = dict(split="train", sources=["openeds"], max_videos=5, max_frames_per_video=120, labels_dir=MASKS)
    bank = CacheBank(OPENEDS, image_size=96, hflip=True, **kw)
    plain = CacheBank(OPENEDS, image_size=96, hflip=True, **{k: v for k, v in kw.items() if k != "labels_dir"})
    assert bank.has_labels and not plain.has_labels
    g1, g2 = torch.Generator().manual_seed(4), torch.Generator().manual_seed(4)
    b = bank.sample(24, 4, generator=g1, train=True)
    p = plain.sample(24, 4, generator=g2, train=True)
    assert torch.equal(b["images"], p["images"]) and set(p) == {"images", "is_colour"}, "labels must not change pixels or RNG order"
    ends = np.cumsum([v["frames"] for v in bank.videos]); starts = ends - np.array([v["frames"] for v in bank.videos])
    n_checked = 0
    for i in range(24):
        for t in range(4):
            gi = int(b["index"][i, t]); k = int(np.searchsorted(ends, gi, side="right")); local = gi - starts[k]
            mk = groups[bank.videos[k]["group"]][0][local]
            off, fl = int(b["off"][i]), bool(b["flip"][i])
            crop = mk[:, 8:120][:, off:off + 96]
            crop = crop[:, ::-1] if fl else crop
            ys, xs = np.nonzero(mk == 3)
            if len(xs) < 12 or not ((xs >= 8 + off).all() and (xs < 8 + off + 96).all()):
                continue                                       # pupil leaves the crop: the label is still the full-eye one
            cx, cy, area = _centroid_px(crop == 3)
            lab = b["geom"][i, t].numpy()
            assert abs(lab[0] - (cx - 47.5) / 48) < 1e-4 and abs(lab[1] - (cy - 47.5) / 48) < 1e-4, (lab[:2], cx, cy, fl, off)
            assert abs(lab[2] - np.sqrt(area / np.pi) / 48) < 1e-4
            if b["has_cmd"][i, t]:
                assert abs(float(b["cmd"][i, t, 0]) - lab[0]) < 0.15 and abs(float(b["cmd"][i, t, 1]) - lab[1]) < 0.15, "command stays near the pupil"
            n_checked += 1
    assert n_checked >= 60 and bool(b["flip"].any()) and len(set(b["off"].tolist())) > 1, "test must exercise flips and different crops"
    val = CacheBank(OPENEDS, image_size=96, hflip=False, split="val", sources=["openeds"], labels_dir=MASKS)
    from ivs import eyemetric as E
    M = E.aligned_masks(val, MASKS)
    bv = val.sample(32, 2, generator=torch.Generator().manual_seed(1), train=False)
    for i in range(32):
        gi = int(bv["index"][i, -1]); ys, xs = np.nonzero(M[gi] == 3)
        if len(xs) >= 12 and bv["geom_valid"][i, -1, 0]:
            assert abs(float(bv["geom"][i, -1, 0]) - (xs.mean() - 47.5) / 48) < 1e-4, "val labels must match the aligned-mask centroid"
    small = CacheBank(OPENEDS, image_size=64, hflip=False, split="val", sources=["openeds"], max_videos=3, labels_dir=MASKS)
    bs = small.sample(16, 1, generator=torch.Generator().manual_seed(2), train=False)
    sev = [int(v["frames"]) for v in small.videos]; ends_s = np.cumsum(sev); starts_s = ends_s - np.array(sev)
    errs = []
    for i in range(16):
        if not bs["geom_valid"][i, 0, 0]:
            continue
        gi = int(bs["index"][i, 0]); k = int(np.searchsorted(ends_s, gi, side="right"))
        mk = groups[small.videos[k]["group"]][0][gi - starts_s[k]]
        rs = cv2.resize(mk, (85, 64), interpolation=cv2.INTER_NEAREST)
        sub = rs[:, (85 - small.store_w) // 2 if 85 >= small.store_w else 0:][:, :small.store_w]
        off = (small.store_w - 64) // 2
        cx, cy, _ = _centroid_px(sub[:, off:off + 64] == 3)
        errs.append(abs(float(bs["geom"][i, 0, 0]) * 32 + 31.5 - cx))
    assert errs and max(errs) < 1.5, f"64 px labels must follow the resized pixels (px error {max(errs)})"


def test_synthetic_shift_moves_a_real_pupil_exactly_as_the_labels_assume():
    from ivs.actions import InterventionSampler
    from ivs import eyemetric as E
    val = CacheBank(OPENEDS, image_size=96, hflip=False, split="val", sources=["openeds"], max_videos=4, labels_dir=MASKS)
    b = val.sample(16, 3, generator=torch.Generator().manual_seed(3), train=False)
    M = torch.from_numpy(E.aligned_masks(val, MASKS))[b["index"]]                      # B,T,H,W real masks of the sampled frames
    ind = (M == 3).float().unsqueeze(2)                                                 # pupil indicator image B,T,1,H,W
    sampler = InterventionSampler(p=1.0)
    imgs, acts, st = sampler(ind, generator=torch.Generator().manual_seed(9), return_states=True)
    ref_imgs, ref_acts = sampler(ind, generator=torch.Generator().manual_seed(9))
    assert torch.equal(imgs, ref_imgs) and torch.equal(acts, ref_acts), "return_states must not change images/actions"
    assert st.shape == (16, 3, 5) and float(st[..., :2].abs().max()) > 0.01
    sh = torch.stack([sampler._shift(ind[:, i], st[:, i], 1.0) for i in range(3)], 1)    # shift only (no gain/light)
    n_ok = 0
    for i in range(16):
        for t in range(3):
            before, after = ind[i, t, 0], sh[i, t, 0]
            if before.sum() < 12 or after.sum() < 0.9 * before.sum():
                continue                                                                # pupil pushed out of the frame
            yb, xb = torch.nonzero(before, as_tuple=True); ya, xa = torch.nonzero(after > 0.5, as_tuple=True)
            dx = (xa.float().mean() - xb.float().mean()) / 48.0; dy = (ya.float().mean() - yb.float().mean()) / 48.0
            assert abs(float(dx) - 2 * float(st[i, t, 0])) < 0.03 and abs(float(dy) - 2 * float(st[i, t, 1])) < 0.03, (float(dx), float(st[i, t, 0]))
            n_ok += 1
    assert n_ok >= 20, n_ok
    _, _, st0 = InterventionSampler(p=0.0)(ind, return_states=True)
    assert float(st0.abs().max()) == 0.0, "passive clips must report zero state"


def test_make_batch_labels_follow_the_shift_and_command_has_dropout():
    import dataclasses
    from ivs.actions import InterventionSampler
    from ivs.gaze import POS_PAIRS
    bank = CacheBank(OPENEDS, image_size=64, hflip=True, split="train", sources=["openeds"], max_videos=4, max_frames_per_video=150,
                     labels_dir=MASKS)
    base = pilot.build_run("G4_cmd_geom_prior", 3)
    rc = dataclasses.replace(base, batch=12)
    sampler = InterventionSampler(p=0.5)
    torch.manual_seed(5); gen = torch.Generator().manual_seed(2)
    inp, clean, act, aux = pilot.make_batch(bank, rc, gen, torch.Generator().manual_seed(1), sampler, 8, "cpu", with_aux=True)
    assert act.shape == (12, 8, 8) and aux["geom"].shape == (12, 8, 7) and inp.shape == (12, 8, 3, 64, 64)
    torch.manual_seed(5); gen2 = torch.Generator().manual_seed(2)
    b = bank.sample(12, 8, generator=gen2, train=True)
    _, _, states = sampler(pilot.to_channels(b["images"], 3), return_states=True)
    exp = b["geom"].clone()
    for xi, yi in POS_PAIRS:
        exp[..., xi] += 2 * states[..., 0]; exp[..., yi] += 2 * states[..., 1]
    assert torch.allclose(aux["geom"], exp, atol=1e-6), "geometry labels must move with the synthetic shift"
    assert float(states.abs().sum()) > 0 and bool((states.abs().sum((1, 2)) == 0).any()), "mix of intervened and passive clips"
    flag, cmd = act[..., 5], act[..., 6:8]
    assert set(flag.unique().tolist()) <= {0.0, 1.0} and bool((flag == 0).any()) and bool((flag == 1).any())
    assert float(cmd[flag == 0].abs().sum()) == 0.0, "no command -> zero gaze channels (zero action = 'dream alone')"
    assert torch.allclose(cmd[flag == 1], b["cmd"][flag == 1], atol=1e-6), "command is the pure gaze target, not shifted by the touch"
    assert bool((flag <= b["has_cmd"].float()).all())
    rc0 = dataclasses.replace(base, batch=12, cmd_dropout=1.0); rc1 = dataclasses.replace(base, batch=12, cmd_dropout=0.0)
    a0 = pilot.make_batch(bank, rc0, torch.Generator().manual_seed(2), torch.Generator().manual_seed(1), sampler, 8, "cpu")[2]
    a1 = pilot.make_batch(bank, rc1, torch.Generator().manual_seed(2), torch.Generator().manual_seed(1), sampler, 8, "cpu")[2]
    assert float(a0[..., 5:].abs().sum()) == 0.0 and float(a1[..., 5].sum()) > 0
    plain = pilot.build_run("V6_bal99_wide_lr1e3_openeds", 3)
    assert pilot.make_batch(bank, plain, torch.Generator().manual_seed(2), torch.Generator().manual_seed(1), sampler, 8, "cpu", with_aux=True)[3] is None
    assert pilot.make_batch(bank, plain, torch.Generator().manual_seed(2), torch.Generator().manual_seed(1), sampler, 8, "cpu")[2].shape[-1] == 5


def test_geometry_head_loss_gradients_and_checkpoint_compat():
    from dataclasses import asdict
    from ivs.config import ModelConfig
    from ivs.learning import geom_loss
    from ivs.model import EyeWorldModel
    bank = CacheBank(OPENEDS, image_size=64, hflip=False, split="train", sources=["openeds"], max_videos=4, labels_dir=MASKS)
    mean, std = bank.label_stats()
    assert mean.shape == (7,) and bool((std >= 0.01).all()) and float(std[0]) < 0.5 and 0 < float(mean[5]) < 1, (mean, std)
    b = bank.sample(8, 4, generator=torch.Generator().manual_seed(1), train=False)
    tgt, valid = b["geom"], b["geom_valid"].float()
    perfect = (tgt - mean) / std
    loss, err = geom_loss(perfect, tgt, valid, mean, std)
    assert float(loss) < 1e-6 and float(torch.nan_to_num(err).max()) < 1e-5
    junk = tgt.clone(); junk[valid == 0] = 99.0
    assert abs(float(geom_loss(perfect + 0.3, junk, valid, mean, std)[0]) - float(geom_loss(perfect + 0.3, tgt, valid, mean, std)[0])) < 1e-6, "targets on invalid entries must not matter"
    l_inv, _ = geom_loss(perfect + (1 - valid) * 5.0, tgt, valid, mean, std)
    assert float(l_inv) < 1e-6, "predictions on invalid entries must not matter"
    assert float(geom_loss(perfect + 1.0, tgt, valid, mean, std)[0]) > 1.0
    cfg = ModelConfig(image_size=64, channels=3, geom_dim=7, action_dim=8, stoch_dim=16, deter_dim=32, feature_dim=32, hidden_dim=32)
    model = EyeWorldModel(cfg)
    model.geom_mean.copy_(mean); model.geom_std.copy_(std)
    x = pilot.to_channels(b["images"], 3)
    out = model.observe(x, torch.randn(8, 4, 8))                     # non-zero actions: a zero input has zero weight gradient
    assert out["geom"].shape == (8, 4, 7) and out["geom_prior"].shape == (8, 4, 7)
    lg, _ = geom_loss(out["geom_prior"], tgt, valid, model.geom_mean, model.geom_std)
    lg.backward()
    assert float(model.geom_head[0].weight.grad.abs().sum()) > 0 and float(model.rssm.gru.weight_ih.grad[:, -8:].abs().sum()) > 0, \
        "prior-state geometry loss must reach the GRU (it is what makes the command matter)"
    assert float(model.rssm.gru.weight_ih.grad[:, :16].abs().sum()) > 0 and out["geom"].requires_grad
    assert model.geometry(out["h"][:, 0], out["z"][:, 0]).shape == (8, 7)
    plain = EyeWorldModel(ModelConfig(image_size=64, channels=3, stoch_dim=16, deter_dim=32, feature_dim=32, hidden_dim=32))
    assert not any(k.startswith("geom") for k in plain.state_dict()) and "geom_head.0.weight" in model.state_dict()
    po = plain.observe(x, torch.zeros(8, 4, 5))
    assert "geom" not in po and plain.cfg.action_dim == 5, "models without a head are unchanged"
    old = {k: v for k, v in asdict(cfg).items() if k != "geom_dim"}
    assert ModelConfig(**old).geom_dim == 0, "configs saved before this field existed still load"


def test_gaze_presets_train_eval_and_refuse_missing_labels():
    import dataclasses
    with tempfile.TemporaryDirectory() as root:
        dv, ds = _mini_multisource(root, n_video=6, n_s=10)
        names = ["G0_v6_control", "G1_geom", "G2_cmd", "G3_cmd_geom", "G4_cmd_geom_prior"]
        out = str(Path(root) / "g")
        s = pilot.main(["--cache", str(dv), str(ds), "--source-weights", "video=0.5,openeds=0.5", "--out", out, "--runs", *names,
                        "--steps", "3", "--batch", "4", "--image-size", "64", "--device", "cpu", "--max-frames-per-video", "200",
                        "--val-frac", "0.34", "--eval-windows", "64", "--log-every", "1", "--labels", str(MASKS)])
        for n in names:
            ck = torch.load(Path(out) / n / "model.pt", map_location="cpu")
            rc = pilot.build_run(n, 3)
            assert ck["config"]["action_dim"] == (8 if rc.gaze_command else 5)
            has_head = rc.geom_weight > 0 or rc.geom_prior_weight > 0
            assert ck["config"]["geom_dim"] == (7 if has_head else 0) and ("geom_std" in ck["model"]) == has_head
            g = s[n]["geometry"]
            assert ("skipped" in g) != has_head
            log = [json.loads(l) for l in open(Path(out) / n / "train_log.jsonl")]
            assert all(("geom" in r) == (rc.geom_weight > 0) and ("geom_prior" in r) == (rc.geom_prior_weight > 0) for r in log), n
            assert all(np.isfinite(r["loss"]) for r in log)
            if has_head:
                assert ("with_command" in g) == rc.gaze_command and set(g["no_command"]["geom"]) == set(pilot.GEOM_NAMES)
                assert g["no_command"]["geom"]["open_r"] is not None and g["no_command"]["n_valid"]["open_r"] > 0
                assert float(ck["model"]["geom_std"].min()) >= 0.01 and float((ck["model"]["geom_mean"] != 0).sum()) > 0
            assert np.isfinite(s[n]["heldout"]["passive"]["recon"]) and np.isfinite(s[n]["heldout"]["intervened"]["recon"])
            fr = s[n]["freerun"]
            assert "skipped" in fr or np.isfinite(fr["by_horizon"][1]["model_mse"])
        refused = False
        try:
            pilot.main(["--cache", str(dv), str(ds), "--out", str(Path(root) / "nolabels"), "--runs", "G3_cmd_geom", "--steps", "2",
                        "--batch", "4", "--image-size", "64", "--device", "cpu", "--max-frames-per-video", "200", "--val-frac", "0.34",
                        "--eval-windows", "64"])
        except ValueError as e:
            refused = "--labels" in str(e)
        assert refused, "a label-dependent preset without --labels must fail loudly, not train silently without labels"


def test_command_response_metric_separates_obeying_from_ignoring_on_real_frames():
    """Measurement test: two oracles built from REAL frames (one shifts the frame exactly as commanded, one ignores the command) must
    score gain ~1 / ~0; plus the real model path (untrained 8-dim-action net) runs and returns the full structure."""
    from ivs import eyemetric as E
    from ivs.actions import InterventionSampler
    from ivs.config import ModelConfig
    from ivs.model import EyeWorldModel
    seg_path = Path(__file__).resolve().parents[1] / "pilot_results" / "eyemetric" / "segmenter.pt"
    assert seg_path.exists(), "needs the trained eye segmenter (python scripts/eval_eye_metric.py trains and caches it)"
    seg = E.load_segmenter(seg_path)
    val = CacheBank(OPENEDS, image_size=96, hflip=False, split="val", sources=["openeds"], labels_dir=MASKS)
    M = E.aligned_masks(val, MASKS)
    x, _, pick = E.eval_windows(val, M, 24, 8, seed=7, need_pupil=True)
    lab = val.labels_at(pick[:, None] - np.arange(7, -1, -1)[None, :])
    assert lab["cmd"].shape == (24, 8, 2) and bool(lab["has_cmd"][:, -1].all())

    class Oracle:
        def __init__(self, obey):
            self.cfg, self.obey, self.sampler = ModelConfig(image_size=96, channels=3, action_dim=8), obey, InterventionSampler(p=0.0)

        def eval(self):
            return self

        def observe(self, xin, a):
            return {"h": xin.flatten(2), "z": a[..., 6:8]}                       # h[:, -1] = last real frame, z[:, -1] = its target

        def imagine(self, h, z, actions):
            frame, cmd0 = h.reshape(h.shape[0], 3, 96, 96), z
            out = []
            for i in range(actions.shape[1]):
                moved = (actions[:, i, 6:8] - cmd0) / 2.0 * actions[:, i, 5:6]    # no command (flag 0) -> no shift
                st = torch.zeros(len(frame), 5); st[:, :2] = moved if self.obey else 0.0
                out.append(self.sampler._shift(frame, st, 1.0))
            return torch.stack(out, 1), None

    good = E.command_response(Oracle(True), seg, x, lab["cmd"], lab["has_cmd"], steps=(1, 4), horizon=4)
    bad = E.command_response(Oracle(False), seg, x, lab["cmd"], lab["has_cmd"], steps=(1, 4), horizon=4)
    for s in (1, 4):
        g, b_ = good["by_step"][s], bad["by_step"][s]
        assert 0.8 < g["gain_x"] < 1.2 and 0.8 < g["gain_y"] < 1.2 and g["sign_ok_x"] > 0.9 and g["sign_ok_y"] > 0.9, g
        assert abs(b_["gain_x"]) < 0.15 and abs(b_["gain_y"]) < 0.15, b_
        assert g["track_px"] < 4.0 and g["none_track_px"] < 4.0, g          # held target = real pupil: reached with or without
    assert good["_shots"]["mx"].shape == (24, 3, 96, 96) and not torch.equal(good["_shots"]["mx"], good["_shots"]["px"])
    cfg = ModelConfig(image_size=96, channels=3, action_dim=8, stoch_dim=16, deter_dim=32, feature_dim=32, hidden_dim=32)
    real = E.command_response(EyeWorldModel(cfg), seg, x[:6], lab["cmd"][:6], lab["has_cmd"][:6], steps=(1, 2), horizon=3)
    assert set(real["by_step"]) == {1, 2} and real["n"] == 6 and "pupil_found" in real["by_step"][1]
    five = E.command_response(EyeWorldModel(ModelConfig(image_size=96, channels=3)), seg, x[:2], lab["cmd"][:2], lab["has_cmd"][:2])
    assert "skipped" in five


def test_overshoot_kl_indexing_gradients_and_short_clips():
    """overshoot_kl vs a brute-force per-start loop on real windows (deterministic prior step so both are comparable), only the dynamics
    get gradients (start states and targets are detached), and clips too short for depth 2 give zero."""
    from ivs.config import ModelConfig
    from ivs.learning import gaussian_kl, overshoot_kl
    from ivs.model import EyeWorldModel
    bank = CacheBank(OPENEDS, image_size=64, hflip=False, split="train", sources=["openeds"], max_videos=4)
    b = bank.sample(4, 7, generator=torch.Generator().manual_seed(3), train=False)
    x = pilot.to_channels(b["images"], 3)
    cfg = ModelConfig(image_size=64, channels=3, stoch_dim=16, deter_dim=32, feature_dim=32, hidden_dim=32)
    model = EyeWorldModel(cfg)
    act = torch.randn(4, 7, 5)
    torch.manual_seed(0)
    out = model.observe(x, act)
    rssm = model.rssm

    def det_step(h, z, a):                                   # same maths as imagine_step but z = prior mean (no sampling)
        h = rssm.gru(torch.cat([z, a], -1), h)
        mean, std = rssm._stats(rssm.prior(h))
        return h, mean, (mean, std)

    def brute(from_start, depth):
        t = x.shape[1]
        depth = min(depth, t - 1)
        per_d = []
        for d in range(2, depth + 1):
            kls = []
            for tau in range((-1 if from_start else 0), t):
                if tau + d > t - 1:
                    continue
                if tau < 0:
                    h, z = rssm.initial(4, x.device)
                else:
                    h, z = out["h"][:, tau].detach(), out["z"][:, tau].detach()
                for j in range(1, d + 1):
                    h, z, (mean, std) = det_step(h, z, act[:, tau + j])
                post = (out["posterior"][tau + d][0].detach(), out["posterior"][tau + d][1].detach())
                kls.append(gaussian_kl(post, (mean, std)).mean())
            per_d.append(torch.stack(kls).mean())
        return torch.stack(per_d).mean()

    rssm.imagine_step = det_step
    for from_start in (False, True):
        for depth in (2, 4, 31):
            _, raw = overshoot_kl(rssm, out, act, depth, 1.0, from_start)
            ref = brute(from_start, depth)
            assert torch.allclose(raw, ref, rtol=1e-4, atol=1e-5), (from_start, depth, float(raw), float(ref))
    assert float(overshoot_kl(rssm, out, act, 4, 1.0, True)[1]) != float(overshoot_kl(rssm, out, act, 4, 1.0, False)[1])
    del rssm.imagine_step                                     # back to the sampling step
    for p in model.parameters():
        p.grad = None
    torch.manual_seed(1)
    out = model.observe(x, act)
    loss, raw = overshoot_kl(model.rssm, out, act, 5, free_nats=0.0, from_start=True)
    assert float(raw) > 0 and torch.isfinite(loss)
    loss.backward()
    g = lambda m: sum(float(p.grad.abs().sum()) for p in m.parameters() if p.grad is not None)   # noqa: E731
    assert g(model.rssm.gru) > 0 and g(model.rssm.prior) > 0, "the dynamics must learn from overshooting"
    assert g(model.encoder) == 0 and g(model.decoder) == 0 and g(model.rssm.posterior) == 0, "start states / targets are detached"
    for t_short in (1, 2):
        o2 = model.observe(x[:, :t_short], act[:, :t_short])
        l, r = overshoot_kl(model.rssm, o2, act[:, :t_short], 31, 1.0, True)
        assert float(l) == 0.0 and float(r) == 0.0
    l3, r3 = overshoot_kl(model.rssm, model.observe(x[:, :3], act[:, :3]), act[:, :3], 31, 0.0, True)
    assert float(r3) > 0, "T=3 gives depth 2 (from the zero start)"


def test_latent_refinement_matches_observe_at_zero_steps_lowers_the_loss_and_leaves_the_model_alone():
    import torch.nn.functional as F
    from ivs.config import ModelConfig
    from ivs.model import EyeWorldModel
    bank = CacheBank(OPENEDS, image_size=64, hflip=False, split="train", sources=["openeds"], max_videos=4)
    b = bank.sample(4, 5, generator=torch.Generator().manual_seed(5), train=False)
    x = pilot.to_channels(b["images"], 3)
    cfg = ModelConfig(image_size=64, channels=3, stoch_dim=16, deter_dim=32, feature_dim=32, hidden_dim=32)
    model = EyeWorldModel(cfg).eval()
    act = torch.randn(4, 5, 5)
    with torch.no_grad():
        torch.manual_seed(2)
        ref = model.observe(x, act)
    torch.manual_seed(2)
    zero = model.observe_refined(x, act, steps=0)
    for k in ("h", "z", "recon"):
        assert torch.allclose(zero[k], ref[k], atol=1e-6), k
    assert torch.equal(zero["z"], zero["z_unrefined"])
    before = {k: v.clone() for k, v in model.state_dict().items()}
    torch.manual_seed(2)
    r = model.observe_refined(x, act, steps=15, lr=0.05, prior_weight=0.0)
    pix = lambda rec: float(F.smooth_l1_loss(rec, x, reduction="none").flatten(2).sum(-1).mean())    # noqa: E731
    assert pix(r["recon"]) < pix(zero["recon"]), (pix(r["recon"]), pix(zero["recon"]))
    assert not torch.equal(r["z"], r["z_unrefined"]) and r["recon"].shape == x.shape
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items()), "refinement must not touch the weights"
    assert all(p.grad is None for p in model.parameters()), "refinement must not write parameter gradients"
    assert not r["recon"].requires_grad and not r["z"].requires_grad
    strong = model.observe_refined(x, act, steps=15, lr=0.05, prior_weight=1e4)           # a very strong prior keeps z near the prior mean
    d = lambda o: float(((o["z"][:, 0] - o["prior"][0][0]) ** 2).mean())                   # noqa: E731
    assert d(strong) < d(r)


def test_overshoot_presets_train_log_and_change_the_loss():
    with tempfile.TemporaryDirectory() as root:
        dv, ds = _mini_multisource(root, n_video=6, n_s=10)
        base = pilot.PRESETS["L1_v6_long"]
        for n in ("O0_l1_control", "O1_overshoot", "O2_overshoot_start"):     # the real presets differ from L1 ONLY by the overshoot knobs
            assert {k: v for k, v in pilot.PRESETS[n].items() if not k.startswith("overshoot")} == base, n
        names = []
        for n in ("O0_l1_control", "O1_overshoot", "O2_overshoot_start"):     # mini cache has no appearance source: same presets, image_weight 0
            pilot.PRESETS["T" + n] = {**pilot.PRESETS[n], "image_weight": 0.0}
            names.append("T" + n)
        out = str(Path(root) / "o")
        s = pilot.main(["--cache", str(dv), str(ds), "--source-weights", "video=0.5,openeds=0.5", "--out", out, "--runs", *names,
                        "--steps", "6", "--batch", "4", "--image-size", "64", "--device", "cpu", "--max-frames-per-video", "200",
                        "--val-frac", "0.34", "--eval-windows", "64", "--log-every", "1"])
        loss = {}
        for n in names:
            rc = pilot.build_run(n, 6)
            log = [json.loads(l) for l in open(Path(out) / n / "train_log.jsonl")]
            assert all(np.isfinite(r["loss"]) for r in log), n
            assert all(("overshoot" in r) == (rc.overshoot_weight > 0) for r in log), n
            if rc.overshoot_weight > 0:
                assert all(r["overshoot"] > 0 for r in log), n
            loss[n] = [r["loss"] for r in log]
            assert np.isfinite(s[n]["heldout"]["passive"]["recon"])
        for n in names:
            del pilot.PRESETS[n]
        assert loss["TO0_l1_control"][0] != loss["TO1_overshoot"][0] and loss["TO1_overshoot"][0] != loss["TO2_overshoot_start"][0], \
            "the overshoot term must enter the loss (same seed, same first batch)"
        rc1, rc2 = pilot.build_run("O1_overshoot", 6), pilot.build_run("O2_overshoot_start", 6)
        assert not rc1.overshoot_from_start and rc2.overshoot_from_start and rc1.overshoot_depth == 31


def test_long_gaze_preset_is_l1_plus_g4_and_trains_staged_with_real_masks():
    base, g4, g5 = pilot.PRESETS["L1_v6_long"], pilot.PRESETS["G4_cmd_geom_prior"], pilot.PRESETS["G5_long_cmd_geom_prior"]
    assert {k: v for k, v in g5.items() if k not in ("gaze_command", "geom_weight", "geom_prior_weight")} == base
    assert all(g5[k] == g4[k] for k in ("gaze_command", "geom_weight", "geom_prior_weight"))
    with tempfile.TemporaryDirectory() as root:
        dv, ds = _mini_multisource(root, n_video=6, n_s=10)
        pilot.PRESETS["TG5"] = {**g5, "image_weight": 0.0}               # the mini cache has no appearance source
        try:
            out = str(Path(root) / "g5")
            s = pilot.main(["--cache", str(dv), str(ds), "--source-weights", "video=0.5,openeds=0.5", "--out", out, "--runs", "TG5",
                            "--steps", "6", "--batch", "4", "--image-size", "64", "--device", "cpu", "--max-frames-per-video", "200",
                            "--val-frac", "0.34", "--eval-windows", "64", "--log-every", "1", "--labels", str(MASKS)])
        finally:
            del pilot.PRESETS["TG5"]
        log = [json.loads(l) for l in open(Path(out) / "TG5" / "train_log.jsonl")]
        assert all(np.isfinite(r["loss"]) and "geom" in r and "geom_prior" in r for r in log)
        assert len({r["T"] for r in log}) > 1, "the staged curriculum must change the rollout length"
        ck = torch.load(Path(out) / "TG5" / "model.pt", map_location="cpu")
        assert ck["config"]["action_dim"] == 8 and ck["config"]["geom_dim"] == 7
        assert np.isfinite(s["TG5"]["heldout"]["passive"]["recon"]) and "with_command" in s["TG5"]["geometry"]


def test_ui_dream_frames_reproduce_the_ui_world_tick_for_tick_on_a_real_checkpoint():
    import base64, io
    from PIL import Image
    from ivs import eyemetric as E, ui
    from ivs.config import ModelConfig
    from ivs.model import EyeWorldModel
    ckpt = Path(__file__).resolve().parents[1] / "pilot_results" / "kaggle_gaze1_8000" / "pilot" / "G4_cmd_geom_prior" / "model.pt"
    assert ckpt.exists(), "needs the trained G4 checkpoint (action_dim 8, RGB)"
    with tempfile.TemporaryDirectory() as tmp:
        w = ui.World(ui.parse_args(["--checkpoint", str(ckpt), "--world", str(Path(tmp) / "w.pt"), "--fresh", "--seed", "21"]))
        shown = []
        for _ in range(10):
            r = w.tick({"target": [0, 0, 0, 0, 0], "dream": True})
            shown.append(np.asarray(Image.open(io.BytesIO(base64.b64decode(r["img"]["stim"].split(",", 1)[1])))))
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    m = EyeWorldModel(ModelConfig(**ck["config"])); m.load_state_dict(ck["model"])
    mine = (E.ui_dream_frames(m, ticks=10, n=1, noise_seed=21)[:, 0] * 255).round().byte().permute(0, 2, 3, 1).numpy()
    assert all(np.array_equal(a, b) for a, b in zip(shown, mine)), "the offline dream must be the UI's dream"
    many = E.ui_dream_frames(m, ticks=6, n=3, noise_seed=21)
    assert many.shape == (6, 3, 3, ck["config"]["image_size"], ck["config"]["image_size"]) and torch.isfinite(many).all()
    assert (many[:, 0] - many[:, 1]).abs().max() > 0.01, "rows of a batch must be different worlds"


def test_eye_episodes_counts_runs_after_dropping_flicker():
    from ivs import eyemetric as E
    f = np.zeros(200, bool)
    f[10:40] = True                 # one 30-tick episode
    f[60] = True                    # single-tick flicker: must vanish
    f[90:105] = True                # 15 ticks: counts as an episode, not a long one
    f[100] = False                  # one-tick gap inside it: must be bridged
    f[170:200] = True               # ends at the last tick: must be counted
    e = E.eye_episodes(f)
    assert e["n_episodes"] == 3 and e["n_long"] == 2 and e["max_len"] == 30
    assert abs(e["eye_frac"] - (30 + 15 + 30) / 200) < 1e-9 and e["median_len"] == 30
    assert E.eye_episodes(np.zeros(50, bool)) == dict(eye_frac=0.0, n_episodes=0, median_len=0.0, max_len=0, n_long=0)
    cls = torch.zeros(2, 32, 32, dtype=torch.long)
    cls[0, 10:22, 10:22] = 2; cls[0, 14:18, 14:18] = 3          # pupil centred in the iris
    cls[1, 10:22, 10:22] = 2; cls[1, 4:8, 4:8] = 3              # pupil far off-centre / outside
    assert E.nested_flags(cls).tolist() == [True, False]
    assert E.structure_stats(cls)["nested_eye"] == 0.5


def test_lr_schedules_default_unchanged_and_continuation_arms_are_l1_plus_only_lr_knobs():
    import math
    from dataclasses import replace
    old = lambda rc, s, T: rc.lr * min(1.0, (s + 1) / max(1, rc.warmup)) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / max(1, T))))
    rc = pilot.build_run("L1_v6_long", 30000)
    assert all(abs(pilot.lr_at(rc, s, 30000) - old(rc, s, 30000)) < 1e-15 for s in (0, 1, 199, 200, 5000, 29999, 30000))
    base = pilot.PRESETS["L1_v6_long"]
    arms = ("LR0_const1e4", "LR0b_const1e4_seed1", "LR1_const3e5", "LR2_cos3e4_to3e5", "LR3_cos1e3_to1e4")
    for n in arms:                      # the model, data and loss recipe must be exactly L1's (so --init-from loads and the test is about the LR only)
        assert {k: v for k, v in pilot.PRESETS[n].items() if k not in ("stages", "lr", "lr_floor", "train_seed")} == \
               {k: v for k, v in base.items() if k not in ("stages", "lr")}, n
        assert pilot.PRESETS[n]["stages"] == ((32, 1.0),)
    c = pilot.build_run("LR1_const3e5", 4000)
    assert abs(pilot.lr_at(c, 3999, 4000) - 3e-5) < 1e-12 and abs(pilot.lr_at(c, 2000, 4000) - 3e-5) < 1e-12      # constant after warmup
    assert pilot.lr_at(c, 0, 4000) < 3e-5 / 100                                                                    # but the fresh Adam still warms up
    r = pilot.build_run("LR2_cos3e4_to3e5", 4000)
    assert abs(pilot.lr_at(r, 200, 4000) - 3e-4) < 1e-5 and abs(pilot.lr_at(r, 4000, 4000) - 3e-5) < 1e-12
    assert pilot.build_run("LR0b_const1e4_seed1", 4000).train_seed == 1 and replace(c, lr_floor=0.1).lr_floor == 0.1


def test_continuation_init_from_loads_weights_rejects_other_architectures_and_logs_gradient_norms():
    from ivs.config import ModelConfig
    from ivs.model import EyeWorldModel
    with tempfile.TemporaryDirectory() as root:
        dv, ds = _mini_multisource(root, n_video=6, n_s=10)
        for n in ("LR0_const1e4", "LR2_cos3e4_to3e5"):          # the mini cache has no appearance source: same presets, image_weight 0, small clips
            pilot.PRESETS["T" + n] = {**pilot.PRESETS[n], "image_weight": 0.0, "stages": ((8, 1.0),), "warmup": 2, "stoch_dim": 8, "deter_dim": 16,
                                      "feature_dim": 32, "hidden_dim": 16}
        pilot.PRESETS["TLR0_seed1"] = {**pilot.PRESETS["TLR0_const1e4"], "train_seed": 1}
        common = ["--cache", str(dv), str(ds), "--source-weights", "video=0.5,openeds=0.5", "--steps", "30", "--batch", "4", "--image-size", "64",
                  "--device", "cpu", "--max-frames-per-video", "200", "--val-frac", "0.34", "--eval-windows", "64", "--log-every", "1"]
        first = str(Path(root) / "a")
        pilot.main([*common, "--out", first, "--runs", "TLR0_const1e4", "TLR0_seed1"])
        log = lambda out, n: [json.loads(l) for l in open(Path(out) / n / "train_log.jsonl")]
        a, a1 = log(first, "TLR0_const1e4"), log(first, "TLR0_seed1")
        assert a[0]["loss"] != a1[0]["loss"], "train_seed must change the batches"
        for r in a:                      # interval means (log_every=1 -> the mean of one step), gradient norm before clipping, clip share
            assert abs(r["recon_avg"] - r["recon"]) < 1e-3 * max(1.0, r["recon"]) and abs(r["loss_avg"] - r["loss"]) < 1e-3 * max(1.0, r["loss"])
            assert r["gnorm"] > 0 and r["gnorm_max"] >= r["gnorm"] - 1e-6 and r["gclip"] in (0.0, 1.0) and "kl_avg" in r and "pred_avg" in r
        ckpt = Path(first) / "TLR0_const1e4" / "model.pt"
        second = str(Path(root) / "b")
        pilot.main([*common, "--steps", "6", "--out", second, "--runs", "TLR0_const1e4", "TLR2_cos3e4_to3e5", "--init-from", str(ckpt)])
        fresh = str(Path(root) / "c")
        pilot.main([*common, "--steps", "6", "--out", fresh, "--runs", "TLR0_const1e4"])
        cont = log(second, "TLR0_const1e4")[0]["loss"]
        assert log(second, "TLR2_cos3e4_to3e5")[0]["loss"] == cont, "same seed + same weights -> identical first batch and loss for every arm"
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        dist = lambda p: float(sum(((v.float() - ck["model"][k].float()) ** 2).sum() for k, v in torch.load(p, map_location="cpu", weights_only=False)["model"].items()
                                   if v.dtype.is_floating_point) ** .5)
        d_cont, d_fresh = dist(Path(second) / "TLR0_const1e4" / "model.pt"), dist(Path(fresh) / "TLR0_const1e4" / "model.pt")
        n_par = sum(v.numel() for v in ck["model"].values() if v.dtype.is_floating_point)
        assert d_cont <= 1.1 * 6 * 1e-4 * n_par ** .5, f"Adam moves each weight by <= ~lr per step: 6 steps from the checkpoint cannot go further than {6e-4 * n_par ** .5:.3f} (moved {d_cont:.3f})"
        assert d_cont < 0.5 * d_fresh, f"continuing stays next to the checkpoint ({d_cont:.3f}), a fresh run does not ({d_fresh:.3f})"
        m = EyeWorldModel(ModelConfig(**ck["config"]))
        pilot.load_init(m, ModelConfig(**ck["config"]), ckpt)
        assert all(torch.equal(v, ck["model"][k]) for k, v in m.state_dict().items())
        other = ModelConfig(**{**ck["config"], "deter_dim": 24})
        try:
            pilot.load_init(EyeWorldModel(other), other, ckpt)
            raise AssertionError("an architecture mismatch must be refused")
        except ValueError as e:
            assert "deter_dim" in str(e)
        for n in ("TLR0_const1e4", "TLR2_cos3e4_to3e5", "TLR0_seed1"):
            del pilot.PRESETS[n]


def _builder(name):
    import importlib
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module(name)


def test_window_spans_are_distinct_stretches_and_fall_back_to_the_whole_video():
    bc = _builder("build_cache")
    sp = bc.window_spans(10000, 4, 100)
    assert len(sp) == 4 and all(l == 100 for _, l in sp)
    starts = [s for s, _ in sp]
    assert starts == sorted(starts) and all(b - a >= 100 for a, b in zip(starts, starts[1:])), "spans must never overlap (no frame used twice)"
    assert starts[0] >= 0 and starts[-1] + 100 <= 10000
    assert [s for s, _ in sp] == [1200, 3700, 6200, 8700], "each span is centred in its quarter of the video"
    assert bc.window_spans(300, 4, 100) == [(0, None)], "a video the spans would mostly cover is read whole"
    assert bc.window_spans(10000, 0, 100) == [(0, None)], "windows=0 is the original read-everything behaviour"


def _smallest_video_meta(windows, win_len):
    metas = [m for m in json.loads((CACHE / "manifest.json").read_text()) if m["source"] == "video"]
    ok = [m for m in metas if windows * win_len * m["stride"] < 0.8 * m["total_raw"]]
    return min(ok, key=lambda m: m["total_raw"])


def test_default_video_job_reproduces_the_stored_96px_cache_bit_for_bit():
    bc = _builder("build_cache")
    m = _smallest_video_meta(3, 20)
    src = CACHE.parents[2] / "set" / m["name"]
    with tempfile.TemporaryDirectory() as root:
        name, status, meta = bc.process((str(src), root, m["cap"], 20.0, 0.30, 16, 300, False))
        assert status == "ok", status
        with np.load(Path(root) / (bc.safe(src.stem) + ".npz")) as a, np.load(CACHE / (bc.safe(src.stem) + ".npz")) as b:
            assert np.array_equal(a["frames"], b["frames"]) and np.array_equal(a["seq_bounds"], b["seq_bounds"]), "adding --height/--windows must not change the 96 px cache"
        assert meta["height"] == 96 and "windows" not in meta


def test_windowed_hires_video_cache_matches_an_independent_sequential_decode_and_never_joins_spans():
    import cv2
    bc = _builder("build_cache")
    windows, win_len, height = 3, 20, 128
    m = _smallest_video_meta(windows, win_len)
    src = CACHE.parents[2] / "set" / m["name"]
    with tempfile.TemporaryDirectory() as root:
        name, status, meta = bc.process((str(src), root, 3000, 20.0, 0.30, 16, 300, False, height, windows, win_len))
        assert status == "ok", status
        with np.load(Path(root) / (bc.safe(src.stem) + ".npz")) as z:
            fr, b = z["frames"], z["seq_bounds"]
    x0, y0, x1, y1 = m["crop_box"]
    assert meta["crop_box"] == m["crop_box"] and meta["height"] == height and meta["width"] == round(height * (x1 - x0) / (y1 - y0))
    assert fr.shape[1:] == (height, meta["width"], 3) and fr.dtype == np.uint8 and b[-1] == len(fr) == meta["kept"]
    assert 0 < len(fr) <= windows * win_len and max(np.diff(b)) <= win_len, "a sequence can never be longer than one span"
    stride = m["stride"]
    expect, cap = {}, cv2.VideoCapture(str(src))
    spans = bc.window_spans(m["total_raw"], windows, win_len * stride)
    want = {s + j * stride for s, l in spans for j in range(l // stride + (l % stride > 0))}
    i = 0
    while i <= max(want):                                           # sequential decode: no seek, so it checks the seek landed on the right frames
        ok, f = cap.read()
        if not ok:
            break
        if i in want:
            rgb = cv2.cvtColor(cv2.resize(f[y0:y1, x0:x1], (meta["width"], height), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            expect[rgb.tobytes()] = i
        i += 1
    cap.release()
    got = [expect.get(f.tobytes()) for f in fr]
    assert all(g is not None for g in got), f"{sum(g is None for g in got)}/{len(got)} stored frames are not the frames the spans should contain"
    assert len(set(got)) == len(got), "no frame may be stored twice"
    assert len(fr) >= 0.8 * len(want)


def test_openeds_hires_job_keeps_the_crop_and_default_job_reproduces_the_stored_cache():
    import cv2
    bc = _builder("build_cache_openeds")
    m = json.loads((OPENEDS / "manifest.json").read_text())[7]
    folder = CACHE.parents[2] / "set" / m["group"]
    with tempfile.TemporaryDirectory() as root:
        _, st, meta96 = bc.process((str(folder), root, 2.5, 160, 0.30, False))
        _, st2, meta128 = bc.process((str(folder), root + "_h", 2.5, 160, 0.30, False, 128))
        assert st == st2 == "ok", (st, st2)
        with np.load(Path(root) / (m["group"] + ".npz")) as a, np.load(OPENEDS / (m["group"] + ".npz")) as b, np.load(Path(root + "_h") / (m["group"] + ".npz")) as h:
            assert np.array_equal(a["frames"], b["frames"]) and np.array_equal(a["seq_bounds"], b["seq_bounds"]), "default job must reproduce the stored 96 px cache"
            fh = h["frames"]
            assert np.array_equal(h["seq_bounds"], b["seq_bounds"]), "the same frames survive the cut/bad-frame filter at every height"
    assert meta128["crop_box"] == meta96["crop_box"] == m["crop_box"], "every height must see the same native pixels"
    assert fh.shape[1:] == (128, 171, 3) and meta128["width"] == 171 and meta128["height"] == 128
    assert (fh[..., 0] == fh[..., 1]).all() and (fh[..., 1] == fh[..., 2]).all()
    x0, y0, x1, y1 = m["crop_box"]
    raw = cv2.imread(str(folder / "0.png"), cv2.IMREAD_GRAYSCALE)[y0:y1, x0:x1]
    assert np.array_equal(fh[0, :, :, 0], cv2.resize(raw, (171, 128), interpolation=cv2.INTER_AREA))
    import shutil
    shutil.rmtree(root + "_h", ignore_errors=True)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print("ok", fn.__name__)
    print("all pilot tests passed")
