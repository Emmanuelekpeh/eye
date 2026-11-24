"""Tests for scripts/heldout_clarity.py on REAL data: the 128 px caches in data/cache_hr (v128, o128) and the raw videos / PNGs in set/. CPU.
Run: python tests\\test_heldout.py (from ivs2; needs the caches built with build_cache.py / build_cache_openeds.py --height 128 into data/cache_hr)."""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
import heldout_clarity as H           # noqa: E402
from ivs.cachedata import CacheBank, colour_class   # noqa: E402

ROOT = HERE.parent
HR, SET = ROOT / "data" / "cache_hr", ROOT / "set"
HEIGHT = 128
_TMP = tempfile.TemporaryDirectory()
TMP = Path(_TMP.name)


def _manifest(d):
    return json.loads((HR / d / "manifest.json").read_text())


def _entries():
    """Real entries: a 4:3 video (stored 171 wide), the widest video, a colour one, an OpenEDS sequence. Every cached frame is wider than the store width, so each view is a centre crop."""
    vids, openeds = _manifest(f"v{HEIGHT}"), _manifest(f"o{HEIGHT}")
    narrow = next(m for m in vids if m["width"] == 171 and colour_class(m) != "colour")
    wide = max(vids, key=lambda m: m["width"])
    colour = next(m for m in vids if colour_class(m) == "colour" and m["name"] not in (narrow["name"], wide["name"]))
    out, seen = [], set()
    for m, d in ((narrow, "v"), (wide, "v"), (colour, "v"), (openeds[7], "o")):
        if m["name"] not in seen:
            seen.add(m["name"]); out.append((m, f"{d}{HEIGHT}"))
    return out


def _frames(m, d):
    with np.load(HR / d / (H.safe(Path(m["name"]).stem) + ".npz")) as z:
        return z["frames"], z["seq_bounds"]


def _natives(m, d, k=3):
    frames, bounds = _frames(m, d)
    got = H.openeds_natives(SET, m, frames, k) if m["source"] == "openeds" else H.video_natives(SET, m, frames, k)
    return frames, bounds, got


def _mini_cache(entries, out):
    out.mkdir(parents=True, exist_ok=True)
    metas = []
    for m, d in entries:
        f = H.safe(Path(m["name"]).stem) + ".npz"
        shutil.copy2(HR / d / f, out / f)
        metas.append(m)
    (out / "manifest.json").write_text(json.dumps(metas))
    return out


def test_hires_manifests_keep_the_96px_held_out_split_crops_and_classes():
    base = {"v": json.loads((ROOT / "data" / "cache" / "videos" / "manifest.json").read_text()), "o": json.loads((ROOT / "data" / "cache" / "openeds" / "manifest.json").read_text())}
    from ivs.cachedata import split_groups_stratified
    for kind, hs in (("v", (96, 128, 192)), ("o", (128, 192))):
        want_val = split_groups_stratified(base[kind], H.VAL_FRAC, H.SEED)
        ref = {m["name"]: m for m in base[kind]}
        for h in hs:
            new = _manifest(f"{kind}{h}")
            assert {m["name"] for m in new} == set(ref), f"{kind}{h}: different entries"
            assert split_groups_stratified(new, H.VAL_FRAC, H.SEED) == want_val, f"{kind}{h}: the held-out split changed"
            for m in new:
                r = ref[m["name"]]
                assert m["group"] == r["group"] and m["crop_box"] == r["crop_box"] and colour_class(m) == colour_class(r) and m["height"] == h, (kind, h, m["name"])
                if kind == "o":
                    assert m["kept"] == r["kept"] and m["sequences"] == r["sequences"], "OpenEDS keeps the same frames at every height"
    assert len(want_val) >= 14


def test_view_geometry_covers_padding_and_cropping():
    store_w, o, j0, j1 = H.view_geometry(128, 128)
    assert (store_w, o, j0, j1) == (149, 0, 0, 128), "a square frame is shown whole"
    store_w, o, j0, j1 = H.view_geometry(100, 128)
    assert o == 10 - (149 - 100) // 2 and j0 == -o and j1 == 100 - o and 0 < j1 - j0 < 128, "a narrow frame is edge-padded on both sides: only the middle is real"
    store_w, o, j0, j1 = H.view_geometry(171, 128)
    assert (j0, j1) == (0, 128) and o == (171 - 149) // 2 + 10, "a wide frame is centre-cropped"
    assert H.native_cols(0, 0, 128, 128, 700) == (0, 700)
    a, b = H.native_cols(21, 0, 128, 171, 400)
    assert (a, b) == (round(21 * 400 / 171), round(149 * 400 / 171))


def test_eval_view_is_bit_identical_to_the_cachebank_view_and_the_native_columns_line_up():
    seen_crop = False
    for m, d in _entries():
        frames, bounds, got = _natives(m, d)
        assert len(got) >= 3, f"{m['name']}: natives not found again"
        mini = _mini_cache([(m, d)], TMP / f"bank_{d}_{H.safe(m['name'])}")
        bank = CacheBank(mini, image_size=HEIGHT, split="all", hflip=False, tint_mode="gray")
        for i, nat, mk in got:
            view, j0, j1, o, w = H.eval_view(nat, HEIGHT, colour_class(m))
            assert np.array_equal(H.stored_frame(nat, HEIGHT), frames[i]), "native crop must reproduce the cache frame"
            seq = int(np.searchsorted(bounds, i, side="right")) - 1
            ref = (bank.window_at(seq, i - int(bounds[seq]), 1)[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            assert np.array_equal(view, ref), f"{m['name']} frame {i}: eval view differs from what CacheBank shows"
            a, b = H.native_cols(o, j0, j1, w, nat.shape[1])
            seen_crop |= o > 0 and (j0, j1) == (0, HEIGHT)
            real = H.O.gray_of(cv2.resize(nat[:, a:b], (j1 - j0, HEIGHT), interpolation=cv2.INTER_AREA)) if nat.ndim == 3 else cv2.resize(nat[:, a:b], (j1 - j0, HEIGHT), interpolation=cv2.INTER_AREA)
            diff = np.abs(real.astype(float) - H.O.gray_of(view[:, j0:j1]).astype(float)).mean()
            assert diff < 5.0, f"{m['name']}: native columns [{a},{b}) do not cover the view's pixels (mean abs diff {diff:.1f})"
            if mk is not None:
                assert mk.shape == nat.shape[:2]
    assert seen_crop, "the test entries must exercise the cropped view"


def test_padded_view_geometry_on_a_real_frame_cropped_narrow():
    """No cached frame is narrow, but the code handles it (edge padding): a real frame cut to 70% width must give a view whose real columns line up with the native pixels."""
    m, d = _entries()[0]
    _, _, got = _natives(m, d, 1)
    nat = got[0][1]
    nat = nat[:, :int(nat.shape[0] * 0.7)]
    view, j0, j1, o, w = H.eval_view(nat, HEIGHT, colour_class(m))
    assert w < H.view_geometry(w, HEIGHT)[0] and j0 > 0 and j1 < HEIGHT and j1 - j0 == w, "narrow frame: padded both sides, the middle j1-j0 columns are real"
    a, b = H.native_cols(o, j0, j1, w, nat.shape[1])
    assert (a, b) == (0, nat.shape[1])
    real = cv2.resize(nat, (j1 - j0, HEIGHT), interpolation=cv2.INTER_AREA)
    assert np.abs(H.O.gray_of(real).astype(float) - H.O.gray_of(view[:, j0:j1]).astype(float)).mean() < 1.0
    assert np.array_equal(view[:, 0], view[:, j0]) and np.array_equal(view[:, -1], view[:, j1 - 1]), "outside the real columns the view is the replicated edge"


def _pack_from(entries, out, k=3):
    out.mkdir(parents=True, exist_ok=True)
    arrays, meta = {}, []
    for m, d in entries:
        _, _, got = _natives(m, d, k)
        for i, nat, mk in got:
            n = len(meta)
            arrays[f"n{n}"] = nat
            if mk is not None:
                arrays[f"m{n}"] = mk
            meta.append(dict(i=n, name=m["name"], group=m["group"], source=m["source"], cls=colour_class(m), slice=H.slice_of(m), stored_idx=i, crop_hw=list(nat.shape[:2])))
    np.savez_compressed(out / "eval_native.npz", **arrays)
    (out / "eval_meta.json").write_text(json.dumps(meta))
    return out


def test_scoring_ranks_ceiling_above_blur_and_gate_follows_the_stated_thresholds():
    pack = _pack_from(_entries(), TMP / "pack_a")
    items = H.build_items(pack, HEIGHT)
    assert {it["slice"] for it in items} == set(H.SLICES), "pad/crop/colour/openeds entries must give all three slices"
    views = [it["view"] for it in items]
    ceil = H.score_items(items, views)
    blur = H.score_items(items, [cv2.GaussianBlur(v, (0, 0), 2.0) for v in views])
    for s in H.SLICES:
        assert ceil[s]["n_frames"] == sum(1 for it in items if it["slice"] == s) and 0.2 < ceil[s]["edge_keep"] < 1.05 and ceil[s]["psnr"] > 15, (s, ceil[s])
        assert blur[s]["edge_keep"] < ceil[s]["edge_keep"] - 0.05 and blur[s]["edge_fidelity"] < ceil[s]["edge_fidelity"] and blur[s]["psnr"] < ceil[s]["psnr"], (s, blur[s], ceil[s])
    assert "pupil_edge_grad_ratio" in ceil["openeds"] and "pupil_edge_grad_ratio" not in ceil["video_colour"]
    assert blur["openeds"]["pupil_edge_grad_ratio"] < ceil["openeds"]["pupil_edge_grad_ratio"]
    g = H.gate_of(blur["openeds"], ceil["openeds"])
    assert g["edge_keep"] == bool(H.GATE * ceil["openeds"]["edge_keep"] <= blur["openeds"]["edge_keep"] <= H.KEEP_MAX)
    assert g["edge_fidelity"] == bool(blur["openeds"]["edge_fidelity"] >= H.GATE * ceil["openeds"]["edge_fidelity"]) and "pupil_edge" in g
    assert all(H.gate_of(ceil[s], ceil[s]).values()), "the ceiling itself passes its own gate"
    over = dict(ceil["video_gray"], edge_keep=1.2)
    assert not H.gate_of(over, ceil["video_gray"])["edge_keep"], "over-sharpening beyond 1.05 fails"


def test_sheet_patch_stays_in_the_central_half_of_real_frames():
    pack = _pack_from(_entries(), TMP / "pack_c")
    for it in H.build_items(pack, HEIGHT):
        g = it["native_g"]
        s = min(H.C.PATCH, min(g.shape) // 2)
        y, x, size = H.central_patch(g, s)
        assert size == s and g.shape[0] // 4 <= y and y + size <= g.shape[0] - g.shape[0] // 4 and g.shape[1] // 4 <= x and x + size <= g.shape[1] - g.shape[1] // 4, (it["name"], g.shape, (y, x, size))


def test_merge_refuses_duplicates_and_cpu_run_writes_results_sheets_and_summary():
    ents = _entries()
    vid = _mini_cache([e for e in ents if e[1].startswith("v")], TMP / "mini_v")
    op = _mini_cache([e for e in ents if e[1].startswith("o")], TMP / "mini_o")
    n, frames = H.merge_dirs([vid, op], TMP / "merged")
    assert n == len(ents) and frames == sum(m["kept"] for m, _ in ents)
    try:
        H.merge_dirs([vid, vid], TMP / "merged_dup")
        raise AssertionError("duplicate entries must be refused")
    except ValueError:
        pass
    pack = _pack_from(ents, TMP / "pack_b")
    out = TMP / "run"
    out.mkdir()
    res = H.run_height(TMP / "merged", pack, HEIGHT, out, steps=12, batch=8, lr=1e-3, device=torch.device("cpu"), seed=0, probe_every=6)
    mm = json.loads((TMP / "merged" / "manifest.json").read_text())
    val = H.val_groups_of(mm)
    assert res["train_frames"] == sum(m["kept"] for m in mm if m["group"] not in val) < frames, "training excludes the held-out groups"
    assert len(res["curve"]) == 2 and all(np.isfinite(c["loss"]) and "openeds/edge_keep" in c for c in res["curve"])
    assert set(res["slices"]) == set(H.SLICES)
    for s, v in res["slices"].items():
        assert v["passed"] == all(v["gate"].values()) and v["ae_over_ceiling_keep"] == v["ae"]["edge_keep"] / v["ceiling"]["edge_keep"]
        assert (out / f"sheet_{s}_h{HEIGHT}.png").exists()
    img = cv2.imread(str(out / "sheet_openeds_h128.png"))
    assert img.shape == (200, 1200, 3) and img.std() > 10, "2 frames x (native | ceiling | AE) tiles of 200 px"
    # two heights -> summary: gain over the smallest height and the stacked sheets
    (out / "r.json").write_text(json.dumps({"96": res, "128": res}))
    shutil.copy(out / "sheet_openeds_h128.png", out / "sheet_openeds_h96.png")
    rows = H.summary_main(type("A", (), dict(results=[str(out / "r.json")], out=str(out)))())   # sheets are read from --out, where the kernel merges the parts
    assert len(rows) == 2 * len(H.SLICES) and all(r["keep_gain_vs_smallest"] is None for r in rows if r["height"] == 96)
    assert all(r["keep_gain_vs_smallest"] == 0 and r["worth_it"] is False for r in rows if r["height"] == 128)
    stacked = cv2.imread(str(out / "sheet_openeds.png"))
    assert stacked.shape == (400, 1200, 3), "rows are stacked by height"
    # R3b arms: shared context, arm-aware keys / sheets / curve parts, gain over the baseline, WORKS verdicts, captioned arm sheets
    ctx = H.prepare_height(TMP / "merged", pack, HEIGHT)
    arms_out = TMP / "arms"
    arms_out.mkdir()
    r_ssim = H.run_height(TMP / "merged", pack, HEIGHT, arms_out, steps=12, batch=8, lr=1e-3, device=torch.device("cpu"), seed=0, probe_every=6, arm="S8_ssim", ctx=ctx)
    r_s1 = H.run_height(TMP / "merged", pack, HEIGHT, arms_out, steps=12, batch=8, lr=1e-3, device=torch.device("cpu"), seed=0, probe_every=6, arm="AE512_s1", ctx=ctx)
    assert r_ssim["latent_numbers"] == 8 * 8 * 8 and r_s1["latent_numbers"] == 512 and r_s1["seed"] == 1 and r_ssim["seed"] == 0
    assert all(np.isfinite(c["ssim_term"]) and 0 < c["one_minus_ssim"] < 1 and c["loss"] > c["pixel"] for c in r_ssim["curve"]) and "ssim_term" not in r_s1["curve"][0]
    assert (arms_out / "sheet_openeds_S8_ssim_h128.png").exists() and (arms_out / "sheet_openeds_AE512_s1_h128.png").exists()
    (arms_out / "r.json").write_text(json.dumps({"128": res, "S8_ssim@128": r_ssim, "AE512_s1@128": r_s1}))
    rows = H.summary_main(type("A", (), dict(results=[str(arms_out / "r.json")], out=str(arms_out), baseline_sheets=[str(out)]))())
    mine = [r for r in rows if r["arm"] == "S8_ssim"]
    assert len(mine) == len(H.SLICES) and all(abs(r["keep_gain_vs_base"] - (r["ae_keep"] - res["slices"][r["slice"]]["ae"]["edge_keep"])) < 1e-9 for r in mine)
    assert all(r["keep_gain_vs_base"] is None for r in rows if r["arm"] == "AE512")
    verdicts = json.loads((arms_out / "verdicts.json").read_text())
    assert set(verdicts) == {"S8_ssim@128"}, "the baseline and the seed repeat get no verdict"
    cmp_sheet = cv2.imread(str(arms_out / "sheet_openeds_arms_h128.png"))
    assert cmp_sheet.shape == (3 * 220, 1200, 3), "baseline (taken from the baseline dir) + the two arms, captioned"
    # R3c: steps_x multiplies steps and probe interval, a new base arm, cache routing in train_main (arm cache="f" -> key '128f')
    r_x3 = H.run_height(TMP / "merged", pack, HEIGHT, arms_out, steps=4, batch=8, lr=1e-3, device=torch.device("cpu"), seed=0, probe_every=2, arm="S16_ssim_x3", ctx=ctx)
    assert r_x3["steps"] == 12 and [c["step"] for c in r_x3["curve"]] == [6, 12] and r_x3["latent_numbers"] == 1024
    for arm, n in (("V1024_ssim", 1024), ("S32_ssim", 2048), ("F8_ssim", 2048)):
        r = H.run_height(TMP / "merged", pack, HEIGHT, arms_out, steps=2, batch=8, lr=1e-3, device=torch.device("cpu"), seed=0, probe_every=2, arm=arm, ctx=ctx)
        assert r["latent_numbers"] == n and (arms_out / f"sheet_openeds_{arm}_h128.png").exists(), arm
    (arms_out / "r2.json").write_text(json.dumps({"S8_ssim@128": r_ssim, "AE512_s1@128": r_s1, "S16_ssim_x3@128": r_x3}))
    rows2 = H.summarise(json.loads((arms_out / "r2.json").read_text()), base_arm="S8_ssim")
    assert all(r["keep_gain_vs_base"] is None for r in rows2 if r["arm"] == "S8_ssim") and all(r["keep_gain_vs_base"] is not None for r in rows2 if r["arm"] != "S8_ssim")
    assert set(H.works_verdicts(rows2, keep=0.03, noise_floor=0.005)) == {"S16_ssim_x3@128"}
    args = type("A", (), dict(out=str(TMP / "tm"), cache=[f"128={TMP / 'merged'}", f"128f={TMP / 'merged'}"], device="cpu", steps=2, batch=8, lr=1e-3, seed=0, probe_every=2,
                              heights=[128], pack=str(pack), arms=["S16_ssim_full", "S8", "S16_ssim_full"]))()
    tm = H.train_main(args)
    assert set(tm) == {"S16_ssim_full@128", "S8@128"} and (TMP / "tm" / "sheet_openeds_S16_ssim_full_h128.png").exists()
    args.cache, args.arms, args.out = [f"128={TMP / 'merged'}"], ["S16_ssim_full"], str(TMP / "tm2")
    try:
        H.train_main(args)
        raise AssertionError("an arm with cache='f' needs the 128f dataset")
    except KeyError:
        pass


def test_works_rule_needs_gain_on_openeds_and_gray_above_seed_noise_without_losing_fidelity():
    def rows(gains, fids, noise):
        r = []
        for arm, g, f in (("A", gains, fids), ("AE512_s1", noise, {s: 0.0 for s in H.SLICES})):
            for s in H.SLICES:
                r.append(dict(arm=arm, height=128, slice=s, keep_gain_vs_base=g[s], fid_change_vs_base=f[s]))
        return r
    z = {s: 0.0 for s in H.SLICES}
    good = {"openeds": 0.08, "video_gray": 0.06, "video_colour": 0.0}
    assert H.works_verdicts(rows(good, z, {s: 0.005 for s in H.SLICES}))["A@128"]["works"]
    assert not H.works_verdicts(rows({**good, "video_gray": 0.04}, z, z))["A@128"]["works"], "below the +0.05 line"
    assert not H.works_verdicts(rows(good, z, {s: 0.04 for s in H.SLICES}))["A@128"]["works"], "gain not above 2 x seed noise"
    assert not H.works_verdicts(rows(good, {**z, "video_colour": -0.03}, z))["A@128"]["works"], "fidelity loss on any slice"


def test_ssim_loss_is_zero_for_identical_penalises_lost_contrast_and_flows_gradients():
    from ivs.learning import ssim_loss
    ents = _entries()
    frames = np.concatenate([np.ascontiguousarray(_frames(m, d)[0][:4, :, :HEIGHT]) for m, d in ents[:2]])    # square left crops of stored frames of different widths
    x = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    assert float(ssim_loss(x, x)) < 1e-6
    blur = torch.from_numpy(np.stack([cv2.GaussianBlur(f, (0, 0), 2.0) for f in frames])).permute(0, 3, 1, 2).float() / 255.0
    flat = x.mean((2, 3), keepdim=True).expand_as(x).contiguous()
    lb, lf = float(ssim_loss(blur, x)), float(ssim_loss(flat, x))
    assert 0.02 < lb < lf < 1.0, (lb, lf)
    p = (x + 0.05 * torch.randn_like(x)).clamp(0, 1).requires_grad_(True)
    ssim_loss(p, x).backward()
    assert p.grad is not None and float(p.grad.abs().sum()) > 0


def test_spatial_plain_ae_shapes_latent_budget_gradients_and_unchanged_vector_init():
    from ivs.aeprobe import PlainAE
    from ivs.config import ModelConfig
    from ivs.model import ConvEncoder, build_decoder
    ents = _entries()
    x = torch.from_numpy(np.ascontiguousarray(_frames(*ents[0])[0][:2, :, :HEIGHT])).permute(0, 3, 1, 2).float() / 255.0
    for k, n in ((8, 512), (16, 1024)):
        m = PlainAE(HEIGHT, 512, spatial=k)
        assert m.latent_numbers == n and m.encoder(x).shape == (2, k, HEIGHT // 16, HEIGHT // 16)
        y = m(x)
        assert y.shape == x.shape and float(y.min()) >= 0 and float(y.max()) <= 1
        y.sum().backward()
        assert all(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in m.parameters()), "every parameter takes part"
    try:
        PlainAE(HEIGHT, 512, vae=1.0, spatial=8)
        raise AssertionError("spatial + vae must be refused")
    except ValueError:
        pass
    f = PlainAE(HEIGHT, 512, spatial=8, fine=True)
    assert f.latent_numbers == 8 * 16 * 16 and f.encoder(x).shape == (2, 8, HEIGHT // 8, HEIGHT // 8)
    y = f(x)
    assert y.shape == x.shape
    y.sum().backward()
    assert all(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in f.parameters())
    assert PlainAE(HEIGHT, 1024).latent_numbers == 1024 and PlainAE(HEIGHT, 1024)(x).shape == x.shape
    try:
        PlainAE(HEIGHT, 512, fine=True)
        raise AssertionError("fine needs a spatial latent")
    except ValueError:
        pass
    torch.manual_seed(3)
    a = PlainAE(96, 512)
    torch.manual_seed(3)
    cfg = ModelConfig(image_size=96, channels=3, feature_dim=512, conv_mult=1.0)
    enc, dec = ConvEncoder(cfg), build_decoder(cfg, 512)
    assert a.latent_numbers == 512 and all(torch.equal(p, q) for p, q in zip(list(a.encoder.parameters()) + list(a.decoder.parameters()), list(enc.parameters()) + list(dec.parameters()))), \
        "vector AE keeps the seeded init of the earlier results"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
