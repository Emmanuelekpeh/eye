"""Clarity test, stage 3 (R3): how sharp is a plain autoencoder on UNSEEN recordings at 96 / 128 / 192 px?
Stage 1 (clarity_ceiling.py) = what the resolution allows, stage 2 (overfit_clarity.py) = the decoder is not the limit on frames it has seen.
This stage trains ONE AE512 per height (ivs.aeprobe.PlainAE: the world model's conv stacks, no RSSM / KL / time) on the TRAIN groups of all sources
(videos 50% / OpenEDS 50% of the samples, the world-model recipe: random crop + flip, gray for gray/tint videos) and scores it on the VAL groups, i.e.
recordings it never saw. The split is ivs.cachedata.split_groups_stratified per source (seed 0, 15%) = the split of every earlier 96 px result.

  merge  --dirs <video cache> <openeds cache> --out <dir>        one CacheBank directory per height (copies the npz, concatenates the manifests)
  pack   --cache <v192> <o192> --out <dir>                      held-out eval pack: per val group K native frames (matched bit-for-bit to the 192 px cache
                                                               frames, so they are frames the cache really holds), OpenEDS native masks, eval_meta.json
  train  --cache 96=<dir> 128=<dir> 192=<dir> --pack <dir> --out <dir> --heights ...
  summary --results results.json [...] --out <dir>              gate table + height comparison, stacks the per-height sheets

Scoring = stage-2 pipeline on held-out frames. For each eval frame the VIEW is exactly what CacheBank(split='val', image_size=H) shows (same resize,
edge-pad / centre-crop; the cache code itself builds it). The AE reconstruction of that view and the view itself (= the CEILING of height H) are scaled
back to native size (cubic) over the native pixels the view covers and compared with the native crop: edge_fidelity, edge_keep, PSNR and, for OpenEDS,
the ground-truth pupil / limbus metrics with the native masks.

GATE (written in plan.md before any result): per slice (openeds / video_colour / video_gray = gray + tint videos) and height, PASS when
  edge_fidelity  AE >= 0.8 x ceiling
  edge_keep      0.8 x ceiling <= AE <= 1.05
  pupil_edge_grad_ratio (OpenEDS) AE >= 0.8 x ceiling
Not gated, reported: absolute AE edge_keep gain over the 96 px AE (a larger H is only worth its CPU cost if it is >= +0.03), PSNR, training curve.

R3b (plan.md): `train --arms ...` trains the ARMS variants (spatial latent / SSIM loss / seed repeat) on the same bank and scores them on the same pack; results keys are 'ARM@H'
(the default arm keeps the bare height). `summary` adds the gain over the baseline arm at the same height and works_verdicts() (the pre-registered WORKS rule in plan.md).
"""
import argparse
import hashlib
import json
import re
import shutil
import sys
import time
import types
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import clarity_ceiling as C                                        # noqa: E402
import overfit_clarity as O                                        # noqa: E402
from ivs.cachedata import CacheBank, STORE_W, colour_class, split_groups_stratified   # noqa: E402
from ivs.eyemetric import gt_metrics                               # noqa: E402

HEIGHTS = (96, 128, 192)
SLICES = ("openeds", "video_colour", "video_gray")
VAL_FRAC, SEED = 0.15, 0
K_PER_GROUP = 8
GATE, KEEP_MAX, WORTH = 0.8, 1.05, 0.03
SOURCE_WEIGHTS = {"video": 0.5, "openeds": 0.5}
GT_KEYS = ("pupil_contrast_ratio_median", "pupil_edge_grad_ratio", "limbus_grad_ratio")

# R3b arms (plan.md). AE512 = the R3 recipe (512-d vector code, pixel smooth-L1). spatial=k: (k, H/16, H/16) feature-map latent (aeprobe.PlainAE).
# ssim=w: + w * (3*H*W) * (1 - SSIM), the same size as the pixel term at ~29 dB. seed overrides --seed (AE512_s1 = the seed-noise yardstick).
ARMS = {
    "AE512": dict(),
    "AE512_s1": dict(seed=1),
    "AE512_ssim": dict(ssim=0.01),
    "S8": dict(spatial=8), "S8_ssim": dict(spatial=8, ssim=0.01),
    "S16": dict(spatial=16), "S16_ssim": dict(spatial=16, ssim=0.01),
    # R3c (plan.md): reference = S16_ssim. code = vector size; fine = (k, H/8, H/8) grid; steps_x = multiple of --steps; cache = suffix of the cache key ("f" = whole-video cache)
    "V1024_ssim": dict(code=1024, ssim=0.01), "S32_ssim": dict(spatial=32, ssim=0.01), "F8_ssim": dict(spatial=8, fine=True, ssim=0.01),
    "S16_ssim_x3": dict(spatial=16, ssim=0.01, steps_x=3), "S16_ssim_full": dict(spatial=16, ssim=0.01, cache="f"),
}
BASE_ARM = "AE512"
WORKS_KEEP, WORKS_FID, NOISE_MULT, NOISE_FLOOR = 0.05, 0.02, 2.0, 0.01   # R3b "WORKS": keep gain on openeds + video_gray, max fidelity loss, gain > 2 x seed noise (>= 0.01)


def result_key(arm, h):
    """Key in results.json: the bare height for the default arm (R3 files keep loading), 'ARM@H' otherwise."""
    return str(h) if arm == BASE_ARM else f"{arm}@{h}"


def split_key(k):
    arm, _, h = k.rpartition("@")
    return (arm or BASE_ARM), int(h)


def safe(stem):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")


def slice_of(m):
    if m["source"] == "openeds":
        return "openeds"
    return "video_colour" if colour_class(m) == "colour" else "video_gray"


def digest(a):
    return hashlib.blake2b(np.ascontiguousarray(a).tobytes(), digest_size=16).digest()


# ----------------------------------------------------------------------------------------------- the view CacheBank shows
def view_geometry(w, h):
    """(store_w, o, j0, j1) for a frame stored w wide at height h: the evaluation view is columns [o, o + h) of the stored frame (o < 0 = edge padding
    on the left), of which view columns [j0, j1) are real pixels (the rest is replicated padding)."""
    store_w = max(h, int(round(STORE_W * h / 96.0)))
    vs = (store_w - h) // 2
    if w >= store_w:
        return store_w, (w - store_w) // 2 + vs, 0, h
    o = vs - (store_w - w) // 2
    return store_w, o, max(0, -o), min(h, w - o)


def stored_frame(native, h):
    """Native crop (ch, cw[, 3]) uint8 -> the h-high frame the cache builders store (INTER_AREA, width = h * cw / ch), always 3 channels."""
    ch, cw = native.shape[:2]
    s = cv2.resize(native, (int(round(h * cw / ch)), h), interpolation=cv2.INTER_AREA)
    return np.repeat(s[..., None], 3, axis=2) if s.ndim == 2 else s


def eval_view(native, h, cls):
    """(view (h,h,3) uint8, j0, j1, o, w): what CacheBank(split='val', image_size=h, tint_mode='gray') shows for this frame (CacheBank._prepare itself builds it)."""
    s = stored_frame(native, h)
    store_w, o, j0, j1 = view_geometry(s.shape[1], h)
    fr = CacheBank._prepare(types.SimpleNamespace(size=h, store_w=store_w), s[None], h / 96.0, cls, "gray")
    v = (store_w - h) // 2
    return np.ascontiguousarray(fr[0, :, v:v + h]), j0, j1, o, s.shape[1]


def native_cols(o, j0, j1, w, cw):
    """Native crop columns [a, b) that view columns [j0, j1) cover (stored column c spans native [c * cw / w, (c + 1) * cw / w))."""
    return max(0, int(round((o + j0) * cw / w))), min(cw, int(round((o + j1) * cw / w)))


# ----------------------------------------------------------------------------------------------- merge + eval pack
def merge_dirs(dirs, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    metas, seen = [], set()
    for d in map(Path, dirs):
        for m in json.loads((d / "manifest.json").read_text()):
            f = safe(Path(m["name"]).stem) + ".npz"
            if f in seen:
                raise ValueError(f"{f} exists in two caches")
            seen.add(f)
            shutil.copy2(d / f, out / f)
            metas.append(m)
    (out / "manifest.json").write_text(json.dumps(metas, indent=1))
    return len(metas), sum(m["kept"] for m in metas)


def val_groups_of(metas):
    val = set()
    for src in sorted({m.get("source", "video") for m in metas}):
        val |= split_groups_stratified([m for m in metas if m.get("source", "video") == src], VAL_FRAC, SEED)
    return val


def pick_even(idx, k):
    idx = np.asarray(idx)
    return idx[np.unique(np.linspace(0, len(idx) - 1, min(k, len(idx))).round().astype(int))]


def openeds_natives(root, m, frames, k):
    """k native (gray crop, mask crop) pairs of an OpenEDS group, chosen among the cache frames with a visible pupil + iris, matched to the PNGs bit-for-bit."""
    fol = Path(root) / m["group"]
    x0, y0, x1, y1 = m["crop_box"]
    hs, ws = frames.shape[1:3]
    by_hash = {digest(f): i for i, f in enumerate(frames)}
    found = {}
    for p in sorted(fol.glob("*.png"), key=lambda p: int(p.stem) if p.stem.isdigit() else -1):
        if not p.stem.isdigit():
            continue
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)[y0:y1, x0:x1]
        i = by_hash.get(digest(np.repeat(cv2.resize(g, (ws, hs), interpolation=cv2.INTER_AREA)[..., None], 3, axis=2)))
        if i is not None and (fol / f"{p.stem}.npy").exists():
            mk = np.load(fol / f"{p.stem}.npy")[y0:y1, x0:x1]
            if mk.shape == g.shape and int((mk == 3).sum()) >= 60 and int((mk == 2).sum()) >= 60:
                found[i] = (g, mk)
    if len(found) < 4:
        return []
    ids = pick_even(sorted(found), k)
    return [(int(i), found[int(i)][0], found[int(i)][1]) for i in ids]


def video_natives(root, m, frames, k):
    """k native RGB crops of a video group, evenly over its cached frames, found again by re-decoding the same spans (same seek, stride, crop, INTER_AREA)."""
    import build_cache as BC
    ids = pick_even(np.arange(len(frames)), k)
    want = {digest(frames[i]): int(i) for i in ids}
    x0, y0, x1, y1 = m["crop_box"]
    hs, ws = frames.shape[1:3]
    stride = m["stride"]
    spans = BC.window_spans(m["total_raw"], m.get("windows", 0), m.get("window_len", 0) * stride)
    cap = cv2.VideoCapture(str(Path(root) / m["name"]))
    found = {}
    for start, length in spans:
        if start:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        j = 0
        while (length is None or j < length) and len(found) < len(want):
            if not cap.grab():
                break
            if j % stride == 0:
                ok, f = cap.retrieve()
                if not ok:
                    break
                crop = f[y0:y1, x0:x1]
                rgb = cv2.cvtColor(cv2.resize(crop, (ws, hs), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
                i = want.get(digest(rgb))
                if i is not None:
                    found[i] = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            j += 1
    cap.release()
    return [(i, found[i], None) for i in sorted(found)]


def build_pack(cache_dirs, root, out, per_group=K_PER_GROUP):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    metas = []
    for d in cache_dirs:
        for m in json.loads((Path(d) / "manifest.json").read_text()):
            m["_dir"] = str(d); m.setdefault("source", "video")
            metas.append(m)
    val = val_groups_of(metas)
    arrays, meta, skipped = {}, [], []
    for m in sorted([m for m in metas if m["group"] in val], key=lambda m: m["name"]):
        with np.load(Path(m["_dir"]) / (safe(Path(m["name"]).stem) + ".npz")) as z:
            frames = z["frames"]
        got = openeds_natives(root, m, frames, per_group) if m["source"] == "openeds" else video_natives(root, m, frames, per_group)
        if not got:
            skipped.append(m["name"]); continue
        for i, nat, mk in got:
            n = len(meta)
            arrays[f"n{n}"] = nat
            if mk is not None:
                arrays[f"m{n}"] = mk
            meta.append(dict(i=n, name=m["name"], group=m["group"], source=m["source"], cls=colour_class(m), slice=slice_of(m), stored_idx=i, crop_hw=list(nat.shape[:2])))
        print(f"{m['name']}: {len(got)} frames ({slice_of(m)})", flush=True)
    np.savez_compressed(out / "eval_native.npz", **arrays)
    (out / "eval_meta.json").write_text(json.dumps(meta, indent=1))
    (out / "val_groups.json").write_text(json.dumps(sorted(val)))
    cnt = {s: sum(1 for x in meta if x["slice"] == s) for s in SLICES}
    print(f"eval pack: {len(meta)} frames {cnt}, {len(val)} val groups, skipped {skipped}", flush=True)
    return meta


# ----------------------------------------------------------------------------------------------- scoring
def build_items(pack, h):
    """Per eval frame: its view at height h, the native pixels it covers (gray, shown, mask) and the columns used."""
    pack = Path(pack)
    meta = json.loads((pack / "eval_meta.json").read_text())
    z = np.load(pack / "eval_native.npz")
    items = []
    for m in meta:
        nat = z[f"n{m['i']}"]
        view, j0, j1, o, w = eval_view(nat, h, m["cls"])
        a, b = native_cols(o, j0, j1, w, nat.shape[1])
        if b - a < 16 or j1 - j0 < 16:
            continue
        mk = z[f"m{m['i']}"][:, a:b] if f"m{m['i']}" in z.files else None
        items.append(dict(m, view=view, j0=j0, j1=j1, a=a, b=b, native=nat[:, a:b], native_g=O.gray_of(nat[:, a:b]) if nat.ndim == 3 else nat[:, a:b], mask=mk))
    return items


def to_native(rgb_view, it):
    """(h,h,3) uint8 view / reconstruction -> RGB at the native size of the pixels the view covers (cubic, as in stages 1 and 2)."""
    nh, nw = it["native_g"].shape
    return cv2.resize(rgb_view[:, it["j0"]:it["j1"]], (nw, nh), interpolation=cv2.INTER_CUBIC)


def score_items(items, recs):
    """{slice: metrics} for reconstructions `recs` (one (h,h,3) uint8 per item). frame metrics are averaged over frames; the OpenEDS ground-truth
    metrics are computed per group (same crop, so frames stack) and averaged over groups."""
    out = {}
    for s in SLICES:
        idx = [i for i, it in enumerate(items) if it["slice"] == s]
        if not idx:
            continue
        rec_g = {i: O.gray_of(to_native(recs[i], items[i])) for i in idx}
        res = C.mean_dict([C.frame_metrics(items[i]["native_g"], rec_g[i]) for i in idx])
        if s == "openeds":
            rows = []
            for g in sorted({items[i]["group"] for i in idx}):
                gi = [i for i in idx if items[i]["group"] == g]
                rows.append(gt_metrics(torch.from_numpy(np.stack([items[i]["native_g"] for i in gi])).float() / 255.0,
                                       torch.from_numpy(np.stack([rec_g[i] for i in gi])).float() / 255.0,
                                       torch.from_numpy(np.stack([items[i]["mask"] for i in gi])).to(torch.uint8)))
            gt = C.mean_dict(rows)
            res.update({k: gt[k] for k in GT_KEYS if k in gt})
        res["n_frames"] = len(idx)
        res["n_groups"] = len({items[i]["group"] for i in idx})
        out[s] = res
    return out


def gate_of(ae, ceil):
    g = dict(edge_keep=bool(GATE * ceil["edge_keep"] <= ae["edge_keep"] <= KEEP_MAX), edge_fidelity=bool(ae["edge_fidelity"] >= GATE * ceil["edge_fidelity"]))
    if "pupil_edge_grad_ratio" in ceil and "pupil_edge_grad_ratio" in ae:
        g["pupil_edge"] = bool(ae["pupil_edge_grad_ratio"] >= GATE * ceil["pupil_edge_grad_ratio"])
    return g


def recon_all(model, items, device, chunk=48):
    out = []
    for i in range(0, len(items), chunk):
        x = torch.from_numpy(np.stack([it["view"] for it in items[i:i + chunk]])).permute(0, 3, 1, 2).float() / 255.0
        out.extend(O.reconstruct(model, x, device))
    return out


def flat(scores):
    """Curve row for train_one: mean over slices of the headline numbers, plus every slice's own."""
    row = {k: float(np.mean([v[k] for v in scores.values()])) for k in ("edge_fidelity", "edge_keep", "psnr")}
    for s, v in scores.items():
        row.update({f"{s}/edge_fidelity": v["edge_fidelity"], f"{s}/edge_keep": v["edge_keep"], f"{s}/psnr": v["psnr"]})
    return row


def probe_subset(items, step=4, least=2):
    """Every `step`-th held-out frame of EACH slice (>= `least` per slice): cheap scoring for the training curve that never drops a slice."""
    out = []
    for s in SLICES:
        mine = [it for it in items if it["slice"] == s]
        out.extend(mine[::step] if len(mine[::step]) >= least else mine[:least])
    return out


def show_pairs(items, sheet_ids):
    """For each slice the two items whose pack ids are in sheet_ids -> list of positions in `items`."""
    pos = {it["i"]: p for p, it in enumerate(items)}
    return {s: [pos[i] for i in sheet_ids[s] if i in pos] for s in SLICES}


def sheet_ids_of(pack):
    meta = json.loads((Path(pack) / "eval_meta.json").read_text())
    out = {}
    for s in SLICES:
        ids = [m["i"] for m in meta if m["slice"] == s]
        out[s] = [ids[len(ids) // 3], ids[2 * len(ids) // 3]] if len(ids) >= 3 else ids[:2]
    return out


def central_patch(gray, size):
    """(y, x, size): the most detailed size x size native window inside the central half of the frame, so overlays / watermarks / frame borders
    (the detail_patch of the whole frame picks text on some videos) do not decide what the sheet shows."""
    h, w = gray.shape
    y0, x0 = h // 4, w // 4
    sub = gray[y0:h - y0, x0:w - x0]
    size = min(size, *sub.shape)
    y, x, _ = C.detail_patch(sub, size)
    return y + y0, x + x0, size


def sheet_row(items, recs, ids):
    """One tile row for one slice: for two held-out frames native | ceiling (the view) | AE, all on the same most-detailed native patch."""
    colour = items[ids[0]]["slice"] == "video_colour"
    g = (lambda x: x) if colour else O.gray_of
    nat = [items[i]["native"] if colour else items[i]["native_g"] for i in ids]
    ceil = [g(to_native(items[i]["view"], items[i])) for i in ids]
    ae = [g(to_native(recs[i], items[i])) for i in ids]
    s = min(C.PATCH, *(min(items[i]["native_g"].shape) // 2 for i in ids))
    pat = [central_patch(items[i]["native_g"], s) for i in ids]
    return O.sheet_tiles(nat, ceil, ae, list(range(len(ids))), ([p[0] for p in pat], [p[1] for p in pat], s))


def sheet_name(s, arm, h):
    return f"sheet_{s}_h{h}.png" if arm == BASE_ARM else f"sheet_{s}_{arm}_h{h}.png"


def arm_parts(arm, h):
    """(make_model, loss_fn) of an arm for train_one."""
    from ivs.aeprobe import PlainAE
    from ivs.learning import _pixel_sum_loss, ssim_loss
    spec = ARMS[arm]
    w = float(spec.get("ssim", 0.0))

    def loss_fn(pred, x):
        pix = _pixel_sum_loss(pred, x)
        if not w:
            return pix, {}
        one_minus = ssim_loss(pred, x)
        term = w * x[0].numel() * one_minus
        return pix + term, dict(pixel=pix.detach(), ssim_term=term.detach(), one_minus_ssim=one_minus.detach())
    return (lambda: PlainAE(h, int(spec.get("code", 512)), spatial=int(spec.get("spatial", 0)), fine=bool(spec.get("fine", False)))), loss_fn


def prepare_height(cache, pack, h):
    """Everything the arms of one height share: held-out items + ceilings, the training bank, the probe subset."""
    items = build_items(pack, h)
    ceil = score_items(items, [it["view"] for it in items])
    print(f"[h{h}] {len(items)} held-out frames; ceilings " + " | ".join(f"{s} keep {v['edge_keep']:.3f} fid {v['edge_fidelity']:.3f}" for s, v in ceil.items()), flush=True)
    bank = CacheBank(cache, image_size=h, split="train", val_frac=VAL_FRAC, seed=SEED, tint_mode="gray", source_weights=SOURCE_WEIGHTS, device="cpu")
    info = bank.info()
    print(f"[h{h}] train bank: {info['frames']} frames, {info['groups']} groups, by source {info['frames_by_source']}", flush=True)
    return dict(items=items, ceil=ceil, bank=bank, info=info, pack=pack, probe_items=probe_subset(items))


def run_height(cache, pack, h, out, steps, batch, lr, device, seed, probe_every, arm=BASE_ARM, ctx=None, save_to=None):
    ctx = ctx or prepare_height(cache, pack, h)
    items, ceil, bank, info = ctx["items"], ctx["ceil"], ctx["bank"], ctx["info"]
    seed = int(ARMS[arm].get("seed", seed))
    mult = int(ARMS[arm].get("steps_x", 1))
    steps, probe_every = steps * mult, probe_every * mult
    tag = f"[{arm}] "
    probe_items = ctx["probe_items"]
    probe = lambda m: flat(score_items(probe_items, recon_all(m, probe_items, device)))
    sampler = lambda b, gen: bank.sample(b, 1, generator=gen, train=True)["images"][:, 0]
    make_model, loss_fn = arm_parts(arm, h)
    model, curve, sec = O.train_one(None, h, steps, batch, lr, device, seed, probe, probe_every, sampler=sampler, keys=None, make_model=make_model, loss_fn=loss_fn, tag=tag)
    if save_to:                                                    # R4: the trained AE is the frozen base of the latent dynamics (ivs.latentdyn.load_ae)
        from ivs.latentdyn import save_ae
        save_ae(save_to, model, meta=dict(arm=arm, h=h, steps=steps, batch=batch, seed=seed, train_frames=info["frames"]))
    recs = recon_all(model, items, device)
    ae = score_items(items, recs)
    res = {}
    for s in ae:
        gate = gate_of(ae[s], ceil[s])
        res[s] = dict(ceiling=ceil[s], ae=ae[s], ae_over_ceiling_keep=ae[s]["edge_keep"] / ceil[s]["edge_keep"],
                      ae_over_ceiling_fidelity=ae[s]["edge_fidelity"] / ceil[s]["edge_fidelity"], gate=gate, passed=bool(all(gate.values())))
        print(f"{tag}h{h} {s}: AE keep {ae[s]['edge_keep']:.3f} / ceiling {ceil[s]['edge_keep']:.3f}, fidelity {ae[s]['edge_fidelity']:.3f} / {ceil[s]['edge_fidelity']:.3f}, "
              f"psnr {ae[s]['psnr']:.1f} / {ceil[s]['psnr']:.1f} -> {'PASS' if res[s]['passed'] else 'FAIL'}", flush=True)
    pairs = show_pairs(items, sheet_ids_of(ctx["pack"]))
    for s, ids in pairs.items():
        if len(ids) == 2:
            cv2.imwrite(str(Path(out) / sheet_name(s, arm, h)), sheet_row(items, recs, ids))
    return dict(slices=res, curve=curve, train_seconds=sec, steps=steps, batch=batch, train_frames=info["frames"], train_groups=info["groups"],
                train_frames_by_source=info["frames_by_source"], eval_frames=len(items), arm=arm, arm_spec=ARMS[arm], seed=seed,
                latent_numbers=int(model.latent_numbers), params=int(sum(p.numel() for p in model.parameters())))


def train_main(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    caches = dict(kv.split("=", 1) for kv in a.cache)
    device = torch.device("cuda" if a.device == "auto" and torch.cuda.is_available() else ("cpu" if a.device == "auto" else a.device))
    print(f"device={device} steps={a.steps} batch={a.batch} heights={a.heights} arms={a.arms}", flush=True)
    results = {}
    for h in a.heights:
        ctxs = {}                                                      # one training bank per cache key ("128", or "128f" for arms with cache="f")
        for arm in a.arms:
            ck = f"{h}{ARMS[arm].get('cache', '')}"
            if ck not in ctxs:
                ctxs.clear()
                ctxs[ck] = prepare_height(caches[ck], a.pack, h)
            results[result_key(arm, h)] = run_height(caches[ck], a.pack, h, out, a.steps, a.batch, a.lr, device, a.seed, a.probe_every, arm=arm, ctx=ctxs[ck])
            (out / "results.json").write_text(json.dumps(results, indent=1))
    return results


# ----------------------------------------------------------------------------------------------- summary
def summarise(results, base_arm=BASE_ARM):
    """Gate table rows + absolute AE gain over the smallest height and over `base_arm` at the same height, from merged results {key: run_height output}."""
    by_arm = {}
    for k, v in results.items():
        arm, h = split_key(k)
        by_arm.setdefault(arm, {})[h] = v
    rows = []
    for arm, per_h in by_arm.items():
        hs = sorted(per_h)
        for h in hs:
            for s, v in per_h[h]["slices"].items():
                base = per_h[hs[0]]["slices"].get(s)
                ref = by_arm.get(base_arm, {}).get(h, {}).get("slices", {}).get(s) if arm != base_arm else None
                rows.append(dict(arm=arm, height=h, slice=s, ceiling_keep=v["ceiling"]["edge_keep"], ae_keep=v["ae"]["edge_keep"], ceiling_fid=v["ceiling"]["edge_fidelity"],
                                 ae_fid=v["ae"]["edge_fidelity"], keep_ratio=v["ae_over_ceiling_keep"], fid_ratio=v["ae_over_ceiling_fidelity"], psnr=v["ae"]["psnr"],
                                 pupil_edge=(v["ae"].get("pupil_edge_grad_ratio"), v["ceiling"].get("pupil_edge_grad_ratio")), passed=v["passed"], gate=v["gate"],
                                 keep_gain_vs_smallest=(v["ae"]["edge_keep"] - base["ae"]["edge_keep"]) if base and h != hs[0] else None,
                                 worth_it=(v["ae"]["edge_keep"] - base["ae"]["edge_keep"] >= WORTH) if base and h != hs[0] else None,
                                 keep_gain_vs_base=(v["ae"]["edge_keep"] - ref["ae"]["edge_keep"]) if ref else None,
                                 fid_change_vs_base=(v["ae"]["edge_fidelity"] - ref["ae"]["edge_fidelity"]) if ref else None,
                                 latent_numbers=per_h[h].get("latent_numbers"), params=per_h[h].get("params")))
    return rows


def works_verdicts(rows, keep=WORKS_KEEP, noise_floor=NOISE_FLOOR):
    """{'ARM@H': dict(works, why)} for every arm that has the base arm (summarise's base_arm) at the same height. WORKS = keep gain over the base >= `keep` on openeds AND
    video_gray, each gain > NOISE_MULT x the seed noise of that slice (|AE512_s1 - base| when that seed twin is in the rows, floor `noise_floor`), and edge_fidelity not lower
    than the base - WORKS_FID on any slice. AE512_s1 is the yardstick itself and gets no verdict. R3b: keep .05, floor .01; R3c: keep .03 (data arm .02), floor .005 (plan.md)."""
    out = {}
    cells = {(r["arm"], r["height"], r["slice"]): r for r in rows}
    for arm, h in sorted({(r["arm"], r["height"]) for r in rows if r["keep_gain_vs_base"] is not None and r["arm"] != "AE512_s1"}):
        why, ok = [], True
        for s in SLICES:
            r = cells.get((arm, h, s))
            if r is None:
                continue
            twin = cells.get(("AE512_s1", h, s))
            noise = max(noise_floor, abs(twin["keep_gain_vs_base"])) if twin and twin["keep_gain_vs_base"] is not None else noise_floor
            if s in ("openeds", "video_gray") and not (r["keep_gain_vs_base"] >= keep and r["keep_gain_vs_base"] > NOISE_MULT * noise):
                ok = False; why.append(f"{s} keep gain {r['keep_gain_vs_base']:+.3f} < max({keep}, {NOISE_MULT}x noise {noise:.3f})")
            if r["fid_change_vs_base"] < -WORKS_FID:
                ok = False; why.append(f"{s} fidelity {r['fid_change_vs_base']:+.3f}")
        out[result_key(arm, h)] = dict(works=ok, why=why)
    return out


def label_row(img, text):
    """A 20 px caption strip above a sheet row."""
    strip = np.zeros((20, img.shape[1], 3), np.uint8)
    cv2.putText(strip, text, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([strip, img], 0)


def stack_sheets(out, baseline_dirs=(), base_arm=BASE_ARM):
    """sheet_<slice>.png = the per-height rows of the default arm (R3, rows by height); sheet_<slice>_arms_h<H>.png = one captioned row per arm at H, `base_arm`'s row
    taken from `out` or the first of `baseline_dirs` that has it."""
    out = Path(out)
    for s in SLICES:
        rows = [cv2.imread(str(p)) for h in HEIGHTS for p in [out / sheet_name(s, BASE_ARM, h)] if p.exists()]
        if rows:
            cv2.imwrite(str(out / f"sheet_{s}.png"), np.concatenate(rows, 0))
        for h in HEIGHTS:
            arms = [(arm, p) for arm in ARMS for p in [out / sheet_name(s, arm, h)] if p.exists()]
            if arms and not any(a == base_arm for a, _ in arms):
                base = next((Path(d) / sheet_name(s, base_arm, h) for d in baseline_dirs if (Path(d) / sheet_name(s, base_arm, h)).exists()), None)
                arms = ([(base_arm, base)] if base else []) + arms
            if len(arms) > 1:
                cv2.imwrite(str(out / f"sheet_{s}_arms_h{h}.png"), np.concatenate([label_row(cv2.imread(str(p)), f"{a} (rows: native | ceiling | AE, 2 frames)") for a, p in arms], 0))


def summary_main(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    results = {}
    for f in a.results:
        results.update(json.loads(Path(f).read_text()))
    base_arm = getattr(a, "base_arm", None) or BASE_ARM
    rows = summarise(results, base_arm)
    verdicts = works_verdicts(rows, getattr(a, "works_keep", None) or WORKS_KEEP, getattr(a, "noise_floor", None) or NOISE_FLOOR)
    (out / "results.json").write_text(json.dumps(results, indent=1))
    (out / "summary.json").write_text(json.dumps(rows, indent=1))
    (out / "verdicts.json").write_text(json.dumps(verdicts, indent=1))
    stack_sheets(out, getattr(a, "baseline_sheets", None) or (), base_arm)
    print(f"{'arm':<13} {'H':>4} {'slice':<13} {'keep AE/ceil':>14} {'fid AE/ceil':>14} {'AE/ceil keep':>12} {'fid':>5} {'psnr':>5} {'dKeep vs 96':>11} {'dKeep vs ' + base_arm:>14}  gate")
    for r in rows:
        d = "" if r["keep_gain_vs_smallest"] is None else f"{r['keep_gain_vs_smallest']:+.3f}"
        b = "" if r["keep_gain_vs_base"] is None else f"{r['keep_gain_vs_base']:+.3f}"
        print(f"{r['arm']:<13} {r['height']:>4} {r['slice']:<13} {r['ae_keep']:.3f}/{r['ceiling_keep']:.3f} {r['ae_fid']:.3f}/{r['ceiling_fid']:.3f} {r['keep_ratio']:>12.3f} {r['fid_ratio']:>5.2f} {r['psnr']:>5.1f} {d:>11} {b:>14}  "
              f"{'PASS' if r['passed'] else 'FAIL ' + ','.join(k for k, v in r['gate'].items() if not v)}")
    for k, v in verdicts.items():
        print(f"WORKS {k}: {v['works']} {'; '.join(v['why'])}")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("merge"); p.add_argument("--dirs", nargs="+", required=True); p.add_argument("--out", required=True)
    p = sub.add_parser("pack")
    p.add_argument("--cache", nargs="+", required=True); p.add_argument("--out", required=True)
    p.add_argument("--root", default=str(C.ROOT / "set")); p.add_argument("--per-group", type=int, default=K_PER_GROUP)
    p = sub.add_parser("train")
    p.add_argument("--cache", nargs="+", required=True, help="H=dir pairs"); p.add_argument("--pack", required=True); p.add_argument("--out", required=True)
    p.add_argument("--heights", type=int, nargs="+", default=list(HEIGHTS)); p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--batch", type=int, default=128); p.add_argument("--lr", type=float, default=1e-3); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--probe-every", type=int, default=2000); p.add_argument("--device", default="auto")
    p.add_argument("--arms", nargs="+", default=[BASE_ARM], choices=list(ARMS))
    p = sub.add_parser("summary"); p.add_argument("--results", nargs="+", required=True); p.add_argument("--out", required=True)
    p.add_argument("--baseline-sheets", nargs="*", default=[], help="dirs holding the base arm's sheets (e.g. earlier results) to caption into the per-arm comparison sheets")
    p.add_argument("--base-arm", default=BASE_ARM, choices=list(ARMS), help="arm the gains / verdicts are measured against (R3b AE512, R3c S16_ssim)")
    p.add_argument("--works-keep", type=float, default=WORKS_KEEP, help="edge_keep gain the verdict needs on openeds and video_gray")
    p.add_argument("--noise-floor", type=float, default=NOISE_FLOOR)
    a = ap.parse_args(argv)
    if a.cmd == "merge":
        n, f = merge_dirs(a.dirs, a.out)
        print(f"merged {n} entries, {f} frames -> {a.out}")
    elif a.cmd == "pack":
        build_pack(a.cache, a.root, a.out, a.per_group)
    elif a.cmd == "train":
        return train_main(a)
    else:
        return summary_main(a)


if __name__ == "__main__":
    main()
