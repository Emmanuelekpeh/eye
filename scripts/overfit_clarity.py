"""Clarity test, stage 2: can a plain autoencoder reproduce ONE real clip sharply at 96 / 128 / 192 px? (overfit = train-frame fit)
Separates "the resolution limits detail" (stage 1, scripts/clarity_ceiling.py) from "the decoder / loss cannot render it".

pack   (local, read-only on set/): one colour clip and one OpenEDS sequence, cut to their centre SQUARE at native size (the same centre crop the
       CacheBank evaluation view takes). Each clip file holds the training frames at every height (INTER_AREA, as the cache builders do), 24 evenly
       spaced NATIVE eval frames (+ their native masks for OpenEDS) and the clip meta.
train  (CPU or GPU, same code): a plain AE512 (ivs.aeprobe.PlainAE: the world model's conv stacks, no RSSM / KL / time) trained only on that clip's frames
       with the world model's pixel-sum smooth-L1. At the end its reconstructions of the eval frames are scaled to native size and scored exactly like
       stage 1 (edge_fidelity = the gate metric: share of the native edge energy the reconstruction explains, so blur, noise and exaggerated contrast all lower it; edge_keep, PSNR, and for OpenEDS the ground-truth pupil / limbus metrics with the native masks) next to the CEILING of the same height
       on the same frames (shrink + scale back, no model). Writes results.json, a convergence curve per run and sheet_<clip>.png
       (rows = heights; tiles = native | ceiling | AE for two eval frames, the most detailed 160 px patch). LOOK at the sheets.

GATE (written before any result; every guard exists because a real failure showed it is needed): a run PASSES when, on the eval frames, ALL hold
  edge_keep      0.9 x ceiling <= AE <= 1.05   (blur / wash-out fail low; a copy cannot have more edge strength than native, so more is over-sharpening)
  edge_fidelity  AE >= 0.9 x ceiling           (share of native edge energy explained; noise, artefacts and misplaced edges fail even when edge_keep looks fine:
                                                after 60 steps the OpenEDS AE had edge_keep .56 = the ceiling but edge_fidelity .19 vs .73)
  pupil_edge_grad_ratio (OpenEDS only) AE >= 0.9 x ceiling
Pass = the decoder is not what limits detail on seen frames, so remaining blur comes from regularisation (KL, sampling) or the dynamics path; fail = decoder /
loss / steps limit it even on frames it has seen. The convergence curve says whether more steps would help. PSNR is reported next to it, not gated.
  python scripts/overfit_clarity.py pack --out ../overfit_pack
  python scripts/overfit_clarity.py train --pack ../overfit_pack --out pilot_results/overfit_clarity --steps 6000
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import clarity_ceiling as C                       # noqa: E402
from ivs.eyemetric import gt_metrics              # noqa: E402

COLOUR_CLIP, OPENEDS_CLIP = "yt_-01-XBHXFq8.mp4", "S_105"
HEIGHTS = (96, 128, 192)
CLIPS = ("colour", "openeds")
EVAL_N = 24
GATE = 0.9
KEEP_MAX = 1.05
CUT_THR = 0.30                                    # build_cache.py's scene-cut threshold on 48 px gray frames


def centre_square(img):
    h, w = img.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return img[y0:y0 + s, x0:x0 + s]


def longest_clean_run(frames):
    """(start, end) of the longest run without a scene cut (mean |diff| of 48 px gray frames > CUT_THR, as in the cache builders)."""
    small = np.stack([cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f, (48, 48), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0 for f in frames])
    cut = np.abs(np.diff(small, axis=0)).mean(axis=(1, 2)) > CUT_THR
    best, start = (0, 0), 0
    for i in range(len(frames)):
        if i == len(frames) - 1 or cut[i]:
            if i + 1 - start > best[1] - best[0]:
                best = (start, i + 1)
            start = i + 1
    return best


def read_colour_clip(root, name, n, start_frac=0.25, target_fps=20.0):
    """n consecutive frames (stride like the cache: fps / 20) of a raw video from start_frac on, cropped to the manifest crop box, then to the centre
    square, trimmed to the longest cut-free run. Returns RGB uint8 (N,S,S,3) and the clip meta."""
    m = next(m for m in json.loads((root / "data" / "cache" / "videos" / "manifest.json").read_text()) if m["name"] == name)
    cap = cv2.VideoCapture(str(root / "set" / name))
    fps, total = cap.get(cv2.CAP_PROP_FPS) or 25.0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(1, int(round(fps / target_fps)))
    x0, y0, x1, y1 = m["crop_box"]
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * start_frac))
    frames, i = [], 0
    while len(frames) < n:
        if not cap.grab():
            break
        if i % stride == 0:
            ok, f = cap.retrieve()
            if not ok:
                break
            frames.append(centre_square(f[y0:y1, x0:x1]))
        i += 1
    cap.release()
    a, b = longest_clean_run(frames)
    frames = frames[a:b]
    if len(frames) < 60:
        raise RuntimeError(f"{name}: only {len(frames)} cut-free frames after {int(total * start_frac)}")
    rgb = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames])
    return rgb, dict(name=name, kind="colour", native=m["native"], crop_box=m["crop_box"], fps=fps, stride=stride, frames=len(rgb), square=int(rgb.shape[1]))


def read_openeds_clip(root, group):
    """Every frame of set/<group> cropped with the cache's crop box, then to the centre square; frames with a mask are eval candidates."""
    m = next(m for m in json.loads((root / "data" / "cache" / "openeds" / "manifest.json").read_text()) if m["group"] == group)
    fol = root / "set" / group
    nums = sorted(int(p.stem) for p in fol.glob("*.png") if p.stem.isdigit())
    x0, y0, x1, y1 = m["crop_box"]
    frames, masks = [], []
    for k in nums:
        g = cv2.imread(str(fol / f"{k}.png"), cv2.IMREAD_GRAYSCALE)
        f = fol / f"{k}.npy"
        mk = np.load(f) if f.exists() else np.full(g.shape, 255, np.uint8)
        frames.append(centre_square(g[y0:y1, x0:x1]))
        masks.append(centre_square(mk[y0:y1, x0:x1]))
    a, b = longest_clean_run(frames)
    frames, masks = np.stack(frames[a:b]), np.stack(masks[a:b])
    return frames, masks, dict(name=group, kind="openeds", native=m["native"], crop_box=m["crop_box"], fps=20.0, stride=1, frames=len(frames), square=int(frames.shape[1]))


def eval_indices(n_frames, ok, k=EVAL_N):
    cand = np.nonzero(ok)[0]
    if len(cand) < 4:
        raise RuntimeError(f"only {len(cand)} usable eval frames")
    return cand[np.unique(np.linspace(0, len(cand) - 1, min(k, len(cand))).round().astype(int))]


def build_pack(out, root=C.ROOT, colour=COLOUR_CLIP, openeds=OPENEDS_CLIP, n_colour=240, heights=HEIGHTS):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    clips = {}
    rgb, meta = read_colour_clip(root, colour, n_colour)
    clips["colour"] = (rgb, None, meta, np.ones(len(rgb), bool))
    g, masks, meta = read_openeds_clip(root, openeds)
    ok = ((masks == 3).sum((1, 2)) >= 60) & ((masks == 2).sum((1, 2)) >= 60)
    clips["openeds"] = (np.repeat(g[..., None], 3, axis=3), masks, meta, ok)
    for key, (frames, masks, meta, ok) in clips.items():
        idx = eval_indices(len(frames), ok)
        pack = dict(meta=json.dumps(dict(meta, eval_idx=idx.tolist(), heights=list(heights))), eval_idx=idx,
                    native_eval=frames[idx] if key == "colour" else frames[idx][..., 0])
        if masks is not None:
            pack["masks_eval"] = masks[idx]
        for h in heights:
            pack[f"train_{h}"] = np.stack([cv2.resize(f, (h, h), interpolation=cv2.INTER_AREA) for f in frames])
        np.savez_compressed(out / f"{key}.npz", **pack)
        print(f"{key}: {meta['name']} {len(frames)} frames, native square {meta['square']}, eval {len(idx)}", flush=True)
    return out


def gray_of(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb


def score(native, rec, masks):
    """native, rec: lists/arrays of uint8 gray (S,S) at native size. Stage-1 metrics (+ ground-truth eye metrics when masks are given)."""
    res = C.mean_dict([C.frame_metrics(n, r) for n, r in zip(native, rec)])
    if masks is not None:
        real = torch.from_numpy(np.stack(native)).float() / 255.0
        rect = torch.from_numpy(np.stack(rec)).float() / 255.0
        gt = gt_metrics(real, rect, torch.from_numpy(masks).to(torch.uint8))
        res.update({k: gt[k] for k in ("pupil_contrast_ratio_median", "pupil_edge_grad_ratio", "limbus_grad_ratio", "n_pupil_frames") if gt.get(k) is not None})
    return res


def up_to_native(rec_u8, s):
    """(E,H,H,3) uint8 RGB -> list of (S,S) uint8 gray, cubic like the stage-1 ceiling."""
    return [gray_of(cv2.resize(r, (s, s), interpolation=cv2.INTER_CUBIC)) for r in rec_u8]


def reconstruct(model, x_eval, device):
    model.eval()
    with torch.no_grad():
        y = model(x_eval.to(device)).clamp(0, 1).cpu()
    model.train()
    return (y.permute(0, 2, 3, 1).numpy() * 255).round().astype(np.uint8)


CURVE_KEYS = ("edge_fidelity", "edge_keep", "psnr", "pupil_edge_grad_ratio")


def train_one(frames, h, steps, batch, lr, device, seed, probe, probe_every, sampler=None, keys=CURVE_KEYS, make_model=None, loss_fn=None, tag=""):
    """AE512 on `frames` (N,h,h,3 uint8) only, or on `sampler(batch, generator) -> (B,3,h,h) float` when given (then `frames` is unused, e.g. a
    CacheBank). probe(model) -> dict of eval numbers, called every probe_every steps; `keys` picks what goes into the curve (None = all).
    make_model() -> module (default PlainAE(h, 512)); loss_fn(pred, x) -> (loss, {name: detached scalar tensor}) (default the world model's pixel smooth-L1);
    the named parts are logged into the curve rows. Returns model, curve, seconds."""
    from ivs.aeprobe import PlainAE
    from ivs.learning import _pixel_sum_loss
    x_all = None if sampler else (torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0).to(device)
    torch.manual_seed(seed)
    model = (make_model() if make_model else PlainAE(h, 512)).to(device)
    loss_fn = loss_fn or (lambda pred, x: (_pixel_sum_loss(pred, x), {}))
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    curve, t0 = [], time.time()
    for step in range(steps):
        cur = lr * min(1.0, (step + 1) / 200) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / steps)))
        for g in opt.param_groups:
            g["lr"] = cur
        x = sampler(batch, gen).to(device) if sampler else x_all[torch.randint(0, len(x_all), (batch,), generator=gen).to(device)]
        loss, parts = loss_fn(model(x), x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
        opt.step()
        if (step + 1) % probe_every == 0 or step + 1 == steps:
            row = dict(step=step + 1, loss=float(loss), sec=round(time.time() - t0, 1), **{k: float(v) for k, v in parts.items()},
                       **{k: v for k, v in probe(model).items() if keys is None or k in keys})
            curve.append(row)
            print(f"  {tag}h{h} step {row['step']} loss {row['loss']:.1f}{''.join(f' {k} {float(v):.3g}' for k, v in parts.items())} edge_fidelity {row['edge_fidelity']:.3f} edge_keep {row['edge_keep']:.3f} psnr {row['psnr']:.2f} ({row['sec']}s)", flush=True)
    return model, curve, round(time.time() - t0, 1)


def sheet_tiles(native_show, ceil_show, ae_show, frames_at, patch):
    y, x, s = patch
    row = []
    for i in frames_at:
        for name, im in (("native", native_show[i]), ("ceiling", ceil_show[i]), ("AE", ae_show[i])):
            t = im[y[i]:y[i] + s, x[i]:x[i] + s]
            t = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR) if t.ndim == 2 else cv2.cvtColor(t, cv2.COLOR_RGB2BGR)
            t = cv2.resize(t, (200, 200), interpolation=cv2.INTER_NEAREST)
            cv2.putText(t, name, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            row.append(t)
    return np.concatenate(row, 1)


def run_clip(pack_file, key, out, heights, steps, batch, lr, device, seed, probe_every):
    z = np.load(pack_file)
    meta = json.loads(str(z["meta"]))
    native = z["native_eval"]                                          # (E,S,S,3) RGB or (E,S,S) gray
    masks = z["masks_eval"] if "masks_eval" in z.files else None
    s = native.shape[1]
    native_g = [gray_of(n) for n in native]
    native_show = list(native) if native.ndim == 4 else [n for n in native]
    patches = [C.detail_patch(g) for g in native_g]
    py, px = [p[0] for p in patches], [p[1] for p in patches]
    ps = patches[0][2]
    pick = [len(native) // 3, 2 * len(native) // 3]
    res, rows = {}, []
    for h in heights:
        train = z[f"train_{h}"]
        ceil_u8 = np.stack([C.down_up(n, h) for n in native])          # (E,S,S[,3]) shrink + cubic back, no model
        ceil = score(native_g, [gray_of(c) for c in ceil_u8], masks)
        x_eval = torch.from_numpy(train[z["eval_idx"]]).permute(0, 3, 1, 2).float() / 255.0
        probe = lambda m, h=h: score(native_g, up_to_native(reconstruct(m, x_eval, device), s), masks)
        print(f"[{key}] h={h}: {len(train)} train frames, ceiling edge_fidelity {ceil['edge_fidelity']:.3f} (edge_keep {ceil['edge_keep']:.3f})", flush=True)
        model, curve, sec = train_one(train, h, steps, batch, lr, device, seed, probe, probe_every)
        rec = reconstruct(model, x_eval, device)
        ae = score(native_g, up_to_native(rec, s), masks)
        gate = dict(edge_keep=bool(GATE * ceil["edge_keep"] <= ae["edge_keep"] <= KEEP_MAX), edge_fidelity=bool(ae["edge_fidelity"] >= GATE * ceil["edge_fidelity"]))
        if masks is not None:
            gate["pupil_edge"] = bool(ae["pupil_edge_grad_ratio"] >= GATE * ceil["pupil_edge_grad_ratio"])
        res[str(h)] = dict(ceiling=ceil, ae=ae, ae_over_ceiling_keep=ae["edge_keep"] / ceil["edge_keep"], ae_over_ceiling_fidelity=ae["edge_fidelity"] / ceil["edge_fidelity"], gate=gate, passed=bool(all(gate.values())),
                           curve=curve, train_seconds=sec, train_frames=int(len(train)), steps=steps)
        ae_show = [cv2.resize(r, (s, s), interpolation=cv2.INTER_CUBIC) for r in rec]
        ae_show = ae_show if native.ndim == 4 else [gray_of(a) for a in ae_show]
        ceil_show = list(ceil_u8) if native.ndim == 4 else [c for c in ceil_u8]
        rows.append(sheet_tiles(native_show, ceil_show, ae_show, pick, (py, px, ps)))
        print(f"[{key}] h={h}: AE edge_fidelity {ae['edge_fidelity']:.3f} vs ceiling {ceil['edge_fidelity']:.3f} (edge_keep {ae['edge_keep']:.3f} vs {ceil['edge_keep']:.3f}, psnr {ae['psnr']:.1f} vs {ceil['psnr']:.1f}) -> {'PASS' if res[str(h)]['passed'] else 'FAIL'}", flush=True)
    cv2.imwrite(str(Path(out) / f"sheet_{key}.png"), np.concatenate(rows, 0))
    return dict(meta={k: meta[k] for k in ("name", "kind", "native", "frames", "square")}, heights=res)


def train_main(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if a.device == "auto" and torch.cuda.is_available() else ("cpu" if a.device == "auto" else a.device))
    print(f"device={device} steps={a.steps} batch={a.batch} heights={a.heights} clips={a.clips}", flush=True)
    results = {}
    for key in a.clips:
        results[key] = run_clip(Path(a.pack) / f"{key}.npz", key, out, a.heights, a.steps, a.batch, a.lr, device, a.seed, a.probe_every)
        (out / f"results_{key}.json").write_text(json.dumps(results[key], indent=1))
    (out / "results.json").write_text(json.dumps(results, indent=1))
    for key, r in results.items():
        for h, v in r["heights"].items():
            print(f"{key} {h}px: AE/ceiling keep {v['ae_over_ceiling_keep']:.3f} fidelity {v['ae_over_ceiling_fidelity']:.3f}  {'PASS' if v['passed'] else 'FAIL'}  ({v['train_seconds']}s)")
    return results


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack")
    p.add_argument("--out", required=True)
    p.add_argument("--colour", default=COLOUR_CLIP)
    p.add_argument("--openeds", default=OPENEDS_CLIP)
    p.add_argument("--n-colour", type=int, default=240)
    t = sub.add_parser("train")
    t.add_argument("--pack", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--clips", nargs="+", default=list(CLIPS), choices=CLIPS)
    t.add_argument("--heights", type=int, nargs="+", default=list(HEIGHTS))
    t.add_argument("--steps", type=int, default=6000)
    t.add_argument("--batch", type=int, default=32)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--probe-every", type=int, default=1000)
    t.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    if a.cmd == "pack":
        return build_pack(a.out, colour=a.colour, openeds=a.openeds, n_colour=a.n_colour)
    return train_main(a)


if __name__ == "__main__":
    main()


