"""Dream pool: the okay models' own dreams as extra TRAINING clips (plan.md "SCRATCH"). CPU, writes a CacheBank-compatible cache (source 'dream', manifest entries marked train_only).
For every teacher (the eye / some tier 96 px models of ivs.modelzoo, or --models NAME=model.pt) and --contexts distinct TRAIN recordings (CacheBank split='train', never a validation
recording, so nothing the held-out scores use can leak in): the teacher perceives a real --context-len frame window, then dreams --ticks UI ticks (ivs.eyemetric.ui_dream_frames, zero action,
the loop the UI shows). A dream frame is KEPT when ALL hold: tick >= --min-tick (earlier ticks are soft copies of a real training frame, no new information), the eye segmenter calls it an eye
(smoothed over 5 ticks), the picture is not frozen (scripts/eval_ui_dream.window_change >= STILL), its Laplacian-variance sharpness >= --sharp-ratio x the median of real training frames.
Only runs of >= --min-run consecutive kept frames are stored (one sequence each, so training windows never straddle a gap). Colour class is INHERITED from the context: a grey context's dream
is written as luma (chroma 0) so a model's grey->colour drift is not taught as eye colour; a colour context's dream stays colour (chroma/hue_spread fields set to the 'colour' class).
Output: <out>/manifest.json + one npz per (teacher, context) + pool.json (kept frames per teacher, thresholds) + sheet_pool.png (rows = teachers, random kept frames): LOOK at it.
  python scripts/make_dream_pool.py --tiers eye some --out ../data/cache/dreams [--contexts 64 --ticks 200]
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
import make_birth as MB                                           # noqa: E402
from eval_ui_dream import STILL, window_change                    # noqa: E402
from ivs import eyemetric as E                                    # noqa: E402
from ivs import modelzoo                                          # noqa: E402
from ivs.birth import perceive                                    # noqa: E402
from ivs.cachedata import CLASSES, CacheBank                      # noqa: E402
from ivs.pilot import _LAP                                        # noqa: E402
from identity_curve import load                                   # noqa: E402

SMOOTH = 5


def frame_sharpness(x):
    """(N,3,H,W) -> (N,) variance of the Laplacian of the gray image, per frame (ivs.pilot.sharpness is one number per batch)."""
    g = x.mean(1, keepdim=True)
    return F.conv2d(g, _LAP.to(x.device)).flatten(1).var(1)


def smooth_flags(flags, k=SMOOTH):
    """(T,n) bool -> median filter of width k along time (drops one-tick flicker, as eyemetric.eye_episodes does)."""
    f = torch.as_tensor(flags).float().T.contiguous()                  # n,T
    pad = k // 2
    f = F.pad(f[:, None], (pad, pad), mode="replicate")[:, 0]
    return (f.unfold(1, k, 1).median(-1).values > 0.5).T.numpy()


def runs_of(mask, min_len):
    """[(start, end)) of the runs of True in a 1-D bool array that are at least min_len long."""
    out, start = [], None
    for i, v in enumerate(list(mask) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_len:
                out.append((start, i))
            start = None
    return out


def thumbs(x, s=24):
    """(N,3,H,W) -> (N, s*s) gray thumbnails (area resize)."""
    return F.interpolate(x.mean(1, keepdim=True), size=(s, s), mode="area").flatten(1)


def nn_each(a, b, chunk=2048):
    """(N,d) x (M,d) -> (N,) distance (RMS per pixel) from every row of a to its nearest row of b."""
    out = [(torch.cdist(a[i:i + chunk], b) / a.shape[1] ** 0.5).min(1).values for i in range(0, len(a), chunk)]
    return torch.cat(out)


def keep_mask(flags, still, sharp, sharp_thr, min_tick, plaus=None, plaus_thr=None):
    """(T,n) flags / still / sharp (/ plaus = distance to the nearest real frame) -> (T,n) bool of kept frames (tick t is row t-1)."""
    m = smooth_flags(flags) & ~still & (np.asarray(sharp) >= sharp_thr)
    if plaus is not None:
        m &= np.asarray(plaus) <= plaus_thr
    m[:max(0, min_tick - 1)] = False
    return m


def safe(s):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", s).strip("_")


def teachers_from(root, tiers, models):
    out = {}
    for e in modelzoo.discover(root):
        if e["tier"] in tiers and e["size"] == 96 and e["channels"] == 3:
            out[safe(f"{Path(e['id']).parent.name}_{Path(e['id']).stem}")] = str(Path(root) / e["id"])
    for spec in models or []:
        name, _, path = spec.partition("=")
        out[safe(name)] = path
    return out


def write_clip(out, teacher, ctx, frames, bounds, cls):
    """One npz + manifest entry for the kept runs of one (teacher, context) dream. frames (N,96,96,3) uint8."""
    group = safe(f"dream_{teacher}_{ctx['group']}")
    name = f"{group}.dream"
    colour = cls == "colour"
    meta = dict(name=name, group=group, source="dream", pad_mode="reflect", fps=20.0, stride=1, eff_fps=20.0, native=[96, 96], total_raw=int(len(frames)),
                crop_box=[0, 0, 0, 0], chroma=10.0 if colour else 0.0, hue_spread=0.05 if colour else 0.0, cuts=0, bad_frames=0, decoded=int(len(frames)),
                clean=int(len(frames)), kept=int(len(frames)), sequences=len(bounds) - 1, width=96, height=96, train_only=True,
                teacher=teacher, context_group=ctx["group"], context_source=ctx["source"], context_class=cls)
    np.savez_compressed(out / f"{group}.npz", frames=frames, seq_bounds=np.asarray(bounds, np.int64), meta=json.dumps(meta))
    return meta


@torch.no_grad()
def dream_teacher(model, seg, bank, ctxs, ticks, context_len, noise_seed, chunk, ref=None):
    """All contexts of one teacher -> frames (T,n,3,H,W), flags (T,n), still (T,n), sharp (T,n), plaus (T,n) (nearest-real-frame distance of the thumbnail, zeros without `ref`)."""
    fr_all, fl_all, st_all, sh_all, pl_all = [], [], [], [], []
    for i in range(0, len(ctxs), chunk):
        part = ctxs[i:i + chunk]
        wins = torch.stack([bank.window_at(c["seq"], c["offset"], context_len) for c in part])
        p = perceive(model, wins, seed=noise_seed + i)
        fr = E.ui_dream_frames(model, ticks=ticks, n=len(part), noise_seed=noise_seed + 1000 + i, init_state=(p["h"], p["z"]))
        st = MB._stats(fr, seg)
        still = window_change(fr) < STILL
        sh = frame_sharpness(fr.reshape(-1, *fr.shape[2:])).reshape(fr.shape[0], fr.shape[1]).numpy()
        pl = (nn_each(thumbs(fr.reshape(-1, *fr.shape[2:])), ref).reshape(fr.shape[0], fr.shape[1]).numpy() if ref is not None else np.zeros(sh.shape, np.float32))
        fr_all.append(fr); fl_all.append(st["flags"]); st_all.append(still); sh_all.append(sh); pl_all.append(pl)
    return torch.cat(fr_all, 1), np.concatenate(fl_all, 1), np.concatenate(st_all, 1), np.concatenate(sh_all, 1), np.concatenate(pl_all, 1)


def plausibility_reference(bank, plaus_q, seed):
    """(ref thumbnails of 4096 real train frames, threshold). Threshold = the plaus_q quantile of the nearest-real-frame distance of real frames of OTHER recordings (odd groups queried
    against even groups, so a frame never finds its own near-duplicates): 'plausible' = as close to real eyes as real eyes are to each other."""
    frame_group = torch.as_tensor(bank.seq_group[np.searchsorted(bank.seq_start, np.arange(len(bank)), side="right") - 1])
    x0 = (bank.store_w - bank.size) // 2
    view = lambda idx: bank.frames[idx][:, :, x0:x0 + bank.size].permute(0, 3, 1, 2).float() / 255.0
    rng = np.random.RandomState(seed)
    pick = lambda idx, n: idx[torch.as_tensor(rng.choice(len(idx), min(n, len(idx)), replace=False))]
    even, odd = torch.nonzero(frame_group % 2 == 0)[:, 0], torch.nonzero(frame_group % 2 == 1)[:, 0]
    thr = float(torch.quantile(nn_each(thumbs(view(pick(odd, 1500))), thumbs(view(pick(even, 3000)))), plaus_q))
    return thumbs(view(pick(torch.arange(len(bank)), 4096))), thr


def build(teachers, caches, out, contexts, ticks, context_len, min_tick, min_run, sharp_ratio, seed, segmenter, plaus_q=0.9, chunk=32, log=print):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    seg = E.load_segmenter(segmenter)
    bank = CacheBank(caches, image_size=96, split="train", val_frac=0.15, seed=0, tint_mode="gray", sources=["video", "openeds"], hflip=False)
    g = torch.Generator().manual_seed(seed)
    real = bank.sample(1024, 1, generator=g, train=False)["images"][:, 0]
    sharp_thr = float(sharp_ratio * frame_sharpness(real).median())
    ref, plaus_thr = plausibility_reference(bank, plaus_q, seed)
    log(f"train bank {bank.info()['groups']} groups; real median sharpness {float(frame_sharpness(real).median()):.5f} -> threshold {sharp_thr:.5f}; plausibility threshold {plaus_thr:.4f} (q{plaus_q})")
    manifest, stats, picks = [], {}, {}
    for ti, (name, path) in enumerate(sorted(teachers.items())):
        t0 = time.time()
        model = load(path)
        ctxs = MB.candidates(bank, context_len, contexts, seed + ti)
        fr, flags, still, sharp, plaus = dream_teacher(model, seg, bank, ctxs, ticks, context_len, seed + 100 * ti, chunk, ref)
        mask = keep_mask(flags, still, sharp, sharp_thr, min_tick, plaus, plaus_thr)
        n_kept = 0
        for k, c in enumerate(ctxs):
            runs = runs_of(mask[:, k], min_run)
            if not runs:
                continue
            cls = CLASSES[int(bank.seq_class[c["seq"]])]
            pieces, bounds = [], [0]
            for a, b in runs:
                x = fr[a:b, k].clamp(0, 1)
                if cls != "colour":
                    x = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]).expand(-1, 3, -1, -1)
                pieces.append((x.permute(0, 2, 3, 1).numpy() * 255).round().astype(np.uint8))
                bounds.append(bounds[-1] + (b - a))
            frames = np.concatenate(pieces)
            manifest.append(write_clip(out, name, c, frames, bounds, cls))
            n_kept += len(frames)
            picks.setdefault(name, []).append(frames)
        stats[name] = dict(path=path, contexts=len(ctxs), dreamed=int(fr.shape[0] * fr.shape[1]), kept=n_kept, keep_rate=round(n_kept / (fr.shape[0] * fr.shape[1]), 3),
                           eye_flag_rate=round(float(smooth_flags(flags).mean()), 3), still_rate=round(float(still.mean()), 3), sec=round(time.time() - t0, 1))
        log(f"{name:40s} kept {n_kept:6d} / {fr.shape[0] * fr.shape[1]} ({stats[name]['keep_rate']:.2f}) eye {stats[name]['eye_flag_rate']:.2f} still {stats[name]['still_rate']:.2f} [{stats[name]['sec']}s]")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    info = dict(teachers=stats, total_frames=sum(s["kept"] for s in stats.values()), clips=len(manifest), contexts=contexts, ticks=ticks, context_len=context_len, min_tick=min_tick,
                min_run=min_run, sharp_ratio=sharp_ratio, sharp_thr=sharp_thr, plaus_q=plaus_q, plaus_thr=plaus_thr, seed=seed, val_groups_used=0)
    (out / "pool.json").write_text(json.dumps(info, indent=1))
    sheet(picks, out / "sheet_pool.png", np.random.RandomState(seed))
    return info


def sheet(picks, path, rng, cols=8, tile=72):
    rows = []
    for name in sorted(picks):
        allf = np.concatenate(picks[name])
        sel = allf[rng.choice(len(allf), min(cols, len(allf)), replace=False)]
        row = [cv2.resize(f, (tile, tile), interpolation=cv2.INTER_AREA) for f in sel] + [np.zeros((tile, tile, 3), np.uint8)] * (cols - len(sel))
        r = np.concatenate(row, 1)
        cv2.putText(r, name[:30], (2, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)
        rows.append(r)
    if rows:
        cv2.imwrite(str(path), cv2.cvtColor(np.concatenate(rows), cv2.COLOR_RGB2BGR))


def curate(out, drop, log=print):
    """Visual curation of a finished pool (plan.md SCRATCH): clips of the teachers in `drop` are MOVED to <out>/rejected (never deleted), manifest.json and pool.json are rewritten.
    Returns the number of frames kept."""
    out = Path(out)
    man = json.loads((out / "manifest.json").read_text())
    (out / "rejected").mkdir(exist_ok=True)
    keep, moved = [], 0
    for m in man:
        if m["teacher"] in drop:
            (out / (m["group"] + ".npz")).replace(out / "rejected" / (m["group"] + ".npz"))
            moved += 1
        else:
            keep.append(m)
    (out / "manifest.rejected.json").write_text(json.dumps([m for m in man if m["teacher"] in drop], indent=1))
    (out / "manifest.json").write_text(json.dumps(keep, indent=1))
    info = json.loads((out / "pool.json").read_text())
    info["dropped_teachers"] = sorted(drop)
    info["clips"] = len(keep)
    info["total_frames"] = sum(s["kept"] for t, s in info["teachers"].items() if t not in drop)
    (out / "pool.json").write_text(json.dumps(info, indent=1))
    log(f"curated: moved {moved} clips of {sorted(drop)} to {out / 'rejected'}; {len(keep)} clips / {info['total_frames']} frames remain")
    return info["total_frames"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", nargs="*", default=[], help="modelzoo curation tiers to use as teachers (eye / some); 96 px RGB models only")
    ap.add_argument("--models", nargs="*", default=[], help="extra NAME=model.pt teachers")
    ap.add_argument("--cache", nargs="+", default=[str(HERE.parent / "data" / "cache" / "videos"), str(HERE.parent / "data" / "cache" / "openeds")])
    ap.add_argument("--segmenter", default=str(HERE / "pilot_results" / "eyemetric" / "segmenter.pt"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--contexts", type=int, default=64)
    ap.add_argument("--ticks", type=int, default=200)
    ap.add_argument("--context-len", type=int, default=16)
    ap.add_argument("--min-tick", type=int, default=16)
    ap.add_argument("--min-run", type=int, default=16)
    ap.add_argument("--sharp-ratio", type=float, default=0.03, help="x the median Laplacian variance of real train frames (dreams are far softer: adv .15, W3 .07, L1 .02, LR2 .01; plan.md SCRATCH)")
    ap.add_argument("--plaus-q", type=float, default=0.9, help="quantile of real-to-real nearest-frame distance a dream frame must stay under")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--curate", nargs="+", default=None, metavar="TEACHER", help="do not build: move these teachers' clips of the existing pool at --out into <out>/rejected")
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    out = Path(a.out)
    if a.curate:
        curate(out, set(a.curate))
        return
    if (out / "manifest.json").exists() and not a.force:
        raise SystemExit(f"{out} already holds a pool; pass --force to rebuild it (nothing is deleted otherwise)")
    teachers = teachers_from(HERE, set(a.tiers), a.models)
    if not teachers:
        raise SystemExit("no teachers: give --tiers and/or --models")
    print(f"{len(teachers)} teachers: {sorted(teachers)}", flush=True)
    info = build(teachers, a.cache, out, a.contexts, a.ticks, a.context_len, a.min_tick, a.min_run, a.sharp_ratio, a.seed, a.segmenter, plaus_q=a.plaus_q, log=lambda s: print(s, flush=True))
    print(f"pool: {info['total_frames']} frames in {info['clips']} clips")


if __name__ == "__main__":
    main()
