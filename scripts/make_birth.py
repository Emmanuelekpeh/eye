"""Make a birth memory for a trained world model (ivs.birth): the model perceives real HELD-OUT eye windows once and we keep only its own
latent state (h, z). The UI then starts from one of those states (python -m ivs.ui --checkpoint M --birth FILE) instead of from noise.

Procedure (everything on the model's own UI dream loop, ivs.eyemetric.ui_dream_frames, which is parity-tested against ivs.ui.World):
  1. one window of --context frames per held-out recording (group) of the validation split (CacheBank split='val', the split the pilot used),
  2. the model observes it with zero action -> candidate state; its render is scored against the real last frame (MSE) and the eye segmenter,
  3. every candidate is dreamed for --dwell-ticks UI ticks; score = share of ticks that look like a moving eye (eye flag AND not frozen, the
     conservative measure from scripts/eval_ui_dream.py) -> the best --k states, at most one per recording, are saved,
  4. HONESTY CHECK: the saved states are dreamed again with a DIFFERENT tick-noise seed and compared with the same number of noise starts
     (what the UI did before) and with the average candidate; the selection never saw that run.
Writes the .pt, a sheet (real last frame | birth render | dream at several ticks, rows = saved states, green frame = flagged eye) and a json.
The detector is lenient (it can call a blob an eye): LOOK at the sheet.
CPU only; a few minutes for 64 candidates. Usage:
  python scripts/make_birth.py --checkpoint pilot_results/kaggle_long3_30k/pilot/L1_v6_long/model.pt --out checkpoints/birth_L1.pt
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
from ivs import eyemetric as E                      # noqa: E402
from ivs.birth import perceive, save_birth, to_model_channels   # noqa: E402
from ivs.cachedata import CacheBank                 # noqa: E402
from ivs.config import ModelConfig                  # noqa: E402
from ivs.model import EyeWorldModel                 # noqa: E402
from eval_ui_dream import STILL, window_change      # noqa: E402


def dwell(model, seg, state, ticks, noise_seed):
    """UI dream (n = len(state[0]) worlds) from the latent `state` = (h, z). Returns frames (T,n,3,H,W) and the per-world statistics of _stats."""
    fr = E.ui_dream_frames(model, ticks=ticks, n=len(state[0]), noise_seed=noise_seed, init_state=state)
    return fr, _stats(fr, seg)


def _stats(fr, seg):
    T, n = fr.shape[:2]
    flat = fr.reshape(T * n, *fr.shape[2:])
    cls = torch.cat([E.segment(seg, E.gray(flat[i:i + 512])) for i in range(0, len(flat), 512)])
    flags = E.nested_flags(cls).reshape(T, n).numpy()
    still = window_change(fr) < STILL
    per = [E.eye_episodes(flags[:, k]) for k in range(n)]
    return dict(flags=flags, eye_frac=np.array([p["eye_frac"] for p in per]), max_len=np.array([p["max_len"] for p in per]),
                n_long=np.array([p["n_long"] for p in per]), still_frac=still.mean(0),
                eye_moving=(flags & ~still).mean(0), first100=flags[:100].mean(0))


def noise_dwell(model, seg, n, ticks, noise_seed):
    fr = E.ui_dream_frames(model, ticks=ticks, n=n, noise_seed=noise_seed)
    return fr, _stats(fr, seg)


def candidates(bank, context, n_max, seed):
    """One (seq, offset) per held-out group that has a sequence of >= context frames, then a seeded subsample of n_max groups."""
    rng = np.random.RandomState(seed)
    names = bank.group_names()
    out = []
    for g in sorted(names):
        seqs = [s for s in np.flatnonzero(bank.seq_group == g) if bank.seq_len[s] >= context]
        if seqs:
            s = int(rng.choice(seqs))
            out.append(dict(group=names[g], source=bank.group_source[g], seq=s,
                            offset=int(rng.randint(0, int(bank.seq_len[s]) - context + 1))))
    if len(out) > n_max:
        out = [out[i] for i in sorted(rng.choice(len(out), n_max, replace=False))]
    return out


def tile(img, flag, label, size):
    a = (img.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
    a = np.ascontiguousarray(cv2.resize(a, (size, size), interpolation=cv2.INTER_CUBIC))
    if flag:
        cv2.rectangle(a, (0, 0), (size - 1, size - 1), (0, 255, 0), 2)
    cv2.putText(a, label, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
    return a


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cache", nargs="+", default=[str(HERE.parent / "data" / "cache" / "videos"), str(HERE.parent / "data" / "cache" / "openeds")])
    ap.add_argument("--segmenter", default=str(HERE / "pilot_results" / "eyemetric" / "segmenter.pt"))
    ap.add_argument("--out", required=True, help="birth memory .pt (never overwrites an existing file unless --force)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--candidates", type=int, default=64)
    ap.add_argument("--context", type=int, default=16, help="real frames the model perceives (training windows went up to 32)")
    ap.add_argument("--dwell-ticks", type=int, default=300)
    ap.add_argument("--check-ticks", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.15, help="must equal the training run's split (pilot default)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists() and not a.force:
        raise SystemExit(f"{out} exists; pass --force to replace it (nothing is deleted otherwise)")
    t0 = time.time()
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ck["config"])
    model = EyeWorldModel(cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    seg = E.load_segmenter(a.segmenter)
    bank = CacheBank(a.cache, image_size=cfg.image_size, split="val", val_frac=a.val_frac, seed=0, tint_mode="gray", sources=["video", "openeds"])
    cands = candidates(bank, a.context, a.candidates, a.seed)
    if len(cands) < a.k:
        raise SystemExit(f"only {len(cands)} held-out recordings have {a.context} frames; need k={a.k}")
    wins = torch.stack([bank.window_at(c["seq"], c["offset"], a.context) for c in cands])          # K,T,3,H,W
    p = perceive(model, wins, seed=a.seed)
    real_last = to_model_channels(wins[:, -1], cfg.channels)
    mse = ((p["recon"] - real_last) ** 2).mean((1, 2, 3)).numpy()
    recon3 = p["recon"] if p["recon"].shape[1] == 3 else p["recon"].expand(-1, 3, -1, -1)
    birth_eye = E.nested_flags(E.segment(seg, E.gray(recon3))).numpy()
    print(f"{len(cands)} candidates ({sum(c['source'] == 'video' for c in cands)} video / {sum(c['source'] == 'openeds' for c in cands)} openeds), "
          f"birth render flagged eye {birth_eye.mean():.2f}, recon mse mean {mse.mean():.4f}  [{time.time() - t0:.0f}s]", flush=True)

    _, st = dwell(model, seg, (p["h"], p["z"]), a.dwell_ticks, noise_seed=a.seed + 21)
    print(f"selection dream ({a.dwell_ticks} ticks) done  [{time.time() - t0:.0f}s]", flush=True)
    # rank: moving-eye share first (ignores frozen blobs), then a better birth render
    order = sorted(range(len(cands)), key=lambda i: (-(st["eye_moving"][i]), mse[i]))
    sel = [i for i in order if birth_eye[i]][:a.k]
    if len(sel) < a.k:
        sel += [i for i in order if i not in sel][:a.k - len(sel)]
        print(f"WARNING: only {int(birth_eye.sum())} candidate renders were flagged as eyes; padded with the next best", flush=True)
    sel = sorted(sel, key=lambda i: order.index(i))

    h, z = p["h"][sel], p["z"][sel]
    chk_seed = a.seed + 1000
    fr, ch = dwell(model, seg, (h, z), a.check_ticks, chk_seed)            # fresh tick noise: the selection never saw it
    _, nz = noise_dwell(model, seg, len(sel), a.check_ticks, chk_seed)
    _, allc = dwell(model, seg, (p["h"], p["z"]), a.check_ticks, chk_seed)
    row = lambda s: {k: round(float(np.mean(s[k])), 3) for k in ("eye_frac", "eye_moving", "still_frac", "first100", "max_len", "n_long")}
    summary = dict(saved_birth_fresh_noise=row(ch), all_candidates_fresh_noise=row(allc), noise_start_fresh_noise=row(nz),
                   selection=dict(dwell_ticks=a.dwell_ticks, check_ticks=a.check_ticks, candidates=len(cands), k=len(sel), context=a.context))
    print(json.dumps(summary, indent=1), flush=True)

    meta = dict(group=[cands[i]["group"] for i in sel], source=[cands[i]["source"] for i in sel], seq=[cands[i]["seq"] for i in sel],
                offset=[cands[i]["offset"] for i in sel], context=a.context, recon_mse=[round(float(mse[i]), 5) for i in sel],
                selection_eye_moving=[round(float(st["eye_moving"][i]), 3) for i in sel], seed=a.seed,
                note="held-out windows perceived once; only the latent state is stored")
    save_birth(out, h, z, a.checkpoint, meta)
    sheet_path = out.with_suffix(".sheet.png")
    marks = [0, 10, 25, 50, 100, 200, 300, 450, 599]
    marks = [m for m in marks if m < a.check_ticks]
    rows = []
    for r, i in enumerate(sel):
        tiles = [tile(wins[i, -1], False, "real", 100), tile(p["recon"][i].expand(3, -1, -1) if cfg.channels == 1 else p["recon"][i], bool(birth_eye[i]), "birth", 100)]
        tiles += [tile(fr[m, r], bool(ch["flags"][m, r]), f"t{m}", 100) for m in marks]
        rows.append(np.concatenate(tiles, 1))
    cv2.imwrite(str(sheet_path), cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR))
    out.with_suffix(".json").write_text(json.dumps(dict(summary=summary, meta=meta), indent=1))
    print(f"wrote {out}, {sheet_path}  [{time.time() - t0:.0f}s]")
    return summary


if __name__ == "__main__":
    main()
