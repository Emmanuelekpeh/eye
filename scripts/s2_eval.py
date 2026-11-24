"""R5 stage 2 evaluation (CPU): L1 vs the stage-2 arms (ctl, adv, and their 1/3, 2/3 checkpoints) on the SAME protocol as scripts/r4_baseline.py (plan.md "R5 STAGE 2"):
one 16-frame real context per held-out recording (CacheBank split='val', 42 groups), the UI dream loop (ivs.eyemetric.ui_dream_frames, parity-tested against ivs.ui.World) for
--ticks ticks, scored by make_birth._stats (eye_frac / eye_moving / still_frac / first100 / max_len / n_long). Adds: sharpness of the dream vs the ORIGINAL L1's own render of the real
last frame (same reference for every model) and contrast vs the real frame, held-out posterior recon / open-loop prediction (ivs.pilot.eval_heldout), touch response in the UI loop vs
copy-last (shift_x, shift_y, gain, light_x), paired bootstrap over worlds for the arm differences, the pre-registered gates S2-G1..G5 (G6 = the sheets, looked at by a person),
per-model dream sheets and one comparison sheet.
  python scripts/s2_eval.py --models L1=<L1 model.pt> ctl=<ctl/model.pt> adv=<adv/model.pt> [adv_s2=...] --out pilot_results/kaggle_r5s2/eval
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
import make_birth as MB                                          # noqa: E402
from ivs import advdream as A                                    # noqa: E402
from ivs import eyemetric as E                                   # noqa: E402
from ivs.actions import InterventionSampler                      # noqa: E402
from ivs.birth import perceive                                   # noqa: E402
from ivs.cachedata import CacheBank                              # noqa: E402
from ivs.config import ModelConfig                               # noqa: E402
from ivs.model import EyeWorldModel                              # noqa: E402
from ivs.pilot import build_run, eval_heldout, sharpness         # noqa: E402

KEYS = ("eye_frac", "eye_moving", "still_frac", "first100", "max_len", "n_long")
TOUCH = {"shift_x": (0, 1.0), "shift_y": (1, 1.0), "gain": (2, 1.0), "light_x": (3, -1.0)}      # same pushes as R4 (r4_latent.TOUCH)
TOUCH_TICKS, TOUCH_READ, S0 = 3, (3, 8, 16), (0.0, 0.0, 0.0, 0.4, -0.4)
SHEET_TICKS = (1, 5, 15, 30, 60, 100, 200, 299)
COMPARE_TICKS = (1, 30, 100, 299)
SHARP_TICKS = (29, 99, 299)
# pre-registered gates (plan.md R5 STAGE 2)
G1 = dict(eye_frac=0.65, first100=0.65, max_len=90.0)
G2_GAIN, G3_STILL = 0.08, 0.05
G4_SHARP, G4_CONTRAST = (0.7, 1.3), 0.7
G5_RECON, G5_TOUCH = 1.10, 0.15


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = EyeWorldModel(ModelConfig(**ck["config"]))
    model.load_state_dict(ck["model"])
    return model.eval(), ck


def paired_bootstrap(a, b, n=2000, seed=0):
    """Mean of (a - b) over worlds with a bootstrap over worlds: (mean, 5th percentile, 95th percentile)."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.RandomState(seed)
    boots = d[rng.randint(0, len(d), (n, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(boots, 5)), float(np.percentile(boots, 95))


@torch.no_grad()
def touch_test(model, ctx_real, device="cpu"):
    """3-tick push of one touch dimension (action +-1) from a static touched state, in the UI loop (the action enters the prediction and the perception, like ivs.ui). Truth = the
    last real frame under the new state. ratio = mse(dream, truth) / mse(copy-last, truth) at ticks 3 / 8 / 16 (< 1: the touch moved the picture the right way)."""
    isamp = InterventionSampler()
    b, tc = ctx_real.shape[:2]
    ticks = max(TOUCH_READ)
    s0 = torch.tensor(S0).view(1, 1, 5).expand(b, 1, 5)
    ctx_i = isamp.apply_state(ctx_real, s0.expand(b, tc, 5))
    base = ctx_real[:, -1:]
    step_unit = isamp.step * isamp.limit / isamp.limit[0]
    torch.manual_seed(0)
    out = model.observe(ctx_i, torch.zeros(b, tc, 5))
    res = {}
    for name, (dim, sign) in TOUCH.items():
        act = torch.zeros(b, ticks, 5)
        act[:, :TOUCH_TICKS, dim] = sign
        states = s0.expand(b, ticks, 5).clone()
        for t in range(ticks):
            states[:, t] = (states[:, t - 1] if t else s0[:, 0]) + act[:, t] * step_unit
        states = torch.max(torch.min(states, isamp.limit), -isamp.limit)
        truth = isamp.apply_state(base.expand(b, ticks, *base.shape[2:]), states)
        copy = ctx_i[:, -1:].expand(-1, ticks, -1, -1, -1)
        torch.manual_seed(1)
        fr, _ = A.ui_rollout(model, out["h"][:, -1], out["z"][:, -1], act)
        res[name] = {str(r): dict(dream=float(((fr[:, r - 1] - truth[:, r - 1]) ** 2).mean()), copy_last=float(((copy[:, r - 1] - truth[:, r - 1]) ** 2).mean())) for r in TOUCH_READ}
        for r in TOUCH_READ:
            res[name][str(r)]["ratio"] = res[name][str(r)]["dream"] / max(res[name][str(r)]["copy_last"], 1e-9)
    return res


def _pairwise(x):
    """Mean squared difference between different rows of x (n,C,H,W)."""
    f = x.flatten(1)
    d = torch.cdist(f, f).pow(2) / f.shape[1]
    n = len(f)
    return float(d.sum() / (n * (n - 1)))


def collapse_stats(fr, wins, ticks=(99, 299)):
    """Does the dream remember WHICH eye it started from, and are different worlds still different? (not gated; added after the stage-2 sheets showed every world converging on one picture)
    diversity_ratio = pairwise mse between worlds' dream frames at tick t / the same for the real last context frames (1 = as varied as real eyes, ~0 = one picture);
    own_vs_other = mse(dream_t, own context's last frame) / mean mse(dream_t, the other worlds' context frames) (< 1: still resembles its own start; ~1: no memory of it);
    motion_200_299 = mean |frame_t - frame_t-1| over ticks 200-299; chroma = mean |channel difference| of the dream at tick t (real context frames: gray recordings are ~0)."""
    from ivs.pilot import chroma
    real = wins[:, -1]
    base = _pairwise(real)
    out = dict(real_pairwise_mse=base, real_chroma=chroma(real))
    n = len(real)
    for t in ticks:
        if t >= fr.shape[0]:
            continue
        x = fr[t]
        d = ((x.flatten(1)[:, None] - real.flatten(1)[None]) ** 2).mean(-1)               # (n dream, n context)
        own = d.diag().mean()
        other = (d.sum() - d.diag().sum()) / (n * (n - 1))
        out[str(t)] = dict(diversity_ratio=_pairwise(x) / base, own_vs_other=float(own / other), chroma=chroma(x))
    if fr.shape[0] >= 300:
        out["motion_200_299"] = float((fr[201:300] - fr[200:299]).abs().mean())
    return out


def score_model(name, model, ck, wins, seg, ref_sharp, real_last, val, ticks, seed):
    t0 = time.time()
    p = perceive(model, wins, seed=seed)
    fr, st = MB.dwell(model, seg, (p["h"], p["z"]), ticks, noise_seed=seed + 1000)
    contrast = lambda x: float(x.std(dim=(1, 2, 3)).mean())
    row = dict(name=name, mean={k: round(float(np.mean(st[k])), 4) for k in KEYS}, per_world={k: [float(v) for v in st[k]] for k in KEYS},
               sharp_ratio={str(t): sharpness(fr[t]) / ref_sharp for t in SHARP_TICKS if t < ticks},
               contrast_ratio={str(t): contrast(fr[t]) / contrast(real_last) for t in SHARP_TICKS if t < ticks})
    row["collapse"] = collapse_stats(fr, wins)
    rc = build_run("L1_v6_long", 10)
    row["heldout"] = eval_heldout(model, rc, val, ck["mean_image"], n=192)
    row["touch"] = touch_test(model, wins)
    row["seconds"] = round(time.time() - t0)
    print(f"[{name}] dream {row['mean']} | sharp/ref {({k: round(v, 3) for k, v in row['sharp_ratio'].items()})} contrast/real {({k: round(v, 2) for k, v in row['contrast_ratio'].items()})} | recon {row['heldout']['passive']['recon']:.5f} "
          f"| touch@8 {({k: round(v['8']['ratio'], 2) for k, v in row['touch'].items()})} | collapse {json.dumps(row['collapse'], default=lambda v: round(v, 4))} [{row['seconds']}s]", flush=True)
    return row, fr, st


def gates(res):
    """Pre-registered S2-G1..G5 for the arm 'adv' against 'ctl' and 'L1' (names as given with --models). Pure function of the per-model rows (tested)."""
    adv, ctl, l1 = res["adv"], res["ctl"], res["L1"]
    m = adv["mean"]
    g1 = m["eye_frac"] >= G1["eye_frac"] and m["first100"] >= G1["first100"] and m["max_len"] >= G1["max_len"]
    d_ef, lo_ef, _ = paired_bootstrap(adv["per_world"]["eye_frac"], ctl["per_world"]["eye_frac"])
    d_f1, lo_f1, _ = paired_bootstrap(adv["per_world"]["first100"], ctl["per_world"]["first100"])
    g2 = d_ef >= G2_GAIN and d_f1 >= G2_GAIN and lo_ef > 0
    g3 = m["still_frac"] <= G3_STILL and m["eye_moving"] >= l1["mean"]["eye_moving"]
    sr, cr = adv["sharp_ratio"].get("99"), adv["contrast_ratio"].get("99")
    g4 = sr is not None and G4_SHARP[0] <= sr <= G4_SHARP[1] and cr >= G4_CONTRAST
    rec = adv["heldout"]["passive"]["recon"] <= G5_RECON * l1["heldout"]["passive"]["recon"]
    touch = all(adv["touch"][k]["8"]["ratio"] <= l1["touch"][k]["8"]["ratio"] + G5_TOUCH for k in TOUCH)
    g5 = bool(rec and touch)
    return dict(G1_persistence=bool(g1), G2_attribution=bool(g2), G3_alive=bool(g3), G4_sharp_not_grain=bool(g4), G5_no_regression=g5, G6="LOOK at the sheets",
                detail=dict(adv_minus_ctl=dict(eye_frac=round(d_ef, 3), eye_frac_p5=round(lo_ef, 3), first100=round(d_f1, 3), first100_p5=round(lo_f1, 3)),
                            adv_minus_L1=dict(eye_frac=round(paired_bootstrap(adv["per_world"]["eye_frac"], l1["per_world"]["eye_frac"])[0], 3),
                                              first100=round(paired_bootstrap(adv["per_world"]["first100"], l1["per_world"]["first100"])[0], 3)),
                            ctl_minus_L1=dict(eye_frac=round(paired_bootstrap(ctl["per_world"]["eye_frac"], l1["per_world"]["eye_frac"])[0], 3),
                                              first100=round(paired_bootstrap(ctl["per_world"]["first100"], l1["per_world"]["first100"])[0], 3)),
                            recon_ratio_to_L1=round(adv["heldout"]["passive"]["recon"] / l1["heldout"]["passive"]["recon"], 3), recon_ok=bool(rec), touch_ok=bool(touch)))


def sheets(out, names, frames, stats, wins, sel):
    size = wins.shape[-1]
    for n in names:
        fr, st = frames[n], stats[n]
        rows = [np.concatenate([MB.tile(wins[i, -1], False, "real", size)] + [MB.tile(fr[t, i], bool(st["flags"][t, i]), f"t{t}", size) for t in SHEET_TICKS if t < fr.shape[0]], 1) for i in sel]
        cv2.imwrite(str(Path(out) / f"sheet_{n}.png"), cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR))
    rows = []
    for i in sel:
        cells = [MB.tile(wins[i, -1], False, "real", size)]
        for n in names:
            cells += [MB.tile(frames[n][t, i], bool(stats[n]["flags"][t, i]), f"{n[:6]} t{t}", size) for t in COMPARE_TICKS if t < frames[n].shape[0]]
        rows.append(np.concatenate(cells, 1))
    cv2.imwrite(str(Path(out) / "compare.png"), cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="name=path ...; the first one is L1 (names L1, ctl, adv enable the gates)")
    ap.add_argument("--cache", nargs="+", default=[str(HERE.parent / "data" / "cache" / "videos"), str(HERE.parent / "data" / "cache" / "openeds")])
    ap.add_argument("--segmenter", default=str(HERE / "pilot_results" / "eyemetric" / "segmenter.pt"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--context", type=int, default=16)
    ap.add_argument("--ticks", type=int, default=300)
    ap.add_argument("--candidates", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sheet-worlds", type=int, default=8)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    specs = [m.split("=", 1) for m in a.models]
    torch.set_num_threads(max(1, min(6, torch.get_num_threads())))
    seg = E.load_segmenter(a.segmenter)
    val = CacheBank(a.cache, image_size=96, split="val", val_frac=0.15, seed=0, tint_mode="gray", sources=["video", "openeds"])
    cands = MB.candidates(val, a.context, a.candidates, a.seed)
    wins = torch.stack([val.window_at(c["seq"], c["offset"], a.context) for c in cands])
    ref_model, _ = load(specs[0][1])
    ref_recon = perceive(ref_model, wins, seed=a.seed)["recon"]
    ref_sharp, real_last = sharpness(ref_recon), wins[:, -1]
    print(f"{len(cands)} held-out worlds ({sum(c['source'] == 'video' for c in cands)} video / {sum(c['source'] == 'openeds' for c in cands)} openeds); reference = {specs[0][0]}'s own render of the real last frame, sharpness {ref_sharp:.5f}", flush=True)
    res, frames, stats = {}, {}, {}
    for name, path in specs:
        model, ck = load(path)
        res[name], frames[name], stats[name] = score_model(name, model, ck, wins, seg, ref_sharp, real_last, val, a.ticks, a.seed)
        res[name]["checkpoint"] = path
        res[name]["stage2"] = ck.get("stage2")
    sel = sorted(set(range(0, len(cands), max(1, len(cands) // a.sheet_worlds))))[:a.sheet_worlds]
    sheets(out, [n for n, _ in specs], frames, stats, wins, sel)
    verdict = gates(res) if all(k in res for k in ("L1", "ctl", "adv")) else {}
    result = dict(n_worlds=len(cands), ticks=a.ticks, context=a.context, sources=[c["source"] for c in cands], groups=[c["group"] for c in cands], models=res, verdict=verdict)
    (out / "s2_eval.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(verdict, indent=1), flush=True)
    return result


if __name__ == "__main__":
    main()
