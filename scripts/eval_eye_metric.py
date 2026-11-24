"""Pupil/iris-aware evaluation of world-model / AE checkpoints on held-out S_ subjects (CPU). See ivs/eyemetric.py.

  python scripts/eval_eye_metric.py --wm V6=<model.pt> --wm V1=<model.pt> [--ae AE128=<model.pt>] [--dreams 64] --out pilot_results/eyemetric_x
First run trains the segmenter (CPU, a few minutes) and caches it at --segmenter; its quality on REAL val frames (and blurred real
frames) is written to eye_metric.json as `segmenter_ceiling`. Needs the aligned mask cache (build_cache_openeds.py --masks).
Writes eye_metric.json + sheet.png (rows: real | ground-truth mask | per model: recon, segmentation of recon) + sheet_columns.txt,
and with --dreams also sheet_dreams_<name>.png (rows = noise seeds, cols = t 0,10,30,100,300) + `dreams` stats vs real frames;
with --response N and an 8-dim-action (gaze command) model: `command_response` (gain / sign / tracking of the pupil when the command target moves, see
eyemetric.command_response) + sheet_response_<name>.png (cols: last context frame | x-0.15 | held real target | x+0.15 | y-0.15 | y+0.15 | no command, imagined step 8).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ivs import eyemetric as E  # noqa: E402
from ivs.cachedata import CacheBank  # noqa: E402
from ivs.pilot import save_sheet  # noqa: E402

HERE = Path(__file__).resolve().parents[1]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(HERE.parent / "data" / "cache" / "openeds"))
    ap.add_argument("--masks", default=str(HERE.parent / "data" / "cache" / "openeds_masks"))
    ap.add_argument("--segmenter", default=str(HERE / "pilot_results" / "eyemetric" / "segmenter.pt"))
    ap.add_argument("--seg-steps", type=int, default=2500)
    ap.add_argument("--wm", nargs="*", action="extend", default=[], help="NAME=path/to/model.pt (world model)")
    ap.add_argument("--ae", nargs="*", action="extend", default=[], help="NAME=path/to/AE_model.pt (plain AE probe)")
    ap.add_argument("--n", type=int, default=384, help="held-out windows")
    ap.add_argument("--t", type=int, default=8, help="window length the world model observes (metric on the last frame)")
    ap.add_argument("--dreams", type=int, default=0, help="noise-seed dreams per world model (0 = skip)")
    ap.add_argument("--response", type=int, default=0, help="windows for the gaze-command response test of models with the 8-dim action (0 = skip)")
    ap.add_argument("--out", default=str(HERE / "pilot_results" / "eyemetric_out"))
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    res = {"n_windows": a.n, "T": a.t}

    seg_path = Path(a.segmenter)
    data = None
    if seg_path.exists():
        seg = E.load_segmenter(seg_path)
        res["segmenter"] = {"path": str(seg_path), "loaded": True}
    else:
        t0 = time.time()
        data = E.load_masked_gray(a.cache, a.masks)
        print(f"segmenter data: train {data['train'][0].shape[0]} val {data['val'][0].shape[0]} frames", flush=True)
        seg = E.train_segmenter(data, steps=a.seg_steps)
        seg_path.parent.mkdir(parents=True, exist_ok=True)
        E.save_segmenter(seg, seg_path, dict(steps=a.seg_steps))
        res["segmenter"] = {"path": str(seg_path), "trained_s": round(time.time() - t0)}
    ceil_path = seg_path.with_suffix(".ceiling.json")
    if not ceil_path.exists():
        data = data or E.load_masked_gray(a.cache, a.masks)
        ceil_path.write_text(json.dumps(E.segmenter_ceiling(seg, *data["val"]), indent=1))
    res["segmenter_ceiling"] = json.loads(ceil_path.read_text())

    bank = CacheBank([a.cache], split="val", hflip=False, sources=["openeds"], labels_dir=a.masks)
    masks = E.aligned_masks(bank, a.masks)
    x, m, _ = E.eval_windows(bank, masks, a.n, a.t, seed=0)
    real = E.gray(x[:, -1])
    m_np = m
    res["reference"] = {}
    for name, rec in (("blur_sigma_1", E.gaussian_blur(real, 1.0)), ("blur_sigma_2", E.gaussian_blur(real, 2.0)),
                      ("mean_image_of_eval_frames", real.mean(0, keepdim=True).expand_as(real))):
        res["reference"][name] = dict(gt=E.gt_metrics(real, rec, m_np), seg=E.seg_metrics(E.segment(seg, rec), m_np))
    res["reference"]["real_through_segmenter"] = dict(seg=E.seg_metrics(E.segment(seg, real), m_np))

    recon, fns = {}, {}
    runs = [("wm", s) for s in a.wm] + [("ae", s) for s in a.ae]
    for kind, spec in runs:
        name, path = spec.split("=", 1)
        fn = fns[name] = E.load_recon_fn(kind, path, bank.size)
        with torch.no_grad():
            r = torch.cat([fn(x[i:i + 64]) for i in range(0, len(x), 64)])
        recon[name] = r
        g = E.gray(r)
        res[name] = dict(kind=kind, gt=E.gt_metrics(real, g, m_np), seg=E.seg_metrics(E.segment(seg, g), m_np))
        if kind == "wm" and a.dreams:
            fr = E.dream_frames(fn.model, n=a.dreams)
            real_stats = E.structure_stats(E.segment(seg, real[:a.dreams]))
            res.setdefault("dreams_real_reference", real_stats)
            res[name]["dreams"] = {str(t): E.structure_stats(E.segment(seg, E.gray(f))) for t, f in fr.items()}
            save_sheet([[E.gray(fr[t][i])[None] for t in fr] for i in range(min(12, a.dreams))], out / f"sheet_dreams_{name}.png")
        if kind == "wm" and a.response and fn.cfg.action_dim >= 8:
            import numpy as np
            xr, _, pk = E.eval_windows(bank, masks, a.response, 8, seed=7, need_pupil=True)
            lab = bank.labels_at(pk[:, None] - np.arange(7, -1, -1)[None, :])
            cr = E.command_response(fn.model, seg, xr, lab["cmd"], lab["has_cmd"])
            shots = cr.pop("_shots")
            res[name]["command_response"] = cr
            save_sheet([[xr[i, -1]] + [shots[k][i] for k in ("mx", "hold", "px", "my", "py", "none")] for i in range(min(10, len(xr)))],
                       out / f"sheet_response_{name}.png")
        print(f"[{name}] gt: pupil mse {res[name]['gt'].get('mse_pupil')} contrast ratio {res[name]['gt'].get('pupil_contrast_ratio_median')} "
              f"edge ratio {res[name]['gt'].get('pupil_edge_grad_ratio')} | seg: pupil IoU {res[name]['seg'].get('iou_pupil')}", flush=True)

    (out / "eye_metric.json").write_text(json.dumps(res, indent=1))
    # sheet: windows with a visible pupil, fixed seed
    xs, ms, _ = E.eval_windows(bank, masks, 8, a.t, seed=5, need_pupil=True)
    cols = ["real", "GT mask"] + [f"{n} recon | {n} seg" for n in recon]
    rows = []
    with torch.no_grad():
        recs = {name: fn(xs) for name, fn in fns.items()}
    for i in range(len(xs)):
        row = [xs[i, -1], E.colour_mask(ms[i].numpy())]
        for name in recon:
            g = E.gray(recs[name][i:i + 1])
            row += [recs[name][i], E.colour_mask(E.segment(seg, g)[0].numpy())]
        rows.append(row)
    save_sheet(rows, out / "sheet.png")
    (out / "sheet_columns.txt").write_text(" | ".join(cols) + "   (windows whose last frame has a visible ground-truth pupil)")
    print(f"written {out}", flush=True)
    return res


if __name__ == "__main__":
    main()
