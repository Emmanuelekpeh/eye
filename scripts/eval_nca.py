"""Head-to-head EyeNCA vs the RSSM (and plain AEs) on REAL held-out S_ subjects. Gates G2 / G3 of backlog.md P0g are computed here.

  python scripts/eval_nca.py --nca N50=<model.pt> --nca N135=<model.pt> --wm L1=pilot_results/kaggle_long3_30k/pilot/L1_v6_long/model.pt \
      [--ae AE512=<AE512_model.pt>] --n 96 --dreams 64 --out pilot_results/nca_eval_x

REPAIR (G2). Held-out windows whose last frame has a visible ground-truth pupil. Damage = the RSSM's own scatter test (ivs.actions.scatter_batch,
3 discs, radius .15-.25, strength .7-1.0) applied to EVERY frame of the window, and a "pupil_erased" variant (last frame only: the pupil, dilated
2 px, is painted with the mean iris gray). Every method sees the SAME damaged last frame:
  damaged       the damaged frame as it is (what a model that ignores the damage would output)
  <wm> T8 / T1  RSSM observing the damaged 8-frame window (extra information: 8 independently damaged views) / the damaged last frame alone
  <ae>          plain autoencoder on the damaged last frame
  <nca> s=...   EyeNCA blind repair (SDEdit) at sigma_start s; the sigma marked `chosen` was picked on a SEPARATE selection set of windows
                (same val subjects, different frames), the others are shown for information
  <nca> masked  EyeNCA inpainting with the damaged pixels given as a mask (ORACLE information, not comparable to the blind rows)
Scores: ground-truth-mask metrics (ivs.eyemetric.gt_metrics) and the pupil segmenter (seg_metrics) on the repaired frame.
DREAMS (G3). N noise samples per model (EyeNCA: Heun from pure noise, EMA weights; RSSM: ivs.eyemetric.dream_frames at t=0,10,30,100,300). With no
ground truth, contrast / edge are defined by the SEGMENTER's own pupil / iris masks on the dream and measured the same way on real frames.
COST. Seconds per frame on this machine for a dream, a blind repair and one denoise call (= one live tick of the planned E1 loop).
Writes eval_nca.json, sheet_repair_<variant>.png (+ _columns.txt), sheet_dreams_<name>.png (8x8; the first 16 are rows 1-2). LOOK at the sheets.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ivs import eyemetric as E  # noqa: E402
from ivs import nca  # noqa: E402
from ivs.actions import scatter_batch  # noqa: E402
from ivs.cachedata import CacheBank  # noqa: E402
from ivs.pilot import nn_distance, save_sheet, sharpness, small_gray  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
DATA = HERE.parent / "data" / "cache"
SCATTER = dict(max_discs=3, radius=(0.15, 0.25), strength=(0.7, 1.0), p_disc=1.0)
SIGMAS = (0.3, 0.6, 1.0)
STEPS = 12
DREAM_TIMES = (0, 10, 30, 100, 300)
GATE_MARGIN = 0.05
FLAT_MAX = 1.5                       # noise guard: max sclera_grad_ratio for a repair to count (see score())
LIVE_SIGMAS = (0.02, 0.05, 0.1)      # living-mode noise levels (ivs.pnca.SIGMA_LIVE); the P1/P2 gates use LIVE_GATE_SIGMA, FIXED (not selected)
LIVE_GATE_SIGMA = 0.05
LIVE_REPAIR_TICKS = 30               # P2
LIVE_TIMES = (0, 10, 30, 100, 300)   # P1: dream, then living ticks


def load_nca(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = nca.net_from_config(ck["config"]); m.load_state_dict(ck["model"])
    m.persist = bool((ck.get("run") or {}).get("persist", False))     # PD-NCA checkpoints (ivs.pnca) carry the hidden state; E0 ones do not
    return m.to(device).eval()


def is_persistent(net):
    return bool(getattr(net, "persist", False))


def erase_pupil(x_last, mask):
    """x_last (n,3,H,W), mask (n,H,W) uint8: paint the (2 px dilated) pupil with the mean iris gray of that frame."""
    pupil = (torch.as_tensor(mask) == 3).float()[:, None]
    region = F.max_pool2d(pupil, 5, 1, 2) > 0
    g = E.gray(x_last)
    iris = torch.as_tensor(mask) == 2
    fill = E._masked_mean(g, iris)
    fill = torch.where(torch.isnan(fill), g.mean((1, 2)), fill).view(-1, 1, 1, 1).expand_as(x_last)
    return torch.where(region, fill, x_last)


def damage(kind, x, mask, seed):
    """x (n,t,3,H,W) real window. Returns (damaged window, damaged last frame, known-pixel mask (n,1,H,W) of the last frame)."""
    n, t, c, h, w = x.shape
    if kind == "scatter":
        gen = torch.Generator().manual_seed(seed)
        win = scatter_batch(x.reshape(-1, c, h, w), gen, **SCATTER).view(n, t, c, h, w)
    elif kind == "pupil_erased":
        win = x.clone()
        win[:, -1] = erase_pupil(x[:, -1], mask)
    else:
        raise ValueError(kind)
    last = win[:, -1]
    known = ((last - x[:, -1]).abs().amax(1, keepdim=True) <= 1e-6).float()
    return win, last, known


def batched(fn, *tensors, chunk=16):
    return torch.cat([fn(*[t[i:i + chunk] for t in tensors]) for i in range(0, len(tensors[0]), chunk)])


def score(seg, real_g, rec, m):
    """Ground-truth-mask metrics + segmenter metrics + a NOISE GUARD: sclera_grad_ratio = Sobel strength of the repaired frame / of the real
    frame over the (smooth) sclera pixels. The pupil-edge ratio alone rewards noise (an untrained denoiser's static scores 1.7-2.2 on it),
    so a result only counts if sclera_grad_ratio stays near 1 (<= FLAT_MAX)."""
    g = E.gray(rec)
    sclera = torch.as_tensor(m) == 1
    flat = float((E.grad_mag(g) * sclera).sum() / (E.grad_mag(real_g) * sclera).sum().clamp(min=1e-9))
    return dict(gt=E.gt_metrics(real_g, g, m), seg=E.seg_metrics(E.segment(seg, g), m), mse_full=float(((g - real_g) ** 2).mean()),
                sclera_grad_ratio=flat)


def nca_repair(net, last, sigma, seed, steps=STEPS):
    dev = next(net.parameters()).device

    def one(x):
        out = nca.repair_blind(net, x.to(dev), sigma, steps=steps, generator=torch.Generator().manual_seed(seed), persist=is_persistent(net))
        return (out[0] if is_persistent(net) else out).cpu()
    return batched(one, last)


def nca_live_repair(net, last, ticks, sigma, seed):
    """LIVING-MODE repair (P2 of backlog P0j): the damaged real frame is the cell grid's image and the network just runs on its own render
    (nca.live: re-noise to sigma, denoise, feed back) for `ticks` ticks; the hidden state starts at zero (a real frame has no dream history)."""
    dev = next(net.parameters()).device

    def one(x):
        return nca.live(net, x.to(dev), None, ticks, sigma=sigma, generator=torch.Generator().manual_seed(seed))[0].cpu()
    return batched(one, last)


def nca_masked(net, last, known, seed, steps=18):
    dev = next(net.parameters()).device

    def one(x, k):
        out = nca.repair_masked(net, x.to(dev), k.to(dev), steps=steps, generator=torch.Generator().manual_seed(seed), persist=is_persistent(net))
        return (out[0] if is_persistent(net) else out).cpu()
    return batched(one, last, known)


def pick_sigma(net, seg, bank, masks, t, n, seed):
    """sigma_start chosen on a SEPARATE selection set (scatter damage): the best pupil IoU + pupil-edge ratio (the two gate quantities)."""
    x, m, _ = E.eval_windows(bank, masks, n, t, seed=seed, need_pupil=True)
    _, last, _ = damage("scatter", x, m, seed + 1)
    real_g = E.gray(x[:, -1])
    table = {}
    for s in SIGMAS:
        r = score(seg, real_g, nca_repair(net, last, s, seed + 2), m)
        table[s] = dict(iou_pupil=r["seg"].get("iou_pupil"), edge=r["gt"].get("pupil_edge_grad_ratio"))
    best = max(SIGMAS, key=lambda s: (table[s]["iou_pupil"] or 0) + (table[s]["edge"] or 0))
    return best, {str(k): v for k, v in table.items()}


# ----------------------------------------------------------------------------------------------- dreams
def seg_contrast_edge(g, pred):
    """Segmenter-defined pupil/iris contrast and edge strength of frames WITHOUT ground truth. g (N,H,W) gray, pred (N,H,W) class ids."""
    pupil, iris = pred == 3, pred == 2
    has = (pupil.sum((1, 2)) >= E.MIN_PUPIL_PX) & (iris.sum((1, 2)) >= E.MIN_PUPIL_PX)
    contrast = E._masked_mean(g, iris) - E._masked_mean(g, pupil)
    band = E.edge_band(pupil) & has[:, None, None]
    gm = E.grad_mag(g)
    edge = (gm * band).sum((1, 2)) / band.sum((1, 2)).clamp(min=1)
    sclera = pred == 1
    flat = gm[sclera].mean() if sclera.any() else None             # noise guard: Sobel strength over the predicted (smooth) sclera
    return dict(n_with_pupil_and_iris=int(has.sum()), contrast_median=float(contrast[has].median()) if has.any() else None,
                edge_median=float(edge[has].median()) if has.any() else None, sclera_grad_mean=float(flat) if flat is not None else None)


def dream_stats(seg, frames, train_ref, val_ref):
    """frames (N,3,H,W) in [0,1]."""
    g = E.gray(frames)
    pred = E.segment(seg, g)
    sg = small_gray(frames)
    return dict(structure=E.structure_stats(pred), seg_contrast_edge=seg_contrast_edge(g, pred),
                diversity_nn=nn_distance(sg, sg, exclude_self=True), sharp=sharpness(frames), mean=float(frames.mean()),
                std=float(frames.flatten(1).std(1).mean()), nn_dist_to_train=nn_distance(sg, train_ref),
                nn_dist_to_val_real=nn_distance(sg, val_ref))


def grid(frames, path, cols=8):
    rows = [[frames[r * cols + c] for c in range(cols) if r * cols + c < len(frames)] for r in range(-(-len(frames) // cols))]
    save_sheet(rows, path)


# ----------------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--nca", nargs="*", action="extend", default=[], help="NAME=path/to/model.pt (EyeNCA, ivs.ncapilot)")
    ap.add_argument("--wm", nargs="*", action="extend", default=[], help="NAME=path/to/model.pt (RSSM world model)")
    ap.add_argument("--ae", nargs="*", action="extend", default=[], help="NAME=path/to/AE_model.pt (plain AE probe)")
    ap.add_argument("--gate-wm", default="L1", help="the RSSM the gates compare against (must be among --wm)")
    ap.add_argument("--cache-openeds", default=str(DATA / "openeds"))
    ap.add_argument("--cache-video", default=str(DATA / "videos"))
    ap.add_argument("--masks", default=str(DATA / "openeds_masks"))
    ap.add_argument("--segmenter", default=str(HERE / "pilot_results" / "eyemetric" / "segmenter.pt"))
    ap.add_argument("--n", type=int, default=96, help="held-out repair windows (test)")
    ap.add_argument("--select-n", type=int, default=32, help="windows of the separate sigma-selection set")
    ap.add_argument("--t", type=int, default=8)
    ap.add_argument("--dreams", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--skip-dreams", action="store_true")
    ap.add_argument("--out", default=str(HERE / "pilot_results" / "nca_eval_out"))
    a = ap.parse_args(argv)
    device = torch.device("cuda" if a.device == "auto" and torch.cuda.is_available() else ("cpu" if a.device == "auto" else a.device))
    torch.set_num_threads(max(1, min(6, torch.get_num_threads())))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    seg = E.load_segmenter(a.segmenter)
    res = dict(args=vars(a), repair={}, dreams={}, living={}, cost={}, gates={})

    nets = {s.split("=", 1)[0]: load_nca(s.split("=", 1)[1], device) for s in a.nca}
    wms = {s.split("=", 1)[0]: E.load_recon_fn("wm", s.split("=", 1)[1]) for s in a.wm}
    aes = {s.split("=", 1)[0]: E.load_recon_fn("ae", s.split("=", 1)[1], 96) for s in a.ae}
    res["params"] = {k: nca.count_params(v) for k, v in nets.items()}

    bank = CacheBank([a.cache_openeds], split="val", hflip=False, sources=["openeds"], labels_dir=a.masks)
    masks = E.aligned_masks(bank, a.masks)
    chosen, selection = {}, {}
    for name, net in nets.items():
        chosen[name], selection[name] = pick_sigma(net, seg, bank, masks, a.t, a.select_n, seed=100 + a.seed)
        print(f"[{name}] sigma_start chosen on the selection set: {chosen[name]} {selection[name]}", flush=True)
    res["sigma_selection"] = dict(chosen=chosen, table=selection)

    x, m, _ = E.eval_windows(bank, masks, a.n, a.t, seed=a.seed, need_pupil=True)
    real_g, m_np = E.gray(x[:, -1]), m                  # mask tensors (what ivs.eyemetric expects)
    res["repair"]["real_through_segmenter"] = dict(seg=E.seg_metrics(E.segment(seg, real_g), m_np))
    xs, ms, _ = E.eval_windows(bank, masks, 8, a.t, seed=5, need_pupil=True)             # sheet windows

    for kind in ("scatter", "pupil_erased"):
        win, last, known = damage(kind, x, m, 1000 + a.seed)
        rows = {"damaged": last}
        for name, fn in wms.items():
            rows[f"{name} T8"] = batched(lambda w: fn(w), win)
            rows[f"{name} T1"] = batched(lambda w: fn(w[:, -1:]), win)
        for name, fn in aes.items():
            rows[name] = batched(lambda w: fn(w), win)
        for name, net in nets.items():
            for s in SIGMAS:
                rows[f"{name} s={s}" + (" chosen" if s == chosen[name] else "")] = nca_repair(net, last, s, 2000 + a.seed)
            rows[f"{name} masked(oracle)"] = nca_masked(net, last, known, 2000 + a.seed)
            if is_persistent(net):
                for s in LIVE_SIGMAS:
                    rows[f"{name} live{LIVE_REPAIR_TICKS} s={s}"] = nca_live_repair(net, last, LIVE_REPAIR_TICKS, s, 2000 + a.seed)
        res["repair"][kind] = {k: score(seg, real_g, v, m_np) for k, v in rows.items()}
        for k, v in res["repair"][kind].items():
            print(f"[{kind}] {k:28s} pupil IoU {v['seg'].get('iou_pupil')}  edge {v['gt'].get('pupil_edge_grad_ratio')}  contrast {v['gt'].get('pupil_contrast_ratio_median')}  "
                  f"sclera_grad {v['sclera_grad_ratio']:.2f}  mse {v['mse_full']:.5f}", flush=True)
        # sheet: the 8 sheet windows with the same damage protocol
        win_s, last_s, known_s = damage(kind, xs, ms, 1000 + a.seed)
        cols, tiles = ["real", "damaged"], [[xs[i, -1], last_s[i]] for i in range(len(xs))]
        for name, fn in wms.items():
            for tag, inp in (("T8", win_s), ("T1", win_s[:, -1:])):
                cols.append(f"{name} {tag}")
                r = fn(inp)
                for i in range(len(xs)):
                    tiles[i].append(r[i])
        for name, fn in aes.items():
            cols.append(name)
            r = fn(win_s)
            for i in range(len(xs)):
                tiles[i].append(r[i])
        for name, net in nets.items():
            for s in SIGMAS:
                cols.append(f"{name} s={s}")
                r = nca_repair(net, last_s, s, 2000 + a.seed)
                for i in range(len(xs)):
                    tiles[i].append(r[i])
            cols.append(f"{name} masked(oracle)")
            r = nca_masked(net, last_s, known_s, 2000 + a.seed)
            for i in range(len(xs)):
                tiles[i].append(r[i])
            if is_persistent(net):
                for s in LIVE_SIGMAS:
                    cols.append(f"{name} live{LIVE_REPAIR_TICKS} s={s}")
                    r = nca_live_repair(net, last_s, LIVE_REPAIR_TICKS, s, 2000 + a.seed)
                    for i in range(len(xs)):
                        tiles[i].append(r[i])
        save_sheet(tiles, out / f"sheet_repair_{kind}.png")
        (out / f"sheet_repair_{kind}_columns.txt").write_text(" | ".join(cols))

    # ------------------------------------------------------------------ dreams
    if not a.skip_dreams and (nets or wms):
        mixed = CacheBank([a.cache_video, a.cache_openeds], split="val", hflip=False, source_weights=dict(video=0.5, openeds=0.5))
        train = CacheBank([a.cache_video, a.cache_openeds], split="train", hflip=False, source_weights=dict(video=0.5, openeds=0.5))
        g = torch.Generator().manual_seed(77)
        train_ref = small_gray(train.sample(2048, 1, generator=g, train=False)["images"][:, 0])
        real_mixed = mixed.sample(a.dreams, 1, generator=g, train=False)["images"][:, 0]
        real_sec = bank.sample(a.dreams, 1, generator=g, train=False)["images"][:, 0]
        val_ref = small_gray(mixed.sample(1024, 1, generator=g, train=False)["images"][:, 0])
        res["dreams"]["real_openeds_val"] = dream_stats(seg, real_sec, train_ref, val_ref)
        res["dreams"]["real_mixed_val"] = dream_stats(seg, real_mixed, train_ref, val_ref)
        grid(real_mixed, out / "sheet_real_mixed.png")
        for name, net in nets.items():
            t0 = time.time()
            gen13 = torch.Generator().manual_seed(13)
            if is_persistent(net):      # the dream keeps its hidden state; living ticks continue from it (P1 of backlog P0j)
                fr, hid = nca.sample(net, a.dreams, steps=18, generator=gen13, device=device, persist=True)
                fr = fr.cpu()
            else:
                fr = nca.sample(net, a.dreams, steps=18, generator=gen13, device=device).cpu()
            res["cost"][f"{name}_dream_sec_per_frame"] = round((time.time() - t0) / a.dreams, 3)
            if is_persistent(net):
                res["living"][name] = {}
                img, prev = fr.to(device), 0
                snap = {0: fr}
                for t in LIVE_TIMES[1:]:
                    img, hid = nca.live(net, img, hid, t - prev, sigma=LIVE_GATE_SIGMA, generator=gen13)
                    prev = t
                    snap[t] = img.cpu()
                for t, f in snap.items():
                    res["living"][name][str(t)] = dream_stats(seg, f, train_ref, val_ref)
                    res["living"][name][str(t)]["change_from_dream"] = float((f - fr).abs().mean())
                    if t in (100, LIVE_TIMES[-1]):
                        grid(f, out / f"sheet_living_{name}_t{t}.png")
                print(f"[living {name}] " + " | ".join(f"t{t}: nested {v['structure']['nested_eye']:.2f} sclera {v['seg_contrast_edge']['sclera_grad_mean']} "
                                                       f"contrast {v['seg_contrast_edge']['contrast_median']}" for t, v in res["living"][name].items()), flush=True)
            res["dreams"][name] = dream_stats(seg, fr, train_ref, val_ref)
            grid(fr, out / f"sheet_dreams_{name}.png")
            print(f"[dream {name}] {json.dumps(res['dreams'][name]['structure'])} {res['dreams'][name]['seg_contrast_edge']}", flush=True)
        for name, fn in wms.items():
            fr = E.dream_frames(fn.model, times=DREAM_TIMES, n=a.dreams)
            res["dreams"][name] = {str(t): dream_stats(seg, f.cpu(), train_ref, val_ref) for t, f in fr.items()}
            grid(fr[30].cpu(), out / f"sheet_dreams_{name}_t30.png")
            print(f"[dream {name}] t30 {res['dreams'][name]['30']['structure']} {res['dreams'][name]['30']['seg_contrast_edge']}", flush=True)

    # ------------------------------------------------------------------ cost
    one = x[:1, -1].to(device)
    for name, net in nets.items():
        with torch.no_grad():
            xm = nca.to_model(one)
            net.denoise(xm, 0.5)
            t0 = time.time()
            for _ in range(3):
                net.denoise(xm, 0.5)
            res["cost"][f"{name}_denoise_call_sec"] = round((time.time() - t0) / 3, 3)
            t0 = time.time()
            nca.repair_blind(net, one, chosen[name], steps=STEPS, generator=torch.Generator().manual_seed(0))
            res["cost"][f"{name}_repair_sec_per_frame"] = round(time.time() - t0, 3)
    for name, fn in wms.items():                       # the RSSM recon fns (ivs.eyemetric.load_recon_fn) always run on the CPU: keep the input there too
        w = x[:1]
        fn(w)
        t0 = time.time()
        for _ in range(3):
            fn(w)
        res["cost"][f"{name}_observe_T{a.t}_window_sec_cpu"] = round((time.time() - t0) / 3, 3)

    # ------------------------------------------------------------------ gates (numbers only; the visual part is a human judgement)
    gw = a.gate_wm
    if nets and gw in wms:
        sc = res["repair"]["scatter"]
        ref = {tag: (sc[f"{gw} {tag}"]["seg"].get("iou_pupil") or 0, sc[f"{gw} {tag}"]["gt"].get("pupil_edge_grad_ratio") or 0) for tag in ("T8", "T1")}
        iou_ref, edge_ref = max(r[0] for r in ref.values()), max(r[1] for r in ref.values())
        for name in nets:
            row = sc[f"{name} s={chosen[name]} chosen"]
            iou, edge = row["seg"].get("iou_pupil") or 0, row["gt"].get("pupil_edge_grad_ratio") or 0
            flat = row["sclera_grad_ratio"]
            res["gates"][name] = dict(
                G2_repair=dict(nca_iou=iou, nca_edge=edge, nca_sclera_grad_ratio=flat, ref_wm=gw, ref_iou_best_of_T8_T1=iou_ref,
                               ref_edge_best_of_T8_T1=edge_ref,
                               pass_numbers=bool(iou >= iou_ref + GATE_MARGIN and edge >= edge_ref + GATE_MARGIN and flat <= FLAT_MAX),
                               note="also needs the repair sheet to agree (look at it); sclera_grad_ratio is the noise guard"))
            if res["dreams"].get(gw):
                ce = {t: res["dreams"][gw][t]["seg_contrast_edge"] for t in res["dreams"][gw]}
                best_c = max((v["contrast_median"] or 0) for v in ce.values())
                best_e = max((v["edge_median"] or 0) for v in ce.values())
                d = res["dreams"][name]
                real_div = res["dreams"]["real_mixed_val"]["diversity_nn"]
                real_to_train = res["dreams"]["real_mixed_val"]["nn_dist_to_train"]
                real_flat = res["dreams"]["real_openeds_val"]["seg_contrast_edge"]["sclera_grad_mean"]
                flat_ok = d["seg_contrast_edge"]["sclera_grad_mean"] is not None and real_flat is not None \
                    and d["seg_contrast_edge"]["sclera_grad_mean"] <= FLAT_MAX * real_flat
                res["gates"][name]["G3_dream"] = dict(
                    nca_contrast=d["seg_contrast_edge"]["contrast_median"], nca_edge=d["seg_contrast_edge"]["edge_median"],
                    ref_wm_best_over_t_contrast=best_c, ref_wm_best_over_t_edge=best_e,
                    nca_sclera_grad=d["seg_contrast_edge"]["sclera_grad_mean"], real_sclera_grad=real_flat, noise_guard_ok=bool(flat_ok),
                    diversity=d["diversity_nn"], real_diversity=real_div, diversity_ok=bool(d["diversity_nn"] >= 0.5 * real_div),
                    nn_dist_to_train=d["nn_dist_to_train"], real_val_nn_dist_to_train=real_to_train,
                    pass_numbers=bool((d["seg_contrast_edge"]["contrast_median"] or 0) >= best_c and (d["seg_contrast_edge"]["edge_median"] or 0) >= best_e
                                      and d["diversity_nn"] >= 0.5 * real_div and flat_ok),
                    note="also needs >= half of 16 viewed dreams recognisable as eyes (count them in sheet_dreams_<name>.png rows 1-2)")
    # P1 persistence / P2 regeneration (backlog P0j) - numbers only, persistent models only. Interpretation fixed here, before any PD-NCA result:
    #   P1: at living tick 100, nested_eye >= 0.8 x its dream value, contrast_median >= 0.5 x its dream value, and the NOISE GUARD
    #       sclera_grad_mean(t100) / sclera_grad_mean(t0) <= FLAT_MAX (the Sobel strength over predicted sclera must not creep up).
    #   P2: scatter repair, sigma LIVE_GATE_SIGMA (fixed, not selected): live30 pupil IoU >= the damaged frame's pupil IoU + .05 and sclera_grad_ratio <= FLAT_MAX.
    for name, net in nets.items():
        if not is_persistent(net):
            continue
        g_ = res["gates"].setdefault(name, {})
        lv = res["living"].get(name)
        if lv:
            t0s, t100 = lv["0"], lv["100"]
            n0, n100 = t0s["structure"]["nested_eye"], t100["structure"]["nested_eye"]
            c0, c100 = t0s["seg_contrast_edge"]["contrast_median"], t100["seg_contrast_edge"]["contrast_median"]
            f0, f100 = t0s["seg_contrast_edge"]["sclera_grad_mean"], t100["seg_contrast_edge"]["sclera_grad_mean"]
            ratio = (f100 / f0) if (f0 and f100 is not None) else None
            g_["P1_persistence"] = dict(
                nested_eye_t0=n0, nested_eye_t100=n100, contrast_t0=c0, contrast_t100=c100, sclera_grad_t0=f0, sclera_grad_t100=f100, sclera_grad_ratio=ratio,
                nested_eye_t300=lv[str(LIVE_TIMES[-1])]["structure"]["nested_eye"],
                pass_numbers=bool(n0 > 0 and n100 >= 0.8 * n0 and c0 is not None and c100 is not None and c100 >= 0.5 * c0 and ratio is not None and ratio <= FLAT_MAX),
                note="needs the sheets to agree (sheet_dreams vs sheet_living_<name>_t100.png): same frames, still eyes, not noise, not flat grey")
        sc = res["repair"]["scatter"]
        row = sc[f"{name} live{LIVE_REPAIR_TICKS} s={LIVE_GATE_SIGMA}"]
        dmg = sc["damaged"]
        iou, iou0 = row["seg"].get("iou_pupil") or 0, dmg["seg"].get("iou_pupil") or 0
        g_["P2_regen"] = dict(live_iou=iou, damaged_iou=iou0, live_edge=row["gt"].get("pupil_edge_grad_ratio"), sclera_grad_ratio=row["sclera_grad_ratio"],
                              pass_numbers=bool(iou >= iou0 + GATE_MARGIN and row["sclera_grad_ratio"] <= FLAT_MAX),
                              note="needs repair sheet columns 'live30 s=0.05' to agree: damage healed, no noise, no flat paint")
    (out / "eval_nca.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res["gates"], indent=1, default=float), flush=True)
    print(f"written {out}", flush=True)
    return res


if __name__ == "__main__":
    main()
