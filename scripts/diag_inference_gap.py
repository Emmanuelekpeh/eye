"""Where do bad dreams come from: the INFERENCE (posterior can't use the frame well) or the PRIOR/DYNAMICS (free-run drifts)?

Usage: python scripts/diag_inference_gap.py --wm L1=<model.pt> [--wm L2=<model.pt>] --out pilot_results/diag_inference_gap.json

On real held-out S_ frames (windows of 8 frames, zero action, noise seed fixed), per model:
  A  amortisation gap: eye-segmenter pupil IoU / MSE of the LAST frame's reconstruction when z is (i) sampled from the posterior
     (as in training), (ii) the posterior mean, (iii) refined by K Adam steps on z to minimise the recon MSE with h fixed. (iii) is an UPPER
     BOUND for any 'decode, compare with the observation, update the latent, repeat' loop (it uses unlimited, perfectly informed updates).
  B  prior vs posterior: the same readouts for z = prior mean at the last step (what h alone knew before seeing the frame).
  D  refinement inside the observation chain (model.observe_refined: the refined z feeds the next GRU step): recon quality of the last frame
     (mse, pupil IoU, contrast / edge ratio vs real) with and without, and the free-run after the refined context (does refining hurt dreams?).
  C  free-run from a REAL context: eye structure (ivs.eyemetric.structure_stats) of imagined frames k steps after the 8 context frames, zero action,
     next to the free-run from the NOISE start (eyemetric.dream_frames) and the real-frame reference.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ivs import eyemetric as E  # noqa: E402
from ivs.cachedata import CacheBank  # noqa: E402
from ivs.config import ModelConfig  # noqa: E402
from ivs.model import EyeWorldModel  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def pupil_iou(seg, rec, mask):
    pred = E.segment(seg, E.gray(rec)) == 3
    gt = mask == 3
    inter = (pred & gt).sum((1, 2)).float()
    union = (pred | gt).sum((1, 2)).float()
    return float((inter / union.clamp(min=1)).mean())


def recon_stats(seg, rec, real, mask):
    return dict(mse=float(((rec - real) ** 2).mean()), pupil_iou=pupil_iou(seg, rec, mask))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm", nargs="*", action="extend", default=[], help="NAME=path/to/model.pt")
    ap.add_argument("--cache", default=str(ROOT / "data" / "cache" / "openeds"))
    ap.add_argument("--masks", default=str(ROOT / "data" / "cache" / "openeds_masks"))
    ap.add_argument("--segmenter", default=str(Path(__file__).resolve().parents[1] / "pilot_results" / "eyemetric" / "segmenter.pt"))
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--refine-steps", type=int, nargs="*", default=[5, 20, 60])
    ap.add_argument("--horizon", type=int, default=64)
    ap.add_argument("--obs-refine", type=int, default=20, help="D: Adam steps per observed frame of refine_latent inside the observation chain (0 = skip)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    seg = E.load_segmenter(a.segmenter)
    val = CacheBank([a.cache], image_size=96, hflip=False, split="val", sources=["openeds"])
    masks = E.aligned_masks(val, a.masks)
    ctx = 8
    x, _m_end, pick = E.eval_windows(val, masks, a.n, ctx + a.horizon, seed=7)
    real_last = x[:, ctx - 1]
    m_ctx = torch.from_numpy(masks[pick - a.horizon])        # eval_windows returns the mask of the window's LAST frame; we need the context's
    out = {}
    for item in a.wm:
        name, path = item.split("=", 1)
        ck = torch.load(path, map_location="cpu")
        cfg = ModelConfig(**ck["config"])
        model = EyeWorldModel(cfg)
        model.load_state_dict(ck["model"])
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        xin = x[:, :ctx] if cfg.channels == 3 else x[:, :ctx, :1]
        act = torch.zeros(len(x), ctx, cfg.action_dim)
        three = lambda r: r if r.shape[1] == 3 else r.expand(-1, 3, -1, -1)  # noqa: E731
        res = {"model": path}
        with torch.no_grad():
            torch.manual_seed(0)
            o = model.observe(xin, act)
            h_last, z_last = o["h"][:, -1], o["z"][:, -1]
            post_mean = o["posterior"][-1][0]
            prior_mean = o["prior"][-1][0]
            res["A_posterior_sample"] = recon_stats(seg, three(model.decode(h_last, z_last)), real_last, m_ctx)
            res["A_posterior_mean"] = recon_stats(seg, three(model.decode(h_last, post_mean)), real_last, m_ctx)
            res["B_prior_mean"] = recon_stats(seg, three(model.decode(h_last, prior_mean)), real_last, m_ctx)
            res["B_zero_z"] = recon_stats(seg, three(model.decode(h_last, torch.zeros_like(z_last))), real_last, m_ctx)
        target = real_last if cfg.channels == 3 else real_last[:, :1]
        z = post_mean.clone().requires_grad_(True)
        opt = torch.optim.Adam([z], lr=0.05)
        done = 0
        for k in sorted(a.refine_steps):
            while done < k:
                opt.zero_grad()
                loss = ((model.decode(h_last, z) - target) ** 2).mean()
                loss.backward()
                opt.step()
                done += 1
            with torch.no_grad():
                rec = three(model.decode(h_last, z))
                res[f"A_refined_{k}"] = recon_stats(seg, rec, real_last, m_ctx)
                res[f"A_refined_{k}"]["z_moved_vs_prior_std"] = float((z - prior_mean).abs().mean() / o["prior"][-1][1].mean())
        # C: free-run from the real context vs from noise
        times = (1, 4, 16, 32, a.horizon)
        with torch.no_grad():
            hh, zz = h_last, z_last
            zero = torch.zeros(len(x), cfg.action_dim)
            torch.manual_seed(1)
            real_ctx_free, noise_free = {}, {}
            for t in range(1, a.horizon + 1):
                hh, zz, _p = model.rssm.imagine_step(hh, zz, zero)
                if t in times:
                    fr = three(model.decode(hh, zz))
                    real_ctx_free[t] = E.structure_stats(E.segment(seg, E.gray(fr)))
                    real_ctx_free[t]["mse_to_last_context_frame"] = float(((fr - real_last) ** 2).mean())
            dreams = E.dream_frames(model, times=(0,) + times, n=a.n)
            for t, fr in dreams.items():
                noise_free[t] = E.structure_stats(E.segment(seg, E.gray(fr)))
        if a.obs_refine > 0:                                    # D: refinement INSIDE the observation chain (refined z feeds the next GRU step)
            torch.manual_seed(0)
            rf = model.observe_refined(xin, act, steps=a.obs_refine)
            with torch.no_grad():
                real_g = E.gray(real_last)
                for tag, rec in (("D_unrefined", three(model.decode(h_last, z_last))), ("D_refined", three(model.decode(rf["h"][:, -1], rf["z"][:, -1])))):
                    gm = E.gt_metrics(real_g, E.gray(rec), m_ctx)
                    res[tag] = dict(recon_stats(seg, rec, real_last, m_ctx), contrast_ratio=gm.get("pupil_contrast_ratio_median"),
                                    edge_ratio=gm.get("pupil_edge_grad_ratio"), washed_out=gm.get("frac_pupil_washed_out"))
                hh, zz = rf["h"][:, -1], rf["z"][:, -1]
                zero = torch.zeros(len(x), cfg.action_dim)
                torch.manual_seed(1)
                after = {}
                for t in range(1, a.horizon + 1):
                    hh, zz, _p = model.rssm.imagine_step(hh, zz, zero)
                    if t in times:
                        st = E.structure_stats(E.segment(seg, E.gray(three(model.decode(hh, zz)))))
                        after[str(t)] = {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in st.items()}
                res["D_free_run_after_refined_context"] = after
        res["C_free_run_from_real_context"] = {str(k): {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in real_ctx_free.items()}
        res["C_free_run_from_noise"] = {str(k): {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in noise_free.items()}
        out[name] = res
        print(f"[{name}] A/B recon (mse, pupil IoU):")
        for k, v in res.items():
            if k[0] in "AB" and isinstance(v, dict):
                print(f"   {k:22s} mse {v['mse']:.5f}  pupil_iou {v['pupil_iou']:.3f}" + (f"  |z-prior|/std {v['z_moved_vs_prior_std']:.2f}" if "z_moved_vs_prior_std" in v else ""))
        if "D_refined" in res:
            for k in ("D_unrefined", "D_refined"):
                v = res[k]; print(f"   {k:22s} mse {v['mse']:.5f}  pupil_iou {v['pupil_iou']:.3f}  contrast_ratio {v['contrast_ratio']}  edge_ratio {v['edge_ratio']}  washed_out {v['washed_out']}")
            print("   free-run nested_eye after the REFINED context: " + " ".join(f"t{t}={s['nested_eye']:.2f}" for t, s in res["D_free_run_after_refined_context"].items()))
        print(f"[{name}] C nested_eye / pupil_present by steps after a REAL context | from NOISE:")
        for t in times:
            r_, n_ = res["C_free_run_from_real_context"][str(t)], res["C_free_run_from_noise"][str(t)]
            print(f"   t={t:3d}  real-ctx nested {r_['nested_eye']:.2f} pupil {r_['pupil_present']:.2f} mse_to_ctx {r_['mse_to_last_context_frame']:.4f} | noise nested {n_['nested_eye']:.2f} pupil {n_['pupil_present']:.2f}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
