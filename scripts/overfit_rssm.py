"""RSSM overfit test: does L1's ARCHITECTURE (encoder + GRU/RSSM + decoder, 4.14 M params) reproduce ONE real clip sharply when it may memorise it?
Follow-up to scripts/overfit_clarity.py (plain AE512 passes the gate on seen frames: the decoder / loss / resolution are NOT the limit). Here the same
clip, frames, eval frames and metrics go through the full RSSM path, so a gap to the AE is caused by something the RSSM adds:
  A_l1     L1's recipe as is: sampled z (std >= min_std .1), kl_weight 1, kl_balance .99, free nats 1  (posterior penalty = .01 x KL above 1 nat)
  B_nokl   same, kl_weight 0                                         -> isolates the KL term
  C_det    kl_weight 0 and z = posterior mean (no sampling)          -> isolates the sampling noise (an RSSM-shaped autoencoder)
  D_wide   L1's recipe with stoch_dim 512 (decoder sees [h(256) | z(512)])  -> isolates the 128-d latent bottleneck
  E_kl01 / F_kl001  L1's recipe with kl_weight .1 / .01                      -> how much KL the detail tolerates
  G_stopgrad  L1's recipe, prior head reads h.detach() (prior loss trains only the prior head, no gradient into the GRU state) -> separates the
              dynamics-side KL cost on h from the posterior-side cost. Round 2 prediction (written before the run): round 1 had A .614 vs no-KL .654;
              if the dynamics-side gradient into h is the cost, G ~ .65 (near B) and E/F in between; if G ~ .61 (near A) the cost is on the posterior side.
  H_det_t1 / I_det_wide / J_det_long  round 3, the gap left when KL and sampling are gone (C_det .668 vs plain AE .716): single-frame windows (no recurrence), z 512,
              6000 steps. Prediction (written before the run): H_t1 ~ C_det (the recurrence is not the cost, the 128-d z is), I_wide > C_det by >= .02, J_long > C_det
              by ~.01-.02 but still below the AE (.716). P(any of the three reaches .70) ~ 30%.
Training data = that clip's 96 px frames only, T=8 windows from random starts, zero actions, no augmentation (as the AE test: pure memorisation). Loss =
learning.world_model_loss exactly as pilot.train_loss uses it for L1 (pred_weight 0). lr 1e-3, 200 warm-up steps, cosine to 0.1x, grad clip 100.
Scoring = overfit_clarity.score on the eval frames: window = the 8 frames ending at the eval frame (clip start for the first), state at the last step,
decoded; 'sample' = the model's own sampled z (mean over --draws noise draws), 'mean' = the same history decoded with the last posterior mean.
Same GATE as the AE test (edge_keep .9-1.05 x ceiling, edge_fidelity >= .9 x ceiling, pupil edge >= .9 x ceiling).

PREDICTION written before any result (so it can be wrong): the posterior is almost unregularised in L1, so A ~ B; if sampling matters C is sharper than B;
if D passes while A fails the 128-d bottleneck is the limit; if all four fail, the recurrent path / step budget limits detail, not KL or sampling.
P(A passes) ~ 35%.
  python scripts/overfit_rssm.py train  --pack ../overfit_pack --out pilot_results/overfit_rssm --arms A_l1 --threads 3
  python scripts/overfit_rssm.py report --pack ../overfit_pack --out pilot_results/overfit_rssm
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
import clarity_ceiling as C                                          # noqa: E402
import overfit_clarity as O                                          # noqa: E402
from ivs.config import ModelConfig                                   # noqa: E402
from ivs.learning import world_model_loss                            # noqa: E402
from ivs.model import EyeWorldModel                                  # noqa: E402

ARMS = {                                                              # name -> overrides on L1's recipe
    "A_l1": dict(),
    "B_nokl": dict(kl_weight=0.0),
    "C_det": dict(kl_weight=0.0, det=True),
    "D_wide": dict(stoch_dim=512),
    "E_kl01": dict(kl_weight=0.1),
    "F_kl001": dict(kl_weight=0.01),
    "G_stopgrad": dict(stopgrad=True),
    # round 3 (gap to the plain AE with KL and sampling already removed; all = C_det unless stated): T=1 windows (batch 64 keeps 64 frames/step; h is constant, so the
    # decoder sees only z = a 128-d AE with the RSSM's encoder/decoder), z 512, and twice the steps (6000, cosine over the longer run)
    "H_det_t1": dict(kl_weight=0.0, det=True, t=1, batch=64),
    "I_det_wide": dict(kl_weight=0.0, det=True, stoch_dim=512),
    "J_det_long": dict(kl_weight=0.0, det=True, steps_x=2),
}
L1 = dict(feature_dim=512, deter_dim=256, stoch_dim=128, hidden_dim=256, kl_weight=1.0, free_nats=1.0, kl_balance=0.99, lr=1e-3, warmup=200, lr_floor=0.1, grad_clip=100.0)
CURVE_KEYS = ("edge_fidelity", "edge_keep", "psnr", "pupil_edge_grad_ratio")


def make_deterministic(model):
    """z = the posterior mean at every step (training and eval): the same network without sampling noise."""
    orig = model.rssm.observe_step

    def step(prev_h, prev_z, action, embed):
        h, _z, prior, post = orig(prev_h, prev_z, action, embed)
        return h, post[0], prior, post
    model.rssm.observe_step = step
    return model


def make_stopgrad_prior(model):
    """The prior head reads h.detach(): the prior loss (kl_dyn, weight kl_balance) still trains the prior head, but sends no gradient into the GRU state h
    (nor through h into earlier z / the encoder). The posterior-side KL (kl_rep) is untouched. Isolates what the dynamics-side KL costs the state."""
    model.rssm.prior_stopgrad = True                                 # the RSSM's own training switch (ivs.model.RSSM.prior_stopgrad)
    return model


def build_model(arm, size=96, channels=3, action_dim=5):
    p = dict(L1, **ARMS[arm])
    cfg = ModelConfig(image_size=size, channels=channels, feature_dim=p["feature_dim"], deter_dim=p["deter_dim"], stoch_dim=p["stoch_dim"],
                      hidden_dim=p["hidden_dim"], action_dim=action_dim)
    model = EyeWorldModel(cfg)
    if p.get("stopgrad"):
        model = make_stopgrad_prior(model)
    return make_deterministic(model) if p.get("det") else model, p


def eval_windows(n_frames, eval_idx, t):
    """(starts, pos): window k = frames[starts[k]:starts[k]+t] and the eval frame sits at position pos[k] inside it (the window ends at the eval
    frame; the first frames of the clip use the first window)."""
    starts = np.clip(np.asarray(eval_idx) - (t - 1), 0, n_frames - t)
    return starts, np.asarray(eval_idx) - starts


def batch_windows(x_all, starts, t):
    """x_all (N,C,H,W) -> (len(starts), t, C,H,W)."""
    return torch.stack([x_all[s:s + t] for s in starts])


def lr_at(p, step, total):
    warm = min(1.0, (step + 1) / max(1, p["warmup"]))
    return p["lr"] * warm * (p["lr_floor"] + (1.0 - p["lr_floor"]) * 0.5 * (1 + math.cos(math.pi * step / max(1, total))))


def train_step(model, p, x, actions):
    out = model.observe(x, actions)
    total, m = world_model_loss(x, out, p["kl_weight"], p["free_nats"], p["kl_balance"])
    std = torch.stack([q[1] for q in out["posterior"]]).mean()
    return total, dict(recon=float(m["reconstruction"]), kl=float(m["kl"]), post_std=float(std))


@torch.no_grad()
def reconstruct_eval(model, x_all, eval_idx, t, draws, seed):
    """Returns {'sample': [draws x (E,H,W,3) uint8], 'mean': same}: eval-frame reconstructions from windows ending at each eval frame."""
    starts, pos = eval_windows(len(x_all), eval_idx, t)
    x = batch_windows(x_all, starts, t)
    actions = torch.zeros(x.shape[0], t, model.cfg.action_dim)
    sel = torch.as_tensor(pos)
    ar = torch.arange(len(pos))
    res = {"sample": [], "mean": []}
    to_u8 = lambda y: (y.clamp(0, 1).permute(0, 2, 3, 1).numpy() * 255).round().astype(np.uint8)
    was_training = model.training
    model.eval()
    for d in range(draws):
        torch.manual_seed(seed + d)
        out = model.observe(x, actions)
        res["sample"].append(to_u8(out["recon"][ar, sel]))
        pm = torch.stack([q[0] for q in out["posterior"]], 1)
        res["mean"].append(to_u8(model.decode(out["h"][ar, sel], pm[ar, sel])))
    model.train(was_training)
    return res


def mean_scores(rec_list, native_g, s, masks):
    sc = [O.score(native_g, O.up_to_native(r, s), masks) for r in rec_list]
    keys = [k for k in sc[0] if isinstance(sc[0][k], (int, float))]
    out = {k: float(np.mean([d[k] for d in sc])) for k in keys}
    out["edge_fidelity_sd"] = float(np.std([d["edge_fidelity"] for d in sc]))
    return out


def gate_of(sc, ceil, masks_given):
    g = dict(edge_keep=bool(O.GATE * ceil["edge_keep"] <= sc["edge_keep"] <= O.KEEP_MAX), edge_fidelity=bool(sc["edge_fidelity"] >= O.GATE * ceil["edge_fidelity"]))
    if masks_given:
        g["pupil_edge"] = bool(sc["pupil_edge_grad_ratio"] >= O.GATE * ceil["pupil_edge_grad_ratio"])
    return g


def load_clip(pack, key="openeds", h=96):
    z = np.load(Path(pack) / f"{key}.npz")
    native = z["native_eval"]
    return dict(train=z[f"train_{h}"], eval_idx=z["eval_idx"], native=native, native_g=[O.gray_of(n) for n in native], s=native.shape[1],
                masks=z["masks_eval"] if "masks_eval" in z.files else None, meta=json.loads(str(z["meta"])), h=h)


def run_arm(arm, clip, out, steps, batch, t, seed, probe_every, draws, threads):
    torch.set_num_threads(threads)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    model, p = build_model(arm, clip["h"])
    t, batch, steps = int(p.get("t", t)), int(p.get("batch", batch)), steps * int(p.get("steps_x", 1))      # per-arm window length / batch / step multiple (gap arms)
    x_all = torch.from_numpy(clip["train"]).permute(0, 3, 1, 2).float() / 255.0
    n = len(x_all)
    opt = torch.optim.AdamW(model.parameters(), lr=p["lr"])
    gen = torch.Generator().manual_seed(seed)
    actions = torch.zeros(batch, t, model.cfg.action_dim)
    ceil = O.score(clip["native_g"], [O.gray_of(C.down_up(nn_, clip["h"])) for nn_ in clip["native"]], clip["masks"])
    print(f"[{arm}] {n} train frames, {sum(q.numel() for q in model.parameters()) / 1e6:.2f} M params, steps {steps}, batch {batch}x{t}, threads {threads}; "
          f"ceiling edge_fidelity {ceil['edge_fidelity']:.3f} edge_keep {ceil['edge_keep']:.3f}", flush=True)
    curve, t0 = [], time.time()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(p, step, steps)
        starts = torch.randint(0, n - t + 1, (batch,), generator=gen).tolist()
        loss, parts = train_step(model, p, batch_windows(x_all, starts, t), actions)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), p["grad_clip"])
        opt.step()
        if (step + 1) % probe_every == 0 or step + 1 == steps:
            rec = reconstruct_eval(model, x_all, clip["eval_idx"], t, draws, seed=10_000)
            sc = mean_scores(rec["sample"], clip["native_g"], clip["s"], clip["masks"])
            row = dict(step=step + 1, loss=float(loss), sec=round(time.time() - t0, 1), **parts, **{k: v for k, v in sc.items() if k in CURVE_KEYS})
            curve.append(row)
            print(f"  [{arm}] step {row['step']} recon {parts['recon']:.1f} kl {parts['kl']:.2f} post_std {parts['post_std']:.3f} edge_fidelity {sc['edge_fidelity']:.3f} "
                  f"edge_keep {sc['edge_keep']:.3f} psnr {sc['psnr']:.2f} ({row['sec']}s)", flush=True)
    rec = reconstruct_eval(model, x_all, clip["eval_idx"], t, draws, seed=10_000)
    res = dict(arm=arm, recipe={k: v for k, v in p.items()}, steps=steps, batch=batch, t=t, train_frames=int(n), params=int(sum(q.numel() for q in model.parameters())),
               ceiling=ceil, train_seconds=round(time.time() - t0, 1), curve=curve)
    for variant in ("sample", "mean"):
        sc = mean_scores(rec[variant], clip["native_g"], clip["s"], clip["masks"])
        res[variant] = dict(scores=sc, over_ceiling_keep=sc["edge_keep"] / ceil["edge_keep"], over_ceiling_fidelity=sc["edge_fidelity"] / ceil["edge_fidelity"],
                            gate=gate_of(sc, ceil, clip["masks"] is not None))
        res[variant]["passed"] = bool(all(res[variant]["gate"].values()))
        print(f"[{arm}] {variant}: fidelity {sc['edge_fidelity']:.3f} vs ceiling {ceil['edge_fidelity']:.3f} (x{res[variant]['over_ceiling_fidelity']:.2f}), keep {sc['edge_keep']:.3f} "
              f"vs {ceil['edge_keep']:.3f}, pupil_edge {sc.get('pupil_edge_grad_ratio', float('nan')):.3f} vs {ceil.get('pupil_edge_grad_ratio', float('nan')):.3f} -> "
              f"{'PASS' if res[variant]['passed'] else 'FAIL'}", flush=True)
    (out / f"arm_{arm}.json").write_text(json.dumps(res, indent=1))
    np.savez_compressed(out / f"rec_{arm}.npz", sample=rec["sample"][0], mean=rec["mean"][0])
    torch.save(dict(config=model.cfg.__dict__, arm=arm, state_dict=model.state_dict()), out / f"model_{arm}.pt")
    return res


def report(pack, out):
    """Collects arm_*.json: summary table + sheet_openeds_rssm.png (rows = two eval frames; tiles = native | ceiling | each arm's sampled reconstruction)."""
    out = Path(out)
    clip = load_clip(pack)
    names = [a for a in ARMS if (out / f"arm_{a}.json").exists()]
    results = {a: json.loads((out / f"arm_{a}.json").read_text()) for a in names}
    if not names:
        raise FileNotFoundError(f"no arm_*.json in {out}")
    s = clip["s"]
    ceil_u8 = [O.gray_of(C.down_up(n, clip["h"])) for n in clip["native"]]
    native_show = [n for n in clip["native"]]
    patches = [C.detail_patch(g) for g in clip["native_g"]]
    pick = [len(native_show) // 3, 2 * len(native_show) // 3]
    recs = {a: [cv2.resize(r, (s, s), interpolation=cv2.INTER_CUBIC) for r in np.load(out / f"rec_{a}.npz")["sample"]] for a in names}
    recs = {a: [O.gray_of(r) for r in v] for a, v in recs.items()}
    rows = []
    for i in pick:
        y, x, sz = patches[i]
        tiles = []
        for name, im in [("native", native_show[i]), ("ceiling", ceil_u8[i])] + [(a, recs[a][i]) for a in names]:
            t_ = cv2.cvtColor(np.ascontiguousarray(im[y:y + sz, x:x + sz]), cv2.COLOR_GRAY2BGR)
            t_ = cv2.resize(t_, (200, 200), interpolation=cv2.INTER_NEAREST)
            cv2.putText(t_, name, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            tiles.append(t_)
        rows.append(np.concatenate(tiles, 1))
    cv2.imwrite(str(out / "sheet_openeds_rssm.png"), np.concatenate(rows, 0))
    ceil = results[names[0]]["ceiling"]
    print(f"clip {clip['meta']['name']} @ {clip['h']} px; ceiling edge_fidelity {ceil['edge_fidelity']:.3f} edge_keep {ceil['edge_keep']:.3f} pupil_edge {ceil.get('pupil_edge_grad_ratio', float('nan')):.3f}")
    print("(plain AE512 seen-frame reference, kaggle_overfit1: fidelity x0.98, keep x0.97, PASS)")
    for a in names:
        for v in ("sample", "mean"):
            r = results[a][v]
            sc = r["scores"]
            print(f"{a:8s} {v:6s} fidelity {sc['edge_fidelity']:.3f} (x{r['over_ceiling_fidelity']:.2f}) keep {sc['edge_keep']:.3f} (x{r['over_ceiling_keep']:.2f}) "
                  f"psnr {sc['psnr']:.1f} pupil_edge {sc.get('pupil_edge_grad_ratio', float('nan')):.3f}  {'PASS' if r['passed'] else 'FAIL'}")
    (out / "results.json").write_text(json.dumps(results, indent=1))
    return results


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--pack", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    t.add_argument("--steps", type=int, default=3000)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--t", type=int, default=8)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--probe-every", type=int, default=500)
    t.add_argument("--draws", type=int, default=3)
    t.add_argument("--threads", type=int, default=3)
    r = sub.add_parser("report")
    r.add_argument("--pack", required=True)
    r.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "report":
        return report(a.pack, a.out)
    clip = load_clip(a.pack)
    return {arm: run_arm(arm, clip, a.out, a.steps, a.batch, a.t, a.seed, a.probe_every, a.draws, a.threads) for arm in a.arms}


if __name__ == "__main__":
    main()
