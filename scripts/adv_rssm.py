"""R5 stage 2 (plan.md "R5 STAGE 2"): adversarial fine-tuning of the L1 RSSM on its own UI dream loop, plus its control.
  ctl = continue L1's own objective (recon + balanced KL on real T=32 windows, interventions / scatter / desaturation exactly as L1_v6_long) for the same steps / lr;
  adv = the same + a conditional PatchGAN on the dream: fake = the UI loop (predict -> render -> perceive the render) for K ticks from a clean real context, real = the real
        continuation under the same interventions, cond = the last context frame; weight set by ivs.advdream.adaptive_weight.
Only rssm.gru, rssm.prior and the decoder learn (encoder + posterior frozen). Appearance images (mrl) are not in the kernel's cache, so the image loss L1 trained with is absent in
BOTH arms. Both arms use the same data seed, so their real batches are identical.
  python scripts/adv_rssm.py train --arm adv --ckpt <L1 model.pt> --cache <96 px CacheBank dir> --out <dir> [--steps 6000 --max-minutes 60]
Outputs: <out>/<arm>/{model.pt (final), model_s1.pt (1/3), model_s2.pt (2/3), train_log.jsonl, summary.json}. model.pt keeps the pilot format ({model, config, run, mean_image}) + `stage2`.
Evaluation is separate and local (scripts/s2_eval.py, CPU).
"""
import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from ivs import advdream as A                                          # noqa: E402
from ivs.actions import InterventionSampler                           # noqa: E402
from ivs.cachedata import CacheBank                                   # noqa: E402
from ivs.config import ModelConfig                                    # noqa: E402
from ivs.learning import world_model_loss                             # noqa: E402
from ivs.model import EyeWorldModel                                   # noqa: E402
from ivs.pilot import build_run, make_batch, mse, sharpness           # noqa: E402

ARMS = ("ctl", "adv")
RUN = "L1_v6_long"
CTX, K, T = 8, 24, 32                       # real context, dream ticks, window length (CTX + K = T = L1's longest training window)
TICKS = list(range(2, K, 3))                # the dream frames the discriminator sees (every 3rd tick: 8 per clip)
SOURCE_WEIGHTS = {"video": 0.5, "openeds": 0.5}
VAL_FRAC = 0.15


def make_banks(cache, device, size=96, splits=("train", "val")):
    kw = dict(image_size=size, val_frac=VAL_FRAC, seed=0, tint_mode="gray", device=device, sources=["video", "openeds"])
    out = {}
    if "train" in splits:
        out["train"] = CacheBank(cache, split="train", hflip=True, source_weights=SOURCE_WEIGHTS, **kw)
    if "val" in splits:
        out["val"] = CacheBank(cache, split="val", hflip=False, **kw)
    return out


def load_model(ckpt, device):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = EyeWorldModel(ModelConfig(**ck["config"]))
    model.load_state_dict(ck["model"])
    return model.to(device), ck


def real_loss(model, rc, inp, clean, actions):
    """L1's objective on a real window = ivs.pilot.train_loss with pred_weight 0 (no geometry / consistency / overshoot in L1_v6_long): recon + balanced KL. Skips the unused
    open-loop prediction rollout (its weight is 0, so it only costs time)."""
    out = model.observe(inp, actions)
    return world_model_loss(clean, out, rc.kl_weight, rc.free_nats, rc.kl_balance, recon_kind=rc.recon_kind, edge_weight=rc.edge_weight)


@torch.no_grad()
def heldout_probe(model, val, device, n=48, ctx=CTX, ticks=48, seed=5):
    """Cheap monitor: posterior recon mse on held-out windows (teacher-forced, T=16) and the contrast ratio (dream / real) of the UI loop at ticks 24 and 48 from a real context.
    Uses the model in place (no state change except the RNG, which is re-seeded)."""
    g = torch.Generator().manual_seed(seed)
    x = val.sample(n, ctx + ticks, generator=g, train=False)["images"].to(device) if val.max_window() >= ctx + ticks else val.sample(n, 16, generator=g, train=False)["images"].to(device)
    torch.manual_seed(seed)
    zero = torch.zeros(x.shape[0], x.shape[1], model.cfg.action_dim, device=device)
    out = model.observe(x[:, :16], zero[:, :16])
    row = dict(recon_mse=float(mse(out["recon"], x[:, :16])))
    if x.shape[1] >= ctx + ticks:
        h = model.observe(x[:, :ctx], zero[:, :ctx])
        fr, _ = A.ui_rollout(model, h["h"][:, -1], h["z"][:, -1], zero[:, ctx:ctx + ticks])
        contrast = lambda f: float(f.flatten(1).std(1).mean())
        for t in (24, 48):
            row[f"contrast_ratio_t{t}"] = contrast(fr[:, t - 1]) / contrast(x[:, ctx + t - 1])
            row[f"sharp_ratio_t{t}"] = sharpness(fr[:, t - 1]) / sharpness(x[:, ctx + t - 1])
            row[f"mse_to_real_t{t}"] = float(mse(fr[:, t - 1], x[:, ctx + t - 1]))
    return row


def lr_at(base, frac, warm=0.03, floor=0.1):
    cos = floor + (1.0 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, frac)))
    return base * min(1.0, frac / warm) * cos


def train_arm(arm, ckpt, cache, out, steps, max_minutes, batch, lr, d_lr, w_adv, device, seed, log_every=50, probe_every=500, banks=None):
    assert arm in ARMS
    out = Path(out) / arm
    out.mkdir(parents=True, exist_ok=True)
    model, ck = load_model(ckpt, device)
    params = A.freeze_for_stage2(model)
    rc = build_run(RUN, steps)
    import dataclasses
    rc = dataclasses.replace(rc, batch=batch)
    banks = banks or make_banks(cache, device, model.cfg.image_size)
    train, val = banks["train"], banks["val"]
    opt = torch.optim.AdamW(params, lr=lr)
    d = A.PatchDisc().to(device) if arm == "adv" else None
    od = torch.optim.Adam(d.parameters(), lr=d_lr, betas=(0.5, 0.999)) if d is not None else None
    gen = torch.Generator().manual_seed(seed)                                      # same seed in both arms: identical real batches
    aug_gen = torch.Generator(device=device).manual_seed(seed + 1)
    torch.manual_seed(seed)
    sampler = InterventionSampler(p=rc.p_intervene)
    last = model.decoder[-2].weight                                               # reference layer of the adaptive weight (ConvTranspose2d before the Sigmoid)
    log = open(out / "train_log.jsonl", "w")
    n_train = sum(p.numel() for p in params)
    print(f"[{arm}] device={device} trainable {n_train:,} of {sum(p.numel() for p in model.parameters()):,} params | batch {batch} T={T} (ctx {CTX} + dream {K}) | lr {lr} d_lr {d_lr} w_adv {w_adv} | steps {steps} / {max_minutes} min", flush=True)
    base_probe = heldout_probe(model, val, device)
    print(f"[{arm}] step 0 probe {json.dumps({k: round(v, 4) for k, v in base_probe.items()})}", flush=True)
    t0, acc, acc_n, probes, step, skipped = time.time(), {}, 0, [dict(step=0, **base_probe)], 0, 0
    marks = {int(round(steps / 3)): "model_s1.pt", int(round(2 * steps / 3)): "model_s2.pt"}
    while True:
        frac = max(step / steps, (time.time() - t0) / (max_minutes * 60.0))
        if frac >= 1.0:
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(lr, frac)
        if od is not None:
            for g in od.param_groups:
                g["lr"] = lr_at(d_lr, frac)
        inp, clean, actions = make_batch(train, rc, gen, aug_gen, sampler, T, device)
        l_real, m = real_loss(model, rc, inp, clean, actions)
        parts = dict(real=float(l_real.detach()), recon=float(m["reconstruction"]), kl=float(m["kl"]))
        total = l_real
        if d is not None:
            fake, real, cond = A.dream_from_clean_context(model, clean, actions, CTX, K)
            ld = A.d_hinge(A.disc_logits(d, cond, real, TICKS), A.disc_logits(d, cond, fake.detach(), TICKS))
            od.zero_grad(set_to_none=True)
            ld.backward()
            od.step()
            l_adv = A.g_hinge(A.disc_logits(d, cond, fake, TICKS))
            lam = A.adaptive_weight(l_real, l_adv, last, w_adv)
            total = l_real + lam * l_adv
            parts.update(d=float(ld.detach()), adv=float(l_adv.detach()), lam=float(lam), dream_mse=float(mse(fake, real)))
        opt.zero_grad(set_to_none=True)
        total.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(params, rc.grad_clip))
        if not math.isfinite(gn):
            skipped += 1
            opt.zero_grad(set_to_none=True)
            step += 1
            print(f"[{arm}] step {step}: non-finite gradient, step skipped ({skipped} so far)", flush=True)
            continue
        opt.step()
        step += 1
        parts["gn"] = gn
        for k, v in parts.items():
            acc[k] = acc.get(k, 0.0) + v
        acc_n += 1
        if step % log_every == 0:
            row = {k: round(v / acc_n, 4) for k, v in acc.items()}
            row.update(step=step, lr=round(lr_at(lr, frac), 7), min=round((time.time() - t0) / 60, 1))
            if probe_every and step % probe_every == 0:
                pr = heldout_probe(model, val, device)
                row["probe"] = {k: round(v, 4) for k, v in pr.items()}
                probes.append(dict(step=step, **pr))
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(f"[{arm}] " + " ".join(f"{k}={v}" for k, v in row.items()), flush=True)
            acc, acc_n = {}, 0
        if step in marks:
            save_model(out / marks[step], model, ck, arm, step, steps, lr, w_adv)
    save_model(out / "model.pt", model, ck, arm, step, steps, lr, w_adv)
    final_probe = heldout_probe(model, val, device)
    probes.append(dict(step=step, **final_probe))
    summ = dict(arm=arm, steps_done=step, steps_planned=steps, minutes=round((time.time() - t0) / 60, 1), skipped=skipped, batch=batch, lr=lr, d_lr=d_lr, w_adv=w_adv, ctx=CTX, k=K, probes=probes)
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(f"[{arm}] done {step} steps in {summ['minutes']} min | final probe {json.dumps({k: round(v, 4) for k, v in final_probe.items()})}", flush=True)
    return model, summ


def save_model(path, model, ck, arm, step, steps, lr, w_adv):
    """pilot-format checkpoint (loadable by ivs.ui / scripts) with the stage-2 provenance."""
    new = dict(ck)
    new["model"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    new["stage2"] = dict(arm=arm, step=step, steps_planned=steps, lr=lr, w_adv=w_adv, trainable=list(A.TRAINABLE), frozen=list(A.FROZEN), ctx=CTX, k=K)
    torch.save(new, path)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="stage", required=True)
    t = sub.add_parser("train")
    t.add_argument("--arm", choices=ARMS, required=True)
    t.add_argument("--ckpt", required=True)
    t.add_argument("--cache", required=True, nargs="+", help="one or more CacheBank dirs; only the video / openeds sources are used (the adversary's 'real' side is real recordings only)")
    t.add_argument("--out", required=True)
    t.add_argument("--steps", type=int, default=6000)
    t.add_argument("--max-minutes", type=float, default=60.0, help="wall-clock cap; the lr schedule follows whichever of steps / minutes is further along")
    t.add_argument("--batch", type=int, default=12)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--d-lr", type=float, default=2e-4)
    t.add_argument("--w-adv", type=float, default=0.3)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default="auto")
    t.add_argument("--probe-every", type=int, default=500)
    a = ap.parse_args(argv)
    device = torch.device("cuda" if a.device == "auto" and torch.cuda.is_available() else ("cpu" if a.device == "auto" else a.device))
    torch.set_num_threads(max(1, min(6, torch.get_num_threads())))
    return train_arm(a.arm, a.ckpt, a.cache, a.out, a.steps, a.max_minutes, a.batch, a.lr, a.d_lr, a.w_adv, device, a.seed, probe_every=a.probe_every)


if __name__ == "__main__":
    main()
