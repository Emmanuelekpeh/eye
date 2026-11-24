import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .config import ModelConfig, TrainConfig
from .data import EyeSequenceDataset
from .model import EyeWorldModel
from .learning import world_model_loss, PlasticityController
from .actions import InterventionSampler


def compute_loss(model, images, actions, a):
    """Observe the first half as context, then predict the second half open-loop."""
    outputs = model.observe(images, actions)
    ctx = images.shape[1] // 2
    
    # Check if this batch is a static image (flagged by action[0,0] == -999)
    # If it is, we only care about reconstruction loss, not prediction or KL
    is_static = (actions[0, 0, 0] == -999.0).item()
    
    if is_static:
        # Only compute reconstruction loss to train the autoencoder
        recon_loss = ((outputs["recon"] - images) ** 2).mean()
        # Zero out the other losses
        kl_loss = torch.tensor(0.0, device=images.device)
        pred_loss = torch.tensor(0.0, device=images.device)
        total_loss = recon_loss
        
        metrics = {
            "error": float(recon_loss),
            "reconstruction": float(recon_loss),
            "kl": 0.0,
            "prediction": 0.0,
            "total": float(total_loss)
        }
        return total_loss, metrics
        
    pred = model.rollout(outputs["h"][:, ctx - 1], outputs["z"][:, ctx - 1], actions[:, ctx:])
    return world_model_loss(images, outputs, a.kl_weight, a.free_nats, a.kl_balance,
                            pred_frames=pred, pred_targets=images[:, ctx:], pred_weight=a.pred_weight)


@torch.no_grad()
def evaluate(model, loader, device, mean_img, sampler=None, seed=123):
    """Per-pixel MSE for reconstruction and open-loop prediction, with trivial baselines."""
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    keys = ["recon", "pred", "mean_baseline_recon", "mean_baseline_pred", "copy_last_baseline_pred"]
    sums, n = dict.fromkeys(keys, 0.0), 0
    mean_img = mean_img.to(device)[None, None]
    for images, actions in loader:
        images, actions = images.to(device), actions.to(device)
        if sampler is not None:
            images, actions = sampler(images, generator=gen)
        b, ctx = images.shape[0], images.shape[1] // 2
        out = model.observe(images, actions)
        pred = model.rollout(out["h"][:, ctx - 1], out["z"][:, ctx - 1], actions[:, ctx:])
        target = images[:, ctx:]
        vals = {
            "recon": ((out["recon"] - images) ** 2).mean(),
            "pred": ((pred - target) ** 2).mean(),
            "mean_baseline_recon": ((mean_img - images) ** 2).mean(),
            "mean_baseline_pred": ((mean_img - target) ** 2).mean(),
            "copy_last_baseline_pred": ((images[:, ctx - 1:ctx] - target) ** 2).mean(),
        }
        for k in keys:
            sums[k] += float(vals[k]) * b
        n += b
    model.train()
    return {k: v / max(1, n) for k, v in sums.items()}


def verdict(m):
    return {"recon_beats_mean_image": m["recon"] < m["mean_baseline_recon"],
            "pred_beats_copy_last": m["pred"] < m["copy_last_baseline_pred"]}


def main():
    d = TrainConfig()
    p = argparse.ArgumentParser(description="Train the IVS eye-world model (CPU-first prototype)")
    p.add_argument("--data", default=d.data)
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--sequence-length", type=int, default=d.sequence_length)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--device", default=d.device)
    p.add_argument("--checkpoint-dir", default=d.checkpoint_dir)
    p.add_argument("--lr", type=float, default=d.learning_rate)
    p.add_argument("--kl-weight", type=float, default=d.kl_weight)
    p.add_argument("--free-nats", type=float, default=d.free_nats)
    p.add_argument("--kl-balance", type=float, default=d.kl_balance)
    p.add_argument("--pred-weight", type=float, default=d.pred_weight)
    p.add_argument("--grad-clip", type=float, default=d.grad_clip)
    p.add_argument("--intervention-prob", type=float, default=d.intervention_prob)
    p.add_argument("--val-frac", type=float, default=d.val_frac)
    p.add_argument("--probe-every", type=int, default=d.probe_every)
    p.add_argument("--include-stills", action="store_true",
                   help="also load loose root images as repeated single-frame clips (not recommended)")
    p.add_argument("--seed", type=int, default=d.seed)
    a = p.parse_args()
    if a.sequence_length < 4:
        raise SystemExit("--sequence-length must be >= 4 (context half + prediction half)")

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    device = torch.device(a.device if a.device == "cpu" or torch.cuda.is_available() else "cpu")
    cfg = ModelConfig(image_size=a.image_size)

    ds_args = dict(sequence_length=a.sequence_length, image_size=a.image_size,
                   val_frac=a.val_frac, seed=a.seed, include_stills=a.include_stills)
    train_ds = EyeSequenceDataset(a.data, split="train", **ds_args)
    try:
        val_ds = EyeSequenceDataset(a.data, split="val", **ds_args)
    except RuntimeError:
        val_ds = None
        print("WARNING: fewer than 2 subject groups -> no validation split. "
              "Metrics below are TRAIN metrics and say nothing about generalisation.")
    train_loader = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=a.batch_size, shuffle=False) if val_ds else None
    mean_img = train_ds.mean_image()

    model = EyeWorldModel(cfg).to(device)
    groups = {"encoder": model.encoder, "rssm": model.rssm, "decoder": model.decoder}
    optimizer = torch.optim.AdamW([{"params": list(m.parameters()), "name": n} for n, m in groups.items()],
                                  lr=a.lr)
    plasticity = PlasticityController(base_lr=a.lr)
    sampler = InterventionSampler(p=a.intervention_prob)
    eval_sampler = InterventionSampler(p=1.0)
    out = Path(a.checkpoint_dir); out.mkdir(parents=True, exist_ok=True)

    print(f"train windows={len(train_ds)} groups={len(set(train_ds.keys))} | "
          f"val windows={len(val_ds) if val_ds else 0} | device={device} | "
          f"parameters={sum(p.numel() for p in model.parameters()):,}")

    probe, probe_ema, probe_best, retention_ok, history, step_total = None, None, None, True, [], 0
    for epoch in range(1, a.epochs + 1):
        model.train()
        running = {"total": 0.0, "reconstruction": 0.0, "kl": 0.0, "prediction": 0.0}
        for step, (images, actions) in enumerate(train_loader, 1):
            images, actions = images.to(device), actions.to(device)
            if probe is None:
                probe = (images.clone(), actions.clone())     # fixed early batch = retention probe
            images, actions = sampler(images)
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = compute_loss(model, images, actions, a)
            loss.backward()
            
            # If it's a static image, we DO NOT want to update the RSSM (temporal model)
            # We only want to update the encoder and decoder.
            is_static = (actions[0, 0, 0] == -999.0).item()
            if is_static:
                # Zero out gradients for the RSSM so it doesn't learn from static frames
                for p in model.rssm.parameters():
                    if p.grad is not None:
                        p.grad.zero_()
                        
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
            lrs = plasticity.update(float(metrics["error"]), retention_ok)
            for group in optimizer.param_groups:
                group["lr"] = lrs[group["name"]]
            optimizer.step()
            step_total += 1
            for key in running:
                running[key] += float(metrics[key])
            if step_total % a.probe_every == 0:               # measure retention, don't assume it
                with torch.no_grad():
                    e = float(((model.observe(*probe)["recon"] - probe[0]) ** 2).mean())
                probe_ema = e if probe_ema is None else 0.5 * probe_ema + 0.5 * e
                probe_best = probe_ema if probe_best is None else min(probe_best, probe_ema)
                retention_ok = probe_ema <= 1.25 * probe_best
            if step % 20 == 0:
                print(f"epoch={epoch} step={step} loss={float(loss):.2f} recon={float(metrics['reconstruction']):.2f} "
                      f"kl={float(metrics['kl']):.2f} pred={float(metrics['prediction']):.2f} "
                      f"scales={ {k: round(v, 2) for k, v in plasticity.scales.items()} } retention_ok={retention_ok}")
        count = max(1, len(train_loader))
        row = {k: v / count for k, v in running.items()}
        row.update({"epoch": epoch, "plasticity_scales": dict(plasticity.scales),
                    "probe_error": probe_ema, "retention_ok": retention_ok})
        eval_loader = val_loader or train_loader
        tag = "val" if val_loader else "train(no-val!)"
        row[f"{tag}_passive"] = evaluate(model, eval_loader, device, mean_img)
        row[f"{tag}_intervened"] = evaluate(model, eval_loader, device, mean_img, eval_sampler)
        row["verdict_passive"] = verdict(row[f"{tag}_passive"])
        history.append(row)
        torch.save({"model": model.state_dict(), "config": asdict(cfg), "epoch": epoch, "history": history,
                    "plasticity": plasticity.state_dict(), "mean_image": mean_img}, out / "latest.pt")
        (out / "history.json").write_text(json.dumps(history, indent=2, default=float))
        mp = row[f"{tag}_passive"]
        print(f"epoch={epoch} [{tag}] recon_mse={mp['recon']:.4f} (mean-img baseline {mp['mean_baseline_recon']:.4f}) "
              f"pred_mse={mp['pred']:.4f} (copy-last {mp['copy_last_baseline_pred']:.4f}) verdict={row['verdict_passive']}")


if __name__ == "__main__":
    main()
