"""Diagnostic: can the SAME conv encoder/decoder stacks reach the linear (PCA) baseline when trained as a plain
autoencoder on single frames (no RSSM, no KL, no time)? Separates "the world-model machinery costs us detail" from
"the conv architecture / optimisation cannot learn detail".

  python -m ivs.aeprobe --cache ../data/cache/videos --out ae_out --variants AE32 AE128 --steps 8000

Code = encoder output (feature_dim = K, LayerNorm+SiLU as in the world model) -> decoder. Loss = the world model's pixel-sum
smooth-L1. Same CacheBank split/val frames as the pilots (video source only unless --sources says otherwise).
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch

from .cachedata import CacheBank
from .config import ModelConfig
from .learning import _pixel_sum_loss
from .model import ConvEncoder, build_decoder, conv_channels
from .pilot import mse, save_sheet, sharpness

VARIANTS = {
    "AE32": dict(code=32),
    "AE128": dict(code=128),
    "AE512": dict(code=512),
    "AE128_wide": dict(code=128, conv_mult=2.0),
    # Optimisation / VAE-ness tests: the world model trains at batch 32 (clips), lr 3e-4, sampled z, KL weight 1.
    "AE32_b32": dict(code=32, batch=32, lr=3e-4),
    "AE128_b32": dict(code=128, batch=32, lr=3e-4),
    "VAE32": dict(code=32, vae=1.0),                                  # sampled 32-d z (std=softplus+0.1), KL to N(0,1), WM loss scale
    "VAE32_b32": dict(code=32, vae=1.0, batch=32, lr=3e-4),
    "VAE128": dict(code=128, vae=1.0),
    # KL-weight sweep at the same loss scale (VAE32/VAE128 at weight 1 both stall at ~24 nats, val .0074-.0076):
    "VAE32_kl01": dict(code=32, vae=0.1),
    "VAE32_kl001": dict(code=32, vae=0.01),
    "VAE128_kl01": dict(code=128, vae=0.1),
    "VAE128_kl001": dict(code=128, vae=0.01),
}
MIN_STD, FREE_NATS = 0.1, 1.0                                         # = ModelConfig.min_std and the world model's free nats


class PlainAE(torch.nn.Module):
    """vae=0: encoder(feature_dim=code) -> decoder (deterministic code). vae>0: encoder(128) -> (mean, std) of a sampled code,
    sampling in train mode only (eval decodes the mean), `last_kl` = KL(q || N(0,1)) in nats per image.
    spatial=k (vae=0 only, `code` unused): the latent is a (k, H/16, H/16) feature map instead of one vector: the encoder's conv stack, then 1x1 conv -> GroupNorm(1) -> SiLU;
    the decoder starts with a 1x1 conv back to the stack's width and reuses the world model's transposed convs. `latent_numbers` = what the bottleneck holds.
    fine=True (spatial only): a finer grid, (k, H/8, H/8): the encoder drops its last stride-2 stage and the decoder its first transposed conv (a shallower net, not a drop-in)."""
    def __init__(self, image_size, code, conv_mult=1.0, vae=0.0, spatial=0, fine=False):
        super().__init__()
        self.vae = vae
        self.image_size, self.code, self.spatial, self.fine = image_size, code, spatial, fine      # constructor arguments (ivs.latentdyn.save_ae / load_ae rebuild the net from them)
        cfg = ModelConfig(image_size=image_size, channels=3, feature_dim=128 if vae else code, conv_mult=conv_mult)
        self.last_kl = None
        if spatial:
            if vae:
                raise ValueError("spatial latent is deterministic (vae=0)")
            ch = conv_channels(cfg)
            cut, first, width, s = (8, 6, ch[2], image_size // 8) if fine else (11, 3, ch[3], image_size // 16)    # layer indices in ConvEncoder.net / build_decoder
            self.head = None
            self.encoder = torch.nn.Sequential(*list(ConvEncoder(cfg).net)[:cut], torch.nn.Conv2d(width, spatial, 1), torch.nn.GroupNorm(1, spatial), torch.nn.SiLU())
            self.decoder = torch.nn.Sequential(torch.nn.Conv2d(spatial, width, 1), torch.nn.SiLU(), *list(build_decoder(cfg, 1))[first:])
            self.latent_numbers = spatial * s * s
        elif fine:
            raise ValueError("fine needs a spatial latent")
        else:                                                         # creation order encoder -> head -> decoder is unchanged (same seeded init as before)
            self.encoder = ConvEncoder(cfg)
            self.head = torch.nn.Linear(128, 2 * code) if vae else None
            self.decoder = build_decoder(cfg, code)
            self.latent_numbers = code

    def forward(self, x):
        e = self.encoder(x)
        if not self.vae:
            return self.decoder(e)
        mean, raw = self.head(e).chunk(2, dim=-1)
        std = torch.nn.functional.softplus(raw) + MIN_STD
        self.last_kl = (-torch.log(std) + (std ** 2 + mean ** 2) / 2 - 0.5).sum(-1).mean()
        return self.decoder(mean + std * torch.randn_like(std) if self.training else mean)


def frames(bank, n, seed, train=False):
    g = torch.Generator().manual_seed(seed)
    return bank.sample(n, 1, generator=g, train=train)["images"][:, 0]


@torch.no_grad()
def evaluate(model, bank, n=2048, chunk=256, seed=5):
    model.eval()
    rec, sh_r, sh_x, done = 0.0, 0.0, 0.0, 0
    for i in range(n // chunk):
        x = frames(bank, chunk, seed + i)
        y = model(x.to(next(model.parameters()).device))
        x = x.to(y.device)
        rec += mse(y, x) * chunk; sh_r += sharpness(y) * chunk; sh_x += sharpness(x) * chunk; done += chunk
    model.train()
    return dict(mse=rec / done, sharp_ratio=sh_r / max(sh_x, 1e-12))


def pca_baseline(train, val, dims=(32, 128, 512), n_train=4000, n_val=1024):
    xt = frames(train, n_train, 1).flatten(1).cpu()
    xv = frames(val, n_val, 2).flatten(1).cpu()
    mu = xt.mean(0, keepdim=True)
    q = min(max(dims) + 8, n_train - 1)
    _, _, V = torch.pca_lowrank(xt - mu, q=q, center=False, niter=3)
    out = {"mean_image": float(((mu - xv) ** 2).mean())}
    for k in dims:
        P = V[:, :k]
        out[f"pca{k}"] = float(((((xv - mu) @ P) @ P.T + mu - xv) ** 2).mean())
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, nargs="+")
    ap.add_argument("--out", default="ae_out")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--sources", nargs="+", default=["video"])
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--image-size", type=int, default=96)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-videos", type=int, default=None, help="smoke tests only")
    ap.add_argument("--max-frames-per-video", type=int, default=None, help="smoke tests only")
    ap.add_argument("--eval-frames", type=int, default=2048)
    ap.add_argument("--no-pca", action="store_true")
    a = ap.parse_args(argv)
    device = torch.device("cuda" if a.device == "auto" and torch.cuda.is_available() else ("cpu" if a.device == "auto" else a.device))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    kw = dict(image_size=a.image_size, device=device, sources=a.sources, seed=a.seed, max_videos=a.max_videos,
              max_frames_per_video=a.max_frames_per_video)
    train = CacheBank(a.cache, split="train", **kw)
    val = CacheBank(a.cache, split="val", hflip=False, **kw)
    print(f"device={device} train={train.info()['frames']} frames/{train.n_groups} groups val={val.info()['frames']}/{val.n_groups}", flush=True)
    ref = {} if a.no_pca else pca_baseline(train, val)
    print("linear baselines (val MSE):", ref, flush=True)
    if ref:
        (out / "baselines.json").write_text(json.dumps(ref, indent=1))
    summary = {}
    for name in a.variants:
        spec = VARIANTS[name]
        torch.manual_seed(a.seed)
        vae_w = float(spec.get("vae", 0.0))
        batch, lr0 = spec.get("batch", a.batch), spec.get("lr", a.lr)
        model = PlainAE(a.image_size, spec["code"], spec.get("conv_mult", 1.0), vae=vae_w).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr0)
        gen = torch.Generator().manual_seed(a.seed)
        log = open(out / f"{name}_train_log.jsonl", "w")
        t0 = time.time()
        for step in range(a.steps):
            lr = lr0 * min(1.0, (step + 1) / 200) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / a.steps)))
            for g in opt.param_groups:
                g["lr"] = lr
            x = train.sample(batch, 1, generator=gen, train=True)["images"][:, 0]
            if vae_w:        # same scale as the world model's loss: pixel sum over C,H,W + free-nats KL
                loss = _pixel_sum_loss(model(x)[:, None], x[:, None]) + vae_w * model.last_kl.clamp(min=FREE_NATS)
            else:
                loss = _pixel_sum_loss(model(x), x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
            opt.step()
            if (step + 1) % 250 == 0 or step == 0:
                row = dict(step=step + 1, loss=float(loss), lr=lr, sec=round(time.time() - t0, 1))
                if vae_w:
                    row["kl"] = round(float(model.last_kl), 2)
                log.write(json.dumps(row) + "\n"); log.flush()
                print(f"[{name}] " + " ".join(f"{k}={v}" for k, v in row.items()), flush=True)
        log.close()
        res = dict(variant=name, spec=spec, steps=a.steps, train_seconds=round(time.time() - t0, 1),
                   params=sum(p.numel() for p in model.parameters()),
                   train_fit=evaluate(model, train, a.eval_frames), val=evaluate(model, val, a.eval_frames), baselines=ref)
        with torch.no_grad():
            model.eval()
            xv = frames(val, 8, 99).to(device)
            xt = frames(train, 8, 98).to(device)
            save_sheet([[xv[i], model(xv)[i]] for i in range(8)] + [[xt[i], model(xt)[i]] for i in range(8)], out / f"{name}_sheet.png")
        (out / f"{name}_metrics.json").write_text(json.dumps(res, indent=1))
        torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()}, "spec": spec}, out / f"{name}_model.pt")
        summary[name] = res
        print(f"[{name}] val mse {res['val']['mse']:.5f} (train {res['train_fit']['mse']:.5f}) sharp {res['val']['sharp_ratio']:.3f} "
              f"| {res['train_seconds']}s", flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    main()
