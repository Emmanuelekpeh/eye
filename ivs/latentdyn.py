"""R4: latent-grid dynamics on a FROZEN sharp autoencoder (ivs.aeprobe.PlainAE, spatial latent, R3c S32_ssim).

The RSSM (ivs.model) keeps a compact state and decodes it to a soft image. R3b/R3c showed the sharpness is bounded by latent SIZE, so the
sharp AE has a (32, H/16, H/16) grid latent (2048 numbers at 128 px). This module predicts that grid over time:

    e_t = Enc(x_t)                       (frozen)         x_hat_t = Dec(z_t)  (frozen)
    h_t, (mu_t, sigma_t) = ConvGRU(h_{t-1}, [z_in_{t-1}, A(a_t)])      z_t ~ N(z_in_{t-1} + delta, sigma)

z_in is the real latent while the model is still reading the context, afterwards its own previous sample (optionally re-projected through
Enc(Dec(.)), the analogue of the UI re-perceiving its own render). `a_t` is the 5-d touch action of ivs.actions (state change per step).
Everything is plain torch and runs on CPU; the Kaggle script only changes the sizes.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from .aeprobe import PlainAE
from .learning import ssim_loss

MIN_STD = 0.02


class ConvGRUCell(nn.Module):
    """GRU whose gates are 3x3 convolutions over the latent grid (state h: B,hid,g,g); GroupNorm(1) on gates/candidate keeps long rollouts stable."""

    def __init__(self, in_ch, hid, k=3):
        super().__init__()
        p = k // 2
        self.hid = hid
        self.gates = nn.Conv2d(in_ch + hid, 2 * hid, k, padding=p)
        self.cand = nn.Conv2d(in_ch + hid, hid, k, padding=p)
        self.norm_g, self.norm_c = nn.GroupNorm(1, 2 * hid), nn.GroupNorm(1, hid)

    def forward(self, x, h):
        r, u = torch.sigmoid(self.norm_g(self.gates(torch.cat([x, h], 1)))).chunk(2, 1)
        c = torch.tanh(self.norm_c(self.cand(torch.cat([x, r * h], 1))))
        return (1 - u) * h + u * c


class LatentDynamics(nn.Module):
    """Residual Gaussian next-latent model over a (z_ch, g, g) grid. The head starts as 'copy the last latent with sigma ~0.1'."""

    def __init__(self, z_ch=32, grid=8, action_dim=5, hid=96, act_ch=16, layers=2, anchor=False):
        super().__init__()
        self.z_ch, self.grid, self.action_dim, self.hid, self.layers, self.anchor = z_ch, grid, action_dim, hid, layers, anchor
        self.act = nn.Sequential(nn.Linear(action_dim, act_ch), nn.SiLU())
        self.cells = nn.ModuleList([ConvGRUCell(z_ch + act_ch + (z_ch if anchor else 0) if i == 0 else hid, hid) for i in range(layers)])
        self.head = nn.Sequential(nn.Conv2d(hid, hid, 3, padding=1), nn.SiLU(), nn.Conv2d(hid, 2 * z_ch, 1))
        last = self.head[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        with torch.no_grad():
            last.bias[z_ch:] = math.log(math.expm1(0.1))                       # softplus^-1(0.1)

    def config(self):
        return dict(z_ch=self.z_ch, grid=self.grid, action_dim=self.action_dim, hid=self.hid, act_ch=self.act[0].out_features, layers=self.layers, anchor=self.anchor)

    def initial(self, b, device="cpu"):
        return [torch.zeros(b, self.hid, self.grid, self.grid, device=device) for _ in range(self.layers)]

    def step(self, h, z_in, a, anchor=None):
        """h: list of (B,hid,g,g); z_in (B,z_ch,g,g); a (B,action_dim); anchor (B,z_ch,g,g) = latent of the last REAL context frame (only used when
        self.anchor; defaults to z_in) -> new h, mean, std (B,z_ch,g,g)."""
        am = self.act(a)[:, :, None, None].expand(-1, -1, self.grid, self.grid)
        x, new = torch.cat([z_in, am] + ([z_in if anchor is None else anchor] if self.anchor else []), 1), []
        for cell, hi in zip(self.cells, h):
            x = cell(x, hi)
            new.append(x)
        mean_off, raw = self.head(x).chunk(2, 1)
        return new, z_in + mean_off, F.softplus(raw) + MIN_STD


def nll(mean, std, target):
    """Gaussian negative log-likelihood per element (constants dropped)."""
    return (0.5 * ((target - mean) / std) ** 2 + torch.log(std)).mean()


class FrozenAE(nn.Module):
    """Encoder / decoder halves of a trained PlainAE with gradients off. `project(z)` = Enc(Dec(z))."""

    def __init__(self, ae):
        super().__init__()
        self.ae = ae.eval()
        for p in self.ae.parameters():
            p.requires_grad_(False)
        self.image_size, self.latent_numbers = ae.image_size, ae.latent_numbers

    def train(self, mode=True):                                                    # always eval (GroupNorm / no dropout, but be explicit)
        return super().train(False)

    def encode(self, x):
        """x (...,3,H,W) -> (...,z_ch,g,g)."""
        lead = x.shape[:-3]
        z = self.ae.encoder(x.reshape(-1, *x.shape[-3:]))
        return z.reshape(*lead, *z.shape[1:])

    def decode(self, z):
        lead = z.shape[:-3]
        x = self.ae.decoder(z.reshape(-1, *z.shape[-3:]))
        return x.reshape(*lead, *x.shape[1:])

    def project(self, z):
        return self.encode(self.decode(z).clamp(0, 1))


def save_ae(path, ae, meta=None):
    torch.save(dict(state=ae.state_dict(), image_size=ae.image_size, code=ae.code, spatial=ae.spatial, fine=ae.fine, meta=meta or {}), path)


def load_ae(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    ae = PlainAE(ck["image_size"], ck["code"], spatial=ck["spatial"], fine=ck["fine"])
    ae.load_state_dict(ck["state"])
    return FrozenAE(ae)


def save_dyn(path, dyn, meta=None):
    torch.save(dict(state=dyn.state_dict(), config=dyn.config(), meta=meta or {}), path)


def load_dyn(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    dyn = LatentDynamics(**ck["config"])
    dyn.load_state_dict(ck["state"])
    return dyn.eval(), ck["meta"]


def unroll(dyn, e, actions, ctx, free, ae=None, reproject=False, temperature=1.0, generator=None, detach_every=0):
    """Teacher-forced for the first `ctx` predicted frames, then free-running for `free` frames.
    e: (B,T,z_ch,g,g) real latents with T >= ctx+free; actions (B,T,A) (actions[:, t] moves frame t-1 -> t).
    Frame t (1..ctx+free-1... see below) is predicted from input latent e[t-1] while t <= ctx, afterwards from the model's own previous
    sample (re-projected through Enc(Dec(.)) when `reproject`). So the first free frame is still conditioned on the last REAL context frame.
    anchor (dyn.anchor): the real latent of frame min(t, ctx)-1, i.e. the input itself while reading the context and the last real context frame afterwards.
    detach_every=k: truncated BPTT, the recurrent state and the fed-back sample are detached every k free steps (long horizons without a 100-step gradient chain).
    Returns means, stds, samples: (B, ctx+free-1, z_ch, g, g) for frames 1..ctx+free-1; frame t's entry is index t-1."""
    b = e.shape[0]
    h = dyn.initial(b, e.device)
    means, stds, samples = [], [], []
    prev = None
    for t in range(1, ctx + free):
        z_in = e[:, t - 1] if t <= ctx else prev
        if detach_every and t > ctx and (t - ctx) % detach_every == 0:
            h, z_in = [x.detach() for x in h], z_in.detach()
        h, mu, sd = dyn.step(h, z_in, actions[:, t], anchor=e[:, min(t, ctx) - 1])
        eps = torch.randn(mu.shape, generator=generator, device=mu.device) if temperature > 0 else torch.zeros_like(mu)
        s = mu + temperature * sd * eps
        means.append(mu), stds.append(sd), samples.append(s)
        if reproject and ae is not None:                                          # straight-through: forward = Enc(Dec(s)), backward = identity (no decoder/encoder activations kept: memory + stability)
            with torch.no_grad():
                proj = ae.project(s.detach())
            prev = s + (proj - s).detach()
        else:
            prev = s
    return torch.stack(means, 1), torch.stack(stds, 1), torch.stack(samples, 1)


def dyn_loss(dyn, ae, x, actions, ctx, free, w_pix=10.0, w_ssim=2.0, pix_every=3, reproject=False, generator=None, detach_every=0):
    """One training loss on real windows x (B,T,3,H,W), T = ctx+free. Returns total, parts (detached floats).
    nll: Gaussian NLL of every predicted frame (teacher-forced and free-running). pix: decoder(mean) vs the real frame on every `pix_every`-th free
    frame (pixel L1 + SSIM through the frozen decoder), the term that keeps the predicted mean on the manifold the decoder renders sharply."""
    with torch.no_grad():
        e = ae.encode(x)
    mu, sd, _ = unroll(dyn, e, actions, ctx, free, ae=ae, reproject=reproject, temperature=1.0, generator=generator, detach_every=detach_every)
    tgt = e[:, 1:ctx + free]
    l_nll = nll(mu, sd, tgt)
    idx = list(range(ctx, ctx + free, pix_every))
    pick = [i - 1 for i in idx]                                                    # frame t sits at index t-1
    pred = ae.decode(mu[:, pick]).reshape(-1, *x.shape[-3:])
    real = x[:, idx].reshape(-1, *x.shape[-3:])
    l_l1 = (pred - real).abs().mean()
    l_ss = ssim_loss(pred.clamp(0, 1), real)
    total = l_nll + w_pix * l_l1 + w_ssim * l_ss
    return total, dict(nll=float(l_nll.detach()), l1=float(l_l1.detach()), ssim=float(l_ss.detach()))


@torch.no_grad()
def dream(dyn, ae, context, ticks, temperature=0.5, reproject=True, actions=None, seed=0, quantise=True):
    """What the UI would show: read `context` (B,Tc,3,H,W) with zero action, then `ticks` times: predict the next latent, render it (shown), feed
    it back (re-projected through Enc(Dec(.)) when `reproject`, the analogue of the UI re-perceiving its own render).
    actions: optional (B,ticks,A) touch actions for the ticks. Returns frames (ticks,B,3,H,W) in [0,1] (8-bit quantised like the page's PNG)."""
    dyn.eval()
    g = torch.Generator().manual_seed(int(seed) & 0x7FFFFFFF)
    b, tc = context.shape[:2]
    dev = context.device
    e = ae.encode(context)
    zero = torch.zeros(b, dyn.action_dim, device=dev)
    h = dyn.initial(b, dev)
    for t in range(1, tc):
        h, _, _ = dyn.step(h, e[:, t - 1], zero)
    anchor, z_in, out = e[:, tc - 1], e[:, tc - 1], []
    for k in range(ticks):
        a = zero if actions is None else actions[:, k]
        h, mu, sd = dyn.step(h, z_in, a, anchor=anchor)
        z = mu + temperature * sd * torch.randn(mu.shape, generator=g).to(dev) if temperature > 0 else mu
        x = ae.decode(z).clamp(0, 1)
        out.append((x * 255).round() / 255 if quantise else x)
        z_in = ae.project(z) if reproject else z
    return torch.stack(out)


@torch.no_grad()
def one_step_mse(dyn, ae, windows, ctx):
    """Teacher-forced one-step image MSE of decoder(mean) vs the real next frame, and the copy-last baseline (same frames), over frames ctx..T-1.
    Reference for how much of the error is the AE itself: mse of decoder(encoder(real next frame)). windows (B,T,3,H,W)."""
    e = ae.encode(windows)
    zero = torch.zeros(windows.shape[0], windows.shape[1], dyn.action_dim, device=windows.device)
    mu, _, _ = unroll(dyn, e, zero, windows.shape[1] - 1, 1, ae=ae, temperature=0.0)
    idx = list(range(ctx, windows.shape[1]))
    pred = ae.decode(mu[:, [i - 1 for i in idx]]).clamp(0, 1)
    real = windows[:, idx]
    copy = windows[:, [i - 1 for i in idx]]
    recon = ae.decode(e[:, idx]).clamp(0, 1)
    f = lambda a, b: float(((a - b) ** 2).mean())
    return dict(model=f(pred, real), copy_last=f(copy, real), ae_floor=f(recon, real))
