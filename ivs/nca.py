"""EyeNCA: a tiny conditional DENOISING cellular automaton (Diff-NCA style) with EDM preconditioning (Karras et al. 2022).
A second architecture next to the RSSM (ivs/model.py is untouched). See backlog.md P0g for the design and the pre-set gates.

One denoise call D(x, sigma) = c_skip*x + c_out*F(c_in*x, c_noise): F runs `n_inner` NCA updates on a (B, C, H, W) cell grid
(3 visible channels start as c_in*x, the hidden channels start at 0) and reads the visible part out with a 1x1 conv. Every update is
local: each cell sees its own state, Sobel x/y + Laplacian of its state at dilations 1/2/4 (reach 4 px per update, so 8 updates see ~36 px),
the global mean of every channel (one number per channel: lets a 96 px grid agree on global structure), 2 fixed coordinate planes, the noise
level and the (constant) noisy input. The update MLP's last layer is zero-initialised (residual update: an untrained net leaves the state alone).
The same denoiser is used for (1) generation (Heun sampler from noise), (2) blind repair (SDEdit: noise the damaged frame to sigma_start,
denoise back), (3) masked repair (RePaint-style replacement when the damaged region is known) and, later (stage E1), a live loop.
Images are in [-1, 1] inside the model; `to_model` / `to_unit` convert from / to [0, 1].
Weights are plain tensors; pass the EMA copy (`ema_update`) to every sampler.
"""
import copy
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class NCAConfig:
    image_size: int = 96
    channels: int = 3                      # visible channels (RGB)
    state: int = 32                        # C: visible + hidden channels per cell
    hidden: int = 128                      # width of the per-cell update MLP
    n_inner: int = 8                       # NCA updates per denoise call
    dilations: tuple = (1, 2, 4)
    cond_dim: int = 0                      # extra per-cell conditioning planes (E1: previous frame, gaze command, touch)
    sigma_data: float = 0.5
    fire_rate: float = 1.0                 # <1: each cell updates with this probability per step (asynchronous NCA)
    # ---- PD-NCA (persistent denoising NCA, backlog P0j). All off = the E0 network, bit-for-bit and parameter-for-parameter.
    gate: bool = False                     # GRU-style update on the HIDDEN channels: h' = (1-u)*h + u*tanh(cand); visible channels stay additive
    fourier: bool = False                  # global pathway: rfft2 of the state -> 1x1 MLP over [re, im, freq coords] -> irfft2 -> extra perception features
    fourier_hidden: int = 64
    lattn: bool = False                    # neighbourhood attention over 3x3 at every dilation (27 keys/cell) with learned relative-position bias
    heads: int = 2
    attn_dk: int = 4                       # query/key width per head
    attn_dv: int = 4                       # value width per head
    film: bool = False                     # sigma conditioning as scale+shift on the update MLP's hidden layer, plus per-cell RMS normalisation
    arch: str = "nca"                      # network family marker saved in checkpoints ('vitca' = ivs/vitca.py); see net_from_config

    def hidden_shape(self):
        """(channels, H, W) of the persistent hidden state of ONE sample."""
        return (self.state - self.channels, self.image_size, self.image_size)


NCA_PRESETS = {                            # parameter counts are measured, see tests/test_nca.py
    "N50": dict(state=32, hidden=128),
    "N135": dict(state=48, hidden=232),
}


def to_model(x01):
    return x01 * 2.0 - 1.0


def to_unit(x):
    return ((x + 1.0) * 0.5).clamp(0.0, 1.0)


def edm_coeffs(sigma, sigma_data):
    """EDM preconditioning (Karras et al. 2022, table 1): c_skip, c_out, c_in, c_noise for noise level(s) sigma (tensor)."""
    s2 = sigma ** 2 + sigma_data ** 2
    return sigma_data ** 2 / s2, sigma * sigma_data / s2.sqrt(), 1.0 / s2.sqrt(), sigma.log() / 4.0


def edm_weight(sigma, sigma_data):
    return (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2


def _filters():
    sx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8.0
    lap = torch.tensor([[1.0, 2.0, 1.0], [2.0, -12.0, 2.0], [1.0, 2.0, 1.0]]) / 16.0
    return torch.stack([sx, sx.t().contiguous(), lap]).unsqueeze(1)          # 3,1,3,3


class EyeNCA(nn.Module):
    def __init__(self, cfg: NCAConfig):
        super().__init__()
        assert cfg.state > cfg.channels, "need at least one hidden channel"
        self.cfg = cfg
        c, h = cfg.state, cfg.hidden
        self.register_buffer("filt", _filters().repeat(c, 1, 1, 1), persistent=False)      # groups=c: channel g -> outputs 3g..3g+2
        n_local = c + 3 * c * len(cfg.dilations) + cfg.channels + cfg.cond_dim
        if cfg.fourier:
            n_local += c
        if cfg.lattn:
            n_local += cfg.heads * cfg.attn_dv
        # first layer of the update MLP == a 1x1 conv over [local features | global mean | noise level | coordinates]; the last three are
        # spatially constant or fixed, so they enter as biases (mathematically identical to concatenating the planes, far less memory)
        self.local = nn.Conv2d(n_local, h, 1)
        self.glob = nn.Linear(c + 1, h, bias=False)
        self.pos = nn.Conv2d(2, h, 1, bias=False)
        self.update = nn.Conv2d(h, c, 1)
        nn.init.zeros_(self.update.weight)
        nn.init.zeros_(self.update.bias)
        self.readout = nn.Conv2d(c, cfg.channels, 1)
        s = cfg.image_size
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, s), torch.linspace(-1, 1, s), indexing="ij")
        self.register_buffer("coords", torch.stack([xs, ys]).unsqueeze(0), persistent=False)   # 1,2,H,W
        if cfg.gate:
            self.gate_conv = nn.Conv2d(h, c - cfg.channels, 1)
            nn.init.zeros_(self.gate_conv.weight)
            nn.init.constant_(self.gate_conv.bias, -1.5)                    # u ~ 0.18 at init: hidden state is kept rather than overwritten
        if cfg.fourier:
            self.fw1 = nn.Conv2d(2 * c + 2, cfg.fourier_hidden, 1)
            self.fw2 = nn.Conv2d(cfg.fourier_hidden, 2 * c, 1)
            nn.init.zeros_(self.fw2.weight)
            nn.init.zeros_(self.fw2.bias)                                   # the global pathway starts silent
            fu = torch.fft.fftfreq(s).view(s, 1).expand(s, s // 2 + 1) * 2.0
            fv = torch.fft.rfftfreq(s).view(1, s // 2 + 1).expand(s, s // 2 + 1) * 2.0
            self.register_buffer("fcoords", torch.stack([fu, fv]).unsqueeze(0), persistent=False)   # 1,2,H,W//2+1 (frequency position)
        if cfg.lattn:
            self.wq = nn.Conv2d(c, cfg.heads * cfg.attn_dk, 1)
            self.wk = nn.Conv2d(c, cfg.heads * cfg.attn_dk, 1)
            self.wv = nn.Conv2d(c, cfg.heads * cfg.attn_dv, 1)
            self.rel = nn.Parameter(torch.zeros(cfg.heads, 9 * len(cfg.dilations)))     # relative-position bias per head and neighbour
        if cfg.film:
            self.film_net = nn.Sequential(nn.Linear(16, 64), nn.SiLU(), nn.Linear(64, 2 * h))
            nn.init.zeros_(self.film_net[2].weight)
            nn.init.zeros_(self.film_net[2].bias)                           # scale 0 / shift 0 at init
            self.register_buffer("film_freqs", torch.tensor([2.0 ** i for i in range(8)]), persistent=False)

    def perceive(self, state):
        feats = [state]
        for d in self.cfg.dilations:
            feats.append(F.conv2d(F.pad(state, (d, d, d, d), mode="replicate"), self.filt, groups=self.cfg.state, dilation=d))
        return feats

    def _fourier(self, state):
        """Global pathway (FourierDiff-NCA idea): a 1x1 MLP applied per frequency, then back to image space (B, C, H, W)."""
        c = self.cfg.state
        X = torch.fft.rfft2(state, norm="ortho")
        y = self.fw2(F.silu(self.fw1(torch.cat([X.real, X.imag, self.fcoords.expand(state.shape[0], -1, -1, -1)], 1))))
        return torch.fft.irfft2(torch.complex(y[:, :c], y[:, c:]), s=state.shape[-2:], norm="ortho")

    def _attn(self, state):
        """Neighbourhood attention: each cell attends over its 3x3 neighbours at every dilation (27 keys), per head, with a learned bias per
        neighbour position. Returns (B, heads*dv, H, W)."""
        cf = self.cfg
        b, _, hh, ww = state.shape
        q = self.wq(state).view(b, 1, cf.heads, cf.attn_dk, hh, ww)
        k, v = self.wk(state), self.wv(state)
        ks, vs = [], []
        for d in cf.dilations:
            kp, vp = F.pad(k, (d, d, d, d), mode="replicate"), F.pad(v, (d, d, d, d), mode="replicate")
            for i in range(3):
                for j in range(3):
                    ks.append(kp[:, :, i * d:i * d + hh, j * d:j * d + ww])
                    vs.append(vp[:, :, i * d:i * d + hh, j * d:j * d + ww])
        K = torch.stack(ks, 1).view(b, len(ks), cf.heads, cf.attn_dk, hh, ww)
        V = torch.stack(vs, 1).view(b, len(vs), cf.heads, cf.attn_dv, hh, ww)
        logits = (q * K).sum(3) * cf.attn_dk ** -0.5 + self.rel.t()[None, :, :, None, None]       # B,27,heads,H,W
        a = torch.softmax(logits, dim=1)
        return (a.unsqueeze(3) * V).sum(1).reshape(b, cf.heads * cf.attn_dv, hh, ww)

    def _film(self, c_noise):
        """(scale, shift) of the update MLP's hidden layer from the noise level, or None when film is off."""
        if not self.cfg.film:
            return None
        e = c_noise[:, None] * self.film_freqs[None]
        out = self.film_net(torch.cat([e.sin(), e.cos()], 1))
        scale, shift = out.chunk(2, 1)
        return scale[:, :, None, None], shift[:, :, None, None]

    def _step(self, state, x_in, cond, c_noise, pos, film=None):
        cf = self.cfg
        feats = self.perceive(state) + [x_in] + ([cond] if cond is not None else [])
        if cf.fourier:
            feats.append(self._fourier(state))
        if cf.lattn:
            feats.append(self._attn(state))
        g = self.glob(torch.cat([state.mean((2, 3)), c_noise[:, None]], 1))[:, :, None, None]
        pre = self.local(torch.cat(feats, 1))
        if film is not None:
            pre = pre * torch.rsqrt(pre.pow(2).mean(1, keepdim=True) + 1e-6)             # per-cell RMS normalisation over channels
            pre = pre * (1.0 + film[0]) + film[1]
        z = F.silu(pre + g + pos)
        upd = self.update(z)
        if cf.gate:
            ch = cf.channels
            u = torch.sigmoid(self.gate_conv(z))
            new = torch.cat([state[:, :ch] + upd[:, :ch], (1.0 - u) * state[:, ch:] + u * torch.tanh(upd[:, ch:])], 1)
            if cf.fire_rate < 1.0:
                m = (torch.rand(state.shape[0], 1, *state.shape[2:], device=state.device) < cf.fire_rate).to(new.dtype)
                new = state + m * (new - state)
            return new
        if cf.fire_rate < 1.0:
            upd = upd * (torch.rand(state.shape[0], 1, *state.shape[2:], device=state.device) < cf.fire_rate).to(upd.dtype)
        return state + upd

    def run_state(self, x_in, c_noise, cond=None, checkpoint_steps=False, hidden=None):
        """The cell grid after n_inner updates, (B, C, H, W). x_in = c_in * noisy image. hidden (B, C-ch, H, W) = the persistent hidden
        channels carried over from the previous call (None = zeros = the E0 behaviour)."""
        b, _, hh, ww = x_in.shape
        if hidden is None:
            hidden = x_in.new_zeros(b, self.cfg.state - self.cfg.channels, hh, ww)
        state = torch.cat([x_in, hidden], 1)
        pos = self.pos(self.coords)
        film = self._film(c_noise)
        for _ in range(self.cfg.n_inner):
            if checkpoint_steps and torch.is_grad_enabled():
                state = checkpoint(self._step, state, x_in, cond, c_noise, pos, film, use_reentrant=False)
            else:
                state = self._step(state, x_in, cond, c_noise, pos, film)
        return state

    def denoise(self, x, sigma, cond=None, checkpoint_steps=False, hidden=None, return_hidden=False):
        """D(x, sigma): estimate of the clean image given the noisy x (B,ch,H,W, model range) and noise level sigma (float or (B,)).
        return_hidden=True returns (D, hidden after this call) so the caller can carry the state to the next call (persistence)."""
        sigma = torch.as_tensor(sigma, dtype=x.dtype, device=x.device).reshape(-1).expand(x.shape[0])
        c_skip, c_out, c_in, c_noise = edm_coeffs(sigma, self.cfg.sigma_data)
        v = lambda t: t.view(-1, 1, 1, 1)
        state = self.run_state(v(c_in) * x, c_noise, cond, checkpoint_steps, hidden)
        d = v(c_skip) * x + v(c_out) * self.readout(state)
        return (d, state[:, self.cfg.channels:]) if return_hidden else d

    forward = denoise


def net_from_config(config):
    """The network for a saved config dict (checkpoint['config']): EyeNCA (arch 'nca', also every checkpoint saved before the marker existed) or EyeViTCA."""
    config = dict(config)
    if config.get("arch", "nca") == "vitca":
        from . import vitca
        return vitca.EyeViTCA(vitca.ViTCAConfig(**config))
    config["dilations"] = tuple(config["dilations"])
    return EyeNCA(NCAConfig(**config))


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def build(preset=None, **kw):
    """EyeNCA from a preset name (NCA_PRESETS) and/or NCAConfig overrides."""
    return EyeNCA(NCAConfig(**{**(NCA_PRESETS[preset] if preset else {}), **kw}))


@torch.no_grad()
def ema_update(ema, model, decay):
    for pe, p in zip(ema.parameters(), model.parameters()):
        pe.lerp_(p, 1.0 - decay)
    for be, b in zip(ema.buffers(), model.buffers()):
        be.copy_(b)


def make_ema(model):
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    return ema


def _randn(shape, generator, device):
    return torch.randn(shape, generator=generator, device=generator.device if generator is not None else device).to(device)


# ----------------------------------------------------------------------------------------------- training loss
def edm_loss(model, x, sigma=None, generator=None, p_mean=-1.2, p_std=1.2, cond=None, checkpoint_steps=False):
    """EDM denoising loss on clean images x (B,ch,H,W) in model range. sigma ~ lognormal(p_mean, p_std) unless given (B,).
    Returns (loss, per-sample unweighted MSE (detached), sigma)."""
    b = x.shape[0]
    if sigma is None:
        sigma = torch.exp(p_mean + p_std * _randn((b,), generator, x.device))
    sigma = torch.as_tensor(sigma, dtype=x.dtype, device=x.device).reshape(-1).expand(b)
    noise = _randn(x.shape, generator, x.device)
    d = model.denoise(x + noise * sigma.view(-1, 1, 1, 1), sigma, cond, checkpoint_steps)
    per = ((d - x) ** 2).mean((1, 2, 3))
    return (edm_weight(sigma, model.cfg.sigma_data) * per).mean(), per.detach(), sigma


# ----------------------------------------------------------------------------------------------- samplers
def sigma_schedule(n, sigma_min, sigma_max, rho=7.0):
    """EDM time steps: n levels from sigma_max down to sigma_min, followed by 0."""
    assert n >= 2 and 0 < sigma_min < sigma_max
    i = torch.arange(n, dtype=torch.float64) / (n - 1)
    s = (sigma_max ** (1 / rho) + i * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    return torch.cat([s, torch.zeros(1, dtype=torch.float64)]).float()


@torch.no_grad()
def run_sampler(model, x, sigmas, cond=None, known=None, mask=None, generator=None, repeats=1, hidden=None, persist=False):
    """Heun sampler (EDM algorithm 1) from x (at noise level sigmas[0]) down to sigmas[-1] = 0. Deterministic given x.
    known/mask (B,ch,H,W) / (B,1,H,W) (mask 1 = known pixel): after every step the known region is replaced by the known image diffused to
    the current level (replacement inpainting) and the final image keeps the known pixels exactly. repeats > 1: RePaint-style resampling
    (after each level except the last repeat, jump back up to that level with fresh noise and denoise again).
    persist=True: the hidden channels are carried through EVERY denoise call of the trajectory (starting from `hidden`, None = zeros) and the
    function returns (x, final hidden); persist=False is the E0 behaviour (state rebuilt on every call) and returns x."""
    st = [hidden]

    def D(z, s):
        if not persist:
            return model.denoise(z, s, cond)
        d, st[0] = model.denoise(z, s, cond, hidden=st[0], return_hidden=True)
        return d

    for i in range(len(sigmas) - 1):
        s, sn = float(sigmas[i]), float(sigmas[i + 1])
        for r in range(repeats):
            d = (x - D(x, s)) / s
            xn = x + (sn - s) * d
            if sn > 0:
                d2 = (xn - D(xn, sn)) / sn
                xn = x + (sn - s) * 0.5 * (d + d2)
            if mask is not None:
                kn = known + sn * _randn(known.shape, generator, x.device) if sn > 0 else known
                xn = mask * kn + (1 - mask) * xn
            if r < repeats - 1 and sn > 0:
                xn = xn + math.sqrt(max(s * s - sn * sn, 0.0)) * _randn(xn.shape, generator, x.device)
            x = xn
    return (x, st[0]) if persist else x


@torch.no_grad()
def sample(model, n, steps=18, sigma_min=0.01, sigma_max=5.0, rho=7.0, cond=None, generator=None, device=None, persist=False):
    """n dreams from pure noise, returned in [0,1]. persist=True also returns the final hidden state (for living-mode ticks)."""
    dev = device or next(model.parameters()).device
    sig = sigma_schedule(steps, sigma_min, sigma_max, rho)
    c, s = model.cfg.channels, model.cfg.image_size
    x = _randn((n, c, s, s), generator, dev) * sig[0]
    out = run_sampler(model, x, sig, cond, generator=generator, persist=persist)
    return (to_unit(out[0]), out[1]) if persist else to_unit(out)


@torch.no_grad()
def repair_blind(model, x01, sigma_start, steps=12, rho=7.0, cond=None, generator=None, hidden=None, persist=False):
    """SDEdit: diffuse the damaged frame (in [0,1]) to sigma_start, then denoise back. No knowledge of where the damage is.
    sigma_start -> 0 returns the input. persist=True: carry (and return) the hidden state, starting from `hidden`."""
    sig = sigma_schedule(steps, min(0.01, sigma_start / 2), sigma_start, rho)
    x = to_model(x01)
    x = x + sig[0] * _randn(x.shape, generator, x.device)
    out = run_sampler(model, x, sig, cond, generator=generator, hidden=hidden, persist=persist)
    return (to_unit(out[0]), out[1]) if persist else to_unit(out)


@torch.no_grad()
def repair_masked(model, x01, known_mask, steps=18, sigma_start=5.0, sigma_min=0.01, rho=7.0, repeats=1, cond=None, generator=None,
                  hidden=None, persist=False):
    """Inpaint the pixels where known_mask (B,1,H,W) is 0, keeping the known ones exactly (returned in [0,1])."""
    sig = sigma_schedule(steps, sigma_min, sigma_start, rho)
    known = to_model(x01)
    m = known_mask.to(known.dtype)
    x = m * (known + sig[0] * _randn(known.shape, generator, known.device)) + (1 - m) * sig[0] * _randn(known.shape, generator, known.device)
    out = run_sampler(model, x, sig, cond, known=known, mask=m, generator=generator, repeats=repeats, hidden=hidden, persist=persist)
    return (to_unit(out[0]), out[1]) if persist else to_unit(out)


@torch.no_grad()
def live(model, x01, hidden, n_ticks, sigma=0.05, cond=None, generator=None, every=None):
    """LIVING MODE: the network running on its own render. Each tick re-noises the current image to `sigma`, denoises it with the persistent
    hidden state, and feeds the result back (the CA's homeostasis loop; denoising = self-repair). Returns (final image [0,1], hidden) or,
    with `every=k`, also a list of (tick, image) snapshots every k ticks (tick 0 = the input)."""
    x, h = to_model(x01), hidden
    snaps = [(0, to_unit(x))] if every else None
    for t in range(1, n_ticks + 1):
        d, h = model.denoise(x + sigma * _randn(x.shape, generator, x.device), sigma, cond, hidden=h, return_hidden=True)
        x = d.clamp(-1, 1)
        if every and t % every == 0:
            snaps.append((t, to_unit(x)))
    return (to_unit(x), h, snaps) if every else (to_unit(x), h)
