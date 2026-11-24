"""EyeViTCA: a recurrent, attention-based denoising cellular automaton (ViTCA, Tesfaldet et al. 2022, with EDM preconditioning like ivs/nca.py).

Why: EyeNCA / PD-NCA (backlog P0g, P0j) denoise well but dream cloud textures: local cells never agree on a global layout. Here every CELL is a patch
TOKEN and every update is GLOBAL self-attention (144 tokens at patch 8 on 96 px), while the living-strata properties are kept: the per-token state is
PERSISTENT (carried across denoise calls, `hidden=` / `return_hidden=`, exactly the EyeNCA contract) and BOUNDED (GRU-style gate with a tanh candidate,
the only living loop that stayed stable in P0j round 1).

One denoise call D(x, sigma) = c_skip*x + c_out*G(c_in*x, c_noise, state):
  e = patch_embed(c_in * x) + learned position            (constant over the loops of one call)
  repeat `loops` times with the SAME weights (ViTCA weight sharing = recurrent depth):
      z = state + e ; for each of `blocks` pre-LN transformer blocks: z += attn(FiLM(LN z)) ; z += mlp(FiLM(LN z))     FiLM = sigma-conditioned scale+shift
      f = LN(z) ; u = sigmoid(W_u f + b_u) ; state = (1-u)*state + u*tanh(W_c f)           b_u = -1.5: slow forgetting at the start
  image = unpatchify(LN(state)) (+ a zero-initialised depthwise 3x3 refinement against patch seams)
The unpatchify layer is zero-initialised, so an untrained net is the EDM skip connection (D = c_skip*x) like the NCA's zero-initialised update.
Interface (duck-typed against ivs.nca): .cfg (image_size, channels, sigma_data, cond_dim, hidden_shape()), denoise(x, sigma, cond=None, checkpoint_steps=False,
hidden=None, return_hidden=False). hidden is laid out (B, dim, H/patch, W/patch) so pools / samplers / viewer treat it like the NCA's hidden channels.
Checkpoints carry config['arch'] == 'vitca'; ivs.nca.net_from_config builds the right class.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from . import nca


@dataclass
class ViTCAConfig:
    arch: str = "vitca"
    image_size: int = 96
    channels: int = 3
    patch: int = 8
    dim: int = 256
    blocks: int = 3                        # distinct transformer blocks, shared across the loops
    loops: int = 4                         # recurrent applications of the block stack per denoise call
    vheads: int = 8
    mlp_mult: int = 4
    cond_width: int = 64                   # width of the sigma embedding that drives the FiLM layers
    cond_dim: int = 0                      # unused (kept so the viewer / eval treat both architectures alike)
    sigma_data: float = 0.5
    gate_bias: float = -1.5

    def grid(self):
        assert self.image_size % self.patch == 0 and self.dim % self.vheads == 0
        return self.image_size // self.patch

    def hidden_shape(self):
        g = self.grid()
        return (self.dim, g, g)


VITCA_PRESETS = {                          # parameter counts are measured in tests/test_vitca.py
    "V2": dict(blocks=2),
    "V3": dict(blocks=3),
}


class Block(nn.Module):
    def __init__(self, dim, heads, mult, cond_width):
        super().__init__()
        self.heads = heads
        self.n1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.n2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.fc1 = nn.Linear(dim, mult * dim)
        self.fc2 = nn.Linear(mult * dim, dim)
        self.film = nn.Linear(cond_width, 4 * dim)           # scale1, shift1, scale2, shift2 from the sigma embedding
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, z, c):
        sc1, sh1, sc2, sh2 = self.film(c).unsqueeze(1).chunk(4, -1)
        b, n, d = z.shape
        h = self.n1(z) * (1 + sc1) + sh1
        q, k, v = self.qkv(h).view(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v)                       # global attention over every token
        z = z + self.proj(a.transpose(1, 2).reshape(b, n, d))
        h = self.n2(z) * (1 + sc2) + sh2
        return z + self.fc2(F.gelu(self.fc1(h)))


class EyeViTCA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        g, d, p, ch = cfg.grid(), cfg.dim, cfg.patch, cfg.channels
        self.embed = nn.Conv2d(ch, d, p, p)
        self.pos = nn.Parameter(torch.randn(1, g * g, d) * 0.02)
        self.register_buffer("freqs", torch.tensor([2.0 ** i for i in range(8)]), persistent=False)
        self.cmlp = nn.Sequential(nn.Linear(17, cfg.cond_width), nn.SiLU(), nn.Linear(cfg.cond_width, cfg.cond_width), nn.SiLU())
        self.blocks = nn.ModuleList([Block(d, cfg.vheads, cfg.mlp_mult, cfg.cond_width) for _ in range(cfg.blocks)])
        self.out_norm = nn.LayerNorm(d)
        self.cand = nn.Linear(d, d)
        self.gate = nn.Linear(d, d)
        nn.init.constant_(self.gate.bias, cfg.gate_bias)
        self.read_norm = nn.LayerNorm(d)
        self.readout = nn.ConvTranspose2d(d, ch, p, p)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)
        self.refine = nn.Conv2d(ch, ch, 3, padding=1, groups=ch)
        nn.init.zeros_(self.refine.weight)
        nn.init.zeros_(self.refine.bias)

    def _cond(self, c_noise):
        a = c_noise.view(-1, 1) * self.freqs.view(1, -1) * math.pi
        return self.cmlp(torch.cat([c_noise.view(-1, 1), a.sin(), a.cos()], 1))

    def _loop(self, s, e, c):
        z = s + e
        for blk in self.blocks:
            z = blk(z, c)
        f = self.out_norm(z)
        u = torch.sigmoid(self.gate(f))
        return (1 - u) * s + u * torch.tanh(self.cand(f))

    def run_state(self, x_in, c_noise, checkpoint_steps=False, hidden=None):
        """Token state (B, N, dim) after `loops` recurrent updates. hidden (B, dim, g, g) = the state carried from the previous call (None = zeros)."""
        b = x_in.shape[0]
        g, d = self.cfg.grid(), self.cfg.dim
        e = self.embed(x_in).flatten(2).transpose(1, 2) + self.pos
        c = self._cond(c_noise)
        s = x_in.new_zeros(b, g * g, d) if hidden is None else hidden.flatten(2).transpose(1, 2)
        for _ in range(self.cfg.loops):
            if checkpoint_steps and torch.is_grad_enabled():
                s = checkpoint(self._loop, s, e, c, use_reentrant=False)
            else:
                s = self._loop(s, e, c)
        return s

    def denoise(self, x, sigma, cond=None, checkpoint_steps=False, hidden=None, return_hidden=False):
        """D(x, sigma): estimate of the clean image given the noisy x (B,ch,H,W, model range) at noise level sigma (float or (B,)).
        return_hidden=True returns (D, state after this call) for persistence. cond is accepted for interface parity and must be None."""
        assert cond is None, "EyeViTCA has no conditioning planes"
        sigma = torch.as_tensor(sigma, dtype=x.dtype, device=x.device).reshape(-1).expand(x.shape[0])
        c_skip, c_out, c_in, c_noise = nca.edm_coeffs(sigma, self.cfg.sigma_data)
        v = lambda t: t.view(-1, 1, 1, 1)
        s = self.run_state(v(c_in) * x, c_noise, checkpoint_steps, hidden)
        g, d = self.cfg.grid(), self.cfg.dim
        grid = self.read_norm(s).transpose(1, 2).reshape(x.shape[0], d, g, g)
        y = self.readout(grid)
        y = y + self.refine(y)
        out = v(c_skip) * x + v(c_out) * y
        return (out, s.transpose(1, 2).reshape(x.shape[0], d, g, g)) if return_hidden else out

    forward = denoise
