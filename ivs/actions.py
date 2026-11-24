"""Synthetic, action-labelled interventions applied to real frames.

The hidden intervention state s = [shift_x, shift_y, log_gain, light_x, light_y]
evolves as a smooth random walk. The IMAGES show the effect of s; the ACTION fed
to the dynamics model is the per-step change (s_t - s_{t-1}) / step. The model is
never given s itself and must infer it from pixels, which keeps the 'no privileged
state' rule while giving the dynamics a real cause->effect signal.

Caveat: this teaches the model OUR transformation family, not real optics. Keep
evaluating real-video dynamics (blink, pupil) separately from intervention dynamics.
"""
import torch
import torch.nn.functional as F


def luma(x):
    """RGB (…,3,H,W) -> gray replicated to 3 channels (ITU-R BT.601). 1-channel input is returned unchanged."""
    if x.shape[-3] == 1:
        return x
    g = 0.299 * x[..., 0:1, :, :] + 0.587 * x[..., 1:2, :, :] + 0.114 * x[..., 2:3, :, :]
    return g.expand_as(x).contiguous()


@torch.no_grad()
def scatter_batch(x, generator=None, max_discs=4, radius=(0.08, 0.25), strength=(0.3, 1.0), p_disc=0.7):
    """Batched, channel-safe version of the UI's scatter: shuffle WHOLE pixels (all channels together) inside random
    discs. x: N,C,H,W. Pixel values are only moved, never invented, so each frame keeps its value multiset; pixels
    outside the selected discs are untouched. Disc centres/radii are fractions of the width.
    Vectorised with a sort trick: selected pixels get key (disc, random) vs (disc, position); sorting both and pairing
    ranks gives a uniform random permutation inside each disc."""
    n, c, h, w = x.shape
    dev = x.device
    gdev = generator.device if generator is not None else dev        # draw where the generator lives (GPU-native when possible)
    u = lambda *s: torch.rand(*s, generator=generator, device=gdev).to(dev)
    xs = ((torch.arange(w, device=dev) + 0.5) / w).view(1, 1, w)
    ys = ((torch.arange(h, device=dev) + 0.5) / h).view(1, h, 1)
    label = torch.full((n, h, w), -1, dtype=torch.long, device=dev)
    for d in range(max_discs):
        cx, cy = u(n).view(n, 1, 1) * 0.8 + 0.1, u(n).view(n, 1, 1) * 0.8 + 0.1
        r = (radius[0] + (radius[1] - radius[0]) * u(n)).view(n, 1, 1)
        k = (strength[0] + (strength[1] - strength[0]) * u(n)).view(n, 1, 1)
        active = (u(n) < (p_disc if d else 1.0)).view(n, 1, 1)
        inside = ((xs - cx) ** 2 + (ys - cy) ** 2 <= r ** 2) & (u(n, h, w) < k) & active
        label = torch.where(inside & (label < 0), torch.full_like(label, d), label)
    hw = h * w
    lab = label.view(n, hw)
    sel = lab >= 0
    pos = torch.arange(hw, device=dev, dtype=torch.float32).view(1, hw) / hw
    key_pos = torch.where(sel, lab.float() * 2.0 + pos, 1000.0 + pos)           # selected: (disc, position); rest: position
    key_rnd = torch.where(sel, lab.float() * 2.0 + u(n, hw), 1000.0 + pos)      # selected: (disc, random);   rest: position
    src = torch.argsort(key_rnd, dim=1)
    dst = torch.argsort(key_pos, dim=1)
    src_of = torch.empty_like(dst).scatter_(1, dst, src)                          # out[dst[i]] = x[src[i]]
    out = x.reshape(n, c, hw).gather(2, src_of.unsqueeze(1).expand(-1, c, -1))
    return out.view(n, c, h, w)


class InterventionSampler:
    ACTION_DIM = 5

    def __init__(self, p: float = 0.5, step: float = 0.06, max_shift: float = 0.2,
                 gain_range: float = 0.5, light_sigma: float = 0.3, light_strength: float = 0.5):
        self.p, self.step = p, step
        self.max_shift, self.gain_range = max_shift, gain_range
        self.light_sigma, self.light_strength = light_sigma, light_strength
        self.limit = torch.tensor([max_shift, max_shift, gain_range, 0.8, 0.8])

    def _walk(self, b, t, generator):
        lim = self.limit
        s0 = (torch.rand(b, 5, generator=generator) * 2 - 1) * lim * 0.5
        deltas = torch.randn(b, t, 5, generator=generator) * self.step * lim / lim[0]
        states, s = [], s0
        for i in range(t):
            if i > 0:
                s = torch.max(torch.min(s + deltas[:, i], lim), -lim)
            states.append(s)
        states = torch.stack(states, 1)                                   # B,T,5
        prev = torch.cat([states[:, :1], states[:, :-1]], 1)
        actions = (states - prev) / (self.step * lim / lim[0])             # ~unit scale; action_0 = 0
        return states, actions

    def _shift(self, x, st, sign):
        theta = torch.zeros(x.shape[0], 2, 3, device=x.device)
        theta[:, 0, 0] = 1.0
        theta[:, 1, 1] = 1.0
        theta[:, 0, 2] = -2 * sign * st[:, 0]
        theta[:, 1, 2] = -2 * sign * st[:, 1]
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        return F.grid_sample(x, grid, padding_mode="border", align_corners=False)

    def _blob(self, st, h, w, device):
        ys = torch.linspace(-1, 1, h, device=device).view(1, 1, h, 1)
        xs = torch.linspace(-1, 1, w, device=device).view(1, 1, 1, w)
        d2 = (xs - st[:, 3].view(-1, 1, 1, 1)) ** 2 + (ys - st[:, 4].view(-1, 1, 1, 1)) ** 2
        return torch.exp(-d2 / (2 * self.light_sigma ** 2))

    @staticmethod
    def _light_gain(light, device):
        """light=None -> 1 (training: every state carries its light). Otherwise a 0..1 gain per sample (scalar or (b*t,)): the UI passes 0 while the
        user has not switched the light on, so a zero touch state means NO light (the state's x,y are only the light's position)."""
        return 1.0 if light is None else torch.as_tensor(light, dtype=torch.float32, device=device).reshape(-1, 1, 1, 1)

    def apply_state(self, images, states, light=None):
        b, t, c, h, w = images.shape
        x = images.reshape(b * t, c, h, w)
        st = states.reshape(b * t, 5).to(images.device)
        x = self._shift(x, st, 1.0)
        x = x * torch.exp(st[:, 2]).view(-1, 1, 1, 1) + self.light_strength * self._light_gain(light, images.device) * self._blob(st, h, w, images.device)
        return x.clamp(0, 1).reshape(b, t, c, h, w)

    def invert_state(self, images, states, light=None):
        """Approximate inverse of apply_state: strip light, exposure and shift of a known state.
        Lossy where apply_state clamped or where a shift exposed border-replicated pixels. Used by the
        UI so the model's own canvas is stored 'untouched' and the touch is re-applied each tick
        instead of compounding."""
        b, t, c, h, w = images.shape
        x = images.reshape(b * t, c, h, w)
        st = states.reshape(b * t, 5).to(images.device)
        x = (x - self.light_strength * self._light_gain(light, images.device) * self._blob(st, h, w, images.device)) / torch.exp(st[:, 2]).view(-1, 1, 1, 1)
        x = self._shift(x.clamp(0, 1), st, -1.0)
        return x.clamp(0, 1).reshape(b, t, c, h, w)

    @torch.no_grad()
    def __call__(self, images, generator=None, return_states=False):
        """images: B,T,C,H,W -> (images_aug, actions B,T,5). Per-sample mask keeps
        a fraction p of clips intervened; the rest stay passive with zero actions.
        return_states=True also returns the applied hidden state (B,T,5; zeros for passive clips) so labels that live in image
        coordinates (pupil position) can follow the shift: position += 2 * state[..., :2] in [-1,1] image units."""
        b, t = images.shape[:2]
        states, actions = self._walk(b, t, generator)
        mask = (torch.rand(b, generator=generator) < self.p)
        aug = self.apply_state(images, states)
        m = mask.to(images.device).view(b, 1, 1, 1, 1)
        out_images = torch.where(m, aug, images)
        out_actions = torch.where(mask.view(b, 1, 1), actions, torch.zeros_like(actions)).to(images.device)
        if return_states:
            return out_images, out_actions, (states * mask.view(b, 1, 1)).to(images.device)
        return out_images, out_actions
