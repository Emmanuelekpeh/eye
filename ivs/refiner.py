"""R5 stage 1: a conditional GAN refiner. A residual U-Net sharpens the soft render of a base model (L1 RSSM decode or the S32 autoencoder
reconstruction); a spectral-norm PatchGAN judges [render, candidate] pairs. plan.md "R5". Plain torch, CPU-capable (sizes are multiples of 8).
"""
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm

from .learning import ssim_loss


def _block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(4, o), nn.SiLU(), nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(4, o), nn.SiLU())


class Refiner(nn.Module):
    """out = x + delta(x). 3 down / 3 up stages with skip connections; the last conv is zero-initialised, so an untrained refiner is the identity."""

    def __init__(self, ch=3, width=32):
        super().__init__()
        w = width
        self.e1, self.e2, self.e3, self.mid = _block(ch, w), _block(w, 2 * w), _block(2 * w, 4 * w), _block(4 * w, 4 * w)
        self.d3, self.d2, self.d1 = _block(8 * w, 2 * w), _block(4 * w, w), _block(2 * w, w)
        self.out = nn.Conv2d(w, ch, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        if x.shape[-1] % 8 or x.shape[-2] % 8:
            raise ValueError(f"image sides must be multiples of 8, got {tuple(x.shape[-2:])}")
        up = lambda t: F.interpolate(t, scale_factor=2, mode="bilinear", align_corners=False)
        e1 = self.e1(x)
        e2 = self.e2(F.avg_pool2d(e1, 2))
        e3 = self.e3(F.avg_pool2d(e2, 2))
        m = self.mid(F.avg_pool2d(e3, 2))
        d3 = self.d3(torch.cat([up(m), e3], 1))
        d2 = self.d2(torch.cat([up(d3), e2], 1))
        d1 = self.d1(torch.cat([up(d2), e1], 1))
        return x + self.out(d1)


class PatchDisc(nn.Module):
    """Conditional PatchGAN on cat([render, candidate]) (2*ch channels) -> a map of realism logits (one per ~34 px receptive patch)."""

    def __init__(self, ch=3, width=48):
        super().__init__()
        sn = spectral_norm
        w = width
        self.net = nn.Sequential(
            sn(nn.Conv2d(2 * ch, w, 4, 2, 1)), nn.LeakyReLU(0.2),
            sn(nn.Conv2d(w, 2 * w, 4, 2, 1)), nn.LeakyReLU(0.2),
            sn(nn.Conv2d(2 * w, 4 * w, 4, 2, 1)), nn.LeakyReLU(0.2),
            sn(nn.Conv2d(4 * w, 4 * w, 3, 1, 1)), nn.LeakyReLU(0.2),
            sn(nn.Conv2d(4 * w, 1, 3, 1, 1)))

    def forward(self, render, candidate):
        return self.net(torch.cat([render, candidate], 1))


def d_hinge(real_logits, fake_logits):
    return F.relu(1 - real_logits).mean() + F.relu(1 + fake_logits).mean()


def g_hinge(fake_logits):
    return -fake_logits.mean()


def g_loss(render, out, target, d=None, w_adv=1.0, w_l1=100.0, w_ssim=20.0):
    """Generator loss: w_l1 x L1 + w_ssim x (1 - SSIM) [+ w_adv x hinge adversarial when a discriminator is given]. Returns total, parts (detached)."""
    l1 = (out - target).abs().mean()
    ss = ssim_loss(out.clamp(0, 1), target)
    adv = g_hinge(d(render, out)) if d is not None else out.new_zeros(())
    return w_l1 * l1 + w_ssim * ss + (w_adv * adv if d is not None else 0.0), dict(l1=l1.detach(), ssim=ss.detach(), adv=adv.detach())
