import torch
import torch.nn.functional as F


def gaussian_kl(posterior, prior):
    # KL(q || p), q = observation-conditioned posterior, p = prior. Summed over latent dims.
    q_mean, q_std = posterior
    p_mean, p_std = prior
    return (torch.log(p_std / q_std) +
            (q_std.pow(2) + (q_mean - p_mean).pow(2)) / (2 * p_std.pow(2)) - 0.5).sum(-1)


def _pixel_sum_loss(pred, target, kind="smoothl1"):
    # summed over C,H,W, averaged over batch/time (comparable scale to a KL in nats).
    # smoothl1 (default, = 0.5*d^2 for |d|<1) | l1 (sharper: constant gradient at small errors)
    fn = F.l1_loss if kind == "l1" else F.smooth_l1_loss
    return fn(pred, target, reduction="none").flatten(start_dim=2).sum(-1).mean()


def edge_loss(pred, target):
    """L1 between horizontal/vertical finite-difference images (B,T,C,H,W), summed over pixels, mean over B,T.
    Pixel losses reward blur; matching gradients directly rewards edges/texture."""
    dx = lambda x: x[..., :, 1:] - x[..., :, :-1]
    dy = lambda x: x[..., 1:, :] - x[..., :-1, :]
    s = lambda a, b: (a - b).abs().flatten(start_dim=2).sum(-1).mean()
    return s(dx(pred), dx(target)) + s(dy(pred), dy(target))


def ssim_loss(pred, target, win=11, sigma=1.5):
    """1 - mean SSIM between (B,C,H,W) images in [0,1] (gaussian window, 'valid' region, per channel), averaged over batch: 0 for identical images.
    Unlike an L1 on gradients it penalises a prediction whose local CONTRAST is lower than the target's, so a flat blur is not its optimum."""
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    k = torch.exp(-((torch.arange(win, dtype=pred.dtype, device=pred.device) - (win - 1) / 2) ** 2) / (2 * sigma ** 2))
    k = k / k.sum()
    w = (k[:, None] * k[None, :])[None, None].repeat(pred.shape[1], 1, 1, 1)
    blur = lambda x: F.conv2d(x, w, groups=x.shape[1])
    mp, mt = blur(pred), blur(target)
    vp, vt, cov = blur(pred * pred) - mp * mp, blur(target * target) - mt * mt, blur(pred * target) - mp * mt
    ssim = ((2 * mp * mt + c1) * (2 * cov + c2)) / ((mp * mp + mt * mt + c1) * (vp + vt + c2))
    return 1.0 - ssim.mean()


def geom_loss(pred, target, valid, mean, std):
    """Auxiliary eye-geometry loss. pred (B,T,G) is STANDARDISED; target (B,T,G) in image units, standardised here with
    (mean, std). Smooth-L1 averaged over the VALID entries of each dim, summed over dims (so ~G at chance, comparable to nats);
    entries without a label (video frames, closed eyes for pupil dims) contribute nothing. Returns (loss, per-dim mean error
    in image units over valid entries, NaN where none)."""
    t = (target - mean) / std
    err = F.smooth_l1_loss(pred, t, reduction="none") * valid
    n = valid.sum((0, 1)).clamp(min=1)
    loss = (err.sum((0, 1)) / n).sum()
    abs_err = ((pred * std + mean - target).abs() * valid).sum((0, 1)) / n
    return loss, torch.where(valid.sum((0, 1)) > 0, abs_err, torch.full_like(abs_err, float("nan")))


def overshoot_kl(rssm, outputs, actions, depth, free_nats=1.0, from_start=True):
    """Latent overshooting (PlaNet): train the OPEN-LOOP prior against the posterior, in latent space (no pixel loss, so no blur).

    Every start state rolls `depth` steps with the prior only (z sampled, real actions); at step d the d-step prior is pulled toward the
    posterior of the frame that really is d steps later (KL(posterior || prior), posterior detached). Start states and targets are
    detached, so only the dynamics (GRU, prior) learn: the posterior stays as free as kl_balance makes it. d = 1 is already the normal
    kl_dyn, so d = 2..depth. `from_start` adds the zero state (what a dream starts from: no observation yet) as one more start, which trains
    the blind first steps with prior-SAMPLED z as the GRU input (the normal loss only ever feeds it posterior z).
    outputs = RSSM.observe outputs (h, z (B,T,.), posterior list of (mean, std)); actions (B,T,A) with actions[:, t] entering step t.
    Returns (loss clamped at free_nats, raw mean KL per step in nats); zeros when the clip is too short (needs depth >= 2, T >= 3)."""
    h, z = outputs["h"].detach(), outputs["z"].detach()
    b, t = h.shape[:2]
    depth = min(depth, t - 1)
    zero = torch.zeros((), device=h.device)
    if depth < 2:
        return zero, zero
    pm = torch.stack([p[0] for p in outputs["posterior"]], 1).detach()
    ps = torch.stack([p[1] for p in outputs["posterior"]], 1).detach()
    lag = 1 if from_start else 0              # start k is the state at time k-lag (time -1 = the zero state)
    if from_start:
        h0, z0 = rssm.initial(b, h.device)
        h, z = torch.cat([h0[:, None], h[:, :-1]], 1), torch.cat([z0[:, None], z[:, :-1]], 1)
    kls = []
    for d in range(1, depth + 1):
        n, s = t - d + lag, d - lag           # n starts still have a target; their d-th step lands on times s..t-1
        h, z = h[:, :n], z[:, :n]
        h, z, (mean, std) = rssm.imagine_step(h.reshape(b * n, -1), z.reshape(b * n, -1), actions[:, s:s + n].reshape(b * n, -1))
        h, z = h.view(b, n, -1), z.view(b, n, -1)
        if d >= 2:
            kls.append(gaussian_kl((pm[:, s:], ps[:, s:]), (mean.view(b, n, -1), std.view(b, n, -1))).mean())
    raw = torch.stack(kls).mean()
    return raw.clamp(min=free_nats), raw


def world_model_loss(images, outputs, kl_weight=1.0, free_nats=1.0, balance=0.8,
                     pred_frames=None, pred_targets=None, pred_weight=1.0, recon_kind="smoothl1", edge_weight=0.0):
    """recon (pixel-summed) + balanced, free-bits KL + optional open-loop prediction loss.

    Balancing: the prior is trained toward a detached posterior (weight `balance`) and the
    posterior toward a detached prior (weight 1-balance), so the prior learns to predict
    without the posterior collapsing onto it. Free nats stop the KL from being squeezed to 0.
    """
    reconstruction = _pixel_sum_loss(outputs["recon"], images, recon_kind)
    if edge_weight > 0:
        reconstruction = reconstruction + edge_weight * edge_loss(outputs["recon"], images)
    prior, post = outputs["prior"], outputs["posterior"]
    detach = lambda d: (d[0].detach(), d[1].detach())
    kl_dyn = torch.stack([gaussian_kl(detach(post[t]), prior[t]) for t in range(len(prior))], 1).mean()
    kl_rep = torch.stack([gaussian_kl(post[t], detach(prior[t])) for t in range(len(prior))], 1).mean()
    kl_loss = balance * kl_dyn.clamp(min=free_nats) + (1 - balance) * kl_rep.clamp(min=free_nats)
    total = reconstruction + kl_weight * kl_loss
    prediction = torch.zeros((), device=images.device)
    if pred_frames is not None:
        prediction = _pixel_sum_loss(pred_frames, pred_targets, recon_kind)
        if edge_weight > 0:
            prediction = prediction + edge_weight * edge_loss(pred_frames, pred_targets)
        total = total + pred_weight * prediction
    npix = images[0, 0].numel()
    metrics = {
        "total": total.detach(), "reconstruction": reconstruction.detach(),
        "kl": kl_dyn.detach(), "prediction": prediction.detach(),
        "error": ((reconstruction + prediction) / npix).detach(),   # per-pixel, for the controller
    }
    return total, metrics


class PlasticityController:
    """Heuristic per-module learning-rate controller (still NOT a learned meta-controller).

    One scale per module (encoder / rssm / decoder). Inputs: smoothed *relative* loss progress
    and a measured retention flag (probe-batch error vs its own best). Rules:
      retention failed  -> shrink all scales (encoder hardest: protect perception)
      improving         -> grow slowly
      worsening         -> shrink slightly
      stalled           -> grow very slowly ('plasticity recovery' so units can re-adapt)
    Scales are clamped and logged, so every change is inspectable.
    """
    def __init__(self, base_lr=2e-4, groups=("encoder", "rssm", "decoder"),
                 group_bias=None, min_scale=0.1, max_scale=3.0, momentum=0.95, tol=1e-4):
        self.base_lr = base_lr
        self.bias = group_bias or {"encoder": 0.5, "rssm": 1.0, "decoder": 1.0}
        self.scales = {g: 1.0 for g in groups}
        self.min_scale, self.max_scale, self.momentum, self.tol = min_scale, max_scale, momentum, tol
        self.ema_error = None
        self.ema_progress = 0.0
        self.steps = 0

    def update(self, error: float, retention_ok: bool = True):
        if self.ema_error is None:
            self.ema_error = error
        prev = self.ema_error
        self.ema_error = self.momentum * prev + (1 - self.momentum) * error
        rel = (prev - self.ema_error) / max(abs(prev), 1e-8)
        self.ema_progress = self.momentum * self.ema_progress + (1 - self.momentum) * rel
        for g in self.scales:
            if not retention_ok:
                f = 0.6 if g == "encoder" else 0.8
            elif self.ema_progress > self.tol:
                f = 1.02
            elif self.ema_progress < -self.tol:
                f = 0.97
            else:
                f = 1.005
            self.scales[g] = min(self.max_scale, max(self.min_scale, self.scales[g] * f))
        self.steps += 1
        return {g: self.base_lr * self.bias.get(g, 1.0) * s for g, s in self.scales.items()}

    def state_dict(self):
        return {"scales": dict(self.scales), "ema_error": self.ema_error,
                "ema_progress": self.ema_progress, "steps": self.steps}
