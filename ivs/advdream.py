"""R5 stage 2 (plan.md "R5"): adversarial fine-tuning of the RSSM on its OWN dream loop.
The dream is what the UI shows (ivs.ui.World.tick with dream=True): every tick the model PREDICTS (prior), RENDERS the prediction, PERCEIVES that render (posterior of the
encoded render) and shows the render. `ui_rollout` is that loop, differentiable (no 8-bit quantisation), so a conditional PatchGAN (ivs.refiner.PatchDisc, cond = the last
real context frame, candidate = a dream frame vs the real continuation frame at the same tick) can push the dream toward "a real continuation of this eye".
Only rssm.gru, rssm.prior and the decoder learn; the encoder and the posterior stay frozen (the perception the dream loop re-uses cannot move). Plain torch, CPU-capable.
"""
import torch

from .refiner import PatchDisc, d_hinge, g_hinge          # noqa: F401  (re-exported for the training script)

TRAINABLE = ("rssm.gru", "rssm.prior", "decoder")
FROZEN = ("encoder", "rssm.posterior")


def freeze_for_stage2(model):
    """requires_grad only for TRAINABLE modules; returns the list of trainable parameters. Raises when a parameter belongs to neither list (a new module must be classified)."""
    params = []
    for name, p in model.named_parameters():
        learn = name.startswith(TRAINABLE)
        if not learn and not name.startswith(FROZEN):
            raise ValueError(f"parameter {name} is neither in TRAINABLE {TRAINABLE} nor in FROZEN {FROZEN}")
        p.requires_grad_(learn)
        if learn:
            params.append(p)
    return params


def ui_rollout(model, h, z, actions):
    """The UI dream loop from the latent state (h, z), one step per row of `actions` (B,K,A): predict -> render -> perceive the render with the same action (ivs.ui.World).
    Returns the renders (B,K,C,H,W) in [0,1] (differentiable) and the final (h, z)."""
    frames = []
    for k in range(actions.shape[1]):
        a = actions[:, k]
        ph, pz, _ = model.rssm.imagine_step(h, z, a)
        pred = model.decode(ph, pz)
        h, z, _, _ = model.rssm.observe_step(h, z, a, model.encoder(pred))
        frames.append(pred)
    return torch.stack(frames, 1), (h, z)


def dream_from_clean_context(model, clean, actions, ctx, k):
    """Observe the first `ctx` clean frames (no gradient: the posterior path is frozen, and the context only provides the start state), then run the UI loop for the next `k`
    ticks with the clip's own actions. Returns (fake (B,k,C,H,W), real (B,k,C,H,W) = the real continuation under the same interventions, cond (B,C,H,W) = last context frame)."""
    with torch.no_grad():
        out = model.observe(clean[:, :ctx], actions[:, :ctx])
        h, z = out["h"][:, -1], out["z"][:, -1]
    fake, _ = ui_rollout(model, h, z, actions[:, ctx:ctx + k])
    return fake, clean[:, ctx:ctx + k], clean[:, ctx - 1]


def disc_logits(d, cond, frames, ticks):
    """PatchDisc logits for frames (B,K,C,H,W) at the chosen tick indices, all conditioned on `cond` (B,C,H,W): (B*len(ticks), 1, h, w)."""
    b = frames.shape[0]
    sel = frames[:, ticks]
    n = sel.shape[1]
    c = cond[:, None].expand(-1, n, -1, -1, -1).reshape(b * n, *cond.shape[1:])
    return d(c, sel.reshape(b * n, *sel.shape[2:]))


def adaptive_weight(l_real, l_adv, layer_weight, scale=0.3, max_weight=1e3):
    """VQGAN-style balance: scale * |grad of the real loss| / |grad of the adversarial loss| w.r.t. one reference weight (the decoder's last layer), both measured now, so the
    adversarial term always pushes with a fixed fraction of the reconstruction force, whatever the loss units are. Returns a detached scalar (0 when the adversarial gradient is exactly 0)."""
    g_real = torch.autograd.grad(l_real, layer_weight, retain_graph=True)[0]
    g_adv = torch.autograd.grad(l_adv, layer_weight, retain_graph=True, allow_unused=True)[0]
    if g_adv is None:
        return l_real.new_zeros(())
    return (scale * g_real.norm() / (g_adv.norm() + 1e-4)).clamp(0.0, max_weight).detach()
