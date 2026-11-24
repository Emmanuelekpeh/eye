"""Pilot trainer + evaluation suite on the video cache (CPU-aware, GPU-fast). One identical data/val/seed/step budget
per run so runs differ ONLY in the idea being tested (see PRESETS).

  python -m ivs.pilot --cache ../data/cache/videos --out pilot_out --runs R1_rgb_base --steps 4000 --device auto

Ideas under test: corrected pipeline (R0/R1), colour cost (R0 vs R1), staged rollout lengths 8->16->32 (R2),
scatter-denoising (R3), decode->re-encode consistency (R4), old plasticity controller vs plain cosine LR (R5),
real-colour objective gray-input -> colour-target (R6).

Every run writes metrics.json, train_log.jsonl, model.pt (plain tensors/primitives only) and PNG sheets that must be
LOOKED AT: numbers cannot tell you whether a frame is an eye.
"""
import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .actions import InterventionSampler, luma, scatter_batch
from .cachedata import CacheBank, source_kinds, train_only_sources
from .config import ModelConfig
from .gaze import GEOM_DIM, GEOM_NAMES, POS_PAIRS
from .learning import PlasticityController, geom_loss, overshoot_kl, world_model_loss
from .model import EyeWorldModel

SEED_STEPS = 4            # zero-action prior steps that seed a noise canvas (same value as ivs/ui.py)
HORIZONS = (1, 2, 4, 8, 16, 32, 64, 128)


@dataclass
class RunConfig:
    name: str = "R1_rgb_base"
    channels: int = 3
    stages: tuple = ((8, 1.0),)            # (sequence length, fraction of total steps)
    lr: float = 3e-4
    lr_mode: str = "cosine"                # cosine | plasticity (the old controller + retention probe)
    batch: int = 32
    p_intervene: float = 0.5
    p_scatter: float = 0.0                 # fraction of clips whose INPUT is scatter-corrupted (target stays clean)
    p_desat: float = 0.0                   # fraction of colour-class clips shown gray (target stays colour)
    cons_weight: float = 0.0               # decode->re-encode embedding agreement
    kl_weight: float = 1.0
    free_nats: float = 1.0
    kl_balance: float = 0.8
    pred_weight: float = 1.0
    grad_clip: float = 100.0
    warmup: int = 200
    lr_floor: float = 0.1                  # cosine ends at lr * lr_floor (1.0 = constant LR after warmup)
    train_seed: int = 0                    # added to --seed for the training batches / noise only (val split and eval stay the same)
    recon_kind: str = "smoothl1"           # smoothl1 | l1
    edge_weight: float = 0.0               # gradient-matching term added to recon/prediction losses
    feature_dim: int = 128                 # model size knobs (defaults = original model)
    deter_dim: int = 128
    stoch_dim: int = 32
    hidden_dim: int = 128
    conv_mult: float = 1.0
    train_group_frac: float = 1.0          # data-scaling: train on this fraction of SUBJECTS (val set is unchanged)
    train_sources: tuple = None            # temporal sources this run trains on (None = all temporal sources in the caches)
    image_weight: float = 0.0              # >0: also train encoder/decoder on single images (appearance sources), T=1
    image_batch: int = 64
    # Real eye labels from the S_ masks (need --labels; see ivs/gaze.py). gaze_command: the action grows 5 -> 8 dims
    # [touch x5, has_cmd, gaze_x, gaze_y] with the cleaned absolute pupil target; cmd_dropout zeroes the command per clip so that
    # a zero command still means 'dream on your own' (video clips never have one). geom_weight / geom_prior_weight: auxiliary
    # geometry head on the posterior state / on the PRIOR state (what the state knew before seeing the frame: forces the command to matter).
    gaze_command: bool = False
    cmd_dropout: float = 0.3
    geom_weight: float = 0.0
    geom_prior_weight: float = 0.0
    # Latent overshooting (learning.overshoot_kl): open-loop prior trained against the posterior up to `overshoot_depth` steps ahead, in latent
    # space only. overshoot_from_start also trains a dream's own start (zero state, prior-sampled z). Weight is relative to kl_weight.
    overshoot_weight: float = 0.0
    overshoot_depth: int = 31
    overshoot_from_start: bool = True
    # RSSM.prior_stopgrad (overfit_rssm.py G_stopgrad): the prior loss trains the prior head only, no gradient into the GRU state / encoder
    prior_stopgrad: bool = False


STAGED = ((8, 0.4), (16, 0.35), (32, 0.25))
PRESETS = {
    "R0_gray_base": dict(channels=1),
    "R1_rgb_base": dict(),
    "R2_rgb_long": dict(stages=STAGED),
    "R3_scatter": dict(stages=STAGED, p_scatter=0.5),
    "R4_consistency": dict(stages=STAGED, p_scatter=0.5, cons_weight=1.0),
    "R5_plasticity": dict(stages=STAGED, p_scatter=0.5, cons_weight=1.0, lr_mode="plasticity"),
    "R6_colourise": dict(stages=STAGED, p_scatter=0.5, cons_weight=1.0, p_desat=0.3),
    # Ablations vs R2_rgb_long (staged, no augmentation): is the blur loss- or capacity-limited?
    "A1_kl01": dict(stages=STAGED, kl_weight=0.1),
    "A2_kl001": dict(stages=STAGED, kl_weight=0.01),
    "A3_l1": dict(stages=STAGED, recon_kind="l1"),
    "A4_edge": dict(stages=STAGED, edge_weight=1.0),
    "A5_wide": dict(stages=STAGED, feature_dim=512, deter_dim=256, stoch_dim=64, hidden_dim=256, conv_mult=2.0),
    "A6_wide_kl01_edge": dict(stages=STAGED, feature_dim=512, deter_dim=256, stoch_dim=64, hidden_dim=256, conv_mult=2.0,
                              kl_weight=0.1, edge_weight=1.0),
    # Fast feature tests: fixed T=8 (~8 steps/s on a T4, so 8000 steps ~ 16 min), plain base config, one change each.
    "F0_base": dict(),
    "F1_lr1e3": dict(lr=1e-3),
    "F2_kl001": dict(kl_weight=0.01),
    "F3_embed512": dict(feature_dim=512, hidden_dim=256),
    "F4_half_subjects": dict(train_group_frac=0.5),
    "F5_quarter_subjects": dict(train_group_frac=0.25),
    # Latent-width tests (AE probe: a plain AE with the same conv stacks reaches PCA level, and its val error falls with code size
    # .0060/.0026/.0016 for 32/128/512; in the RSSM the current frame reaches the decoder only through the 32-d sampled z).
    "S1_stoch128": dict(stoch_dim=128),
    "S2_stoch256": dict(stoch_dim=256),
    "S3_stoch128_kl001": dict(stoch_dim=128, kl_weight=0.01),
    "S4_stoch128_embed512_lr1e3": dict(stoch_dim=128, feature_dim=512, hidden_dim=256, lr=1e-3),
    "S5_stoch256_embed512_lr1e3_kl001": dict(stoch_dim=256, feature_dim=512, hidden_dim=256, lr=1e-3, kl_weight=0.01),
    "S6_stoch128_deter256": dict(stoch_dim=128, deter_dim=256),
    # What does the RSSM add on top of a plain VAE? (probe: VAE32 val .0074/.0063/.0061 at KL weight 1/.1/.01 and VAE128_kl001 .0042,
    # world model F0 .0090, F2(kl .01) .0086 on the same split): drop the open-loop prediction loss and/or the random interventions.
    "W1_kl001_nopred": dict(kl_weight=0.01, pred_weight=0.0),
    "W2_kl001_nointv": dict(kl_weight=0.01, p_intervene=0.0),
    "W3_kl001_recon_only": dict(kl_weight=0.01, pred_weight=0.0, p_intervene=0.0),
    "W4_kl001_recon_only_stoch128": dict(kl_weight=0.01, pred_weight=0.0, p_intervene=0.0, stoch_dim=128),
    "W5_kl01_recon_only": dict(kl_weight=0.1, pred_weight=0.0, p_intervene=0.0),
    "W6_kl1_recon_only": dict(kl_weight=1.0, pred_weight=0.0, p_intervene=0.0),
    # Candidate defaults from the W round (val recon / open-loop pred: F0 .0090/.0115, W1 .0064/.0138, W2 .0110/.0128, W3 .0072/.0135,
    # W6 .0075/.0108): no pixel prediction loss (it blurs the shared decoder; the KL-dyn term already trains the prior), interventions
    # ON (augmentation: removing them hurt). The posterior is penalised ONLY by the rep term (weight (1-kl_balance)*kl_weight; the dyn
    # term detaches the posterior and just trains the prior), so kl_balance .99 frees the posterior's information (probe: VAE val .0074
    # at KL 1 -> .0061 / .0042 at .01 for 32 / 128 codes, width only helps once the penalty is low) while the prior keeps learning
    # at full strength (what gave W6 its best open-loop pred). V1 = control with the default balance .8.
    "V1_base": dict(pred_weight=0.0, train_sources=("video",)),
    "V2_bal99": dict(pred_weight=0.0, train_sources=("video",), kl_balance=0.99),
    "V3_bal99_wide": dict(pred_weight=0.0, train_sources=("video",), kl_balance=0.99, stoch_dim=128, deter_dim=256, feature_dim=512,
                          hidden_dim=256),
    "V4_bal99_wide_lr1e3": dict(pred_weight=0.0, train_sources=("video",), kl_balance=0.99, stoch_dim=128, deter_dim=256,
                                feature_dim=512, hidden_dim=256, lr=1e-3),
    "V5_bal99_wide_pred01": dict(pred_weight=0.1, train_sources=("video",), kl_balance=0.99, stoch_dim=128, deter_dim=256,
                                 feature_dim=512, hidden_dim=256),
    "V6_bal99_wide_lr1e3_openeds": dict(pred_weight=0.0, train_sources=("video", "openeds"), kl_balance=0.99, stoch_dim=128,
                                        deter_dim=256, feature_dim=512, hidden_dim=256, lr=1e-3),
    # Long staged runs on the best config so far (V6) plus what the project needs: scatter denoising (self-organisation), the gray<->colour
    # objective (p_desat) and low-weight MRL appearance images. L2 differs from L1 ONLY by a small open-loop pixel prediction loss
    # (V5: neutral for recon, better intervened pred; a staged T=32 run may need it to avoid free-run drift).
    "L1_v6_long": dict(pred_weight=0.0, train_sources=("video", "openeds"), kl_balance=0.99, stoch_dim=128, deter_dim=256,
                       feature_dim=512, hidden_dim=256, lr=1e-3, stages=STAGED, p_scatter=0.5, p_desat=0.3, image_weight=0.3),
    "L2_v6_long_pred01": dict(pred_weight=0.1, train_sources=("video", "openeds"), kl_balance=0.99, stoch_dim=128, deter_dim=256,
                              feature_dim=512, hidden_dim=256, lr=1e-3, stages=STAGED, p_scatter=0.5, p_desat=0.3, image_weight=0.3),
    # B (auxiliary geometry head) and C (real gaze command) on the V6 base (same seed/steps/val as V6; G0 re-runs V6 under the new code
    # as the control). Needs --labels. G4 additionally puts the geometry head on the PRIOR state.
    "G0_v6_control": dict(pred_weight=0.0, train_sources=("video", "openeds"), kl_balance=0.99, stoch_dim=128, deter_dim=256,
                          feature_dim=512, hidden_dim=256, lr=1e-3),
    "G1_geom": dict(pred_weight=0.0, train_sources=("video", "openeds"), kl_balance=0.99, stoch_dim=128, deter_dim=256,
                    feature_dim=512, hidden_dim=256, lr=1e-3, geom_weight=10.0),
    "G2_cmd": dict(pred_weight=0.0, train_sources=("video", "openeds"), kl_balance=0.99, stoch_dim=128, deter_dim=256,
                   feature_dim=512, hidden_dim=256, lr=1e-3, gaze_command=True),
    "G3_cmd_geom": dict(pred_weight=0.0, train_sources=("video", "openeds"), kl_balance=0.99, stoch_dim=128, deter_dim=256,
                        feature_dim=512, hidden_dim=256, lr=1e-3, gaze_command=True, geom_weight=10.0),
    "G4_cmd_geom_prior": dict(pred_weight=0.0, train_sources=("video", "openeds"), kl_balance=0.99, stoch_dim=128, deter_dim=256,
                              feature_dim=512, hidden_dim=256, lr=1e-3, gaze_command=True, geom_weight=10.0, geom_prior_weight=10.0),
    # The long staged run with the gaze command: L1's recipe (staged T=8/16/32, scatter, desat, appearance images) + G4's command and
    # geometry heads on posterior and prior state. Chosen after eye-os1 showed overshooting hurts (backlog P0e).
    "G5_long_cmd_geom_prior": dict(pred_weight=0.0, train_sources=("video", "openeds"), kl_balance=0.99, stoch_dim=128, deter_dim=256,
                                   feature_dim=512, hidden_dim=256, lr=1e-3, stages=STAGED, p_scatter=0.5, p_desat=0.3, image_weight=0.3,
                                   gaze_command=True, geom_weight=10.0, geom_prior_weight=10.0),
    # Latent overshooting on the L1 base (the 30k eye-long3 run: dreams drift from a real context within ~30 steps and lose the eye at
    # once from a noise start; backlog P0e / handoff LATEST 8). O0 = control (L1), O1 = overshooting from the posterior states only,
    # O2 = O1 + the zero start of a dream.
    "O0_l1_control": dict(pred_weight=0.0, train_sources=("video", "openeds"), kl_balance=0.99, stoch_dim=128, deter_dim=256,
                          feature_dim=512, hidden_dim=256, lr=1e-3, stages=STAGED, p_scatter=0.5, p_desat=0.3, image_weight=0.3),
    "O1_overshoot": dict(pred_weight=0.0, train_sources=("video", "openeds"), kl_balance=0.99, stoch_dim=128, deter_dim=256,
                         feature_dim=512, hidden_dim=256, lr=1e-3, stages=STAGED, p_scatter=0.5, p_desat=0.3, image_weight=0.3,
                         overshoot_weight=1.0, overshoot_from_start=False),
    "O2_overshoot_start": dict(pred_weight=0.0, train_sources=("video", "openeds"), kl_balance=0.99, stoch_dim=128, deter_dim=256,
                               feature_dim=512, hidden_dim=256, lr=1e-3, stages=STAGED, p_scatter=0.5, p_desat=0.3, image_weight=0.3,
                               overshoot_weight=1.0, overshoot_from_start=True),
    # Data-mix tests (same T=8 base as F0; the data comes from --cache/--source-weights; every run is scored on the SAME
    # per-source val sets): temporal sources 'video' and 'openeds', appearance source 'mrl' (single images, T=1 objective).
    "D0_video": dict(train_sources=("video",)),
    "D1_video_openeds": dict(train_sources=("video", "openeds")),
    "D2_video_openeds_mrl03": dict(train_sources=("video", "openeds"), image_weight=0.3),
    "D3_video_openeds_mrl10": dict(train_sources=("video", "openeds"), image_weight=1.0),
}

# LR continuation test (backlog P0e): continue the trained L1 (30k steps, cosine 1e-3 -> 1e-4, recon flat over the last third while the LR sat at
# its floor) for N steps at T=32 with --init-from L1's model.pt, one LR schedule per arm, same data and seed so the batches are paired. Fresh Adam
# state in every arm (the checkpoint has none), so arms are compared with each other, not with L1's last log row. Question: is the plateau
# LR/optimisation-limited (a lower floor or a re-heat helps) or steps/capacity-limited (nothing helps beyond noise)?
# LR0 = continue at the old floor (control), LR0b = same with another batch order (training noise), LR1 = lower constant, LR2/LR3 = warm restart.
PRESETS.update({
    "LR0_const1e4": dict(PRESETS["L1_v6_long"], stages=((32, 1.0),), lr=1e-4, lr_floor=1.0),
    "LR0b_const1e4_seed1": dict(PRESETS["L1_v6_long"], stages=((32, 1.0),), lr=1e-4, lr_floor=1.0, train_seed=1),
    "LR1_const3e5": dict(PRESETS["L1_v6_long"], stages=((32, 1.0),), lr=3e-5, lr_floor=1.0),
    "LR2_cos3e4_to3e5": dict(PRESETS["L1_v6_long"], stages=((32, 1.0),), lr=3e-4, lr_floor=0.1),
    "LR3_cos1e3_to1e4": dict(PRESETS["L1_v6_long"], stages=((32, 1.0),), lr=1e-3, lr_floor=0.1),
})

# CPU-sized paired fine-tune of L1 (--init-from L1's model.pt, plan.md "RSSM OVERFIT"): L1's recipe with T=8 windows only (L1's staged T up to 32 is too slow on CPU),
# run with --batch 8. FT0 = control (L1's loss), FT0_s1 = control with another batch order (the noise yardstick), FT1 = the same with the prior loss stopped at the GRU state.
PRESETS.update({
    "P0_ft_control": dict(PRESETS["L1_v6_long"], stages=((8, 1.0),)),
    "P0_ft_control_s1": dict(PRESETS["L1_v6_long"], stages=((8, 1.0),), train_seed=1),
    "P1_ft_stopgrad": dict(PRESETS["L1_v6_long"], stages=((8, 1.0),), prior_stopgrad=True),
})

# From-scratch run of a ONE-GPU-HOUR budget (plan.md "SCRATCH"): L1's recipe and architecture, random init, staged T=8 -> 16 only (L1's T=32 stage costs 2/3 of its 2.3 h; the T=32 windows
# come in the adversarial stage that follows). train_sources=None = every temporal source in --cache, so the SAME preset is the control (caches without the dream pool) and the dream
# arm (the dream cache attached, weights from --source-weights).
PRESETS["N0_scratch"] = dict(PRESETS["L1_v6_long"], train_sources=None, stages=((8, 0.75), (16, 0.25)))


# ----------------------------------------------------------------------------------------------- helpers
def to_channels(x, c):
    """x: B,T,3,H,W (RGB) -> model channels (1 = luma)."""
    return x if c == 3 else luma(x)[:, :, :1]


def mse(a, b):
    return float(((a - b) ** 2).mean())


def chroma(x):
    """Mean |channel difference| over pixels, (…,3,H,W) -> float."""
    r, g, b = x[..., 0, :, :], x[..., 1, :, :], x[..., 2, :, :]
    return float(((r - g).abs() + (g - b).abs() + (r - b).abs()).mean() / 3.0)


_LAP = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3)


def sharpness(x):
    """Variance of the Laplacian of the gray image, x: N,C,H,W. Blurry output -> low value."""
    g = x.mean(1, keepdim=True)
    return float(F.conv2d(g, _LAP.to(x.device)).var())


def small_gray(x, s=24):
    return F.interpolate(x.mean(1, keepdim=True), size=(s, s), mode="area").flatten(1)


def nn_distance(a, b, exclude_self=False):
    """Median over rows of a of the L1 distance (per pixel) to the nearest row of b."""
    d = torch.cdist(a, b, p=1) / a.shape[1]
    if exclude_self:
        d.fill_diagonal_(float("inf"))
    return float(d.min(1).values.median())


def save_sheet(rows, path):
    """rows: list of rows, each a list of (C,H,W) tensors in [0,1]."""
    out = []
    for row in rows:
        tiles = []
        for t in row:
            t = t.detach().float().clamp(0, 1).cpu()
            if t.shape[0] == 1:
                t = t.expand(3, -1, -1)
            tiles.append((t.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))
        out.append(np.concatenate(tiles, 1))
    cv2.imwrite(str(path), cv2.cvtColor(np.concatenate(out, 0), cv2.COLOR_RGB2BGR))


# ----------------------------------------------------------------------------------------------- training
def uses_labels(rc):
    return rc.gaze_command or rc.geom_weight > 0 or rc.geom_prior_weight > 0


def action_dim(rc):
    return 8 if rc.gaze_command else 5           # [touch x5] (+ [has_cmd, gaze_x, gaze_y])


def pad_actions(a, model):
    """Zero-pad touch-only actions (B,T,5) to the model's action size (extra dims = 'no gaze command')."""
    extra = model.cfg.action_dim - a.shape[-1]
    return a if extra <= 0 else torch.cat([a, torch.zeros(*a.shape[:-1], extra, device=a.device)], -1)


def make_batch(bank, rc, gen, aug_gen, sampler, t, device, with_aux=False):
    """Returns (inp, clean, actions); with_aux=True also the label dict {'geom','valid'} (None when the run uses no labels).
    Labels follow the pixels: the bank applied crop/flip, here the synthetic shift moves the pupil/iris positions too
    (position += 2*shift in [-1,1] units); the gaze command is the eye's own gaze and is NOT shifted by the touch."""
    b = bank.sample(rc.batch, t, generator=gen, train=True)
    clean = to_channels(b["images"], rc.channels)
    aux = None
    if uses_labels(rc):
        clean, actions, states = sampler(clean, return_states=True)    # interventions change the world: target moves too
        geom = b["geom"].to(device).clone()
        for xi, yi in POS_PAIRS:
            geom[..., xi] += 2.0 * states[..., 0]
            geom[..., yi] += 2.0 * states[..., 1]
        aux = dict(geom=geom, valid=b["geom_valid"].to(device))
        if rc.gaze_command:
            drop = (torch.rand(rc.batch, generator=gen) < rc.cmd_dropout).to(device).view(-1, 1, 1)
            keep = b["has_cmd"].to(device).unsqueeze(-1) & ~drop                              # B,T,1
            cmd = b["cmd"].to(device)
            actions = torch.cat([actions.to(device), keep.float(), torch.where(keep, cmd, torch.zeros_like(cmd))], -1)
    else:
        clean, actions = sampler(clean)
    actions = actions.to(device)
    inp = clean
    if rc.p_desat > 0 and rc.channels == 3:
        m = (b["is_colour"] & (torch.rand(rc.batch, generator=gen) < rc.p_desat).to(device)).view(-1, 1, 1, 1, 1)
        inp = torch.where(m, luma(clean), inp)
    if rc.p_scatter > 0:
        n, t_, c, h, w = inp.shape
        corrupt = scatter_batch(inp.reshape(n * t_, c, h, w), aug_gen).view(n, t_, c, h, w)
        m = (torch.rand(n, generator=gen) < rc.p_scatter).to(device).view(-1, 1, 1, 1, 1)
        inp = torch.where(m, corrupt, inp)
    return (inp, clean, actions, aux) if with_aux else (inp, clean, actions)


def train_loss(model, rc, inp, clean, actions, aux=None):
    out = model.observe(inp, actions)
    ctx = clean.shape[1] // 2
    pred = model.rollout(out["h"][:, ctx - 1], out["z"][:, ctx - 1], actions[:, ctx:])
    total, metrics = world_model_loss(clean, out, rc.kl_weight, rc.free_nats, rc.kl_balance,
                                      pred_frames=pred, pred_targets=clean[:, ctx:], pred_weight=rc.pred_weight,
                                      recon_kind=rc.recon_kind, edge_weight=rc.edge_weight)
    if aux is not None and model.cfg.geom_dim > 0:
        for key, w in (("geom", rc.geom_weight), ("geom_prior", rc.geom_prior_weight)):
            if w > 0:
                lg, _ = geom_loss(out[key], aux["geom"], aux["valid"], model.geom_mean, model.geom_std)
                total = total + w * lg
                metrics[key] = lg.detach()
    if rc.overshoot_weight > 0:
        os_loss, os_raw = overshoot_kl(model.rssm, out, actions, rc.overshoot_depth, rc.free_nats, rc.overshoot_from_start)
        total = total + rc.kl_weight * rc.overshoot_weight * os_loss
        metrics["overshoot"] = os_raw.detach()
    cons = torch.zeros((), device=clean.device)
    if rc.cons_weight > 0:
        b, t, c, h, w = clean.shape
        e_r = model.encoder(out["recon"].reshape(b * t, c, h, w))
        with torch.no_grad():
            e_t = model.encoder(clean.reshape(b * t, c, h, w))
        cons = ((e_r - e_t) ** 2).sum(-1).mean()
        total = total + rc.cons_weight * cons
    metrics = {k: float(v) for k, v in metrics.items()}
    metrics["consistency"] = float(cons)
    return total, metrics


def image_loss(model, rc, bank, gen, device):
    """Appearance objective: a batch of real single images through the same encoder/RSSM/decoder as ONE step from the
    initial state with zero action (no dynamics to learn; this trains how images are encoded and drawn)."""
    b = bank.sample(rc.image_batch, 1, generator=gen, train=True)
    x = to_channels(b["images"], rc.channels)
    out = model.observe(x, torch.zeros(x.shape[0], 1, model.cfg.action_dim, device=device))
    total, metrics = world_model_loss(x, out, rc.kl_weight, rc.free_nats, rc.kl_balance,
                                      recon_kind=rc.recon_kind, edge_weight=rc.edge_weight)
    return total, float(metrics["reconstruction"])


def lr_at(rc, step, total):
    warm = min(1.0, (step + 1) / max(1, rc.warmup))
    cos = rc.lr_floor + (1.0 - rc.lr_floor) * 0.5 * (1 + math.cos(math.pi * step / max(1, total)))
    return rc.lr * warm * cos


def load_init(model, cfg, path):
    """Start from a trained checkpoint (continuation runs). The architecture must match exactly; the optimiser starts fresh (checkpoints
    store no Adam state), which is why continuation runs keep the LR warmup."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    want, have = asdict(cfg), asdict(ModelConfig(**ck["config"]))        # fields added after the checkpoint was written take their defaults
    diff = {k: (have[k], want[k]) for k in want if have[k] != want[k]}
    if diff:
        raise ValueError(f"--init-from {path}: model config differs from this run's (checkpoint, run): {diff}")
    model.load_state_dict(ck["model"])


def train_run(rc, train_bank, device, total_steps, seed, log_path, log_every=100, mean_probe=None, ckpt_every=0,
              image_bank=None, init_from=None):
    if rc.image_weight > 0 and image_bank is None:
        raise ValueError(f"{rc.name}: image_weight>0 needs an appearance source (a cache with kind='appearance')")
    if uses_labels(rc):
        if not getattr(train_bank, "has_labels", False) or not bool(train_bank.has_cmd.any()):
            raise ValueError(f"{rc.name}: gaze_command / geom_* need the mask cache (--labels) and 'openeds' among the train sources "
                             f"(this train bank has {'no labels' if not getattr(train_bank, 'has_labels', False) else 'no labelled frames'})")
    seed = seed + rc.train_seed
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    cfg = ModelConfig(image_size=train_bank.size, channels=rc.channels, feature_dim=rc.feature_dim, deter_dim=rc.deter_dim,
                      stoch_dim=rc.stoch_dim, hidden_dim=rc.hidden_dim, conv_mult=rc.conv_mult, action_dim=action_dim(rc),
                      geom_dim=GEOM_DIM if (rc.geom_weight > 0 or rc.geom_prior_weight > 0) else 0)
    model = EyeWorldModel(cfg).to(device)
    model.rssm.prior_stopgrad = bool(rc.prior_stopgrad)
    if cfg.geom_dim > 0:
        mean, std = train_bank.label_stats()
        model.geom_mean.copy_(mean.to(device)); model.geom_std.copy_(std.to(device))
    if init_from:
        load_init(model, cfg, init_from)
    groups = {"encoder": model.encoder, "rssm": model.rssm, "decoder": model.decoder}
    opt = torch.optim.AdamW([{"params": list(m.parameters()), "name": n} for n, m in groups.items()], lr=rc.lr)
    plast = PlasticityController(base_lr=rc.lr) if rc.lr_mode == "plasticity" else None
    gen = torch.Generator().manual_seed(seed)
    aug_gen = torch.Generator(device=device).manual_seed(seed + 1)
    sampler = InterventionSampler(p=rc.p_intervene)
    schedule = []
    for t, frac in rc.stages:
        schedule += [t] * int(round(frac * total_steps))
    schedule = (schedule + [rc.stages[-1][0]] * total_steps)[:total_steps]
    probe, probe_ema, probe_best, retention_ok = None, None, None, True
    log = open(log_path, "w")
    t0, hist = time.time(), []
    acc, acc_n = {}, 0                       # per-log-interval means (a single batch's recon swings +-8% at T=32) + gradient-norm statistics
    for step, t in enumerate(schedule):
        inp, clean, actions, aux = make_batch(train_bank, rc, gen, aug_gen, sampler, t, device, with_aux=True)
        if probe is None:
            probe = (inp.clone(), actions.clone(), clean.clone())
        opt.zero_grad(set_to_none=True)
        loss, m = train_loss(model, rc, inp, clean, actions, aux)
        if rc.image_weight > 0:
            l_img, m["image_recon"] = image_loss(model, rc, image_bank, gen, device)
            loss = loss + rc.image_weight * l_img
        loss.backward()
        gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), rc.grad_clip))     # norm BEFORE clipping
        for k, v in (("loss", float(loss)), ("recon", m["reconstruction"]), ("kl", m["kl"]), ("pred", m["prediction"]),
                     ("img", m.get("image_recon")), ("gnorm", gnorm), ("gclip", float(gnorm > rc.grad_clip))):
            if v is not None:
                acc[k] = acc.get(k, 0.0) + v
        acc["gnorm_max"] = max(acc.get("gnorm_max", 0.0), gnorm)
        acc_n += 1
        if plast is not None:
            lrs = plast.update(m["error"], retention_ok)
            for g in opt.param_groups:
                g["lr"] = lrs[g["name"]]
        else:
            for g in opt.param_groups:
                g["lr"] = lr_at(rc, step, total_steps)
        opt.step()
        if plast is not None and (step + 1) % 20 == 0:                 # old behaviour: measure retention on a fixed batch
            with torch.no_grad():
                e = mse(model.observe(probe[0], probe[1])["recon"], probe[2])
            probe_ema = e if probe_ema is None else 0.5 * probe_ema + 0.5 * e
            probe_best = probe_ema if probe_best is None else min(probe_best, probe_ema)
            retention_ok = probe_ema <= 1.25 * probe_best
        if (step + 1) % log_every == 0 or step == 0:
            row = dict(step=step + 1, T=t, loss=float(loss), recon=m["reconstruction"], kl=m["kl"], pred=m["prediction"],
                       cons=m["consistency"], lr=opt.param_groups[0]["lr"], sec=round(time.time() - t0, 1))
            if rc.image_weight > 0:
                row["img"] = m["image_recon"]
            for k, v in acc.items():         # *_avg = mean over the steps since the previous row (the first row covers step 1 only)
                row[k + "_avg" if k not in ("gnorm", "gclip", "gnorm_max") else k] = round(v / acc_n if k != "gnorm_max" else v, 4)
            acc, acc_n = {}, 0
            for k in ("geom", "geom_prior", "overshoot"):
                if k in m:
                    row[k] = round(m[k], 4)
            if plast is not None:
                row["scales"] = {k: round(v, 3) for k, v in plast.scales.items()}; row["retention_ok"] = retention_ok
            hist.append(row)
            log.write(json.dumps(row) + "\n"); log.flush()
            print(f"[{rc.name}] " + " ".join(f"{k}={v}" for k, v in row.items() if k != "scales"), flush=True)
        if ckpt_every and (step + 1) % ckpt_every == 0 and step + 1 < total_steps:   # insurance for long runs
            torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()}, "config": asdict(cfg), "run": asdict(rc), "step": step + 1},
                       Path(log_path).parent / "model_partial.pt")
    log.close()
    return model, cfg, hist, time.time() - t0


# ----------------------------------------------------------------------------------------------- evaluation
@torch.no_grad()
def eval_heldout(model, rc, val, mean_img, n=192, t=16, chunk=64, seed=11):
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    if n < chunk:
        raise ValueError(f"eval_heldout: n={n} windows is smaller than one chunk of {chunk} (--eval-windows must be >= {chunk})")
    sampler = InterventionSampler(p=1.0)
    res = {}
    for tag, intervene in (("passive", False), ("intervened", True)):
        s = dict.fromkeys(["recon", "pred", "mean_recon", "mean_pred", "copy_pred", "recon_colour", "recon_other"], 0.0)
        n_col = n_oth = 0
        for _ in range(n // chunk):
            b = val.sample(chunk, t, generator=gen, train=False)
            x = to_channels(b["images"], rc.channels)
            a = torch.zeros(chunk, t, model.cfg.action_dim, device=x.device)
            if intervene:
                x, a = sampler(x, generator=gen)
                a = pad_actions(a.to(x.device), model)
            out = model.observe(x, a)
            ctx = t // 2
            pred = model.rollout(out["h"][:, ctx - 1], out["z"][:, ctx - 1], a[:, ctx:])
            tgt = x[:, ctx:]
            mi = mean_img.to(x.device)[None, None]
            s["recon"] += mse(out["recon"], x) * chunk; s["pred"] += mse(pred, tgt) * chunk
            s["mean_recon"] += mse(mi, x) * chunk; s["mean_pred"] += mse(mi, tgt) * chunk
            s["copy_pred"] += mse(x[:, ctx - 1:ctx], tgt) * chunk
            col = b["is_colour"]
            if col.any():
                s["recon_colour"] += mse(out["recon"][col], x[col]) * int(col.sum()); n_col += int(col.sum())
            if (~col).any():
                s["recon_other"] += mse(out["recon"][~col], x[~col]) * int((~col).sum()); n_oth += int((~col).sum())
        n_tot = (n // chunk) * chunk
        res[tag] = {k: v / n_tot for k, v in s.items() if k not in ("recon_colour", "recon_other")}
        res[tag]["recon_colour"] = s["recon_colour"] / max(1, n_col)
        res[tag]["recon_other"] = s["recon_other"] / max(1, n_oth)
    model.train()
    return res


@torch.no_grad()
def eval_freerun(model, rc, val, out_dir, n=16, ctx=8, horizon=128, seed=12):
    """Observe `ctx` real frames, then imagine with zero action. Error vs the real continuation per horizon, drift of
    image statistics, and how many generated frames have gone flat."""
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    longest = val.max_window()
    horizon = next((h for h in (horizon, 128, 64, 32, 16, 8) if ctx + h <= longest and h <= horizon), None)
    if horizon is None:
        return {"skipped": f"longest val sequence {longest} < {ctx + 8}"}
    t = ctx + horizon
    b = val.sample(n, t, generator=gen, train=False)
    x = to_channels(b["images"], rc.channels)
    torch.manual_seed(seed)
    out = model.observe(x[:, :ctx], torch.zeros(n, ctx, model.cfg.action_dim, device=x.device))
    frames, _ = model.imagine(out["h"][:, -1], out["z"][:, -1], torch.zeros(n, horizon, model.cfg.action_dim, device=x.device))
    res = {"horizon_available": horizon, "by_horizon": {}}
    for hz in HORIZONS:
        if hz > horizon:
            continue
        g, r = frames[:, hz - 1], x[:, ctx + hz - 1]
        res["by_horizon"][hz] = dict(model_mse=mse(g, r), copy_last_mse=mse(x[:, ctx - 1], r),
                                     gen_mean=float(g.mean()), real_mean=float(r.mean()),
                                     gen_std=float(g.flatten(1).std(1).mean()), real_std=float(r.flatten(1).std(1).mean()),
                                     gen_flat_frac=float((g.flatten(1).std(1) < 0.03).float().mean()))
    rows = []
    sel = [h for h in (1, 8, 32, 128) if h <= horizon]
    for i in range(min(6, n)):
        rows.append([x[i, ctx - 1]] + [frames[i, h - 1] for h in sel] + [x[i, ctx + h - 1] for h in sel])
    save_sheet(rows, out_dir / "sheet_freerun.png")          # cols: last context | model at +h ... | real at +h ...
    model.train()
    return res


@torch.no_grad()
def eval_noise_seeds(model, rc, val, out_dir, n=16, steps=300, seed=13):
    """The UI's start: sample the model's own prior from a zero state (SEED_STEPS zero-action steps), then free-run."""
    model.eval()
    dev = next(model.parameters()).device
    torch.manual_seed(seed)
    h, z = model.rssm.initial(n, dev)
    zero = torch.zeros(n, model.cfg.action_dim, device=dev)
    for _ in range(SEED_STEPS):
        h, z, _ = model.rssm.imagine_step(h, z, zero)
    times = [0, 10, 30, 100, 300]
    frames = {0: model.decode(h, z)}
    for t in range(1, steps + 1):
        h, z, _ = model.rssm.imagine_step(h, z, zero)
        if t in times:
            frames[t] = model.decode(h, z)
    gen = torch.Generator().manual_seed(seed)
    real = to_channels(val.sample(1024, 1, generator=gen, train=False)["images"], rc.channels)[:, 0]
    ref = small_gray(real)
    res = {"by_time": {}}
    for t in times:
        g = frames[t]
        res["by_time"][t] = dict(mean=float(g.mean()), std=float(g.flatten(1).std(1).mean()), sharp=sharpness(g),
                                 flat_frac=float((g.flatten(1).std(1) < 0.03).float().mean()),
                                 nn_dist_to_real=nn_distance(small_gray(g), ref))
    res["real_reference"] = dict(mean=float(real.mean()), std=float(real.flatten(1).std(1).mean()), sharp=sharpness(real),
                                 nn_dist_real_to_real=nn_distance(ref[:128], ref[128:]))
    save_sheet([[frames[t][i] for t in times] for i in range(n)], out_dir / "sheet_noise_seeds.png")   # cols t=0,10,30,100,300
    model.train()
    return res


@torch.no_grad()
def eval_scatter(model, rc, val, out_dir, n=32, t=8, seed=14):
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    x = to_channels(val.sample(n, t, generator=gen, train=False)["images"], rc.channels)
    b_, t_, c, h, w = x.shape
    aug = torch.Generator(device=x.device).manual_seed(seed)
    cor = scatter_batch(x.reshape(-1, c, h, w), aug, max_discs=3, radius=(0.15, 0.25), strength=(0.7, 1.0),
                        p_disc=1.0).view(b_, t_, c, h, w)
    torch.manual_seed(seed)
    rc_clean = model.observe(x, torch.zeros(n, t, model.cfg.action_dim, device=x.device))["recon"]
    torch.manual_seed(seed)
    rc_cor = model.observe(cor, torch.zeros(n, t, model.cfg.action_dim, device=x.device))["recon"]
    res = dict(mse_corrupt_vs_clean=mse(cor, x), mse_recon_clean_vs_clean=mse(rc_clean, x),
               mse_recon_corrupt_vs_clean=mse(rc_cor, x), recon_dependence_on_corruption=mse(rc_cor, rc_clean),
               sharp_recon_corrupt=sharpness(rc_cor.reshape(-1, c, h, w)), sharp_recon_clean=sharpness(rc_clean.reshape(-1, c, h, w)),
               sharp_real=sharpness(x.reshape(-1, c, h, w)))
    save_sheet([[cor[i, -1], rc_cor[i, -1], rc_clean[i, -1], x[i, -1]] for i in range(6)], out_dir / "sheet_scatter.png")
    model.train()                                            # cols: corrupted | recon(corrupted) | recon(clean) | clean
    return res


@torch.no_grad()
def eval_images(model, rc, bank, mean_img, n=256, chunk=64, seed=21):
    """Held-out single images (T=1, zero action): recon MSE, the mean-image baseline, sharpness of recon / sharpness of real."""
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    rec = base = sh_r = sh_x = 0.0
    done = 0
    for _ in range(max(1, n // chunk)):
        x = to_channels(bank.sample(chunk, 1, generator=gen, train=False)["images"], rc.channels)
        out = model.observe(x, torch.zeros(chunk, 1, model.cfg.action_dim, device=x.device))
        rec += mse(out["recon"], x) * chunk
        base += mse(mean_img.to(x.device)[None, None], x) * chunk
        sh_r += sharpness(out["recon"][:, 0]) * chunk
        sh_x += sharpness(x[:, 0]) * chunk
        done += chunk
    model.train()
    return dict(recon=rec / done, mean_image=base / done, sharp_ratio=sh_r / max(sh_x, 1e-12), n=done)


@torch.no_grad()
def eval_colour(model, rc, val, out_dir, n=32, t=8, seed=15):
    if rc.channels != 3:
        return {"skipped": "grayscale model"}
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    xs, others = [], []
    for _ in range(40):
        b = val.sample(64, t, generator=gen, train=False)
        xs.append(b["images"][b["is_colour"]]); others.append(b["images"][~b["is_colour"]])
        if sum(len(v) for v in xs) >= n and sum(len(v) for v in others) >= n:
            break
    col, oth = torch.cat(xs)[:n], torch.cat(others)[:n]
    if len(col) == 0:
        return {"skipped": "no colour clips in val"}
    z5 = lambda k: torch.zeros(k, t, model.cfg.action_dim, device=col.device)
    torch.manual_seed(seed)
    from_gray = model.observe(luma(col), z5(len(col)))["recon"]
    torch.manual_seed(seed)
    from_col = model.observe(col, z5(len(col)))["recon"]
    res = dict(n_colour=len(col), mse_gray_input_vs_colour=mse(luma(col), col), mse_recon_from_gray_vs_colour=mse(from_gray, col),
               mse_recon_from_colour_vs_colour=mse(from_col, col), chroma_target=chroma(col), chroma_recon_from_gray=chroma(from_gray),
               chroma_recon_from_colour=chroma(from_col))
    if len(oth):
        torch.manual_seed(seed)
        o = model.observe(oth, z5(len(oth)))["recon"]
        res.update(chroma_target_on_gray_clips=chroma(oth), chroma_recon_on_gray_clips=chroma(o))
    save_sheet([[luma(col)[i, -1], from_gray[i, -1], col[i, -1]] for i in range(6)], out_dir / "sheet_colour.png")
    model.train()                                             # cols: gray input | recon from gray | true colour
    return res


@torch.no_grad()
def eval_geometry(model, rc, val, n=192, t=8, chunk=64, seed=17):
    """Held-out S_ windows: error of the geometry head in PIXELS of the image (positions / radii x size/2), averaged over the valid
    labels of the last half of each window. 'head' reads the posterior state (it saw the frame), 'prior' reads the state BEFORE the
    frame (what the dynamics + command alone knew). no_command = zero command; with_command = the real cleaned command (gaze models).
    mean_baseline = always predicting the training mean. Skipped for models without a geometry head or banks without labels."""
    if model.cfg.geom_dim == 0 or not getattr(val, "has_labels", False):
        return {"skipped": "no geometry head" if model.cfg.geom_dim == 0 else "bank has no labels"}
    model.eval()
    half = val.size / 2.0
    res = {"px_per_unit": half, "dims": list(GEOM_NAMES)}
    for tag, use_cmd in (("no_command", False),) + ((("with_command", True),) if rc.gaze_command else ()):
        gen = torch.Generator().manual_seed(seed)
        err = {k: torch.zeros(GEOM_DIM) for k in ("geom", "geom_prior", "mean_baseline")}
        cnt = torch.zeros(GEOM_DIM)
        for _ in range(max(1, n // chunk)):
            b = val.sample(chunk, t, generator=gen, train=False)
            x = to_channels(b["images"], rc.channels)
            a = torch.zeros(chunk, t, model.cfg.action_dim, device=x.device)
            if use_cmd:
                a[..., 5] = b["has_cmd"].float(); a[..., 6:8] = b["cmd"]
            out = model.observe(x, a)
            valid = b["geom_valid"][:, t // 2:].float()
            tgt = b["geom"][:, t // 2:]
            cnt += valid.sum((0, 1)).cpu()
            for k in ("geom", "geom_prior"):
                pred = out[k][:, t // 2:] * model.geom_std + model.geom_mean
                err[k] += ((pred - tgt).abs() * valid).sum((0, 1)).cpu()
            err["mean_baseline"] += ((model.geom_mean - tgt).abs() * valid).sum((0, 1)).cpu()
        res[tag] = {k: {GEOM_NAMES[i]: (round(float(v[i] / cnt[i] * half), 3) if cnt[i] > 0 else None) for i in range(GEOM_DIM)}
                    for k, v in err.items()}
        res[tag]["n_valid"] = {GEOM_NAMES[i]: int(cnt[i]) for i in range(GEOM_DIM)}
    model.train()
    return res


# ----------------------------------------------------------------------------------------------- driver
def build_run(name, steps):
    base = PRESETS[name]
    return replace(RunConfig(name=name), **base)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, nargs="+", help="one or more cache dirs (each with manifest.json)")
    ap.add_argument("--source-weights", default=None,
                    help="share of samples per source, e.g. video=0.5,openeds=0.5 (default: every group equally likely)")
    ap.add_argument("--out", default="pilot_out")
    ap.add_argument("--runs", nargs="+", default=list(PRESETS))
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--image-size", type=int, default=96)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tint-mode", default="gray", choices=["gray", "keep"])
    ap.add_argument("--max-videos", type=int, default=None, help="smoke tests only")
    ap.add_argument("--max-frames-per-video", type=int, default=None, help="smoke tests only")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--eval-windows", type=int, default=192)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--ckpt-every", type=int, default=0, help="save model_partial.pt every N steps (long runs)")
    ap.add_argument("--labels", default=None,
                    help="aligned mask cache dir (build_cache_openeds.py --masks): enables eye geometry labels + gaze command for S_ frames")
    ap.add_argument("--init-from", default=None, help="model.pt to continue from (every run in this call starts from it; architecture must match)")
    a = ap.parse_args(argv)
    device = torch.device("cuda" if a.device == "auto" and torch.cuda.is_available() else ("cpu" if a.device == "auto" else a.device))
    torch.set_num_threads(max(1, min(6, torch.get_num_threads())))
    out_root = Path(a.out); out_root.mkdir(parents=True, exist_ok=True)
    weights = ({k: float(v) for k, v in (p.split("=") for p in a.source_weights.split(","))} if a.source_weights else {})
    kw = dict(image_size=a.image_size, val_frac=a.val_frac, seed=a.seed, tint_mode=a.tint_mode, max_videos=a.max_videos,
              max_frames_per_video=a.max_frames_per_video, device=device, labels_dir=a.labels)
    dyn_sources, img_sources = source_kinds(a.cache)           # temporal sources vs single-image (appearance) sources
    if not dyn_sources:
        raise SystemExit("no temporal source in --cache")

    def bank(split, srcs, hflip=True, **extra):
        w = {s: v for s, v in weights.items() if s in srcs}      # weights only for the sources actually in this bank
        return CacheBank(a.cache, split=split, sources=list(srcs), hflip=hflip, source_weights=w or None, **kw, **extra)

    t0 = time.time()
    train = bank("train", dyn_sources)
    val_sources = [s for s in dyn_sources if s not in train_only_sources(a.cache)]     # synthetic train-only sources (the dream pool) have no val groups
    val = bank("val", val_sources, hflip=False)
    # one val set per temporal source (a video-only cache gives the old val set), plus one per appearance source
    val_by_source = {s: bank("val", [s], hflip=False) for s in val_sources} if len(val_sources) > 1 else {}
    img_val = {s: bank("val", [s], hflip=False) for s in img_sources}
    img_train = bank("train", img_sources) if img_sources and any(
        build_run(n, a.steps).image_weight > 0 for n in a.runs) else None
    print(f"device={device} dynamics={dyn_sources} appearance={img_sources} train={train.info()} val={val.info()} "
          f"load={time.time() - t0:.0f}s", flush=True)
    (out_root / "data_info.json").write_text(json.dumps(dict(
        train=train.info(), val=val.info(), appearance_train=img_train.info() if img_train else None,
        appearance_val={s: b.info() for s, b in img_val.items()}, args=vars(a)), indent=1))
    g = torch.Generator().manual_seed(a.seed)
    mean_img = to_channels(train.sample(1024, 1, generator=g, train=False)["images"], 3)[:, 0].mean(0)   # C,H,W (RGB)
    img_mean = None
    if img_val:                                                  # train-set mean image of the appearance sources (baseline)
        ref = img_train or bank("train", img_sources)
        img_mean = to_channels(ref.sample(1024, 1, generator=g, train=False)["images"], 3)[:, 0].mean(0)
        del ref
    summary = {}
    for name in a.runs:
        run_dir = out_root / name
        if a.skip_existing and (run_dir / "metrics.json").exists():
            summary[name] = json.loads((run_dir / "metrics.json").read_text()); continue
        run_dir.mkdir(exist_ok=True)
        rc = build_run(name, a.steps)
        if a.batch:
            rc = replace(rc, batch=a.batch)
        mi = mean_img if rc.channels == 3 else luma(mean_img[None])[:, :1][0]
        run_srcs = list(rc.train_sources) if rc.train_sources else dyn_sources
        bad = sorted(set(run_srcs) - set(dyn_sources))
        if bad:
            raise SystemExit(f"[{name}] train_sources {bad} are not temporal sources in --cache (have {dyn_sources})")
        run_train = train
        if rc.train_group_frac < 1.0 or run_srcs != dyn_sources:
            run_train = bank("train", run_srcs, group_frac=rc.train_group_frac)
            print(f"[{name}] own train bank: {run_train.info()}", flush=True)
        model, cfg, hist, secs = train_run(rc, run_train, device, a.steps, a.seed, run_dir / "train_log.jsonl", a.log_every,
                                         ckpt_every=a.ckpt_every, image_bank=img_train, init_from=a.init_from)
        t1 = time.time()
        metrics = dict(run=asdict(rc), train_seconds=round(secs, 1), steps_per_sec=round(a.steps / secs, 2),
                       params=sum(p.numel() for p in model.parameters()),
                       train_data=run_train.info(),
                       train_fit=eval_heldout(model, rc, run_train, mi, n=a.eval_windows),   # same eval on TRAIN subjects -> gap
                       heldout=eval_heldout(model, rc, val, mi, n=a.eval_windows),
                       heldout_by_source={s: eval_heldout(model, rc, b, mi, n=a.eval_windows) for s, b in val_by_source.items()},
                       heldout_images_by_source={s: eval_images(model, rc, b, img_mean if rc.channels == 3 else luma(img_mean[None])[:, :1][0])
                                                 for s, b in img_val.items()},
                       freerun=eval_freerun(model, rc, val, run_dir),
                       noise_seeds=eval_noise_seeds(model, rc, val, run_dir),
                       scatter=eval_scatter(model, rc, val, run_dir),
                       colour=eval_colour(model, rc, val, run_dir),
                       geometry=eval_geometry(model, rc, val_by_source.get("openeds", val), n=a.eval_windows))
        metrics["eval_seconds"] = round(time.time() - t1, 1)
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=1, default=float))
        torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()}, "config": asdict(cfg), "run": asdict(rc), "mean_image": mi.cpu()},
                   run_dir / "model.pt")
        summary[name] = metrics
        print(f"[{name}] done: train {secs:.0f}s, eval {metrics['eval_seconds']}s", flush=True)
    (out_root / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    return summary


if __name__ == "__main__":
    main()
