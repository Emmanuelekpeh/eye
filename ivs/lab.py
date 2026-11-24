"""Mechanistic-interpretability lab for the RSSM eye models (CPU only, no dataset: the only pixels are the models' own dreams).

State = [h | z] (deter_dim + stoch_dim). Everything runs the UI's own dream loop (`dream` = ivs.eyemetric.ui_dream_frames + actions + recording +
interventions; bit-identical to it without actions/edits, tests/test_lab.py): predict -> render -> perceive the render -> show.

Labels (no privileged data): (a) the cumulative action = the hidden touch state s = [shift_x, shift_y, gain, light_x, light_y] that WE drive the dream with,
(b) eye variables measured on the model's render by the eye segmenter (pupil x/y/radius, iris radius, brightness, nested-eye flag).

Sections (each returns plain dicts; `analyse` runs them all for one checkpoint, `run_model` caches them under pilot_results/lab/<model>/):
  capacity   per-unit KL (posterior vs prior), active units (KL > 0.01 nats; Burda: variance of the posterior mean > 0.01), participation ratio of the states
  probes     ridge probes h / z / [h|z] -> every label, held-out BY WORLD, with controls: shuffled labels and a random-init network of the same architecture
             that gets the same actions; per-unit correlations, tuning curves, probe directions (reused for steering)
  steer      CAUSAL test: add k * (std along the probe direction) * direction to the decoded state; response of every measured variable vs random directions
  lesion     clamp unit sets to their mean: single units, active vs inactive z, random matched sets, top probe units; paired with the unlesioned dream
  patch      copy unit sets from a donor world (same tick, same noise) into a recipient world
  dynamics   GRU Jacobian spectrum, closed-loop perturbation growth, drift / collapse of worlds towards a prototype, deterministic (noise-free) loop
  images     decoder filmstrips along probe directions, encoder channel activation maximisation
  cka        linear CKA between models on the same stimuli (comparative anatomy; `cka_matrix`)
"""
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from . import eyemetric as E
from .actions import InterventionSampler
from .config import ModelConfig
from .gaze import MIN_IRIS_PX, MIN_PUPIL_PX
from .model import EyeWorldModel

VERSION = 1
SEED_STEPS = 4                      # = ivs.ui.SEED_STEPS
S_NAMES = ("shift_x", "shift_y", "gain", "light_x", "light_y")
M_NAMES = ("pupil_x", "pupil_y", "pupil_r", "iris_r", "brightness", "eye")
TARGETS = S_NAMES + M_NAMES
KL_ACTIVE = 0.01
ROOT = Path(__file__).resolve().parents[1]
LAB_DIR = ROOT / "pilot_results" / "lab"
SEG_PATH = ROOT / "pilot_results" / "eyemetric" / "segmenter.pt"


@dataclass
class LabConfig:
    seed: int = 21
    probe_worlds: int = 16
    probe_ticks: int = 100
    burn: int = 10                   # ticks dropped before probing (start-up transient)
    stride: int = 2                  # probe on every 2nd tick (neighbouring ticks are nearly the same sample)
    folds: int = 5
    alphas: tuple = (1.0, 10.0, 100.0, 1000.0, 10000.0)
    steer_worlds: int = 8
    steer_ticks: int = 50
    steer_t0: int = 15
    steer_t1: int = 45
    steer_alphas: tuple = (-2.0, 2.0)  # in standard deviations of the state's spread along the probe direction
    steer_random: int = 4
    lesion_units: int = 8            # single units tried per space (z by KL, h by std)
    lesion_random: int = 3
    group_size: int = 8              # size of the "top units of a variable" and random matched sets
    dyn_worlds: int = 8
    dyn_ticks: int = 200
    perturb: float = 0.01
    jac_ticks: tuple = (10, 40, 100, 190)
    am_channels: int = 8
    am_steps: int = 60
    sections: tuple = ("capacity", "probes", "steer", "lesion", "patch", "dynamics", "images")

    @staticmethod
    def quick():
        return LabConfig(probe_worlds=8, probe_ticks=50, burn=6, steer_worlds=4, steer_ticks=30, steer_t0=8, steer_t1=24, steer_random=2,
                         lesion_units=3, lesion_random=1, group_size=4, dyn_worlds=4, dyn_ticks=60, jac_ticks=(10, 40), am_channels=2, am_steps=10)


# ---------------------------------------------------------------------------------- loading / basics
def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = EyeWorldModel(ModelConfig(**ck["config"]))
    m.load_state_dict(ck["model"])
    return m.eval()


def load_segmenter():
    return E.load_segmenter(str(SEG_PATH))


def random_model(cfg, seed=0):
    """Untrained network of the same architecture (default init): the control every probe is compared with."""
    torch.manual_seed(seed)
    return EyeWorldModel(cfg).eval()


def pad_actions(a, action_dim):
    extra = action_dim - a.shape[-1]
    return a if extra <= 0 else torch.cat([a, torch.zeros(*a.shape[:-1], extra)], -1)


def action_walk(n, ticks, seed, action_dim=5):
    """The training distribution of touch: a smooth random walk of the hidden state, the model sees only its per-step change (a[:, 0] = 0).
    Returns (actions (n,T,action_dim), s (n,T,5) = cumulative state change in state units = exactly what the model has been told so far)."""
    sam = InterventionSampler(p=1.0)
    _, a = sam._walk(n, ticks, torch.Generator().manual_seed(int(seed) & 0x7FFFFFFF))
    unit = sam.step * sam.limit / sam.limit[0]
    return pad_actions(a, action_dim), (a * unit).cumsum(1)


def seed_state(model, n, noise_seed, seed_steps=SEED_STEPS):
    """The UI's noise start: the prior sampled from the zero state for `seed_steps` zero-action steps."""
    with torch.no_grad():
        torch.manual_seed(int(noise_seed) & 0x7FFFFFFF)
        h, z = model.rssm.initial(n, "cpu")
        zero = torch.zeros(n, model.cfg.action_dim)
        for _ in range(seed_steps):
            h, z, _ = model.rssm.imagine_step(h, z, zero)
    return h, z


def _observe(R, h, z, a, embed, temperature=1.0):
    """RSSM.observe_step with the per-unit KL(posterior || prior) and the posterior mean returned (same RNG consumption)."""
    h2 = R.gru(torch.cat([z, a], dim=-1), h)
    pm, ps = R._stats(R.prior(h2))
    qm, qs = R._stats(R.posterior(torch.cat([h2, embed], dim=-1)))
    z2 = qm + temperature * qs * torch.randn_like(qs)
    kl = torch.log(ps / qs) + (qs ** 2 + (qm - pm) ** 2) / (2 * ps ** 2) - 0.5
    return h2, z2, kl, qm


class Intervention:
    """Edit of the latent state while dreaming. State = [h | z] (length D = deter + stoch).
    clamp_idx/clamp_val: those units are forced to the value every tick in [t0, t1) (lesion = clamp to the mean).
    add: a length-D vector added to the state (steering); by default only to the state that is DECODED this tick (stage 'pred') so it does not
    accumulate through the recurrence; the model then perceives the steered render like any other frame.
    patch = (idx, src): src = {'pred': (T,n,D), 'post': (T,n,D)} donor states; units idx are copied from the donor at the same tick."""

    def __init__(self, deter, t0=0, t1=10 ** 9, clamp_idx=None, clamp_val=None, add=None, stages=("pred", "post"), patch=None):
        self.deter, self.t0, self.t1, self.stages = deter, t0, t1, tuple(stages)
        self.clamp_idx = None if clamp_idx is None else torch.as_tensor(list(clamp_idx), dtype=torch.long)
        self.clamp_val = None if clamp_val is None else torch.as_tensor(clamp_val, dtype=torch.float32)
        self.add = None if add is None else torch.as_tensor(add, dtype=torch.float32)
        self.patch = None if patch is None else (torch.as_tensor(list(patch[0]), dtype=torch.long), patch[1])

    def __call__(self, h, z, t, stage):
        if not (self.t0 <= t < self.t1) or stage not in self.stages:
            return h, z
        s = torch.cat([h, z], -1).clone()
        if self.clamp_idx is not None and len(self.clamp_idx):
            v = self.clamp_val
            s[:, self.clamp_idx] = v if v.ndim == 0 else v.to(s.dtype)
        if self.patch is not None and len(self.patch[0]):
            idx, src = self.patch
            s[:, idx] = src[stage][t][:, idx]
        if self.add is not None:
            s = s + self.add
        return s[:, :self.deter], s[:, self.deter:]


# ---------------------------------------------------------------------------------- the dream loop
@torch.no_grad()
def dream(model, ticks=120, n=8, noise_seed=21, actions=None, init_state=None, temperature=1.0, intervention=None, seed_steps=SEED_STEPS):
    """ivs.eyemetric.ui_dream_frames with actions, recording and interventions. actions (n,T,A) or None (zero). temperature scales the sampling noise of
    the prior AND the posterior (1 = the UI, 0 = noise-free loop). Returns dict: frames uint8 (T,n,3,H,W) (exactly the page's 8-bit render),
    ph/pz = state decoded at tick t, h/z = state after perceiving the render, kl (T,n,Z) per-unit KL, qm posterior mean."""
    model.eval()
    R, cfg = model.rssm, model.cfg
    zero = torch.zeros(n, cfg.action_dim)
    if init_state is not None:
        if len(init_state[0]) != n:
            raise ValueError(f"init_state holds {len(init_state[0])} states but n={n} worlds were asked for")
        h, z = init_state[0].clone(), init_state[1].clone()
    else:
        h, z = seed_state(model, n, noise_seed, seed_steps)
    out = {k: [] for k in ("frames", "ph", "pz", "h", "z", "kl", "qm")}
    for t in range(ticks):
        seed = (int(noise_seed) * 1000003 + t * 7919) & 0x7FFFFFFF
        a = zero if actions is None else actions[:, t]
        torch.manual_seed(seed)
        ph, pz, _ = R.imagine_step(h, z, a, temperature)
        if intervention is not None:
            ph, pz = intervention(ph, pz, t, "pred")
        pred = model.decode(ph, pz)
        torch.manual_seed(seed + 1)
        h, z, kl, qm = _observe(R, h, z, a, model.encoder(pred), temperature)
        if intervention is not None:
            h, z = intervention(h, z, t, "post")
        q = (pred.clamp(0, 1) * 255).round().to(torch.uint8)
        out["frames"].append(q if q.shape[1] == 3 else q.expand(-1, 3, -1, -1).contiguous())
        for k, v in (("ph", ph), ("pz", pz), ("h", h), ("z", z), ("kl", kl), ("qm", qm)):
            out[k].append(v)
    out = {k: torch.stack(v) for k, v in out.items()}
    out["actions"] = None if actions is None else actions.transpose(0, 1)
    return out


def as_float(frames):
    """uint8 frames -> float in [0,1] exactly as ui_dream_frames returns them."""
    return frames.float() / 255


@torch.no_grad()
def measure(frames, seg, chunk=256):
    """Eye variables of every frame. frames (T,n,3,H,W) uint8 or float. Returns {name: (T,n) float array}, NaN where undefined (pupil/iris absent).
    pupil_x/y: centroid in image units [-1,1]; pupil_r / iris_r: sqrt(area/pi)/96 (fraction of the 96 px frame); brightness: mean luma; eye: nested flag."""
    T, n = frames.shape[:2]
    fr = frames.flatten(0, 1)
    fr = fr.float() / 255 if fr.dtype == torch.uint8 else fr
    if fr.shape[-1] != 96:
        fr = F.interpolate(fr, size=(96, 96), mode="bilinear", align_corners=False)
    g = E.gray(fr)
    cls = torch.cat([E.segment(seg, g[i:i + chunk]) for i in range(0, len(g), chunk)])
    pupil, iris = cls == 3, (cls == 2) | (cls == 3)
    n_p, n_i = pupil.sum((1, 2)).float(), iris.sum((1, 2)).float()
    c = E._centroid(pupil)
    nan = torch.full_like(n_p, float("nan"))
    ok_p, ok_i = n_p >= MIN_PUPIL_PX, n_i >= MIN_IRIS_PX
    unit = lambda v: (v + 0.5) / 96 * 2 - 1
    m = dict(pupil_x=torch.where(ok_p, unit(c[:, 0]), nan), pupil_y=torch.where(ok_p, unit(c[:, 1]), nan),
             pupil_r=torch.where(ok_p, torch.sqrt(n_p / math.pi) / 96, nan), iris_r=torch.where(ok_i, torch.sqrt(n_i / math.pi) / 96, nan),
             brightness=g.mean((1, 2)), eye=E.nested_flags(cls).float())
    return {k: v.reshape(T, n).numpy().astype(np.float64) for k, v in m.items()}


def _nanmean(a, axis=None):
    a = np.asarray(a, dtype=np.float64)
    ok = np.isfinite(a)
    cnt = ok.sum(axis=axis)
    s = np.where(ok, a, 0.0).sum(axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)


def _states(rec, kind="pred"):
    """(T,n,D) numpy: the state decoded at tick t ('pred') or carried after perceiving ('post')."""
    h, z = (rec["ph"], rec["pz"]) if kind == "pred" else (rec["h"], rec["z"])
    return torch.cat([h, z], -1).numpy().astype(np.float64)


# ---------------------------------------------------------------------------------- capacity
def _pr(x):
    """participation ratio and #components for 90% variance of the rows of x (N,d)."""
    x = x - x.mean(0, keepdims=True)
    ev = np.clip(np.linalg.eigvalsh(x.T @ x / max(1, len(x) - 1)), 0, None)[::-1]
    if ev.sum() <= 1e-12:
        return 0.0, 0
    return float(ev.sum() ** 2 / (ev ** 2).sum()), int(np.searchsorted(np.cumsum(ev) / ev.sum(), 0.9) + 1)


def capacity(rec, deter, burn=10):
    kl = rec["kl"][burn:].numpy().astype(np.float64)
    qm = rec["qm"][burn:].numpy().astype(np.float64)
    kl_unit = kl.mean((0, 1))
    var_unit = qm.reshape(-1, qm.shape[-1]).var(0)
    X = _states(rec)[burn:].reshape(-1, deter + kl.shape[-1])
    h_pr, h_n90 = _pr(X[:, :deter])
    z_pr, z_n90 = _pr(X[:, deter:])
    hz_pr, hz_n90 = _pr(X)
    h_std, z_std = X[:, :deter].std(0), X[:, deter:].std(0)
    return dict(kl_total=float(kl_unit.sum()), kl_unit=kl_unit.tolist(), active_kl=int((kl_unit > KL_ACTIVE).sum()),
                active_var=int((var_unit > KL_ACTIVE).sum()), z_dim=int(kl.shape[-1]), h_dim=int(deter),
                h_std=h_std.tolist(), z_std=z_std.tolist(), h_silent=int((h_std < 0.01).sum()),
                pr=dict(h=h_pr, z=z_pr, hz=hz_pr), n90=dict(h=h_n90, z=z_n90, hz=hz_n90))


# ---------------------------------------------------------------------------------- probes
def _fold_ids(groups, folds, seed):
    ug = np.unique(groups)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(ug))
    fold_of = {int(ug[p]): i % folds for i, p in enumerate(perm)}
    return np.array([fold_of[int(g)] for g in groups])


def _ridge_preds(Xtr, ytr, Xte, alphas):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    ym = ytr.mean()
    lam, V = np.linalg.eigh(A.T @ A)                         # ridge via the Gram matrix: same solution as the SVD, D x D instead of N x D
    lam = np.clip(lam, 0, None)
    c = V.T @ (A.T @ (ytr - ym))
    return [B @ (V @ (c / (lam + al))) + ym for al in alphas]


def cv_r2(X, y, groups, alphas, folds=5, seed=0):
    """Held-out R2 of a ridge probe, folds split BY WORLD (no tick of a test world is ever in training). X (N,D), y (N,) with NaN allowed.
    Returns (best R2 over alphas, best alpha, N rows used); R2 is NaN when there is not enough variation."""
    ok = np.isfinite(y)
    X, y, groups = X[ok], y[ok], groups[ok]
    if len(y) < 40 or len(np.unique(groups)) < folds or y.std() < 1e-9:
        return float("nan"), float("nan"), int(len(y))
    fid = _fold_ids(groups, folds, seed)
    pred = np.zeros((len(alphas), len(y)))
    for f in range(folds):
        te = fid == f
        if te.sum() == 0 or (~te).sum() < 10:
            continue
        for i, p in enumerate(_ridge_preds(X[~te], y[~te], X[te], alphas)):
            pred[i, te] = p
    sst = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ((pred - y[None]) ** 2).sum(1) / sst
    b = int(np.argmax(r2))
    return float(r2[b]), float(alphas[b]), int(len(y))


def fit_direction(X, y, alpha):
    """Ridge on all valid rows -> unit direction in raw state space, gain (target units per unit move along it), spread of the states along it."""
    ok = np.isfinite(y)
    X, y = X[ok], y[ok]
    mu, sd = X.mean(0), X.std(0) + 1e-8
    A = (X - mu) / sd
    lam, V = np.linalg.eigh(A.T @ A)
    w = (V @ ((V.T @ (A.T @ (y - y.mean()))) / (np.clip(lam, 0, None) + alpha))) / sd
    gain = float(np.linalg.norm(w))
    d = w / max(gain, 1e-12)
    return d, gain, float((X @ d).std())


def _labels(sel, meas, walk_s):
    """(len(sel),n,K) labels in TARGETS order; meas was measured on the ticks `sel` only."""
    s = walk_s.transpose(0, 1).numpy().astype(np.float64)[sel]
    return np.stack([s[..., i] for i in range(5)] + [meas[k] for k in M_NAMES], -1)


def _shuffled(Y, rng):
    """Circularly shift every world's label series by a random offset (>= T/4): same marginals and autocorrelation, no link to the states."""
    out = np.empty_like(Y)
    T = Y.shape[0]
    for w in range(Y.shape[1]):
        out[:, w] = np.roll(Y[:, w], int(rng.randint(T // 4, 3 * T // 4 + 1)), axis=0)
    return out


def probe_ticks(T, lc):
    return np.arange(lc.burn, T, lc.stride)


def probes(rec, meas, walk_s, deter, lc, ctrl=None):
    """meas = eye variables measured on the ticks probe_ticks(T, lc) only (the segmenter is the expensive part).
    ctrl = (rec_control, meas_control) from the random-init network driven by the same actions."""
    T, n = rec["ph"].shape[:2]
    sel = probe_ticks(T, lc)
    X3 = _states(rec)[sel]                                   # (t,n,D)
    Y3 = _labels(sel, meas, walk_s)
    D = X3.shape[-1]
    groups = np.broadcast_to(np.arange(n)[None], X3.shape[:2]).reshape(-1)
    X, Y = X3.reshape(-1, D), Y3.reshape(-1, Y3.shape[-1])
    Ys = _shuffled(Y3, np.random.RandomState(lc.seed)).reshape(Y.shape)
    Xc = Yc = None
    if ctrl is not None:
        Xc = _states(ctrl[0])[sel].reshape(-1, D)
        Yc = _labels(sel, ctrl[1], walk_s).reshape(Y.shape)
    out = {}
    for k, name in enumerate(TARGETS):
        y = Y[:, k]
        r2 = {}
        for kind, cols in (("h", slice(0, deter)), ("z", slice(deter, D)), ("hz", slice(0, D))):
            r2[kind], alpha, rows = cv_r2(X[:, cols], y, groups, lc.alphas, lc.folds, lc.seed)
            if kind == "hz":
                alpha_hz = alpha
        rec_k = dict(r2=r2, alpha=alpha_hz, n=rows, r2_shuffle=cv_r2(X, Ys[:, k], groups, lc.alphas, lc.folds, lc.seed)[0],
                     r2_random_init=cv_r2(Xc, Yc[:, k], groups, lc.alphas, lc.folds, lc.seed)[0] if Xc is not None else None,
                     y_mean=float(np.nanmean(y)) if np.isfinite(y).any() else None, y_std=float(np.nanstd(y)) if np.isfinite(y).any() else None)
        ok = np.isfinite(y)
        if np.isfinite(r2["hz"]) and r2["hz"] > 0 and ok.sum() > 40:
            d, gain, psd = fit_direction(X, y, alpha_hz)
            rec_k.update(dir=d.tolist(), gain=gain, proj_sd=psd)
            xs, ys_ = X[ok], y[ok]
            xc = xs - xs.mean(0)
            r = (xc * (ys_ - ys_.mean())[:, None]).sum(0) / (np.sqrt((xc ** 2).sum(0) * ((ys_ - ys_.mean()) ** 2).sum()) + 1e-12)
            order = np.argsort(-np.abs(r))[:max(8, lc.group_size)]
            rec_k["top_units"] = [dict(unit=int(u), space="h" if u < deter else "z", index=int(u if u < deter else u - deter), r=float(r[u])) for u in order]
            rec_k["tuning"] = [_tuning(xs[:, u], ys_, int(u), deter) for u in order[:3]]
        out[name] = rec_k
    return out


def _tuning(x, y, unit, deter, bins=8):
    q = np.unique(np.quantile(y, np.linspace(0, 1, bins + 1)))
    if len(q) < 3:
        return dict(unit=unit, x=[], y=[])
    idx = np.clip(np.searchsorted(q, y, side="right") - 1, 0, len(q) - 2)
    return dict(unit=unit, space="h" if unit < deter else "z", x=[float(y[idx == b].mean()) for b in range(len(q) - 1)],
                y=[float(x[idx == b].mean()) for b in range(len(q) - 1)])


# ---------------------------------------------------------------------------------- causal tests
def _resp(base, other, nat_sd):
    """paired response of every measured variable (steered/lesioned minus base, valid pairs only; both measured on the same ticks) in raw units and
    in natural standard deviations."""
    raw = {k: float(_nanmean(other[k] - base[k])) for k in M_NAMES}
    return raw, {k: (raw[k] / nat_sd[k] if nat_sd.get(k) else None) for k in M_NAMES}


def _nat_sd(meas):
    return {k: (float(np.nanstd(meas[k])) if np.isfinite(meas[k]).any() else None) for k in M_NAMES}


def steer(model, seg, probe_res, lc, nat_sd, rng_seed=0):
    """Add alpha * proj_sd * direction (alpha in lc.steer_alphas) to the decoded state during [t0, t1). Compare the response of every measured variable with
    that of random unit directions of the same size. response_sd = change per unit alpha in natural standard deviations of the variable;
    effect/predicted (measured variables only) = achieved change vs the change the probe itself predicts for that move, in the variable's units.
    Measured on every 2nd tick of [t0+8, t1)."""
    deter = model.cfg.deter_dim
    D = deter + model.cfg.stoch_dim
    lo, hi = lc.steer_t0 + 8, lc.steer_t1
    win = lambda rec: measure(rec["frames"][lo:hi:2], seg)
    base = win(dream(model, lc.steer_ticks, lc.steer_worlds, lc.seed))

    def run(vec):
        iv = Intervention(deter, lc.steer_t0, lc.steer_t1, add=vec, stages=("pred",))
        return win(dream(model, lc.steer_ticks, lc.steer_worlds, lc.seed, intervention=iv))

    def slope(d, size):
        rs, raw_hi = {}, None
        for a in lc.steer_alphas:
            raw, rs[a] = _resp(base, run(torch.as_tensor(a * size * d, dtype=torch.float32)), nat_sd)
            if a == max(lc.steer_alphas):
                raw_hi = raw
        den = sum(a * a for a in lc.steer_alphas)
        sl = {k: (None if any(rs[a][k] is None or not np.isfinite(rs[a][k]) for a in lc.steer_alphas) else float(sum(a * rs[a][k] for a in lc.steer_alphas) / den)) for k in M_NAMES}
        return sl, raw_hi

    res = {}
    for name, pr in probe_res.items():
        if "dir" not in pr:
            continue
        sl, raw_hi = slope(np.asarray(pr["dir"]), pr["proj_sd"])
        entry = dict(response_sd=sl, proj_sd=pr["proj_sd"])
        if name in M_NAMES:
            a = max(lc.steer_alphas)
            entry.update(effect=raw_hi[name], predicted=a * pr["proj_sd"] * pr["gain"], alpha=a)
        res[name] = entry
    rng = np.random.RandomState(rng_seed)
    rand = {k: [] for k in M_NAMES}
    sizes = [pr["proj_sd"] for pr in probe_res.values() if "dir" in pr]
    size = float(np.median(sizes)) if sizes else 1.0
    for _ in range(lc.steer_random):
        d = rng.randn(D)
        d /= np.linalg.norm(d)
        sl, _ = slope(d, size)
        for k in M_NAMES:
            if sl[k] is not None:
                rand[k].append(abs(sl[k]))
    return dict(directions=res, random_abs_response_sd={k: (float(np.mean(v)) if v else None) for k, v in rand.items()}, random_size=size,
                window=[lc.steer_t0, lc.steer_t1], alphas=list(lc.steer_alphas))


def _eye(m):
    return float(_nanmean(m["eye"]))


def lesion(model, seg, probe_res, cap, lc, nat_sd):
    """Clamp unit sets to their mean (from the unlesioned dream) from tick t0 on, paired with that dream (same noise).
    pix = mean |pixel change| over the rest of the dream (dominated by chaotic divergence: compare with the random sets); pix_early = over the first 3 clamped
    ticks, before trajectories diverge (the direct effect on the render). d_eye = change of the nested-eye share."""
    deter, Z = model.cfg.deter_dim, model.cfg.stoch_dim
    D = deter + Z
    base_rec = dream(model, lc.steer_ticks, lc.steer_worlds, lc.seed)
    lo, t0 = lc.steer_t0 + 5, lc.steer_t0
    base = measure(base_rec["frames"][lo::2], seg)
    mean = _states(base_rec, "pred")[lc.burn:].reshape(-1, D).mean(0)
    kl = np.asarray(cap["kl_unit"])
    zs = [deter + int(i) for i in np.argsort(-kl)[:lc.lesion_units]]
    hs = [int(i) for i in np.argsort(-np.asarray(cap["h_std"]))[:lc.lesion_units]]
    groups = {f"z{u - deter}": [u] for u in zs}
    groups.update({f"h{u}": [u] for u in hs})
    act = [deter + int(i) for i in np.flatnonzero(kl > KL_ACTIVE)]
    ina = [deter + int(i) for i in np.flatnonzero(kl <= KL_ACTIVE)]
    if act:
        groups["z_active"] = act
    if ina:
        groups["z_inactive"] = ina
    groups["h_all"], groups["z_all"] = list(range(deter)), list(range(deter, D))
    for name in ("pupil_x", "pupil_y", "pupil_r", "brightness", "shift_x"):
        pr = probe_res.get(name, {})
        if "top_units" in pr:
            groups[f"top_{name}"] = [u["unit"] for u in pr["top_units"][:lc.group_size]]
    rng = np.random.RandomState(lc.seed)
    for i in range(lc.lesion_random):
        groups[f"random{lc.group_size}_{i}"] = [int(u) for u in rng.choice(D, lc.group_size, replace=False)]
    fb = base_rec["frames"].float()
    out = {}
    for name, idx in groups.items():
        iv = Intervention(deter, t0, 10 ** 9, clamp_idx=idx, clamp_val=mean[idx])
        rec = dream(model, lc.steer_ticks, lc.steer_worlds, lc.seed, intervention=iv)
        m = measure(rec["frames"][lo::2], seg)
        raw, sd = _resp(base, m, nat_sd)
        f = rec["frames"].float()
        out[name] = dict(units=len(idx), pix=float((f[lo:] - fb[lo:]).abs().mean() / 255), pix_early=float((f[t0:t0 + 3] - fb[t0:t0 + 3]).abs().mean() / 255),
                         eye=_eye(m), eye_base=_eye(base), d_eye=_eye(m) - _eye(base), response_sd=sd)
    return dict(groups=out, window=[lc.steer_t0, lc.steer_ticks], clamp="mean of the unlesioned dream", eye_base=_eye(base))


def patch(model, seg, lc):
    """Copy unit sets from a donor world (high eye share) into a recipient world (low eye share) of the same dream (same noise, same tick)."""
    deter, Z = model.cfg.deter_dim, model.cfg.stoch_dim
    D = deter + Z
    n, T = lc.steer_worlds, lc.steer_ticks
    rec = dream(model, T, n, lc.seed)
    lo = lc.steer_t0 + 5
    eye = _nanmean(measure(rec["frames"][lo::2], seg)["eye"], axis=0)
    order = np.argsort(eye)
    k = n // 2
    rec_i, don_i = order[:k].copy(), order[::-1][:k].copy()
    if k == 0 or not (eye[don_i].mean() - eye[rec_i].mean() > 0.2):
        return dict(skipped="no world pair with a clear eye-share gap in this dream", eye_per_world=eye.tolist())
    # Full batch (a world's noise depends on the batch layout): recipient rows get the donor's state, every other row gets its own unchanged state.
    src = {}
    for stage in ("pred", "post"):
        s = torch.tensor(_states(rec, stage), dtype=torch.float32)
        s[:, rec_i] = s[:, don_i]
        src[stage] = s
    out = {}
    groups = {"all_units": list(range(D)), "h_all": list(range(deter)), "z_all": list(range(deter, D))}
    for g, idx in groups.items():
        iv = Intervention(deter, lc.steer_t0, 10 ** 9, patch=(idx, src))
        mm = measure(dream(model, T, n, lc.seed, intervention=iv)["frames"][lo::2], seg)
        out[g] = dict(units=len(idx), eye=float(_nanmean(mm["eye"][:, rec_i])))
    return dict(groups=out, recipient_eye=float(eye[rec_i].mean()), donor_eye=float(eye[don_i].mean()), recipients=[int(i) for i in rec_i], donors=[int(i) for i in don_i])


# ---------------------------------------------------------------------------------- dynamics
def jacobian_spectrum(model, rec, ticks, worlds=4):
    """Spectrum of d h_{t+1} / d h_t of the GRU at the dream's own states (zero action). Excludes the feedback through render + encoder
    (that is `perturbation_growth`)."""
    R = model.rssm
    rows = []
    for t in ticks:
        if t >= rec["h"].shape[0]:
            continue
        for w in range(min(worlds, rec["h"].shape[1])):
            z, h = rec["z"][t, w], rec["h"][t, w]
            x = torch.cat([z, torch.zeros(model.cfg.action_dim)])[None]
            J = torch.autograd.functional.jacobian(lambda hh: R.gru(x, hh[None])[0], h).numpy().astype(np.float64)
            rows.append((t, float(np.abs(np.linalg.eigvals(J)).max()), float(np.linalg.svd(J, compute_uv=False)[0])))
    arr = np.array(rows)
    if not len(arr):
        return dict(ticks=[], radius_mean=None)
    return dict(ticks=sorted(set(int(t) for t in arr[:, 0])), radius_by_tick=[float(arr[arr[:, 0] == t, 1].mean()) for t in sorted(set(arr[:, 0]))],
                radius_mean=float(arr[:, 1].mean()), radius_max=float(arr[:, 1].max()), smax_mean=float(arr[:, 2].mean()))


def _pairwise(x):
    """mean pairwise L2 distance between worlds. x (n,d)."""
    d = torch.cdist(x, x)
    n = len(x)
    return float(d.sum() / max(1, n * (n - 1)))


def drift_stats(rec, seg_meas, lc):
    T, n = rec["h"].shape[:2]
    S = torch.cat([rec["h"], rec["z"]], -1)
    marks = [t for t in (0, 10, 30, 60, 100, 150, 199) if t < T] + ([T - 1] if (T - 1) not in (0, 10, 30, 60, 100, 150, 199) else [])
    d0 = _pairwise(S[0]) + 1e-9
    fr = rec["frames"].float() / 255
    step = (S[1:] - S[:-1]).norm(dim=-1).mean(1)
    early = float(step[min(5, len(step) - 1):min(15, len(step))].median()) if len(step) > 5 else float("nan")
    still = next((int(t) for t in range(5, len(step)) if step[t] < 0.1 * early), None) if early == early else None
    eye = seg_meas["eye"]                                   # measured on every 2nd tick
    k30 = 15
    return dict(marks=marks, from_start=[float((S[t] - S[0]).norm(dim=-1).mean()) for t in marks],
                world_dispersion=[_pairwise(S[t]) / d0 for t in marks],
                render_dispersion=[_pairwise(fr[t].flatten(1)) for t in marks],
                step_norm=[float(step[min(t, len(step) - 1)]) for t in marks], stationary_tick=still,
                eye_frac=float(np.nanmean(eye)), eye_first=float(np.nanmean(eye[:k30])), eye_last=float(np.nanmean(eye[-k30:])))


def perturbation_growth(model, lc):
    """Closed-loop (render -> perceive) sensitivity: two dreams with identical noise whose initial h differs by `perturb` in a random direction.
    lambda = slope of log(distance of [h|z]) over the first 25 ticks (> 0 amplifies, < 0 contracts)."""
    n, T = lc.dyn_worlds, min(lc.dyn_ticks, 120)
    h0, z0 = seed_state(model, n, lc.seed)
    g = torch.Generator().manual_seed(lc.seed)
    u = torch.randn(n, h0.shape[1], generator=g)
    u = u / u.norm(dim=1, keepdim=True)
    a = dream(model, T, n, lc.seed, init_state=(h0, z0))
    b = dream(model, T, n, lc.seed, init_state=(h0 + lc.perturb * u, z0))
    d = (torch.cat([a["h"], a["z"]], -1) - torch.cat([b["h"], b["z"]], -1)).norm(dim=-1)         # (T,n)
    k = min(25, T)
    ld = torch.log(d[:k].clamp_min(1e-12)).median(1).values.numpy()
    slope = float(np.polyfit(np.arange(k), ld, 1)[0])
    ref = _pairwise(torch.cat([a["h"], a["z"]], -1)[T - 1])
    return dict(lambda_per_tick=slope, perturb=lc.perturb, d_marks={str(t): float(d[t].median()) for t in (0, 5, 10, 25, 50, T - 1) if t < T}, saturation=float(d[T - 1].median()) / (ref + 1e-9))


def dynamics(model, seg, lc):
    rec = dream(model, lc.dyn_ticks, lc.dyn_worlds, lc.seed)
    m = measure(rec["frames"][::2], seg)
    det = dream(model, lc.dyn_ticks, lc.dyn_worlds, lc.seed, init_state=seed_state(model, lc.dyn_worlds, lc.seed), temperature=0.0)
    md = measure(det["frames"][::2], seg)
    return dict(noisy=drift_stats(rec, m, lc), deterministic=drift_stats(det, md, lc), jacobian=jacobian_spectrum(model, rec, lc.jac_ticks),
                perturbation=perturbation_growth(model, lc))


# ---------------------------------------------------------------------------------- images
def _tile(img_f, size=96):
    a = (img_f.clamp(0, 1) * 255).round().byte().permute(1, 2, 0).numpy()
    if a.shape[2] == 1:
        a = np.repeat(a, 3, 2)
    return cv2.resize(a, (size, size), interpolation=cv2.INTER_NEAREST if a.shape[0] < size else cv2.INTER_AREA)


def eye_reference(meas, sel, T):
    """(tick, world) of the dream state the filmstrip is drawn around: the frame the segmenter finds an eye in (after tick 15) with the largest iris,
    so the effect of a direction can be read on an actual eye; falls back to tick 30 of world 0 when no frame qualifies."""
    ok = (meas["eye"] == 1) & (sel[:, None] >= 15) & np.isfinite(meas["iris_r"])
    if not ok.any():
        return min(30, T - 1), 0, False
    score = np.where(ok, meas["iris_r"], -1.0)
    i, w = np.unravel_index(int(np.argmax(score)), score.shape)
    return int(sel[i]), int(w), True


def filmstrip(model, rows, ref_state, ks=(-3, -1.5, 0, 1.5, 3), size=96):
    """rows: {name: (unit direction (D,), spread along it)}; ref_state (n,D). One row per direction: decode(state + k * spread * direction) for k in ks, for the
    first reference state. Returns an RGB uint8 image with the row names on the left."""
    deter = model.cfg.deter_dim
    pad = 120
    lines = []
    with torch.no_grad():
        for name, (d, sd) in rows.items():
            tiles = []
            for k in ks:
                s = torch.as_tensor(ref_state[:1], dtype=torch.float32) + float(k) * float(sd) * torch.as_tensor(d, dtype=torch.float32)
                tiles.append(_tile(model.decode(s[:, :deter], s[:, deter:])[0], size))
            label = np.zeros((size, pad, 3), np.uint8)
            cv2.putText(label, name, (4, size // 2 + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            lines.append(np.concatenate([label] + tiles, 1))
    head = np.zeros((18, lines[0].shape[1], 3), np.uint8)
    for i, k in enumerate(ks):
        cv2.putText(head, f"{k:+g} sd", (pad + i * size + 6, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    return np.concatenate([head] + lines, 0)


RF = (4, 10, 22)      # receptive field (px) of the conv stages 1-3 (k4 s2 each)


def encoder_am(model, frames, per_layer=8, steps=60, layers=(1, 4, 7), size=96, tv_weights=(10.0, 60.0, 100.0)):
    """Activation maximisation: for the most variable channels of encoder stages 1-3 (ranked by the spread of their spatial mean over the dream
    frames), optimise an image (sigmoid-parametrised, total-variation regularised) to maximise the channel's response at the middle of its map and show
    the receptive field of that unit (4, 10, 22 px; upsampled). Rows = stages. The smoothness weight is per stage because one weight collapses stage 1
    to grey (weight 100) or leaves stage 3 as noise (weight 10); values picked by looking at L1's images."""
    net = model.encoder.net
    fr = frames.float() / 255 if frames.dtype == torch.uint8 else frames
    fr = fr.flatten(0, 1)[::max(1, len(fr.flatten(0, 1)) // 64)]
    if fr.shape[-1] != model.cfg.image_size:
        fr = F.interpolate(fr, size=(model.cfg.image_size,) * 2, mode="bilinear", align_corners=False)
    if model.cfg.channels == 1:
        fr = fr.mean(1, keepdim=True)
    rows = []
    for L in layers:
        with torch.no_grad():
            x = fr
            for i in range(L + 1):
                x = net[i](x)
            ch = torch.argsort(-x.mean((2, 3)).std(0))[:per_layer]
        img = torch.randn(len(ch), model.cfg.channels, model.cfg.image_size, model.cfg.image_size) * 0.1
        img.requires_grad_(True)
        opt = torch.optim.Adam([img], lr=0.1)
        rf, c = RF[layers.index(L)], model.cfg.image_size // 2
        crop = lambda t: t[..., c - rf // 2:c + rf // 2, c - rf // 2:c + rf // 2]
        for _ in range(steps):
            s = torch.sigmoid(img)
            x = s
            for i in range(L + 1):
                x = net[i](x)
            m = x.shape[-1] // 2
            act = x[torch.arange(len(ch)), ch, m, m]                            # the unit in the middle of the map: only its receptive field matters
            sc = crop(s)
            tv = (sc[..., 1:, :] - sc[..., :-1, :]).abs().mean() + (sc[..., 1:] - sc[..., :-1]).abs().mean()
            img.grad = torch.autograd.grad(-act.sum() + tv_weights[layers.index(L)] * tv, img)[0]       # gradient w.r.t. the image only: the model's weights stay untouched
            opt.step()
        rows.append(np.concatenate([_tile(crop(torch.sigmoid(img[i].detach())), size) for i in range(len(ch))], 1))
    w = max(r.shape[1] for r in rows)
    return np.concatenate([np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows], 0)


# ---------------------------------------------------------------------------------- comparative anatomy
def linear_cka(X, Y):
    """Linear CKA (Kornblith et al. 2019) between two feature matrices (N,dx), (N,dy) of the same N stimuli."""
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    num = np.linalg.norm(Y.T @ X) ** 2
    den = np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)
    return float(num / den) if den > 0 else float("nan")


@torch.no_grad()
def stimulus_features(model, clips, seed=0):
    """Feed identical clips (n,T,3,96,96 float) with zero action through a model: encoder embedding, deterministic state h, posterior mean of z (all (n*T, d))."""
    n, T = clips.shape[:2]
    torch.manual_seed(seed)
    x = clips if model.cfg.channels == 3 else clips.mean(2, keepdim=True)
    out = model.observe(x, torch.zeros(n, T, model.cfg.action_dim))
    emb = model.encoder(x.flatten(0, 1))
    zm = torch.stack([p[0] for p in out["posterior"]], 1)
    return dict(embed=emb.numpy().astype(np.float64), h=out["h"].flatten(0, 1).numpy().astype(np.float64), z=zm.flatten(0, 1).numpy().astype(np.float64))


# ---------------------------------------------------------------------------------- orchestration / cache
def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


def safe_id(model_id):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(model_id))


def lab_dir(model_id, root=None):
    return Path(root or ROOT) / "pilot_results" / "lab" / safe_id(model_id)


def analyse(path, seg, lc=None, progress=None, controls=None):
    """All sections of `lc.sections` for one checkpoint. Returns (results dict, images {name: uint8 RGB array}).
    controls: optional dict (arch key -> (random model, record, meas)) shared across calls to avoid re-dreaming the random-init network."""
    lc = lc or LabConfig()
    marks = []

    def say(stage, frac):
        marks.append((stage, time.time()))
        if progress:
            progress(stage, frac)

    t0 = time.time()
    model = load(path)
    cfg = model.cfg
    deter, Z = cfg.deter_dim, cfg.stoch_dim
    res = dict(version=VERSION, lab=asdict(lc), config=asdict(cfg))
    imgs = {}
    acts, walk_s = action_walk(lc.probe_worlds, lc.probe_ticks, lc.seed, cfg.action_dim)
    say("dream", 0.02)
    rec = dream(model, lc.probe_ticks, lc.probe_worlds, lc.seed, actions=acts)
    sel = probe_ticks(lc.probe_ticks, lc)
    meas = measure(rec["frames"][sel], seg)
    nat = _nat_sd(meas)
    res["nat_sd"] = nat
    X = _states(rec)[lc.burn:].reshape(-1, deter + Z)
    res["reference"] = dict(mean=X.mean(0).tolist(), sd=X.std(0).tolist())
    cap = capacity(rec, deter, lc.burn)
    if "capacity" in lc.sections:
        res["capacity"] = cap
    pr = None
    if "probes" in lc.sections or {"steer", "lesion", "images"} & set(lc.sections):
        say("probes", 0.1)
        key = (tuple(sorted(asdict(cfg).items())), lc.seed, lc.probe_worlds, lc.probe_ticks)
        if controls is not None and key in controls:
            rm, crec, cmeas = controls[key]
        else:
            rm = random_model(cfg, 0)
            crec = dream(rm, lc.probe_ticks, lc.probe_worlds, lc.seed, actions=acts)
            cmeas = measure(crec["frames"][sel], seg)
            if controls is not None:
                controls[key] = (rm, crec, cmeas)
        pr = probes(rec, meas, walk_s, deter, lc, ctrl=(crec, cmeas))
        res["probes"] = pr
    if "steer" in lc.sections:
        say("steer", 0.3)
        res["steer"] = steer(model, seg, pr, lc, nat)
    if "lesion" in lc.sections:
        say("lesion", 0.5)
        res["lesion"] = lesion(model, seg, pr, cap, lc, nat)
    if "patch" in lc.sections:
        say("patch", 0.7)
        res["patch"] = patch(model, seg, lc)
    if "dynamics" in lc.sections:
        say("dynamics", 0.8)
        res["dynamics"] = dynamics(model, seg, lc)
    if "images" in lc.sections and pr is not None:
        say("images", 0.93)
        dirs = {k: (np.asarray(v["dir"]), v["proj_sd"]) for k, v in pr.items() if "dir" in v}
        if dirs:
            t_ref, w_ref, found = eye_reference(meas, sel, lc.probe_ticks)
            imgs["filmstrip"] = filmstrip(model, dirs, _states(rec, "pred")[t_ref][w_ref][None])
            res["filmstrip_reference"] = dict(tick=t_ref, world=w_ref, found_eye=found)
        imgs["encoder_am"] = encoder_am(model, rec["frames"], lc.am_channels, lc.am_steps)
        imgs["dream"] = np.concatenate([_tile(as_float(rec["frames"][t, w]), 96) for w in range(min(4, lc.probe_worlds)) for t in (min(20, lc.probe_ticks - 1), lc.probe_ticks - 1)][:8], 1)
    marks.append(("end", time.time()))
    res["timing"] = {marks[i][0]: round(marks[i + 1][1] - marks[i][1], 1) for i in range(len(marks) - 1)}
    res["seconds"] = round(time.time() - t0, 1)
    return _clean(res), imgs


def save_results(model_id, nbytes, res, imgs, root=None):
    d = lab_dir(model_id, root)
    d.mkdir(parents=True, exist_ok=True)
    for name, im in imgs.items():
        cv2.imwrite(str(d / f"{name}.png"), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
    (d / "results.json").write_text(json.dumps(dict(model=str(model_id), bytes=int(nbytes), images=sorted(imgs), **res)))
    return d


def load_results(model_id, nbytes=None, root=None):
    f = lab_dir(model_id, root) / "results.json"
    if not f.exists():
        return None
    try:
        r = json.loads(f.read_text())
    except ValueError:
        return None
    return r if (nbytes is None or r.get("bytes") == nbytes) else None


def interventions_from_request(req, results, deter, stoch):
    """Live UI edit -> Intervention for the running world, or None. req: {} / {'clear': True} -> None;
    {'lesion': {'space': 'h'|'z'|'hz', 'idx': [..] (indices within the space) or 'group': name, 'value': 'mean'|'zero'}, 'steer': {'target': name, 'alpha': float}}.
    Needs `results` (the cached lab run of this model) for the reference means, probe directions and named groups."""
    if not req or req.get("clear"):
        return None
    D = deter + stoch
    kw = {}
    les = req.get("lesion")
    if les:
        if "group" in les:
            if not results or les["group"] not in (results.get("lesion") or {}).get("groups", {}):
                raise ValueError(f"unknown lesion group {les['group']!r}: run the lab for this model first")
            idx = _group_units(les["group"], results, deter, stoch)
        else:
            off = {"h": 0, "z": deter, "hz": 0}[les.get("space", "z")]
            top = {"h": deter, "z": stoch, "hz": D}[les.get("space", "z")]
            idx = [off + int(i) for i in les["idx"]]
            if any(i < off or i >= off + top for i in idx):
                raise ValueError("unit index out of range")
        mode = les.get("value", "mean")
        if mode == "mean":
            if not results or "reference" not in results:
                raise ValueError("'mean' lesions need the reference means: run the lab for this model first (or use value 'zero')")
            val = np.asarray(results["reference"]["mean"], dtype=np.float32)[idx]
        elif mode == "zero":
            val = np.zeros(len(idx), np.float32)
        else:
            raise ValueError("lesion value must be 'mean' or 'zero'")
        kw.update(clamp_idx=idx, clamp_val=val)
    st = req.get("steer")
    if st:
        p = ((results or {}).get("probes") or {}).get(st.get("target"), {})
        if "dir" not in p:
            raise ValueError(f"no probe direction for {st.get('target')!r}: run the lab for this model first (the probe must beat chance)")
        kw["add"] = float(st.get("alpha", 1.0)) * p["proj_sd"] * np.asarray(p["dir"], dtype=np.float32)
        kw["stages"] = ("pred",) if not les else ("pred", "post")
    if not kw:
        return None
    return Intervention(deter, **kw)


def _group_units(name, results, deter, stoch):
    """Unit indices (into [h|z]) of a named lesion group of a stored lab run."""
    cap = results.get("capacity") or {}
    kl = np.asarray(cap.get("kl_unit", []))
    if name == "z_active":
        return [deter + int(i) for i in np.flatnonzero(kl > KL_ACTIVE)]
    if name == "z_inactive":
        return [deter + int(i) for i in np.flatnonzero(kl <= KL_ACTIVE)]
    if name == "h_all":
        return list(range(deter))
    if name == "z_all":
        return list(range(deter, deter + stoch))
    m = re.fullmatch(r"top_(\w+)", name)
    if m:
        tu = results["probes"][m.group(1)]["top_units"]
        return [u["unit"] for u in tu[:int(results["lab"]["group_size"])]]
    m = re.fullmatch(r"([hz])(\d+)", name)
    if m:
        return [(0 if m.group(1) == "h" else deter) + int(m.group(2))]
    m = re.fullmatch(r"random(\d+)_(\d+)", name)
    if m:
        rng = np.random.RandomState(int(results["lab"]["seed"]))
        out = None
        for _ in range(int(m.group(2)) + 1):
            out = [int(u) for u in rng.choice(deter + stoch, int(m.group(1)), replace=False)]
        return out
    raise ValueError(f"unknown lesion group {name!r}")
