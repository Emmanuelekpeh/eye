"""Pupil/iris-aware evaluation on the S_ (OpenEDS-style) sequences. MSE is dominated by the background and hides a missing pupil,
so this module scores the eye itself. Everything uses REAL data: the aligned mask cache written by
`scripts/build_cache_openeds.py --masks` (0 bg, 1 sclera, 2 iris, 3 pupil, 255 unknown).

Layer 1 - ground-truth masks applied to a reconstruction (no learned component):
    per-class MSE, pupil/iris contrast (recon vs real), edge softness (Sobel strength in a +-2 px band around the pupil edge and the
    limbus, recon/real; 1 = as sharp as real, ~0.3 = blurry), band MSE.
Layer 2 - `EyeSegmenter` (small U-Net trained on REAL train-group frames with blur/noise/exposure augmentation, so it can read soft
    recon): segmentation IoU of the recon, pupil found / hallucinated / centroid error / area ratio, and `structure_stats` for frames that
    have no ground truth (dreams): pupil present, iris present, pupil nested in iris, relative pupil size.
The segmenter's own error is measured on real val frames (`segmenter_ceiling`) and on blurred real frames; read layer 2 against that.
Train/val membership is the CacheBank group split (per source, deterministic), so a model's held-out S_ subjects are the segmenter's too.
"""
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cachedata import split_groups_stratified
from .gaze import MIN_IRIS_PX, MIN_PUPIL_PX      # noqa: F401  (single definition of "pupil/iris present")

CLASSES = ("background", "sclera", "iris", "pupil")
UNKNOWN = 255
BAND_K = 5                 # edge band = dilate(k) minus erode(k): +-2 px around the boundary


# ----------------------------------------------------------------------------------------------- aligned data
def aligned_masks(bank, masks_dir):
    """(N,size,size) uint8 masks for every frame of a CacheBank, same order and same centre crop as bank.sample(train=False).
    255 for frames of non-S_ sources or without a mask file."""
    size = bank.size
    off = (bank.store_w - size) // 2
    out = []
    for v in bank.videos:
        n = v["frames"]
        f = Path(masks_dir) / (v["group"] + ".npz")
        if v["source"] != "openeds" or not f.exists():
            out.append(np.full((n, size, size), UNKNOWN, np.uint8))
            continue
        with np.load(f) as z:
            m = z["masks"][:n]
        w = int(round(m.shape[2] * size / 96.0))
        if m.shape[1] != size:
            m = np.stack([cv2.resize(x, (w, size), interpolation=cv2.INTER_NEAREST) for x in m])
        if w >= bank.store_w:
            x0 = (w - bank.store_w) // 2
            m = m[:, :, x0:x0 + bank.store_w]
        else:
            pad = bank.store_w - w
            m = np.pad(m, ((0, 0), (0, 0), (pad // 2, pad - pad // 2)), mode="edge")
        out.append(np.ascontiguousarray(m[:, :, off:off + size]))
    return np.concatenate(out)


def eval_windows(bank, masks, n, t, seed=0, need_pupil=False):
    """n windows of t consecutive frames; returns (x (n,t,3,H,W) float, mask of the LAST frame (n,H,W), end indices).
    Deterministic for a seed. need_pupil keeps only windows whose last frame has a visible ground-truth pupil."""
    rng = np.random.RandomState(seed)
    ends = []
    for s, ln in zip(bank.seq_start, bank.seq_len):
        ends.extend(range(int(s) + t - 1, int(s) + int(ln)))
    ends = np.array(ends)
    ends = ends[(masks[ends] != UNKNOWN).any((1, 2))]
    if need_pupil:
        ends = ends[((masks[ends] == 3).sum((1, 2)) >= MIN_PUPIL_PX)]
    if len(ends) < n:
        raise RuntimeError(f"only {len(ends)} eligible windows for n={n}, t={t}")
    pick = np.sort(rng.choice(ends, n, replace=False))
    idx = pick[:, None] - np.arange(t - 1, -1, -1)[None, :]
    off = (bank.store_w - bank.size) // 2
    fr = bank.frames[torch.from_numpy(idx)][:, :, :, off:off + bank.size]          # n,t,H,W,3 uint8
    x = fr.permute(0, 1, 4, 2, 3).float() / 255.0
    return x, torch.from_numpy(masks[pick]), pick


# ----------------------------------------------------------------------------------------------- models -> recon functions
def load_recon_fn(kind, path, image_size=96):
    """kind 'wm' (EyeWorldModel checkpoint, observes the window with zero action, returns the LAST frame) or 'ae' (PlainAE probe,
    last frame only; the checkpoint stores no image size, so pass it). Returns fn(x (B,T,3,H,W)) -> (B,3,H,W)."""
    ck = torch.load(path, map_location="cpu")
    if kind == "ae":
        from .aeprobe import PlainAE
        spec = ck["spec"]
        m = PlainAE(image_size, spec["code"], spec.get("conv_mult", 1.0), vae=spec.get("vae", 0.0)); m.load_state_dict(ck["model"]); m.eval()
        return lambda x: m(x[:, -1]).detach()
    from .config import ModelConfig
    from .model import EyeWorldModel
    cfg = ModelConfig(**ck["config"])
    m = EyeWorldModel(cfg); m.load_state_dict(ck["model"]); m.eval()

    @torch.no_grad()
    def fn(x):
        xin = x if cfg.channels == 3 else x[:, :, :1]
        torch.manual_seed(0)
        r = m.observe(xin, torch.zeros(x.shape[0], x.shape[1], cfg.action_dim))["recon"][:, -1]
        return r if r.shape[1] == 3 else r.expand(-1, 3, -1, -1)
    fn.model, fn.cfg = m, cfg
    return fn


@torch.no_grad()
def dream_frames(model, times=(0, 10, 30, 100, 300), n=64, seed=13, seed_steps=4, temperature=1.0):
    """The UI's start: sample the prior from the zero state (`seed_steps` zero-action steps = ivs.pilot.SEED_STEPS), then free-run
    with zero action. Returns {t: (n,3,H,W)} (same procedure as ivs.pilot.eval_noise_seeds). temperature scales the prior noise
    of every step (the start-up steps stay at 1 so all temperatures begin from the same kind of state)."""
    model.eval()
    dev = next(model.parameters()).device
    torch.manual_seed(seed)
    h, z = model.rssm.initial(n, dev)
    zero = torch.zeros(n, model.cfg.action_dim, device=dev)
    for _ in range(seed_steps):
        h, z, _ = model.rssm.imagine_step(h, z, zero)
    out = {0: model.decode(h, z)}
    for t in range(1, max(times) + 1):
        h, z, _ = model.rssm.imagine_step(h, z, zero, temperature)
        if t in times:
            out[t] = model.decode(h, z)
    return {t: (f if f.shape[1] == 3 else f.expand(-1, 3, -1, -1)) for t, f in out.items() if t in times}


@torch.no_grad()
def ui_dream_frames(model, ticks=1200, n=4, noise_seed=21, seed_steps=4, init_state=None):
    """What the UI shows in DREAM mode (ivs.ui.World.tick with dream=True, no touch, no gaze command), without the server: the world is
    seeded from the prior (zero state, `seed_steps` zero-action steps), then every tick the model PREDICTS, RENDERS the prediction, PERCEIVES
    that render (posterior update of h,z) and shows the render (the page's `stim`). Seeds follow the UI (seed = noise_seed*1000003 + tick*7919,
    perceive uses seed+1), so n=1 reproduces World exactly (tests/test_pilot.py); n>1 runs n independent worlds as a batch (same noise
    streams per tick, different draws per row). Returns (ticks, n, 3, H, W) float in [0,1], quantised to 8 bit like the page's PNG.
    init_state=(h, z) with batch n starts every world from that latent (a birth memory, ivs.birth) instead of from noise."""
    model.eval()
    zero = torch.zeros(n, model.cfg.action_dim)
    torch.manual_seed(int(noise_seed) & 0x7FFFFFFF)
    if init_state is not None:
        h, z = init_state[0].clone(), init_state[1].clone()
        if len(h) != n or len(z) != n:
            raise ValueError(f"init_state holds {len(h)} states but n={n} worlds were asked for")
    else:
        h, z = model.rssm.initial(n, "cpu")
        for _ in range(seed_steps):
            h, z, _ = model.rssm.imagine_step(h, z, zero)
    out = []
    for t in range(ticks):
        seed = (int(noise_seed) * 1000003 + t * 7919) & 0x7FFFFFFF
        torch.manual_seed(seed)
        ph, pz, _ = model.rssm.imagine_step(h, z, zero)
        pred = model.decode(ph, pz)
        torch.manual_seed(seed + 1)
        h, z, _, _ = model.rssm.observe_step(h, z, zero, model.encoder(pred))
        q = (pred.clamp(0, 1) * 255).round() / 255
        out.append(q if q.shape[1] == 3 else q.expand(-1, 3, -1, -1))
    return torch.stack(out)


def eye_episodes(flags, smooth=5, min_len=20):
    """Persistence of 'looks like an eye' along one dream. flags: (T,) bool per tick (nested_flags). Median-`smooth` filter (drops one-tick
    flicker), then runs of True. Returns eye_frac (share of ticks), n_episodes, median/max length, n_long (>= min_len ticks)."""
    f = np.asarray(flags, dtype=int)
    k = smooth // 2
    s = np.array([int(np.median(f[max(0, i - k):i + k + 1])) for i in range(len(f))])
    runs, cur = [], 0
    for v in s:
        if v:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    return dict(eye_frac=float(s.mean()), n_episodes=len(runs), median_len=float(np.median(runs)) if runs else 0.0,
                max_len=int(max(runs)) if runs else 0, n_long=int(sum(r >= min_len for r in runs)))


def gray(x):
    """(…,3,H,W) -> (…,H,W) BT.601 luma; (…,1,H,W) -> (…,H,W)."""
    if x.shape[-3] == 1:
        return x[..., 0, :, :]
    return 0.299 * x[..., 0, :, :] + 0.587 * x[..., 1, :, :] + 0.114 * x[..., 2, :, :]


def gaussian_blur(x, sigma):
    """x (N,H,W) or (N,1,H,W); separable Gaussian, replicate padding."""
    sq = x.dim() == 3
    if sq:
        x = x[:, None]
    if sigma <= 0:
        return x[:, 0] if sq else x
    r = max(1, int(math.ceil(3 * sigma)))
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1).float() / sigma) ** 2); k = (k / k.sum()).to(x.device)
    y = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    y = F.conv2d(F.pad(y, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1))
    return y[:, 0] if sq else y


# ----------------------------------------------------------------------------------------------- layer 1 (ground-truth masks)
_SOBEL = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])


def grad_mag(g):
    """Sobel gradient magnitude of a gray batch (N,H,W)."""
    gx = F.conv2d(g[:, None], _SOBEL.view(1, 1, 3, 3), padding=1)
    gy = F.conv2d(g[:, None], _SOBEL.t().contiguous().view(1, 1, 3, 3), padding=1)
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-12)[:, 0]


def edge_band(region):
    """(N,H,W) bool -> bool band of +-2 px around the region boundary."""
    r = region.float()[:, None]
    dil = F.max_pool2d(r, BAND_K, 1, BAND_K // 2)
    ero = -F.max_pool2d(-r, BAND_K, 1, BAND_K // 2)
    return ((dil - ero) > 0)[:, 0]


def _masked_mean(v, m):
    """Per-frame mean of v over bool mask m (N,H,W); NaN where the mask is empty."""
    n = m.sum((1, 2)).float()
    return torch.where(n > 0, (v * m).sum((1, 2)) / n.clamp(min=1), torch.full_like(n, float("nan")))


def gt_metrics(real, rec, mask):
    """real, rec: (N,H,W) float gray in [0,1]; mask (N,H,W) uint8. Pixel/frame-weighted summary (python floats / None)."""
    known = mask != UNKNOWN
    se = (rec - real) ** 2
    res = {"n_frames": int(len(real))}
    for c, name in enumerate(CLASSES):
        m = (mask == c)
        res[f"mse_{name}"] = float(se[m].mean()) if m.any() else None
    res["mse_known"] = float(se[known].mean()) if known.any() else None
    pupil, iris_all = mask == 3, (mask == 2) | (mask == 3)
    has = (pupil.sum((1, 2)) >= MIN_PUPIL_PX) & ((mask == 2).sum((1, 2)) >= MIN_PUPIL_PX)
    res["n_pupil_frames"] = int(has.sum())
    if has.any():
        iris_ring = mask == 2
        c_real = _masked_mean(real, iris_ring) - _masked_mean(real, pupil)
        c_rec = _masked_mean(rec, iris_ring) - _masked_mean(rec, pupil)
        ok = has & (c_real > 0.02)
        res["pupil_contrast_real"] = float(c_real[ok].mean()) if ok.any() else None
        res["pupil_contrast_recon"] = float(c_rec[ok].mean()) if ok.any() else None
        if ok.any():
            ratio = c_rec[ok] / c_real[ok]
            res["pupil_contrast_ratio_median"] = float(ratio.median())
            res["frac_pupil_washed_out"] = float((ratio < 0.5).float().mean())    # less than half the real pupil/iris contrast
        gr, gc = grad_mag(real), grad_mag(rec)
        for name, region in (("pupil_edge", pupil), ("limbus", iris_all)):
            band = edge_band(region) & has[:, None, None]
            res[f"{name}_grad_ratio"] = float((gc * band).sum() / (gr * band).sum().clamp(min=1e-9))
            res[f"{name}_band_mse"] = float(se[band].mean())
    return res


# ----------------------------------------------------------------------------------------------- layer 2 (segmenter)
def _blk(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(4, o), nn.ReLU(True),
                         nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(4, o), nn.ReLU(True))


class EyeSegmenter(nn.Module):
    """Gray (N,1,H,W) in [0,1] -> 4-class logits (N,4,H,W); H and W divisible by 8."""

    def __init__(self, base=16):
        super().__init__()
        b = base
        self.e1, self.e2, self.e3, self.mid = _blk(1, b), _blk(b, 2 * b), _blk(2 * b, 4 * b), _blk(4 * b, 4 * b)
        self.d3, self.d2, self.d1 = _blk(8 * b, 2 * b), _blk(4 * b, b), _blk(2 * b, b)
        self.out = nn.Conv2d(b, len(CLASSES), 1)
        self.base = base

    def forward(self, x):
        x = x - 0.5
        e1 = self.e1(x); e2 = self.e2(F.max_pool2d(e1, 2)); e3 = self.e3(F.max_pool2d(e2, 2)); m = self.mid(F.max_pool2d(e3, 2))
        up = lambda t: F.interpolate(t, scale_factor=2, mode="bilinear", align_corners=False)
        d3 = self.d3(torch.cat([up(m), e3], 1)); d2 = self.d2(torch.cat([up(d3), e2], 1)); d1 = self.d1(torch.cat([up(d2), e1], 1))
        return self.out(d1)


@torch.no_grad()
def segment(seg, g, chunk=128):
    """g (N,H,W) float gray -> (N,H,W) uint8 class ids."""
    seg.eval()
    return torch.cat([seg(g[i:i + chunk, None]).argmax(1) for i in range(0, len(g), chunk)]).to(torch.uint8)


def load_masked_gray(cache_dir, masks_dir, val_frac=0.15, seed=0):
    """Real gray frames + aligned masks for every S_ group, split exactly like CacheBank (per source, deterministic).
    Returns {'train': (G,M), 'val': (G,M)}, each (N,96,128) uint8 (full stored width; crop at use)."""
    cache_dir, masks_dir = Path(cache_dir), Path(masks_dir)
    metas = [m for m in json.loads((cache_dir / "manifest.json").read_text()) if m.get("source", "video") == "openeds"]
    val = split_groups_stratified(metas, val_frac, seed)
    out = {"train": ([], []), "val": ([], [])}
    for m in sorted(metas, key=lambda m: m["name"]):
        with np.load(cache_dir / (m["group"] + ".npz")) as z:
            fr = z["frames"][..., 0]
        with np.load(masks_dir / (m["group"] + ".npz")) as z:
            mk = z["masks"]
        assert fr.shape == mk.shape, (m["group"], fr.shape, mk.shape)
        k = "val" if m["group"] in val else "train"
        out[k][0].append(fr); out[k][1].append(mk)
    return {k: (np.concatenate(v[0]), np.concatenate(v[1])) for k, v in out.items()}


def _augment(g, gen):
    """g (B,H,W) float. Blur sigma in {0,.8,1.5,2.5} per chunk, exposure/contrast, noise: so soft recon is readable."""
    b = len(g)
    sig = [0.0, 0.8, 1.5, 2.5]
    parts = [gaussian_blur(g[i::4], sig[i]) for i in range(4)]
    out = torch.empty_like(g)
    for i in range(4):
        out[i::4] = parts[i]
    gain = torch.exp((torch.rand(b, 1, 1, generator=gen) - 0.5) * 0.8)
    bias = (torch.rand(b, 1, 1, generator=gen) - 0.5) * 0.2
    out = (out - 0.5) * gain + 0.5 + bias
    out = out + torch.randn(out.shape, generator=gen) * 0.02 * torch.rand(b, 1, 1, generator=gen)
    return out.clamp(0, 1)


def train_segmenter(data, steps=2500, batch=32, lr=2e-3, base=16, seed=0, log=print):
    """data from load_masked_gray. Random 96-wide crop + h-flip; CE with a heavier pupil weight; masks 255 ignored."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    G, M = (torch.from_numpy(a) for a in data["train"])
    seg = EyeSegmenter(base)
    opt = torch.optim.Adam(seg.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    w = torch.tensor([0.5, 1.0, 1.0, 3.0])
    for step in range(steps):
        i = torch.randint(0, len(G), (batch,), generator=gen)
        x0 = int(torch.randint(0, G.shape[2] - 96 + 1, (1,), generator=gen))
        g = G[i][:, :, x0:x0 + 96].float() / 255.0
        m = M[i][:, :, x0:x0 + 96].long()
        flip = torch.rand(batch, generator=gen) < 0.5
        g = torch.where(flip[:, None, None], g.flip(2), g); m = torch.where(flip[:, None, None], m.flip(2), m)
        loss = F.cross_entropy(seg(_augment(g, gen)[:, None]), m, weight=w, ignore_index=UNKNOWN)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 250 == 0 or step == steps - 1:
            log(f"[segmenter] step {step} loss {float(loss):.4f}")
    seg.eval()
    return seg


def _centroid(m):
    """(N,H,W) bool -> (N,2) x,y centroid (NaN if empty)."""
    n = m.sum((1, 2)).float()
    ys = torch.arange(m.shape[1]).float().view(1, -1, 1); xs = torch.arange(m.shape[2]).float().view(1, 1, -1)
    cx = (m * xs).sum((1, 2)) / n; cy = (m * ys).sum((1, 2)) / n
    return torch.stack([cx, cy], 1)


def seg_metrics(pred, mask):
    """pred, mask (N,H,W) uint8 class ids (mask 255 ignored). IoU per class, pupil found/hallucinated, centroid error, area ratio."""
    known = mask != UNKNOWN
    res = {}
    for c, name in enumerate(CLASSES):
        p, g = (pred == c) & known, (mask == c)
        u = (p | g).sum()
        res[f"iou_{name}"] = float((p & g).sum() / u) if u > 0 else None
    pp, gp = (pred == 3) & known, (mask == 3)
    has_g, has_p = gp.sum((1, 2)) >= MIN_PUPIL_PX, pp.sum((1, 2)) >= MIN_PUPIL_PX
    if has_g.any():
        res["pupil_found_rate"] = float((has_g & has_p).sum() / has_g.sum())
        both = has_g & has_p
        if both.any():
            err = (_centroid(pp)[both] - _centroid(gp)[both]).norm(dim=1)
            res["pupil_centroid_err_px_median"] = float(err.median())
            res["pupil_area_ratio_median"] = float((pp.sum((1, 2)).float()[both] / gp.sum((1, 2)).float()[both]).median())
    res["pupil_hallucinated_rate"] = float((~has_g & has_p).sum() / max(1, (~has_g).sum()))     # pupil predicted where none exists
    return res


def nested_flags(pred):
    """pred (N,H,W) class ids -> (N,) bool: pupil and iris both present and the pupil centroid within half an iris radius of the iris centroid.
    LENIENT: a blob the segmenter labels pupil-in-iris counts (look at the frames next to the numbers)."""
    pupil, iris = pred == 3, (pred == 2) | (pred == 3)
    n_p, n_i = pupil.sum((1, 2)).float(), iris.sum((1, 2)).float()
    d = (_centroid(pupil) - _centroid(iris)).norm(dim=1)
    return (n_p >= MIN_PUPIL_PX) & (n_i >= MIN_IRIS_PX) & (d < 0.5 * torch.sqrt(n_i / math.pi))


def structure_stats(pred):
    """For frames WITHOUT ground truth (dreams): does it look like an eye? pred (N,H,W) class ids.
    nested_eye = nested_flags (pupil and iris both present and the pupil centroid lies within half an iris radius of the iris centroid)."""
    pupil, iris = pred == 3, (pred == 2) | (pred == 3)
    n_p, n_i = pupil.sum((1, 2)).float(), iris.sum((1, 2)).float()
    has_p, has_i = n_p >= MIN_PUPIL_PX, n_i >= MIN_IRIS_PX
    nested = nested_flags(pred)
    both = has_p & has_i
    return dict(n=int(len(pred)), pupil_present=float(has_p.float().mean()), iris_present=float(has_i.float().mean()),
                nested_eye=float(nested.float().mean()),
                pupil_area_median=float(n_p[has_p].median()) if has_p.any() else None,
                iris_area_median=float(n_i[has_i].median()) if has_i.any() else None,
                pupil_to_iris_radius_median=float(torch.sqrt(n_p[both] / n_i[both]).median()) if both.any() else None)


@torch.no_grad()
def command_response(model, seg, x_ctx, ctx_cmd, ctx_has, delta=0.15, horizon=16, steps=(1, 4, 8, 16), seed=0, frame_step=8):
    """Does the gaze command control the dream? For a model with the 8-dim action (ivs.pilot gaze_command): observe the real context
    frames with their own commands, then imagine `horizon` steps with the command held at (a) the last real target, (b) shifted by
    +-delta in x / y (image units, [-1,1]), (c) no command. Identical noise per condition, so the only difference is the command.
    The pupil is located in the imagined frames by the segmenter.
      gain_x = median over windows of (pupil x under +delta  -  pupil x under -delta) / (2*delta*size/2 px):  1 = obeys exactly, 0 = ignores,
      <0 = moves the wrong way; sign_ok = fraction of windows moving the right way; same for y.
      track_px = median |pupil position under the held real target - target| in px (how well the commanded position is reached);
      none_track_px = the same with NO command (what the model does by itself), so the gap is the command's effect.
    x_ctx (n,ctx,3,H,W), ctx_cmd (n,ctx,2) / ctx_has (n,ctx) = real commands of the context (image units).
    The result also carries "_shots": {condition: imagined frames at `frame_step` (n,3,H,W)} for the sheet (not JSON)."""
    cfg = model.cfg
    if cfg.action_dim < 8:
        return {"skipped": f"action_dim {cfg.action_dim}: model has no gaze command"}
    model.eval()
    steps = tuple(s for s in steps if s <= horizon)
    frame_step = min(frame_step, horizon)
    n, ctx = x_ctx.shape[:2]
    half = x_ctx.shape[-1] / 2.0
    xin = x_ctx if cfg.channels == 3 else x_ctx[:, :, :1]
    a = torch.zeros(n, ctx, cfg.action_dim)
    a[..., 5] = ctx_has.float(); a[..., 6:8] = ctx_cmd
    torch.manual_seed(seed)
    out = model.observe(xin, a)
    h, z = out["h"][:, -1], out["z"][:, -1]
    target = ctx_cmd[:, -1]
    ok_target = ctx_has[:, -1]

    shots = {}

    def pupil(cmd, name):                                 # cmd (n,2) or None -> {step: (x_norm, y_norm, found)}
        act = torch.zeros(n, horizon, cfg.action_dim)
        if cmd is not None:
            act[..., 5] = 1.0; act[..., 6:8] = cmd[:, None, :]
        torch.manual_seed(seed + 1)
        fr, _ = model.imagine(h, z, act)
        shots[name] = fr[:, frame_step - 1] if fr.shape[2] == 3 else fr[:, frame_step - 1].expand(-1, 3, -1, -1)
        res = {}
        for s in steps:
            f = fr[:, s - 1]
            f = f if f.shape[1] == 3 else f.expand(-1, 3, -1, -1)
            pred = segment(seg, gray(f))
            p = pred == 3
            area = p.sum((1, 2))
            xy = _centroid(p)
            res[s] = (((xy[:, 0] - (p.shape[2] - 1) / 2) / half), ((xy[:, 1] - (p.shape[1] - 1) / 2) / half), area >= MIN_PUPIL_PX)
        return res

    d = torch.tensor([delta, 0.0]); dy = torch.tensor([0.0, delta])
    conds = {"hold": pupil(target, "hold"), "px": pupil(target + d, "px"), "mx": pupil(target - d, "mx"),
             "py": pupil(target + dy, "py"), "my": pupil(target - dy, "my"), "none": pupil(None, "none")}
    res = {"n": int(ok_target.sum()), "delta": delta, "by_step": {}, "_shots": shots}
    for s in steps:
        both = lambda a_, b_: ok_target & conds[a_][s][2] & conds[b_][s][2]
        row = {}
        for axis, (p_, m_, k) in (("x", ("px", "mx", 0)), ("y", ("py", "my", 1))):
            ok = both(p_, m_)
            if ok.any():
                disp = (conds[p_][s][k] - conds[m_][s][k])[ok] * half                    # px
                row[f"gain_{axis}"] = float(disp.median() / (2 * delta * half))
                row[f"sign_ok_{axis}"] = float((disp > 0).float().mean())
                row[f"n_{axis}"] = int(ok.sum())
        for name in ("hold", "none"):
            okk = ok_target & conds[name][s][2]
            if okk.any():
                err = torch.hypot(conds[name][s][0] - target[:, 0], conds[name][s][1] - target[:, 1])[okk] * half
                row[f"{'track_px' if name == 'hold' else 'none_track_px'}"] = float(err.median())
        row["pupil_found"] = {k: float((ok_target & conds[k][s][2]).float().sum() / max(1, int(ok_target.sum()))) for k in conds}
        res["by_step"][s] = row
    return res


def segmenter_ceiling(seg, g_val, m_val, n=600, seed=0):
    """Segmenter quality on REAL val frames (centre 96 columns), at several blur levels: the bound for any layer-2 number."""
    rng = np.random.RandomState(seed)
    i = np.sort(rng.choice(len(g_val), min(n, len(g_val)), replace=False))
    g = torch.from_numpy(g_val[i][:, :, 16:112]).float() / 255.0
    m = torch.from_numpy(m_val[i][:, :, 16:112])
    out = {}
    for s in (0.0, 1.0, 2.0):
        out[f"blur_sigma_{s}"] = seg_metrics(segment(seg, gaussian_blur(g, s)), m)
    return out


def save_segmenter(seg, path, extra=None):
    torch.save({"model": seg.state_dict(), "base": seg.base, **(extra or {})}, path)


def load_segmenter(path):
    ck = torch.load(path, map_location="cpu")
    seg = EyeSegmenter(ck.get("base", 16)); seg.load_state_dict(ck["model"]); seg.eval()
    return seg


PALETTE = np.array([[0, 0, 0], [140, 140, 140], [40, 90, 255], [255, 60, 40]], np.uint8)     # bg, sclera, iris, pupil


def colour_mask(m):
    """(H,W) uint8 class ids -> (3,H,W) float tensor; 255 shown as mid-yellow."""
    img = np.zeros(m.shape + (3,), np.uint8)
    for c in range(4):
        img[m == c] = PALETTE[c]
    img[m == UNKNOWN] = (200, 200, 0)
    return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
