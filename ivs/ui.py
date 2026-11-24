"""IVS touch/observer UI server.  python -m ivs.ui --checkpoint checkpoints/latest.pt

Stdlib HTTP server + torch/PIL/numpy. Single local user, one lock around all model work. No dataset is
read: the only pixels in the loop are the trained model's own render.

The CANVAS is the model's own rendered eye. It starts from noise (the prior sampled from a zero state) or, with --birth, from the
model's birth memory: its own latent state after it perceived a real eye once offline (ivs.birth, scripts/make_birth.py).
The person touches it: shift / exposure / light go through the same hidden-intervention transform the
model was trained on, and SCATTER shuffles pixels where the person grabs and shakes. The model never sees
the intervention state, only the resulting pixels (watching) and the per-step action deltas, which are
rate-limited like training. After every tick the canvas is overwritten by the model's own belief (a blend
of what it perceived and what it expected), with the known touch stripped off so it is not compounded.
A frozen copy of the checkpoint runs on identical noise and input as the control.

Modes. Watching: the model perceives the touched canvas. Dream: the touch only reaches the model as
actions; each model perceives its own render (scattered if you scatter), so dissonance (prior vs posterior)
is what it feels.

Learning (off by default): trains the RSSM only; encoder and decoder stay frozen. The signal is
dissonance on what it perceived, plus pseudo-rehearsal: the live dynamics are also pulled toward the
frozen copy's imagination under random actions. Retention = distance to the frozen copy's imagination on a
fixed probe.
"""
import argparse
import base64
import collections
import copy
import hashlib
import io
import json
import random
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from .actions import InterventionSampler
from . import lab, modelzoo
from .birth import load_birth
from .config import ModelConfig, TrainConfig
from .learning import PlasticityController, gaussian_kl, _pixel_sum_loss
from .model import EyeWorldModel
from .train import compute_loss

WEB = Path(__file__).parent / "web" / "index.html"
SEQ_T = 8
SEED_STEPS = 4          # zero-action imagination steps used to seed the canvas from noise
SCATTER_RADIUS = 0.18   # fraction of the box width
MAX_SCATTER_PTS = 64    # per tick


def png(arr_u8):
    """(H,W) -> grayscale PNG, (H,W,3) -> RGB PNG."""
    buf = io.BytesIO()
    Image.fromarray(arr_u8, "RGB" if arr_u8.ndim == 3 else "L").save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def u8(t):
    """(H,W) or (C,H,W) float in [0,1] -> uint8 (H,W) for 1 channel, (H,W,3) for 3."""
    a = (t.detach().clamp(0, 1) * 255).round().byte().cpu().numpy()
    if a.ndim == 3:
        a = a[0] if a.shape[0] == 1 else a.transpose(1, 2, 0)
    return a


def scatter_pixels(img, pts, radius, seed):
    """Shuffle pixels inside a disc around each (x, y, strength) point (box coords 0..1, strength 0..1).
    Strength is the fraction of the disc's pixels that get permuted among themselves. img: 1,C,H,W (any C: a pixel moves with all its
    channels). Pixel values are only moved, never invented, so the multiset of values is preserved."""
    if not pts:
        return img
    out = img.clone()
    _, c, h, w = out.shape
    flat = out.view(1, c, h * w)
    gen = torch.Generator().manual_seed(int(seed) & 0x7FFFFFFF)
    xs = ((torch.arange(w) + 0.5) / w).view(1, w)
    ys = ((torch.arange(h) + 0.5) / h).view(h, 1)
    for x, y, k in pts[-MAX_SCATTER_PTS:]:
        k = min(1.0, max(0.0, float(k)))
        if k <= 0:
            continue
        idx = (((xs - float(x)) ** 2 + (ys - float(y)) ** 2) <= radius ** 2).flatten().nonzero().squeeze(1)
        if idx.numel() < 2:
            continue
        chosen = idx[torch.rand(idx.numel(), generator=gen) < k]
        if chosen.numel() < 2:
            continue
        flat[:, :, chosen] = flat[:, :, chosen[torch.randperm(chosen.numel(), generator=gen)]]
    return out


class World:
    def __init__(self, a):
        self.lock = threading.RLock()
        self.a = a
        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
        if a.birth and not Path(a.checkpoint).exists():
            raise ValueError("--birth needs a trained checkpoint: a birth memory is that model's own latent state")
        self.mods = lambda m: {"encoder": m.encoder, "rssm": m.rssm, "decoder": m.decoder}
        self.lab_job, self.lab_thread = None, None
        cfg, trained, model, frozen = self._build(a.checkpoint, a.image_size)
        self._install(a.checkpoint, cfg, trained, model, frozen, load_birth(a.birth, a.checkpoint, cfg) if a.birth else None)   # None = start from noise
        self.sampler = InterventionSampler(p=1.0)
        self.lim = self.sampler.limit.numpy().astype(np.float32)
        self.unit = (self.sampler.step * self.lim / self.lim[0]).astype(np.float32)
        d = TrainConfig()
        self.targs = SimpleNamespace(kl_weight=d.kl_weight, free_nats=d.free_nats,
                                     kl_balance=d.kl_balance, pred_weight=d.pred_weight)
        self.snapshots, self._snap_id = [], 0
        self.world_file = Path(a.world)
        self.noise_seed = a.seed if a.seed is not None else random.randrange(2 ** 30)
        self._reset_runtime(load_world=not a.fresh)

    # ---- model (re)loading ------------------------------------------------------------
    @staticmethod
    def _build(checkpoint, image_size=64):
        """(cfg, trained, model, frozen) for a checkpoint path. Touches nothing on a running world, so a bad file raises before any state changes."""
        ck, cfg = None, ModelConfig(image_size=image_size)
        if Path(checkpoint).exists():
            ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
            cfg = ModelConfig(**ck["config"])
        model = EyeWorldModel(cfg)
        if ck:
            model.load_state_dict(ck["model"])
        model.eval()
        frozen = copy.deepcopy(model).eval()
        for p in frozen.parameters():
            p.requires_grad_(False)
        for m in (model.encoder, model.decoder):    # perception and rendering never change here
            for p in m.parameters():
                p.requires_grad_(False)
        return cfg, ck is not None, model, frozen

    def _install(self, checkpoint, cfg, trained, model, frozen, birth):
        self.checkpoint, self.cfg, self.trained, self.model, self.frozen, self.birth = checkpoint, cfg, trained, model, frozen, birth
        self.lab_iv, self.lab_req = None, None   # live lab edit (ivs.lab.Intervention) on the LIVE model only; the frozen copy is the untouched control
        self.birth_index = None
        self.gaze_ok = cfg.action_dim >= 8      # models trained with the gaze command: action = [touch x5, has_cmd, gaze_x, gaze_y]
        self.has_geom = cfg.geom_dim > 0        # auxiliary pupil/iris readout (ivs.gaze) -> where the model thinks the pupil is
        self.base_params = {f"{k}.{n}": p.detach().clone()
                            for k, m in self.mods(self.model).items() for n, p in m.named_parameters()}

    # ---- model switching (dropdown) -----------------------------------------------------
    def _root(self):
        return Path(__file__).resolve().parents[1]

    def _world_path(self, checkpoint):
        """The live-world file of a model: the --world path for the model the server started with, a per-checkpoint sibling for the others."""
        if Path(checkpoint).resolve() == Path(self.a.checkpoint).resolve():
            return Path(self.a.world)
        tag = hashlib.sha256(str(Path(checkpoint).resolve()).encode()).hexdigest()[:10]
        return Path(self.a.world).with_name(f"world_{Path(checkpoint).parent.name}_{Path(checkpoint).stem}_{tag}.pt")

    def _find_birth(self, checkpoint, cfg):
        """A birth memory made for exactly this checkpoint (sha256 match), looked for next to --world in birth_*.pt; None = start from noise."""
        for f in sorted(Path(self.a.world).parent.glob("birth_*.pt")):
            try:
                return load_birth(f, checkpoint, cfg)
            except Exception:
                continue
        return None

    def models(self):
        cur = Path(self.checkpoint).resolve()
        out = []
        for e in modelzoo.discover(self._root()):
            out.append({**e, "current": (self._root() / e["id"]).resolve() == cur})
        return {"models": out, "current": next((e["id"] for e in out if e["current"]), None)}

    def load_model(self, req):
        """Swap the running world to another trained checkpoint from the discovered list. Starts a fresh world (touch, snapshots, replay buffer
        cleared); the saved live world of that model is loaded only if it was saved for exactly that checkpoint."""
        path = modelzoo.resolve(self._root(), str(req.get("id")))
        with self.lock:
            if path.resolve() == Path(self.checkpoint).resolve():
                return {"ok": True, "changed": False, "info": self.info()}
            cfg, trained, model, frozen = self._build(path, self.a.image_size)       # raises before anything below runs
            self._install(str(path), cfg, trained, model, frozen, self._find_birth(path, cfg))
            self.world_file = self._world_path(path)
            self.snapshots, self.replay = [], collections.deque(maxlen=300)
            self._reset_runtime(load_world=self._world_matches(self.world_file, path))
            self.snapshot("birth")
            return {"ok": True, "changed": True, "info": self.info()}

    # ---- lab (mechanistic interpretability, ivs.lab) ---------------------------------------
    def lab_root(self):
        """Where cached lab runs live (<root>/pilot_results/lab); tests point this at a temp folder."""
        return self._root()

    def _lab_entry(self):
        """The zoo entry of the running checkpoint (the lab keys its cached runs by zoo id + file size); None outside pilot_results."""
        cur = Path(self.checkpoint).resolve()
        return next((e for e in modelzoo.discover(self._root()) if (self._root() / e["id"]).resolve() == cur), None)

    def _lab_results(self):
        e = self._lab_entry()
        return (e, lab.load_results(e["id"], e["bytes"], self.lab_root())) if e else (None, None)

    def lab_state(self):
        e, res = self._lab_results()
        if e is None:
            return {"available": False, "reason": "this checkpoint is outside pilot_results/checkpoints: the lab runs on discovered models only", "intervention": self.lab_req}
        job = self.lab_job if self.lab_job and self.lab_job.get("model") == e["id"] else None
        return {"available": True, "model": e["id"], "label": e["label"], "tier": e.get("tier"), "results": res, "job": job, "intervention": self.lab_req}

    def lab_run(self, req):
        """Analyse the running checkpoint in a background thread (own copy of the weights, the live world keeps ticking). req: {quick: bool}."""
        e = self._lab_entry()
        if e is None:
            raise ValueError("the lab runs on discovered models only")
        with self.lock:
            if self.lab_thread is not None and self.lab_thread.is_alive():
                raise ValueError("a lab run is already in progress")
            quick = bool(req.get("quick"))
            lc = lab.LabConfig.quick() if quick else lab.LabConfig()
            job = {"model": e["id"], "state": "running", "stage": "starting", "frac": 0.0, "quick": quick, "started": time.time()}
            self.lab_job = job
        root, path = self.lab_root(), self._root() / e["id"]

        def work():
            try:
                res, imgs = lab.analyse(path, lab.load_segmenter(), lc, progress=lambda s, f: job.update(stage=s, frac=f))
                lab.save_results(e["id"], e["bytes"], res, imgs, root)
                job.update(state="done", stage="done", frac=1.0)
            except Exception as ex:
                job.update(state="error", error=f"{type(ex).__name__}: {ex}")
        self.lab_thread = threading.Thread(target=work, daemon=True)
        self.lab_thread.start()
        return {"ok": True, "job": job}

    def lab_set(self, req):
        """Live edit of the running model's latent state (lesion / steer, see lab.interventions_from_request). {} or {clear: true} removes it.
        The frozen copy is never edited, so its panel is the control. Validation happens before anything changes."""
        _, res = self._lab_results()
        with self.lock:
            iv = lab.interventions_from_request(req, res, self.cfg.deter_dim, self.cfg.stoch_dim)
            self.lab_iv, self.lab_req = iv, (None if iv is None else req)
            return {"ok": True, "active": iv is not None, "intervention": self.lab_req}

    def lab_image(self, name):
        e, res = self._lab_results()
        if not res or not re.fullmatch(r"[A-Za-z0-9_]+", name) or name not in res.get("images", []):
            raise FileNotFoundError(name)
        return (lab.lab_dir(e["id"], self.lab_root()) / f"{name}.png").read_bytes()

    @staticmethod
    def _world_matches(world_file, checkpoint):
        if not world_file.exists():
            return False
        w = torch.load(world_file, map_location="cpu", weights_only=False)
        return Path(w.get("checkpoint", "")).resolve() == Path(checkpoint).resolve() if w.get("checkpoint") else False

    # ---- state ------------------------------------------------------------------------
    def _reset_runtime(self, load_world=False):
        self.s = np.zeros(5, np.float32)
        self.light_on = False     # zero touch = no light; the state's light x,y only say where the light is once the user switches it on
        self.tick_n, self.learn_steps, self.last_loss = 0, 0, None
        self.hist = collections.deque(maxlen=SEQ_T)
        self.replay = collections.deque(maxlen=300)
        self.was_dream = False
        self.ctrl = PlasticityController(base_lr=self.a.lr)
        self.resumed = False
        if load_world and self.world_file.exists():
            w = torch.load(self.world_file, map_location="cpu", weights_only=False)
            self.model.load_state_dict(w["model"])
            self._restore_ctrl(w["ctrl"])
            self.learn_steps, self.tick_n = w.get("learn_steps", 0), w.get("tick", 0)
            self.resumed = True
        self._make_opt()
        self._build_probe()
        self._update_retention(reset=True)
        self._new_world(self.noise_seed)
        self._drift_cache = self.drift()

    def _pad(self, a):
        """Touch-only actions (...,5) -> the model's action size (extra dims = no gaze command)."""
        extra = self.cfg.action_dim - a.shape[-1]
        return a if extra <= 0 else torch.cat([a, torch.zeros(*a.shape[:-1], extra)], -1)

    def _action(self, delta, gaze):
        """One tick's action: the rate-limited touch deltas, plus (gaze models) the absolute target pupil position in image units
        [-1,1] with has_cmd = 1; no gaze = zero command = the model dreams its own gaze."""
        a = self._pad(torch.from_numpy((delta / self.unit).astype(np.float32))[None])
        if self.gaze_ok and gaze is not None:
            a[0, 5] = 1.0
            a[0, 6:8] = torch.tensor([min(1.0, max(-1.0, float(gaze[0]))), min(1.0, max(-1.0, float(gaze[1])))])
        return a

    def _seed_state(self, model, b, seed):
        """Noise-seeded latent: sample the model's own prior a few steps from a zero state (same seed -> same
        noise for live and frozen, so they only differ through their weights)."""
        torch.manual_seed(int(seed) & 0x7FFFFFFF)
        h, z = model.rssm.initial(b, "cpu")
        zero = torch.zeros(b, self.cfg.action_dim)
        for _ in range(SEED_STEPS):
            h, z, _ = model.rssm.imagine_step(h, z, zero)
        return h, z

    def _new_world(self, seed):
        self.noise_seed = int(seed)
        with torch.no_grad():
            if self.birth is not None:     # the seed picks which remembered eye; live and frozen start from the same state
                k = self.birth_index = self.noise_seed % len(self.birth["h"])
                self.h, self.z = self.birth["h"][k:k + 1].clone(), self.birth["z"][k:k + 1].clone()
                self.hf, self.zf = self.h.clone(), self.z.clone()
            else:
                self.h, self.z = self._seed_state(self.model, 1, seed)
                self.hf, self.zf = self._seed_state(self.frozen, 1, seed)
            self.canvas = self.model.decode(self.h, self.z).clamp(0, 1)      # 1,1,H,W: the model's own eye
        self.hist.clear()

    def _make_opt(self):
        self.opt = torch.optim.AdamW([{"params": list(self.model.rssm.parameters()), "name": "rssm"}],
                                     lr=self.a.lr)

    def _restore_ctrl(self, st):
        self.ctrl.scales, self.ctrl.ema_error = dict(st["scales"]), st["ema_error"]
        self.ctrl.ema_progress, self.ctrl.steps = st["ema_progress"], st["steps"]

    # ---- retention: distance to the frozen copy's imagination ------------------------------
    def _imagine(self, model, actions, seed):
        h, z = self._seed_state(model, actions.shape[0], seed)
        return model.rollout(h, z, actions)

    def _build_probe(self):
        gen = torch.Generator().manual_seed(1234)
        _, actions = self.sampler._walk(4, SEQ_T, gen)
        actions = self._pad(actions)
        with torch.no_grad():
            target = self._imagine(self.frozen, actions, 777)
        self.probe = (actions, target)
        var = float(target.var(dim=0).mean())
        self.probe_tol = max(1e-3, self.a.retain_tol * var)

    @torch.no_grad()
    def _probe_err(self):
        actions, target = self.probe
        return float(((self._imagine(self.model, actions, 777) - target) ** 2).mean())

    def _update_retention(self, reset=False):
        e = self._probe_err()
        self.probe_ema = e if reset else 0.5 * self.probe_ema + 0.5 * e
        self.retention_ok = self.probe_ema <= self.probe_tol

    def drift(self):
        mod, layers = {}, []
        for k, m in self.mods(self.model).items():
            num = den = 0.0
            for n, p in m.named_parameters():
                w0 = self.base_params[f"{k}.{n}"]
                d, w = float(((p.detach() - w0) ** 2).sum()), float((w0 ** 2).sum())
                num, den = num + d, den + w
                if p.ndim > 1:
                    layers.append({"name": f"{k}.{n}".replace(".weight", ""), "rel": (d / (w + 1e-12)) ** .5})
            mod[k] = (num / (den + 1e-12)) ** .5
        return {"modules": mod, "layers": layers}

    # ---- model stepping ----------------------------------------------------------------
    def _predict(self, model, h, z, a, seed):
        torch.manual_seed(seed)
        ph, pz, _ = model.rssm.imagine_step(h, z, a)
        if self.lab_iv is not None and model is self.model:
            ph, pz = self.lab_iv(ph, pz, self.tick_n, "pred")
        return dict(ph=ph, pz=pz, pred=model.decode(ph, pz))

    def _perceive(self, model, h, z, a, obs, seed):
        torch.manual_seed(seed + 1)
        nh, nz, prior, post = model.rssm.observe_step(h, z, a, model.encoder(obs))
        if self.lab_iv is not None and model is self.model:
            nh, nz = self.lab_iv(nh, nz, self.tick_n, "post")
        return dict(h=nh, z=nz, recon=model.decode(nh, nz), mean=post[0], std=post[1],
                    kl=float(gaussian_kl(post, prior).mean()))

    def _acts(self, model, obs):
        x, out = obs, []
        for i, layer in enumerate(model.encoder.net):
            x = layer(x)
            if i in (1, 4, 7):
                out.append(x[0])
            if i == 7:
                break
        return out

    @staticmethod
    def _tile(a, cols=8):
        c, h, w = a.shape
        a = a - a.amin((1, 2), keepdim=True)
        a = a / (a.amax((1, 2), keepdim=True) + 1e-6)
        rows = (c + cols - 1) // cols
        canvas = torch.zeros(rows * h, cols * w)
        for i in range(c):
            canvas[(i // cols) * h:(i // cols + 1) * h, (i % cols) * w:(i % cols + 1) * w] = a[i]
        return png(u8(canvas))

    # ---- one tick ----------------------------------------------------------------------
    def tick(self, req):
        with self.lock:
            intervene = bool(req.get("intervene", True))
            # Intervention off: all touch input is ignored (no scatter, zero actions, the state is held), so the
            # world model runs on its own. Any touch already applied stays as it was; "reset touch" clears it.
            tgt = np.clip(np.array(req.get("target", self.s) if intervene else self.s, dtype=np.float32), -self.lim, self.lim)
            delta = np.clip(tgt - self.s, -3 * self.unit, 3 * self.unit)   # speed-limited like training
            self.s = self.s + delta
            if intervene and "light" in req:                                  # intervention off holds the light as it was, like the rest of the touch
                self.light_on = bool(req["light"])
            lg = torch.tensor([float(self.light_on)])
            gz = req.get("gaze")                                              # None | [x, y] absolute target pupil position
            gaze = [float(gz[0]), float(gz[1])] if (gz is not None and self.gaze_ok) else None
            act = self._action(delta, gaze)
            dream, learn = bool(req.get("dream")), bool(req.get("learn"))
            trust = min(1.0, max(0.0, float(req.get("trust", 0.5))))      # heal source: 0 perceived .. 1 expected
            pts = [(float(p[0]), float(p[1]), float(p[2])) for p in (req.get("scatter") or [])] if intervene else []
            if dream != self.was_dream:
                self.hist.clear()
                self.was_dream = dream
            seed = (self.noise_seed * 1000003 + self.tick_n * 7919) & 0x7FFFFFFF
            gseed = seed ^ 0x2545F491
            with torch.no_grad():
                PL = self._predict(self.model, self.h, self.z, act, seed)
                PF = self._predict(self.frozen, self.hf, self.zf, act, seed)
                if dream:       # each model perceives its own render
                    obs_l = scatter_pixels(PL["pred"], pts, SCATTER_RADIUS, gseed)
                    obs_f = scatter_pixels(PF["pred"], pts, SCATTER_RADIUS, gseed)
                else:           # both perceive the touched canvas
                    touched = self.sampler.apply_state(self.canvas[:, None], torch.from_numpy(self.s)[None, None], light=lg)[:, 0]
                    obs_l = obs_f = scatter_pixels(touched, pts, SCATTER_RADIUS, gseed)
                L = {**PL, **self._perceive(self.model, self.h, self.z, act, obs_l, seed)}
                F_ = {**PF, **self._perceive(self.frozen, self.hf, self.zf, act, obs_f, seed)}
                al, af = self._acts(self.model, obs_l), self._acts(self.frozen, obs_f)
                belief = (1 - trust) * L["recon"] + trust * L["pred"]
                self.canvas = self.sampler.invert_state(belief[:, None], torch.from_numpy(self.s)[None, None], light=lg)[:, 0]
            self.h, self.z, self.hf, self.zf = L["h"], L["z"], F_["h"], F_["z"]
            mse = lambda x, y: float(((x - y) ** 2).mean())
            if dream:   # nothing external to predict: the error is dissonance (expected vs perceived)
                e_l, e_f, ref, metric = mse(L["recon"], L["pred"]), mse(F_["recon"], F_["pred"]), L["recon"], "dissonance"
            else:
                e_l, e_f, ref, metric = mse(L["pred"], obs_l), mse(F_["pred"], obs_f), obs_l, "prediction error"
            self.hist.append((obs_l[0].numpy().copy(), act[0].numpy().copy()))
            if len(self.hist) == SEQ_T and self.tick_n % 2 == 0:
                self.replay.append((np.stack([x for x, _ in self.hist]), np.stack([y for _, y in self.hist])))
            self.tick_n += 1
            if learn and self.tick_n % self.a.learn_every == 0 and len(self.replay) >= 4:
                self._learn()
            if self.tick_n % 10 == 0:
                self._drift_cache = self.drift()
            diff = [((x - y).abs().mean((1, 2)) / (y.abs().mean((1, 2)) + 1e-6)).tolist() for x, y in zip(al, af)]
            return {
                "tick": self.tick_n, "state": self.s.tolist(), "light": self.light_on, "dream": dream, "metric": metric, "intervene": intervene,
                "img": {"stim": png(u8(obs_l[0])), "pred": png(u8(L["pred"][0])),
                        "recon": png(u8(L["recon"][0])), "frozen": png(u8(F_["pred"][0])),
                        "surprise": png(u8((L["pred"] - ref).abs()[0].mean(0) * 4))},
                "gaze": {"command": gaze, "pupil": self.model.geometry(L["h"], L["z"])[0, :3].tolist() if self.has_geom else None},
                "mse": {"pred": e_l, "pred_frozen": e_f,
                        "recon": mse(L["recon"], obs_l), "recon_frozen": mse(F_["recon"], obs_f)},
                "kl": L["kl"],
                "latent": {"mean": L["mean"][0].tolist(), "std": L["std"][0].tolist()},
                "acts": [self._tile(x) for x in al], "act_diff": diff,
                "drift": self._drift_cache,
                "plasticity": {"scales": self.ctrl.scales, "retention_ok": self.retention_ok,
                               "probe": self.probe_ema, "tol": self.probe_tol,
                               "learn_steps": self.learn_steps, "loss": self.last_loss,
                               "replay": len(self.replay)},
            }

    def _rehearse_loss(self):
        """Pseudo-rehearsal: the live dynamics imagine what the frozen copy imagines (same noise)."""
        seed = random.getrandbits(30)
        _, actions = self.sampler._walk(2, SEQ_T, None)
        actions = self._pad(actions)
        with torch.no_grad():
            target = self._imagine(self.frozen, actions, seed)
        return _pixel_sum_loss(self._imagine(self.model, actions, seed), target)

    def _learn(self):
        own = random.sample(list(self.replay), min(4, len(self.replay)))
        images = torch.from_numpy(np.stack([c[0] for c in own]))
        actions = torch.from_numpy(np.stack([c[1] for c in own]))
        self.model.train()
        self.opt.zero_grad(set_to_none=True)
        loss, m = compute_loss(self.model, images, actions, self.targs)
        loss = loss + self.a.rehearse_weight * self._rehearse_loss()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.rssm.parameters(), 100.0)
        lrs = self.ctrl.update(float(m["error"]), self.retention_ok)
        for g in self.opt.param_groups:
            g["lr"] = lrs[g["name"]]
        self.opt.step()
        self.model.eval()
        self.learn_steps += 1
        self.last_loss = float(loss)
        if self.learn_steps % self.a.probe_every == 0:
            self._update_retention()
        self._drift_cache = self.drift()

    # ---- time travel / persistence -----------------------------------------------------
    def snapshot(self, name=None):
        with self.lock:
            self._snap_id += 1
            self.snapshots.append({"id": self._snap_id, "name": name or f"snapshot {self._snap_id}", "tick": self.tick_n,
                                   "model": {k: v.clone() for k, v in self.model.state_dict().items()},
                                   "ctrl": copy.deepcopy(self.ctrl.state_dict()),
                                   "h": self.h.clone(), "z": self.z.clone(), "hf": self.hf.clone(), "zf": self.zf.clone(),
                                   "canvas": self.canvas.clone(), "learn_steps": self.learn_steps})
            self.snapshots = self.snapshots[-20:]
            return self.list_snapshots()

    def list_snapshots(self):
        return [{"id": s["id"], "name": s["name"], "tick": s["tick"], "learn_steps": s["learn_steps"]} for s in self.snapshots]

    def restore(self, sid):
        with self.lock:
            s = next(x for x in self.snapshots if x["id"] == sid)
            self.model.load_state_dict(s["model"])
            self._restore_ctrl(s["ctrl"])
            self.h, self.z, self.learn_steps = s["h"].clone(), s["z"].clone(), s["learn_steps"]
            self.hf, self.zf, self.canvas = s["hf"].clone(), s["zf"].clone(), s["canvas"].clone()
            self._make_opt()
            self.hist.clear()
            self._update_retention(reset=True)
            self._drift_cache = self.drift()
            return {"ok": True, "tick": s["tick"]}

    def set_state(self, req):
        with self.lock:
            if "state" in req:
                self.s = np.clip(np.array(req["state"], np.float32), -self.lim, self.lim)
            if "light" in req:
                self.light_on = bool(req["light"])
            return {"ok": True}

    def reseed(self, req):
        with self.lock:
            seed = int(req["seed"]) if "seed" in req else random.randrange(2 ** 30)
            self.s = np.zeros(5, np.float32)
            self.light_on = False
            self._new_world(seed)
            return {"ok": True, "seed": seed}

    def save(self):
        with self.lock:
            self.world_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": self.model.state_dict(), "ctrl": self.ctrl.state_dict(), "checkpoint": str(Path(self.checkpoint).resolve()),
                        "learn_steps": self.learn_steps, "tick": self.tick_n}, self.world_file)
            return {"ok": True, "path": str(self.world_file)}

    def reset(self):
        with self.lock:
            birth = {f"{k}": v for k, v in self.frozen.state_dict().items()}
            self.model.load_state_dict(birth)
            self.snapshots = []
            self._reset_runtime(load_world=False)
            return {"ok": True}

    def info(self):
        return {"size": self.cfg.image_size, "trained": self.trained,
                "source": "birth memory (model's own render)" if self.birth is not None else "model's own render",
                "birth": None if self.birth is None else {"index": self.birth_index, "count": len(self.birth["h"])},
                "checkpoint": str(self.checkpoint), "seed": self.noise_seed, "limits": self.lim.tolist(), "resumed": self.resumed,
                "channels": self.cfg.channels, "action_dim": self.cfg.action_dim, "gaze_command": self.gaze_ok, "geometry": self.has_geom,
                "learn_every": self.a.learn_every, "lab_active": self.lab_iv is not None}


def make_handler(world):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path, _, query = self.path.partition("?")
            if path in ("/", "/index.html"):
                self._send(200, WEB.read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/info":
                self._send(200, world.info())
            elif path == "/api/snapshots":
                self._send(200, world.list_snapshots())
            elif path == "/api/models":
                self._send(200, world.models())
            elif path == "/api/lab":
                self._send(200, world.lab_state())
            elif path == "/api/lab/img":
                try:
                    self._send(200, world.lab_image(query.partition("f=")[2]), "image/png")
                except FileNotFoundError:
                    self._send(404, {"error": "no such lab image"})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            try:
                n = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(n) or b"{}")
                route = {"/api/tick": world.tick, "/api/set_state": world.set_state, "/api/reseed": world.reseed,
                         "/api/snapshot": lambda r: world.snapshot(r.get("name")),
                         "/api/restore": lambda r: world.restore(int(r["id"])),
                         "/api/save": lambda r: world.save(), "/api/reset": lambda r: world.reset(), "/api/load_model": world.load_model,
                         "/api/lab/run": world.lab_run, "/api/lab/set": world.lab_set}.get(self.path)
                self._send(200, route(req)) if route else self._send(404, {"error": "not found"})
            except Exception as e:  # surface errors to the page instead of dropping the connection
                self._send(500, {"error": f"{type(e).__name__}: {e}"})
    return H


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="IVS touch/observer UI")
    p.add_argument("--checkpoint", default="checkpoints/latest.pt")
    p.add_argument("--world", default="checkpoints/world.pt", help="persisted live world (weights + plasticity)")
    p.add_argument("--fresh", action="store_true", help="ignore a saved world file")
    p.add_argument("--image-size", type=int, default=64, help="only used when no checkpoint exists")
    p.add_argument("--seed", type=int, default=None, help="noise seed for the first canvas (default random); with --birth it also picks the remembered eye (seed %% count)")
    p.add_argument("--birth", default=None, help="birth memory file made for this exact checkpoint (scripts/make_birth.py): start from the model's "
                   "own state after it perceived a real eye once, instead of from noise")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--learn-every", type=int, default=4, help="ticks between online updates")
    p.add_argument("--probe-every", type=int, default=10, help="learning steps between retention checks")
    p.add_argument("--rehearse-weight", type=float, default=1.0, help="weight of pseudo-rehearsal vs the frozen copy")
    p.add_argument("--retain-tol", type=float, default=0.25,
                   help="retention limit as a fraction of the probe's pixel variance (floor 1e-3)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    return p.parse_args(argv)


def main():
    a = parse_args()
    w = World(a)
    if not w.trained:
        print(f"NOTE: {a.checkpoint} not found: running an UNTRAINED model (the UI will say so).")
    w.snapshot("birth")
    start = f"birth memory {w.birth_index + 1}/{len(w.birth['h'])}" if w.birth is not None else "noise"
    print(f"IVS UI on http://{a.host}:{a.port}  (canvas = model's own render, start = {start}, seed={w.noise_seed}, resumed={w.resumed})")
    ThreadingHTTPServer((a.host, a.port), make_handler(w)).serve_forever()


if __name__ == "__main__":
    main()
