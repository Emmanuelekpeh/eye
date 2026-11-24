"""EyeNCA dream viewer.  python -m ivs.nca_ui --checkpoint pilot_results/kaggle_nca1_15000steps/pilot/E0_N50/model.pt --seed 21 --port 8767

ivs/ui.py is RSSM-only (it needs rssm / encoder / action interventions), so it cannot load an EyeNCA checkpoint. This is the NCA counterpart:
stdlib HTTP server, one lock, nothing read from disk except the checkpoint. "Evolve" feeds the model's own finished render back in (re-noised to
a chosen sigma, then denoised) so a dream continues from its own previous output instead of restarting from pixel noise. A "dream" is the Heun sampler (ivs.nca.run_sampler, same math,
advanced one noise level per request) started from pure noise, so you watch the denoising trajectory: the noisy state x_t and the model's
current clean estimate D(x_t, sigma). When it finishes you can SCATTER pixels on the result (drag on it) and ask the same network to repair it:
  blind  = SDEdit (diffuse the touched image to sigma_start and denoise; the model is not told where the damage is),
  masked = RePaint-style replacement (the model is told that everything outside the touched discs is known).
Runs on CPU (about 1 s per noise level for N50 at 96 px) or CUDA when available.
"""
import argparse
import json
import random
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

from . import nca
from .ui import png, u8, scatter_pixels, SCATTER_RADIUS

WEB = Path(__file__).parent / "web" / "nca.html"


def load_model(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = nca.net_from_config(ck["config"])
    m.load_state_dict(ck["model"])
    return m.to(device).eval(), ck.get("run", {})


def heun_step(model, x, s, sn, known=None, mask=None, gen=None, st=None):
    """One level of nca.run_sampler (repeats=1) from sigma s to sn. Returns (x_next, D(x, s)). st = [hidden] (a one-element list, updated in place)
    carries the persistent hidden state through both denoise calls exactly like run_sampler(persist=True); None = the E0 behaviour."""
    def D(z, sg):
        if st is None:
            return model.denoise(z, sg)
        d_, st[0] = model.denoise(z, sg, hidden=st[0], return_hidden=True)
        return d_

    den = D(x, s)
    d = (x - den) / s
    xn = x + (sn - s) * d
    if sn > 0:
        d2 = (xn - D(xn, sn)) / sn
        xn = x + (sn - s) * 0.5 * (d + d2)
    if mask is not None:
        kn = known + sn * nca._randn(known.shape, gen, x.device) if sn > 0 else known
        xn = mask * kn + (1 - mask) * xn
    return xn, den


class Session:
    """A sampler trajectory advanced one noise level per call."""
    def __init__(self, x, sigmas, known=None, mask=None, gen=None, hidden=None, persist=False):
        self.x, self.sig, self.i, self.known, self.mask, self.gen = x, sigmas, 0, known, mask, gen
        self.den = None
        self.persist, self.st = persist, [hidden]              # st[0] = the hidden channels (persistent models only)

    @property
    def done(self):
        return self.i >= len(self.sig) - 1

    @property
    def hidden(self):
        return self.st[0]

    @property
    def view(self):                 # what the "state" panel shows
        return self.x

    @property
    def result(self):               # what the "result" panel shows once done
        return self.x

    @torch.no_grad()
    def step(self, model):
        s, sn = float(self.sig[self.i]), float(self.sig[self.i + 1])
        self.x, self.den = heun_step(model, self.x, s, sn, self.known, self.mask, self.gen, self.st if self.persist else None)
        self.i += 1
        return s


class LiveSession(Session):
    """LIVING MODE (persistent models): n ticks of nca.live, one per request. Each tick re-noises the own render to sigma, denoises it with the
    carried hidden state and feeds the result back. view = the noisy input of the last tick, result = the own render."""
    def __init__(self, x, n, sigma, gen, hidden):
        super().__init__(x, [sigma] * (n + 1), gen=gen, hidden=hidden, persist=True)
        self.noisy = x

    @property
    def view(self):
        return self.noisy

    @torch.no_grad()
    def step(self, model):
        sg = float(self.sig[self.i])
        self.noisy = self.x + sg * nca._randn(self.x.shape, self.gen, self.x.device)
        d, self.st[0] = model.denoise(self.noisy, sg, hidden=self.st[0], return_hidden=True)
        self.x, self.den = d.clamp(-1, 1), d
        self.i += 1
        return sg


class NCAWorld:
    def __init__(self, a):
        self.lock = threading.RLock()
        self.a = a
        self.device = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
        torch.set_num_threads(max(1, a.threads))
        self.model, self.run = load_model(a.checkpoint, self.device)
        self.size = self.model.cfg.image_size
        self.persist = bool(self.run.get("persist", False))      # PD-NCA checkpoint: the hidden channels live on between sessions
        self.hidden = None          # the cell grid's hidden channels after the last finished session (persistent models only)
        self.final = None           # the last finished image in [0,1], (1,3,H,W) on cpu
        self.sess, self.seed, self.kind, self.evolve_n, self.live_n = None, a.seed, "dream", 0, 0
        self.dream(self.seed, a.steps)

    def _gen(self, seed):
        return torch.Generator(device="cpu").manual_seed(int(seed) & 0x7FFFFFFF)

    def dream(self, seed, steps):
        with self.lock:
            self.seed, self.kind, self.evolve_n, self.live_n = int(seed), "dream", 0, 0
            g = self._gen(seed)
            sig = nca.sigma_schedule(int(steps), 0.01, 5.0)
            x = nca._randn((1, self.model.cfg.channels, self.size, self.size), g, self.device) * sig[0]
            self.hidden = None
            self.sess = Session(x, sig, gen=g, persist=self.persist)
            return self.state(None)

    def repair(self, req):
        with self.lock:
            if self.final is None:
                raise ValueError("nothing to repair yet: let a dream finish first")
            pts = [(float(p[0]), float(p[1]), float(p[2])) for p in (req.get("scatter") or [])]
            seed = int(req.get("seed", self.seed)) + 1
            steps, sigma = int(req.get("steps", 12)), float(req.get("sigma", 0.6))
            damaged = scatter_pixels(self.final, pts, SCATTER_RADIUS, seed)
            g = self._gen(seed)
            dm = damaged.to(self.device)
            if req.get("masked"):
                hit = (damaged != self.final).any(1, keepdim=True).float()
                hit = torch.nn.functional.max_pool2d(hit, 5, 1, 2)             # discs, not just the pixels that changed value
                mask = (1 - hit).to(self.device)
                sig = nca.sigma_schedule(steps, 0.01, 5.0)
                known = nca.to_model(dm)
                x = mask * (known + sig[0] * nca._randn(known.shape, g, self.device)) + (1 - mask) * sig[0] * nca._randn(known.shape, g, self.device)
                self.sess = Session(x, sig, known=known, mask=mask, gen=g, hidden=self.hidden, persist=self.persist)
                self.kind = "repair (masked)"
            else:
                self.sess = self._blind_session(dm, sigma, steps, g)
                self.kind = "repair (blind)"
            self.damaged = damaged
            return self.state(None)

    def _blind_session(self, base01, sigma, steps, g):
        """SDEdit start (same draw order as nca.repair_blind): diffuse base01 [0,1] to sigma, then denoise back."""
        sig = nca.sigma_schedule(steps, min(0.01, sigma / 2), sigma)
        x = nca.to_model(base01) + sig[0] * nca._randn(base01.shape, g, self.device)
        return Session(x, sig, gen=g, hidden=self.hidden, persist=self.persist)

    def live(self, req):
        """Living mode (persistent models): n ticks at sigma on the model's own last render with its own hidden state (nca.live)."""
        with self.lock:
            if not self.persist:
                raise ValueError("living mode needs a persistent (PD-NCA) checkpoint; this one rebuilds its hidden state on every call")
            if self.final is None:
                raise ValueError("nothing to run yet: let a dream finish first")
            self.live_n += 1
            n, sigma = int(req.get("ticks", 30)), float(req.get("sigma", 0.05))
            self.sess = LiveSession(nca.to_model(self.final.to(self.device)), n, sigma, self._gen(self.seed * 999983 + self.live_n), self.hidden)
            self.kind = f"live #{self.live_n}"
            return self.state(None)

    def evolve(self, req):
        """Self-feeding loop: the next dream starts from the model's own previous render, re-noised to sigma and denoised again.
        Small sigma = the image drifts slowly; large sigma = it is mostly re-dreamed."""
        with self.lock:
            if self.final is None:
                raise ValueError("nothing to evolve yet: let a dream finish first")
            self.evolve_n += 1
            sigma, steps = float(req.get("sigma", 0.8)), int(req.get("steps", 12))
            g = self._gen(self.seed * 1000003 + self.evolve_n)
            self.sess = self._blind_session(self.final.to(self.device), sigma, steps, g)
            self.kind = f"evolve #{self.evolve_n}"
            return self.state(None)

    def state(self, sigma):
        s = self.sess
        out = {"kind": self.kind, "seed": self.seed, "i": s.i, "n": len(s.sig) - 1, "done": s.done,
               "sigma": sigma if sigma is not None else float(s.sig[0]), "size": self.size,
               "x": png(u8(nca.to_unit(s.view[0]).cpu())),
               "est": png(u8(nca.to_unit(s.den[0]).cpu())) if s.den is not None else None}
        if s.done:
            self.final = nca.to_unit(s.result).cpu()
            self.hidden = s.hidden
            out["final"] = png(u8(self.final[0]))
        return out

    def tick(self, req):
        with self.lock:
            s = self.sess
            sigma = None
            if not s.done:
                sigma = s.step(self.model)
            return self.state(sigma)

    def info(self):
        return {"size": self.size, "params": nca.count_params(self.model), "run": self.run.get("name"), "device": str(self.device),
                "seed": self.seed, "steps": self.a.steps, "radius": SCATTER_RADIUS, "cond_dim": self.model.cfg.cond_dim, "persist": self.persist}


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
            if self.path in ("/", "/index.html"):
                self._send(200, WEB.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/info":
                self._send(200, world.info())
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            try:
                n = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(n) or b"{}")
                route = {"/api/tick": world.tick,
                         "/api/dream": lambda r: world.dream(r.get("seed", random.randrange(2 ** 30)), r.get("steps", world.a.steps)),
                         "/api/repair": world.repair, "/api/evolve": world.evolve, "/api/live": world.live}.get(self.path)
                self._send(200, route(req)) if route else self._send(404, {"error": "not found"})
            except Exception as e:  # surface errors to the page instead of dropping the connection
                self._send(500, {"error": f"{type(e).__name__}: {e}"})
    return H


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="EyeNCA dream viewer")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--seed", type=int, default=None, help="noise seed of the first dream (default random)")
    p.add_argument("--steps", type=int, default=18, help="noise levels per dream (the quality the eval used)")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--cpu", action="store_true", help="force CPU even if CUDA exists")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8767)
    a = p.parse_args(argv)
    if a.seed is None:
        a.seed = random.randrange(2 ** 30)
    return a


def main():
    a = parse_args()
    w = NCAWorld(a)
    print(f"EyeNCA viewer on http://{a.host}:{a.port}  (device={w.device}, params={nca.count_params(w.model)}, seed={w.seed})")
    ThreadingHTTPServer((a.host, a.port), make_handler(w)).serve_forever()


if __name__ == "__main__":
    main()
