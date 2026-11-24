import torch
from torch import nn
import torch.nn.functional as F
from .config import ModelConfig


def conv_channels(cfg: ModelConfig):
    """Channel widths of the 4 conv stages; multiples of 8 so GroupNorm(4|8) always divides them."""
    return tuple(max(8, int(round(n * cfg.conv_mult / 8)) * 8) for n in (32, 48, 64, 96))


class ConvEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        c = cfg.channels
        c1, c2, c3, c4 = conv_channels(cfg)
        self.net = nn.Sequential(
            nn.Conv2d(c, c1, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(c1, c2, 4, 2, 1), nn.GroupNorm(4, c2), nn.SiLU(),
            nn.Conv2d(c2, c3, 4, 2, 1), nn.GroupNorm(8, c3), nn.SiLU(),
            nn.Conv2d(c3, c4, 4, 2, 1), nn.GroupNorm(8, c4), nn.SiLU(),
            nn.Flatten(),
            nn.Linear(c4 * (cfg.image_size // 16) ** 2, cfg.feature_dim),
            nn.LayerNorm(cfg.feature_dim), nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class RSSM(nn.Module):
    """Compact recurrent state-space model with deterministic and stochastic state."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.gru = nn.GRUCell(cfg.stoch_dim + cfg.action_dim, cfg.deter_dim)
        hd = cfg.hidden_dim
        self.prior = nn.Sequential(nn.Linear(cfg.deter_dim, hd), nn.SiLU(),
                                   nn.Linear(hd, 2 * cfg.stoch_dim))
        self.posterior = nn.Sequential(nn.Linear(cfg.deter_dim + cfg.feature_dim, hd), nn.SiLU(),
                                       nn.Linear(hd, 2 * cfg.stoch_dim))

    def _stats(self, x):
        mean, raw_std = x.chunk(2, dim=-1)
        std = F.softplus(raw_std) + self.cfg.min_std
        return mean, std

    def initial(self, batch, device):
        h = torch.zeros(batch, self.cfg.deter_dim, device=device)
        z = torch.zeros(batch, self.cfg.stoch_dim, device=device)
        return h, z

    # Training-only switch (never in the checkpoint; the forward values are identical): the prior head reads h.detach(), so the prior loss (kl_dyn) trains
    # the prior head but sends no gradient into the GRU state / encoder. Set by pilot.train_run from RunConfig.prior_stopgrad (scripts/overfit_rssm.py G_stopgrad).
    prior_stopgrad = False

    def observe_step(self, prev_h, prev_z, action, embed):
        h = self.gru(torch.cat([prev_z, action], dim=-1), prev_h)
        prior_mean, prior_std = self._stats(self.prior(h.detach() if self.prior_stopgrad else h))
        post_mean, post_std = self._stats(self.posterior(torch.cat([h, embed], dim=-1)))
        z = post_mean + post_std * torch.randn_like(post_std)
        return h, z, (prior_mean, prior_std), (post_mean, post_std)

    def imagine_step(self, prev_h, prev_z, action, temperature=1.0):
        """One open-loop step. temperature scales the prior's sampling noise (1 = the trained distribution, 0 = the prior mean)."""
        h = self.gru(torch.cat([prev_z, action], dim=-1), prev_h)
        mean, std = self._stats(self.prior(h))
        z = mean + temperature * std * torch.randn_like(std)
        return h, z, (mean, std)


def build_decoder(cfg: ModelConfig, in_dim: int):
    """Latent (B, in_dim) -> image. Same layers/indices as before (checkpoints keep loading); reusable for probes."""
    c1, c2, c3, c4 = conv_channels(cfg)
    s = cfg.image_size // 16
    return nn.Sequential(
        nn.Linear(in_dim, c4 * s * s),
        nn.SiLU(),
        nn.Unflatten(1, (c4, s, s)),
        nn.ConvTranspose2d(c4, c3, 4, 2, 1), nn.GroupNorm(8, c3), nn.SiLU(),
        nn.ConvTranspose2d(c3, c2, 4, 2, 1), nn.GroupNorm(4, c2), nn.SiLU(),
        nn.ConvTranspose2d(c2, c1, 4, 2, 1), nn.SiLU(),
        nn.ConvTranspose2d(c1, cfg.channels, 4, 2, 1), nn.Sigmoid(),
    )


class EyeWorldModel(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        self.encoder = ConvEncoder(self.cfg)
        self.rssm = RSSM(self.cfg)
        self.decoder = build_decoder(self.cfg, self.cfg.deter_dim + self.cfg.stoch_dim)
        if self.cfg.geom_dim > 0:
            # auxiliary eye-geometry readout (training signal only; never fed back into the dynamics). Predicts STANDARDISED
            # geometry; geom_mean/std are saved with the checkpoint so the readout can be turned back into image units.
            self.geom_head = nn.Sequential(nn.Linear(self.cfg.deter_dim + self.cfg.stoch_dim, self.cfg.hidden_dim), nn.SiLU(),
                                           nn.Linear(self.cfg.hidden_dim, self.cfg.geom_dim))
            self.register_buffer("geom_mean", torch.zeros(self.cfg.geom_dim))
            self.register_buffer("geom_std", torch.ones(self.cfg.geom_dim))

    def decode(self, h, z):
        return self.decoder(torch.cat([h, z], dim=-1))

    def geometry(self, h, z):
        """Eye geometry in image units (normalised positions / radii, see ivs.gaze) read from a latent state."""
        return self.geom_head(torch.cat([h, z], dim=-1)) * self.geom_std + self.geom_mean

    def observe(self, images, actions):
        """images: B,T,C,H,W; actions: B,T,A. Returns sequence states and diagnostics. With a geometry head also
        'geom' (head on posterior state (h,z)) and 'geom_prior' (head on (h, prior mean): what the state knew BEFORE seeing
        the frame, i.e. what the action/command alone produced), both standardised, B,T,G."""
        b, t, c, h_img, w_img = images.shape
        embeds = self.encoder(images.reshape(b * t, c, h_img, w_img)).reshape(b, t, -1)
        h, z = self.rssm.initial(b, images.device)
        outputs = {"h": [], "z": [], "prior": [], "posterior": [], "recon": []}
        if self.cfg.geom_dim > 0:
            outputs.update(geom=[], geom_prior=[])
        for i in range(t):
            h, z, prior, posterior = self.rssm.observe_step(h, z, actions[:, i], embeds[:, i])
            outputs["h"].append(h)
            outputs["z"].append(z)
            outputs["prior"].append(prior)
            outputs["posterior"].append(posterior)
            outputs["recon"].append(self.decode(h, z))
            if self.cfg.geom_dim > 0:
                outputs["geom"].append(self.geom_head(torch.cat([h, z], dim=-1)))
                outputs["geom_prior"].append(self.geom_head(torch.cat([h, prior[0]], dim=-1)))
        for key in ("h", "z", "recon") + (("geom", "geom_prior") if self.cfg.geom_dim > 0 else ()):
            outputs[key] = torch.stack(outputs[key], dim=1)
        return outputs

    def refine_latent(self, h, z, target, prior, steps=20, lr=0.05, prior_weight=0.01):
        """Predictive-coding style inference (test time, not trained): `steps` Adam updates of the stochastic latent z (h fixed) that
        decode -> compare with the observed frame `target` -> update, starting from the posterior sample `z`. Objective = the training one:
        pixel smooth-L1 summed over C,H,W + prior_weight (= kl_weight*(1-kl_balance) of the V6 base) * the Gaussian log-prior of z under
        `prior` = (mean, std). Returns the refined z (detached). Unlimited-power upper bound of what a learned refinement loop could do."""
        if steps <= 0:
            return z
        pm, ps = prior
        with torch.enable_grad():
            z = z.detach().clone().requires_grad_(True)
            opt = torch.optim.Adam([z], lr=lr)
            for _ in range(steps):
                pix = F.smooth_l1_loss(self.decode(h.detach(), z), target, reduction="none").flatten(1).sum(-1)
                reg = (0.5 * ((z - pm.detach()) / ps.detach()) ** 2).sum(-1)
                z.grad = torch.autograd.grad((pix + prior_weight * reg).sum(), z)[0]      # only z gets a gradient (model params stay clean)
                opt.step()
        return z.detach()

    def observe_refined(self, images, actions, steps=20, lr=0.05, prior_weight=0.01):
        """`observe` with refine_latent after each posterior sample; the REFINED z is what the next GRU step and the decoder see.
        Same outputs as observe (recon from the refined z; 'z_unrefined' = the posterior samples). steps=0 reproduces observe's maths."""
        b, t, c, h_img, w_img = images.shape
        with torch.no_grad():
            embeds = self.encoder(images.reshape(b * t, c, h_img, w_img)).reshape(b, t, -1)
        h, z = self.rssm.initial(b, images.device)
        outputs = {"h": [], "z": [], "z_unrefined": [], "prior": [], "posterior": [], "recon": []}
        for i in range(t):
            with torch.no_grad():
                h, z_post, prior, posterior = self.rssm.observe_step(h, z, actions[:, i], embeds[:, i])
            z = self.refine_latent(h, z_post, images[:, i], prior, steps, lr, prior_weight)
            with torch.no_grad():
                outputs["recon"].append(self.decode(h, z))
            outputs["h"].append(h); outputs["z"].append(z); outputs["z_unrefined"].append(z_post)
            outputs["prior"].append(prior); outputs["posterior"].append(posterior)
        for key in ("h", "z", "z_unrefined", "recon"):
            outputs[key] = torch.stack(outputs[key], dim=1)
        return outputs

    def rollout(self, initial_h, initial_z, actions):
        """Differentiable open-loop prediction (used by the prediction loss). actions: B,T,A."""
        h, z = initial_h, initial_z
        frames = []
        for i in range(actions.shape[1]):
            h, z, _ = self.rssm.imagine_step(h, z, actions[:, i])
            frames.append(self.decode(h, z))
        return torch.stack(frames, dim=1)

    @torch.no_grad()
    def imagine(self, initial_h, initial_z, actions):
        """Generate future frames from latent state and action sequence. actions: B,T,A."""
        h, z = initial_h, initial_z
        frames, states = [], []
        for i in range(actions.shape[1]):
            h, z, _ = self.rssm.imagine_step(h, z, actions[:, i])
            frames.append(self.decode(h, z))
            states.append(torch.cat([h, z], dim=-1))
        return torch.stack(frames, dim=1), torch.stack(states, dim=1)
