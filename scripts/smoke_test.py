from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from ivs.config import ModelConfig
from ivs.model import EyeWorldModel
from ivs.learning import world_model_loss, gaussian_kl
from ivs.actions import InterventionSampler

cfg = ModelConfig(image_size=64)
model = EyeWorldModel(cfg)
x = torch.rand(2, 4, 1, 64, 64)
a = torch.zeros(2, 4, cfg.action_dim)
x_aug, a_aug = InterventionSampler(p=1.0)(x)
assert x_aug.shape == x.shape and a_aug.shape == a.shape and a_aug[:, 0].abs().max() == 0
out = model.observe(x_aug, a_aug)
pred = model.rollout(out["h"][:, 1], out["z"][:, 1], a_aug[:, 2:])
loss, metrics = world_model_loss(x_aug, out, pred_frames=pred, pred_targets=x_aug[:, 2:])
loss.backward()
assert torch.isfinite(loss)
assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
mu, std = torch.zeros(3, 8), torch.ones(3, 8)
kl_identity = gaussian_kl((mu, std), (mu, std)).abs().max().item()
assert kl_identity < 1e-5, kl_identity
print("smoke test passed")
print("recon:", tuple(out["recon"].shape), "pred:", tuple(pred.shape))
print("loss:", {k: round(float(v), 4) for k, v in metrics.items()})
