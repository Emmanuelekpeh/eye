from dataclasses import dataclass


@dataclass
class ModelConfig:
    image_size: int = 64
    channels: int = 1                 # eye-tracking datasets are often infrared
    feature_dim: int = 128
    deter_dim: int = 128
    stoch_dim: int = 32
    action_dim: int = 5
    min_std: float = 0.1
    conv_mult: float = 1.0            # conv channel multiplier (1.0 = original 32/48/64/96)
    hidden_dim: int = 128             # prior/posterior MLP width
    geom_dim: int = 0                 # >0: auxiliary eye-geometry head on the latent state (ivs.gaze.GEOM_DIM); 0 = no head


@dataclass
class TrainConfig:
    data: str = "data/eyes"
    sequence_length: int = 8          # needs >= 4: first half = context, second half = predicted
    batch_size: int = 4
    epochs: int = 10
    learning_rate: float = 2e-4
    kl_weight: float = 1.0            # recon is now summed over pixels, so beta=1 is sane
    free_nats: float = 1.0            # KL below this (per step) is not penalised -> avoids collapse
    kl_balance: float = 0.8           # weight on prior-training KL term (Dreamer-style)
    pred_weight: float = 1.0
    grad_clip: float = 100.0
    intervention_prob: float = 0.5    # fraction of training clips given synthetic interventions
    val_frac: float = 0.2             # fraction of subject groups held out
    probe_every: int = 20
    seed: int = 0
    workers: int = 0
    device: str = "cpu"
    checkpoint_dir: str = "checkpoints"
    log_every: int = 20
    max_frames_per_video: int = 600
