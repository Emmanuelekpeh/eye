"""Run: python -m pytest -p no:xonsh tests/test_hybrids.py   (component swap between real EyeWorldModel state dicts, tiny config)."""
import dataclasses
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import pytest
import torch

import make_hybrids as MH
from ivs.config import ModelConfig
from ivs.model import EyeWorldModel

CFG = ModelConfig(image_size=32, channels=1, feature_dim=16, deter_dim=16, stoch_dim=8, hidden_dim=16)


def _model(seed, cfg=CFG):
    torch.manual_seed(seed)
    return EyeWorldModel(cfg)


def _save(tmp, name, seed, cfg=CFG):
    p = Path(tmp) / f"{name}.pt"
    torch.save(dict(model=_model(seed, cfg).state_dict(), config=dataclasses.asdict(cfg), mean_image=torch.zeros(1, 32, 32)), p)
    return f"{name}={p}"


def test_components_come_from_the_named_sources_and_the_result_loads_and_runs():
    with tempfile.TemporaryDirectory() as tmp:
        a, b = _save(tmp, "a", 1), _save(tmp, "b", 2)
        MH.main(["--src", a, b, "--hybrid", "h=E:a,R:b,D:a", "--out", tmp])
        h = torch.load(Path(tmp) / "h.pt", weights_only=False)
        sa, sb = _model(1).state_dict(), _model(2).state_dict()
        for k, v in h["model"].items():
            want = sb if k.startswith("rssm.") else sa
            assert torch.equal(v, want[k]), k
        assert set(h["model"]) == set(sa) and h["hybrid"] == {"E": "a", "R": "b", "D": "a"}
        m = EyeWorldModel(ModelConfig(**h["config"]))
        m.load_state_dict(h["model"])
        out = m.observe(torch.rand(2, 3, 1, 32, 32), torch.zeros(2, 3, 5))
        assert torch.isfinite(out["recon"]).all()


def test_random_source_is_deterministic_and_differs_from_trained_models():
    with tempfile.TemporaryDirectory() as tmp:
        a = _save(tmp, "a", 1)
        for tag in ("x", "y"):
            MH.main(["--src", a, "--hybrid", f"{tag}=E:random,R:a,D:a", "--out", tmp])
        x, y = (torch.load(Path(tmp) / f"{t}.pt", weights_only=False)["model"] for t in "xy")
        assert all(torch.equal(x[k], y[k]) for k in x)
        assert not torch.equal(x["encoder.net.0.weight"], _model(1).state_dict()["encoder.net.0.weight"])


def test_a_config_saved_before_geom_dim_existed_is_the_same_architecture():
    with tempfile.TemporaryDirectory() as tmp:
        a = _save(tmp, "a", 1)
        old = dataclasses.asdict(CFG)
        del old["geom_dim"]                                    # what pre-geometry checkpoints stored
        p = Path(tmp) / "b.pt"
        torch.save(dict(model=_model(2).state_dict(), config=old, mean_image=torch.zeros(1, 32, 32)), p)
        MH.main(["--src", a, f"b={p}", "--hybrid", "h=E:a,R:b,D:a", "--out", tmp])
        assert (Path(tmp) / "h.pt").exists()


def test_mismatched_configs_and_malformed_specs_are_refused():
    with tempfile.TemporaryDirectory() as tmp:
        a = _save(tmp, "a", 1)
        b = _save(tmp, "b", 2, dataclasses.replace(CFG, deter_dim=24))
        with pytest.raises(ValueError, match="share one config"):
            MH.main(["--src", a, b, "--hybrid", "h=E:a,R:b,D:a", "--out", tmp])
    with pytest.raises(ValueError):
        MH.parse_hybrid("h=E:a,R:b")
    with pytest.raises(ValueError):
        MH.parse_hybrid("h=E:a,R:b,E:c")
