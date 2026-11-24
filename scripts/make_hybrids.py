"""Plug one model's encoder / RSSM / decoder into another (CPU, reads checkpoints, writes new files only).
Components: E = `encoder.*`, R = `rssm.*` (gru + prior + posterior: the posterior is what reads the encoder's embedding), D = `decoder.*`.
Sources are named checkpoints; the source `random` is a freshly initialised EyeWorldModel of the same config (seed --random-seed): a control for "completely broken".
All sources must share one ModelConfig (same shapes), otherwise the swap is refused. Output files are named `<hybrid>.pt` (NOT model.pt), so ivs.modelzoo never lists them in the UI.
  python scripts/make_hybrids.py --src L1=<model.pt> adv=<model.pt> --hybrid "L1dyn_advdec=E:L1,R:L1,D:adv" --out pilot_results/hybrid_eval/ckpt
"""
import argparse
import dataclasses
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from ivs.config import ModelConfig       # noqa: E402
from ivs.model import EyeWorldModel      # noqa: E402

PREFIX = {"E": ("encoder.",), "R": ("rssm.",), "D": ("decoder.",)}


def parse_hybrid(text):
    """'name=E:a,R:b,D:c' -> (name, {'E': 'a', 'R': 'b', 'D': 'c'}); all three components must be given exactly once."""
    name, spec = text.split("=", 1)
    parts = dict(p.split(":", 1) for p in spec.split(","))
    if set(parts) != set(PREFIX) or len(spec.split(",")) != 3:
        raise ValueError(f"hybrid {name!r} needs exactly E:, R: and D: sources, got {spec!r}")
    return name, parts


def load_source(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return dict(model=ck["model"], config=ck["config"], mean_image=ck.get("mean_image"))


def random_source(config, seed):
    torch.manual_seed(seed)
    return dict(model=EyeWorldModel(ModelConfig(**config)).state_dict(), config=config, mean_image=None)


def combine(sources, parts):
    """State dict whose encoder / rssm / decoder tensors come from the sources named in parts; raises on config mismatch or uncovered keys."""
    # canonical config: older checkpoints do not store fields added later (e.g. geom_dim=0), which is the same architecture
    cfgs = {json.dumps(dataclasses.asdict(ModelConfig(**sources[s]["config"])), sort_keys=True) for s in parts.values() if s != "random"}
    if len(cfgs) != 1:
        raise ValueError(f"sources of one hybrid must share one config, got {len(cfgs)} different")
    sd, taken = {}, set()
    for comp, src in parts.items():
        for k, v in sources[src]["model"].items():
            if k.startswith(PREFIX[comp]):
                sd[k] = v.clone()
                taken.add(k)
    base = sources[next(s for s in parts.values() if s != "random")]["model"]
    extra = sorted(set(base) - taken)
    if extra:
        raise ValueError(f"keys not covered by E/R/D (a geometry head?): {extra[:4]}")
    return sd


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True, help="name=path ...")
    ap.add_argument("--hybrid", nargs="+", required=True, help="name=E:src,R:src,D:src ...")
    ap.add_argument("--out", required=True)
    ap.add_argument("--random-seed", type=int, default=12345)
    a = ap.parse_args(argv)
    sources = {n: load_source(p) for n, p in (s.split("=", 1) for s in a.src)}
    first = next(iter(sources.values()))
    sources["random"] = random_source(first["config"], a.random_seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for text in a.hybrid:
        name, parts = parse_hybrid(text)
        sd = combine(sources, parts)
        ck = dict(model=sd, config=sources[parts["R"]]["config"], mean_image=sources[next(s for s in (parts["R"], parts["E"], parts["D"]) if s != "random")]["mean_image"],
                  hybrid=parts)
        torch.save(ck, out / f"{name}.pt")
        print(f"{name}: E<-{parts['E']} R<-{parts['R']} D<-{parts['D']}  ({sum(v.numel() for v in sd.values())} params)", flush=True)


if __name__ == "__main__":
    main()
