"""Which components of two EyeWorldModel checkpoints are the same weights, close relatives, or unrelated? (CPU, reads only)
Components follow the model's own modules: encoder, rssm.gru, rssm.prior, rssm.posterior, decoder (geom_head is ignored).
For each pair and component: rel = ||a - b|| / ||a|| over the concatenated weight matrices (0 = identical weights, ~1.4 = two independent tensors of equal scale)
and cos = cosine of the two flattened vectors (1 = same direction, ~0 = unrelated). Only checkpoints with identical tensor shapes are compared.
  python scripts/lineage_map.py --models L1=<model.pt> adv=<model.pt> ... --out pilot_results/hybrid_eval/lineage.json
"""
import argparse
import json
import sys
from pathlib import Path

import torch

COMPONENTS = ("encoder", "rssm.gru", "rssm.prior", "rssm.posterior", "decoder")


def component(sd, comp):
    """Flat vector of the WEIGHT MATRICES / kernels (ndim >= 2) of one component, sorted by key so two checkpoints line up. Biases and norm gains
    are left out on purpose: they start at 0 / 1 in every network, which would make two unrelated models look correlated."""
    keys = sorted(k for k in sd if k.startswith(comp + ".") and sd[k].ndim >= 2)
    return keys, torch.cat([sd[k].flatten().float() for k in keys]) if keys else torch.zeros(0)


def distance(sd_a, sd_b):
    """{component: dict(rel, cos, n)} or dict(shape_mismatch=True) when the tensors differ in shape."""
    out = {}
    for c in COMPONENTS:
        ka, a = component(sd_a, c)
        kb, b = component(sd_b, c)
        if ka != kb or any(sd_a[k].shape != sd_b[k].shape for k in ka):
            out[c] = dict(shape_mismatch=True)
            continue
        out[c] = dict(rel=float((a - b).norm() / a.norm().clamp_min(1e-12)),
                      cos=float(torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(1e-12)), n=int(a.numel()))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="name=path ...")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    specs = [m.split("=", 1) for m in a.models]
    sds = {n: torch.load(p, map_location="cpu", weights_only=False)["model"] for n, p in specs}
    names = [n for n, _ in specs]
    res = {}
    for i, n in enumerate(names):
        for m in names[i + 1:]:
            res[f"{n} vs {m}"] = distance(sds[n], sds[m])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"{'pair':<22}" + "".join(f"{c:>26}" for c in COMPONENTS))
    for pair, d in res.items():
        print(f"{pair:<22}" + "".join(f"{('mismatch' if 'shape_mismatch' in d[c] else 'rel %.3f cos %.3f' % (d[c]['rel'], d[c]['cos'])):>26}" for c in COMPONENTS))
    return res


if __name__ == "__main__":
    main()
