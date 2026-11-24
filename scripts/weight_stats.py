"""Per-model weight statistics for every EyeWorldModel checkpoint the zoo finds (CPU, no data read).
Per module group: rms, top singular value, stable rank (|W|_F^2 / s_max^2), effective rank (exp of the entropy of the normalised
singular values). GRU recurrent blocks also get their spectral radius (max |eigenvalue| of each of the 3 gate blocks of weight_hh).
Output: pilot_results/weight_stats.json  {model id: {tier, params, config, groups:{name:{...}}, gru_radius:[r,z,n]}}.
Run: python scripts/weight_stats.py [--out PATH]"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ivs import modelzoo  # noqa: E402

GROUPS = (("enc_conv", "encoder.net.", True), ("enc_fc", "encoder.net.12.weight", False), ("gru_ih", "rssm.gru.weight_ih", False),
          ("gru_hh", "rssm.gru.weight_hh", False), ("prior", "rssm.prior.", False), ("posterior", "rssm.posterior.", False),
          ("dec_fc", "decoder.0.weight", False), ("dec_conv", "decoder.", True))


def spectrum(w):
    """w: any weight tensor with ndim>=2 -> matrix (out, rest). Returns rms, s_max, stable rank, effective rank."""
    m = w.reshape(w.shape[0], -1).double()
    s = torch.linalg.svdvals(m).numpy()
    p = s / s.sum()
    return dict(rms=float(m.pow(2).mean().sqrt()), smax=float(s[0]), stable_rank=float((s ** 2).sum() / s[0] ** 2),
                eff_rank=float(np.exp(-(p * np.log(p + 1e-12)).sum())), dim=int(min(m.shape)))


def group_weights(sd, prefix, conv):
    out = []
    for k, v in sd.items():
        if not k.endswith("weight") or v.ndim < 2 or not k.startswith(prefix):
            continue
        if conv and v.ndim != 4:
            continue
        if not conv and v.ndim == 4:
            continue
        if prefix == "encoder.net." and k == "encoder.net.12.weight":
            continue
        if prefix == "decoder." and k == "decoder.0.weight":
            continue
        out.append((k, v))
    return out


def stats(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model"]
    res = {}
    for name, prefix, conv in GROUPS:
        ws = group_weights(sd, prefix, conv)
        if not ws:
            continue
        per = [spectrum(v) for _, v in ws]
        res[name] = {k: float(np.mean([p[k] for p in per])) for k in per[0]}
        res[name]["layers"] = len(per)
    hh = sd["rssm.gru.weight_hh"].double()
    d = hh.shape[1]
    radius = [float(np.abs(np.linalg.eigvals(hh[i * d:(i + 1) * d].numpy())).max()) for i in range(3)]
    return ck["config"], res, radius


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "pilot_results" / "weight_stats.json"))
    a = ap.parse_args(argv)
    out = {}
    for e in modelzoo.discover(ROOT):
        cfg, groups, radius = stats(ROOT / e["id"])
        out[e["id"]] = dict(label=e["label"], group=e["group"], tier=e["tier"], params=e["params"], bytes=e["bytes"], config=cfg,
                            groups=groups, gru_radius=radius)
        print(e["label"], e["group"], e["tier"], flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main()
