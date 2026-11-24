"""Held-out clarity of RSSM checkpoints (R3 pack, 96 px): does a training change make UNSEEN eyes sharper? Companion of scripts/overfit_rssm.py (seen frames).
Protocol = the one the R5 'l1' base used (refine_gan.Base.single): every held-out view (exactly what CacheBank(split='val', image_size=96) shows; 344 frames of 42 recordings
the model never trained on, data/heldout_pack) is encoded as a ONE-frame window from the zero state and decoded from the posterior sample (seeded), scored with
heldout_clarity.score_items against the native frames next to the CEILING of the same view, with heldout_clarity's held-out gate (0.8 x ceiling).
Reference numbers in plan.md / results files: L1 openeds edge_keep .353, plain AE512 (R3) .401, ceiling .597.
  python scripts/heldout_rssm.py score --pack ../data/heldout_pack --models L1=<model.pt> FT0=<model.pt> FT1=<model.pt> --out pilot_results/heldout_rssm
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import heldout_clarity as H                                         # noqa: E402
import overfit_clarity as O                                         # noqa: E402
from ivs.config import ModelConfig                                  # noqa: E402
from ivs.model import EyeWorldModel                                 # noqa: E402

SIZE = 96
SEED = 123
AE_REF = HERE / "pilot_results" / "kaggle_heldout1" / "heldout" / "results.json"      # plain AE512 per height (R3)


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = EyeWorldModel(ModelConfig(**ck["config"]))
    model.load_state_dict(ck["model"])
    return model.eval()


@torch.no_grad()
def recon_views(model, items, chunk=48, seed=SEED):
    """One (96,96,3) uint8 reconstruction per held-out item: one-frame window, zero state, zero action, posterior sample (seeded)."""
    out = []
    torch.manual_seed(seed)
    for i in range(0, len(items), chunk):
        x = torch.from_numpy(np.stack([it["view"] for it in items[i:i + chunk]])).permute(0, 3, 1, 2).float() / 255.0
        o = model.observe(x[:, None], torch.zeros(x.shape[0], 1, model.cfg.action_dim))
        y = o["recon"][:, 0].clamp(0, 1).permute(0, 2, 3, 1).numpy()
        out.extend((y * 255).round().astype(np.uint8))
    return out


def score_model(model, items, ceil):
    scores = H.score_items(items, recon_views(model, items))
    res = {}
    for s, v in scores.items():
        gate = H.gate_of(v, ceil[s])
        res[s] = dict(scores=v, over_ceiling_keep=v["edge_keep"] / ceil[s]["edge_keep"], over_ceiling_fidelity=v["edge_fidelity"] / ceil[s]["edge_fidelity"],
                      gate=gate, passed=bool(all(gate.values())))
    return res


def sheet(items, recs_by_model, out_path):
    """Per slice, two held-out frames: native | ceiling | each model on the same most-detailed central patch."""
    pos = {it["i"]: p for p, it in enumerate(items)}
    rows = []
    for s, ids in H.sheet_ids_of(PACK_FOR_SHEET[0]).items():
        ids = [pos[i] for i in ids if i in pos]
        for i in ids:
            it = items[i]
            colour = it["slice"] == "video_colour"
            g = (lambda x: x) if colour else O.gray_of
            nat = it["native"] if colour else it["native_g"]
            side = min(C_PATCH, *(min(it["native_g"].shape) // 2,))
            y, x, sz = H.central_patch(it["native_g"], side)
            tiles = []
            for name, im in [("native", nat), ("ceiling", g(H.to_native(it["view"], it)))] + [(m, g(H.to_native(r[i], it))) for m, r in recs_by_model.items()]:
                t = np.ascontiguousarray(im[y:y + sz, x:x + sz])
                t = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR) if t.ndim == 2 else cv2.cvtColor(t, cv2.COLOR_RGB2BGR)
                t = cv2.resize(t, (180, 180), interpolation=cv2.INTER_NEAREST)
                cv2.putText(t, f"{s[:7]} {name}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
                tiles.append(t)
            rows.append(np.concatenate(tiles, 1))
    cv2.imwrite(str(out_path), np.concatenate(rows, 0))


PACK_FOR_SHEET = [None]
C_PATCH = 160


def score_main(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    PACK_FOR_SHEET[0] = a.pack
    torch.set_num_threads(a.threads)
    items = H.build_items(a.pack, SIZE)
    ceil = H.score_items(items, [it["view"] for it in items])
    print(f"{len(items)} held-out frames; ceilings " + " | ".join(f"{s} keep {v['edge_keep']:.3f} fid {v['edge_fidelity']:.3f}" for s, v in ceil.items()), flush=True)
    results, recs = dict(ceiling=ceil, models={}), {}
    for spec in a.models:
        name, _, path = spec.partition("=")
        model = load_model(path)
        recs[name] = recon_views(model, items)
        results["models"][name] = dict(path=path, **{"slices": score_model(model, items, ceil)})
        for s, r in results["models"][name]["slices"].items():
            sc = r["scores"]
            print(f"{name:12s} {s:13s} keep {sc['edge_keep']:.3f} (x{r['over_ceiling_keep']:.2f}) fidelity {sc['edge_fidelity']:.3f} (x{r['over_ceiling_fidelity']:.2f}) psnr {sc['psnr']:.1f} "
                  f"pupil_edge {sc.get('pupil_edge_grad_ratio', float('nan')):.3f}  {'PASS' if r['passed'] else 'FAIL ' + ','.join(k for k, v in r['gate'].items() if not v)}", flush=True)
    if AE_REF.exists():
        ae = json.loads(AE_REF.read_text()).get(str(SIZE))
        if ae:
            results["ae512_r3"] = {s: v["ae"] for s, v in ae["slices"].items()}
            for s, v in results["ae512_r3"].items():
                print(f"{'AE512 (R3)':12s} {s:13s} keep {v['edge_keep']:.3f} fidelity {v['edge_fidelity']:.3f} psnr {v['psnr']:.1f}")
    (out / "results.json").write_text(json.dumps(results, indent=1, default=float))
    sheet(items, recs, out / "sheet_heldout_rssm.png")
    return results


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--pack", required=True)
    s.add_argument("--models", nargs="+", required=True, help="NAME=model.pt ...")
    s.add_argument("--out", required=True)
    s.add_argument("--threads", type=int, default=4)
    return score_main(ap.parse_args(argv))


if __name__ == "__main__":
    main()
