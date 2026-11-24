"""Run the mechanistic-interpretability lab (ivs/lab.py) over models and cache the results under pilot_results/lab/<model>/ (results.json + PNGs).
  python scripts/run_lab.py --only L1_v6_long          # one model (regex on the id)
  python scripts/run_lab.py                            # tiers eye + some, plus the blob controls
  python scripts/run_lab.py --compare                  # comparative anatomy: linear CKA between models on the same stimuli -> pilot_results/lab/cka.json
  python scripts/run_lab.py --report                   # summary table + cross-model statistics -> pilot_results/lab/summary.json
Resumable: a model is skipped when the stored run has the same file size and lab settings (--force reruns). CPU only."""
import argparse
import json
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from ivs import eyemetric as E       # noqa: E402
from ivs import lab, modelzoo        # noqa: E402

CONTROLS = r"V3_bal99_wide/|V5_|V6_|O2_|F0_base|D0_video|F3_embed"      # blob makers (same architecture where possible) as negative controls


def pick(zoo, tiers, controls, only):
    out = []
    for e in zoo:
        if only and not re.search(only, e["id"]):
            continue
        if only or e["tier"] in tiers or (controls and re.search(controls, e["id"])):
            out.append(e)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--tiers", default="eye,some")
    ap.add_argument("--controls", default=CONTROLS)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--sections", default=None, help="comma list, default all")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    zoo = modelzoo.discover(HERE)
    if a.compare:
        return compare(zoo)
    if a.report:
        return report(zoo)
    lc = lab.LabConfig.quick() if a.quick else lab.LabConfig()
    if a.sections:
        lc.sections = tuple(a.sections.split(","))
    seg = E.load_segmenter(str(lab.SEG_PATH))
    todo = pick(zoo, a.tiers.split(","), a.controls, a.only)
    print(f"{len(todo)} models", flush=True)
    ctrl_cache, t0 = {}, time.time()
    for i, e in enumerate(todo):
        old = lab.load_results(e["id"], e["bytes"], HERE)
        if old and old.get("lab") == json.loads(json.dumps(asdict(lc))) and not a.force:
            print(f"[{i + 1}/{len(todo)}] {e['id']}: cached", flush=True)
            continue
        t1 = time.time()
        res, imgs = lab.analyse(HERE / e["id"], seg, lc, progress=lambda s, f: print(f"    {s}", flush=True), controls=ctrl_cache)
        lab.save_results(e["id"], e["bytes"], res, imgs, HERE)
        c = res.get("capacity", {})
        print(f"[{i + 1}/{len(todo)}] {e['label']} ({e['tier']}): active_kl {c.get('active_kl')}/{c.get('z_dim')} pr_hz {c.get('pr', {}).get('hz', 0):.1f}  "
              f"[{time.time() - t1:.0f}s, total {time.time() - t0:.0f}s]", flush=True)


def _arch(r):
    c = r["config"]
    return f"{c['feature_dim']}/{c['deter_dim']}/{c['stoch_dim']}"


def compare(zoo, n_clips=12, ticks=30):
    """Linear CKA between all 96 px RGB models on identical stimuli = clips dreamed by L1 and by `adv` (both models' own renders, so the stimuli
    are eye-like; this biases towards those two, which is why seed pairs are reported as the reference)."""
    seg = None
    ref = {e["label"]: e for e in zoo}
    clips = []
    for label in ("L1_v6_long", "adv"):
        m = lab.load(HERE / ref[label]["id"])
        fr = lab.as_float(lab.dream(m, ticks + 10, n_clips // 2, 21)["frames"][10:])
        clips.append(fr.permute(1, 0, 2, 3, 4))
    clips = torch.cat(clips)
    ids = [e for e in zoo if e["size"] == 96 and e["channels"] == 3]
    feats = {}
    for e in ids:
        feats[e["id"]] = lab.stimulus_features(lab.load(HERE / e["id"]), clips)
        print("features", e["label"], flush=True)
    names = [e["id"] for e in ids]
    out = dict(models=[dict(id=e["id"], label=e["label"], group=e["group"], tier=e["tier"], params=e["params"]) for e in ids], stimuli=int(clips.shape[0] * clips.shape[1]))
    for kind in ("embed", "h", "z"):
        M = np.full((len(names), len(names)), np.nan)
        for i, p in enumerate(names):
            for j, q in enumerate(names):
                if j >= i:
                    M[i, j] = M[j, i] = lab.linear_cka(feats[p][kind], feats[q][kind])
        out[kind] = M.tolist()
    (HERE / "pilot_results" / "lab").mkdir(parents=True, exist_ok=True)
    (HERE / "pilot_results" / "lab" / "cka.json").write_text(json.dumps(lab._clean(out)))
    print("wrote cka.json", flush=True)
    return out


FAMILY = 0.05      # relative weight distance to L1 below this = a warm start / fine-tune of L1, not an independent training run


def lineage(entries, ref_label="L1_v6_long"):
    """||w - w_L1||^2 / ||w_L1||^2 for every model with L1's tensor layout: ~0 = fine-tune of L1 (the LR*, ctl, adv runs), ~1 = unrelated init.
    Models in one family are not independent samples, so cross-model statistics over them overstate n."""
    ref = next(e for e in entries if e["label"] == ref_label)
    base = {k: v.detach() for k, v in lab.load(HERE / ref["id"]).state_dict().items()}
    den = sum(float(v.pow(2).sum()) for v in base.values())
    out = {}
    for e in entries:
        w = lab.load(HERE / e["id"]).state_dict()
        if w.keys() == base.keys() and all(w[k].shape == base[k].shape for k in base):
            out[e["id"]] = sum(float((w[k].detach() - base[k]).pow(2).sum()) for k in base) / den
    return out


def cka_summary(zoo, lin):
    """Mean CKA of embedding / h / z over model pairs of the 4.14 M architecture (cka.json from --compare), by lineage and by tier."""
    p = HERE / "pilot_results" / "lab" / "cka.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    M, mats = d["models"], {k: np.array(d[k], float) for k in ("embed", "h", "z")}
    idx = [i for i, m in enumerate(M) if m["id"] in lin]
    fam = lambda i: lin[M[i]["id"]] < FAMILY
    tier = lambda i: M[i]["tier"]
    pairs = [(a, b) for k, a in enumerate(idx) for b in idx[k + 1:]]
    classes = {"both L1 lineage": [q for q in pairs if fam(q[0]) and fam(q[1])], "both independent init": [q for q in pairs if not fam(q[0]) and not fam(q[1])],
               "one of each": [q for q in pairs if fam(q[0]) != fam(q[1])]}
    ind = [q for q in pairs if not fam(q[0]) and not fam(q[1])]
    for a, b in (("eye", "eye"), ("eye", "some"), ("eye", "no"), ("some", "some"), ("some", "no"), ("no", "no")):
        classes[f"independent only, {a}-{b}"] = [q for q in ind if {tier(q[0]), tier(q[1])} == {a, b}]
    out = {}
    print("CKA between models (4.14 M architecture, same stimuli):")
    for name, ps in classes.items():
        if ps:
            out[name] = dict(n=len(ps), **{k: float(np.mean([mats[k][a, b] for a, b in ps])) for k in mats})
            print(f"  {name:32s} n={len(ps):3d}  embed {out[name]['embed']:.3f}  h {out[name]['h']:.3f}  z {out[name]['z']:.3f}")
    return out


def report(zoo):
    from scipy.stats import spearmanr
    rows = []
    for e in zoo:
        r = lab.load_results(e["id"], e["bytes"], HERE)
        if r:
            rows.append((e, r))
    lin = lineage([e for e, _ in rows])
    summ = {}
    for e, r in rows:
        c, p = r.get("capacity", {}), r.get("probes", {})
        st, ls, dy = r.get("steer", {}), r.get("lesion", {}), r.get("dynamics", {})
        best = {t: p[t]["r2"]["hz"] for t in p if p[t]["r2"]["hz"] is not None}
        gain = {t: (p[t]["r2"]["hz"] - (p[t]["r2_random_init"] or 0)) for t in p if p[t]["r2"]["hz"] is not None and p[t].get("r2_random_init") is not None}
        summ[e["id"]] = dict(label=e["label"], tier=e["tier"], arch=_arch(r), active_kl=c.get("active_kl"), z_dim=c.get("z_dim"), kl_total=c.get("kl_total"),
                             pr_hz=c.get("pr", {}).get("hz"), n90_hz=c.get("n90", {}).get("hz"), h_silent=c.get("h_silent"),
                             r2=best, r2_vs_random=gain, jac=dy.get("jacobian", {}).get("radius_mean"), lyap=dy.get("perturbation", {}).get("lambda_per_tick"),
                             saturation=dy.get("perturbation", {}).get("saturation"),
                             collapse=(dy.get("noisy", {}).get("world_dispersion") or [None])[-1], collapse_det=(dy.get("deterministic", {}).get("world_dispersion") or [None])[-1],
                             eye_first=dy.get("noisy", {}).get("eye_first"), eye_last=dy.get("noisy", {}).get("eye_last"),
                             lesion_active_pix=(ls.get("groups", {}).get("z_active") or {}).get("pix"), lesion_random_pix=float(np.mean([v["pix"] for k, v in ls.get("groups", {}).items() if k.startswith("random")] or [np.nan])),
                             lesion_inactive_pix=(ls.get("groups", {}).get("z_inactive") or {}).get("pix"), lineage=lin.get(e["id"]),
                             **{f"r2_{t}": v for t, v in best.items()})
    out = dict(models=summ)
    n_ind = {t: sum(1 for s in summ.values() if s["lineage"] is not None and s["lineage"] >= FAMILY and s["tier"] == t) for t in ("eye", "some", "no")}
    n_fam = sum(1 for s in summ.values() if s["lineage"] is not None and s["lineage"] < FAMILY)
    print(f"{n_fam} models are fine-tunes of L1 (weight distance < {FAMILY}); independent inits of the 4.14 M architecture: {n_ind}")
    tier = {"eye": 2, "some": 1, "no": 0}
    print(f"{len(summ)} models with a lab run")
    for arch in sorted(set(s["arch"] for s in summ.values())):
        ss = [s for s in summ.values() if s["arch"] == arch and s["tier"] in tier]
        if len(ss) < 6 or len(set(s["tier"] for s in ss)) < 2:
            continue
        y = np.array([tier[s["tier"]] for s in ss])
        tests = []
        for key in ("active_kl", "kl_total", "pr_hz", "n90_hz", "h_silent", "jac", "lyap", "saturation", "collapse", "collapse_det", "eye_first", "eye_last",
                    *[f"r2_{t}" for t in lab.TARGETS]):
            x = np.array([np.nan if s[key] is None else s[key] for s in ss], float)
            ok = np.isfinite(x)
            if ok.sum() >= 6 and x[ok].std() > 0 and len(set(y[ok])) > 1:
                rho, pv = spearmanr(x[ok], y[ok])
                tests.append((key, float(rho), float(pv), int(ok.sum())))
        m = len(tests)
        for t in tests:
            print(f"  arch {arch} n={len(ss)}  {t[0]:14s} rho {t[1]:+.2f} p {t[2]:.3f} (Bonferroni x{m}: {min(1, t[2] * m):.3f})")
        out.setdefault("tests", {})[arch] = [dict(metric=t[0], rho=t[1], p=t[2], p_bonferroni=min(1, t[2] * m), n=t[3]) for t in tests]
    out["cka"] = cka_summary(zoo, lin)
    (HERE / "pilot_results" / "lab" / "summary.json").write_text(json.dumps(lab._clean(out), indent=1))
    return out


if __name__ == "__main__":
    main()
