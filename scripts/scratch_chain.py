"""One arm of the SCRATCH run (plan.md "SCRATCH"): stage 1 = L1's recipe from RANDOM weights (ivs.pilot preset N0_scratch, T=8 -> 16), stage 2 = the adversarial dream stage of
scripts/adv_rssm.py on that model, all inside one wall-clock deadline (the GPU budget is a hard hour). Arms differ ONLY in the data stage 1 sees:
  ctl = real recordings (+ the mrl images, image_weight .3 as L1)           --source-weights video=.5,openeds=.5
  mix = the same + the dream pool (source 'dream', train_only), 25 % share   --source-weights video=.375,openeds=.375,dream=.25
Same seed, so the two random initialisations are IDENTICAL. Stage 2 is identical for both arms and sees real video / OpenEDS only (its discriminator's 'real' side must be real).
Time: stage 1 is a fixed step count (L1's measured T=8 speed on a T4 is 6.8 steps/s, T=16 3.8); stage 2 gets whatever is left of --deadline-min minus a safety margin
(bank load + final probe + save), with its step count set to what that time buys (70 steps/min measured for the adv arm at batch 24), so its lr schedule ends on time.
Outputs: <out>/<arm>/stage1/N0_scratch/{model.pt, metrics.json, train_log.jsonl, sheets}, <out>/<arm>/stage2/adv/{model.pt, model_s1.pt, model_s2.pt, train_log.jsonl, summary.json}, <out>/<arm>/chain.json.
  python scripts/scratch_chain.py --arm mix --real-caches <videos> <openeds> --image-cache <mrl> --dream-cache <dreams> --out <dir> --steps1 8500 --deadline-min 50
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

WEIGHTS = {"ctl": "video=0.5,openeds=0.5", "mix": "video=0.375,openeds=0.375,dream=0.25"}
STAGE2_SAFETY_MIN = 3.0          # bank reload (~1.5 min) + final probe + save
STAGE2_STEPS_PER_MIN = 70        # adv arm, batch 24, T4, two processes sharing the machine (kaggle_r5s2: 3000 steps in 40.9 min)
STAGE2_MIN_MINUTES = 2.0


def stage2_minutes(deadline_min, elapsed_min):
    """Minutes stage 2 may train for (0 = skip it): what is left of the deadline after the safety margin."""
    left = deadline_min - elapsed_min - STAGE2_SAFETY_MIN
    return round(left, 2) if left >= STAGE2_MIN_MINUTES else 0.0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(WEIGHTS), required=True)
    ap.add_argument("--real-caches", nargs="+", required=True, help="video + OpenEDS CacheBank dirs")
    ap.add_argument("--image-cache", default=None, help="appearance cache (mrl) for L1's image loss")
    ap.add_argument("--dream-cache", default=None, help="dream pool (make_dream_pool.py); required for arm mix, ignored for ctl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps1", type=int, default=8500)
    ap.add_argument("--deadline-min", type=float, default=50.0, help="minutes after --t0 by which everything must be finished")
    ap.add_argument("--t0", type=float, default=None, help="epoch seconds of the deadline clock's start (kernel start); default = now")
    ap.add_argument("--batch1", type=int, default=None, help="stage-1 batch (default: the preset's 32)")
    ap.add_argument("--adv-batch", type=int, default=24)
    ap.add_argument("--lr2", type=float, default=1e-4)
    ap.add_argument("--w-adv", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--ckpt-every", type=int, default=2000)
    a = ap.parse_args(argv)
    if a.arm == "mix" and not a.dream_cache:
        raise SystemExit("arm mix needs --dream-cache")
    t_start = a.t0 if a.t0 is not None else time.time()
    out = Path(a.out) / a.arm
    out.mkdir(parents=True, exist_ok=True)
    from ivs import pilot
    import adv_rssm
    import torch

    caches = list(a.real_caches) + ([a.image_cache] if a.image_cache else []) + ([a.dream_cache] if a.arm == "mix" else [])
    s1 = ["--cache", *caches, "--source-weights", WEIGHTS[a.arm], "--out", str(out / "stage1"), "--runs", "N0_scratch", "--steps", str(a.steps1),
          "--device", a.device, "--seed", str(a.seed), "--ckpt-every", str(a.ckpt_every), "--log-every", "100"] + (["--batch", str(a.batch1)] if a.batch1 else [])
    print(f"[{a.arm}] stage 1: {a.steps1} steps from random init, caches {caches}, weights {WEIGHTS[a.arm]}", flush=True)
    t1 = time.time()
    pilot.main(s1)
    ckpt = out / "stage1" / "N0_scratch" / "model.pt"
    assert ckpt.exists(), f"stage 1 left no {ckpt}"
    elapsed = (time.time() - t_start) / 60
    minutes = stage2_minutes(a.deadline_min, elapsed)
    chain = dict(arm=a.arm, weights=WEIGHTS[a.arm], steps1=a.steps1, stage1_min=round((time.time() - t1) / 60, 1), elapsed_at_stage2_min=round(elapsed, 1), stage2_budget_min=minutes)
    print(f"[{a.arm}] stage 1 done after {chain['stage1_min']} min (clock {elapsed:.1f} of {a.deadline_min}); stage 2 budget {minutes} min", flush=True)
    if minutes <= 0:
        chain["stage2"] = "skipped: no time left"
    else:
        device = torch.device("cuda" if a.device == "auto" and torch.cuda.is_available() else ("cpu" if a.device == "auto" else a.device))
        steps2 = max(10, int(minutes * STAGE2_STEPS_PER_MIN))
        _, summ = adv_rssm.train_arm("adv", str(ckpt), a.real_caches, str(out / "stage2"), steps2, minutes, a.adv_batch, a.lr2, 2e-4, a.w_adv, device, a.seed,
                                     probe_every=max(25, steps2 // 6))
        chain["stage2"] = dict(steps_done=summ["steps_done"], steps_planned=steps2, minutes=summ["minutes"])
    chain["total_min"] = round((time.time() - t_start) / 60, 1)
    (out / "chain.json").write_text(json.dumps(chain, indent=1))
    print(f"[{a.arm}] chain done: {json.dumps(chain)}", flush=True)
    return chain


if __name__ == "__main__":
    main()
