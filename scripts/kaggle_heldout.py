"""Kaggle kernel entry for the held-out clarity tests (scripts/heldout_clarity.py, plan.md R3 / R3b). Private kernel, GPU on, internet off.
Inputs: theradactyl/eye-ivs-code (flat ivs/*.py), theradactyl/eye-heldout-tools (heldout_clarity.py, overfit_clarity.py, clarity_ceiling.py, eval_native.npz,
eval_meta.json), and one cache dataset per height (theradactyl/eye-hr96 / eye-hr128 / eye-hr192: videos + OpenEDS in one CacheBank directory each).
One process per GPU (all on the one GPU if only one is attached). PLAN (env HELDOUT_PLAN, default r3b) says what each process trains:
  r3   GPU 0: 192 px AE512; GPU 1: 96 px then 128 px AE512                                              (finished; results in pilot_results/kaggle_heldout1)
  r3b  128 px, GPU 0: AE512_ssim, S8_ssim, S16_ssim; GPU 1: AE512_s1, S8, S16 (arms in heldout_clarity.ARMS)          (finished; pilot_results/kaggle_heldout2)
  r3c  128 px, GPU 0: S16_ssim_x3 (30k steps); GPU 1: V1024_ssim, S32_ssim, F8_ssim
  r3c_data  128 px, one process: S16_ssim_full (whole-video cache, dataset eye-hr128f -> cache key 128f)
Outputs in /kaggle/working/heldout: results.json, summary.json (gate table + gains over AE512), verdicts.json (WORKS rule), sheet_<slice>*.png, log_part<i>.txt.
Reference for r3c gains = S16_ssim of kaggle_heldout2. Overridable for a local dry run: KAGGLE_INPUT, KAGGLE_WORKING, HELDOUT_PLAN, HELDOUT_STEPS, HELDOUT_BATCH, HELDOUT_ALLOW_CPU=1.
"""
import glob
import os
import re
import shutil
import subprocess
import sys
import time

INPUT = os.environ.get("KAGGLE_INPUT", "/kaggle/input")
WORK = os.environ.get("KAGGLE_WORKING", "/kaggle/working")
STEPS = int(os.environ.get("HELDOUT_STEPS", "10000"))             # edited at push time if needed
BATCH = int(os.environ.get("HELDOUT_BATCH", "128"))
PLANS = {
    "r3": [([192], ["AE512"]), ([96, 128], ["AE512"])],
    "r3b": [([128], ["AE512_ssim", "S8_ssim", "S16_ssim"]), ([128], ["AE512_s1", "S8", "S16"])],
    "r3c": [([128], ["S16_ssim_x3"]), ([128], ["V1024_ssim", "S32_ssim", "F8_ssim"])],
    "r3c_data": [([128], ["S16_ssim_full"])],
}
CACHE_SUFFIX = {"S16_ssim_full": "f"}                              # = heldout_clarity.ARMS[arm]["cache"]: the arm trains on the cache dataset eye-hr<H><suffix>
PLAN = os.environ.get("HELDOUT_PLAN", "r3c")
GROUPS = PLANS[PLAN]

t0 = time.time()
code = sorted(glob.glob(os.path.join(INPUT, "**", "aeprobe.py"), recursive=True))
tool = sorted(glob.glob(os.path.join(INPUT, "**", "heldout_clarity.py"), recursive=True))
assert code and tool, f"inputs not found: code={code} tool={tool}"
scripts = os.path.dirname(tool[0])
for need in ("overfit_clarity.py", "clarity_ceiling.py", "eval_native.npz", "eval_meta.json"):
    assert os.path.exists(os.path.join(scripts, need)), f"{need} missing next to heldout_clarity.py in {scripts}"
caches = {}
for m in sorted(glob.glob(os.path.join(INPUT, "**", "manifest.json"), recursive=True)):
    hit = re.search(r"hr(\d+f?)$", os.path.basename(os.path.dirname(m)))
    if hit:
        caches[hit.group(1)] = os.path.dirname(m)
need_h = sorted({f"{h}{CACHE_SUFFIX.get(arm, '')}" for hs, arms in GROUPS for h in hs for arm in arms})
assert all(h in caches for h in need_h), f"cache datasets missing: have {sorted(caches)}, need {need_h}"
pkg = os.path.join(WORK, "pkg", "ivs")
os.makedirs(pkg, exist_ok=True)
for f in glob.glob(os.path.join(os.path.dirname(code[0]), "*.py")):
    shutil.copy(f, pkg)
import torch  # noqa: E402

n_gpu = torch.cuda.device_count()
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(), "| devices", n_gpu, "| plan", PLAN, GROUPS, "| steps", STEPS, "batch", BATCH, "| caches", caches, "| scripts", scripts, flush=True)
if not torch.cuda.is_available() and os.environ.get("HELDOUT_ALLOW_CPU") != "1":
    raise SystemExit("no GPU attached: refusing to train autoencoders on a Kaggle CPU")

procs = []
for i, (heights, arms) in enumerate(GROUPS):
    cmd = [sys.executable, os.path.join(scripts, "heldout_clarity.py"), "train", "--cache", *[f"{k}={caches[k]}" for k in sorted({f"{h}{CACHE_SUFFIX.get(arm, '')}" for h in heights for arm in arms})], "--pack", scripts,
           "--out", os.path.join(WORK, f"part{i}"), "--heights", *map(str, heights), "--arms", *arms, "--steps", str(STEPS), "--batch", str(BATCH),
           "--device", "auto" if n_gpu else "cpu"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i % max(1, n_gpu)), PYTHONPATH=os.path.join(WORK, "pkg"), PYTHONUNBUFFERED="1")
    log = open(os.path.join(WORK, f"log_part{i}.txt"), "w")
    procs.append((i, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log))
failed = []
for i, p, log in procs:
    rc = p.wait()
    log.close()
    print(f"\n===== part {i} (exit {rc}) tail =====")
    print("".join(open(os.path.join(WORK, f"log_part{i}.txt")).readlines()[-12:])[:3500])
    if rc != 0:
        failed.append(i)

out = os.path.join(WORK, "heldout")
os.makedirs(out, exist_ok=True)
parts = []
for i, _, _ in procs:
    part = os.path.join(WORK, f"part{i}")
    for f in glob.glob(os.path.join(part, "*")):
        name = os.path.basename(f)
        dest = f"results_part{i}.json" if name == "results.json" else name
        shutil.move(f, os.path.join(out, dest))
        if name == "results.json":
            parts.append(os.path.join(out, dest))
    shutil.copy(os.path.join(WORK, f"log_part{i}.txt"), os.path.join(out, f"log_part{i}.txt"))
    shutil.rmtree(part, ignore_errors=True)
if parts:
    sys.path.insert(0, scripts)
    sys.path.insert(0, os.path.join(WORK, "pkg"))
    import heldout_clarity as H  # noqa: E402
    print("\n===== gate table (held-out frames; AE vs ceiling at the same height; PASS = edge_fidelity >= .8 ceiling, .8 ceiling <= edge_keep <= 1.05, OpenEDS pupil edge >= .8 ceiling) =====")
    print("(R3b: the AE512 baseline is not in this kernel; WORKS verdicts are produced locally by `heldout_clarity.py summary` with the R3 results file added)")
    H.summary_main(type("A", (), dict(results=parts, out=out, baseline_sheets=()))())
print("total kernel seconds:", round(time.time() - t0))
shutil.rmtree(os.path.join(WORK, "pkg"), ignore_errors=True)
if failed:
    raise SystemExit(f"parts failed: {failed}")
