"""Kaggle kernel entry for R4 (plan.md "R4"): the complete run = sharp autoencoder (S32_ssim, saved) -> latent-grid dynamics, two variants in parallel
(GPU 0: free-running own sample, GPU 1: re-projected through Enc(Dec(.))) -> held-out dream / touch / one-step evaluation with sheets.
Private kernel, GPU on, internet off. Inputs: theradactyl/eye-ivs-code (flat ivs/*.py incl. latentdyn.py), theradactyl/eye-heldout-tools (scripts + eval pack +
segmenter.pt: r4_latent.py, heldout_clarity.py, overfit_clarity.py, clarity_ceiling.py, make_birth.py, eval_ui_dream.py, eval_native.npz, eval_meta.json, segmenter.pt)
and theradactyl/eye-hr128f (whole-video + OpenEDS 128 px CacheBank directory).
Run 2 (long, anchor) reuses ae.pt from dataset theradactyl/eye-r4-ae. Outputs in /kaggle/working/r4: ae/{ae.pt,results.json,sheets}, one folder per variant (free, proj, long, anchor) {dyn.pt, eval.json, curve.json, dream_sheet_T0.5.png, touch_sheet.png, dream_frames_T0.5.npz}, logs.
Env (local dry run): KAGGLE_INPUT, KAGGLE_WORKING, R4_AE_STEPS (10000), R4_AE_BATCH (128), R4_MINUTES (100), R4_VARIANTS (long,anchor), R4_FREE_MAX, R4_BATCH (16), R4_WORLDS (64), R4_TICKS (300), R4_ALLOW_CPU=1, R4_STAGGER (seconds between variant starts, 300).
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
AE_STEPS, AE_BATCH = os.environ.get("R4_AE_STEPS", "10000"), os.environ.get("R4_AE_BATCH", "128")
MINUTES, BATCH = os.environ.get("R4_MINUTES", "100"), os.environ.get("R4_BATCH", "8")
WORLDS, TICKS = os.environ.get("R4_WORLDS", "64"), os.environ.get("R4_TICKS", "300")
STAGGER = float(os.environ.get("R4_STAGGER", "300"))               # two whole-video banks (~7 GB each) must not be built at the same moment
VARIANTS = tuple(os.environ.get("R4_VARIANTS", "long,anchor").split(","))      # run 1 was free,proj; run 2 = long,anchor (plan.md)
FREE_MAX = os.environ.get("R4_FREE_MAX")                                        # optional override of the horizon (local dry run)

t0 = time.time()
code = sorted(glob.glob(os.path.join(INPUT, "**", "latentdyn.py"), recursive=True))
tool = sorted(glob.glob(os.path.join(INPUT, "**", "r4_latent.py"), recursive=True))
assert code and tool, f"inputs not found: code={code} tool={tool} (re-version eye-ivs-code / eye-heldout-tools)"
scripts = os.path.dirname(tool[0])
for need in ("heldout_clarity.py", "overfit_clarity.py", "clarity_ceiling.py", "make_birth.py", "eval_ui_dream.py", "eval_native.npz", "eval_meta.json", "segmenter.pt"):
    assert os.path.exists(os.path.join(scripts, need)), f"{need} missing next to r4_latent.py in {scripts}"
caches = {}
for m in sorted(glob.glob(os.path.join(INPUT, "**", "manifest.json"), recursive=True)):
    hit = re.search(r"hr(\d+f?)$", os.path.basename(os.path.dirname(m)))
    if hit:
        caches[hit.group(1)] = os.path.dirname(m)
assert "128f" in caches, f"cache dataset eye-hr128f missing: have {sorted(caches)}"
cache = caches["128f"]
pkg = os.path.join(WORK, "pkg", "ivs")
os.makedirs(pkg, exist_ok=True)
for f in glob.glob(os.path.join(os.path.dirname(code[0]), "*.py")):
    shutil.copy(f, pkg)
import torch  # noqa: E402

n_gpu = torch.cuda.device_count()
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(), "| devices", n_gpu, "| cache", cache, "| scripts", scripts, "| ae steps", AE_STEPS, "| dyn minutes", MINUTES, flush=True)
if not torch.cuda.is_available() and os.environ.get("R4_ALLOW_CPU") != "1":
    raise SystemExit("no GPU attached: refusing to train on a Kaggle CPU")

OUT = os.path.join(WORK, "r4")
os.makedirs(OUT, exist_ok=True)
base = [sys.executable, os.path.join(scripts, "r4_latent.py")]
common = ["--cache", cache, "--out", OUT, "--pack", scripts, "--segmenter", os.path.join(scripts, "segmenter.pt"), "--device", "auto" if n_gpu else "cpu"]


def env_for(i):
    return dict(os.environ, CUDA_VISIBLE_DEVICES=str(i % max(1, n_gpu)), PYTHONPATH=os.path.join(WORK, "pkg"), PYTHONUNBUFFERED="1")


def tail(path, n=14):
    print("".join(open(path).readlines()[-n:])[:3500])


# stage 1: autoencoder. Run 2 reuses the run-1 autoencoder (dataset eye-r4-ae holds ae.pt, same S32_ssim recipe/weights, clarity already scored in run 1); otherwise train it (~28 min, one GPU)
saved = sorted(glob.glob(os.path.join(INPUT, "**", "ae.pt"), recursive=True))
if saved:
    os.makedirs(os.path.join(OUT, "ae"), exist_ok=True)
    shutil.copy(saved[0], os.path.join(OUT, "ae", "ae.pt"))
    print("reusing autoencoder", saved[0], flush=True)
else:
    log = os.path.join(OUT, "log_ae.txt")
    with open(log, "w") as lf:
        rc = subprocess.call(base + ["ae", *common, "--ae-steps", AE_STEPS, "--ae-batch", AE_BATCH], env=env_for(0), stdout=lf, stderr=subprocess.STDOUT)
    print(f"===== ae stage exit {rc} after {round((time.time() - t0) / 60, 1)} min =====")
    tail(log)
    if rc != 0 or not os.path.exists(os.path.join(OUT, "ae", "ae.pt")):
        raise SystemExit("ae stage failed: no dynamics without the autoencoder")

# stage 2: dynamics + evaluation, one process per variant / GPU
procs = []
for i, v in enumerate(VARIANTS):
    if i:
        time.sleep(STAGGER)
    lf = open(os.path.join(OUT, f"log_{v}.txt"), "w")
    procs.append((v, subprocess.Popen(base + ["dyn", *common, "--variant", v, "--minutes", MINUTES, "--batch", BATCH, "--worlds", WORLDS, "--ticks", TICKS] + (["--free-max", FREE_MAX] if FREE_MAX else []),
                                      env=env_for(i), stdout=lf, stderr=subprocess.STDOUT), lf))
failed = []
for v, p, lf in procs:
    rc = p.wait()
    lf.close()
    print(f"\n===== dyn variant {v} exit {rc} =====")
    tail(os.path.join(OUT, f"log_{v}.txt"), 30)
    if rc != 0:
        failed.append(v)
print("total kernel seconds:", round(time.time() - t0))
shutil.rmtree(os.path.join(WORK, "pkg"), ignore_errors=True)
if failed:
    raise SystemExit(f"variants failed: {failed}")
