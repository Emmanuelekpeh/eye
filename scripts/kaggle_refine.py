"""Kaggle kernel entry for R5 stage 1 (plan.md "R5"): conditional-GAN refiner on the soft render of two bases, arms reg (control) and gan, scored on the held-out clarity pack.
GPU 0: base l1 (L1 RSSM posterior render, 96 px, cache dataset eye-hr96); GPU 1: base ae (S32_ssim autoencoder, 128 px, cache dataset eye-hr128f). One process per base, both write to /kaggle/working/r5.
Private kernel, GPU on, internet off. Inputs: theradactyl/eye-ivs-code (flat ivs/*.py incl. refiner.py, latentdyn.py), theradactyl/eye-heldout-tools (scripts incl. refine_gan.py, heldout_clarity.py,
the eval pack: eval_native.npz, eval_meta.json, segmenter.pt, val_groups.json ...), theradactyl/eye-hr96, theradactyl/eye-hr128f, theradactyl/eye-r4-ae (ae.pt), theradactyl/eye-l1 (model.pt = the L1 RSSM checkpoint).
Outputs: r5/{l1_reg.pt, l1_gan.pt, ae_reg.pt, ae_gan.pt, results_l1.json, results_ae.json, refine_sheet_<base>_<slice>.png, log_l1.txt, log_ae.txt}.
Continuation (automatic when such files are attached, or R5_RESUME=1/0): the kernel also needs a dataset holding l1_reg.pt, l1_gan.pt, ae_reg.pt, ae_gan.pt (theradactyl/eye-r5-ref = the run-1 outputs); every arm continues from them, outputs go to r5b.
Env (local dry run): KAGGLE_INPUT, KAGGLE_WORKING, R5_STEPS (6000), R5_BATCH (24), R5_LR (2e-4), R5_BASES (l1,ae), R5_RESUME (0), R5_ALLOW_CPU=1.
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
STEPS, BATCH, LR = os.environ.get("R5_STEPS", "6000"), os.environ.get("R5_BATCH", "24"), os.environ.get("R5_LR", "2e-4")
BASES = tuple(os.environ.get("R5_BASES", "l1,ae").split(","))
RESUME = os.environ.get("R5_RESUME", "auto")                      # auto: continue when a dataset with <base>_<arm>.pt refiners is attached (a Kaggle kernel has no env vars)
RESUME = bool(glob.glob(os.path.join(INPUT, "**", "*_gan.pt"), recursive=True)) if RESUME == "auto" else RESUME == "1"
CACHE_KEY = {"l1": "96", "ae": "128f"}

t0 = time.time()
code = sorted(glob.glob(os.path.join(INPUT, "**", "refiner.py"), recursive=True))
tool = sorted(glob.glob(os.path.join(INPUT, "**", "refine_gan.py"), recursive=True))
assert code and tool, f"inputs not found: code={code} tool={tool} (re-version eye-ivs-code / eye-heldout-tools)"
scripts = os.path.dirname(tool[0])
for need in ("heldout_clarity.py", "overfit_clarity.py", "clarity_ceiling.py", "make_birth.py", "eval_ui_dream.py", "eval_native.npz", "eval_meta.json", "segmenter.pt"):
    assert os.path.exists(os.path.join(scripts, need)), f"{need} missing next to refine_gan.py in {scripts}"
caches = {}
for m in sorted(glob.glob(os.path.join(INPUT, "**", "manifest.json"), recursive=True)):
    hit = re.search(r"hr(\d+f?)$", os.path.basename(os.path.dirname(m)))
    if hit:
        caches[hit.group(1)] = os.path.dirname(m)
ckpts = {"l1": sorted(glob.glob(os.path.join(INPUT, "**", "model.pt"), recursive=True)), "ae": sorted(glob.glob(os.path.join(INPUT, "**", "ae.pt"), recursive=True))}
for b in BASES:
    assert CACHE_KEY[b] in caches, f"cache dataset eye-hr{CACHE_KEY[b]} missing: have {sorted(caches)}"
    assert len(ckpts[b]) == 1, f"base {b}: expected exactly one checkpoint, found {ckpts[b]}"
resume_dir = None
if RESUME:
    hits = sorted({os.path.dirname(p) for p in glob.glob(os.path.join(INPUT, "**", "*_gan.pt"), recursive=True)})
    assert len(hits) == 1 and all(os.path.exists(os.path.join(hits[0], f"{b}_{a}.pt")) for b in BASES for a in ("reg", "gan")), \
        f"R5_RESUME=1: need exactly one dataset directory with <base>_reg.pt and <base>_gan.pt, found {hits}"
    resume_dir = hits[0]
pkg = os.path.join(WORK, "pkg", "ivs")
os.makedirs(pkg, exist_ok=True)
for f in glob.glob(os.path.join(os.path.dirname(code[0]), "*.py")):
    shutil.copy(f, pkg)
import torch  # noqa: E402

n_gpu = torch.cuda.device_count()
print("resume", resume_dir, "| torch", torch.__version__, "| cuda", torch.cuda.is_available(), "| devices", n_gpu, "| caches", caches, "| ckpts", ckpts, "| scripts", scripts, "| steps", STEPS, "batch", BATCH, "lr", LR, flush=True)
if not torch.cuda.is_available() and os.environ.get("R5_ALLOW_CPU") != "1":
    raise SystemExit("no GPU attached: refusing to train on a Kaggle CPU")

OUT = os.path.join(WORK, "r5b" if RESUME else "r5")
os.makedirs(OUT, exist_ok=True)
procs = []
for i, b in enumerate(BASES):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i % max(1, n_gpu)), PYTHONPATH=os.path.join(WORK, "pkg"), PYTHONUNBUFFERED="1")
    lf = open(os.path.join(OUT, f"log_{b}.txt"), "w")
    cmd = [sys.executable, os.path.join(scripts, "refine_gan.py"), "train", "--base", b, "--ckpt", ckpts[b][0], "--cache", caches[CACHE_KEY[b]], "--pack", scripts, "--out", OUT,
           "--steps", STEPS, "--batch", BATCH, "--lr", LR, "--device", "auto" if n_gpu else "cpu"] + (["--resume", resume_dir] if resume_dir else [])
    procs.append((b, subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT), lf))
failed = []
for b, p, lf in procs:
    rc = p.wait()
    lf.close()
    print(f"\n===== base {b} exit {rc} after {round((time.time() - t0) / 60, 1)} min =====")
    print("".join(open(os.path.join(OUT, f"log_{b}.txt")).readlines()[-30:])[:4500])
    if rc != 0:
        failed.append(b)
shutil.rmtree(os.path.join(WORK, "pkg"), ignore_errors=True)
if failed:
    raise SystemExit(f"bases failed: {failed}")
