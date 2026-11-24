"""Kaggle kernel entry for the SCRATCH run (plan.md "SCRATCH", pre-registered before it ran): two arms, one per T4, each = stage 1 from random weights + the adversarial stage, finished inside
one hard deadline (the user's last GPU hour). Private kernel, GPU on, internet off.
Inputs (datasets): theradactyl/eye-ivs-code (flat ivs/*.py incl. the train_only cachedata + N0_scratch preset), theradactyl/eye-heldout-tools (scripts incl. scratch_chain.py, adv_rssm.py),
theradactyl/eye-cache-videos, eye-cache-openeds, eye-cache-mrl (the caches L1 trained on) and theradactyl/eye-dreams (the dream pool).
Outputs: scratch/<arm>/{stage1/N0_scratch/model.pt, stage2/adv/model*.pt, chain.json, ...} for arm in (ctl, mix), scratch/log_<arm>.txt. Evaluation is local.
Env (local dry run): KAGGLE_INPUT, KAGGLE_WORKING, SC_STEPS1 (8500), SC_DEADLINE_MIN (50), SC_ARMS (mix,ctl), SC_ALLOW_CPU=1, SC_BATCH1, SC_ADV_BATCH, SC_CKPT_EVERY.
"""
import glob
import os
import shutil
import subprocess
import sys
import time

INPUT = os.environ.get("KAGGLE_INPUT", "/kaggle/input")
WORK = os.environ.get("KAGGLE_WORKING", "/kaggle/working")
STEPS1, DEADLINE = os.environ.get("SC_STEPS1", "8500"), os.environ.get("SC_DEADLINE_MIN", "50")
ARMS = tuple(os.environ.get("SC_ARMS", "mix,ctl").split(","))

T0 = time.time()
code = sorted(glob.glob(os.path.join(INPUT, "**", "advdream.py"), recursive=True))
tool = sorted(glob.glob(os.path.join(INPUT, "**", "scratch_chain.py"), recursive=True))
assert code and tool, f"inputs not found: code={code} tool={tool} (re-version eye-ivs-code / eye-heldout-tools)"
caches = {}
for m in sorted(glob.glob(os.path.join(INPUT, "**", "manifest.json"), recursive=True)):
    caches[os.path.basename(os.path.dirname(m))] = os.path.dirname(m)
need = ("eye-cache-videos", "eye-cache-openeds", "eye-cache-mrl", "eye-dreams")
missing = [n for n in need if n not in caches]
assert not missing, f"datasets missing: {missing} (have {sorted(caches)})"
pkg = os.path.join(WORK, "pkg", "ivs")
os.makedirs(pkg, exist_ok=True)
for f in glob.glob(os.path.join(os.path.dirname(code[0]), "*.py")):
    shutil.copy(f, pkg)
import torch  # noqa: E402

n_gpu = torch.cuda.device_count()
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(), "| devices", n_gpu, "| arms", ARMS, "| steps1", STEPS1, "| deadline", DEADLINE, "min | caches", {k: caches[k] for k in need}, flush=True)
if not torch.cuda.is_available() and os.environ.get("SC_ALLOW_CPU") != "1":
    raise SystemExit("no GPU attached: refusing to train on a Kaggle CPU")
if os.environ.get("SC_ALLOW_CPU") != "1" and n_gpu < len(ARMS):
    raise SystemExit(f"{n_gpu} GPU(s) for {len(ARMS)} arms: the step counts assume one T4 per arm, sharing one would double stage 1 and starve stage 2; not spending the hour on that")

OUT = os.path.join(WORK, "scratch")
os.makedirs(OUT, exist_ok=True)
procs = []
for i, arm in enumerate(ARMS):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i % max(1, n_gpu)), PYTHONPATH=os.path.join(WORK, "pkg"), PYTHONUNBUFFERED="1")
    lf = open(os.path.join(OUT, f"log_{arm}.txt"), "w")
    cmd = [sys.executable, tool[0], "--arm", arm, "--real-caches", caches["eye-cache-videos"], caches["eye-cache-openeds"], "--image-cache", caches["eye-cache-mrl"],
           "--dream-cache", caches["eye-dreams"], "--out", OUT, "--steps1", STEPS1, "--deadline-min", DEADLINE, "--t0", str(T0), "--device", "auto" if n_gpu else "cpu",
           "--ckpt-every", os.environ.get("SC_CKPT_EVERY", "2000")]
    if os.environ.get("SC_BATCH1"):
        cmd += ["--batch1", os.environ["SC_BATCH1"]]
    if os.environ.get("SC_ADV_BATCH"):
        cmd += ["--adv-batch", os.environ["SC_ADV_BATCH"]]
    procs.append((arm, subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT), lf))
failed = []
for arm, p, lf in procs:
    rc = p.wait()
    lf.close()
    print(f"\n===== arm {arm} exit {rc} after {round((time.time() - T0) / 60, 1)} min =====")
    print("".join(open(os.path.join(OUT, f"log_{arm}.txt")).readlines()[-12:])[:3500])
    if rc != 0:
        failed.append(arm)
shutil.rmtree(os.path.join(WORK, "pkg"), ignore_errors=True)
if failed:
    raise SystemExit(f"arms failed: {failed}")
