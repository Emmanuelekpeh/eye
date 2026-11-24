"""Kaggle kernel entry for R5 stage 2 (plan.md "R5 STAGE 2"): adversarial fine-tuning of the L1 RSSM on its own UI dream loop. GPU 0 = arm ctl (L1's own objective, same steps, the
control), GPU 1 = arm adv (the same + the conditional PatchGAN on the dream). One process per arm, same data seed.
Private kernel, GPU on, internet off. Inputs: theradactyl/eye-ivs-code (flat ivs/*.py incl. advdream.py, refiner.py, pilot.py ...), theradactyl/eye-heldout-tools (scripts incl. adv_rssm.py),
theradactyl/eye-hr96 (videos + OpenEDS, 96 px CacheBank directory) and theradactyl/eye-l1 (model.pt = the L1 RSSM checkpoint).
Outputs: s2/<arm>/{model.pt, model_s1.pt, model_s2.pt, train_log.jsonl, summary.json}, s2/log_<arm>.txt. Evaluation is local (scripts/s2_eval.py).
Env (local dry run): KAGGLE_INPUT, KAGGLE_WORKING, S2_STEPS (3000), S2_MINUTES (60), S2_BATCH (24), S2_LR (1e-4), S2_WADV (0.3), S2_ARMS (ctl,adv), S2_ALLOW_CPU=1.
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
STEPS, MINUTES, BATCH = os.environ.get("S2_STEPS", "3000"), os.environ.get("S2_MINUTES", "60"), os.environ.get("S2_BATCH", "24")
LR, WADV = os.environ.get("S2_LR", "1e-4"), os.environ.get("S2_WADV", "0.3")
ARMS = tuple(os.environ.get("S2_ARMS", "ctl,adv").split(","))

t0 = time.time()
code = sorted(glob.glob(os.path.join(INPUT, "**", "advdream.py"), recursive=True))
tool = sorted(glob.glob(os.path.join(INPUT, "**", "adv_rssm.py"), recursive=True))
assert code and tool, f"inputs not found: code={code} tool={tool} (re-version eye-ivs-code / eye-heldout-tools)"
ckpt = sorted(glob.glob(os.path.join(INPUT, "**", "model.pt"), recursive=True))
assert len(ckpt) == 1, f"expected exactly one model.pt (dataset eye-l1), found {ckpt}"
cache = None
for m in sorted(glob.glob(os.path.join(INPUT, "**", "manifest.json"), recursive=True)):
    if re.search(r"hr96$", os.path.basename(os.path.dirname(m))):
        cache = os.path.dirname(m)
assert cache, "cache dataset eye-hr96 missing"
pkg = os.path.join(WORK, "pkg", "ivs")
os.makedirs(pkg, exist_ok=True)
for f in glob.glob(os.path.join(os.path.dirname(code[0]), "*.py")):
    shutil.copy(f, pkg)
import torch  # noqa: E402

n_gpu = torch.cuda.device_count()
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(), "| devices", n_gpu, "| cache", cache, "| ckpt", ckpt[0], "| steps", STEPS, "minutes", MINUTES, "batch", BATCH, "lr", LR, "w_adv", WADV, flush=True)
if not torch.cuda.is_available() and os.environ.get("S2_ALLOW_CPU") != "1":
    raise SystemExit("no GPU attached: refusing to train on a Kaggle CPU")

OUT = os.path.join(WORK, "s2")
os.makedirs(OUT, exist_ok=True)
procs = []
for i, arm in enumerate(ARMS):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i % max(1, n_gpu)), PYTHONPATH=os.path.join(WORK, "pkg"), PYTHONUNBUFFERED="1")
    lf = open(os.path.join(OUT, f"log_{arm}.txt"), "w")
    cmd = [sys.executable, tool[0], "train", "--arm", arm, "--ckpt", ckpt[0], "--cache", cache, "--out", OUT, "--steps", STEPS, "--max-minutes", MINUTES,
           "--batch", BATCH, "--lr", LR, "--w-adv", WADV, "--device", "auto" if n_gpu else "cpu"]
    procs.append((arm, subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT), lf))
failed = []
for arm, p, lf in procs:
    rc = p.wait()
    lf.close()
    print(f"\n===== arm {arm} exit {rc} after {round((time.time() - t0) / 60, 1)} min =====")
    print("".join(open(os.path.join(OUT, f"log_{arm}.txt")).readlines()[-25:])[:5000])
    if rc != 0:
        failed.append(arm)
shutil.rmtree(os.path.join(WORK, "pkg"), ignore_errors=True)
if failed:
    raise SystemExit(f"arms failed: {failed}")
