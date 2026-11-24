"""Kaggle kernel entry for the plain-autoencoder probe (ivs/aeprobe.py). Private kernel, GPU on, internet off.
Inputs: the video cache dataset + theradactyl/eye-ivs-code. Two GPUs: each group of variants runs in its own process.
Outputs in /kaggle/working/ae: summary.json, <variant>_metrics.json / _sheet.png / _model.pt / _train_log.jsonl, baselines.json.
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time

GROUPS = [["AE32", "AE128"], ["AE512", "AE128_wide"]]   # edited at push time: one list of variants per GPU
STEPS = 8000                                            # edited at push time

t0 = time.time()
code = sorted(glob.glob("/kaggle/input/**/aeprobe.py", recursive=True))
man = sorted(glob.glob("/kaggle/input/**/manifest.json", recursive=True))
assert code and man, f"inputs not found: code={code} manifest={man}"
caches = sorted({os.path.dirname(m) for m in man})
pkg = "/kaggle/working/pkg/ivs"
os.makedirs(pkg, exist_ok=True)
for f in glob.glob(os.path.join(os.path.dirname(code[0]), "*.py")):
    shutil.copy(f, pkg)
import torch  # noqa: E402

n_gpu = torch.cuda.device_count()
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(), "| devices", n_gpu, "| caches", caches, flush=True)
procs = []
for i, variants in enumerate(GROUPS):
    args = ["--cache", *caches, "--out", f"/kaggle/working/ae_part{i}", "--variants", *variants, "--steps", str(STEPS),
            "--device", "auto"] + ([] if i == 0 else ["--no-pca"])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i % max(1, n_gpu)), PYTHONUNBUFFERED="1")
    cmd = [sys.executable, "-c", f"import sys; sys.path.insert(0, '/kaggle/working/pkg'); from ivs import aeprobe; aeprobe.main({args!r})"]
    log = open(f"/kaggle/working/log_part{i}.txt", "w")
    procs.append((i, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log))
failed = []
for i, p, log in procs:
    rc = p.wait(); log.close()
    print(f"\n===== part {i} (exit {rc}) tail =====")
    print("".join(open(f"/kaggle/working/log_part{i}.txt").readlines()[-8:])[:3000])
    if rc != 0:
        failed.append(i)
os.makedirs("/kaggle/working/ae", exist_ok=True)
merged = {}
for i, _, _ in procs:
    part = f"/kaggle/working/ae_part{i}"
    for f in glob.glob(part + "/*"):
        shutil.move(f, "/kaggle/working/ae/" + os.path.basename(f))
    shutil.copy(f"/kaggle/working/log_part{i}.txt", f"/kaggle/working/ae/log_part{i}.txt")
    if os.path.exists("/kaggle/working/ae/summary.json"):
        merged.update(json.load(open("/kaggle/working/ae/summary.json")))
        os.rename("/kaggle/working/ae/summary.json", f"/kaggle/working/ae/summary_part{i}.json")
    shutil.rmtree(part, ignore_errors=True)
json.dump(merged, open("/kaggle/working/ae/summary.json", "w"), indent=1)
print("\n===== summary (val MSE) =====")
for name, r in merged.items():
    print(name, "val", round(r["val"]["mse"], 5), "train", round(r["train_fit"]["mse"], 5), "sharp", round(r["val"]["sharp_ratio"], 3), "|", r["train_seconds"], "s")
if os.path.exists("/kaggle/working/ae/baselines.json"):
    print("baselines:", open("/kaggle/working/ae/baselines.json").read().replace("\n", " "))
print("total kernel seconds:", round(time.time() - t0))
shutil.rmtree("/kaggle/working/pkg", ignore_errors=True)
if failed:
    raise SystemExit(f"parts failed: {failed}")
