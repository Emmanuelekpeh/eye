"""Kaggle kernel entry point for the video-cache pilots (private kernel, GPU on, internet off).

Inputs (Kaggle datasets): one or more cache datasets (theradactyl/eye-cache-videos, eye-cache-openeds, ...; every dir with a manifest.json is passed to --cache), optionally theradactyl/eye-masks-openeds (dir with masks.json -> --labels, needed by the G* gaze presets), a checkpoint dataset holding one model.pt (-> --init-from, the LR* continuation presets) and theradactyl/eye-ivs-code (flat copy of the ivs/*.py files).
Edit GPU_GROUPS / STEPS / EXTRA (one line each, marked), then `kaggle kernels push` (see backlog.md P0b).
Kaggle gives 2x T4: each group of runs gets its own process pinned to one GPU, so the groups run in parallel.
Outputs land in /kaggle/working/pilot: summary.json (merged), <run>/metrics.json, train_log.jsonl, sheet_*.png, model.pt.
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time

ENTRY = "pilot"                     # edited at push time: ivs module whose main() runs ("pilot" = RSSM trainer, "ncapilot" = EyeNCA trainer)
GPU_GROUPS = [["R4_consistency"]]   # edited at push time: one list of runs per GPU
STEPS = 300                         # edited at push time
EXTRA = []                          # edited at push time

t0 = time.time()
code = sorted(glob.glob(f"/kaggle/input/**/{ENTRY}.py", recursive=True))
man = sorted(glob.glob("/kaggle/input/**/manifest.json", recursive=True))
assert code and man, f"inputs not found: code={code} manifest={man}"
caches = sorted({os.path.dirname(m) for m in man})          # every attached cache dataset (videos, openeds, ...)
masks = sorted({os.path.dirname(m) for m in glob.glob("/kaggle/input/**/masks.json", recursive=True)})   # eye-masks-openeds -> --labels
assert len(masks) <= 1, f"more than one masks dataset attached: {masks}"
label_args = ["--labels", masks[0]] if masks and ENTRY == "pilot" else []      # only the RSSM trainer takes mask labels
inits = sorted(glob.glob("/kaggle/input/**/model.pt", recursive=True))      # a checkpoint dataset (e.g. theradactyl/eye-init-l1) -> --init-from (continuation runs)
assert len(inits) <= 1, f"more than one model.pt attached: {inits}"
init_args = ["--init-from", inits[0]] if inits else []
pkg = "/kaggle/working/pkg/ivs"
os.makedirs(pkg, exist_ok=True)
for f in glob.glob(os.path.join(os.path.dirname(code[0]), "*.py")):
    shutil.copy(f, pkg)

import torch  # noqa: E402

n_gpu = torch.cuda.device_count()
print("python", sys.version.split()[0], "| torch", torch.__version__, "| cuda", torch.cuda.is_available(), "| devices", n_gpu,
      "|", torch.cuda.get_device_name(0) if n_gpu else "cpu", flush=True)
print("caches:", {c: len(glob.glob(os.path.join(c, "*.npz"))) for c in caches}, "| labels:", masks, "| groups:", GPU_GROUPS, flush=True)

procs = []
for i, runs in enumerate(GPU_GROUPS):
    args = ["--cache", *caches, "--out", f"/kaggle/working/pilot_part{i}", "--runs", *runs, "--steps", str(STEPS),
            "--device", "auto", *label_args, *init_args, *EXTRA]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i % max(1, n_gpu)), PYTHONUNBUFFERED="1")
    cmd = [sys.executable, "-c", f"import sys; sys.path.insert(0, '/kaggle/working/pkg'); import importlib; importlib.import_module('ivs.{ENTRY}').main({args!r})"]
    log = open(f"/kaggle/working/log_part{i}.txt", "w")
    procs.append((i, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log))

failed = []
for i, p, log in procs:
    rc = p.wait()
    log.close()
    print(f"\n===== part {i} (exit {rc}) tail =====")
    print("".join(open(f"/kaggle/working/log_part{i}.txt").readlines()[-6:])[:2500])
    if rc != 0:
        failed.append(i)

os.makedirs("/kaggle/working/pilot", exist_ok=True)
merged = {}
for i, _, _ in procs:
    part = f"/kaggle/working/pilot_part{i}"
    if os.path.exists(part + "/summary.json"):
        merged.update(json.load(open(part + "/summary.json")))
        for d in glob.glob(part + "/*/"):
            shutil.move(d.rstrip("/"), "/kaggle/working/pilot/" + os.path.basename(d.rstrip("/")))
        if i == 0 and os.path.exists(part + "/data_info.json"):                 # only the RSSM trainer writes it
            shutil.copy(part + "/data_info.json", "/kaggle/working/pilot/data_info.json")
json.dump(merged, open("/kaggle/working/pilot/summary.json", "w"), indent=1)
for i, _, _ in procs:
    shutil.copy(f"/kaggle/working/log_part{i}.txt", f"/kaggle/working/pilot/log_part{i}.txt")
    shutil.rmtree(f"/kaggle/working/pilot_part{i}", ignore_errors=True)

print("\n===== speed (steps/sec) by rollout length =====")
for name in merged:
    log = [json.loads(l) for l in open(f"/kaggle/working/pilot/{name}/train_log.jsonl")]
    by_t = {}
    for a, b in zip(log[:-1], log[1:]):
        by_t.setdefault(b.get("T", "-"), []).append((b["step"] - a["step"]) / max(1e-9, b["sec"] - a["sec"]))
    print(name, {t: round(sum(v) / len(v), 2) for t, v in by_t.items()}, "| train", merged[name]["train_seconds"], "s")
print("total kernel seconds:", round(time.time() - t0))
shutil.rmtree("/kaggle/working/pkg", ignore_errors=True)
if failed:
    raise SystemExit(f"parts failed: {failed}")
