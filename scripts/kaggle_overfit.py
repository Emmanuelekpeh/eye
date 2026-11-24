"""Kaggle kernel entry for the one-clip overfit clarity test (scripts/overfit_clarity.py). Private kernel, GPU on, internet off.
Inputs: dataset theradactyl/eye-ivs-code (flat copy of ivs/*.py) + theradactyl/eye-overfit-pack (colour.npz, openeds.npz, overfit_clarity.py,
clarity_ceiling.py). One process per GPU: the colour clip on GPU 0, the OpenEDS clip on GPU 1 (both on one GPU if only one is attached).
Outputs in /kaggle/working/overfit: results.json (+ results_<clip>.json), sheet_<clip>.png, log_<clip>.txt.
Overridable for a local dry run: KAGGLE_INPUT, KAGGLE_WORKING, OVERFIT_STEPS, OVERFIT_ALLOW_CPU=1.
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time

INPUT = os.environ.get("KAGGLE_INPUT", "/kaggle/input")
WORK = os.environ.get("KAGGLE_WORKING", "/kaggle/working")
STEPS = int(os.environ.get("OVERFIT_STEPS", "8000"))              # edited at push time if needed
GROUPS = [["colour"], ["openeds"]]
HEIGHTS = ["96", "128", "192"]

t0 = time.time()
code = sorted(glob.glob(os.path.join(INPUT, "**", "aeprobe.py"), recursive=True))
tool = sorted(glob.glob(os.path.join(INPUT, "**", "overfit_clarity.py"), recursive=True))
assert code and tool, f"inputs not found: code={code} tool={tool}"
scripts = os.path.dirname(tool[0])
for need in ("colour.npz", "openeds.npz", "clarity_ceiling.py"):
    assert os.path.exists(os.path.join(scripts, need)), f"{need} missing next to overfit_clarity.py in {scripts}"
pkg = os.path.join(WORK, "pkg", "ivs")
os.makedirs(pkg, exist_ok=True)
for f in glob.glob(os.path.join(os.path.dirname(code[0]), "*.py")):
    shutil.copy(f, pkg)
import torch  # noqa: E402

n_gpu = torch.cuda.device_count()
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(), "| devices", n_gpu, "| steps", STEPS, "| scripts", scripts, flush=True)
if not torch.cuda.is_available() and os.environ.get("OVERFIT_ALLOW_CPU") != "1":
    raise SystemExit("no GPU attached: refusing to run 6 trainings on a Kaggle CPU")

procs = []
for i, clips in enumerate(GROUPS):
    cmd = [sys.executable, os.path.join(scripts, "overfit_clarity.py"), "train", "--pack", scripts, "--out", os.path.join(WORK, f"part{i}"),
           "--clips", *clips, "--steps", str(STEPS), "--heights", *HEIGHTS, "--device", "auto" if n_gpu else "cpu"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i % max(1, n_gpu)), PYTHONPATH=os.path.join(WORK, "pkg"), PYTHONUNBUFFERED="1")
    log = open(os.path.join(WORK, f"log_{clips[0]}.txt"), "w")
    procs.append((i, clips[0], subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log))
failed = []
for i, name, p, log in procs:
    rc = p.wait()
    log.close()
    print(f"\n===== {name} (exit {rc}) tail =====")
    print("".join(open(os.path.join(WORK, f"log_{name}.txt")).readlines()[-14:])[:3500])
    if rc != 0:
        failed.append(name)

out = os.path.join(WORK, "overfit")
os.makedirs(out, exist_ok=True)
merged = {}
for i, name, _, _ in procs:
    part = os.path.join(WORK, f"part{i}")
    for f in glob.glob(os.path.join(part, "*")):
        if os.path.basename(f) != "results.json":
            shutil.move(f, os.path.join(out, os.path.basename(f)))
    if os.path.exists(os.path.join(part, "results.json")):
        merged.update(json.load(open(os.path.join(part, "results.json"))))
    shutil.copy(os.path.join(WORK, f"log_{name}.txt"), os.path.join(out, f"log_{name}.txt"))
    shutil.rmtree(part, ignore_errors=True)
json.dump(merged, open(os.path.join(out, "results.json"), "w"), indent=1)
print("\n===== summary (AE vs ceiling at the same height; gate = all of edge_keep in [.9 ceiling, 1.05], edge_fidelity >= .9 ceiling, OpenEDS pupil edge >= .9 ceiling) =====")
for clip, r in merged.items():
    for h, v in r["heights"].items():
        print(f"{clip} {h}px: keep {v['ae']['edge_keep']:.3f}/{v['ceiling']['edge_keep']:.3f}  fidelity {v['ae']['edge_fidelity']:.3f}/{v['ceiling']['edge_fidelity']:.3f}  "
              f"psnr {v['ae']['psnr']:.1f}/{v['ceiling']['psnr']:.1f}  {'PASS' if v['passed'] else 'FAIL'}  ({v['train_seconds']}s)")
print("total kernel seconds:", round(time.time() - t0))
shutil.rmtree(os.path.join(WORK, "pkg"), ignore_errors=True)
if failed:
    raise SystemExit(f"parts failed: {failed}")
