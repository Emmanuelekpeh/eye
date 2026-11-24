"""Kaggle kernel entry point for GPU evals (private kernel, GPU on, internet off). STANDING RULE: evals run here, not on CPU.

Inputs (Kaggle datasets, all attached to the kernel):
  theradactyl/eye-ivs-code        flat copy of ivs/*.py            (re-version after any ivs edit)
  theradactyl/eye-cache-videos    video cache (manifest.json)      -> --cache-video
  theradactyl/eye-cache-openeds   openeds cache (manifest.json)    -> --cache-openeds
  theradactyl/eye-masks-openeds   S_ masks (masks.json)            -> --masks
  theradactyl/eye-eval-assets     eval_nca.py, segmenter.pt and the model files named in NCA / WM / AE below
Edit NCA / WM / AE / EXTRA (marked) and `kaggle kernels push` (see backlog.md, STANDING RULE).
The evals themselves are the repo scripts (scripts/eval_nca.py), run unchanged with --device cuda: the NCA repair / sampling runs on the GPU,
the RSSM / AE recon functions and the segmenter stay on the (Kaggle) CPU exactly as in the local script.
Outputs: /kaggle/working/eval/ (eval_nca.json, sheet_*.png, eval.log).
"""
import glob
import importlib.util
import os
import shutil
import sys
import time

NCA = {"N50": "N50_model.pt", "N135": "N135_model.pt"}      # edited at push time: NAME -> file in the assets dataset
WM = {"L1": "L1_model.pt"}                                   # RSSM world models (the gate compares against GATE_WM)
AE = {"AE512": "AE512_model.pt"}                             # plain AE probes
GATE_WM = "L1"
EXTRA = ["--n", "96", "--select-n", "32", "--dreams", "64"]  # edited at push time (more eval args for eval_nca.py)

IN = os.environ.get("EVAL_INPUT", "/kaggle/input")
OUT = os.environ.get("EVAL_WORKING", "/kaggle/working")


def one_dir(pattern, what):
    hits = sorted({os.path.dirname(p) for p in glob.glob(f"{IN}/**/{pattern}", recursive=True)})
    assert len(hits) == 1, f"{what}: expected exactly one dir with {pattern} under {IN}, found {hits}"
    return hits[0]


t0 = time.time()
code = one_dir("nca.py", "code dataset")
assets = one_dir("eval_nca.py", "assets dataset")
caches = sorted({os.path.dirname(p) for p in glob.glob(f"{IN}/**/manifest.json", recursive=True)})
cv = [c for c in caches if "video" in c]
co = [c for c in caches if "openeds" in c and "masks" not in c]
assert len(cv) == 1 and len(co) == 1, f"video / openeds caches not identified: {caches}"
masks = one_dir("masks.json", "masks dataset")

for kind, table in (("nca", NCA), ("wm", WM), ("ae", AE)):
    for name, f in table.items():
        assert os.path.exists(os.path.join(assets, f)), f"{kind} {name}: {f} not in {assets} ({os.listdir(assets)})"
seg = os.path.join(assets, "segmenter.pt")
assert os.path.exists(seg), f"segmenter.pt not in {assets}"

pkg = os.path.join(OUT, "pkg", "ivs")
os.makedirs(pkg, exist_ok=True)
for f in glob.glob(os.path.join(code, "*.py")):
    shutil.copy(f, pkg)
sys.path.insert(0, os.path.join(OUT, "pkg"))

import torch  # noqa: E402

print("python", sys.version.split()[0], "| torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", flush=True)
print("caches:", cv[0], co[0], "| masks:", masks, "| assets:", sorted(os.listdir(assets)), flush=True)
assert torch.cuda.is_available(), "no GPU: the point of this kernel is a GPU eval (turn the accelerator on)"

spec = importlib.util.spec_from_file_location("eval_nca", os.path.join(assets, "eval_nca.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

out = os.path.join(OUT, "eval")
args = ["--cache-video", cv[0], "--cache-openeds", co[0], "--masks", masks, "--segmenter", seg, "--device", "cuda", "--out", out,
        "--gate-wm", GATE_WM, *EXTRA]
for flag, table in (("--nca", NCA), ("--wm", WM), ("--ae", AE)):
    for name, f in table.items():
        args += [flag, f"{name}={os.path.join(assets, f)}"]
print("eval_nca args:", args, flush=True)
mod.main(args)
shutil.rmtree(os.path.join(OUT, "pkg"), ignore_errors=True)
print("total kernel seconds:", round(time.time() - t0), flush=True)
