"""Build a fake /kaggle/input for a local CPU dry run of scripts/kaggle_scratch.py (what the Kaggle kernel will see, with mini REAL caches).
  python scripts/make_dry_inputs.py --root %TEMP%\\dry_scratch [--videos 6 --openeds 10 --mrl 8 --dream 6]
then (PowerShell):  $env:KAGGLE_INPUT='<root>\\input'; $env:KAGGLE_WORKING='<root>\\work'; $env:SC_ALLOW_CPU='1'; $env:SC_STEPS1='8'; $env:SC_BATCH1='4'; $env:SC_ADV_BATCH='2'; $env:SC_DEADLINE_MIN='9'; python scripts/kaggle_scratch.py
Dataset directory names equal the Kaggle slugs (the kernel finds caches by directory name). Code = flat copy of ivs/*.py, tools = the scripts the kernel needs. Nothing is deleted; --force replaces an existing root's input dir only.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
CACHE = HERE.parent / "data" / "cache"
TOOLS = ("adv_rssm.py", "scratch_chain.py")


def safe(stem):
    import re
    return re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")


def copy_cache(src, dst, n, source=None):
    dst.mkdir(parents=True)
    metas = json.loads((src / "manifest.json").read_text())
    metas = [m for m in metas if source is None or m.get("source") == source][:n]
    for m in metas:
        f = src / (safe(Path(m["name"]).stem) + ".npz")
        shutil.copy(f, dst / f.name)
    (dst / "manifest.json").write_text(json.dumps(metas))
    return len(metas)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--videos", type=int, default=6)
    ap.add_argument("--openeds", type=int, default=10)
    ap.add_argument("--mrl", type=int, default=8)
    ap.add_argument("--dream", type=int, default=6)
    ap.add_argument("--dream-src", default=str(CACHE / "dreams"))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    inp = Path(a.root) / "input"
    if inp.exists():
        if not a.force:
            raise SystemExit(f"{inp} exists; pass --force")
        shutil.rmtree(inp)
    code = inp / "eye-ivs-code"
    code.mkdir(parents=True)
    for f in (HERE / "ivs").glob("*.py"):
        shutil.copy(f, code / f.name)
    tools = inp / "eye-heldout-tools"
    tools.mkdir()
    for t in TOOLS:
        shutil.copy(HERE / "scripts" / t, tools / t)
    n = dict(videos=copy_cache(CACHE / "videos", inp / "eye-cache-videos", a.videos), openeds=copy_cache(CACHE / "openeds", inp / "eye-cache-openeds", a.openeds),
             mrl=copy_cache(CACHE / "mrl", inp / "eye-cache-mrl", a.mrl), dreams=copy_cache(Path(a.dream_src), inp / "eye-dreams", a.dream))
    print("fake input at", inp, n)


if __name__ == "__main__":
    main()
