"""Read-only audit of the eye data under <root> (default ../set) and <scraped> (default ../scraped_images).

Nothing is modified. Outputs go to --out (default ../ivs2/audit): report.txt, report.json, and PNG contact
sheets that MUST be looked at (numbers alone do not tell you whether a frame is an eye).

Questions answered, per source class:
  - how many files / what format / native size and aspect (the loader squashes to square),
  - brightness / sharpness / flat-or-black frame rates,
  - temporal continuity: is natural filename order coherent, is the loader's lexicographic order scrambled,
  - duplicates between the loose PNGs and awake/sleepy, and between videos,
  - label (.npy) contents,
  - per-video decode stats with the exact subsampling data.py uses (every 5th frame, <=600 kept).
"""
import argparse, hashlib, json, random, re, sys, time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VID = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
TH = 96


def nat(p):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", Path(p).name)]


def load_gray(path, size=None):
    im = Image.open(path).convert("L")
    native = im.size
    if size:
        im = im.resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0, native


def sharp(g):
    return float(cv2.Laplacian((g * 255).astype(np.uint8), cv2.CV_64F).var())


def img_stats(paths, rng, n):
    paths = rng.sample(paths, min(n, len(paths)))
    rows, natives, thumbs = [], Counter(), []
    for p in paths:
        try:
            g, nat_sz = load_gray(p, TH)
        except Exception as e:  # corrupt / unsupported
            rows.append(dict(path=str(p), error=str(e)[:80])); continue
        natives[f"{nat_sz[0]}x{nat_sz[1]}"] += 1
        rows.append(dict(mean=float(g.mean()), std=float(g.std()), sharp=sharp(g)))
        thumbs.append(g)
    ok = [r for r in rows if "error" in r]
    good = [r for r in rows if "error" not in r]
    out = dict(sampled=len(paths), errors=len(ok), native=dict(natives.most_common(4)))
    if good:
        m = np.array([r["mean"] for r in good]); s = np.array([r["std"] for r in good]); sh = np.array([r["sharp"] for r in good])
        out.update(mean_p5_50_95=[float(x) for x in np.percentile(m, [5, 50, 95])],
                   std_p5_50=[float(x) for x in np.percentile(s, [5, 50])],
                   sharp_p5_50_95=[float(x) for x in np.percentile(sh, [5, 50, 95])],
                   frac_flat=float((s < 0.03).mean()), frac_dark=float((m < 0.05).mean()), frac_bright=float((m > 0.95).mean()))
    return out, thumbs


def sheet(thumbs, cols, path, label=None):
    if not thumbs:
        return
    rows = (len(thumbs) + cols - 1) // cols
    canvas = np.zeros((rows * TH, cols * TH), np.float32)
    for i, t in enumerate(thumbs):
        r, c = divmod(i, cols)
        canvas[r * TH:(r + 1) * TH, c * TH:(c + 1) * TH] = cv2.resize(t, (TH, TH))
    img = (canvas * 255).clip(0, 255).astype(np.uint8)
    if label:
        cv2.putText(img, label, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255, 1)
    cv2.imwrite(str(path), img)


def small(path, s=48):
    g, _ = load_gray(path, s)
    return g


def continuity(ordered, rng, max_frames=60):
    """mean |diff| of consecutive frames vs random pairs inside the same source."""
    ordered = ordered[:max_frames]
    if len(ordered) < 4:
        return None
    fr = [small(p) for p in ordered]
    cons = np.mean([np.abs(fr[i + 1] - fr[i]).mean() for i in range(len(fr) - 1)])
    rnd = np.mean([np.abs(fr[a] - fr[b]).mean() for a, b in (rng.sample(range(len(fr)), 2) for _ in range(60))])
    return float(cons), float(rnd)


def cont_summary(label, sources, rng, strip_dir=None, strip_name=None):
    """sources: list of (name, natural-sorted paths). Compares natural vs lexicographic order on the same files."""
    nat_r, lex_r = [], []
    for name, paths in sources:
        pl = list(paths)
        lex = sorted(pl, key=lambda p: str(p))            # what data._discover does
        n = continuity(pl, rng, 60)
        l = continuity(lex, rng, 60)                       # first 60 in lexicographic order are NOT the first 60 naturally
        if n and l:
            nat_r.append(n[0] / n[1]); lex_r.append(l[0] / l[1])
    if not nat_r:
        return {}
    res = dict(sources_tested=len(nat_r), natural_cons_over_random=float(np.median(nat_r)),
               lexicographic_cons_over_random=float(np.median(lex_r)))
    if strip_dir and sources:
        name, paths = sources[0]
        step = max(1, len(paths) // 16)
        sheet([small(p, TH) for p in paths[:16 * step:step][:16]], 16, strip_dir / f"strip_{strip_name}_natural.png", f"{label} natural")
        lex = sorted(paths, key=lambda p: str(p))
        sheet([small(p, TH) for p in lex[:16]], 16, strip_dir / f"strip_{strip_name}_lexicographic.png", f"{label} lexicographic")
    return res


def video_stats(path, size=TH, max_frames=600):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return dict(error="cannot open")
    fps = cap.get(cv2.CAP_PROP_FPS); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames, idx = [], 0
    while len(frames) < max_frames:
        if idx % 5 == 0:
            ok, f = cap.read()
            if not ok:
                break
            g = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (size, size), interpolation=cv2.INTER_AREA)
            frames.append(g.astype(np.float32) / 255.0)
        else:
            if not cap.grab():
                break
        idx += 1
    cap.release()
    out = dict(fps=round(float(fps), 2), total=total, w=w, h=h, kept=len(frames))
    if len(frames) >= 2:
        d = np.array([np.abs(frames[i + 1] - frames[i]).mean() for i in range(len(frames) - 1)])
        means = np.array([f.mean() for f in frames]); stds = np.array([f.std() for f in frames])
        out.update(diff_mean=float(d.mean()), diff_p95=float(np.percentile(d, 95)), diff_max=float(d.max()),
                   cuts=int((d > 0.20).sum()), frac_dark=float((means < 0.05).mean()), frac_flat=float((stds < 0.03).mean()),
                   bright=float(means.mean()), static=float((d < 0.003).mean()))
    out["_thumbs"] = [frames[i] for i in (0, len(frames) // 2, len(frames) - 1)] if frames else []
    return out


def md5(path, nbytes=None):
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read(nbytes) if nbytes else f.read())
    return h.hexdigest()


def audit_p(root, p_dirs, loose_lr, out, rng, log):
    """P*/C/Data/{Dottest,Htest,OKNtest}/{LE,RE}/*_eye_frame_N.tiff.png : ordered eye-tracker crops."""
    rows, strips, hashes = [], [], defaultdict(list)
    for pd in sorted(p_dirs, key=lambda p: int(p.name[1:])):
        for leaf in sorted({f.parent for f in pd.rglob("*.png")}):
            frames = sorted(leaf.glob("*.png"), key=nat)
            test, side = leaf.parent.name, leaf.name
            if len(frames) < 8:
                rows.append(dict(p=pd.name, test=test, side=side, n=len(frames), note="too short")); continue
            gl = [load_gray(f, None) for f in frames[:1]]
            nums = [int(re.findall(r"frame_(\d+)", f.name)[0]) for f in frames]
            lex = sorted(frames, key=str)
            cn, cl = continuity(frames, rng, 80), continuity(lex, rng, 80)
            sm = [small(f) for f in rng.sample(frames, min(40, len(frames)))]
            means = np.array([s.mean() for s in sm]); stds = np.array([s.std() for s in sm])
            rows.append(dict(p=pd.name, test=test, side=side, n=len(frames), native=f"{gl[0][1][0]}x{gl[0][1][1]}",
                             frame_no_range=[min(nums), max(nums)], missing_numbers=int(max(nums) - min(nums) + 1 - len(set(nums))),
                             nat_ratio=round(cn[0] / cn[1], 3) if cn else None, lex_ratio=round(cl[0] / cl[1], 3) if cl else None,
                             mean=round(float(means.mean()), 3), flat_frac=float((stds < 0.03).mean())))
            for f in frames:
                hashes[md5(f)].append(f"{pd.name}/{test}/{side}/{f.name}")
            if side == "LE" and test == "Dottest" and len(strips) < 8:
                step = max(1, len(frames) // 16)
                strips.append(np.concatenate([cv2.resize(small(f, TH), (TH, TH // 2)) for f in frames[:16 * step:step][:16]], 1))
        log(f"{pd.name}: done")
    if strips:
        cv2.imwrite(str(out / "strip_P_dottest_LE.png"), (np.concatenate(strips, 0) * 255).astype(np.uint8))
    dups = [v for v in hashes.values() if len(v) > 1]
    # are the loose eye_frame files the PNGs missing from P1?
    p1 = [p for p in p_dirs if p.name == "P1"]
    loose_nums = {side: {int(re.findall(r"frame_(\d+)", f.name)[0]) for f in loose_lr[side]} for side in loose_lr}
    p1_info = {}
    for p in p1:
        for t in p.rglob("*_center.txt"):
            n = int(re.findall(r"frame_(\d+)", t.name)[0]); side = "left" if t.name.startswith("left") else "right"
            p1_info.setdefault((t.parent.parent.name, side), []).append(n)
    overlap = {f"{k[0]}/{k[1]}": dict(txt_frames=len(v), in_loose=len(set(v) & loose_nums[k[1]])) for k, v in p1_info.items()}
    rep = dict(rows=rows, duplicate_groups=len(dups), duplicate_examples=dups[:5], p1_txt_vs_loose_frames=overlap)
    (out / "report_p.json").write_text(json.dumps(rep, indent=1, default=str))
    log(f"P audit done: {len(rows)} sequences, {len(dups)} duplicate groups -> report_p.json")


def main():
    here = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(here.parent / "set"))
    ap.add_argument("--scraped", default=str(here.parent / "scraped_images"))
    ap.add_argument("--out", default=str(here / "audit"))
    ap.add_argument("--sample", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", choices=["p"], default=None, help="p = audit only the P* eye-tracking sequences")
    a = ap.parse_args()
    root, scraped, out = Path(a.root), Path(a.scraped), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    rep, t0 = {}, time.time()
    log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)

    top = sorted(root.iterdir())
    awake = sorted((root / "awake").glob("*.png"), key=nat) if (root / "awake").is_dir() else []
    sleepy = sorted((root / "sleepy").glob("*.png"), key=nat) if (root / "sleepy").is_dir() else []
    s_dirs = sorted([p for p in top if p.is_dir() and re.fullmatch(r"S_\d+", p.name)], key=nat)
    p_dirs = [p for p in top if p.is_dir() and re.fullmatch(r"P\d+", p.name)]
    loose = [p for p in top if p.is_file()]
    loose_mrl = sorted([p for p in loose if re.match(r"s\d{4}_", p.name) and p.suffix.lower() == ".png"], key=nat)
    loose_lr = {"left": sorted([p for p in loose if p.name.startswith("left_eye_frame_")], key=nat),
                "right": sorted([p for p in loose if p.name.startswith("right_eye_frame_")], key=nat)}
    pins = [p for p in loose if p.name.startswith("pin_")]
    vids = sorted([p for p in loose if p.suffix.lower() in VID], key=nat)
    known = set(loose_mrl) | set(loose_lr["left"]) | set(loose_lr["right"]) | set(pins) | set(vids)
    other_loose = [p for p in loose if p not in known]

    if a.only == "p":
        audit_p(root, p_dirs, loose_lr, out, rng, log)
        return

    # ---------------- inventory ----------------
    inv = dict(
        mrl_awake=len(awake), mrl_sleepy=len(sleepy), openeds_S_dirs=len(s_dirs),
        openeds_S_pngs=sum(len(list(d.glob("*.png"))) for d in s_dirs),
        openeds_S_npys=sum(len(list(d.glob("*.npy"))) for d in s_dirs),
        P_dirs_label_only=len(p_dirs), P_files_by_ext=dict(Counter(f.suffix for d in p_dirs for f in d.rglob("*") if f.is_file())),
        loose_mrl_png=len(loose_mrl), loose_left_eye=len(loose_lr["left"]), loose_right_eye=len(loose_lr["right"]),
        pins=dict(Counter(p.suffix.lower() for p in pins)), videos=len(vids), other_loose=[p.name for p in other_loose][:20])
    # what the current loader would pick up, to expose the grouping bug
    inv["loader_groups_for_awake_sleepy"] = "awake -> 1 group, sleepy -> 1 group (group key = first path component)"
    rep["inventory"] = inv
    log(f"inventory: {inv}")

    # ---------------- scraped ----------------
    sc = {}
    if scraped.exists():
        for p in sorted(scraped.iterdir()):
            if p.is_dir():
                c = Counter(f.suffix.lower() for f in p.rglob("*") if f.is_file())
                sc[p.name] = dict(c)
        ev = scraped / "archive (10)" / "openEDS_events"
        evs = [f for f in ev.rglob("*.npy")] if ev.exists() else []
        if evs:
            arr = np.load(rng.choice(evs), allow_pickle=True)
            sc["_events_sample"] = dict(shape=list(arr.shape), dtype=str(arr.dtype), n_files=len(evs))
        bb = scraped / "archive (10)" / "bbox"
        bbs = list(bb.rglob("*.txt")) if bb.exists() else []
        if bbs:
            sc["_bbox_sample"] = bbs[0].read_text()[:200]
    rep["scraped"] = sc
    log(f"scraped: {sc}")

    # ---------------- image statistics + contact sheets ----------------
    classes = {"mrl_awake": awake, "mrl_sleepy": sleepy, "openeds_S": [p for d in s_dirs for p in d.glob("*.png")],
               "loose_mrl": loose_mrl, "left_eye_frames": loose_lr["left"], "right_eye_frames": loose_lr["right"],
               "pins_loadable": [p for p in pins if p.suffix.lower() in IMG]}
    stats = {}
    for name, paths in classes.items():
        if not paths:
            continue
        s, thumbs = img_stats(paths, rng, a.sample)
        stats[name] = s
        sheet(thumbs[:48], 8, out / f"sheet_{name}.png", name)
        log(f"{name}: {s}")
    rep["image_stats"] = stats
    heic = [p.name for p in pins if p.suffix.lower() not in IMG]
    rep["pins_not_loadable_by_loader"] = heic

    # ---------------- temporal continuity ----------------
    cont = {}
    subj = defaultdict(list)
    for p in awake + sleepy:
        subj[p.parent.name + "/" + p.name.split("_")[0]].append(p)
    subj_sources = [(k, v) for k, v in subj.items() if len(v) >= 60]
    rng.shuffle(subj_sources)
    cont["mrl_per_subject"] = cont_summary("mrl", subj_sources[:15], rng, out, "mrl")
    gaps = []
    for k, v in subj_sources[:15]:
        fn = [int(p.name.split("_")[1]) for p in v]
        gaps += list(np.diff(sorted(fn)))
    cont["mrl_frame_index_gap_median_p90"] = [float(np.median(gaps)), float(np.percentile(gaps, 90))] if gaps else None
    cont["mrl_subjects_total"] = len(subj)
    s_sources = [(d.name, sorted(d.glob("*.png"), key=nat)) for d in s_dirs]
    rng.shuffle(s_sources)
    cont["openeds_S"] = cont_summary("S", s_sources[:15], rng, out, "S")
    for side in ("left", "right"):
        if loose_lr[side]:
            cont[f"{side}_eye_frames"] = cont_summary(side, [(side, loose_lr[side])], rng, out, side)
    # across-source baseline: unrelated frames
    pool = [small(p) for p in rng.sample(classes["openeds_S"], 40)]
    cont["random_pair_baseline_openeds"] = float(np.mean([np.abs(pool[i] - pool[j]).mean() for i, j in (rng.sample(range(40), 2) for _ in range(80))]))
    rep["continuity"] = cont
    log(f"continuity: {cont}")

    # ---------------- duplicates ----------------
    dup = {}
    names_ad = {p.name: p for p in awake + sleepy}
    dup["loose_mrl_same_name_as_awake_sleepy"] = sum(1 for p in loose_mrl if p.name in names_ad)
    dup["loose_mrl_total"] = len(loose_mrl)
    ad_hash = {}
    for p in awake + sleepy:
        ad_hash.setdefault(md5(p), p.name)
    h_hit = sum(1 for p in loose_mrl if md5(p) in ad_hash)
    dup["loose_mrl_byte_identical_to_awake_sleepy"] = h_hit
    dup["awake_sleepy_internal_byte_duplicates"] = len(awake) + len(sleepy) - len(ad_hash)
    log(f"dups mrl: {dup}")
    # videos: size + head hash
    sizes = defaultdict(list)
    for v in vids:
        sizes[(v.stat().st_size, md5(v, 1 << 20))].append(v.name)
    dup["duplicate_video_groups"] = [v for v in sizes.values() if len(v) > 1]
    # S_ internal sample
    sh = Counter(md5(p, 1 << 16) for p in rng.sample(classes["openeds_S"], min(3000, len(classes["openeds_S"]))))
    dup["openeds_S_sampled_head_duplicates"] = sum(c - 1 for c in sh.values())
    rep["duplicates"] = dup

    # ---------------- labels ----------------
    lab = {}
    npys = [p for d in s_dirs for p in d.glob("*.npy")]
    named = Counter("label_N" if p.name.startswith("label_") else "N" for p in npys)
    lab["npy_name_kinds"] = dict(named)
    cls_count, shapes = Counter(), Counter()
    for p in rng.sample(npys, min(150, len(npys))):
        m = np.load(p)
        shapes[f"{m.shape}/{m.dtype}"] += 1
        u, c = np.unique(m, return_counts=True)
        for k, v in zip(u, c):
            cls_count[int(k)] += int(v)
    tot = sum(cls_count.values()) or 1
    lab["shapes"] = dict(shapes)
    lab["class_fraction"] = {k: round(v / tot, 4) for k, v in sorted(cls_count.items())}
    lab_txt = root / "labels.txt"
    lab["labels_txt_lines"] = sum(1 for _ in open(lab_txt)) if lab_txt.exists() else None
    img_txt = root / "images.txt"
    lab["images_txt_lines"] = sum(1 for _ in open(img_txt)) if img_txt.exists() else None
    # image <-> npy pairing for the unprefixed names
    unpaired = sum(1 for d in s_dirs for p in d.glob("*.png") if not (d / (p.stem + ".npy")).exists())
    lab["S_png_without_matching_N_npy"] = unpaired
    rep["labels"] = lab
    log(f"labels: {lab}")

    # ---------------- videos ----------------
    vs, tiles = {}, []
    for i, v in enumerate(vids):
        st = video_stats(v)
        th = st.pop("_thumbs", [])
        vs[v.name] = st
        if th:
            row = np.concatenate([cv2.resize(t, (TH, TH)) for t in th], 1)
            cv2.putText(row, f"{i}", (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1.0, 1)
            tiles.append(row)
        if (i + 1) % 10 == 0:
            log(f"videos {i + 1}/{len(vids)}")
    rep["videos"] = vs
    # sheets: 12 rows of 3 frames each per image, so each file stays viewable
    for k in range(0, len(tiles), 12):
        chunk = tiles[k:k + 12]
        cols = 2
        rows = [np.concatenate(chunk[r:r + cols] + [np.zeros_like(chunk[0])] * (cols - len(chunk[r:r + cols])), 1) for r in range(0, len(chunk), cols)]
        cv2.imwrite(str(out / f"sheet_videos_{k // 12}.png"), (np.concatenate(rows, 0) * 255).astype(np.uint8))
    good = [(n, s) for n, s in vs.items() if "diff_mean" in s]
    rep["video_summary"] = dict(
        n=len(vs), unreadable=[n for n, s in vs.items() if "error" in s],
        kept_total=sum(s.get("kept", 0) for s in vs.values()),
        with_cuts=[n for n, s in good if s["cuts"] > 0], mostly_dark=[n for n, s in good if s["frac_dark"] > 0.3],
        mostly_flat=[n for n, s in good if s["frac_flat"] > 0.3], mostly_static=[n for n, s in good if s["static"] > 0.7],
        too_short_for_T8=[n for n, s in vs.items() if s.get("kept", 0) < 8],
        res=dict(Counter(f"{s.get('w')}x{s.get('h')}" for s in vs.values()).most_common(6)),
        fps=dict(Counter(s.get("fps") for s in vs.values()).most_common(6)))
    log(f"video summary: {rep['video_summary']}")

    (out / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    lines = []
    for sec, val in rep.items():
        if sec == "videos":
            continue
        lines.append(f"== {sec}")
        lines.append(json.dumps(val, indent=1, default=str))
    (out / "report.txt").write_text("\n".join(lines))
    log(f"done -> {out}")


if __name__ == "__main__":
    sys.exit(main())
