"""Record what the LIVE UI backend shows, tick by tick, through its own /api/tick (the loop the page runs), no browser needed.

Usage (a UI server must be running: python -m ivs.ui --checkpoint <model.pt> --world <tmp.pt> --fresh --port 8767):
  python scripts/record_ui_rollout.py --url http://127.0.0.1:8767 --ticks 400 --dream --out pilot_results/ui_rollouts/L1_dream [--watch-first 20]
  python scripts/record_ui_rollout.py --url http://127.0.0.1:8767 --ticks 400 --out pilot_results/ui_rollouts/L1_watch

--dream     send dream=true every tick (the model perceives its own render, touch = action only); without it the model watches the canvas.
--trust     the 'heal from' slider (0 perceived .. 1 expected), default 0.5 like the page.
--reseed    POST /api/reseed first (new noise eye; default on). --watch-first N: N watching ticks before the dream starts (the page's order).
Saves the panel the page calls `stim` (what the model perceives) at every tick: <out>.mp4 (tick burned in), <out>_contact.png (every --every ticks,
8 per row) and prints z/pupil/dissonance numbers every 50 ticks.
"""
import argparse
import base64
import json
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TILE = 128


def call(url, path, body=None):
    req = urllib.request.Request(url + path, data=json.dumps(body or {}).encode(), headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def decode(data_url):
    raw = base64.b64decode(data_url.split(",", 1)[1])
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1]      # RGB


def labelled(img, text):
    t = cv2.resize(np.ascontiguousarray(img), (TILE, TILE), interpolation=cv2.INTER_CUBIC)
    cv2.putText(t, text, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
    return t


def episode_report(frames, segmenter_path):
    """Dream persistence from the frames the UI showed (ivs.eyemetric.nested_flags + eye_episodes), plus the share of coloured frames
    (chroma > .03) per 100 ticks. The detector is lenient (a white blob can pass), so LOOK at the contact sheet next to these numbers."""
    import torch
    from ivs import eyemetric as E
    x = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0
    flags = E.nested_flags(E.segment(E.load_segmenter(segmenter_path), E.gray(x))).numpy()
    ep = E.eye_episodes(flags)
    chroma = ((x[:, 0] - x[:, 1]).abs().mean((1, 2)) + (x[:, 2] - x[:, 1]).abs().mean((1, 2))).numpy()
    return (f"eye-like ticks {ep['eye_frac']:.2f} | episodes {ep['n_episodes']} (median {ep['median_len']:.0f} ticks, max {ep['max_len']}, "
            f">=20 ticks: {ep['n_long']})\n   coloured per 100 ticks: " + " ".join(f"{(chroma[i:i + 100] > .03).mean():.2f}" for i in range(0, len(chroma), 100)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segmenter", default=None, help="eye segmenter checkpoint (pilot_results/eyemetric/segmenter.pt): print eye-episode statistics")
    ap.add_argument("--url", required=True)
    ap.add_argument("--ticks", type=int, default=400)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--dream", action="store_true")
    ap.add_argument("--watch-first", type=int, default=0)
    ap.add_argument("--trust", type=float, default=0.5)
    ap.add_argument("--no-reseed", action="store_true")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--save-frames", action="store_true", help="also write <out>_frames.npz (uint8 (N,H,W,3)) for analysis")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not a.no_reseed:
        call(a.url, "/api/reseed")
    zero = [0.0] * 5
    frames, rows = [], []
    for i in range(a.watch_first + a.ticks):
        dream = a.dream and i >= a.watch_first
        r = call(a.url, "/api/tick", dict(target=zero, scatter=[], intervene=True, learn=False, dream=dream, trust=a.trust))
        img = decode(r["img"]["stim"])
        frames.append((r["tick"], dream, img))
        if i % 50 == 0:
            lat = np.array(r["latent"]["mean"])
            rows.append(f"   i={i:4d} tick {r['tick']:5d} dream={dream} latent mean rms {np.sqrt((lat ** 2).mean()):.2f} metric {r['metric']} mse {r['mse']['pred']:.4f}"
                        + (f" pupil {np.round(r['gaze']['pupil'], 2).tolist()}" if r["gaze"]["pupil"] else ""))
    print("\n".join(rows))
    if a.segmenter:
        print(episode_report(np.stack([f[2] for f in frames]), a.segmenter))
    if a.save_frames:
        np.savez_compressed(str(out) + "_frames.npz", frames=np.stack([f[2] for f in frames]), dream=np.array([f[1] for f in frames]))
    vw = cv2.VideoWriter(str(out) + ".mp4", cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (TILE, TILE))
    for i, (tick, dream, img) in enumerate(frames):
        vw.write(cv2.cvtColor(labelled(img, f"{'D' if dream else 'W'}{i}"), cv2.COLOR_RGB2BGR))
    vw.release()
    pick = list(range(0, len(frames), a.every))
    tl = [labelled(frames[i][2], f"{'D' if frames[i][1] else 'W'}{i}") for i in pick]
    tl += [np.zeros_like(tl[0])] * (-len(tl) % 8)
    sheet = np.concatenate([np.concatenate(tl[r * 8:(r + 1) * 8], 1) for r in range(len(tl) // 8)], 0)
    cv2.imwrite(str(out) + "_contact.png", cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    print(f"wrote {out}.mp4 and {out}_contact.png ({len(frames)} frames, tiles every {a.every}; D = dreaming, W = watching)")


if __name__ == "__main__":
    main()
