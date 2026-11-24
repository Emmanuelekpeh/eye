# Interactive Visual Substrate (IVS) — V1 research prototype

A CPU-first prototype for a small, continually developing eye-world model.

## What exists in this first build

- A compact CNN visual encoder.
- A recurrent state-space model with deterministic and stochastic latent state.
- A convolutional decoder for image reconstruction.
- A sequence dataset loader for image folders and eye videos.
- A basic reconstruction + latent KL training objective.
- A transparent adaptive learning-rate controller baseline.
- A smoke test for forward/backward execution.

This is **not yet** the complete interactive world model. In particular, the action-conditioned dynamics are not yet trained against controlled interventions, the plasticity controller is a heuristic baseline rather than a learned meta-controller, and the generative decoder is not yet validated for novel eye generation. Those are explicit next milestones, not capabilities we claim already work.

## Hardware target

The defaults are intentionally modest: grayscale 64×64 frames, short sequences, small latent state, batch size 4 and CPU execution. Start with a small sample. Do not begin by loading a multi-gigabyte dataset into memory.

## Environment setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\\Scripts\\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Dataset layout

Place legally usable eye images/videos under `data/eyes`:

```text
data/eyes/
  session_001/
    frame_000001.png
    frame_000002.png
    frame_000003.png
  session_002.mp4
  still_eye_001.png
```

Images in a subfolder are treated as an ordered sequence. Video files are sampled at a modest rate. Loose images are treated as individual still-image sequences. Keep subject/session splits separate for evaluation; do not randomly split adjacent frames from the same recording across train and test.

### Dataset notes

- LPW contains 66 eye-region videos from 22 participants, recorded at approximately 95 FPS across varied everyday lighting and gaze conditions. Its stated use restriction is non-commercial scientific research; check the current dataset agreement before use.
- OpenEDS / OpenEDS2020 are relevant research datasets with real eye sequences. Access and use conditions differ by release; obtain the dataset and confirm its current terms directly from the official research release before downloading or redistributing it.

Do not bundle dataset images in this repository.

## What changed in the patch (v1.1)

See `CHANGES.md`. Short version: loss rescaled + free bits + KL balancing (collapse guard), real
multi-step prediction loss, synthetic action-labelled interventions (`ivs/actions.py`), group-level
subject split, stills excluded by default, lazy cv2 import, per-module plasticity with a *measured*
retention probe, and evaluation against mean-image and copy-last-frame baselines.

## Touch / observer UI

```bash
python -m ivs.ui --checkpoint checkpoints/latest.pt [--seed N]    # then open http://127.0.0.1:8765
```

No dataset is read and nothing synthetic is drawn: the canvas is the trained model's own render, seeded from noise
(its prior sampled from a zero state; `--seed`, or the "New noise eye" button). You touch it: **Grab & shake**
scatters pixels under the pointer (faster shake = stronger), *Move eye / Move light / exposure* use the same
hidden-intervention transform the model was trained on. The model sees only pixels and rate-limited action deltas,
never the hidden state. After every tick the canvas is overwritten by the model's belief (slider: perceived <-> expected)
with the touch stripped off so it is not compounded. **Intervene: off** ignores all touch so the world model runs on
its own. A frozen copy of the checkpoint runs on the same noise and input as the control. Panels: expected vs
perceived vs frozen frames, surprise map, live-vs-frozen error, weight drift per module/layer vs birth, plasticity
scales and retention, latent mean/uncertainty, encoder activations, snapshots, record/replay, save/resume world
(`checkpoints/world.pt`).

*Dream*: touch reaches the model only as actions and each model perceives its own render, so the error chart shows
dissonance (expected vs perceived). *Learning* (off by default, `--learn-every`) trains the RSSM only; encoder and
decoder are frozen. Signal = dissonance on what it perceived, plus pseudo-rehearsal toward the frozen copy's
imagination under random actions. Retention = distance to the frozen copy on a fixed probe (`--retain-tol`).
Observed on the current checkpoint: free-running for hundreds of ticks drifts to a bright blob (watch loop) or to
black (pure imagination); the model was only trained on 8-frame windows.

**Birth memory (default way to run, Nov 22).** Instead of noise, start from the model's own state after it perceived one real
held-out eye (stored as latent only, bound to the checkpoint by sha256):

```bash
python scripts/make_birth.py --checkpoint pilot_results/kaggle_long3_30k/pilot/L1_v6_long/model.pt --out checkpoints/birth_L1.pt
python -m ivs.ui --checkpoint pilot_results/kaggle_long3_30k/pilot/L1_v6_long/model.pt --birth checkpoints/birth_L1.pt --world checkpoints/world_L1.pt
```

`make_birth.py` perceives one window per held-out recording, ranks them by how long the UI dream keeps a moving eye, saves the
best 8 and checks them on fresh noise against noise starts (it writes `birth_L1.sheet.png`: real | birth render | dream ticks;
LOOK at it). "New remembered eye" picks another one (`seed % count`). Use a separate `--world` file per checkpoint.
Honest limits: the start is a clear eye that lasts ~10-20 ticks, then the dream wanders through lids, skin and colour scenes
as before.

Zero touch = no light (v1.6): state `[3,4]` is only the light's *position*, and training always adds the +0.5 blob
(`InterventionSampler.apply_state(light=None)`). The UI now passes a light switch (`light` in `/api/tick`, `/api/set_state`,
reported back): off until you use *Move light* or *Spotlight left*; *Reset touch* and a new eye switch it off. Training code is
untouched and still lights every state, so the model's training inputs always contained the blob; how it reacts to unlit
input in the UI is only checked by looking (see CHANGES v1.6).

## Video cache + pilot (Nov 20, v1.4)

`scripts/build_cache.py` turns the raw videos into `../data/cache/videos/*.npz` (96 px high, aspect kept, native colour,
sequences split at cuts/black frames, letterbox cropped, ~20 fps, <=3000 frames per video as evenly spaced chunks,
`manifest.json` with colour class + group per file). It never modifies raw data. `ivs/cachedata.py` (`CacheBank`) samples
it group-balanced, never across a sequence boundary, with a class-stratified train/val split by group.
`ivs/pilot.py` trains and evaluates one idea per run (R0..R6 in `PRESETS`) and writes metrics + sheets;
`scripts/kaggle_pilot.py` runs a set of presets on Kaggle's 2 GPUs. `train.py`/`data.py` (raw-folder path) are unchanged.

```bash
python tests/test_pilot.py                                      # real cache, CPU, ~40 s
python -m ivs.pilot --cache ../data/cache/videos --out pilot_out --runs R1_rgb_base --steps 4000 --device auto
```

Pilot results and the Kaggle workflow are in `../archive/docs_2026-11-22/backlog_full.md` (P0b, P0d; the live backlog is `../backlog.md`). Colour: the model gets RGB
(`ModelConfig.channels=3`); `tint_mode=gray` treats the monochrome-camera videos (uniform lavender cast) as gray so the
cast is not learned as eye colour. The old UI/checkpoint are grayscale 64 px and unaffected.

## Clarity ceiling (Nov 22, v1.7)
`python scripts/clarity_ceiling.py` (no training, CPU, ~1.5 min; read-only on `set/`) shrinks real native frames to 96/128/192/256 px
like the cache builders and scales them back, so you see what the resolution alone removes. On the full run the 96 px cache keeps
only .56 (OpenEDS) / .42 (LPW gray) / .60 (tint) / .58 (colour) of the edge strength where native frames are sharpest; 128 px: .65/.50/.67/.65,
192 px: .79/.65/.79/.76 (`pilot_results/clarity_ceiling/ceiling.json`, sheets there, LOOK at them). The cache is stored 96 px high, so a
128 px model needs a cache rebuilt from `set/` (`build_cache.py` has `HEIGHT = 96`; not yet parameterised). Tests: `python tests\test_clarity.py`.

## Smoke test

From the project root:

```bash
python scripts/smoke_test.py
python tests/test_core.py
```

Pipeline test on synthetic data (not real eyes):

```bash
python scripts/make_synthetic.py --out data/synthetic
python -m ivs.train --data data/synthetic --epochs 5 --batch-size 4 --sequence-length 8
```

Read the `verdict=` line each epoch: if `recon_beats_mean_image` or `pred_beats_copy_last` is False, the model has not learned anything beyond a trivial baseline, whatever the loss curve says.

## First training run

```bash
python -m ivs.train --data data/eyes --epochs 2 --batch-size 2 --sequence-length 4 --image-size 64 --device cpu
```

The first run should be treated as a pipeline test, not evidence of learned world dynamics. Checkpoints are written to `checkpoints/latest.pt` and scalar history to `checkpoints/history.json`.

## Current limitations / next work

1. Add a proper train/validation/test split by subject and recording session.
2. Add controlled action-labelled transformations so the dynamics model can learn intervention consequences rather than receiving all-zero actions.
3. Add a predictive prior loss and evaluate open-loop multi-step prediction.
4. Add replay memory and retention tests before enabling persistent online updates.
5. Replace the heuristic plasticity controller with a controlled, learnable mechanism only after establishing baselines.
6. Build the touch interface around model-generated observations and action-conditioned latent transitions.
7. Add checkpointed world state, replayable interaction history and a live observer.
8. Add tests for reconstruction, prediction, uncertainty calibration, adaptation and forgetting.

## Research integrity

A changing activation map is not proof of learning. A changing weight is not proof of useful adaptation. Every claimed capability needs an independent test against a frozen baseline and repeatable disturbances.
