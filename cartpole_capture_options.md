# Nominal CartPole Capture Workflow

This document covers the current nominal Gym CartPole workflow used by:

- `simulation/python/teacher_policy.py`
- `simulation/python/collect_teacher_demos.py`
- `simulation/python/student_policy.ipynb`
- `simulation/python/cartpole_frames_to_video.py`

## Unit Convention

The underlying Gym `CartPole-v1` environment still evolves in meters, meters/s, and newtons.

The exported teacher and student interface is now intentionally in step-domain units:

- cart position: `steps`
- cart velocity: `steps/s`
- cart command: `steps/s`
- pole angle: `rad`
- pole angle rate: `rad/s`

The shared conversion is defined in `simulation/python/nominal_cartpole.py` through:

- `meters_per_step`
- `newtons_per_step_rate`

This means:

- `capture_trace.csv` uses `cart_position_steps`, `cart_velocity_steps_s`, and `applied_control_steps_s`
- `capture_metadata.json` records `unit_system = "steps"`
- `teacher_lqr_controller.json` uses `unit_system = "steps"`
- `student_policy.ipynb` fits the observer in the same step-domain state and control units

## Single Teacher Run

Run one nominal Gym CartPole teacher rollout:

```bash
python3 simulation/python/teacher_policy.py \
  --theta0-deg 6 \
  --steps 180 \
  --control-rate-hz 50 \
  --max-step-rate-steps-s 12000 \
  --max-force-n 10 \
  --meters-per-step 0.0000125 \
  --skip-video-assembly \
  --frame-png-dir simulation/captures/cartpole_nominal_steps_demo_frames
```

This example assumes you are running from the repository root.

Key options:

- `--theta0-deg`
  - Fixed initial pole angle in degrees.
- `--steps`
  - Maximum control steps for the rollout.
- `--control-rate-hz`
  - Gym CartPole sample rate. Default is `50`.
- `--max-step-rate-steps-s`
  - Step-domain control saturation used by the teacher/export interface.
- `--max-force-n`
  - Physical CartPole force limit used internally by Gym.
- `--meters-per-step`
  - Conversion from exported cart steps back to Gym meters.
- `--frame-png-dir`
  - Output directory for frames, trace, metadata, and controller JSON.
- `--observer-json`
  - Optional image-only observer JSON for closed-loop rollout from rendered frames.
- `--render`
  - Show the human viewer while the frames are being captured.

Teacher artifacts per run:

- `capture_metadata.json`
- `capture_trace.csv`
- `teacher_lqr_controller.json`
- `frame_000000.png`, `frame_000001.png`, ...
- optional `.mkv` video if `--skip-video-assembly` is omitted

## Multi-Demo Collection Run

Collect multiple teacher demos under one shared configuration and one shared `collection_id`:

```bash
python3 simulation/python/collect_teacher_demos.py \
  --steps 180 \
  --theta0-deg-list -10 -6 -2 2 6 10
```

Example with the current scaled plant settings used in recent robustness sweeps:

```bash
python3 simulation/python/collect_teacher_demos.py \
  --true-masspole-scale 1.1 \
  --true-half-pole-length-scale 0.9 \
  --theta0-deg-list -12 -8 -4 4 8 12
```

This creates a collection directory under:

```text
simulation/captures/collections/<collection_id>/
```

Inside that directory you get:

- one frame directory per initial angle
- one `collection_manifest.json`
- matching metadata in every demo with the same `collection_id`

The collector always reuses the same shared parameters for every demo in that run, so the notebook can safely train on the whole collection as one dataset.

Important behavior:

- by default the collector writes PNGs, trace CSVs, metadata, and controller JSON only
- use `--assemble-videos` if you also want a video per demo
- every teacher demo written by the collector includes `collection_id` and `demo_name` in `capture_metadata.json`

## Notebook Training

Open `simulation/python/student_policy.ipynb` and run the code cells in order.

The notebook now:

- recursively searches `captures/` and `simulation/captures/`
- finds the latest capture bundle
- if that bundle has a `collection_id`, automatically selects all matching demos from that same collection run
- fits the observer using the teacher trace schema from that collection
- exports a nominal step-domain observer JSON

For nominal step-domain collections, the notebook export is:

- `simulation/python/learned_pixels_to_nominal_steps_observer.json`

The observer target is recorded as:

- `pixels-to-nominal-cartpole-steps`

## Image-Only Rollout

After training the notebook, run the image-only observer rollout:

```bash
python3 simulation/python/teacher_policy.py \
  --observer-json simulation/python/learned_pixels_to_nominal_steps_observer.json \
  --theta0-deg 6 \
  --steps 120 \
  --skip-video-assembly \
  --frame-png-dir simulation/captures/cartpole_nominal_steps_observer_frames
```

This rollout:

- keeps the true Gym CartPole state hidden inside the plant
- computes control only from the observer estimate `x_hat`
- logs both the true step-domain state and the estimated step-domain state to `capture_trace.csv`

Estimated-state trace columns include:

- `estimated_cart_position_steps`
- `estimated_cart_velocity_steps_s`
- `estimated_pole_angle_rad`
- `estimated_pole_angle_rate_rad_s`

## Selection Rule In The Notebook

The notebook prefers dataset consistency over mixing arbitrary captures.

Its selection rule is:

1. find the latest complete capture bundle
2. read its rollout model, frame geometry, frame rate, unit system, and collection id
3. if a `collection_id` is present, train on all matching demos from that same collection run
4. otherwise fall back to matching bundles with the same rollout and schema settings

This is the mechanism that ensures the student fit uses one coherent teacher data collection run.

## Current Practical Note

The teacher state-feedback policy in the nominal step-domain workflow is stable, but the learned image observer may still fail early depending on the collected dataset and the observer fit quality.

That is an observer-quality issue, not a unit-consistency issue.
