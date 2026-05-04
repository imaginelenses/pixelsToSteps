# Cart-Pole Capture Script Options

This document explains the Python tools used to render the simulated cart-pole, capture binary frames, and build a fixed-rate video from those frames.

The two scripts are:

- `python/teacher_policy.py`
- `python/cartpole_frames_to_video.py`

## Workflow

The intended workflow is:

1. Run `teacher_policy.py` to simulate the same cart-pole physics used by the LQR script.
2. During capture, save one binary frame per control action as `frame_000000.png`, `frame_000001.png`, and so on.
3. Save a per-frame `capture_trace.csv` containing cart position, angle, and control values aligned to the captured PNG frames.
3. Optionally show a live Gym preview while those frames are being captured.
4. Optionally assemble the saved PNG frames into a lossless video at the control-rate FPS.

Important timing rule:

- one captured frame = one control action
- output video FPS = control rate in Hz

This means the saved frame sequence is temporally aligned with the simulated control loop.

## Output Artifacts

The capture script can produce four artifacts:

- a frame directory containing binary PNGs
- a `capture_metadata.json` file stored in that frame directory
- a `capture_trace.csv` file stored in that frame directory
- a lossless `.mkv` video assembled from the PNG frames

The PNG frames are the source-of-truth capture artifact.

Each frame is:

- binary only: `0` or `255`
- black background
- white pole pixels
- default size `125 x 160` as `height x width`

## Script: `python/teacher_policy.py`

### Purpose

This script:

- reuses the same plant and rollout logic from `python/cartpole_lqr_gain.py`
- simulates the cart-pole state trajectory
- renders Gym frames from that trajectory
- extracts a binary pole mask from each rendered frame
- saves one PNG per simulated control step
- logs per-frame cart position, angle, and control values to `capture_trace.csv`
- optionally previews the run live in a Gym window
- stops capture once the cart-pole reaches the upright goal state
- optionally assembles the final `.mkv` video immediately after capture

### Main Options

#### Simulation horizon and initial condition

- `--steps`
  - Number of control steps to simulate.
  - Default: `1500`

- `--horizon-steps`
  - Alias for `--steps`.
  - If provided, it overrides `--steps`.
  - Default: not set

- `--theta0-deg`
  - Fixed starting pole angle in degrees from upright.
  - If omitted, the script samples a random initial angle.
  - Default: not set

- `--theta0-range-deg`
  - Range for the random initial angle when `--theta0-deg` is not provided.
  - The sampled angle is drawn uniformly from `[-range, +range]`.
  - Default: `8.0`

- `--seed`
  - Random seed used for the initial angle sampling.
  - Default: not set

#### Control and plant parameters

- `--control-rate-hz`
  - Control loop rate in Hz.
  - Also sets the capture FPS and video FPS.
  - Sample time is `1 / control_rate_hz`.
  - Default: `250.0`

- `--max-step-rate-steps-s`
  - Maximum control command magnitude used in the design and saturation logic.
  - Units: `steps/s`
  - Default: `12000.0`

- `--actuator-time-constant-s`
  - First-order actuator time constant used by the same physical model as the LQR script.
  - Units: seconds
  - Default: `0.03`

- `--max-cart-accel-steps-s2`
  - Validation-only cart acceleration limit.
  - Units: `steps/s^2`
  - Default: `150000.0`

- `--r-weight`
  - Scalar control penalty for the LQR design.
  - Default: `1e-6`

- `--command-delay-s`
  - Pure command-to-motion delay used only in the simulation validation.
  - Units: seconds
  - Default: `0.0`

#### Teacher gate thresholds

There are no teacher enable, disable, or sticky gate options in the capture script anymore.

The capture script now behaves like a normal cart-pole stabilization run that still uses the same step-domain physical parameters.

### Goal Stop Behavior

Capture stops early when the full cart-pole state reaches a settled goal state.

The current goal condition is:

- `|cart_position| <= 10 steps`
- `|cart_velocity| <= 20 steps/s`
- `|theta| <= 1.0 deg`
- `|theta_dot| <= 10.0 deg/s`
- all of the above must hold continuously for `0.25 s`

This goal only determines when capture stops.

- The underlying physical model is unchanged.
- The rollout still uses the same step-domain state and the same physical parameters.
- If the goal is never reached, capture continues until failure or the requested horizon.

#### Output frame geometry

- `--frame-height-px`
  - Output frame height in pixels.
  - Default: `125`

- `--frame-width-px`
  - Output frame width in pixels.
  - Default: `160`

Note:

- The default geometry is `125 x 160` as `height x width`.

#### Live preview options

- `--render`
  - Open Gym's human viewer while capture runs.
  - The preview is for inspection only. The saved PNG sequence still contains every captured frame.

- `--render-every`
  - Update the live preview every `N` captured control steps.
  - This affects only the preview window.
  - It does not drop saved frames from the PNG sequence.
  - Default: automatic, chosen to keep the preview near `25 Hz`

Important preview note:

- The saved frames still preserve one-frame-per-control-step timing.
- A desktop display usually cannot visibly present every update at very high control rates such as `250 Hz`.
- The preview window is therefore a realtime inspection aid, not the authoritative timing artifact.
- After capture finishes, the human viewer stays open on the final state until the user interrupts the script.
- By default, the script decimates the human preview cadence automatically so the live preview stays close to realtime instead of trying to draw every control step.

#### Output paths and assembly control

- `--frame-png-dir`
  - Directory where the binary PNG frames are written.
  - Also stores `capture_metadata.json` and `capture_trace.csv`.
  - Default: derived from `--video-output` as `<video_stem>_frames`

- `--video-output`
  - Lossless `.mkv` output path.
  - Default: `python/captures/cartpole_binary_TIMESTAMP.mkv`

- `--skip-video-assembly`
  - Capture PNG frames and metadata only.
  - Skip the final video assembly step.
  - Useful when you want to inspect or post-process the frames separately.

### What the script prints

At the end of a run, the script prints:

- frame directory path
- metadata path
- trace CSV path
- video path or a note that assembly was skipped
- frame count
- frame size
- frame rate
- sample time
- simulated capture time
- wall-clock capture time before the final hold
- preview cadence when live preview is enabled
- initial angle
- capture end reason
- goal step when the goal is reached
- stop state at the last captured frame
- rollout survival summary
- rollout-end state
- rollout-end target command
- peak command, velocity, and acceleration statistics
- saturation counts

### Example Commands

Capture frames only:

```bash
python3 python/teacher_policy.py \
  --steps 1500 \
  --theta0-deg 2 \
  --control-rate-hz 250 \
  --frame-height-px 125 \
  --frame-width-px 160 \
  --frame-png-dir python/captures/run_frames \
  --skip-video-assembly
```

Capture frames and preview live while saving a video:

```bash
python3 python/teacher_policy.py \
  --steps 1500 \
  --theta0-deg 2 \
  --control-rate-hz 250 \
  --render \
  --render-every 1 \
  --frame-png-dir python/captures/run_frames \
  --video-output python/captures/run.mkv
```

Use the alias for horizon and a random start angle:

```bash
python3 python/teacher_policy.py \
  --horizon-steps 2000 \
  --theta0-range-deg 4 \
  --seed 123 \
  --control-rate-hz 200
```

## Script: `python/cartpole_frames_to_video.py`

### Purpose

This script converts an already-captured PNG sequence into a lossless video.

Use it when:

- you ran the capture script with `--skip-video-assembly`
- you want to rebuild the video from the same saved frames
- you want the video generation step to be separate from the capture step

### Options

- `--frame-png-dir`
  - Required.
  - Directory containing `frame_*.png` files.
  - It may also contain `capture_metadata.json`.

- `--video-output`
  - Output video path.
  - If omitted, the script derives it from the frame directory name.
  - Example: `run_frames` becomes `run.mkv`.

- `--control-rate-hz`
  - Explicit FPS to use for the output video.
  - If omitted, the script reads `frame_rate_hz` from `capture_metadata.json`.

### Metadata behavior

If `capture_metadata.json` is present, the script uses it to recover:

- `frame_rate_hz`
- frame size information
- frame count context from the capture step

If the metadata file is missing, you must pass `--control-rate-hz` explicitly.

### What the script checks

Before video assembly, it verifies:

- that `frame_*.png` files exist
- that at least the first frame can be read
- the frame shape of the PNG sequence
- the unique values present in the first PNG frame

### What the script prints

It prints:

- frame directory path
- output video path
- number of PNG frames found
- frame shape
- selected frame rate
- unique values in the sampled frame

### Example Commands

Build a video from a frame directory using stored metadata:

```bash
python3 python/cartpole_frames_to_video.py \
  --frame-png-dir python/captures/run_frames
```

Build a video with an explicit FPS:

```bash
python3 python/cartpole_frames_to_video.py \
  --frame-png-dir python/captures/run_frames \
  --video-output python/captures/run_250hz.mkv \
  --control-rate-hz 250
```

## Notes On Video Encoding

- Video assembly uses `ffmpeg`.
- The current path writes a lossless `FFV1` video in an `.mkv` container.
- The PNG frames remain the exact binary capture artifact.
- The video is a packaged view of those same frames at the chosen fixed FPS.

## Quick Reference

Use `teacher_policy.py` when you want to:

- simulate the cart-pole
- capture binary frames
- log per-frame cart position, angle, and control values
- optionally watch the run live
- optionally build the video immediately

Use `cartpole_frames_to_video.py` when you want to:

- turn an existing frame directory into a video
- rebuild a video later from the saved PNG frames
- keep capture and video assembly as separate steps