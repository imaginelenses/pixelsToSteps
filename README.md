# pixelsToSteps

This repository is split into two working areas:

- `hardware/`: ESP32 firmware, hardware-side Python utilities, and hardware capture output.
- `simulation/`: CartPole simulation, observer training and evaluation scripts, saved observer JSON files, and the live web UI.

## Layout

```text
hardware/
  firmware/
    include/
    src/
  python/
  captures/
simulation/
  python/
  captures/
  observers/
  web/
```

## Firmware

The PlatformIO project now lives at `hardware/platformio.ini`.

Typical commands:

```bash
cd hardware
pio run
pio run -t upload
```

## Simulation Python

The simulation and observer scripts now live under `simulation/python/`.

Examples:

```bash
python3 simulation/python/teacher_policy.py --help
python3 simulation/python/collect_teacher_demos.py --help
python3 simulation/python/cartpole_lqr_gain.py
```

## Live Web UI

The FastAPI backend and vanilla JS frontend live under `simulation/web/`.

Install the simulation dependencies:

```bash
python3 -m pip install -r simulation/python/requirements.txt
```

Run the web server from the repository root:

```bash
python3 -m uvicorn simulation.web.app:app --host 127.0.0.1 --port 8000
```

Then open:

```text
http://127.0.0.1:8000
```

The UI provides:

- live Gym RGB render streaming
- live binary observation streaming
- start, stop, step, reset, and restart controls
- initial-state and runtime-configuration controls
- ground-truth state, estimated state, and control telemetry

## Notes

- The current local environment is Python 3.9, so backend-facing type annotations should stay compatible with Python 3.9.
- Generated firmware LQR headers are written to `hardware/firmware/include/generated_teacher_lqr_gains.h`.
