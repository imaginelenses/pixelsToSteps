# Cart-Pole Pixel-to-Steps Control System — Agent Handoff

## Current Simulation Status (2026-05-04)

The currently working path is the nominal Gym CartPole simulation workflow in native cartpole units, not the older hardware step-domain path described below.

Working configuration:

- teacher and rollout plant: Gym CartPole in `meters`, `meters/s`, `radians`, `radians/s`, and `newtons`
- control rate: `60 Hz`
- frame preprocessing: binary pole mask, vertically cropped to the pole-plus-track region, resized back to `125 x 160`, then dilated
- training collection: `80` demos from one shared collection run at reduced deterministic mismatch
- true rollout mismatch relative to nominal: `masspole = 1.1x`, `half_pole_length = 0.9x`
- observer initialization during image-only rollout: `x_hat_0` is seeded from the true initial state

Working observer:

- the residual LASSO path is not the active rollout path
- the active observer is a hybrid observer that keeps nominal dynamics for prediction and injects a direct Ridge pixel estimate only into `theta_hat`
- observer recurrence:

```python
x_hat_pred = A_L @ x_hat + B_L * u_cmd + d
theta_pixel = theta_pixel_coefficients @ y_pixels + theta_pixel_bias

x_hat[0] = x_hat_pred[0]
x_hat[1] = x_hat_pred[1]
x_hat[2] = (1.0 - theta_pixel_blend_weight) * x_hat_pred[2] + theta_pixel_blend_weight * theta_pixel
x_hat[3] = x_hat_pred[3]
```

Validated training and rollout artifacts:

- reduced-mismatch 80-demo collection: `python/captures/collections/nominal_cartpole_collection_20260504_194403`
- notebook export: `python/hybrid_pixels_to_cartpole_observer_theta_blend_0p7.json`
- direct pixel-to-angle diagnostic in `python/student_policy.ipynb`:
    - `R^2 = 0.980228`
    - angle RMSE `= 0.180844 deg`
    - bootstrap frame: true `+12.000 deg`, predicted `+11.983488 deg`
- validated image-only rollout capture: `python/captures/cartpole_nominal_20260504_195258_frames`

Validated rollout command:

```bash
python3 python/teacher_policy.py \
    --steps 120 \
    --theta0-deg 12 \
    --true-masspole-scale 1.1 \
    --true-half-pole-length-scale 0.9 \
    --observer-json python/hybrid_pixels_to_cartpole_observer_theta_blend_0p7.json \
    --skip-video-assembly
```

Validated result:

- image-only rollout survived `120/120` steps
- capture ended by horizon, not failure
- first-step angle estimate was already close to truth: true `+12.000 deg`, estimated `+11.988 deg`, direct pixel angle `+11.983 deg`

If continuing the simulation path, use the nominal-cartpole files under `python/` and treat the hardware-oriented sections below as historical context rather than the current source of truth.

## Project Overview

This is a physical cart-pole balancing system replicating the paper **"From Pixels to Torques with Linear Feedback"** (Lee et al., CMU, arXiv:2406.18699v3). The system uses a pixel-based Luenberger observer (linear output-feedback policy) to balance an inverted pendulum using only camera feedback — no angle sensor during student policy execution. The control output is a stepper motor step rate, replacing the paper's brushless motor torque output.

A **teacher LQR policy** (encoder-based, runs on ESP32) collects demonstration data. A **student policy** (camera-based Luenberger observer, runs on Mac Mini M4) is learned offline via linear least-squares regression over the teacher trajectories, then deployed for inference.

---

## Hardware Architecture

### Compute
- **Mac Mini M4** — camera host, student policy inference, serial command TX
- **Elegoo ESP32-D0WD-V3** (NodeMCU, dual-core, 240 MHz) — stepper driver, LQR teacher, encoder reading

### Camera
- **VGA global shutter USB monochrome camera** (UVC, USB2)
- Target operating resolution: **QQVGA 160×120** at **200–400 fps**
- Already monochrome — Gaussian blur + Otsu thresholding to convert the grayscale image to a binary image
- Connects to Mac Mini M4 via USB

### Actuator
- **Ender 3 V1 NEMA 17 stepper motor** with GT2 belt + 20-tooth pulley
- **TB6600 stepper driver**, 1/16 microstepping
- mm per step: **0.0125 mm/step**
- Pulse generation via **ESP32 RMT or LEDC PWM peripheral** (hardware, not software-timed)
- **Stepper motor control is already implemented on the ESP32**

### Pole
- Length: **13 cm**, mass: **9.55 g**
- Tip mass: **5-cent coin (~5.0 g)** attached at top
- White pole, black background (high contrast for thresholding)
- Pivot: rotary encoder on ESP32 (used by teacher LQR only)

### Track
- Total usable length: **15 cm** (±7.5 cm from center)
- Designed for balancing only — no swing-up required

### Communication
- **Mac → ESP32**: USB Serial (CP2102 bridge), **921,600 baud** (CP2102 maximum)
- Protocol: **one-way only** — Mac sends a 4-byte float (step rate command) per frame
- No round-trip — student policy is strictly Mac → ESP32

---

## System Dynamics & Constraints

### Pole Dynamics (with tip mass)
- Natural frequency: **9.49 rad/s**
- Fall time (linearised): **~331 ms**
- Required minimum control cycles before fall: **≥ 30** (target ≥ 60)

### Latency Budget (must be satisfied each frame)
| Stage | Budget |
|---|---|
| Camera frame period | 5.0 ms @ 200 fps |
| USB UVC frame delivery | ≤ 1.0 ms |
| Image processing (blur + threshold) | ≤ 0.5 ms |
| Observer matrix multiply | ≤ 1.0 ms |
| Serial TX (4-byte float) | ≤ 0.1 ms |
| ESP32 RMT update | ≤ 0.05 ms |
| **Total end-to-end** | **≤ 7 ms** |

At 200 fps this gives **66 correction cycles** — the minimum viable operating point. **Do not operate at 60 fps** (only 20 cycles — insufficient).

### Stepper Velocity Constraint
- Keep commanded cart velocity **< 250 mm/s** (NEMA 17 torque knee)
- At 1/16 microstepping: 250 mm/s = **20,000 steps/s** — well within TB6600's 200 kHz limit
- **Clamp all step rate commands at ±20,000 steps/s**

### Track Boundary Constraint
- Cart must stay within **±7.5 cm** of center
- Implement a software limit: if encoder reports cart within 1 cm of end, override student command with a hard centering correction
- Failure to implement this will physically damage the hardware

---

## Image Processing Pipeline (Mac, per frame)

```python
import cv2, numpy as np, serial, struct

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)
cap.set(cv2.CAP_PROP_FPS, 200)

ser = serial.Serial('/dev/tty.SLAB_USBtoUART', baudrate=921600)

while True:
    cap.grab()                              # flush buffer, get latest frame
    ret, frame = cap.retrieve()             # already grayscale (mono camera)
    blur = cv2.GaussianBlur(frame, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    y_k = binary.flatten().astype(np.float32)   # 160×120 = 19,200 elements

    # Observer update: x_{k+1} = A_L x_k + B_L u_k + L y_{k+1} + d
    x_k1 = A_L @ x_k + B_L * u_k + L_gain @ y_k + d
    u_k1 = float(K @ x_k1)                 # LQR gain → step rate (steps/s)
    u_k1 = np.clip(u_k1, -20000, 20000)    # clamp to stepper safe range

    ser.write(struct.pack('f', u_k1))       # 4 bytes, ~44 µs TX time
    x_k = x_k1
    u_k = u_k1
```

**Do not** use `cap.read()` — it blocks and returns buffered frames, adding latency. Always use `cap.grab()` + `cap.retrieve()`.

---

## Observer & Policy Architecture

### State Vector
The system state `x_k` has 4 elements: `[cart_position, cart_velocity, pole_angle, pole_angular_velocity]`.

### Teacher Policy (ESP32, data collection phase)
- LQR controller: `u_k = -K @ x_k` where `x_k` is read from rotary encoder
- Output `u_k` is a **step rate (steps/s)** — this is the input space, not force/torque
- Runs on Core 1 at **500–1000 Hz**
- Logs `(x_k, u_k)` pairs at camera frame rate (200 fps) for LLS training

### Student Policy (Mac M4, deployment phase)
- Pixel-based Luenberger observer (equation 17 from paper):
  \[ x_{k+1} = A_L x_k + B_L u_k + L y_{k+1} + d \]
  where `y_{k+1}` is the **flattened binary image** (19,200-element vector)
- Parameters `(A_L, B_L, L, d)` learned offline via LLS regression (equation 21 from paper)
- LQR gain `K` cloned from teacher (same matrix)
- Output `u_k` is **step rate (steps/s)** — same units as teacher

### LLS Regression (offline, on separate machine)
Solve:
\[ \min_{\Theta} \| \Theta W - X_{2:N} \|_2^2 + \lambda \| \Theta \|_1 \]

where `W` stacks `[x_{1:N-1}, u_{1:N-1}, y_{2:N}, 1]` and `X_{2:N}` are the next states.
Optionally enforce stability via LMI constraint (equation 24 from paper).
20 stabilizing demonstrations × 150 samples each is sufficient per paper results.

---

## ESP32 Firmware Requirements

### Core Assignment
- **Core 0**: Serial RX task — receives 4-byte float step rate from Mac, immediately updates RMT/LEDC frequency and DIR pin
- **Core 1**: LQR teacher task — reads encoder, computes `u = -K @ x`, drives stepper (data collection phase only)

### Serial RX Task (Core 0)
```cpp
void serialRxTask(void *pvParams) {
    float step_rate;
    while (true) {
        if (Serial.available() >= 4) {
            Serial.readBytes((char*)&step_rate, sizeof(float));
            step_rate = constrain(step_rate, -20000.0f, 20000.0f);
            // Update RMT frequency + DIR pin
            setStepperVelocity(step_rate);
        }
        vTaskDelay(1);
    }
}
```

### Hardware Pulse Generation
- Use **RMT peripheral** (preferred) or **LEDC PWM** for step pulses — not `delayMicroseconds()`-based software timing
- DIR pin must be updated **before** the first step pulse after a direction change
- Minimum DIR setup time for TB6600: **5 µs**

### Track Limit Safety
- Track soft limits: ±6.5 cm (1 cm before hardware end-stop)
- If cart position (from encoder or step counter) exceeds soft limit, override step rate with centering command regardless of serial input

---

## Training Data Collection Protocol

1. Start with **teacher LQR** active on ESP32 (Core 1)
2. Mac records synchronized `(x_k, u_k, y_k)` triplets at 200 fps
3. Manually balance pole near vertical and release — record 150-sample trajectory (~0.75 s at 200 fps)
4. Collect **20 such trajectories** (per paper: sufficient for 97%+ success rate)
5. Run offline LLS regression to obtain `(A_L, B_L, L, d)`
6. Deploy student policy on Mac — switch ESP32 Core 0 serial RX to active mode

---

## Key Constraints Summary

| Constraint | Value | Reason |
|---|---|---|
| Min camera FPS | 120 fps | <30 correction cycles below this |
| Target camera FPS | 200–400 fps | 66–132 correction cycles |
| Camera resolution | QQVGA 160×120 | Matches paper; low latency |
| Serial baud rate | 921,600 | CP2102 bridge maximum |
| Max step rate | ±20,000 steps/s | NEMA 17 torque knee at 250 mm/s |
| Microstepping | 1/16 (fixed) | Must not change after data collection |
| Max end-to-end latency | 7 ms | 2% of 331 ms fall time |
| Track soft limit | ±6.5 cm | 15 cm total track |
| Control direction | Mac → ESP32 only | No round-trip in student policy |
| Image input | Binary (Otsu) 19,200 floats | Matches paper preprocessing |
| Teacher output units | steps/s | Must match student output units |

---

## What Is Already Implemented

- Stepper motor control on the ESP32 (step pulse generation, direction control)

## What Needs to Be Written

1. **Mac Python inference loop** — camera capture, image processing, observer update, serial TX
2. **ESP32 serial RX task** (Core 0) — receive float, clamp, update RMT velocity
3. **ESP32 LQR teacher task** (Core 1) — encoder read, LQR compute, stepper drive + data logging
4. **Mac Python data collection logger** — synchronized recording of `(x_k, u_k, y_k)` triplets
5. **Offline LLS regression script** — solve for `(A_L, B_L, L, d)` with optional LMI stability constraint

---

## Reference

Paper: Lee, J.H., Schoedel, S., Bhardwaj, A., Manchester, Z. "From Pixels to Torques with Linear Feedback." arXiv:2406.18699v3. CMU Robotics Institute, 2024.
Open-source code: https://roboticexplorationlab.org/projects/linearpixelstotorques.html
