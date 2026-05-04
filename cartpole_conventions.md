# Cart-Pole Wiring And Control Conventions

This file is the bench reference for the current ESP32 firmware in this repo.

## Pins

- `GPIO25` -> TB6600 `PUL-`
- `GPIO26` -> TB6600 `DIR-`
- `GPIO33` -> TB6600 `DIR-ALT` (mirrors `DIR-`)
- `GPIO27` -> TB6600 `ENA-`
- `GPIO32` -> left end-stop home switch input
- `GPIO21` -> AS5600 `SDA`
- `GPIO22` -> AS5600 `SCL`

## TB6600 Wiring

- The driver is wired as common-anode.
- `PUL+`, `DIR+`, and `ENA+` go to the positive control supply.
- The ESP32 only sinks current on the `-` pins using open-drain outputs.
- Logic meaning:
  - drained to GND = active
  - released/high = inactive

## Motion Sign Convention

- Positive signed step rate means `clockwise` in firmware.
- With the current mechanical setup, draining `DIR-` to GND moves the cart to the right.
- Right is the far side.
- Left is the switch side / home side.
- Therefore:
  - positive step rate -> right / away from home
  - negative step rate -> left / toward home

## Home Switch Convention

- The home switch is on the left end only.
- The switch is normally closed.
- Firmware assumes `GPIO32` uses the ESP32 internal pull-up.
- Recommended wiring:
  - `GPIO32` -> switch -> GND
- Because the switch is normally closed:
  - away from the stop: input reads LOW because the switch ties to GND
  - at the left stop: the switch opens and the pull-up makes the input read HIGH
- The firmware treats HIGH on `GPIO32` as `home switch active`.

## Homing Convention

- Start with the cart manually placed near the far-right end.
- Send `HOME`.
- The firmware drives left until the switch opens.
- The measured step count from the start point to the switch is stored as total travel.
- Home is the left reference end used only for calibration.
- The cart then reverses and moves to the midpoint.
- The midpoint is the reported cart position `0`.
- The calibrated motion limits are symmetric around center:
  - `limit = travel_steps / 2 - 50`
  - allowed centered cart range is `[-limit, +limit]`

## Position Convention

- `kMillimetersPerStep = 0.0125 mm/step`
- `METERS_PER_STEP = 0.0000125 m/step`
- Offline simulation should assume a `0.30 m` total track length (`+/- 0.15 m` from center).
- `cart_home_steps` means distance from the left home switch end.
- `cart_center_steps` means distance from the calibrated midpoint.
- `STATUS` reports `cart` in centered coordinates.
- Positive centered cart position means right of center.
- Negative centered cart position means left of center.

## AS5600 Convention

- I2C address: `0x36`
- Bus pins: `SDA=GPIO21`, `SCL=GPIO22`
- Bus speed: `100 kHz`
- Firmware enables weak ESP32 internal pull-ups on `SDA` and `SCL` for bring-up, but proper external I2C pull-ups are still preferred.
- The firmware reads the raw 12-bit angle register and unwraps it across turns.
- `RESTANGLE` captures the pendulum-down resting angle as `+180 deg`.
- Upright is `0 deg`.
- Clockwise pendulum rotation is positive and counter-clockwise is negative.
- `ANGLEZERO` captures the current position as upright `0 deg` by default.
- `ANGLEZERO REST` captures the current position as `+180 deg`.
- `ZEROANGLE` is kept as an alias for `ANGLEZERO`.
- `SENSOR` prints direct AS5600 bus diagnostics, including whether the device acknowledged on I2C.
- Teacher control uses relative angle in radians after zeroing.

## Teacher State Vector

- The teacher controller state is:
  - `x[0] = cart position from center (steps)`
  - `x[1] = cart velocity (steps/second)`
  - `x[2] = pole angle from upright zero (radians)`
  - `x[3] = pole angular velocity (radians/second)`
- Teacher control law in firmware:
  - `u = -(Kx)`
  - `u` is a signed step rate in `steps/second`

## Safety Limits

- Motion commands are clamped to `+/- 20000 steps/s`.
- If a requested speed exceeds the max limit, it is clamped to the max magnitude in the same requested direction.
- If the cart reaches a calibrated hard limit during motion, firmware stops the cart and does not automatically reverse direction.
- If a new command would drive farther outward past a hard limit, firmware clamps that command to zero.
- If the home switch is active, firmware blocks any further motion toward home.

## Serial Modes

- Current firmware console baud: `460800`
- Current serial command path is a human-readable bring-up/debug console.
- Later student-policy mode can switch to the high-rate binary float command path at `921600`.
- Use exactly one serial client at a time.
- Do not run the Python helper and the PlatformIO/VS Code serial monitor simultaneously.
- On macOS, `/dev/cu.*` and `/dev/tty.*` are separate device nodes to the same USB-UART bridge; opening both can split or truncate the same output stream.
- If a second client attaches mid-line, it may show only the tail of a status line such as the end of the `K=[...]` vector.

## Current Commands

- `STATUS`
- `HOME`
- `CENTER`
- `MOVE <signed_steps_per_sec>`
- `SPEED <signed_steps_per_sec>`
- `STOP`
- `RESTANGLE`
- `ANGLEZERO [UPRIGHT|REST]`
- `ZEROANGLE [UPRIGHT|REST]` (alias for `ANGLEZERO`)
- `SENSOR`
- `SETK <cart_steps> <cart_steps_s> <angle_rad> <angle_rate_radps>`
- `GAINS`
- `TEACHER ON`
- `TEACHER OFF`
- `LOG ON`
- `LOG OFF`

## Telemetry Format

- `LOG ON` enables periodic CSV-like lines at `200 Hz`.
- The first emitted header line is:

```text
DATA_HEADER,timestamp_us,cart_home_steps,cart_center_steps,cart_vel_steps_s,angle_raw_counts,angle_deg,angle_rad,angle_vel_rad_s,command_steps_s,mode,homed,angle_zeroed
```

- Each data sample line starts with `DATA,`.
- `timestamp_us` comes from the ESP32 monotonic microsecond timer.
- The camera-side logger can later align frames against these timestamps.

## Bring-Up Order

1. Wire TB6600 and verify manual `MOVE` commands work.
2. Wire the left normally-closed home switch to `GPIO32` and confirm `STATUS` changes when the switch opens.
3. Wire AS5600 to `GPIO21`/`GPIO22` and confirm `STATUS` shows a live sample.
4. Place the cart at the far right and run `HOME`.
5. Let the pole hang at rest and run `RESTANGLE`.
6. Set the teacher LQR gains with `SETK ...`.
7. Optionally enable `LOG ON` for timestamped data.
8. Enable the teacher controller with `TEACHER ON`.