#!/usr/bin/env python3
"""
esp32_serial_schedule.py - Stream ESP32 output and send a predefined command schedule.

Usage:
    python esp32_serial_schedule.py              # auto-detect ESP32 port
    python esp32_serial_schedule.py /dev/cu.xxx  # specify port explicitly

Edit COMMAND_SEQUENCE below to change which motor commands are sent and when.
If the final schedule leaves telemetry enabled with `LOG ON`, the script keeps
streaming the ESP32 output until Ctrl+C so log lines are not cut off.
"""

import sys
import time
import threading
from typing import Optional, Sequence, Tuple

import serial
import serial.tools.list_ports

# Serial timing and the default scripted command sequence used for repeatable bench bring-up.
BAUD_RATE = 460800
POLL_INTERVAL_S = 0.01
POST_SEQUENCE_LISTEN_S = 1.0
BOOT_SETTLE_S = 1.5

CommandStep = Tuple[float, str]

COMMAND_SEQUENCE: Sequence[CommandStep] = (
    (0.25, "STATUS"),
    (0.50, "100"),
    (2.00, "-100"),
    (2.00, "0"),
    (0.50, "STATUS"),
)

_ESP32_VID_PIDS = {
    (0x1A86, 0x7523),  # CH340
    (0x10C4, 0xEA60),  # CP2102 / CP2104
    (0x0403, 0x6001),  # FT232R
    (0x0403, 0x6010),  # FT2232H
    (0x0403, 0x6014),  # FT232H
}


# Port discovery prefers likely ESP32 USB bridges and favors /dev/cu.* on macOS.
def find_esp32_port() -> Optional[str]:
    """Return the first serial port that looks like an ESP32 USB bridge."""
    candidates = []

    for port in serial.tools.list_ports.comports():
        if (port.vid, port.pid) in _ESP32_VID_PIDS:
            candidates.append(port.device)
            continue
        desc = (port.description or "").lower()
        if any(kw in desc for kw in ("ch340", "cp210", "uart", "usb serial", "ft232")):
            candidates.append(port.device)

    if not candidates:
        return None

    cu_candidates = [device for device in candidates if "/dev/cu." in device]
    if cu_candidates:
        return sorted(cu_candidates)[0]

    return sorted(candidates)[0]


def open_serial_port(port: str, baud_rate: int) -> serial.Serial:
    """Open the serial port without pulsing DTR/RTS during initial open."""
    ser = serial.Serial()
    ser.port = port
    ser.baudrate = baud_rate
    ser.timeout = 0
    ser.write_timeout = 0
    ser.xonxoff = False
    ser.rtscts = False
    ser.dsrdtr = False
    ser.dtr = False
    ser.rts = False
    ser.open()
    return ser


def normalized_command(command: str) -> str:
    """Normalize a console command for simple schedule-state checks."""
    return command.strip().upper()


def sequence_leaves_logging_enabled(command_sequence: Sequence[CommandStep]) -> bool:
    """Return true when the scripted sequence ends with telemetry still enabled."""
    logging_enabled = False
    for _delay_s, command in command_sequence:
        normalized = normalized_command(command)
        if normalized == "LOG ON":
            logging_enabled = True
        elif normalized == "LOG OFF":
            logging_enabled = False
    return logging_enabled


# RX rebuilds complete console lines from arbitrary serial chunks.
def rx_thread(ser: serial.Serial, stop_event: threading.Event) -> None:
    """Read lines from the ESP32 and print them without blocking forever."""
    pending = bytearray()

    while not stop_event.is_set():
        try:
            raw = ser.read(ser.in_waiting or 1)
        except serial.SerialException:
            break

        if not raw:
            stop_event.wait(POLL_INTERVAL_S)
            continue

        pending.extend(raw)
        while True:
            line_end = -1
            for separator in (b"\n", b"\r"):
                separator_index = pending.find(separator)
                if separator_index != -1 and ((line_end == -1) or (separator_index < line_end)):
                    line_end = separator_index

            if line_end == -1:
                break

            raw_line = pending[:line_end]
            pending = pending[line_end + 1 :]
            while pending.startswith((b"\n", b"\r")):
                pending = pending[1:]

            line = raw_line.decode("utf-8", errors="ignore").strip()
            if line:
                print(line, flush=True)


# TX replays a fixed schedule so each run sends the same command sequence.
def tx_thread(
    ser: serial.Serial,
    stop_event: threading.Event,
    command_sequence: Sequence[CommandStep],
) -> None:
    """Send a predefined schedule of commands without using stdin."""
    next_send_time = time.monotonic()

    for delay_s, command in command_sequence:
        next_send_time += max(delay_s, 0.0)

        while not stop_event.is_set():
            remaining = next_send_time - time.monotonic()
            if remaining <= 0.0:
                break
            stop_event.wait(min(POLL_INTERVAL_S, remaining))

        if stop_event.is_set():
            return

        normalized = normalized_command(command)

        try:
            ser.write((normalized + "\n").encode("utf-8"))
        except serial.SerialException:
            stop_event.set()
            return

        print(f">>> {normalized}", flush=True)

    if sequence_leaves_logging_enabled(command_sequence):
        print(
            "Telemetry logging left ON by the schedule; streaming ESP32 output until Ctrl+C.",
            flush=True,
        )
        while not stop_event.wait(POLL_INTERVAL_S):
            pass
        return

    stop_event.wait(POST_SEQUENCE_LISTEN_S)
    stop_event.set()


# Main wires together discovery, reset-safe port open, and the RX/TX worker lifecycle.
def main() -> None:
    if len(sys.argv) > 1:
        port = sys.argv[1]
    else:
        port = find_esp32_port()
        if not port:
            print(
                "ERROR: Could not auto-detect an ESP32 serial port.\n"
                "Usage: python esp32_serial_schedule.py [PORT]",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Auto-detected port: {port}", flush=True)

    print(f"Connecting to {port} at {BAUD_RATE} baud ...", flush=True)
    print("Programmatic command schedule:", flush=True)
    for delay_s, command in COMMAND_SEQUENCE:
        print(f"  +{delay_s:0.2f}s -> {command}", flush=True)
    print(f"Waiting {BOOT_SETTLE_S:0.1f}s for ESP32 boot/reset noise to clear ...", flush=True)

    try:
        with open_serial_port(port, BAUD_RATE) as ser:
            time.sleep(BOOT_SETTLE_S)
            ser.reset_input_buffer()
            ser.reset_output_buffer()

            stop_event = threading.Event()
            t_rx = threading.Thread(target=rx_thread, args=(ser, stop_event), daemon=True)
            t_tx = threading.Thread(
                target=tx_thread,
                args=(ser, stop_event, COMMAND_SEQUENCE),
                daemon=True,
            )
            t_rx.start()
            t_tx.start()

            try:
                while t_rx.is_alive() and not stop_event.wait(POLL_INTERVAL_S):
                    pass
            except KeyboardInterrupt:
                stop_event.set()
            finally:
                stop_event.set()
                t_tx.join(timeout=1.0)
                t_rx.join(timeout=1.0)

    except KeyboardInterrupt:
        pass
    except serial.SerialException as exc:
        print(f"Serial error: {exc}", file=sys.stderr)
        sys.exit(1)

    print("\nStopped.", flush=True)


if __name__ == "__main__":
    main()

