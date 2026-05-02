#!/usr/bin/env python3
"""
read_angle.py – Stream AS5048A angle data from the ESP32 and send stepper commands.

Usage:
    python read_angle.py              # auto-detect ESP32 port
    python read_angle.py /dev/cu.xxx  # specify port explicitly

Angle output (CSV, stdout):
    timestamp_us,angle_degrees

Stepper commands (type in the terminal, press Enter):
    MOVE <steps>        relative move; positive = CW, negative = CCW
    STOP                halt immediately

Press Ctrl-C to stop.
"""

import sys
import threading
from typing import Optional
import serial
import serial.tools.list_ports

BAUD_RATE = 921600

_ESP32_VID_PIDS = {
    (0x1A86, 0x7523),  # CH340
    (0x10C4, 0xEA60),  # CP2102 / CP2104
    (0x0403, 0x6001),  # FT232R
    (0x0403, 0x6010),  # FT2232H
    (0x0403, 0x6014),  # FT232H
}


def find_esp32_port() -> Optional[str]:
    """Return the first serial port that looks like an ESP32 USB bridge."""
    for port in serial.tools.list_ports.comports():
        if (port.vid, port.pid) in _ESP32_VID_PIDS:
            return port.device
        desc = (port.description or "").lower()
        if any(kw in desc for kw in ("ch340", "cp210", "uart", "usb serial", "ft232")):
            return port.device
    return None


def rx_thread(ser: serial.Serial, stop_event: threading.Event) -> None:
    """Read lines from the ESP32 and print them."""
    while not stop_event.is_set():
        try:
            raw = ser.readline()
        except serial.SerialException:
            break
        if not raw:
            continue
        try:
            line = raw.decode("utf-8").strip()
        except UnicodeDecodeError:
            continue
        if line:
            print(line, flush=True)
            # continue


def tx_thread(ser: serial.Serial, stop_event: threading.Event) -> None:
    """Read commands from stdin and send them to the ESP32."""
    valid_prefixes = ("MOVE ", "STOP")
    while not stop_event.is_set():
        try:
            cmd = input()
        except EOFError:
            break
        cmd = cmd.strip().upper()
        if not cmd:
            continue
        if any(cmd.startswith(p) for p in valid_prefixes):
            try:
                ser.write((cmd + "\n").encode("utf-8"))
            except serial.SerialException:
                break
        else:
            print(
                f"Unknown command: {cmd!r}\n"
                "  MOVE <steps> | STOP",
                file=sys.stderr,
                flush=True,
            )


def main() -> None:
    if len(sys.argv) > 1:
        port = sys.argv[1]
    else:
        port = find_esp32_port()
        if not port:
            print(
                "ERROR: Could not auto-detect an ESP32 serial port.\n"
                "Usage: python read_angle.py [PORT]",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Auto-detected port: {port}", flush=True)

    print(f"Connecting to {port} at {BAUD_RATE} baud ...", flush=True)
    print("Commands: MOVE <steps>  |  STOP", flush=True)
    print("timestamp_us,angle_degrees", flush=True)

    try:
        with serial.Serial(port, BAUD_RATE, timeout=0.1) as ser:
            ser.reset_input_buffer()

            stop_event = threading.Event()
            t_rx = threading.Thread(target=rx_thread, args=(ser, stop_event), daemon=True)
            t_tx = threading.Thread(target=tx_thread, args=(ser, stop_event), daemon=True)
            t_rx.start()
            t_tx.start()

            try:
                t_rx.join()
            except KeyboardInterrupt:
                pass
            finally:
                stop_event.set()

    except KeyboardInterrupt:
        pass
    except serial.SerialException as exc:
        print(f"Serial error: {exc}", file=sys.stderr)
        sys.exit(1)

    print("\nStopped.", flush=True)


if __name__ == "__main__":
    main()

