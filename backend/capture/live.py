from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterator

import serial
from serial.tools import list_ports

from backend import config
from backend.capture.csi import Frame, parse_line
from backend.capture.meter import Meter

SESSIONS = config.DATA_DIR / "sessions"
ESPRESSIF_USB_VID = 0x303A


def find_port() -> str:
    ports = [p.device for p in list_ports.comports() if p.vid == ESPRESSIF_USB_VID]
    if len(ports) != 1:
        sys.exit(f"Found {len(ports)} ESP32 boards {ports}; pass --port")
    return ports[0]


def live_lines(port: str, log_path: Path | None = None) -> Iterator[tuple[float, str]]:
    log = open(log_path, "a") if log_path else None
    try:
        while True:
            try:
                with serial.Serial(port, 921600, timeout=1) as ser:
                    while True:
                        raw = ser.readline()
                        if not raw:
                            continue
                        t = time.time()
                        line = raw.decode("ascii", errors="ignore").strip()
                        if parse_line(line, t) is None:
                            continue
                        if log:
                            log.write(f"{t:.6f}\t{line}\n")
                        yield t, line
            except (serial.SerialException, OSError):
                print(f"Lost {port}; plug the receiver back in. Retrying...", flush=True)
                time.sleep(1.0)
    finally:
        if log:
            log.close()


def live_frames(port: str, log_path: Path | None = None) -> Iterator[Frame]:
    for t, line in live_lines(port, log_path):
        yield parse_line(line, t)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", help="serial port of the receiver board")
    ap.add_argument("--no-save", action="store_true", help="don't write a session log")
    args = ap.parse_args()

    port = args.port or find_port()
    log_path = None
    if not args.no_save:
        SESSIONS.mkdir(parents=True, exist_ok=True)
        log_path = SESSIONS / f"{datetime.now():%Y%m%d-%H%M%S}.log"
        print(f"Saving to {log_path}")
    print(f"Reading {port}. Walk between the boards; the motion bar should jump.")

    meter = Meter()
    try:
        for frame in live_frames(port, log_path):
            meter.add(frame)
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
