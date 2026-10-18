from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Iterator

import serial
from serial.tools import list_ports

from backend import config
from backend.capture.csi import Frame, log_row, parse_line
from backend.capture.meter import Meter

SESSIONS = config.DATA_DIR / "sessions"
ESPRESSIF_USB_VID = 0x303A


def find_receivers() -> dict[str, str]:
    boards = {(p.serial_number or "").upper(): p.device
              for p in list_ports.comports() if p.vid == ESPRESSIF_USB_VID}
    boards.pop(config.SENDER_BOARD.upper(), None)
    known = {link: (serial or "").upper() for link, serial in config.RECEIVERS.items()}
    ports = {link: boards.pop(serial) for link, serial in known.items() if serial in boards}
    unset = [link for link, serial in known.items() if not serial and link not in ports]
    if len(unset) == 1 and len(boards) == 1:
        ports[unset[0]] = boards.popitem()[1]
    if not ports:
        found = ", ".join(f"{port} ({serial})" for serial, port in boards.items()) or "none"
        sys.exit(f"No receiver found. ESP32 boards: {found}. "
                 "Set config.RECEIVERS to their serial numbers, or pass --port / --port-b.")
    missing = [link for link in config.RECEIVERS if link not in ports]
    if missing:
        print(f"Receiver {', '.join(missing)} not plugged in; using {', '.join(sorted(ports))} only.",
              flush=True)
    return dict(sorted(ports.items()))


def receivers(port: str | None, port_b: str | None) -> dict[str, str]:
    if port or port_b:
        return {link: p for link, p in (("A", port), ("B", port_b)) if p}
    return find_receivers()


def _read_port(link: str, port: str, out: queue.Queue, stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            with serial.Serial(port, 921600, timeout=1) as ser:
                while not stop.is_set():
                    raw = ser.readline()
                    if not raw:
                        continue
                    t = time.time()
                    line = raw.decode("ascii", errors="ignore").strip()
                    if parse_line(line, t) is not None:
                        out.put((t, link, line))
        except (serial.SerialException, OSError):
            if stop.is_set():
                return
            print(f"Lost receiver {link} ({port}); plug it back in. Retrying...", flush=True)
            stop.wait(1.0)


def live_lines(ports: dict[str, str], log_path: Path | None = None,
               stop: threading.Event | None = None) -> Iterator[tuple[float, str, str]]:
    stop = stop if stop is not None else threading.Event()
    out: queue.Queue = queue.Queue()
    threads = []
    for link, port in ports.items():
        thread = threading.Thread(target=_read_port, args=(link, port, out, stop), name=f"receiver-{link}",
                                  daemon=True)
        thread.start()
        threads.append(thread)
    log = open(log_path, "a") if log_path else None
    last = 0.0
    try:
        while not stop.is_set():
            try:
                t, link, line = out.get(timeout=1.0)
            except queue.Empty:
                continue
            t = last = max(t, last)
            if log:
                log.write(log_row(t, link, line))
            yield t, link, line
    finally:
        stop.set()
        for thread in threads:
            thread.join(2)
        if log:
            log.close()


def live_frames(ports: dict[str, str], log_path: Path | None = None) -> Iterator[Frame]:
    for t, link, line in live_lines(ports, log_path):
        yield parse_line(line, t, link)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", help="serial port of receiver A")
    ap.add_argument("--port-b", help="serial port of receiver B")
    ap.add_argument("--no-save", action="store_true", help="don't write a session log")
    args = ap.parse_args()

    ports = receivers(args.port, args.port_b)
    log_path = None
    if not args.no_save:
        SESSIONS.mkdir(parents=True, exist_ok=True)
        log_path = SESSIONS / f"{datetime.now():%Y%m%d-%H%M%S}.log"
        print(f"Saving to {log_path}")
    print(f"Reading {', '.join(f'{link} {port}' for link, port in ports.items())}. "
          "Walk across each link's line; only that link's motion bar should jump.")

    meters = {link: Meter(link) for link in ports}
    try:
        for frame in live_frames(ports, log_path):
            meters[frame.link].add(frame)
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
