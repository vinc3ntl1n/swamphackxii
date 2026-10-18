from __future__ import annotations

import argparse
import json
import logging
import queue
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator, TextIO

import serial
from serial.tools import list_ports

from backend import config
from backend.capture.csi import Frame, log_row, parse_line
from backend.capture.meter import Meter

SESSIONS = config.DATA_DIR / "sessions"
ESPRESSIF_USB_VID = 0x303A
# Detection should follow the room now, rather than work through old movement.
# Recording via live_lines remains lossless and does not use these limits.
LIVE_QUEUE_FRAMES = 256
LIVE_MAX_AGE_S = 0.5
HEALTH_INTERVAL_S = 1.0
LOGGER = logging.getLogger(__name__)


@dataclass
class _Packet:
    frame: Frame
    line: str
    received_mono: float


class _PacketBuffer:
    def __init__(self, max_frames: int, max_age_s: float | None,
                 log: TextIO | None = None,
                 health_callback: Callable[[dict], None] | None = None) -> None:
        self.queue: queue.Queue[_Packet] = queue.Queue(maxsize=max_frames)
        self.max_age_s = max_age_s
        self.log = log
        self.health_callback = health_callback
        self.lock = threading.Lock()
        self.last_t = 0.0
        self.dropped: Counter[str] = Counter()
        self.dropped_links: set[str] = set()
        self.last_drop: dict | None = None
        self.last_report = float("-inf")
        self.reported_drops = 0
        self.last_received_mono: float | None = None
        self.log_error: OSError | None = None

    def _drop(self, packet: _Packet, reason: str, now: float) -> None:
        self.dropped[reason] += 1
        self.dropped_links.add(packet.frame.link)
        self.last_drop = {"reason": reason, "link": packet.frame.link,
                          "capture_t": packet.frame.t,
                          "age_ms": round(max(0.0, now - packet.received_mono) * 1000, 1)}

    def put(self, packet: _Packet) -> bool:
        with self.lock:
            # Log in receive order before applying detection's lossy queue policy.
            # Share this lock across links so rows stay ordered and never interleave.
            packet.frame.t = self.last_t = max(packet.frame.t, self.last_t)
            if self.log:
                try:
                    self.log.write(log_row(packet.frame.t, packet.frame.link, packet.line))
                except OSError as error:
                    self.log_error = error
                    return False
            while True:
                try:
                    self.queue.put_nowait(packet)
                    return True
                except queue.Full:
                    try:
                        old = self.queue.get_nowait()
                    except queue.Empty:  # the consumer freed space in the meantime
                        continue
                    self._drop(old, "queue_overflow", time.monotonic())

    def get(self, timeout: float) -> _Packet | None:
        packet = self.queue.get(timeout=timeout)
        with self.lock:
            now = time.monotonic()
            if self.max_age_s is not None and now - packet.received_mono > self.max_age_s:
                self._drop(packet, "packet_age", now)
                return None
            self.last_received_mono = packet.received_mono
        return packet

    def report(self, *, idle: bool = False) -> None:
        with self.lock:
            now = time.monotonic()
            dropped = sum(self.dropped.values())
            delta = dropped - self.reported_drops
            # Loss must reach the pipeline before its next kept packet. Healthy
            # reports are throttled; drops are batched while expired rows drain.
            if not delta and now - self.last_report < HEALTH_INTERVAL_S:
                return
            health = {"type": "capture_health", "capture_wall_t": time.time(), "idle": idle,
                      "queue_depth": self.queue.qsize(),
                      "queue_capacity": self.queue.maxsize,
                      "capture_age_ms": (round(max(0.0, now - self.last_received_mono) * 1000, 1)
                                         if self.last_received_mono is not None else None),
                      "dropped_frames": dropped, "dropped_since_report": delta,
                      "dropped_links": sorted(self.dropped_links),
                      "drop_reasons": dict(self.dropped), "last_drop": self.last_drop}
            self.last_report = now
            self.reported_drops = dropped
            self.dropped_links.clear()
        LOGGER.log(logging.WARNING if delta else logging.INFO, json.dumps(health))
        if self.health_callback:
            self.health_callback(health)


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


def _read_port(link: str, port: str, out: _PacketBuffer, stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            with serial.Serial(port, 921600, timeout=1) as ser:
                while not stop.is_set():
                    raw = ser.readline()
                    if not raw:
                        continue
                    t = time.time()
                    received_mono = time.monotonic()
                    line = raw.decode("ascii", errors="ignore").strip()
                    frame = parse_line(line, t, link)
                    if frame is not None and not out.put(_Packet(frame, line, received_mono)):
                        stop.set()
                        return
        except (serial.SerialException, OSError):
            if stop.is_set():
                return
            LOGGER.warning(json.dumps({"type": "receiver_disconnected", "link": link,
                                       "port": port, "capture_t": time.time()}))
            print(f"Lost receiver {link} ({port}); plug it back in. Retrying...", flush=True)
            stop.wait(1.0)


def _live_packets(ports: dict[str, str], log_path: Path | None = None,
                  stop: threading.Event | None = None, *, max_queue_frames: int = 0,
                  max_frame_age_s: float | None = None,
                  health_callback: Callable[[dict], None] | None = None) -> Iterator[_Packet]:
    stop = stop if stop is not None else threading.Event()
    log = open(log_path, "a") if log_path else None
    out = _PacketBuffer(max_queue_frames, max_frame_age_s, log, health_callback)
    threads = []
    try:
        for link, port in ports.items():
            thread = threading.Thread(target=_read_port, args=(link, port, out, stop), name=f"receiver-{link}",
                                      daemon=True)
            thread.start()
            threads.append(thread)
        while not stop.is_set():
            try:
                packet = out.get(timeout=0.25)
            except queue.Empty:
                if max_queue_frames or health_callback:
                    out.report(idle=True)
                continue
            if packet is not None:
                if max_queue_frames or health_callback:
                    out.report()
                yield packet
        if out.log_error is not None:
            raise OSError(f"Unable to write raw CSI session log {log_path}") from out.log_error
    finally:
        stop.set()
        for thread in threads:
            thread.join(2)
        # Do not close a log while a serial thread could still be writing to it.
        if log and not any(thread.is_alive() for thread in threads):
            log.close()


def live_lines(ports: dict[str, str], log_path: Path | None = None,
               stop: threading.Event | None = None) -> Iterator[tuple[float, str, str]]:
    """Lossless capture for training; no detection freshness policy is applied."""
    for packet in _live_packets(ports, log_path, stop):
        yield packet.frame.t, packet.frame.link, packet.line


def live_frames(ports: dict[str, str], log_path: Path | None = None, *,
                stop: threading.Event | None = None,
                health_callback: Callable[[dict], None] | None = None) -> Iterator[Frame]:
    """Bounded, fresh frames for detection; the raw session log retains all rows."""
    for packet in _live_packets(ports, log_path, stop, max_queue_frames=LIVE_QUEUE_FRAMES,
                                max_frame_age_s=LIVE_MAX_AGE_S,
                                health_callback=health_callback):
        yield packet.frame


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
