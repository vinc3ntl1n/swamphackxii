from __future__ import annotations

import csv
import gzip
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

DEFAULT_LINK = "A"


@dataclass
class Frame:
    t: float
    rssi: int
    amp: np.ndarray
    link: str = DEFAULT_LINK
    source_id: int | None = None
    source_timestamp_us: int | None = None
    received_mono: float | None = None
    source_lag_s: float | None = None

    def capture_age_s(self, *, now_mono: float | None = None,
                      now_wall: float | None = None) -> float:
        """Host receipt age; monotonic metadata is present only for live frames."""
        if self.received_mono is not None:
            now = time.monotonic() if now_mono is None else now_mono
            return max(0.0, now - self.received_mono)
        now = time.time() if now_wall is None else now_wall
        return max(0.0, now - self.t)

    def relative_age_s(self, *, now_mono: float | None = None,
                       now_wall: float | None = None) -> float:
        """Host receipt age plus observed relative source lag, not absolute age."""
        return self.capture_age_s(now_mono=now_mono, now_wall=now_wall) + max(
            0.0, self.source_lag_s or 0.0)


def _optional_u32(value: str) -> int | None:
    """Accept unsigned counters and their signed 32-bit CSV representation."""
    try:
        number = int(value)
    except (ValueError, OverflowError):
        return None
    return number & 0xFFFFFFFF if -(1 << 31) <= number <= 0xFFFFFFFF else None


def parse_line(line: str, t: float, link: str = DEFAULT_LINK) -> Frame | None:
    if not line.startswith("CSI_DATA") or not math.isfinite(t):
        return None
    try:
        fields = next(csv.reader([line.strip()], strict=True))
        if len(fields) < 6 or fields[0] != "CSI_DATA":
            return None
        rssi = int(fields[3])
        first_word_invalid = int(fields[-2])
        values = fields[-1].strip()
        if first_word_invalid not in (0, 1) or not (values.startswith("[") and values.endswith("]")):
            return None
        raw = np.array([int(v) for v in values[1:-1].split(",")],
                       dtype=np.float32)
    except (ValueError, OverflowError, IndexError, StopIteration, csv.Error):
        return None
    if raw.size == 0 or raw.size % 2 or not np.isfinite(raw).all():
        return None
    if first_word_invalid:
        raw[:4] = 0
    n = raw.size // 2
    imag, real = raw[0:2 * n:2], raw[1:2 * n:2]
    amp = np.hypot(imag, real)
    if not np.isfinite(amp).all():
        return None
    # These positions are verified for the ESP32-S3 25-column export. Legacy
    # short CSVs remain valid; unrecognized metadata must not alter amplitude.
    source_id = _optional_u32(fields[1]) if len(fields) == 25 else None
    source_timestamp_us = _optional_u32(fields[18]) if len(fields) == 25 else None
    return Frame(t=t, rssi=rssi, amp=amp, link=link,
                 source_id=source_id, source_timestamp_us=source_timestamp_us)


def log_row(t: float, link: str, line: str) -> str:
    return f"{t:.6f}\t{link}\t{line}\n"


def read_log(path: str | Path) -> Iterator[tuple[float, str, str]]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        for row in f:
            t, sep, rest = row.rstrip("\n").partition("\t")
            if not sep:
                continue
            link, sep, line = rest.partition("\t")
            if not sep:
                link, line = DEFAULT_LINK, rest
            yield float(t), link, line
