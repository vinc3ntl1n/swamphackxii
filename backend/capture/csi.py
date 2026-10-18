from __future__ import annotations

import csv
import gzip
import math
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
    return Frame(t=t, rssi=rssi, amp=amp, link=link) if np.isfinite(amp).all() else None


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
