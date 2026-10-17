from __future__ import annotations

import csv
import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass
class Frame:
    t: float
    rssi: int
    amp: np.ndarray


def parse_line(line: str, t: float) -> Frame | None:
    if not line.startswith("CSI_DATA"):
        return None
    try:
        fields = next(csv.reader([line.strip()]))
        rssi = int(fields[3])
        first_word_invalid = int(fields[-2])
        raw = np.array([int(v) for v in fields[-1].strip("[]").split(",") if v.strip()],
                       dtype=np.float32)
    except (ValueError, IndexError, StopIteration):
        return None
    if first_word_invalid:
        raw[:4] = 0
    n = raw.size // 2
    imag, real = raw[0:2 * n:2], raw[1:2 * n:2]
    return Frame(t=t, rssi=rssi, amp=np.hypot(imag, real))


def read_log(path: str | Path) -> Iterator[tuple[float, str]]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        for row in f:
            t, sep, line = row.rstrip("\n").partition("\t")
            if sep:
                yield float(t), line
