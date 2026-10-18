from __future__ import annotations

from collections import Counter, deque

import numpy as np

from backend.capture.csi import Frame


class Meter:
    def __init__(self, name: str = "", window_s: float = 0.5, every_s: float = 1.0):
        self.name = name
        self.window_s = window_s
        self.every_s = every_s
        self.recent: deque[Frame] = deque()
        self.history: deque[float] = deque(maxlen=600)
        self.lengths: Counter[int] = Counter()
        self.count = 0
        self.last_print: float | None = None

    def add(self, frame: Frame) -> None:
        if self.last_print is None:
            self.last_print = frame.t
        self.count += 1
        self.lengths[frame.amp.size] += 1
        self.recent.append(frame)
        while frame.t - self.recent[0].t > self.window_s:
            self.recent.popleft()

        elapsed = frame.t - self.last_print
        if elapsed >= self.every_s:
            self._print(self.count / elapsed, frame.rssi)
            self.count = 0
            self.last_print = frame.t

    def _motion(self) -> tuple[float | None, int]:
        size = self.lengths.most_common(1)[0][0]
        same = [f.amp for f in self.recent if f.amp.size == size]
        if len(same) < 10:
            return None, 0
        amps = np.stack(same)
        mean = amps.mean(axis=0)
        live = mean > 0
        if not live.any():
            return None, 0
        return float(amps[:, live].std(axis=0).mean() / mean[live].mean()), int(live.sum())

    def _print(self, rate: float, rssi: int) -> None:
        motion, subcarriers = self._motion()
        prefix = f"{self.name} " if self.name else ""
        head = f"{prefix}{rate:6.1f} Hz | {subcarriers:3d} subcarriers | rssi {rssi:4d} dBm | motion"
        if motion is None:
            print(f"{head}    -- (too few packets)", flush=True)
            return
        self.history.append(motion)
        calm = max(float(np.percentile(self.history, 10)), 1e-6)
        ratio = motion / calm
        bar = "█" * min(40, int(ratio * 4))
        print(f"{head} {ratio:5.1f}x {bar}", flush=True)
