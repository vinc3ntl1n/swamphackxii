from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

import numpy as np

from backend import config
from backend.capture.csi import Frame

CALIBRATING, NO_MOTION, MOVING = "calibrating", "no motion", "moving"
MEASURE = "mean-shift-0.5"
MIN_HALF_FRAMES = 4


@dataclass
class FrameInfo:
    t: float
    level: str
    energy: float | None
    amp: np.ndarray | None


@dataclass
class MotionEvent:
    start: float
    end: float
    t: np.ndarray
    energy: np.ndarray
    amp: np.ndarray
    cut_short: bool

    @property
    def duration(self) -> float:
        return self.end - self.start


def raw_energy(t: np.ndarray, amps: np.ndarray, now: float) -> float | None:
    w = config.WINDOW_S
    recent = amps[t > now - w]
    older = amps[(t > now - 2 * w) & (t <= now - w)]
    if len(recent) < MIN_HALF_FRAMES or len(older) < MIN_HALF_FRAMES:
        return None
    level = float(np.linalg.norm(np.concatenate([recent, older]).mean(axis=0)))
    return float(np.linalg.norm(recent.mean(axis=0) - older.mean(axis=0))) / max(level, 1e-6)


def calibration_from_frames(frames: list[Frame]) -> dict | None:
    if not frames:
        return None
    size = Counter(f.amp.size for f in frames).most_common(1)[0][0]
    kept = [f for f in frames if f.amp.size == size]
    if len(kept) < 50:
        return None
    t = np.array([f.t for f in kept])
    amps = np.stack([f.amp for f in kept])
    mask = amps.mean(axis=0) > 0
    amps = amps[:, mask]
    raws = np.array([r for i in range(len(t)) if (r := raw_energy(t[:i + 1], amps[:i + 1], t[i])) is not None])
    if raws.size < 10:
        return None
    baseline = float(np.median(raws))
    return {"measure": MEASURE, "size": int(size), "mask": np.nonzero(mask)[0].tolist(),
            "baseline": baseline, "noisy_frac": float((raws > config.T_ON * baseline).mean())}


class MotionTracker:
    def __init__(self) -> None:
        self.calibrate()

    def calibrate(self) -> None:
        self.size: int | None = None
        self.mask: np.ndarray | None = None
        self.baseline: float | None = None
        self.cal: dict | None = None
        self.cal_frames: list[Frame] = []
        self.window: deque[tuple[float, np.ndarray]] = deque()
        self.history: deque[tuple[float, float, np.ndarray]] = deque()
        self.level = CALIBRATING
        self.energy = 0.0
        self.event_start: float | None = None
        self.quiet_since: float | None = None

    def calibration(self) -> dict | None:
        return self.cal

    def load_calibration(self, cal: dict) -> None:
        if cal.get("measure") != MEASURE:
            raise ValueError(f"calibration is for motion measure {cal.get('measure')!r}, "
                             f"not {MEASURE!r}: recalibrate")
        self.calibrate()
        self.cal = cal
        self.size = cal["size"]
        self.mask = np.zeros(self.size, dtype=bool)
        self.mask[cal["mask"]] = True
        self.baseline = cal["baseline"]
        self.level = NO_MOTION

    def flush(self) -> MotionEvent | None:
        if self.level != MOVING or not self.history:
            return None
        event = self._cut(self.event_start, self.history[-1][0], cut_short=False)
        self.level, self.event_start, self.quiet_since = NO_MOTION, None, None
        return event

    def update(self, frame: Frame) -> tuple[FrameInfo, MotionEvent | None]:
        if self.baseline is None:
            self._calibrating(frame)
            return FrameInfo(frame.t, self.level, None, None), None
        if frame.amp.size != self.size:
            return FrameInfo(frame.t, self.level, self.energy, None), None

        t, amp = frame.t, frame.amp[self.mask]
        self.window.append((t, amp))
        while t - self.window[0][0] > 2 * config.WINDOW_S:
            self.window.popleft()
        raw = raw_energy(np.array([w[0] for w in self.window]),
                         np.stack([w[1] for w in self.window]), t)
        if raw is not None:
            self.energy = raw / self.baseline

        self.history.append((t, self.energy, amp))
        keep_s = config.PRE_EVENT_S + config.MAX_EVENT_S + config.QUIET_S + 1.0
        while t - self.history[0][0] > keep_s:
            self.history.popleft()

        event = self._step(t, self.energy)
        return FrameInfo(t, self.level, self.energy, amp), event

    def _step(self, t: float, energy: float) -> MotionEvent | None:
        if self.level == NO_MOTION:
            if energy > config.T_ON:
                self.level, self.event_start, self.quiet_since = MOVING, t, None
            return None

        if energy < config.T_OFF:
            if self.quiet_since is None:
                self.quiet_since = t
            if t - self.quiet_since >= config.QUIET_S:
                event = self._cut(self.event_start, self.quiet_since, cut_short=False)
                self.level, self.event_start, self.quiet_since = NO_MOTION, None, None
                return event
            return None

        self.quiet_since = None
        if t - self.event_start >= config.MAX_EVENT_S:
            event = self._cut(self.event_start, t, cut_short=True)
            self.event_start = t
            return event
        return None

    def _cut(self, start: float, end: float, cut_short: bool) -> MotionEvent:
        rows = [h for h in self.history if start - config.PRE_EVENT_S <= h[0] <= end]
        return MotionEvent(
            start=start, end=end, cut_short=cut_short,
            t=np.array([r[0] for r in rows]),
            energy=np.array([r[1] for r in rows]),
            amp=np.stack([r[2] for r in rows]),
        )

    def _calibrating(self, frame: Frame) -> None:
        self.cal_frames.append(frame)
        if frame.t - self.cal_frames[0].t < config.CALIBRATE_S:
            return
        cal = calibration_from_frames(self.cal_frames)
        if cal is None:
            return
        self.load_calibration(cal)
