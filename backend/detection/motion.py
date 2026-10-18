from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

import numpy as np

from backend import config
from backend.capture.csi import DEFAULT_LINK, Frame

CALIBRATING, NO_MOTION, MOVING, STALE, WARMING = "calibrating", "no motion", "moving", "stale", "warming"
MEASURE = "mean-shift-0.5"
MIN_HALF_FRAMES = 4
# Older windows cannot describe a continuous event after capture drops a backlog.
PACKET_GAP_S = 0.5
CALIBRATION_RETRY_S = 0.5


@dataclass
class FrameInfo:
    t: float
    level: str
    energy: float | None
    amp: np.ndarray | None
    reason: str | None = None


@dataclass
class MotionEvent:
    start: float
    end: float
    t: np.ndarray
    energy: np.ndarray
    amp: np.ndarray
    cut_short: bool
    link: str = DEFAULT_LINK

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
    if not np.isfinite(t).all() or not np.isfinite(amps).all():
        return None
    mask = amps.mean(axis=0) > 0
    if not mask.any():
        return None
    amps = amps[:, mask]
    raws = np.array([r for i in range(len(t)) if (r := raw_energy(t[:i + 1], amps[:i + 1], t[i])) is not None])
    if raws.size < 10:
        return None
    baseline = float(np.median(raws))
    if not np.isfinite(baseline) or baseline <= 0:
        return None
    return {"measure": MEASURE, "size": int(size), "mask": np.nonzero(mask)[0].tolist(),
            "baseline": baseline, "noisy_frac": float((raws > config.T_ON * baseline).mean())}


class MotionTracker:
    def __init__(self, link: str = DEFAULT_LINK) -> None:
        self.link = link
        self.calibrate()

    def calibrate(self) -> None:
        self.size: int | None = None
        self.mask: np.ndarray | None = None
        self.baseline: float | None = None
        self.cal: dict | None = None
        self.cal_frames: list[Frame] = []
        self.cal_retry_at: float | None = None
        self.window: deque[tuple[float, np.ndarray]] = deque()
        self.history: deque[tuple[float, float, np.ndarray]] = deque()
        self.level = CALIBRATING
        self.energy = 0.0
        self.event_start: float | None = None
        self.quiet_since: float | None = None
        self.last_valid_t: float | None = None
        self.last_energy_t: float | None = None

    def interrupt(self, *, stale: bool = False) -> None:
        """Discard incomplete signal history without inventing an event ending."""
        self.window.clear()
        self.history.clear()
        self.cal_frames.clear()
        self.cal_retry_at = None
        self.event_start = self.quiet_since = None
        self.energy = 0.0
        self.last_energy_t = None
        self.level = STALE if stale else (CALIBRATING if self.baseline is None else NO_MOTION)

    def calibration(self) -> dict | None:
        return self.cal

    def load_calibration(self, cal: dict) -> None:
        if cal.get("measure") != MEASURE:
            raise ValueError(f"calibration is for motion measure {cal.get('measure')!r}, "
                             f"not {MEASURE!r}: recalibrate")
        size, mask, baseline = cal.get("size"), cal.get("mask"), cal.get("baseline")
        if (type(size) is not int or size <= 0 or not isinstance(mask, list) or not mask
                or any(type(i) is not int or not 0 <= i < size for i in mask)
                or len(set(mask)) != len(mask)):
            raise ValueError("calibration must contain a positive size and unique usable channels")
        if (type(baseline) not in (int, float) or not np.isfinite(baseline) or baseline <= 0):
            raise ValueError("calibration baseline must be positive and finite: recalibrate")
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
        if not frame.amp.size or not np.isfinite(frame.amp).all():
            return FrameInfo(frame.t, self.level, None, None, "invalid_amplitude"), None
        if self.size is not None and frame.amp.size != self.size:
            return FrameInfo(frame.t, self.level, None, None, "wrong_frame_size"), None
        usable = frame.amp if self.mask is None else frame.amp[self.mask]
        if not np.any(usable):
            # Missing CSI is not evidence of a quiet room. In particular, an
            # all-zero vector after a valid reflection would create a large
            # mean-shift spike followed by fabricated post-event stillness.
            return FrameInfo(frame.t, self.level, None, None, "empty_signal"), None

        gap = self.last_valid_t is not None and frame.t - self.last_valid_t > PACKET_GAP_S
        if gap or self.level == STALE:
            self.interrupt()
        self.last_valid_t = frame.t
        if self.baseline is None:
            self._calibrating(frame)
            # load_calibration resets timestamps along with its old history.
            self.last_valid_t = frame.t
            return FrameInfo(frame.t, self.level, None, None), None

        t, amp = frame.t, frame.amp[self.mask]
        self.window.append((t, amp))
        while t - self.window[0][0] > 2 * config.WINDOW_S:
            self.window.popleft()
        raw = raw_energy(np.array([w[0] for w in self.window]),
                         np.stack([w[1] for w in self.window]), t)
        if raw is None:
            # Missing energy cannot advance a quiet interval toward completion.
            self.quiet_since = None
            expired_energy = (self.last_energy_t is not None
                              and t - self.last_energy_t > PACKET_GAP_S)
            if expired_energy:
                self.interrupt()
            if self.level != MOVING:
                self.level = WARMING
            if self.last_energy_t is None and self.event_start is None:
                # These real amplitudes remain useful for the before-event
                # reference. Zero is only a pre-event placeholder; _step is
                # never called until a new energy measurement exists.
                self._remember(t, 0.0, amp)
            return FrameInfo(t, self.level, None, amp,
                             "insufficient_energy_window" if expired_energy
                             else "packet_gap" if gap else "insufficient_window"), None
        self.energy = raw / self.baseline
        self.last_energy_t = t
        if self.level == WARMING:
            self.level = NO_MOTION

        self._remember(t, self.energy, amp)

        event = self._step(t, self.energy)
        return FrameInfo(t, self.level, self.energy, amp), event

    def _remember(self, t: float, energy: float, amp: np.ndarray) -> None:
        self.history.append((t, energy, amp))
        keep_s = config.PRE_EVENT_S + config.MAX_EVENT_S + config.QUIET_S + 1.0
        while t - self.history[0][0] > keep_s:
            self.history.popleft()

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
            start=start, end=end, cut_short=cut_short, link=self.link,
            t=np.array([r[0] for r in rows]),
            energy=np.array([r[1] for r in rows]),
            amp=np.stack([r[2] for r in rows]),
        )

    def _calibrating(self, frame: Frame) -> None:
        self.cal_frames.append(frame)
        if self.cal_retry_at is not None:
            # An unusable baseline should not retain an entire live session.
            # Keep one frame at the boundary so the window still spans 10 s.
            cutoff = frame.t - config.CALIBRATE_S
            while len(self.cal_frames) > 1 and self.cal_frames[1].t <= cutoff:
                self.cal_frames.pop(0)
        if frame.t - self.cal_frames[0].t < config.CALIBRATE_S:
            return
        if self.cal_retry_at is not None and frame.t < self.cal_retry_at:
            return
        cal = calibration_from_frames(self.cal_frames)
        if cal is None:
            self.cal_retry_at = frame.t + CALIBRATION_RETRY_S
            return
        self.load_calibration(cal)
