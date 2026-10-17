from __future__ import annotations

import numpy as np

from backend import config
from backend.detection.motion import MotionEvent

NAMES = ["peak", "static_change", "duration_s", "mean_energy", "time_to_peak_s", "stop_s"]


def static_change(event: MotionEvent, after_amp: np.ndarray) -> float:
    before_rows = ((event.t >= event.start - config.PRE_EVENT_S)
                   & (event.t < event.start - config.BEFORE_GAP_S))
    if not before_rows.any() or len(after_amp) == 0:
        return 0.0
    before = event.amp[before_rows].mean(axis=0)
    after = after_amp.mean(axis=0)
    return float(np.linalg.norm(after - before) / max(np.linalg.norm(before), 1e-6))


def extract(event: MotionEvent, after_amp: np.ndarray) -> np.ndarray:
    inside = event.t >= event.start
    t, e = event.t[inside], event.energy[inside]
    if t.size == 0:
        t, e = event.t, event.energy
    i = int(np.argmax(e))
    peak = float(e[i])
    quiet = np.nonzero((t > t[i]) & (e < config.T_OFF))[0]
    stop_s = float(t[quiet[0]] - t[i]) if quiet.size else float(event.end - t[i])
    return np.array([
        peak, static_change(event, after_amp), event.duration, float(e.mean()),
        float(t[i] - event.start), stop_s,
    ])
