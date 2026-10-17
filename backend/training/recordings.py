from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from backend import config
from backend.capture.csi import Frame, parse_line, read_log
from backend.detection.motion import MotionEvent, MotionTracker

RECORDINGS = config.DATA_DIR / "recordings"
CALIBRATION_FILE = "calibration.json"
EVENT_LABELS = ("walk", "sit-down", "fall")
LABELS = EVENT_LABELS + ("empty",)


@dataclass
class Recording:
    log_path: Path
    meta: dict

    @property
    def label(self) -> str:
        return self.meta["label"]

    def frames(self) -> list[Frame]:
        frames = (parse_line(line, t) for t, line in read_log(self.log_path))
        return [f for f in frames if f is not None]

    def _run(self) -> tuple[list[MotionEvent], np.ndarray, np.ndarray, np.ndarray]:
        if getattr(self, "_ran", None) is None:
            tracker = MotionTracker()
            tracker.load_calibration(self.meta["calibration"])
            events, ts, es, amps = [], [], [], []
            for frame in self.frames():
                info, event = tracker.update(frame)
                if info.amp is not None:
                    ts.append(info.t)
                    es.append(info.energy)
                    amps.append(info.amp)
                if event is not None:
                    events.append(event)
            last = tracker.flush()
            if last is not None:
                events.append(last)
            self._ran = (events, np.array(ts), np.array(es), np.stack(amps) if amps else np.empty((0, 0)))
        return self._ran

    def event(self) -> MotionEvent | None:
        events = [e for e in self._run()[0] if e.end >= self.meta["go_t"]]
        return max(events, key=lambda e: float(e.energy.max()), default=None)

    def decision_inputs(self) -> tuple[MotionEvent, np.ndarray] | None:
        event = self.event()
        if event is None:
            return None
        _, ts, _, amps = self._run()
        decide_t = min(event.end + config.STILLNESS_S, float(ts[-1]))
        after = amps[(ts > decide_t - config.AFTER_S) & (ts <= decide_t)]
        return event, after

    def problems(self) -> list[str]:
        _, ts, es, _ = self._run()
        ts = ts - self.meta["go_t"]
        warm = ts >= ts.min() + 2 * config.WINDOW_S
        found = []
        pre = es[warm & (ts < -0.1)]
        if pre.size and pre.max() >= config.T_ON:
            found.append("moved before GO")
        if self.label == "empty":
            if es[warm].max(initial=0.0) >= config.T_ON:
                found.append("movement in an empty-room clip")
        elif es[ts >= 0].max(initial=0.0) < config.T_ON:
            found.append("no movement detected (move closer to the line between the boards)")
        if self.label in ("fall", "sit-down") and ts.size:
            tail = es[ts >= ts.max() - config.AFTER_S]
            if tail.size and tail.max() >= config.T_ON:
                found.append("moved at the end (stay still until saved)")
        return found


def save(lines: list[tuple[float, str]], meta: dict, root: Path = RECORDINGS) -> Path:
    folder = root / meta["split"]
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{meta['label']}_{datetime.now():%Y%m%d-%H%M%S}"
    base, n = folder / stem, 2
    while base.with_name(base.name + ".json").exists():
        base, n = folder / f"{stem}-{n}", n + 1
    log_path = base.with_name(base.name + ".log.gz")
    with gzip.open(log_path, "wt") as f:
        for t, line in lines:
            f.write(f"{t:.6f}\t{line}\n")
    base.with_name(base.name + ".json").write_text(json.dumps(meta, indent=1))
    return log_path


def load(split: str | None = None, root: Path = RECORDINGS) -> list[Recording]:
    pattern = f"{split}/*.json" if split else "*/*.json"
    recs = []
    for meta_path in sorted(root.glob(pattern)):
        log_path = meta_path.with_name(meta_path.stem + ".log.gz")
        if log_path.exists():
            recs.append(Recording(log_path, json.loads(meta_path.read_text())))
    return recs
