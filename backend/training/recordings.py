from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from backend import config
from backend.capture.csi import DEFAULT_LINK, Frame, log_row, parse_line, read_log
from backend.detection.motion import MotionEvent, MotionTracker

RECORDINGS = config.DATA_DIR / "recordings"
CALIBRATION_FILE = "calibration.json"
EVENT_LABELS = ("walk", "sit-down", "fall")
LABELS = EVENT_LABELS + ("empty",)

Run = tuple[list[MotionEvent], np.ndarray, np.ndarray, np.ndarray]


@dataclass
class Recording:
    log_path: Path
    meta: dict

    @property
    def label(self) -> str:
        return self.meta["label"]

    def frames(self) -> list[Frame]:
        frames = (parse_line(line, t, link) for t, link, line in read_log(self.log_path))
        return [f for f in frames if f is not None]

    def _run(self) -> dict[str, Run]:
        if getattr(self, "_ran", None) is None:
            trackers = {}
            for link, cal in per_link(self.meta["calibration"]).items():
                trackers[link] = MotionTracker(link)
                trackers[link].load_calibration(cal)
            rows = {link: ([], [], [], []) for link in trackers}
            for frame in self.frames():
                tracker = trackers.get(frame.link)
                if tracker is None:
                    continue
                info, event = tracker.update(frame)
                events, ts, es, amps = rows[frame.link]
                if info.amp is not None:
                    ts.append(info.t)
                    es.append(info.energy)
                    amps.append(info.amp)
                if event is not None:
                    events.append(event)
            self._ran = {}
            for link, (events, ts, es, amps) in rows.items():
                last = trackers[link].flush()
                if last is not None:
                    events.append(last)
                self._ran[link] = (events, np.array(ts), np.array(es),
                                   np.stack(amps) if amps else np.empty((0, 0)))
        return self._ran

    def event(self) -> MotionEvent | None:
        events = [e for run in self._run().values() for e in run[0] if e.end >= self.meta["go_t"]]
        return max(events, key=lambda e: float(e.energy.max()), default=None)

    def decision_inputs(self) -> tuple[MotionEvent, np.ndarray] | None:
        event = self.event()
        if event is None:
            return None
        _, ts, _, amps = self._run()[event.link]
        decide_t = min(event.end + config.STILLNESS_S, float(ts[-1]))
        after = amps[(ts > decide_t - config.AFTER_S) & (ts <= decide_t)]
        return event, after

    def problems(self) -> list[str]:
        before = during = after_go = tail = 0.0
        for _, ts, es, _ in self._run().values():
            if not ts.size:
                continue
            ts = ts - self.meta["go_t"]
            warm = ts >= ts.min() + 2 * config.WINDOW_S
            before = max(before, es[warm & (ts < -0.1)].max(initial=0.0))
            during = max(during, es[warm].max(initial=0.0))
            after_go = max(after_go, es[ts >= 0].max(initial=0.0))
            tail = max(tail, es[ts >= ts.max() - config.AFTER_S].max(initial=0.0))
        found = []
        if before >= config.T_ON:
            found.append("moved before GO")
        if self.label == "empty":
            if during >= config.T_ON:
                found.append("movement in an empty-room clip")
        elif after_go < config.T_ON:
            found.append("no movement detected (move closer to the line between the boards)")
        if self.label in ("fall", "sit-down") and tail >= config.T_ON:
            found.append("moved at the end (stay still until saved)")
        return found


def per_link(cal: dict) -> dict[str, dict]:
    return {DEFAULT_LINK: cal} if "measure" in cal else cal


def save(lines: list[tuple[float, str, str]], meta: dict, root: Path = RECORDINGS) -> Path:
    folder = root / meta["split"]
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{meta['label']}_{datetime.now():%Y%m%d-%H%M%S}"
    base, n = folder / stem, 2
    while base.with_name(base.name + ".json").exists():
        base, n = folder / f"{stem}-{n}", n + 1
    log_path = base.with_name(base.name + ".log.gz")
    with gzip.open(log_path, "wt") as f:
        for t, link, line in lines:
            f.write(log_row(t, link, line))
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
