from __future__ import annotations

import numpy as np

from backend import config
from backend.detection import features
from backend.capture.csi import Frame
from backend.detection.detector import ThresholdDetector, Verdict
from backend.detection.fallflow import FallFlow, Sink
from backend.detection.motion import CALIBRATING, MOVING, FrameInfo, MotionEvent, MotionTracker


class Pipeline:
    def __init__(self, sink: Sink, detector=None) -> None:
        self.sink = sink
        self.trackers: dict[str, MotionTracker] = {}
        self.detector = detector or ThresholdDetector()
        self.flow = FallFlow(sink)
        self.ended: dict[str, MotionEvent] = {}
        self.pending: MotionEvent | None = None
        self.links = ""
        self.deadline = 0.0
        self.after: list[tuple[float, np.ndarray]] = []

    def calibrate(self) -> None:
        for tracker in self.trackers.values():
            tracker.calibrate()
        self.ended, self.pending, self.after = {}, None, []

    def process(self, frame: Frame) -> tuple[FrameInfo, tuple[MotionEvent, Verdict] | None]:
        tracker = self.trackers.get(frame.link)
        if tracker is None:
            tracker = self.trackers[frame.link] = MotionTracker(frame.link)
        was_calibrating = tracker.level == CALIBRATING
        info, event = tracker.update(frame)
        if was_calibrating and info.level != CALIBRATING:
            name = f"Link {frame.link} calibrated" if len(self.trackers) > 1 else "Calibrated"
            self.sink.log(info.t, f"{name} (baseline {tracker.baseline:.4f})")

        result = None
        if self.pending is not None:
            if frame.link == self.pending.link and info.amp is not None:
                self.after.append((info.t, info.amp))
            if info.level == MOVING:
                result = self._moved_again(info.t)
            elif info.t >= self.deadline:
                result = self._decide(info.t)

        if event is not None:
            if event.cut_short:
                verdict = Verdict("walk", 1.0, {"duration_s": round(event.duration, 2)})
                self._log(info.t, verdict, event.link, "")
                self.ended = {}
                result = (event, verdict)
            else:
                self.ended[event.link] = event

        if self.ended and not any(tr.level == MOVING for tr in self.trackers.values()):
            self._candidate(info.t)

        self.flow.tick(info.t)
        return info, result

    def _candidate(self, t: float) -> None:
        ended, self.ended = self.ended, {}
        event = max(ended.values(), key=lambda e: float(e.energy.max()))
        deadline = max(e.end for e in ended.values()) + config.STILLNESS_S
        if self.flow.candidate(t, deadline):
            self.pending, self.deadline, self.after = event, deadline, []
            self.links = "+".join(sorted(ended))
            self.sink.log(t, "Fall Candidate: stay still, deciding...")

    def _decide(self, t: float) -> tuple[MotionEvent, Verdict]:
        event, self.pending = self.pending, None
        after = [a for ts, a in self.after if ts > t - config.AFTER_S]
        x = features.extract(event, np.stack(after) if after else np.empty((0, 0)))
        verdict = self.detector.classify(x)
        self._log(t, verdict, self.links, "")
        if verdict.label == "fall":
            self.flow.confirm(t)
        else:
            self.flow.not_a_fall(t)
        return event, verdict

    def _moved_again(self, t: float) -> tuple[MotionEvent, Verdict]:
        event, self.pending = self.pending, None
        label = "walk" if event.duration > config.SIT_MAX_S else "sit-down"
        verdict = Verdict(label, 0.5, {"duration_s": round(event.duration, 2)})
        self._log(t, verdict, self.links, " (moved again, so not a fall)")
        self.flow.not_a_fall(t)
        return event, verdict

    def _log(self, t: float, verdict: Verdict, links: str, note: str) -> None:
        if len(self.trackers) > 1:
            verdict.details["links"] = links
        details = ", ".join(f"{k} {v}" for k, v in verdict.details.items())
        self.sink.log(t, f"Motion Event: {verdict.label} ({details}){note}")
