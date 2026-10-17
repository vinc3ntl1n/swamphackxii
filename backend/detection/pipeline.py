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
        self.motion = MotionTracker()
        self.detector = detector or ThresholdDetector()
        self.flow = FallFlow(sink)
        self.pending: MotionEvent | None = None
        self.after: list[tuple[float, np.ndarray]] = []

    def calibrate(self) -> None:
        self.motion.calibrate()
        self.pending, self.after = None, []

    def process(self, frame: Frame) -> tuple[FrameInfo, tuple[MotionEvent, Verdict] | None]:
        was_calibrating = self.motion.level == CALIBRATING
        info, event = self.motion.update(frame)
        if was_calibrating and info.level != CALIBRATING:
            self.sink.log(info.t, f"Calibrated (baseline {self.motion.baseline:.4f})")

        result = None
        if self.pending is not None:
            if info.amp is not None:
                self.after.append((info.t, info.amp))
            if info.level == MOVING:
                result = self._moved_again(info.t)
            elif info.t >= self.pending.end + config.STILLNESS_S:
                result = self._decide(info.t)

        if event is not None:
            if event.cut_short:
                verdict = Verdict("walk", 1.0, {"duration_s": round(event.duration, 2)})
                self._log(info.t, verdict, "")
                result = (event, verdict)
            elif self.flow.candidate(info.t, event.end + config.STILLNESS_S):
                self.pending, self.after = event, []
                self.sink.log(info.t, "Fall Candidate: stay still, deciding...")

        self.flow.tick(info.t)
        return info, result

    def _decide(self, t: float) -> tuple[MotionEvent, Verdict]:
        event, self.pending = self.pending, None
        after = [a for ts, a in self.after if ts > t - config.AFTER_S]
        x = features.extract(event, np.stack(after) if after else np.empty((0, 0)))
        verdict = self.detector.classify(x)
        self._log(t, verdict, "")
        if verdict.label == "fall":
            self.flow.confirm(t)
        else:
            self.flow.not_a_fall(t)
        return event, verdict

    def _moved_again(self, t: float) -> tuple[MotionEvent, Verdict]:
        event, self.pending = self.pending, None
        label = "walk" if event.duration > config.SIT_MAX_S else "sit-down"
        verdict = Verdict(label, 0.5, {"duration_s": round(event.duration, 2)})
        self._log(t, verdict, " (moved again, so not a fall)")
        self.flow.not_a_fall(t)
        return event, verdict

    def _log(self, t: float, verdict: Verdict, note: str) -> None:
        details = ", ".join(f"{k} {v}" for k, v in verdict.details.items())
        self.sink.log(t, f"Motion Event: {verdict.label} ({details}){note}")
