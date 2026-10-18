from __future__ import annotations

import json
import time
from collections.abc import Iterable

import numpy as np

from backend import config
from backend.detection import features
from backend.capture.csi import Frame
from backend.detection.detector import ThresholdDetector, Verdict
from backend.detection.fallflow import FallFlow, Sink
from backend.detection.motion import (CALIBRATING, MOVING, STALE, MIN_HALF_FRAMES,
                                      WARMING, PACKET_GAP_S, FrameInfo, MotionEvent, MotionTracker)

RECEIVER_STALE_S = 2.0


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
        self._issues: dict[str, str] = {}
        self._waiting_links: tuple[str, ...] = ()
        self.suspended: set[str] = set()

    def calibrate(self, t: float | None = None) -> None:
        for tracker in self.trackers.values():
            tracker.calibrate()
        now = t if t is not None else self.flow.started or 0.0
        if self.pending is not None:
            self._abort(now, "recalibrated")
        self.ended = {}
        self._discard_candidate(now)
        self._issues.clear()
        self._waiting_links = ()

    def health(self, t: float) -> dict[str, dict]:
        """Receiver health uses packet time, so recorded and live streams agree."""
        status = {}
        for link, tracker in self.trackers.items():
            age = None if tracker.last_valid_t is None else max(0.0, t - tracker.last_valid_t)
            if (link in self.suspended or tracker.level == STALE
                    or (age is not None and age >= RECEIVER_STALE_S)):
                state = "stale"
            elif tracker.baseline is None:
                state = "calibrating"
            elif tracker.last_energy_t is None or t - tracker.last_energy_t > PACKET_GAP_S:
                state = "warming"
            else:
                state = "ok"
            status[link] = {"status": state, "last_valid_t": tracker.last_valid_t,
                            "last_energy_t": tracker.last_energy_t, "age_s": age,
                            "level": tracker.level,
                            "energy": tracker.energy if state == "ok" else None,
                            "baseline": tracker.baseline}
        return status

    def tick(self, t: float) -> None:
        """Expire missing links and advance alerts; never fabricate quiet samples."""
        for link, tracker in self.trackers.items():
            if (tracker.level != STALE and tracker.last_valid_t is not None
                    and t - tracker.last_valid_t >= RECEIVER_STALE_S):
                self._diagnostic(t, "receiver_stale", link,
                                 f"age_s={t - tracker.last_valid_t:.3f}; detection degraded")
                tracker.interrupt(stale=True)
                self.ended.pop(link, None)
                if self.pending is not None and self.pending.link == link:
                    self._abort(t, "selected_receiver_stale")
        self._try_candidate(t)
        self.flow.tick(t)

    def capture_gap(self, t: float, links: Iterable[str] | None = None,
                    reason: str = "capture_drop") -> None:
        """Explicit queue loss invalidates histories even when a time gap is short."""
        affected = set(self.trackers if links is None else links)
        self._diagnostic(t, reason, "+".join(sorted(affected)),
                         "capture lost samples; incomplete observations discarded",
                         dropped_links=sorted(affected))
        for link in affected:
            tracker = self.trackers.get(link)
            if tracker is not None:
                tracker.interrupt()
                if tracker.baseline is not None:
                    tracker.level = WARMING
            self.ended.pop(link, None)
        if self.pending is not None and self.pending.link in affected:
            self._abort(t, "selected_receiver_capture_gap")

    def suspend_link(self, link: str, t: float) -> bool:
        """Invalidate a lost live signal; require fresh calibration when it returns.

        Call this from live-source health monitoring, never to pad a replay or to
        infer stillness from missing packets. Confirmed fall countdowns continue.
        """
        tracker = self.trackers.get(link)
        if tracker is None or link in self.suspended:
            return False
        self.suspended.add(link)
        # Losing B must not erase A's real completed event. It may continue
        # in degraded mode if A can supply a complete observation itself.
        self.ended.pop(link, None)
        tracker.calibrate()
        if self.pending is not None and link in self.links.split("+"):
            self._abort(t, "selected_receiver_stale" if link == self.pending.link
                        else "participating_receiver_stale")
        self._diagnostic(t, "receiver_suspended", link,
                         "live signal lost; fresh calibration required on return")
        self.sink.log(t, f"Link {link} signal lost; recalibrating when packets resume")
        return True

    def _discard_candidate(self, t: float) -> None:
        self.pending, self.after, self.links, self.deadline = None, [], "", 0.0
        self.flow.not_a_fall(t)

    def process(self, frame: Frame) -> tuple[FrameInfo, tuple[MotionEvent, Verdict] | None]:
        self.tick(frame.t)
        tracker = self.trackers.get(frame.link)
        if tracker is None:
            tracker = self.trackers[frame.link] = MotionTracker(frame.link)
        was_stale = tracker.level == STALE or frame.link in self.suspended
        previous_level = tracker.level
        last_valid_t = tracker.last_valid_t
        was_calibrating = tracker.level == CALIBRATING
        info, event = tracker.update(frame)
        valid = tracker.last_valid_t == frame.t and info.reason not in (
            "invalid_amplitude", "wrong_frame_size")
        if valid:
            self.suspended.discard(frame.link)
        if valid and last_valid_t is not None and frame.t - last_valid_t > PACKET_GAP_S:
            self._diagnostic(info.t, "packet_gap", frame.link,
                             f"gap_s={frame.t - last_valid_t:.3f}; incomplete event discarded")
            self.ended.pop(frame.link, None)
            if self.pending is not None and self.pending.link == frame.link:
                self._abort(info.t, "selected_receiver_packet_gap")
        if was_stale and valid:
            self._diagnostic(info.t, "receiver_recovered", frame.link, "warming fresh windows")
        if info.reason == "insufficient_energy_window":
            self.ended.pop(frame.link, None)
            if self.pending is not None and self.pending.link == frame.link:
                self._abort(info.t, "selected_receiver_energy_gap")
        issue = info.reason or ("insufficient_window" if valid and info.energy is None
                                and tracker.baseline is not None else "")
        if issue and issue != self._issues.get(frame.link):
            self._diagnostic(info.t, issue, frame.link, "no new motion energy; RF not called")
        elif not issue and self._issues.get(frame.link):
            self._diagnostic(info.t, "signal_ready", frame.link, "fresh motion energy available")
        self._issues[frame.link] = issue
        if previous_level != MOVING and info.level == MOVING:
            self._diagnostic(info.t, "motion_started", frame.link,
                             "motion threshold crossed", energy=info.energy, threshold=config.T_ON)
        if was_calibrating and info.level != CALIBRATING:
            name = f"Link {frame.link} calibrated" if len(self.trackers) > 1 else "Calibrated"
            self.sink.log(info.t, f"{name} (baseline {tracker.baseline:.4f})")

        result = None
        if self.pending is not None:
            if frame.link == self.pending.link and info.amp is not None:
                self.after.append((info.t, info.amp))
            if info.energy is not None and info.level == MOVING:
                result = self._moved_again(info.t, frame.link)
            elif (info.t >= self.deadline and frame.link == self.pending.link
                  and info.amp is not None):
                result = self._decide(info.t)

        if event is not None:
            self._diagnostic(info.t, "event_completed", event.link,
                             "motion event ended", start=event.start, end=event.end,
                             duration_s=event.duration, cut_short=event.cut_short)
            if event.cut_short:
                self._diagnostic(info.t, "max_event_duration", event.link,
                                 "event routed to walking; RF not called", duration_s=event.duration)
                verdict = Verdict("walk", 1.0, {"duration_s": round(event.duration, 2),
                                              "reason": "max_event_duration", "model_called": False})
                self._log(info.t, verdict, event.link, "")
                self.ended = {}
                result = (event, verdict)
            else:
                self.ended[event.link] = event

        self._try_candidate(info.t)

        self.flow.tick(info.t)
        return info, result

    def _try_candidate(self, t: float) -> None:
        waiting = tuple(sorted(link for link, tr in self.trackers.items() if tr.level == MOVING))
        if self.ended and waiting:
            if waiting != self._waiting_links:
                self._diagnostic(t, "waiting_for_quiet", "+".join(waiting),
                                 "completed event waiting for other moving receivers")
            self._waiting_links = waiting
        elif self.ended:
            self._waiting_links = ()
            self._candidate(t)
        else:
            self._waiting_links = ()

    def _candidate(self, t: float) -> None:
        ended, self.ended = self.ended, {}
        event = max(ended.values(), key=lambda e: float(e.energy.max()))
        deadline = max(e.end for e in ended.values()) + config.STILLNESS_S
        if self.flow.candidate(t, deadline):
            self.pending, self.deadline, self.after = event, deadline, []
            self.links = "+".join(sorted(ended))
            self.sink.log(t, "Fall Candidate: stay still, deciding...")
            degraded = "+".join(link for link, h in self.health(t).items() if h["status"] != "ok")
            self._diagnostic(t, "candidate_started", event.link,
                             "post-event observation started", deadline=deadline,
                             degraded_links=degraded.split("+") if degraded else [])
        else:
            self._diagnostic(t, "blocked_alert", event.link,
                             "candidate not accepted by alert state", state=self.flow.state)

    def _decide(self, t: float) -> tuple[MotionEvent, Verdict] | None:
        event = self.pending
        after = [(ts, a) for ts, a in self.after if ts > t - config.AFTER_S]
        tracker = self.trackers[event.link]
        if (tracker.last_energy_t is None or t - tracker.last_energy_t > PACKET_GAP_S
                or len(after) < MIN_HALF_FRAMES
                or after[0][0] > t - config.AFTER_S + config.WINDOW_S
                or after[-1][0] < t - config.WINDOW_S):
            self._abort(t, "insufficient_after_window", f"after_frames={len(after)}")
            return None
        self.pending = None
        x = features.extract(event, np.stack([a for _, a in after]))
        started = time.perf_counter()
        verdict = self.detector.classify(x)
        classify_ms = (time.perf_counter() - started) * 1000
        self._diagnostic(t, "classified", event.link,
                         "event classifier completed",
                         detector=getattr(self.detector, "name", type(self.detector).__name__),
                         label=verdict.label, confidence=verdict.confidence,
                         fall_probability=verdict.details.get("fall_probability"),
                         class_probabilities=verdict.details.get("class_probabilities"),
                         fall_threshold=getattr(self.detector, "fall_threshold", None),
                         classify_ms=round(classify_ms, 3),
                         features=dict(zip(features.NAMES, x.tolist())))
        self._log(t, verdict, self.links, "")
        if verdict.label == "fall":
            self.flow.confirm(t)
        else:
            self.flow.not_a_fall(t)
        return event, verdict

    def _abort(self, t: float, reason: str, note: str = "") -> None:
        link = self.pending.link if self.pending is not None else ""
        self._diagnostic(t, reason, link, f"candidate aborted; RF not called; {note}")
        self._discard_candidate(t)

    def _moved_again(self, t: float, moving_link: str | None = None) -> tuple[MotionEvent, Verdict]:
        event, self.pending = self.pending, None
        label = "walk" if event.duration > config.SIT_MAX_S else "sit-down"
        verdict = Verdict(label, 0.5, {"duration_s": round(event.duration, 2),
                                     "reason": "resumed_movement", "model_called": False})
        self._diagnostic(t, "resumed_movement", event.link, "candidate rejected; RF not called",
                         moving_link=moving_link)
        self._log(t, verdict, self.links, " (moved again, so not a fall)")
        self.flow.not_a_fall(t)
        return event, verdict

    def _diagnostic(self, t: float, reason: str, link: str, note: str, **details) -> None:
        payload = {"reason": reason, "link": link, "message": note, **details}
        self.sink.log(t, "Detection diagnostic: " + json.dumps(payload, sort_keys=True))

    def _log(self, t: float, verdict: Verdict, links: str, note: str) -> None:
        if len(self.trackers) > 1:
            verdict.details["links"] = links
        details = ", ".join(f"{k} {v}" for k, v in verdict.details.items())
        self.sink.log(t, f"Motion Event: {verdict.label} ({details}){note}")
