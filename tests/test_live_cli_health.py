"""Exercise live CLI health routing without receivers, browser, or notifications."""

import contextlib
import io
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from backend import run
from backend.capture import live
from backend.capture.csi import Frame
from backend.detection.detector import Verdict
from backend.detection.fallflow import COUNTDOWN, IDLE
from backend.detection.motion import MotionTracker
from backend.detection.pipeline import Pipeline
from backend.live_runtime import TerminalRuntime


BASE = 1000.0
CAL = {"measure": "mean-shift-0.5", "size": 2, "mask": [0, 1], "baseline": 1.0}


class DetectorSpy:
    name = "rf"
    trained_at = "controlled-test"
    fall_threshold = None

    def __init__(self):
        self.inputs = []

    def classify(self, x):
        self.inputs.append(x)
        return Verdict("fall", 0.9, {"fall_probability": 0.9})


class LiveCliHealthTests(unittest.TestCase):
    def test_terminal_lock_wait_cannot_make_old_capture_look_fresh(self):
        pipe = Pipeline(Mock(), DetectorSpy())
        tracker = pipe.trackers["A"] = MotionTracker("A")
        tracker.load_calibration(CAL)
        runtime = TerminalRuntime(pipe, links=("A",), live_age=True)
        started, result, clock = threading.Event(), [], [BASE]

        def process():
            started.set()
            result.append(runtime.process(Frame(BASE, -40, np.ones(2), "A")))

        with patch("backend.live_runtime.time.time", side_effect=lambda: clock[0]):
            with runtime.lock:
                worker = threading.Thread(target=process)
                worker.start()
                self.assertTrue(started.wait(1))
                clock[0] = BASE + 0.8
            worker.join(1)
        self.assertFalse(worker.is_alive())
        info, verdict = result[0]
        self.assertEqual(info.reason, "stale_capture")
        self.assertIsNone(info.energy)
        self.assertIsNone(info.amp)
        self.assertIsNone(verdict)
        self.assertFalse(pipe.detector.inputs)
        self.assertFalse(runtime.sensors.seen)
        self.assertIsNone(runtime.clock.t)

    def exercise(self, mode):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        model_path = root / "controlled-model.joblib"
        # The controlled spy tests wiring; test_rf_runtime exercises real model loading.
        model_path.write_bytes(b"controlled detector artifact for startup provenance")
        detector = DetectorSpy()
        timeline, pipes, closed = [], [], []
        clock = [BASE]

        def make_pipeline(sink, selected):
            self.assertIs(selected, detector)
            pipe = Pipeline(sink, selected)
            tracker = pipe.trackers["A"] = MotionTracker("A")
            tracker.load_calibration(CAL)
            for name in ("capture_gap", "tick", "process"):
                original = getattr(pipe, name)

                def invoke(*args, _name=name, _original=original, **kwargs):
                    value = args[0].t if _name == "process" else args[0]
                    timeline.append((_name, value))
                    return _original(*args, **kwargs)

                setattr(pipe, name, invoke)
            pipes.append(pipe)
            return pipe

        def frames(ports, log_path, stop=None, *, health_callback):
            def health(*, idle=False, dropped=0):
                row = {"type": "capture_health", "idle": idle,
                       "capture_wall_t": BASE + 9, "capture_age_ms": 0,
                       "queue_depth": 0, "dropped_frames": dropped,
                       "dropped_since_report": dropped, "dropped_links": ["A"] if dropped else [],
                       "last_drop": {"capture_t": BASE + 3.65, "link": "A"} if dropped else None}
                live.LOGGER.info(json.dumps(row))
                health_callback(row)

            try:
                health()
                for n in range(71):
                    clock[0] = BASE + n / 10
                    if mode == "drop" and n == 37:
                        health(dropped=2)
                    yield Frame(BASE + n / 10, -40, np.full(2, 10.0), "A")
                    if mode == "idle" and n == 36:
                        health(idle=True)
                        return
            finally:
                closed.append(True)

        def energy(t, amps, now):
            return 20.0 if 2.0 <= round(now - BASE, 6) < 2.5 else 1.0

        level, handlers = live.LOGGER.level, list(live.LOGGER.handlers)
        argv = ["run", "--no-view", "--detector", "rf", "--model", str(model_path)]
        with patch("sys.argv", argv), \
                patch.object(live, "SESSIONS", root / "sessions"), \
                patch.object(live, "receivers", return_value={"A": "fake-port"}), \
                patch.object(live, "live_frames", side_effect=frames), \
                patch.object(run, "Pipeline", side_effect=make_pipeline), \
                patch.object(run.detector, "make", return_value=detector), \
                patch("backend.live_runtime.time.time", side_effect=lambda: clock[0]), \
                patch("backend.detection.motion.raw_energy", side_effect=energy), \
                patch.object(run, "start_view") as start_view, \
                contextlib.redirect_stdout(io.StringIO()) as output:
            run.main()
        start_view.assert_not_called()
        self.assertIn("Detector: rf", output.getvalue())
        self.assertEqual(live.LOGGER.level, level)
        self.assertEqual(live.LOGGER.handlers, handlers)
        self.assertEqual(closed, [True])
        files = list((root / "sessions").glob("*.events.jsonl"))
        self.assertEqual(len(files), 1)
        rows = [json.loads(line) for line in files[0].read_text().splitlines()]
        self.assertEqual(rows[0]["type"], "run_started")
        self.assertEqual(rows[-1]["type"], "run_stopped")
        self.assertTrue(any(row["type"] == "capture_health" for row in rows))
        self.assertEqual(rows[0]["detector"], "rf")
        self.assertEqual(len(rows[0]["model_sha256"]), 64)
        return pipes[0], detector, rows, timeline

    def test_healthy_live_cli_records_rf_verdict_and_starts_countdown(self):
        pipe, detector, rows, _ = self.exercise("healthy")
        self.assertEqual(len(detector.inputs), 1)
        self.assertEqual(pipe.flow.state, COUNTDOWN)
        classified = [row for row in rows if row.get("reason") == "classified"]
        self.assertEqual(len(classified), 1)
        self.assertEqual(classified[0]["fall_probability"], 0.9)
        self.assertEqual(len(classified[0]["features"]), 6)

    def test_drop_callback_invalidates_candidate_before_next_kept_frame(self):
        pipe, detector, rows, timeline = self.exercise("drop")
        gap = timeline.index(("capture_gap", BASE + 3.65))
        next_frame = timeline.index(("process", BASE + 3.7))
        self.assertLess(gap, next_frame)
        self.assertFalse(detector.inputs)
        self.assertIsNone(pipe.pending)
        self.assertEqual(pipe.flow.state, IDLE)
        self.assertIn("selected_receiver_capture_gap", [row.get("reason") for row in rows])

    def test_idle_callback_expires_receiver_without_fabricating_classification(self):
        pipe, detector, rows, timeline = self.exercise("idle")
        self.assertIn(("tick", BASE + 9), timeline)
        self.assertFalse(detector.inputs)
        self.assertIsNone(pipe.pending)
        self.assertEqual(pipe.flow.state, IDLE)
        self.assertIn("receiver_stale", [row.get("reason") for row in rows])
        self.assertIn("selected_receiver_stale", [row.get("reason") for row in rows])


if __name__ == "__main__":
    unittest.main()
