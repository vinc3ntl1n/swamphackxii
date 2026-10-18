"""Recorded CSI and capture faults through the shared RF/dashboard pipeline.

Private model/recording checks skip on fresh clones without synchronized data.
The bundled demo includes its existing synthetic tail; it is integration evidence,
not an independent measurement of detection accuracy.
"""
import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from backend import dashboard, server
from backend.capture.csi import Frame
from backend.capture.replay import replay_frames
from backend.detection import features
from backend.detection.detector import RandomForestDetector, Verdict
from backend.detection.fallflow import COUNTDOWN, IDLE
from backend.detection.motion import STALE, WARMING
from backend.detection.pipeline import Pipeline
from backend.diagnostics import DiagnosticSink, EventLog


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "data/experiments/batch-20261017-233929"
CAL = {"measure": "mean-shift-0.5", "size": 2, "mask": [0, 1], "baseline": 1.0}


class Detector:
    name = "controlled-rf"
    fall_threshold = None

    def __init__(self):
        self.inputs = []

    def classify(self, x):
        self.inputs.append(x.copy())
        return Verdict("fall", 0.9, {"peak": float(x[0]), "fall_probability": 0.9})


def motion_energy(t, amps, now):
    return 20.0 if 2.0 <= round(now, 6) < 2.5 else 1.0


class LivePipelineIntegrationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.counter = 0

    def make_view(self, detector, calibration=None):
        self.counter += 1
        root = self.root / str(self.counter)
        log = EventLog(root / "events.jsonl")
        self.addCleanup(log.close)
        hub = server.Hub()
        published = []
        original_publish = hub.publish

        def publish(message):
            published.append(message)
            original_publish(message)

        hub.publish = publish
        history = dashboard.History(hub, "replay", root / "history.jsonl")
        notices = dashboard.Notices(hub, history)
        # No messaging sidecar, browser, server, background ticker or API is used.
        notices.send = Mock()
        web = dashboard.WebSink(hub, history, notices)
        sink = DiagnosticSink(web, log)
        pipe = Pipeline(sink, detector)
        dashboard.preload(pipe, calibration or {"A": CAL})
        view = dashboard.Dashboard(hub, pipe, sink, web, source="REPLAY", diagnostics=log, live=False)
        return view, published, log.path

    def rows(self, path):
        return [json.loads(line) for line in path.read_text().splitlines()]

    def feed(self, view, stop, links=("A",)):
        for t in np.arange(0, stop + 0.001, 0.1):
            for link in links:
                view.process(Frame(round(float(t), 6), -40, np.full(2, 10.0), link))

    def replay_model(self, artifact, clip, expected_label, threshold, hold=True):
        if not artifact.exists() or not clip.exists():
            self.skipTest("synchronize the private RF artifacts/recordings to run recorded-data checks")
        before = hashlib.sha256(artifact.read_bytes()).hexdigest()
        detector = RandomForestDetector(artifact)
        classes = detector.model.classes_.copy()
        forest = detector.model
        trees = tuple(detector.model.estimators_)
        params = detector.model.get_params()
        view, published, path = self.make_view(detector, dashboard.recording_calibration(clip))
        frames = replay_frames(clip, 0)
        if hold:
            frames = dashboard.held(frames, dashboard.HOLD_S, 0)
        with patch.object(detector, "classify", wraps=detector.classify) as classify, \
                contextlib.redirect_stdout(io.StringIO()):
            for frame in frames:
                view.process(frame)
        classify.assert_called_once()
        x = classify.call_args.args[0]
        self.assertEqual(x.shape, (len(features.NAMES),))
        self.assertTrue(np.isfinite(x).all())
        classified = [row for row in self.rows(path)
                      if row["type"] == "detection" and row["reason"] == "classified"]
        self.assertEqual(len(classified), 1)
        record = classified[0]
        self.assertEqual(record["detector"], "rf")
        self.assertEqual(record["label"], expected_label)
        self.assertEqual(record["fall_threshold"], threshold)
        self.assertEqual(set(record["features"]), set(features.NAMES))
        np.testing.assert_allclose([record["features"][key] for key in features.NAMES], x)
        p = detector.model.predict_proba(x.reshape(1, -1))[0]
        self.assertAlmostEqual(record["fall_probability"],
                               p[list(detector.model.classes_).index("fall")], places=4)
        ui_events = [message for message in published if message["type"] == "event"]
        self.assertEqual(len(ui_events), 1)
        self.assertEqual((ui_events[0]["label"], ui_events[0]["why"]),
                         (expected_label, "classified"))
        self.assertFalse(view.failed)
        if expected_label == "fall":
            self.assertEqual(view.pipe.flow.state, COUNTDOWN)
            self.assertTrue(any(m["type"] == "state" and m["state"] == COUNTDOWN for m in published))
            self.assertTrue(any(m["type"] == "alarm" and m["on"] for m in published))
            view.web.notices.send.assert_called_once()
        else:
            self.assertEqual(view.pipe.flow.state, IDLE)
            view.web.notices.send.assert_not_called()
        self.assertIs(detector.model, forest)
        self.assertEqual(tuple(detector.model.estimators_), trees)
        self.assertEqual(detector.model.get_params(), params)
        np.testing.assert_array_equal(detector.model.classes_, classes)
        self.assertEqual(detector.fall_threshold, threshold)
        self.assertEqual(hashlib.sha256(artifact.read_bytes()).hexdigest(), before)

    def test_bundled_demo_baseline_keeps_its_actual_nonfall_verdict(self):
        self.replay_model(MODELS / "baseline.joblib", dashboard.SIM_FALL, "walk", None)

    def test_bundled_demo_experimental_threshold_reaches_countdown(self):
        self.replay_model(MODELS / "threshold-035.joblib", dashboard.SIM_FALL, "fall", 0.35)

    def test_measured_multilink_recording_reaches_baseline_countdown_without_padding(self):
        self.replay_model(MODELS / "baseline.joblib",
                          ROOT / "data/recordings/train/fall_20261017-230857.log.gz",
                          "fall", None, hold=False)

    def test_dashboard_idle_tick_expires_all_receivers_without_calling_forest(self):
        detector = Detector()
        view, published, path = self.make_view(detector, {"A": CAL, "B": CAL})
        with patch("backend.detection.motion.raw_energy", side_effect=motion_energy):
            self.feed(view, 3.6, ("A", "B"))
        self.assertIsNotNone(view.pipe.pending)
        with patch.object(view.clock, "now", return_value=7.0), \
                patch.object(view.clock, "stalled", return_value=True):
            view._tick()
        self.assertIsNone(view.pipe.pending)
        self.assertEqual(view.pipe.flow.state, IDLE)
        self.assertTrue(all(tracker.level == STALE for tracker in view.pipe.trackers.values()))
        self.assertFalse(detector.inputs)
        reasons = [row.get("reason") for row in self.rows(path)]
        self.assertEqual(reasons.count("receiver_stale"), 2)
        self.assertIn("selected_receiver_stale", reasons)
        self.assertFalse(any(message["type"] == "event" for message in published))
        view.web.notices.send.assert_not_called()

    def test_capture_drop_aborts_pending_immediately_before_next_frame(self):
        detector = Detector()
        view, _, path = self.make_view(detector)
        with patch("backend.detection.motion.raw_energy", side_effect=motion_energy):
            self.feed(view, 3.6)
        self.assertIsNotNone(view.pipe.pending)
        view.capture_health({"dropped_since_report": 4, "dropped_links": ["A"],
                             "last_drop": {"capture_t": 3.65}, "capture_wall_t": 3.65, "idle": False})
        self.assertIsNone(view.pipe.pending)
        self.assertEqual(view.pipe.flow.state, IDLE)
        self.assertEqual(view.pipe.trackers["A"].level, WARMING)
        self.assertFalse(view.pipe.trackers["A"].window)
        self.assertFalse(detector.inputs)
        with patch("backend.detection.motion.raw_energy", return_value=None):
            view.process(Frame(3.7, -40, np.full(2, 10.0), "A"))
        self.assertFalse(detector.inputs)
        self.assertIn("selected_receiver_capture_gap", [row.get("reason") for row in self.rows(path)])
        view.web.notices.send.assert_not_called()

    def test_capture_drop_preserves_already_confirmed_countdown(self):
        detector = Detector()
        view, published, _ = self.make_view(detector)
        with patch("backend.detection.motion.raw_energy", side_effect=motion_energy):
            self.feed(view, 7.0)
        self.assertEqual(len(detector.inputs), 1)
        self.assertEqual(view.pipe.flow.state, COUNTDOWN)
        deadline = view.pipe.flow.deadline
        view.capture_health({"dropped_since_report": 2, "dropped_links": ["A"],
                             "last_drop": {"capture_t": 7.1}, "capture_wall_t": 7.1, "idle": False})
        self.assertEqual(view.pipe.flow.state, COUNTDOWN)
        self.assertEqual(view.pipe.flow.deadline, deadline)
        self.assertEqual(len(detector.inputs), 1)
        self.assertFalse(any(m["type"] == "alarm" and not m["on"] for m in published))
        view.web.notices.send.assert_called_once()


if __name__ == "__main__":
    unittest.main()
