"""Terminal capture must keep the same health/countdown rules as the dashboard."""
import unittest
from unittest.mock import Mock, patch

import numpy as np

from backend.capture.csi import Frame
from backend.detection.detector import Verdict
from backend.detection.fallflow import ALERTED, COUNTDOWN, FALL_CANDIDATE, IDLE
from backend.detection.motion import CALIBRATING, MEASURE, MotionEvent, MotionTracker
from backend.detection.pipeline import Pipeline
from backend.live_runtime import TerminalRuntime


class TerminalRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.sink = Mock()
        self.detector = Mock()
        self.detector.classify.return_value = Verdict("fall", 0.8)
        self.pipe = Pipeline(self.sink, self.detector)
        for link in ("A", "B"):
            tracker = self.pipe.trackers[link] = MotionTracker(link)
            tracker.load_calibration({"measure": MEASURE, "size": 2,
                                      "mask": [0, 1], "baseline": 0.01})
        self.runtime = TerminalRuntime(self.pipe, links=("A", "B"))
        self.addCleanup(self.runtime.stop)
        self.wall = 100.0
        clock = patch("backend.dashboard.time.monotonic", side_effect=lambda: self.wall)
        clock.start()
        self.addCleanup(clock.stop)
        for link in ("A", "B"):
            self.runtime.process(Frame(10.0, -45, np.ones(2), link))

    def candidate(self):
        event = MotionEvent(start=9.0, end=10.0, t=np.array([8.0, 9.0, 10.0]),
                            energy=np.array([1.0, 15.0, 1.0]), amp=np.ones((3, 2)),
                            cut_short=False, link="A")
        self.pipe.pending, self.pipe.links, self.pipe.deadline = event, "A+B", 14.0
        self.pipe.flow.candidate(10.0, 14.0)

    def test_stale_participant_discards_candidate_while_other_receiver_continues(self):
        self.candidate()
        # A remains fresh; B disappears after its event finished. The terminal
        # must discard the shared candidate before A can reach the RF deadline.
        for elapsed in (1.0, 2.0, 3.0, 4.1):
            self.wall = 100.0 + elapsed
            self.runtime.process(Frame(10.0 + elapsed, -45, np.ones(2), "A"))
        self.assertIsNone(self.pipe.pending)
        self.assertEqual(self.pipe.flow.state, IDLE)
        self.assertIsNone(self.pipe.trackers["B"].baseline)
        self.detector.classify.assert_not_called()

    def test_returning_receiver_recalibrates_before_processing_old_candidate(self):
        self.candidate()
        self.wall += 5.0
        info, result = self.runtime.process(Frame(15.0, -45, np.ones(2), "A"))
        self.assertEqual(info.level, CALIBRATING)
        self.assertIsNone(result)
        self.assertIsNone(self.pipe.pending)
        self.detector.classify.assert_not_called()

    def test_wall_tick_discards_candidate_without_classifying_missing_data(self):
        self.candidate()
        self.assertEqual(self.pipe.flow.state, FALL_CANDIDATE)
        self.wall += 5.0
        self.runtime.tick()
        self.assertEqual(self.pipe.flow.state, IDLE)
        self.assertTrue(all(tracker.baseline is None for tracker in self.pipe.trackers.values()))
        self.detector.classify.assert_not_called()

    def test_confirmed_countdown_alerts_once_when_receivers_stop(self):
        self.candidate()
        self.pipe.flow.confirm(10.0)
        self.assertEqual(self.pipe.flow.state, COUNTDOWN)
        self.wall += 11.0
        self.runtime.tick()
        self.runtime.tick()
        self.assertEqual(self.pipe.flow.state, ALERTED)
        self.sink.ask_resident.assert_called_once_with(10.0)
        self.sink.alert_caregiver.assert_called_once_with(21.0)

    def test_cancelled_countdown_does_not_alert_after_sensor_loss(self):
        self.candidate()
        self.pipe.flow.confirm(10.0)
        self.wall += 3.0
        self.runtime.tick()
        self.assertEqual(self.pipe.flow.state, COUNTDOWN)
        with self.runtime.lock:
            self.pipe.flow.cancel(self.runtime.clock.now(), "the resident")
        self.wall += 10.0
        self.runtime.tick()
        self.assertEqual(self.pipe.flow.state, IDLE)
        self.sink.alert_caregiver.assert_not_called()

    def test_start_is_idempotent_and_stop_joins_the_ticker(self):
        self.runtime.start()
        first = self.runtime.thread
        self.runtime.start()
        self.assertIs(self.runtime.thread, first)
        self.assertTrue(first.is_alive())
        self.runtime.stop()
        self.assertFalse(first.is_alive())

    def test_live_capture_age_is_preserved_by_clock_and_sensor_health(self):
        runtime = TerminalRuntime(self.pipe, links=("A",), live_age=True)
        self.addCleanup(runtime.stop)
        with patch("backend.live_runtime.time.time", return_value=10.4):
            info, _ = runtime.process(Frame(10.1, -45, np.ones(2), "A"))
        self.assertNotEqual(info.reason, "stale_capture")
        self.assertAlmostEqual(runtime.clock.now(), 10.4)
        entry = runtime.sensors.seen["A"]
        self.assertAlmostEqual(entry["valid_at"], self.wall - 0.3)
        self.wall += 0.1
        self.assertAlmostEqual(self.wall - entry["valid_at"], 0.4)

    def test_receiver_restart_resets_calibration_before_stale_timeout(self):
        self.candidate()
        self.runtime.capture_health({"reset_links": ["A"], "capture_wall_t": 10.1,
                                     "dropped_since_report": 0, "idle": False})
        self.assertIsNone(self.pipe.pending)
        self.assertEqual(self.pipe.flow.state, IDLE)
        self.assertIsNone(self.pipe.trackers["A"].baseline)
        self.assertIsNotNone(self.pipe.trackers["B"].baseline)
        self.detector.classify.assert_not_called()

    def test_receipt_age_and_source_lag_combine_without_changing_rf_clock(self):
        runtime = TerminalRuntime(self.pipe, links=("A",), live_age=True)
        self.addCleanup(runtime.stop)
        self.candidate()
        # A frame was read recently but had already accumulated transport delay.
        frame = Frame(10.1, -45, np.ones(2), "A", received_mono=99.7, source_lag_s=.3)
        with patch("backend.live_runtime.time.time", return_value=10.1):
            info, result = runtime.process(frame)
        self.assertEqual(info.reason, "stale_capture")
        self.assertIsNone(result)
        self.assertIsNone(self.pipe.pending)
        self.detector.classify.assert_not_called()
        self.assertEqual(frame.t, 10.1)


if __name__ == "__main__":
    unittest.main()
