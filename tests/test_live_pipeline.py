"""Live packet failures must not become motion or stillness evidence."""
import unittest
from unittest.mock import Mock, patch

import numpy as np

from backend import config
from backend.capture.csi import Frame, parse_line
from backend.detection.detector import Verdict
from backend.detection.fallflow import ALERTED, COUNTDOWN, FALL_CANDIDATE, IDLE
from backend.detection.motion import CALIBRATING, MEASURE, MOVING, MotionEvent, MotionTracker, calibration_from_frames
from backend.detection.pipeline import Pipeline


def calibration(**updates):
    return {"measure": MEASURE, "size": 2, "mask": [0, 1], "baseline": 0.01, **updates}


def event(link, peak=15.0):
    return MotionEvent(start=1.0, end=2.0, t=np.array([0.0, 1.0, 2.0]),
                       energy=np.array([1.0, peak, 1.0]), amp=np.ones((3, 2)),
                       cut_short=False, link=link)


class PacketTests(unittest.TestCase):
    def test_valid_iq_pairs_and_invalid_first_word(self):
        frame = parse_line('CSI_DATA,0,0,-45,1,"[3,4,5,12,8,15]"', 123.0, "B")
        self.assertEqual((frame.t, frame.rssi, frame.link), (123.0, -45, "B"))
        np.testing.assert_array_equal(frame.amp, [0.0, 0.0, 17.0])

    def test_corrupt_serial_rows_are_dropped_instead_of_truncating_iq(self):
        rows = [
            'CSI_DATA,0,0,-45,0,"[]"',
            'CSI_DATA,0,0,-45,0,"[1,2,3]"',
            'CSI_DATA,0,0,-45,0,"[1,,2]"',
            'CSI_DATA,0,0,-45,0,"1,2"',
            'CSI_DATA,0,0,-45,2,"[1,2]"',
            'CSI_DATAgarbage,0,0,-45,0,"[1,2]"',
            'CSI_DATA,0,0,-45,0,"[1,2]',
        ]
        for row in rows:
            with self.subTest(row=row):
                self.assertIsNone(parse_line(row, 1.0))
        self.assertIsNone(parse_line('CSI_DATA,0,0,-45,0,"[1,2]"', float("nan")))

    def test_zero_signal_or_constant_signal_cannot_calibrate(self):
        for amp in (np.zeros(2), np.ones(2)):
            frames = [Frame(i * 0.05, -40, amp) for i in range(100)]
            self.assertIsNone(calibration_from_frames(frames))

    def test_bad_calibration_is_rejected_before_tracker_state_changes(self):
        tracker = MotionTracker()
        tracker.load_calibration(calibration())
        for updates in ({"baseline": 0}, {"baseline": float("inf")}, {"mask": []},
                        {"mask": [2]}, {"mask": [0, 0]}, {"size": 0}):
            with self.subTest(updates=updates):
                with self.assertRaises(ValueError):
                    tracker.load_calibration(calibration(**updates))
                self.assertEqual(tracker.baseline, 0.01)

    def test_failed_live_calibration_keeps_only_recent_frames_and_bounds_retries(self):
        tracker = MotionTracker()
        attempts = []

        def calibrate(frames):
            attempts.append(frames[-1].t)
            return calibration_from_frames(frames)

        with patch("backend.detection.motion.calibration_from_frames", side_effect=calibrate):
            for i in range(701):
                tracker.update(Frame(i * 0.05, -45, np.ones(2)))
        self.assertEqual(tracker.level, CALIBRATING)
        self.assertIsNone(tracker.baseline)
        self.assertLessEqual(len(tracker.cal_frames), 202)
        self.assertGreaterEqual(tracker.cal_frames[0].t, 35.0 - config.CALIBRATE_S - 0.05)
        self.assertEqual(attempts[0], config.CALIBRATE_S)
        self.assertGreater(len(attempts), 2)
        self.assertTrue(all(b - a >= 0.5 - 1e-9 for a, b in zip(attempts, attempts[1:])))

    def test_initial_valid_calibration_uses_original_ten_second_window(self):
        rng = np.random.default_rng(7)
        frames = [Frame(i * 0.05, -45, 10 + rng.normal(0, 0.1, size=2)) for i in range(201)]
        expected = calibration_from_frames(frames)
        tracker = MotionTracker()
        for frame in frames:
            tracker.update(frame)
        self.assertEqual(tracker.calibration(), expected)
        self.assertIsNone(tracker.cal_retry_at)


class LiveSignalLossTests(unittest.TestCase):
    def setUp(self):
        self.sink = Mock()
        self.detector = Mock()
        self.detector.name = "test-rf"
        self.detector.fall_threshold = None
        self.pipe = Pipeline(self.sink, self.detector)
        for link in ("A", "B"):
            tracker = self.pipe.trackers[link] = MotionTracker(link)
            tracker.load_calibration(calibration())

    def candidate(self):
        self.pipe.ended = {"A": event("A", 20), "B": event("B", 15)}
        self.pipe._candidate(3.0)
        self.assertEqual(self.pipe.flow.state, FALL_CANDIDATE)
        self.assertEqual(self.pipe.pending.link, "A")

    def test_loss_of_any_participating_link_discards_candidate(self):
        self.candidate()
        self.pipe.after.append((3.0, np.ones(2)))
        self.assertTrue(self.pipe.suspend_link("B", 3.5))
        self.assertIsNone(self.pipe.pending)
        self.assertEqual(self.pipe.after, [])
        self.assertEqual(self.pipe.flow.state, IDLE)
        self.pipe.process(Frame(7.0, -45, np.ones(2), "A"))
        self.detector.classify.assert_not_called()

    def test_missing_moving_link_preserves_healthy_event_but_needs_real_after_data(self):
        self.pipe.trackers["B"].level = MOVING
        self.pipe.ended = {"A": event("A")}
        self.pipe.suspend_link("B", 3.0)
        self.assertEqual(set(self.pipe.ended), {"A"})
        self.pipe.process(Frame(7.0, -45, np.ones(2), "A"))
        self.assertIsNone(self.pipe.pending)
        self.detector.classify.assert_not_called()
        self.assertTrue(any("insufficient_after_window" in call.args[1]
                            for call in self.sink.log.call_args_list))

    def test_returning_link_requires_fresh_calibration(self):
        self.assertTrue(self.pipe.suspend_link("A", 3.0))
        self.assertFalse(self.pipe.suspend_link("A", 3.1))
        info, result = self.pipe.process(Frame(4.0, -45, np.ones(2), "A"))
        self.assertEqual(info.level, CALIBRATING)
        self.assertIsNone(info.energy)
        self.assertIsNone(result)
        self.assertIsNone(self.pipe.trackers["A"].baseline)
        self.assertEqual(len(self.pipe.trackers["A"].cal_frames), 1)

    def test_decision_requires_valid_selected_link_amplitude_after_deadline(self):
        self.candidate()
        self.detector.classify.return_value = Verdict("sit-down", 0.7)
        # Neither another receiver nor an A frame with a changed CSI dimension
        # supplies the after-event observation needed for A's static change.
        for frame in (Frame(6.1, -45, np.ones(2), "B"),
                      Frame(6.2, -45, np.ones(3), "A")):
            with self.subTest(frame=frame):
                _, result = self.pipe.process(frame)
                self.assertIsNone(result)
                self.assertEqual(self.pipe.flow.state, FALL_CANDIDATE)
                self.assertEqual(self.pipe.after, [])
                self.detector.classify.assert_not_called()
        _, result = self.pipe.process(Frame(6.3, -45, np.ones(2), "A"))
        self.assertIsNone(result)
        self.detector.classify.assert_not_called()
        self.assertEqual(self.pipe.flow.state, IDLE)
        self.assertTrue(any("insufficient_after_window" in call.args[1]
                            for call in self.sink.log.call_args_list))

    def test_continuous_selected_link_after_window_reaches_classifier(self):
        self.candidate()
        self.detector.classify.return_value = Verdict("sit-down", 0.7)
        result = None
        with patch("backend.detection.motion.raw_energy", return_value=0.01):
            for t in np.arange(3.0, 6.11, 0.1):
                _, current = self.pipe.process(Frame(round(float(t), 6), -45, np.ones(2), "A"))
                result = current or result
        self.assertIsNotNone(result)
        self.detector.classify.assert_called_once()
        self.assertEqual(self.pipe.flow.state, IDLE)

    def test_confirmed_countdown_still_alerts_when_sensor_is_lost(self):
        self.candidate()
        self.pipe.flow.confirm(6.0)
        deadline = self.pipe.flow.deadline
        self.pipe.suspend_link("A", 7.0)
        self.assertEqual(self.pipe.flow.state, COUNTDOWN)
        self.assertEqual(self.pipe.flow.deadline, deadline)
        self.pipe.flow.tick(deadline)
        self.assertEqual(self.pipe.flow.state, ALERTED)
        self.sink.alert_caregiver.assert_called_once_with(deadline)

    def test_recalibration_discards_candidate_without_leaving_flow_stuck(self):
        self.candidate()
        self.pipe.calibrate(3.5)
        self.assertEqual(self.pipe.flow.state, IDLE)
        self.assertIsNone(self.pipe.pending)
        self.assertTrue(all(t.level == CALIBRATING for t in self.pipe.trackers.values()))


if __name__ == "__main__":
    unittest.main()
