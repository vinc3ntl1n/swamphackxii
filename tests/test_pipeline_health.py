"""Real packet gaps must not masquerade as motion or post-event stillness."""
import json
import unittest
from unittest.mock import patch

import numpy as np

from backend.capture.csi import Frame
from backend.detection.detector import Verdict
from backend.detection.fallflow import ALERTED, COUNTDOWN, IDLE
from backend.detection.motion import MOVING, STALE, WARMING, MotionTracker
from backend.detection.pipeline import Pipeline


CAL = {"measure": "mean-shift-0.5", "size": 2, "mask": [0, 1], "baseline": 1.0}


class Sink:
    def __init__(self):
        self.logs = []

    def log(self, t, text):
        self.logs.append((t, text))

    def alarm(self, on):
        pass

    def ask_resident(self, t):
        pass

    def alert_caregiver(self, t):
        pass

    def diagnostics(self):
        prefix = "Detection diagnostic: "
        return [json.loads(text[len(prefix):]) for _, text in self.logs if text.startswith(prefix)]

    def reasons(self):
        return [row["reason"] for row in self.diagnostics()]


class Detector:
    name = "test-rf"
    fall_threshold = 0.35

    def __init__(self):
        self.inputs = []

    def classify(self, x):
        self.inputs.append(x)
        return Verdict("fall", 0.9, {"fall_probability": 0.9})


def frame(t, link="A", size=2):
    return Frame(round(t, 6), -40, np.full(size, 10.0), link)


def motion_energy(t, amps, now):
    return 20.0 if 2.0 <= round(now, 6) < 2.5 else 1.0


class PipelineHealthTests(unittest.TestCase):
    def make_pipeline(self, *links):
        sink, detector = Sink(), Detector()
        pipe = Pipeline(sink, detector)
        for link in links:
            tracker = pipe.trackers[link] = MotionTracker(link)
            tracker.load_calibration(CAL)
        return pipe, sink, detector

    def feed(self, pipe, start, stop, link="A"):
        for t in np.arange(start, stop + 0.001, 0.1):
            pipe.process(frame(t, link))

    def test_fresh_fall_keeps_existing_features_and_countdown(self):
        pipe, sink, detector = self.make_pipeline("A")
        with patch("backend.detection.motion.raw_energy", side_effect=motion_energy):
            self.feed(pipe, 0, 7)
        self.assertEqual(len(detector.inputs), 1)
        np.testing.assert_allclose(detector.inputs[0], [20, 0, 0.5, 101 / 6, 0, 0.5])
        self.assertEqual(pipe.flow.state, COUNTDOWN)
        classified = next(row for row in sink.diagnostics() if row["reason"] == "classified")
        self.assertEqual(classified["fall_probability"], 0.9)
        self.assertEqual(classified["fall_threshold"], 0.35)
        self.assertEqual(len(classified["features"]), 6)
        self.assertGreaterEqual(classified["classify_ms"], 0)

    def test_missing_energy_never_reuses_previous_motion_energy(self):
        tracker = MotionTracker()
        tracker.load_calibration(CAL)
        with patch("backend.detection.motion.raw_energy", return_value=20):
            tracker.update(frame(0))
        self.assertEqual(tracker.level, MOVING)
        with patch("backend.detection.motion.raw_energy", return_value=None):
            info, event = tracker.update(frame(0.1))
        self.assertIsNone(info.energy)
        self.assertEqual(info.reason, "insufficient_window")
        self.assertIsNone(event)
        self.assertEqual(tracker.last_valid_t, 0.1)
        self.assertEqual(tracker.last_energy_t, 0)
        self.assertEqual(len(tracker.history), 1)

    def test_real_amplitudes_during_initial_warmup_remain_before_reference(self):
        pipe, _, detector = self.make_pipeline("A")

        def initial_warmup(t, amps, now):
            if now < 0.5:
                return None
            return 20 if 1.2 <= round(now, 6) < 1.7 else 1

        with patch("backend.detection.motion.raw_energy", side_effect=initial_warmup):
            self.feed(pipe, 0, 6)
        self.assertEqual(len(detector.inputs), 1)
        self.assertEqual(detector.inputs[0][1], 0)
        self.assertEqual(pipe.flow.state, COUNTDOWN)

    def test_continuing_valid_packets_without_energy_cannot_latch_moving(self):
        tracker = MotionTracker()
        tracker.load_calibration(CAL)
        with patch("backend.detection.motion.raw_energy", return_value=20):
            tracker.update(frame(0))
        with patch("backend.detection.motion.raw_energy", return_value=None):
            for t in np.arange(0.1, 0.61, 0.1):
                tracker.update(frame(t))
        self.assertEqual(tracker.level, WARMING)
        self.assertIsNone(tracker.event_start)
        self.assertTrue(all(energy == 0 for _, energy, _ in tracker.history))
        self.assertEqual(tracker.last_valid_t, 0.6)

    def test_invalid_dimensions_do_not_keep_receiver_fresh(self):
        pipe, sink, detector = self.make_pipeline("A")
        with patch("backend.detection.motion.raw_energy", return_value=20):
            pipe.process(frame(0))
            for t in np.arange(0.1, 2.2, 0.1):
                info, _ = pipe.process(frame(t, size=3))
        self.assertIsNone(info.energy)
        self.assertEqual(pipe.trackers["A"].last_valid_t, 0)
        self.assertEqual(pipe.trackers["A"].level, STALE)
        self.assertEqual(pipe.health(2.1)["A"]["status"], "stale")
        self.assertIn("wrong_frame_size", sink.reasons())
        self.assertIn("receiver_stale", sink.reasons())
        self.assertFalse(detector.inputs)

    def test_stale_moving_receiver_no_longer_blocks_healthy_fall(self):
        pipe, sink, detector = self.make_pipeline("A", "B")
        with patch("backend.detection.motion.raw_energy", side_effect=motion_energy):
            for t in np.arange(0, 7.01, 0.1):
                pipe.process(frame(t, "A"))
                if t <= 2.01:
                    pipe.process(frame(t, "B"))
        self.assertIn("waiting_for_quiet", sink.reasons())
        self.assertIn("receiver_stale", sink.reasons())
        self.assertEqual(pipe.trackers["B"].level, STALE)
        self.assertEqual(len(detector.inputs), 1)
        self.assertEqual(pipe.flow.state, COUNTDOWN)
        started = next(row for row in sink.diagnostics() if row["reason"] == "candidate_started")
        self.assertEqual(started["degraded_links"], ["B"])

    def test_selected_receiver_dropout_aborts_without_classifying_empty_after(self):
        pipe, sink, detector = self.make_pipeline("A", "B")
        with patch("backend.detection.motion.raw_energy", side_effect=motion_energy):
            self.feed(pipe, 0, 3.6)
        self.assertIsNotNone(pipe.pending)
        with patch("backend.detection.motion.raw_energy", return_value=1):
            self.feed(pipe, 3.7, 8, "B")
        self.assertFalse(detector.inputs)
        self.assertIsNone(pipe.pending)
        self.assertEqual(pipe.flow.state, IDLE)
        self.assertIn("selected_receiver_stale", sink.reasons())

    def test_packet_gap_discards_incomplete_candidate_and_recovers(self):
        pipe, sink, detector = self.make_pipeline("A")
        with patch("backend.detection.motion.raw_energy", side_effect=motion_energy):
            self.feed(pipe, 0, 3.6)
        self.assertIsNotNone(pipe.pending)
        with patch("backend.detection.motion.raw_energy", return_value=None):
            info, _ = pipe.process(frame(4.2))
        self.assertIsNone(pipe.pending)
        self.assertIsNone(info.energy)
        self.assertEqual(pipe.trackers["A"].level, WARMING)
        self.assertEqual(pipe.flow.state, IDLE)
        self.assertIn("selected_receiver_packet_gap", sink.reasons())
        self.assertFalse(detector.inputs)
        self.assertEqual(pipe.health(4.2)["A"]["status"], "warming")
        with patch("backend.detection.motion.raw_energy",
                   side_effect=lambda t, amps, now: motion_energy(t, amps, now - 5)):
            self.feed(pipe, 4.3, 12)
        self.assertEqual(len(detector.inputs), 1)
        self.assertEqual(pipe.flow.state, COUNTDOWN)

    def test_stall_tick_does_not_fabricate_stillness(self):
        pipe, sink, detector = self.make_pipeline("A")
        with patch("backend.detection.motion.raw_energy", side_effect=motion_energy):
            self.feed(pipe, 0, 3.6)
        pipe.tick(9)
        self.assertEqual(pipe.trackers["A"].level, STALE)
        self.assertIsNone(pipe.pending)
        self.assertEqual(pipe.flow.state, IDLE)
        self.assertFalse(detector.inputs)
        self.assertIn("selected_receiver_stale", sink.reasons())

    def test_explicit_capture_loss_aborts_even_without_large_timestamp_gap(self):
        pipe, sink, detector = self.make_pipeline("A")
        with patch("backend.detection.motion.raw_energy", side_effect=motion_energy):
            self.feed(pipe, 0, 3.6)
        pipe.capture_gap(3.65, ["A"])
        self.assertIsNone(pipe.pending)
        self.assertEqual(pipe.flow.state, IDLE)
        self.assertEqual(pipe.trackers["A"].level, WARMING)
        self.assertFalse(pipe.trackers["A"].window)
        self.assertFalse(detector.inputs)
        self.assertIn("selected_receiver_capture_gap", sink.reasons())

    def test_capture_loss_does_not_reset_confirmed_countdown(self):
        pipe, _, detector = self.make_pipeline("A")
        with patch("backend.detection.motion.raw_energy", side_effect=motion_energy):
            self.feed(pipe, 0, 7)
        self.assertEqual(len(detector.inputs), 1)
        pipe.capture_gap(7.1)
        self.assertEqual(pipe.flow.state, COUNTDOWN)

    def test_insufficient_after_window_is_explicit_not_zero_feature(self):
        pipe, sink, detector = self.make_pipeline("A")
        with patch("backend.detection.motion.raw_energy", side_effect=motion_energy):
            self.feed(pipe, 0, 6.4)
            pipe.after.clear()
            pipe.process(frame(6.5))
        self.assertFalse(detector.inputs)
        self.assertIn("insufficient_after_window", sink.reasons())
        self.assertEqual(pipe.flow.state, IDLE)

    def test_resumed_movement_policy_is_preserved_and_logged(self):
        pipe, sink, detector = self.make_pipeline("A")
        with patch("backend.detection.motion.raw_energy", side_effect=motion_energy):
            self.feed(pipe, 0, 3.6)
        with patch("backend.detection.motion.raw_energy", return_value=20):
            _, result = pipe.process(frame(3.7))
        self.assertIsNotNone(result)
        self.assertEqual(result[1].details["reason"], "resumed_movement")
        self.assertFalse(detector.inputs)
        self.assertEqual(pipe.flow.state, IDLE)
        self.assertIn("resumed_movement", sink.reasons())

    def test_unresolved_alert_policy_is_preserved_and_logged(self):
        pipe, sink, detector = self.make_pipeline("A")
        pipe.flow.state = ALERTED
        with patch("backend.detection.motion.raw_energy", side_effect=motion_energy):
            self.feed(pipe, 0, 7)
        self.assertEqual(pipe.flow.state, ALERTED)
        self.assertFalse(detector.inputs)
        blocked = next(row for row in sink.diagnostics() if row["reason"] == "blocked_alert")
        self.assertEqual(blocked["state"], ALERTED)


if __name__ == "__main__":
    unittest.main()
