"""Live calibration excludes walk-away packets without changing its learned baseline."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from backend import config, dashboard, server
from backend.capture.csi import Frame
from backend.detection.detector import ThresholdDetector
from backend.detection.fallflow import COUNTDOWN, IDLE
from backend.detection.motion import CALIBRATING, MEASURE, MotionEvent, calibration_from_frames
from backend.detection.pipeline import Pipeline
from backend.live_runtime import TerminalRuntime


class DashboardRecalibrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.hub = server.Hub()
        history = dashboard.History(self.hub, "live", Path(temporary.name) / "history.jsonl")
        web = dashboard.WebSink(self.hub, history, dashboard.Notices(self.hub, history))
        self.pipe = Pipeline(web, ThresholdDetector())
        self.view = dashboard.Dashboard(self.hub, self.pipe, web, web, links=("A", "B"))
        dashboard.preload(self.pipe, {link: {
            "measure": MEASURE, "size": 4, "mask": [0, 1, 2, 3], "baseline": .1,
        } for link in ("A", "B")})
        self.wall = 100.
        patched = patch("backend.dashboard.time.monotonic", side_effect=lambda: self.wall)
        patched.start()
        self.addCleanup(patched.stop)
        for link in ("A", "B"):
            self.view.process(Frame(10., -48, np.full(4, 20.), link))

    def sensor(self, link="A"):
        return json.loads(self.hub.sticky[f"sensor:{link}"][1])

    def feed(self, frame):
        self.wall = 100. + frame.t - 10.
        return self.view.process(frame)

    def start_calibration(self):
        self.assertTrue(self.view.request_calibration())
        self.view._tick()

    def test_recalibration_resets_both_links_and_excludes_walk_away_packets(self):
        self.start_calibration()
        for tracker in self.pipe.trackers.values():
            self.assertIsNone(tracker.baseline)
        for i in range(1, 30):
            t = 10. + i / 10.
            for link in ("A", "B"):
                info, result = self.feed(Frame(t, -48, np.full(4, i * 10.), link))
                self.assertEqual(info.level, CALIBRATING)
                self.assertIsNone(info.energy)
                self.assertIsNone(result)
                self.assertEqual(self.pipe.trackers[link].cal_frames, [])
        self.view.sensors.refresh(12.9, force=True)
        for link in ("A", "B"):
            status = self.sensor(link)
            self.assertEqual(status["health"], "calibrating")
            self.assertEqual(status["calibration"]["phase"], "settling")
            self.assertEqual(status["calibration"]["progress"], 0.)
            self.assertAlmostEqual(status["calibration"]["settle_remaining_s"], .1)
        self.assertEqual(self.pipe.flow.state, IDLE)

    def test_http_request_queues_once_and_starts_settling(self):
        app = server.make_app(self.hub, self.view.on_command)
        with TestClient(app, base_url="http://127.0.0.1:80") as client:
            first = client.post("/calibrate", json={}, headers={"host": "127.0.0.1:80"})
            duplicate = client.post("/calibrate", json={}, headers={"host": "127.0.0.1:80"})
        self.assertEqual(first.status_code, 202)
        self.assertEqual(duplicate.status_code, 409)
        self.assertEqual(self.view.commands.qsize(), 1)
        self.view._tick()
        self.assertTrue(self.view.commands.empty())
        for link in ("A", "B"):
            self.assertEqual(self.sensor(link)["calibration"]["phase"], "settling")
            self.assertIsNone(self.pipe.trackers[link].baseline)

    def test_recalibration_learns_exact_ten_second_post_settle_window(self):
        self.start_calibration()
        rng = np.random.default_rng(37)
        first = 10. + config.CALIBRATE_SETTLE_S
        frames = {link: [] for link in ("A", "B")}
        for i in range(201):
            t = first + i * .05
            for link in ("A", "B"):
                frame = Frame(t, -48, 20. + rng.normal(0, .1, size=4), link)
                frames[link].append(frame)
                self.feed(frame)
            if i == 100:
                self.view.sensors.refresh(t, force=True)
                for link in ("A", "B"):
                    status = self.sensor(link)
                    self.assertEqual(status["calibration"]["phase"], "learning")
                    self.assertEqual(status["calibration"]["progress"], .5)
                    self.assertIsNone(self.pipe.trackers[link].baseline)
        self.view.sensors.refresh(first + config.CALIBRATE_S, force=True)
        for link in ("A", "B"):
            self.assertEqual(self.pipe.trackers[link].calibration(), calibration_from_frames(frames[link]))
            self.assertEqual(self.sensor(link)["calibration"]["progress"], 1.)
            self.assertEqual(self.sensor(link)["calibration"]["phase"], "ready")

    def test_request_rejects_missing_live_signal_and_recorded_replay(self):
        self.wall += 3.
        self.assertFalse(self.view.request_calibration())
        self.assertTrue(all(tracker.baseline == .1 for tracker in self.pipe.trackers.values()))
        self.wall -= 3.
        self.view.sensors.source = "REPLAY"
        self.assertFalse(self.view.request_calibration())
        self.assertTrue(self.view.commands.empty())

    def test_request_does_not_interrupt_a_confirmed_fall(self):
        event = MotionEvent(9., 10., np.array([9., 10.]), np.array([15., 2.]),
                            np.full((2, 4), 20.), False, "A")
        self.pipe.pending, self.pipe.links, self.pipe.deadline = event, "A+B", 14.
        self.pipe.flow.candidate(10., 14.)
        self.pipe.flow.confirm(10.)
        self.assertFalse(self.view.request_calibration())
        self.view._tick()
        self.assertEqual(self.pipe.flow.state, COUNTDOWN)
        self.assertTrue(all(tracker.baseline == .1 for tracker in self.pipe.trackers.values()))

    def test_request_is_disabled_while_simulation_plays(self):
        self.view._command({"cmd": "simulate"})
        self.assertIsNotNone(self.view.sim)
        self.assertFalse(self.view.request_calibration())
        self.assertTrue(all(tracker.baseline == .1 for tracker in self.pipe.trackers.values()))

    def test_queued_request_rechecks_source_and_active_countdown(self):
        self.assertTrue(self.view.request_calibration())
        self.assertFalse(self.view.request_calibration())
        self.view.sensors.source = "REPLAY"
        self.view._tick()
        self.assertTrue(all(tracker.baseline == .1 for tracker in self.pipe.trackers.values()))

        self.view.sensors.source = "LIVE"
        self.assertTrue(self.view.request_calibration())
        self.pipe.flow.candidate(10., 14.)
        self.pipe.flow.confirm(10.)
        self.view._tick()
        self.assertEqual(self.pipe.flow.state, COUNTDOWN)
        self.assertTrue(all(tracker.baseline == .1 for tracker in self.pipe.trackers.values()))
        self.assertIsNone(self.view.sensors.calibrate_until)

    def test_live_startup_settles_before_collecting_in_dashboard_and_terminal(self):
        for kind in ("dashboard", "terminal"):
            with self.subTest(runtime=kind):
                pipe = Pipeline(self.pipe.sink, ThresholdDetector())
                runtime = (dashboard.Dashboard(server.Hub(), pipe, self.pipe.sink, self.view.web,
                                               links=("A", "B"))
                           if kind == "dashboard" else TerminalRuntime(pipe, links=("A", "B")))
                for t, link in ((30., "A"), (32.9, "B")):
                    self.wall = 200. + t - 30.
                    info, result = runtime.process(Frame(t, -48, np.full(4, 50.), link))
                    self.assertEqual(info.level, CALIBRATING)
                    self.assertIsNone(info.energy)
                    self.assertIsNone(result)
                    self.assertEqual(pipe.trackers[link].cal_frames, [])
                for t, link in ((33., "A"), (33.1, "B")):
                    self.wall = 200. + t - 30.
                    runtime.process(Frame(t, -48, np.full(4, 20.), link))
                    self.assertEqual([frame.t for frame in pipe.trackers[link].cal_frames], [t])


if __name__ == "__main__":
    unittest.main()
