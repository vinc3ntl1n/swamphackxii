import contextlib
import io
import json
import logging
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from backend.capture.csi import Frame
from backend.diagnostics import CaptureLogHandler, DiagnosticSink, EventLog, RuntimeMonitor


class RuntimeDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name) / "diagnostics/events.jsonl"
        self.log = EventLog(self.path)
        self.addCleanup(self.log.close)

    def rows(self):
        return [json.loads(line) for line in self.path.read_text().splitlines()]

    def pipe(self):
        return SimpleNamespace(pending=None, after=[], deadline=0,
                               flow=SimpleNamespace(state="IDLE", started=None, deadline=None),
                               health=lambda t: {"A": {"status": "ok", "age_s": 0}})

    def test_threaded_writes_are_complete_json_and_close_is_idempotent(self):
        threads = [threading.Thread(target=lambda k=k: [self.log.write("sample", i, worker=k)
                                                        for i in range(25)]) for k in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.log.close()
        self.log.close()
        self.log.write("ignored")
        rows = self.rows()
        self.assertEqual(len(rows), 100)
        self.assertTrue(all(row["type"] == "sample" and "wall_time" in row for row in rows))
        self.assertEqual({(r["worker"], r["event_t"]) for r in rows},
                         {(k, i) for k in range(4) for i in range(25)})

    def test_detection_fields_and_sink_actions_are_preserved(self):
        sink = Mock()
        wrapper = DiagnosticSink(sink, self.log)
        wrapper.log(1.0, 'Detection diagnostic: {"reason":"resumed_movement","link":"B"}')
        wrapper.log(2.0, 'Detection diagnostic: {"reason":"classified","label":"fall",'
                          '"features":{"peak":12},"fall_probability":0.7}')
        wrapper.log(3.0, "Detection diagnostic: malformed")
        wrapper.alarm(True)
        wrapper.ask_resident(4.0)
        wrapper.alert_caregiver(5.0)
        rows = self.rows()
        self.assertEqual([r["type"] for r in rows],
                         ["detection", "detection", "log", "alarm", "ask_resident", "alert_caregiver"])
        self.assertEqual(rows[0]["reason"], "resumed_movement")
        self.assertEqual(rows[1]["fall_probability"], 0.7)
        self.assertEqual(rows[1]["features"], {"peak": 12})
        sink.log.assert_any_call(3.0, "Detection diagnostic: malformed")
        sink.alarm.assert_called_once_with(True)
        sink.ask_resident.assert_called_once_with(4.0)
        sink.alert_caregiver.assert_called_once_with(5.0)

    def test_write_failure_warns_once_and_forwarding_continues(self):
        self.log._file.close()
        sink = Mock()
        wrapper = DiagnosticSink(sink, self.log)
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            wrapper.log(1.0, "first")
            wrapper.log(2.0, "second")
            wrapper.alarm(False)
            self.log.close()
        self.assertEqual(stderr.getvalue().count("Runtime diagnostics disabled"), 1)
        self.assertEqual(sink.log.call_count, 2)
        sink.alarm.assert_called_once_with(False)

    def test_startup_file_errors_are_visible(self):
        with self.assertRaises(OSError):
            EventLog(self.path.parent)

    def test_runtime_samples_live_age_cost_health_and_idle_once_per_second(self):
        monitor = RuntimeMonitor(self.log, live=True)
        pipe = self.pipe()
        frame = Frame(98.0, -40, np.ones(2))
        with patch("backend.diagnostics.time.monotonic", return_value=10.0), \
                patch("backend.diagnostics.time.time", return_value=100.0):
            monitor.observe(pipe, frame, 2.0)
            monitor.observe(pipe, frame, 60.0)
            monitor.observe(pipe, frame, 3.0)
            monitor.tick(pipe, 99.0)
        with patch("backend.diagnostics.time.monotonic", return_value=11.0), \
                patch("backend.diagnostics.time.time", return_value=101.0):
            monitor.tick(pipe, 100.0)
        with patch("backend.diagnostics.time.monotonic", return_value=12.0), \
                patch("backend.diagnostics.time.time", return_value=102.0):
            monitor.tick(pipe, 101.0)
        rows = self.rows()
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["capture_age_ms"], 2000)
        self.assertEqual(rows[1]["capture_age_ms"], 3000)
        self.assertEqual(rows[1]["max_processing_ms"], 60)
        self.assertEqual(rows[1]["slow_frames"], 1)
        self.assertEqual(rows[1]["frames"], 2)
        self.assertEqual(rows[1]["receiver_health"]["A"]["status"], "ok")
        self.assertEqual(rows[2]["frames"], 0)
        self.assertIsNone(rows[2]["max_processing_ms"])

    def test_live_age_uses_monotonic_receipt_and_separates_relative_source_delay(self):
        monitor = RuntimeMonitor(self.log, live=True)
        frame = Frame(98.0, -40, np.ones(2), received_mono=10.0,
                      source_id=100, source_timestamp_us=123456, source_lag_s=.2)
        with patch("backend.diagnostics.time.monotonic", return_value=10.3), \
                patch("backend.diagnostics.time.time", return_value=999999.):
            monitor.observe(self.pipe(), frame, 3.0)
        row = self.rows()[0]
        self.assertAlmostEqual(row["capture_age_ms"], 300.)
        self.assertAlmostEqual(row["relative_source_age_ms"], 500.)
        self.assertEqual(row["capture_age_basis"], "monotonic_receipt")
        self.assertEqual(row["source_id"], 100)
        self.assertEqual(row["source_timestamp_us"], 123456)
        self.assertFalse(row["source_age_known"])

    def test_replay_never_treats_historical_time_as_capture_latency(self):
        monitor = RuntimeMonitor(self.log, live=False)
        pipe = self.pipe()
        pipe.pending = SimpleNamespace(link="A", start=10.0, end=11.0)
        pipe.deadline = 15.0
        pipe.after = [1, 2, 3]
        monitor.observe(pipe, SimpleNamespace(t=12.0), 3.0)
        row = self.rows()[0]
        self.assertNotIn("capture_age_ms", row)
        self.assertEqual(row["event_t"], 12.0)
        self.assertEqual(row["pending"]["after_frames"], 3)
        self.assertEqual(row["pending"]["deadline"], 15.0)

    def test_capture_handler_routes_structured_events_without_global_setup(self):
        handler = CaptureLogHandler(self.log)
        handler.emit(logging.LogRecord("capture", logging.WARNING, "", 0,
                                     '{"type":"receiver_disconnected","link":"B"}', (), None))
        handler.emit(logging.LogRecord("capture", logging.INFO, "", 0,
                                     '{"type":"capture_health","queue_depth":7}', (), None))
        handler.emit(logging.LogRecord("capture", logging.WARNING, "", 0,
                                     "ordinary capture message", (), None))
        rows = self.rows()
        self.assertEqual(rows[0]["type"], "receiver_disconnected")
        self.assertEqual(rows[1]["queue_depth"], 7)
        self.assertEqual(rows[2]["type"], "capture_log")


if __name__ == "__main__":
    unittest.main()
