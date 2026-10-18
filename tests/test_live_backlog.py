from __future__ import annotations

import io
import json
import queue
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from backend.capture import live
from backend.capture.csi import Frame, read_log


LINE = 'CSI_DATA,0,0,-45,0,"[0,30,0,31,0,32,0,33]"'


def packet(t: float, received_mono: float, link: str = "A") -> live._Packet:
    return live._Packet(Frame(t, -45, np.ones(4), link), LINE, received_mono)


class PacketBufferTests(unittest.TestCase):
    def test_overflow_keeps_latest_packets_but_logs_every_raw_row(self):
        log = io.StringIO()
        buf = live._PacketBuffer(2, 0.5, log)
        with patch.object(live.time, "monotonic", return_value=10.0):
            for t in (100.0, 101.0, 102.0, 103.0):
                buf.put(packet(t, 10.0))
            self.assertEqual(buf.queue.qsize(), 2)
            self.assertEqual(buf.get(0).frame.t, 102.0)
            self.assertEqual(buf.get(0).frame.t, 103.0)
        self.assertEqual(buf.dropped, {"queue_overflow": 2})
        self.assertEqual([float(row.split("\t")[0]) for row in log.getvalue().splitlines()],
                         [100.0, 101.0, 102.0, 103.0])

    def test_age_cutoff_uses_monotonic_clock_and_reports_drop_reason(self):
        reports = []
        buf = live._PacketBuffer(5, 0.5, health_callback=reports.append)
        # Wall-clock changes do not make an old packet fresh.
        buf.put(packet(999999.0, 5.0))
        buf.put(packet(1.0, 5.8))
        with patch.object(live.time, "monotonic", return_value=6.0), \
                self.assertLogs(live.LOGGER, level="WARNING") as captured:
            self.assertIsNone(buf.get(0))
            fresh = buf.get(0)
            self.assertEqual(fresh.frame.t, 999999.0)  # ordering remains nondecreasing
            buf.report()
        self.assertEqual(reports[0]["dropped_frames"], 1)
        self.assertEqual(reports[0]["drop_reasons"], {"packet_age": 1})
        self.assertEqual(reports[0]["dropped_links"], ["A"])
        self.assertFalse(reports[0]["idle"])
        self.assertEqual(reports[0]["capture_age_ms"], 200.0)
        self.assertEqual(reports[0]["last_drop"]["link"], "A")
        self.assertEqual(json.loads(captured.records[0].getMessage())["type"], "capture_health")

    def test_lossless_buffer_has_no_queue_or_age_limit(self):
        buf = live._PacketBuffer(0, None)
        for t in range(300):
            buf.put(packet(float(t), 1.0))
        with patch.object(live.time, "monotonic", return_value=1000.0):
            result = [buf.get(0).frame.t for _ in range(300)]
        self.assertEqual(result, list(range(300)))
        self.assertEqual(buf.dropped, {})

    def test_health_reporting_is_throttled_and_includes_queue_depth(self):
        reports = []
        buf = live._PacketBuffer(2, 0.5, health_callback=reports.append)
        buf.put(packet(100, 10))
        with patch.object(live.time, "monotonic", return_value=10.0):
            buf.report()
            buf.report()
        with patch.object(live.time, "monotonic", return_value=11.0), \
                patch.object(live.time, "time", return_value=12345.0):
            buf.report(idle=True)
        self.assertEqual(len(reports), 2)
        self.assertEqual(reports[0]["queue_depth"], 1)
        self.assertEqual(reports[0]["queue_capacity"], 2)
        self.assertFalse(reports[0]["idle"])
        self.assertTrue(reports[1]["idle"])
        self.assertEqual(reports[1]["capture_wall_t"], 12345.0)

    def test_drop_report_is_prompt_and_accumulates_links_since_last_report(self):
        reports = []
        buf = live._PacketBuffer(1, 0.5, health_callback=reports.append)
        with patch.object(live.time, "monotonic", return_value=10.0):
            buf.report()
            buf.put(packet(100, 10, "A"))
            buf.put(packet(101, 10, "B"))
            buf.put(packet(102, 10, "A"))
        with patch.object(live.time, "monotonic", return_value=10.0), \
                self.assertLogs(live.LOGGER, level="WARNING"):
            buf.report()
        self.assertEqual(len(reports), 2)
        self.assertEqual(reports[1]["dropped_since_report"], 2)
        self.assertEqual(reports[1]["dropped_links"], ["A", "B"])
        with patch.object(live.time, "monotonic", return_value=11.2):
            buf.report(idle=True)
        self.assertEqual(reports[2]["dropped_links"], [])
        self.assertEqual(reports[2]["dropped_since_report"], 0)


class LiveCaptureTests(unittest.TestCase):
    def test_age_drops_are_batched_and_reported_before_next_kept_frame(self):
        stop = threading.Event()
        reports = []

        def producer(link, port, out, stop):
            out.put(packet(100, 9, "A"))
            out.put(packet(101, 9, "B"))
            out.put(packet(102, 10, "A"))
            stop.wait()

        with patch.object(live, "_read_port", producer), \
                patch.object(live.time, "monotonic", return_value=10.0), \
                self.assertLogs(live.LOGGER, level="WARNING") as captured:
            frames = live.live_frames({"A": "fake"}, stop=stop, health_callback=reports.append)
            try:
                frame = next(frames)
                self.assertEqual(frame.t, 102)
                self.assertEqual(len(reports), 1)
                self.assertEqual(reports[0]["dropped_since_report"], 2)
                self.assertEqual(reports[0]["dropped_links"], ["A", "B"])
                self.assertFalse(reports[0]["idle"])
                self.assertEqual(len(captured.records), 1)
            finally:
                frames.close()

    def test_live_frames_catches_up_preserves_raw_log_and_parses_only_once(self):
        stop, drained = threading.Event(), threading.Event()
        serial_rows: queue.Queue[bytes] = queue.Queue()
        serial_rows.put(b"boot message\n")
        for _ in range(8):
            serial_rows.put((LINE + "\n").encode("ascii"))
        closed = []

        class Serial:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                closed.append(True)

            def readline(self):
                try:
                    return serial_rows.get_nowait()
                except queue.Empty:
                    drained.set()
                    stop.wait(0.01)
                    return b""

        original_get = live._PacketBuffer.get

        def get_after_producer(buf, timeout):
            self.assertTrue(drained.wait(1), "serial reader did not finish the packet burst")
            return original_get(buf, timeout)

        with tempfile.TemporaryDirectory() as temp:
            log_path = Path(temp) / "session.log"
            reports = []
            with patch.object(live.serial, "Serial", Serial), \
                    patch.object(live, "LIVE_QUEUE_FRAMES", 3), \
                    patch.object(live._PacketBuffer, "get", get_after_producer), \
                    patch.object(live, "parse_line", wraps=live.parse_line) as parse, \
                    self.assertLogs(live.LOGGER, level="WARNING"):
                frames = live.live_frames({"B": "fake-port"}, log_path, stop=stop,
                                          health_callback=reports.append)
                try:
                    result = [next(frames) for _ in range(3)]
                finally:
                    frames.close()
                self.assertEqual(parse.call_count, 9)  # each serial line, no consumer reparse
            self.assertEqual([frame.link for frame in result], ["B"] * 3)
            self.assertEqual(len(list(read_log(log_path))), 8)
            self.assertEqual(reports[0]["dropped_frames"], 5)
            self.assertEqual(reports[0]["queue_capacity"], 3)
            self.assertEqual(reports[0]["dropped_links"], ["B"])
            self.assertFalse(reports[0]["idle"])
            self.assertTrue(stop.is_set())
            self.assertEqual(closed, [True])


if __name__ == "__main__":
    unittest.main()
