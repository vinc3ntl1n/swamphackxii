from __future__ import annotations

import io
import csv
import threading
import unittest
from unittest.mock import patch

import numpy as np

from backend.capture import live
from backend.capture.csi import Frame


LINE = 'CSI_DATA,0,0,-45,0,"[0,30,0,31,0,32,0,33]"'


def packet(t, received_mono):
    return live._Packet(Frame(t, -45, np.ones(4), "A"), LINE, received_mono)


def source_row(source_id, source_us):
    fields = ["CSI_DATA", source_id, "1a:00:00:00:00:00", -45] + [0] * 21
    fields[18] = source_us
    fields[-3:] = [8, 0, "[0,30,0,31,0,32,0,33]"]
    output = io.StringIO()
    csv.writer(output, lineterminator="\n").writerow(fields)
    return output.getvalue().encode()


class ChunkSerial:
    """A finite serial stream; chunks deliberately split CSV rows."""

    def __init__(self, chunks, stop):
        self.chunks = list(chunks)
        self.stop = stop
        self.read_sizes = []
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    @property
    def in_waiting(self):
        return len(self.chunks[0]) if self.chunks else 0

    def read(self, size):
        self.read_sizes.append(size)
        if not self.chunks:
            self.stop.set()
            return b""
        result = self.chunks[0][:size]
        self.chunks[0] = self.chunks[0][size:]
        if not self.chunks[0]:
            self.chunks.pop(0)
        return result


class SerialStreamTests(unittest.TestCase):
    def read_stream(self, chunks, *, max_age=None):
        stop = threading.Event()
        stream = ChunkSerial(chunks, stop)
        raw_log = io.StringIO()
        out = live._PacketBuffer(0, max_age, raw_log)
        with patch.object(live.serial, "Serial", return_value=stream):
            live._read_port("A", "fake", out, stop)
        self.assertTrue(stream.closed)
        return stream, out, raw_log.getvalue()

    def test_chunked_reads_reassemble_rows_and_parse_once(self):
        chunks = [b"boot\nCSI_DATA,broken\n" + LINE[:15].encode(),
                  LINE[15:].encode() + b"\r\n" + LINE.encode() + b"\n"]
        with patch.object(live, "parse_line", wraps=live.parse_line) as parse:
            stream, out, raw = self.read_stream(chunks)
        self.assertEqual(parse.call_count, 4)
        self.assertEqual(out.queue.qsize(), 2)
        self.assertEqual(raw.count("\n"), 2)
        self.assertEqual(len(stream.read_sizes), 3)  # two chunks, one idle read
        self.assertGreater(stream.read_sizes[0], 1)
        stats = out.links["A"]
        self.assertEqual(stats["raw_lines"], 4)
        self.assertEqual(stats["csi_valid"], 2)
        self.assertEqual(stats["csi_malformed"], 1)
        self.assertEqual(stats["non_csi_lines"], 1)
        self.assertEqual(stats["read_timeouts"], 1)
        self.assertEqual(stats["bytes_received"], sum(map(len, chunks)))
        self.assertFalse(stats["source"]["source_available"])

    def test_oversized_partial_line_cannot_turn_suffix_into_valid_csi(self):
        with patch.object(live, "SERIAL_MAX_LINE_BYTES", 128):
            _, out, raw = self.read_stream([b"x" * 200, LINE.encode() + b"\n",
                                             LINE.encode() + b"\n"])
        self.assertEqual(out.queue.qsize(), 1)
        self.assertEqual(raw.count("\n"), 1)
        self.assertEqual(out.links["A"]["oversized_lines"], 1)
        self.assertEqual(out.links["A"]["raw_lines"], 2)

    def test_combined_source_lag_and_python_queue_age_expires_packet(self):
        out = live._PacketBuffer(2, 0.5)
        row = packet(100, 10)
        row.source_timing = {"source_lag_s": 0.3}
        out.put(row)
        with patch.object(live.time, "monotonic", return_value=10.3):
            self.assertIsNone(out.get(0))
        self.assertEqual(out.dropped, {"source_backlog": 1})

    def test_receiver_reset_invalidates_queued_detection_and_reports_before_delivery(self):
        reports = []
        out = live._PacketBuffer(4, 0.5, health_callback=reports.append)
        out.put(packet(100, 10))
        out.reset_link("A", "receiver_disconnected")
        out.put(packet(101, 10))
        with patch.object(live.time, "monotonic", return_value=10):
            self.assertIsNone(out.get(0))
            out.report()
            self.assertEqual(out.get(0).frame.t, 101)
        self.assertEqual(reports[0]["reset_links"], ["A"])
        self.assertEqual(reports[0]["dropped_links"], ["A"])

    def test_training_keeps_rows_across_source_reset_or_compression(self):
        out = live._PacketBuffer(0, None, io.StringIO())
        out.put(packet(100, 1))
        out.reset_link("A", "source_timestamp_reset")
        out.put(packet(101, 1), drop_reason="source_compressed")
        self.assertEqual([out.get(0).frame.t for _ in range(2)], [100, 101])
        self.assertEqual(out.dropped, {})

    def test_buffered_device_time_burst_is_logged_but_not_delivered(self):
        rows = b"".join(source_row(index, index * 300000) for index in range(4))
        with patch.object(live.time, "monotonic", return_value=10.0), \
                self.assertLogs(live.LOGGER, level="WARNING"):
            _, out, raw = self.read_stream([rows], max_age=0.5)
            self.assertIsNone(out.get(0))
            self.assertIsNone(out.get(0))
        self.assertEqual(raw.count("\n"), 4)
        self.assertEqual(out.dropped, {"source_compressed": 4})
        self.assertEqual(out.reset_links, set())
        self.assertEqual(out.dropped_links, {"A"})
        self.assertTrue(out.links["A"]["source"]["source_compressed"])
        self.assertFalse(out.links["A"]["source"]["source_age_known"])

    def test_device_clock_reset_discards_pre_reset_detection_queue(self):
        rows = source_row(10, 1000000) + source_row(11, 1100000) + source_row(0, 1000)
        with patch.object(live.time, "monotonic", return_value=10.0), \
                self.assertLogs(live.LOGGER, level="WARNING"):
            _, out, raw = self.read_stream([rows], max_age=0.5)
            self.assertIsNone(out.get(0))
            self.assertIsNone(out.get(0))
            self.assertEqual(out.get(0).frame.source_id, 0)
        self.assertEqual(raw.count("\n"), 3)
        self.assertEqual(out.reset_links, {"A"})
        self.assertEqual(out.links["A"]["source"]["clock_event"], "reset")

    def test_packet_id_gaps_are_telemetry_and_do_not_discard_frames(self):
        with patch.object(live.time, "monotonic", return_value=10.0):
            _, out, _ = self.read_stream([source_row(1, 1000000) + source_row(3, 1100000)],
                                        max_age=0.5)
            self.assertEqual([out.get(0).frame.source_id for _ in range(2)], [1, 3])
        self.assertEqual(out.dropped, {})
        self.assertEqual(out.reset_links, set())
        self.assertEqual(out.links["A"]["source"]["skipped_ids"], 1)

    def test_disconnect_reopens_port_and_never_joins_partial_old_connection(self):
        stop = threading.Event()
        first = ChunkSerial([LINE[:20].encode()], stop)
        second = ChunkSerial([(LINE + "\n").encode()], stop)
        original_read = first.read

        def disconnected_read(size):
            if not first.chunks:
                raise live.serial.SerialException("disconnected")
            return original_read(size)

        first.read = disconnected_read
        out = live._PacketBuffer(10, 0.5, io.StringIO())
        with patch.object(live.serial, "Serial", side_effect=[first, second]), \
                patch.object(stop, "wait", return_value=False), \
                self.assertLogs(live.LOGGER, level="INFO") as events:
            live._read_port("A", "fake", out, stop)
        self.assertTrue(first.closed and second.closed)
        self.assertEqual(out.queue.qsize(), 1)
        self.assertEqual(out.links["A"]["connections"], 2)
        self.assertEqual(out.links["A"]["disconnects"], 1)
        self.assertEqual(out.reset_links, {"A"})
        self.assertEqual(out.links["A"]["csi_valid"], 1)
        self.assertEqual([record.getMessage().count("receiver_connected") for record in events.records],
                         [1, 0, 1])


if __name__ == "__main__":
    unittest.main()
