from __future__ import annotations

import csv
import io
import unittest

import numpy as np

from backend.capture.csi import Frame, parse_line
from backend.capture.timing import SourceTiming


def frame(source_us: int | None, packet_id: int | None = 1, link: str = "A") -> Frame:
    return Frame(1234.5, -45, np.array([5.0, 13.0]), link,
                 source_id=packet_id, source_timestamp_us=source_us)


def csv_line(source_us="123456", packet_id="7"):
    fields = ["CSI_DATA", str(packet_id), "1a:00:00:00:00:00", "-45"] + ["0"] * 21
    fields[18] = str(source_us)
    fields[22], fields[23], fields[24] = "4", "0", "[3,4,5,12]"
    out = io.StringIO()
    csv.writer(out).writerow(fields)
    return out.getvalue()


class SourceMetadataTests(unittest.TestCase):
    def test_full_schema_metadata_preserves_existing_rf_inputs(self):
        parsed = parse_line(csv_line(), 2000.0, "B")
        self.assertEqual((parsed.t, parsed.rssi, parsed.link), (2000.0, -45, "B"))
        np.testing.assert_array_equal(parsed.amp, [5.0, 13.0])
        self.assertEqual((parsed.source_id, parsed.source_timestamp_us), (7, 123456))
        self.assertIsNone(parsed.received_mono)

    def test_signed_unsigned_counter_representation(self):
        parsed = parse_line(csv_line(-1, -2147483648), 1.0)
        self.assertEqual(parsed.source_timestamp_us, (1 << 32) - 1)
        self.assertEqual(parsed.source_id, 1 << 31)

    def test_malformed_optional_metadata_does_not_reject_valid_csi(self):
        for value in ("bad", "nan", "1.5", str(1 << 32), str(-(1 << 31) - 1)):
            with self.subTest(value=value):
                parsed = parse_line(csv_line(value, value), 1.0)
                np.testing.assert_array_equal(parsed.amp, [5.0, 13.0])
                self.assertIsNone(parsed.source_timestamp_us)
                self.assertIsNone(parsed.source_id)

    def test_legacy_csv_and_frame_constructor_remain_compatible(self):
        parsed = parse_line('CSI_DATA,7,device,-45,0,"[3,4,5,12]"', 1.0)
        self.assertIsNone(parsed.source_timestamp_us)
        self.assertIsNone(parsed.source_id)
        self.assertIsNone(Frame(1.0, -45, np.ones(2), "B").received_mono)

    def test_receipt_age_uses_monotonic_clock_and_source_lag_stays_explicit(self):
        captured = frame(100_000)
        captured.received_mono, captured.source_lag_s = 10.0, 0.3
        self.assertAlmostEqual(captured.capture_age_s(now_mono=10.2, now_wall=0.0), 0.2)
        self.assertAlmostEqual(captured.relative_age_s(now_mono=10.2, now_wall=0.0), 0.5)
        legacy = Frame(100.0, -45, np.ones(2))
        self.assertAlmostEqual(legacy.capture_age_s(now_mono=1000.0, now_wall=100.2), 0.2)


class SourceTimingTests(unittest.TestCase):
    def test_healthy_timing_jitter_and_packet_id_gaps_are_not_stale(self):
        timing = SourceTiming()
        first = timing.observe(frame(100_000, 10), 10.0)
        middle = timing.observe(frame(120_000, 12), 10.022)
        final = timing.observe(frame(140_000, 15), 10.041)
        self.assertFalse(first["source_age_known"])
        self.assertAlmostEqual(middle["source_lag_s"], 0.002)
        self.assertEqual(final["skipped_ids"], 3)
        self.assertFalse(final["stale"])
        self.assertFalse(final["timing_discontinuity"])
        self.assertIsNone(final["clock_event"])

    def test_growing_backlog_is_measured_relative_to_freshest_anchor(self):
        timing = SourceTiming()
        timing.observe(frame(0, 1), 10.0)
        delayed = timing.observe(frame(100_000, 2), 10.8)
        self.assertAlmostEqual(delayed["source_lag_s"], 0.7)
        self.assertTrue(delayed["stale"])
        recovered = timing.observe(frame(900_000, 3), 10.9)
        self.assertAlmostEqual(recovered["source_lag_s"], 0.0)
        self.assertFalse(recovered["stale"])
        self.assertTrue(recovered["source_compressed"])
        settled = timing.observe(frame(3_100_000, 4), 13.1)
        self.assertFalse(settled["source_compressed"])

    def test_initial_buffered_burst_is_flagged_even_when_offset_age_is_unknown(self):
        timing = SourceTiming()
        timing.observe(frame(100_000_000, 1), 10.0)
        events = []
        for n in range(1, 11):
            result = timing.observe(frame(100_000_000 + n * 100_000, n + 1), 10 + n * 0.001)
            if result["clock_event"] is not None:
                events.append(result["clock_event"])
        self.assertEqual(result["source_lag_ms"], 0.0)
        self.assertFalse(result["source_age_known"])
        self.assertTrue(result["source_compressed"])
        self.assertEqual(events, ["compressed_catchup"])
        self.assertTrue(timing.observe(frame(102_000_000, 12), 10.02)["source_compressed"])

    def test_uint32_timestamp_and_id_wrap_continue_same_clock(self):
        timing = SourceTiming()
        timing.observe(frame((1 << 32) - 10_000, (1 << 32) - 1), 10)
        wrapped = timing.observe(frame(10_000, 0), 10.02)
        self.assertEqual(wrapped["clock_event"], "wrap")
        self.assertEqual(wrapped["source_step_ms"], 20.0)
        self.assertFalse(wrapped["source_clock_reset"])
        self.assertFalse(wrapped["stale"])
        self.assertEqual(wrapped["skipped_ids"], 0)

    def test_source_timestamp_reset_invalidates_clock_without_fabricating_age(self):
        timing = SourceTiming()
        timing.observe(frame(100_000_000, 100), 10)
        reset = timing.observe(frame(100_000, 1), 10.02)
        self.assertEqual(reset["clock_event"], "reset")
        self.assertEqual(reset["reset_reason"], "source_timestamp_reset")
        self.assertTrue(reset["timing_discontinuity"])
        self.assertEqual(reset["source_clock_elapsed_s"], 0)
        self.assertEqual(reset["source_lag_s"], 0)
        ready = timing.observe(frame(120_000, 2), 10.04)
        self.assertIsNone(ready["clock_event"])
        self.assertFalse(ready["stale"])

    def test_counter_reset_is_detected_even_if_device_time_increases(self):
        timing = SourceTiming()
        timing.observe(frame(100_000, 1000), 10)
        reset = timing.observe(frame(120_000, 0), 10.02)
        self.assertEqual(reset["reset_reason"], "source_id_reset")
        self.assertTrue(reset["source_clock_reset"])
        self.assertEqual(reset["skipped_ids"], 0)

    def test_receivers_have_independent_clocks_and_missing_metadata_is_supported(self):
        clocks = {"A": SourceTiming(), "B": SourceTiming()}
        clocks["A"].observe(frame(0, 1, "A"), 10)
        clocks["B"].observe(frame(1_000_000_000, 1, "B"), 10)
        a = clocks["A"].observe(frame(20_000, 2, "A"), 10.8)
        b = clocks["B"].observe(frame(1_000_800_000, 2, "B"), 10.8)
        self.assertTrue(a["stale"])
        self.assertFalse(b["stale"])
        self.assertFalse(b["source_compressed"])
        unavailable = clocks["A"].observe(frame(None), 10.9)
        self.assertFalse(unavailable["source_available"])
        self.assertIsNone(unavailable["source_lag_ms"])

    def test_explicit_connection_reset_and_host_clock_regression(self):
        timing = SourceTiming()
        timing.observe(frame(1_000_000, 10), 10)
        timing.reset()
        reset = timing.observe(frame(1_020_000, 11), 11)
        self.assertEqual(reset["reset_reason"], "connection_reopened")
        self.assertTrue(reset["source_clock_reset"])
        regressed = timing.observe(frame(1_040_000, 12), 10)
        self.assertEqual(regressed["reset_reason"], "host_clock_reset")
        self.assertFalse(regressed["stale"])


if __name__ == "__main__":
    unittest.main()
