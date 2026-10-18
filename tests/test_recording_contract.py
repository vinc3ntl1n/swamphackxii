import gzip
import json
import tempfile
import unittest
from pathlib import Path

from backend.detection.motion import MEASURE
from backend.training import contract, recordings
from backend.training.validate import audit


class RecordingContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "recordings"

    def save(self, split="train", session="session-1", links=("A", "B")):
        # Two antennas and six samples spanning a complete two-second clip.
        payload = 'CSI_DATA,1,device,-45,0,"[1,2,3,4]"'
        rows = [(t, link, payload) for t in (99.0, 100.0, 101.0) for link in links]
        cal = dict(measure=MEASURE, size=2, mask=[0, 1], baseline=0.01, noisy_frac=0)
        meta = dict(label="empty", split=split, performer="tester", spot="room",
                    go_t=100.0, pre_s=1.0, post_s=1.0, packets=len(rows),
                    rate_hz={link: 1.5 for link in links},
                    recorded_at="2026-10-17T22:00:00-04:00", calibration={link: cal for link in links},
                    session_id=session, setup_id="placement-1",
                    capture=dict(source="live", motion_settings=contract.motion_settings(),
                                 git_commit="test", git_dirty=False,
                                 receivers={link: f"device-{link}" for link in links}, sender="sender"))
        path = recordings.save(rows, meta, self.root)
        return path.with_name(path.name.removesuffix(".log.gz") + ".json")

    def update(self, path, **fields):
        meta = json.loads(path.read_text())
        meta.update(fields)
        path.write_text(json.dumps(meta))

    def test_versioned_pair_round_trip_and_unique_names(self):
        first, second = self.save(), self.save()
        self.assertNotEqual(first.stem, second.stem)
        result = contract.validate_pair(first)
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["counts"], {"A": 3, "B": 3})
        self.assertEqual(len(recordings.load("train", self.root)), 2)

    def test_legacy_single_receiver_and_timestamp_envelope(self):
        path = self.save(links=("A",))
        meta = json.loads(path.read_text())
        for key in ("schema_version", "recording_id", "raw_sha256", "capture", "session_id", "setup_id"):
            meta.pop(key)
        meta["calibration"] = meta["calibration"]["A"]
        meta["rate_hz"] = meta["rate_hz"]["A"]
        meta["recorded_at"] = "2026-10-17T22:00:00"
        path.write_text(json.dumps(meta))
        log = path.with_name(path.stem + ".log.gz")
        with gzip.open(log, "rt") as stream:
            rows = [row.replace("\tA\t", "\t", 1) for row in stream]
        with gzip.open(log, "wt") as stream:
            stream.writelines(rows)
        self.assertEqual(contract.validate_pair(path)["errors"], [])
        self.assertEqual(len(recordings.load("train", self.root)[0].frames()), 3)

    def test_bad_metadata_produces_diagnostics(self):
        for fields in ({"calibration": []}, {"packets": True}, {"go_t": float("nan")},
                       {"schema_version": 99}, {"rate_hz": {"A": "fast"}},
                       {"recorded_at": "2026-10-17T22:00:00"}):
            with self.subTest(fields=fields):
                path = self.save()
                self.update(path, **fields)
                self.assertTrue(contract.validate_pair(path)["errors"])

    def test_checksum_mismatch_and_bad_iq_are_detected(self):
        path = self.save()
        log = path.with_name(path.stem + ".log.gz")
        with gzip.open(log, "rt") as stream:
            text = stream.read().replace("[1,2,3,4]", "[1,2,3]")
        with gzip.open(log, "wt") as stream:
            stream.write(text)
        errors = contract.validate_pair(path)["errors"]
        self.assertIn("raw_sha256 mismatch", errors)
        self.assertTrue(any("I/Q length" in error for error in errors))

    def test_counts_rates_and_timestamp_order_are_checked(self):
        for fields, message in [({"packets": 42}, "packets"),
                                ({"rate_hz": {"A": 10, "B": 10}}, "rate_hz")]:
            path = self.save()
            self.update(path, **fields)
            self.assertTrue(any(message in e for e in contract.validate_pair(path)["errors"]))
        path = self.save()
        log = path.with_name(path.stem + ".log.gz")
        with gzip.open(log, "rt") as stream:
            rows = stream.readlines()
        with gzip.open(log, "wt") as stream:
            stream.writelines(reversed(rows))
        self.update(path, raw_sha256=contract.sha256(log))
        self.assertTrue(any("nondecreasing" in e for e in contract.validate_pair(path)["errors"]))

    def test_missing_pairs_and_cross_split_session_are_reported(self):
        path = self.save()
        self.save(split="test")
        self.assertTrue(all(any("evaluation leakage" in e for e in r["errors"])
                            for r in audit(self.root, quality=False)))
        path.unlink()
        self.assertTrue(any("missing .json partner" in r["errors"]
                            for r in audit(self.root, quality=False)))
        path = self.save(session="another-session")
        path.with_name(path.stem + ".log.gz").unlink()
        self.assertIn("missing .log.gz partner", contract.validate_pair(path)["errors"])

    def test_corrupt_archive_and_unknown_receiver_are_reported(self):
        path = self.save()
        log = path.with_name(path.stem + ".log.gz")
        with gzip.open(log, "rt") as stream:
            text = stream.read().replace("\tA\t", "\tC\t")
        with gzip.open(log, "wt") as stream:
            stream.write(text)
        self.update(path, raw_sha256=contract.sha256(log))
        self.assertTrue(any("no calibration" in e for e in contract.validate_pair(path)["errors"]))
        log.write_bytes(b"broken gzip")
        self.assertTrue(any("cannot read CSI archive" in e for e in contract.validate_pair(path)["errors"]))

    def test_training_rejects_missing_partner_before_fitting(self):
        from unittest.mock import patch
        from backend.training import train

        path = self.save()
        path.with_name(path.stem + ".log.gz").unlink()
        with patch("sys.argv", ["train", "--root", str(self.root)]), \
                patch.object(train, "RandomForestClassifier") as forest:
            with self.assertRaisesRegex(SystemExit, "contract validation failed"):
                train.main()
            forest.assert_not_called()

    def test_quality_audit_reports_insufficient_post_event_time(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        import numpy as np

        self.save()
        event = SimpleNamespace(link="A", duration=0.5, end=100.0, cut_short=False)
        with patch.object(recordings.Recording, "problems", return_value=[]), \
                patch.object(recordings.Recording, "event", return_value=event), \
                patch.object(recordings.Recording, "_run", return_value={"A": ([], np.array([101.0]), [], [])}):
            result = audit(self.root)[0]
        self.assertIn("incomplete post-event observation window", result["warnings"])


if __name__ == "__main__":
    unittest.main()
