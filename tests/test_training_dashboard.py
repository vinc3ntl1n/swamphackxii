import argparse
import asyncio
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend import server
from backend.capture.live import live_lines
from backend.detection.motion import MEASURE
from backend.training import contract, record, recordings
from backend.training.dashboard import Dashboard


def settings(label="fall"):
    return dict(label=label, n=2, split="test", gap=3.0, performer="tester", spot="room")


def args(root):
    return argparse.Namespace(**settings(), root=root, port=None, port_b=None, replay=None,
                              no_view=True, view_port=8000, no_voice=True, recalibrate=False)


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.hub = server.Hub()
        self.dashboard = Dashboard(args(Path(self.temp.name) / "recordings"), self.hub)
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(self.dashboard.close)

    def test_stop_then_start_another_label_without_overlapping_sessions(self):
        entered = threading.Event()
        labels = []

        def run(session_args, hub, stop):
            labels.append(session_args.label)
            hub.publish(dict(type="record", phase="saved", clip=1))
            entered.set()
            stop.wait(2)

        with patch("backend.training.dashboard.record.run_recording", side_effect=run):
            self.assertTrue(self.dashboard.start(settings()))
            self.assertTrue(entered.wait(1))
            first_id = self.dashboard.state["id"]
            self.assertFalse(self.dashboard.start(settings("walk")))
            self.assertFalse(self.dashboard.calibrate())
            worker = self.dashboard.worker
            self.assertTrue(self.dashboard.stop_session())
            worker.join(2)
            self.assertEqual(self.dashboard.state["status"], "idle")
            self.assertEqual(len(self.hub.kept), 1)
            entered.clear()
            self.assertTrue(self.dashboard.start(settings("walk")))
            self.assertTrue(entered.wait(1))
            self.assertNotEqual(first_id, self.dashboard.state["id"])
            self.assertEqual(len(self.hub.kept), 1)
            self.dashboard.close()
            self.assertEqual(labels, ["fall", "walk"])

    def test_receiver_failure_is_visible_and_the_dashboard_can_retry(self):
        with patch("backend.training.dashboard.record.run_recording", side_effect=SystemExit("No receiver found")):
            self.assertTrue(self.dashboard.start(settings()))
            worker = self.dashboard.worker
            if worker is not None:
                worker.join(2)
            self.assertEqual(self.dashboard.state["error"], "No receiver found")
            self.assertEqual(self.dashboard.state["status"], "idle")
        with patch("backend.training.dashboard.run_calibration") as calibrate:
            self.assertTrue(self.hub.actions["calibrate"]())
            worker = self.dashboard.worker
            if worker is not None:
                worker.join(2)
            calibrate.assert_called_once()
            self.assertIsNone(self.dashboard.state["error"])

    def test_reload_restores_session_before_its_saved_clips(self):
        self.hub.publish(dict(type="record", phase="saved", clip=1))
        self.dashboard.state.update(status="idle")
        self.dashboard.publish()

        async def snapshot():
            queue, messages = self.hub.join()
            self.hub.leave(queue)
            return [json.loads(message) for message in messages]

        messages = asyncio.run(snapshot())
        self.assertEqual([message["type"] for message in messages], ["session", "record"])

    def test_api_validates_settings_and_preserves_action_guards(self):
        client = TestClient(server.make_app(self.hub), base_url="http://127.0.0.1:8000")
        with patch.object(self.dashboard, "_start", return_value=True) as start:
            self.assertEqual(client.post("/session/start", json=settings("walk")).status_code, 202)
            self.assertEqual(start.call_args.args[0]["split"], "test")
            for invalid in [dict(label="unknown"), dict(n=0), dict(n=True), dict(n=1001),
                            dict(gap=-1), dict(split="other"), dict(root="/tmp/other")]:
                self.assertEqual(client.post("/session/start", json={**settings(), **invalid}).status_code, 422)
            self.assertEqual(start.call_count, 1)
            self.assertEqual(client.post("/session/start", json=settings(), headers={"host": "other:8000"}).status_code, 403)
            self.assertEqual(client.post("/session/stop", content="{}").status_code, 415)
        self.assertEqual(client.post("/session/stop", json={}).status_code, 409)

    def test_stop_releases_both_serial_ports_even_without_packets(self):
        stop, entered = threading.Event(), threading.Event()
        opened, closed = [], []

        class Serial:
            def __init__(self, port, *unused, **kwargs):
                self.port = port

            def __enter__(self):
                opened.append(self.port)
                if len(opened) == 2:
                    entered.set()
                return self

            def __exit__(self, *unused):
                closed.append(self.port)

            def readline(self):
                stop.wait(0.01)
                return b""

        with patch("backend.capture.live.serial.Serial", Serial):
            reader = record.Reader(live_lines({"A": "port-a", "B": "port-b"}, stop=stop), stop)
            reader.start()
            self.assertTrue(entered.wait(1))
            reader.close()
            self.assertFalse(reader.is_alive())
        self.assertCountEqual(closed, ["port-a", "port-b"])
        with self.assertRaises(record.SessionStopped):
            reader.wait_first()

    def test_complete_batches_save_to_the_selected_split_and_can_run_again(self):
        root = self.dashboard.args.root
        calibration = dict(measure=MEASURE, size=16, mask=list(range(16)), baseline=0.01, noisy_frac=0)
        (root.parent / recordings.CALIBRATION_FILE).write_text(json.dumps({
            "created_at": "test", "calibration": {"A": calibration},
        }))

        def packets(ports, stop):
            t = 1000.0
            line = 'CSI_DATA,0,0,-45,0,"[' + ",".join(["0", "30"] * 16) + ']"'
            while not stop.wait(0.001):
                t += 0.02
                yield t, "A", line

        with patch("backend.capture.live.receivers", return_value={"A": "fake"}), \
                patch("backend.capture.live.live_lines", side_effect=packets), \
                patch("backend.training.record.beep"):
            for split in ("train", "test"):
                self.assertTrue(self.dashboard.start({**settings("empty"), "n": 1, "split": split}))
                worker = self.dashboard.worker
                self.assertIsNotNone(worker)
                worker.join(5)
                self.assertFalse(worker.is_alive())
                self.assertIsNone(self.dashboard.state["error"])
                self.assertEqual(self.dashboard.state["status"], "idle")
                recs = recordings.load(split, root)
                self.assertEqual(len(recs), 1)
                self.assertEqual(recs[0].label, "empty")
                self.assertEqual(recs[0].meta["performer"], "tester")
                self.assertEqual(recs[0].meta["schema_version"], 1)
                self.assertEqual(recs[0].meta["quality"]["problems"], [])
                meta_path = recs[0].log_path.with_name(recs[0].meta["recording_id"] + ".json")
                self.assertEqual(contract.validate_pair(meta_path)["errors"], [])
                self.assertEqual(recs[0].problems(), [])
                phases = [json.loads(message)["phase"] for _, message in self.hub.kept]
                self.assertEqual(phases, ["position", "count", "count", "count", "go", "saved", "done"])


if __name__ == "__main__":
    unittest.main()
