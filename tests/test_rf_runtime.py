"""The merged live/dashboard entry points must honor the selected RF artifact."""
import argparse
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from backend import dashboard, run, server
from backend.detection import features


class RandomForestRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model_path = self.root / "experiment.joblib"
        X = np.random.default_rng(0).normal(size=(12, len(features.NAMES)))
        forest = RandomForestClassifier(n_estimators=3, random_state=0).fit(X, ["fall", "walk"] * 6)
        joblib.dump({"model": forest, "features": features.NAMES, "trained_at": "test",
                     "fall_threshold": 0.35}, self.model_path)

    def test_dashboard_uses_custom_forest_and_keeps_its_threshold(self):
        args = argparse.Namespace(detector="rf", model=self.model_path, replay=None,
                                  history=str(self.root / "history.jsonl"), sim_fall=None, view_port=8000)
        with patch.object(dashboard, "load_env"), \
                patch.object(server, "serve", return_value="http://127.0.0.1:8000/"), \
                patch.object(server, "open_browser"), \
                patch.object(dashboard.Dashboard, "start"), \
                contextlib.redirect_stdout(io.StringIO()):
            view = run.start_view(args, run.ConsoleSink())
        self.assertEqual(view.pipe.detector.name, "rf")
        self.assertEqual(view.pipe.detector.fall_threshold, 0.35)
        self.assertIs(view.pipe.sink, view.sink)

    def live_cli(self, *extra):
        argv = ["run", "--detector", "rf", "--model", str(self.model_path),
                "--history", str(self.root / "history.jsonl"), *extra]
        # No serial ports, web servers, notifications, or background tickers are started.
        with patch("sys.argv", argv), \
                patch("backend.capture.live.receivers", return_value={"A": "test-port"}), \
                patch("backend.capture.live.live_frames", return_value=iter(())), \
                patch("backend.capture.live.SESSIONS", self.root / "sessions"), \
                patch.object(dashboard, "load_env"), \
                patch.object(run.detector, "make", wraps=run.detector.make) as make, \
                contextlib.redirect_stdout(io.StringIO()) as output:
            run.main()
        self.assertIn("Detector: rf", output.getvalue())
        self.assertTrue(make.call_args_list)
        for call in make.call_args_list:
            self.assertEqual(call.args, ("rf", self.model_path))

    def test_terminal_live_mode_uses_custom_model(self):
        with patch.object(run, "start_view") as start:
            self.live_cli("--no-view")
            start.assert_not_called()

    def test_busy_dashboard_falls_back_to_same_custom_model(self):
        with patch.object(server, "serve", return_value=None), \
                patch.object(server, "open_browser") as browser, \
                patch.object(dashboard.Dashboard, "start") as start:
            self.live_cli()
            browser.assert_not_called()
            start.assert_not_called()


if __name__ == "__main__":
    unittest.main()
