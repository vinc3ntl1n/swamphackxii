import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from backend.detection import features
from backend.detection.detector import RandomForestDetector, predict_labels
from backend.detection.motion import MEASURE
from backend.training import contract, evaluation, recordings, train


class DecisionRuleTests(unittest.TestCase):
    def test_threshold_is_inclusive_and_can_override_argmax_both_ways(self):
        classes = ["fall", "sit-down", "walk"]
        p = np.array([[0.35, 0.4, 0.25], [0.6, 0.1, 0.3]])
        self.assertEqual(predict_labels(p, classes).tolist(), ["sit-down", "fall"])
        self.assertEqual(predict_labels(p, classes, 0.35).tolist(), ["fall", "fall"])
        self.assertEqual(predict_labels(p, classes, 0.8).tolist(), ["sit-down", "walk"])

    def test_invalid_probabilities_and_thresholds_fail(self):
        for p in (np.array([[float("nan"), 0.5]]), np.array([[0.1, 0.1]]), np.array([0.5, 0.5])):
            with self.assertRaises(ValueError):
                predict_labels(p, ["fall", "walk"])
        for threshold in (-0.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                predict_labels(np.array([[0.5, 0.5]]), ["fall", "walk"], threshold)

    def test_saved_threshold_and_legacy_bundle_match_offline_predictions(self):
        rng = np.random.default_rng(10)
        X = rng.normal(size=(20, len(features.NAMES)))
        model = RandomForestClassifier(n_estimators=7, random_state=0).fit(X, ["fall", "walk"] * 10)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.joblib"
            for threshold in (None, 0.35, 0.9):
                bundle = {"model": model, "features": features.NAMES, "trained_at": "test"}
                if threshold is not None:
                    bundle["fall_threshold"] = threshold
                joblib.dump(bundle, path)
                detector = RandomForestDetector(path)
                expected = predict_labels(model.predict_proba(X), model.classes_, threshold)
                self.assertEqual([detector.classify(x).label for x in X], expected.tolist())
                verdict = detector.classify(X[0])
                self.assertAlmostEqual(verdict.details["fall_probability"], model.predict_proba(X[:1])[0, 0], places=4)
                with self.assertRaises(ValueError):
                    detector.classify(np.full(len(features.NAMES), np.nan))
            bundle["motion_settings"] = {**contract.motion_settings(), "T_ON": -1}
            joblib.dump(bundle, path)
            with self.assertRaisesRegex(ValueError, "preprocessing"):
                RandomForestDetector(path)


class EvaluationTests(unittest.TestCase):
    def test_grouped_probabilities_come_from_forests_that_exclude_entire_sessions(self):
        X = np.random.default_rng(3).normal(size=(24, 6))
        y = np.array(["fall", "walk", "sit-down"] * 8)
        sessions = np.repeat(["s1", "s2", "s3", "s4"], 6)
        model = RandomForestClassifier(n_estimators=5, random_state=1)
        result = evaluation.out_of_fold(model, X, y, sessions, max_folds=3)
        self.assertEqual(result["mode"], "session")
        for session in set(sessions):
            self.assertEqual(len(set(result["fold_ids"][sessions == session])), 1)
        for fold in set(result["fold_ids"]):
            valid = result["fold_ids"] == fold
            expected = RandomForestClassifier(n_estimators=5, random_state=1).fit(X[~valid], y[~valid])
            np.testing.assert_allclose(result["probabilities"][valid], expected.predict_proba(X[valid]))
        self.assertFalse(hasattr(model, "estimators_"))

    def test_legacy_fallback_and_insufficient_sessions_are_explicit(self):
        X = np.zeros((6, 6))
        y = ["fall", "walk"] * 3
        model = RandomForestClassifier(n_estimators=2, random_state=0)
        self.assertEqual(evaluation.out_of_fold(model, X, y, [None] * 6)["mode"], "clip")
        with self.assertRaisesRegex(ValueError, "mixed session"):
            evaluation.out_of_fold(model, X, y, ["s1", None] * 3)
        with self.assertRaisesRegex(ValueError, "two training sessions"):
            evaluation.out_of_fold(model, X, y, ["s1"] * 6)

    def test_metrics_include_counts_and_undefined_recall(self):
        result = evaluation.metrics(["fall", "fall", "walk"], ["fall", "walk", "fall"])
        self.assertEqual((result["true_falls_detected"], result["missed_falls"], result["false_fall_predictions"]), (1, 1, 1))
        result = evaluation.metrics(["walk"], ["walk"])
        self.assertIsNone(result["fall_recall"])
        self.assertIsNone(result["fall_precision"])


class TrainingIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "recordings"

    def recording(self, label, split, session, quiet=False):
        cal = dict(measure=MEASURE, size=16, mask=list(range(16)), baseline=0.01, noisy_frac=0)
        rows = []
        final = {"fall": 65, "sit-down": 40, "walk": 30}[label]
        for t in np.arange(98.5, 108.501, 0.05):
            amplitude = 30 if quiet or t < 100 else 95 if t < 100.5 else final
            payload = 'CSI_DATA,1,device,-45,0,"[' + ",".join(["0", str(amplitude)] * 16) + ']"'
            rows.append((float(t), "A", payload))
        meta = dict(label=label, split=split, performer="tester", spot="room", go_t=100,
                    pre_s=1.5, post_s=8.5, packets=len(rows), rate_hz={"A": round(len(rows) / 10, 1)},
                    recorded_at="2026-10-17T22:00:00-04:00", calibration={"A": cal},
                    session_id=session, setup_id="layout", capture=contract.capture_context("replay"))
        return recordings.save(rows, meta, self.root)

    def prepare_train(self):
        for session in ("train-1", "train-2", "train-3"):
            for label in ("fall", "sit-down", "walk"):
                self.recording(label, "train", session)

    def run_train(self, *extra):
        argv = ["train", "--root", str(self.root), "--n-estimators", "5", "--folds", "3",
                "--model", str(self.base / "artifacts/model.joblib"),
                "--results", str(self.base / "reports/results.json"), *extra]
        with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()):
            train.main()
        return json.loads((self.base / "reports/results.json").read_text())

    def test_real_csi_to_saved_forest_with_session_validation_and_heldout_test(self):
        self.prepare_train()
        for label in ("fall", "sit-down", "walk"):
            self.recording(label, "test", "test-1")
        result = self.run_train("--evaluate-test", "--fall-threshold", "0.35", "--max-depth", "4", "--fall-weight", "3")
        self.assertEqual(result["validation"]["mode"], "session")
        self.assertEqual(len(result["validation"]["predictions"]), 9)
        self.assertEqual(len(result["test"]["predictions"]), 3)
        self.assertNotIn("threshold_comparison", result["test"])
        self.assertEqual(result["exclusions"], [])
        detector = RandomForestDetector(self.base / "artifacts/model.joblib")
        self.assertEqual(detector.fall_threshold, 0.35)
        self.assertEqual(detector.model.max_depth, 4)
        self.assertEqual(detector.model.class_weight["fall"], 3)
        bundle = joblib.load(self.base / "artifacts/model.joblib")
        self.assertEqual(len(bundle["training_recordings"]), 9)
        self.assertTrue(all(row["session_id"].startswith("train-") for row in bundle["training_recordings"]))
        for row in result["test"]["predictions"]:
            x = np.array([row["features"][name] for name in features.NAMES])
            self.assertEqual(detector.classify(x).label, row["prediction"])

    def test_no_test_uses_validation_and_empty_test_is_an_error(self):
        self.prepare_train()
        result = self.run_train()
        self.assertIsNone(result["test"])
        self.assertIsNone(result["fall_threshold"])
        self.recording("fall", "test", "test-1", quiet=True)
        self.assertIsNone(self.run_train()["test"])
        with self.assertRaisesRegex(SystemExit, "none are usable"):
            self.run_train("--evaluate-test")


if __name__ == "__main__":
    unittest.main()
