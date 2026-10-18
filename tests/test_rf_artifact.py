"""RF startup must fail visibly rather than run a different detector."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import InconsistentVersionWarning

from backend.detection import detector, features
from backend.training.contract import motion_settings


class ForestArtifactTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "baseline.joblib"
        self.X = np.random.default_rng(7).normal(size=(18, len(features.NAMES)))
        self.model = RandomForestClassifier(n_estimators=5, random_state=0).fit(
            self.X, ["fall", "sit-down", "walk"] * 6)
        self.bundle = {"bundle_version": 1, "model": self.model,
                       "features": features.NAMES, "trained_at": "2026-10-17T23:39:29Z",
                       "motion_settings": motion_settings(),
                       "labels": self.model.classes_.tolist(),
                       "library_versions": {"scikit-learn": sklearn.__version__, "numpy": np.__version__}}

    def save(self, **overrides):
        joblib.dump({**self.bundle, **overrides}, self.path)
        return self.path

    def test_environment_configuration_is_resolved_when_called(self):
        self.save()
        with patch.dict(os.environ, {"SWAMP_DETECTOR": "rf", "SWAMP_RF_MODEL": str(self.path)}):
            self.assertEqual(detector.default_name(), "rf")
            self.assertEqual(detector.default_model_path(), self.path)
            forest = detector.make()
        self.assertIsInstance(forest, detector.RandomForestDetector)
        self.assertEqual(forest.metadata(), {
            "name": "rf", "ready": True, "decision_rule": "argmax", "fall_threshold": None,
            "trained_at": self.bundle["trained_at"], "model_path": str(self.path.resolve())})

    def test_missing_model_and_unknown_detector_never_fall_back(self):
        with self.assertRaisesRegex(FileNotFoundError, "SWAMP_RF_MODEL or --model"):
            detector.make("rf", self.path)
        with self.assertRaisesRegex(ValueError, "unknown detector"):
            detector.make("random-forrest", self.path)
        heuristic = detector.make("threshold", self.path)
        self.assertIsInstance(heuristic, detector.ThresholdDetector)
        self.assertEqual(heuristic.metadata()["decision_rule"], "threshold")

    def test_artifact_loads_once_and_reports_features_and_model_scores(self):
        self.save(fall_threshold=0.35)
        with patch.object(joblib, "load", wraps=joblib.load) as load:
            forest = detector.RandomForestDetector(self.path)
            for row, probabilities in zip(self.X, self.model.predict_proba(self.X)):
                verdict = forest.classify(row)
                expected = detector.predict_labels(probabilities.reshape(1, -1), self.model.classes_, 0.35)[0]
                self.assertEqual(verdict.label, expected)
                scores = dict(zip(self.model.classes_, probabilities))
                self.assertAlmostEqual(verdict.confidence, scores[expected])
                self.assertEqual(verdict.details["class_probabilities"], scores)
                self.assertAlmostEqual(verdict.details["fall_probability"], scores["fall"], places=4)
                for name, value in zip(features.NAMES, row):
                    self.assertAlmostEqual(verdict.details[name], value, places=4)
            load.assert_called_once_with(self.path)
        self.assertEqual(forest.metadata()["decision_rule"], "fall_threshold")

    def test_feature_order_and_preprocessing_must_match(self):
        self.save(features=list(reversed(features.NAMES)))
        with self.assertRaisesRegex(detector.ModelCompatibilityError, "names/order"):
            detector.RandomForestDetector(self.path)
        self.save(motion_settings={**motion_settings(), "T_OFF": 0})
        with self.assertRaisesRegex(detector.ModelCompatibilityError, "preprocessing"):
            detector.RandomForestDetector(self.path)
        wrong = RandomForestClassifier(n_estimators=2).fit(self.X[:, :-1], ["fall", "walk"] * 9)
        self.save(model=wrong)
        with self.assertRaisesRegex(detector.ModelCompatibilityError, "feature count"):
            detector.RandomForestDetector(self.path)

    def test_declared_library_mismatch_stops_before_inference(self):
        for library in ("numpy", "scikit-learn"):
            versions = {**self.bundle["library_versions"], library: "0.0.0"}
            self.save(library_versions=versions)
            with self.assertRaisesRegex(detector.ModelCompatibilityError, "requirements-rf.txt"):
                detector.RandomForestDetector(self.path)

    def test_pickle_version_warning_is_an_actionable_startup_error(self):
        self.save()
        def incompatible_load(_):
            warnings.warn(InconsistentVersionWarning(estimator_name="RandomForestClassifier",
                          current_sklearn_version=sklearn.__version__, original_sklearn_version="1.6.1"))
            return self.bundle
        with patch.object(joblib, "load", side_effect=incompatible_load):
            with self.assertRaisesRegex(detector.ModelCompatibilityError, "needs scikit-learn 1.6.1"):
                detector.RandomForestDetector(self.path)

    def test_invalid_labels_and_bundle_version_are_rejected(self):
        self.save(labels=["walk", "fall", "sit-down"])
        with self.assertRaisesRegex(detector.ModelCompatibilityError, "labels differ"):
            detector.RandomForestDetector(self.path)
        self.save(bundle_version=2)
        with self.assertRaisesRegex(detector.ModelCompatibilityError, "unsupported RF bundle_version"):
            detector.RandomForestDetector(self.path)
        joblib.dump(self.model, self.path)
        with self.assertRaisesRegex(detector.ModelCompatibilityError, "model bundle"):
            detector.RandomForestDetector(self.path)


if __name__ == "__main__":
    unittest.main()
