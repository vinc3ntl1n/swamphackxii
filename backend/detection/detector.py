from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np

from backend import config
from backend.detection import features

MODEL_PATH = config.DATA_DIR / "model.joblib"
SHOWN = ("peak", "static_change", "duration_s")


def predict_labels(probabilities: np.ndarray, classes, fall_threshold: float | None = None) -> np.ndarray:
    """Apply the same decision rule in offline evaluation and live inference."""
    p = np.asarray(probabilities, dtype=float)
    labels = np.asarray(classes, dtype=str)
    if (p.ndim != 2 or p.shape[1] != len(labels) or not len(labels)
            or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any()
            or not np.allclose(p.sum(axis=1), 1)):
        raise ValueError("expected finite class probabilities with each row summing to one")
    if fall_threshold is None:
        return labels[np.argmax(p, axis=1)]
    if not np.isfinite(fall_threshold) or not 0 <= fall_threshold <= 1:
        raise ValueError("fall threshold must be between 0 and 1")
    if "fall" not in labels or len(labels) < 2:
        raise ValueError("a fall threshold requires fall and at least one other class")
    fall = int(np.flatnonzero(labels == "fall")[0])
    other = np.flatnonzero(labels != "fall")
    pred = labels[other[np.argmax(p[:, other], axis=1)]].copy()
    pred[p[:, fall] >= fall_threshold] = "fall"
    return pred


@dataclass
class Verdict:
    label: str
    confidence: float
    details: dict = field(default_factory=dict)


def _details(x: np.ndarray) -> dict:
    return {n: round(float(x[features.NAMES.index(n)]), 2) for n in SHOWN}


class ThresholdDetector:
    name = "threshold"

    def classify(self, x: np.ndarray) -> Verdict:
        f = dict(zip(features.NAMES, x))
        if f["peak"] >= config.T_FALL and f["static_change"] >= config.T_STATIC:
            return Verdict("fall", 1.0, _details(x))
        if f["duration_s"] > config.SIT_MAX_S:
            return Verdict("walk", 0.5, _details(x))
        return Verdict("sit-down", 0.5, _details(x))


class RandomForestDetector:
    name = "rf"

    def __init__(self, path: Path = MODEL_PATH) -> None:
        if not path.exists():
            raise FileNotFoundError(f"{path} not found: run python -m backend.training.train first")
        bundle = joblib.load(path)
        if bundle["features"] != features.NAMES:
            raise ValueError("model.joblib was trained on different features: rerun python -m backend.training.train")
        self.model = bundle["model"]
        self.trained_at = bundle["trained_at"]
        self.fall_threshold = bundle.get("fall_threshold")  # Legacy bundles retain argmax.
        settings = bundle.get("motion_settings")
        if settings is not None:
            from backend.training.contract import motion_settings
            if settings != motion_settings():
                raise ValueError("model preprocessing settings differ: restore settings or retrain")
        predict_labels(np.ones((1, len(self.model.classes_))) / len(self.model.classes_),
                       self.model.classes_, self.fall_threshold)

    def classify(self, x: np.ndarray) -> Verdict:
        if x.shape != (len(features.NAMES),) or not np.isfinite(x).all():
            raise ValueError("expected one finite value per model feature")
        proba = self.model.predict_proba(x.reshape(1, -1))[0]
        label = str(predict_labels(proba.reshape(1, -1), self.model.classes_, self.fall_threshold)[0])
        k = int(np.flatnonzero(self.model.classes_ == label)[0])
        details = _details(x)
        if "fall" in self.model.classes_:
            fall = list(self.model.classes_).index("fall")
            details["fall_probability"] = round(float(proba[fall]), 4)
        return Verdict(label, float(proba[k]), details)


def make(name: str, model_path: Path = MODEL_PATH):
    return RandomForestDetector(model_path) if name == "rf" else ThresholdDetector()
