from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np

from backend import config
from backend.detection import features

MODEL_PATH = config.DATA_DIR / "model.joblib"
SHOWN = ("peak", "static_change", "duration_s")


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

    def classify(self, x: np.ndarray) -> Verdict:
        proba = self.model.predict_proba(x.reshape(1, -1))[0]
        k = int(np.argmax(proba))
        return Verdict(str(self.model.classes_[k]), float(proba[k]), _details(x))


def make(name: str):
    return RandomForestDetector() if name == "rf" else ThresholdDetector()
