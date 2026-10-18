from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping
import os
from pathlib import Path
import warnings

import joblib
import numpy as np

from backend import config
from backend.detection import features

MODEL_PATH = config.MODEL_PATH


def default_name() -> str:
    """Resolve after loading .env, so dashboard and terminal share configuration."""
    return os.environ.get("SWAMP_DETECTOR", config.DETECTOR).strip().lower()


def default_model_path() -> Path:
    return Path(os.environ.get("SWAMP_RF_MODEL", str(MODEL_PATH))).expanduser()


class ModelCompatibilityError(ValueError):
    """The requested artifact cannot safely run with this preprocessing/runtime."""


def _runtime_help() -> str:
    return ("Use a Python 3.12 virtual environment with requirements-rf.txt for the "
            "shared baseline (scikit-learn 1.6.1, NumPy 2.1.3), or retrain and "
            "evaluate an artifact in this environment.")


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
    return {name: round(float(value), 4) for name, value in zip(features.NAMES, x)}


class ThresholdDetector:
    name = "threshold"

    def metadata(self) -> dict:
        return {"name": self.name, "ready": True, "decision_rule": "threshold",
                "fall_threshold": None, "trained_at": None, "model_path": None}

    def classify(self, x: np.ndarray) -> Verdict:
        f = dict(zip(features.NAMES, x))
        if f["peak"] >= config.T_FALL and f["static_change"] >= config.T_STATIC:
            return Verdict("fall", 1.0, _details(x))
        if f["duration_s"] > config.SIT_MAX_S:
            return Verdict("walk", 0.5, _details(x))
        return Verdict("sit-down", 0.5, _details(x))


class RandomForestDetector:
    name = "rf"

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path).expanduser() if path is not None else default_model_path()
        if not self.path.is_file():
            raise FileNotFoundError(
                f"RF artifact not found: {self.path}. Copy the trusted baseline to this machine "
                "and set SWAMP_RF_MODEL or --model, or train with python -m backend.training.train."
            )
        import sklearn
        from sklearn.exceptions import InconsistentVersionWarning

        try:
            # Cross-version pickle loading is unsupported by scikit-learn. Never
            # continue through its warning and present an incompatible forest as ready.
            with warnings.catch_warnings():
                warnings.simplefilter("error", InconsistentVersionWarning)
                bundle = joblib.load(self.path)
        except InconsistentVersionWarning as exc:
            raise ModelCompatibilityError(
                f"RF artifact {self.path} needs scikit-learn {exc.original_sklearn_version}; "
                f"running {sklearn.__version__}. {_runtime_help()}"
            ) from exc
        except Exception as exc:
            raise ModelCompatibilityError(f"Cannot load RF artifact {self.path}: {exc}. {_runtime_help()}") from exc
        if not isinstance(bundle, Mapping):
            raise ModelCompatibilityError("RF artifact must be a model bundle, not a bare estimator")
        if bundle.get("bundle_version", 1) != 1:
            raise ModelCompatibilityError(f"unsupported RF bundle_version: {bundle['bundle_version']!r}")
        saved_features = bundle.get("features")
        if not isinstance(saved_features, (list, tuple)) or list(saved_features) != features.NAMES:
            raise ModelCompatibilityError("RF artifact feature names/order differ: restore settings or retrain")
        versions = bundle.get("library_versions", {})
        if not isinstance(versions, Mapping):
            raise ModelCompatibilityError("RF artifact library_versions must be a mapping")
        for library, running in (("scikit-learn", sklearn.__version__), ("numpy", np.__version__)):
            saved = versions.get(library)
            if saved is not None and saved != running:
                raise ModelCompatibilityError(
                    f"RF artifact {self.path} needs {library} {saved}; running {running}. {_runtime_help()}"
                )
        self.model = bundle.get("model")
        if (not callable(getattr(self.model, "predict_proba", None))
                or not hasattr(self.model, "classes_")):
            raise ModelCompatibilityError("RF artifact needs a fitted classifier with predict_proba and classes_")
        self.classes = np.asarray(self.model.classes_, dtype=str)
        if (self.classes.ndim != 1 or len(set(self.classes)) != len(self.classes)
                or "fall" not in self.classes or len(self.classes) < 2
                or not set(self.classes).issubset({"fall", "sit-down", "walk"})):
            raise ModelCompatibilityError("RF classes must include fall and at least one of sit-down/walk")
        if getattr(self.model, "n_features_in_", None) != len(features.NAMES):
            raise ModelCompatibilityError("RF estimator feature count differs from the six event features")
        saved_labels = bundle.get("labels")
        if saved_labels is not None and list(saved_labels) != self.classes.tolist():
            raise ModelCompatibilityError("RF bundle labels differ from the fitted estimator classes")
        self.trained_at = bundle.get("trained_at")
        self.fall_threshold = bundle.get("fall_threshold")  # Legacy bundles retain argmax.
        settings = bundle.get("motion_settings")
        if settings is not None:
            from backend.training.contract import motion_settings
            if settings != motion_settings():
                raise ModelCompatibilityError("model preprocessing settings differ: restore settings or retrain")
        # Validate the saved decision rule and exercise inference once before
        # opening live capture. This does not fit or modify the forest.
        try:
            predict_labels(self.model.predict_proba(np.zeros((1, len(features.NAMES)))),
                           self.classes, self.fall_threshold)
        except Exception as exc:
            raise ModelCompatibilityError(f"RF artifact failed startup inference: {exc}") from exc

    def metadata(self) -> dict:
        return {"name": self.name, "ready": True,
                "decision_rule": "argmax" if self.fall_threshold is None else "fall_threshold",
                "fall_threshold": self.fall_threshold, "trained_at": self.trained_at,
                "model_path": str(self.path.resolve())}

    def classify(self, x: np.ndarray) -> Verdict:
        x = np.asarray(x, dtype=float)
        if x.shape != (len(features.NAMES),) or not np.isfinite(x).all():
            raise ValueError("expected one finite value per model feature")
        proba = self.model.predict_proba(x.reshape(1, -1))[0]
        label = str(predict_labels(proba.reshape(1, -1), self.classes, self.fall_threshold)[0])
        k = int(np.flatnonzero(self.classes == label)[0])
        details = _details(x)
        details["class_probabilities"] = dict(zip(self.classes.tolist(), map(float, proba)))
        fall = self.classes.tolist().index("fall")
        details["fall_probability"] = round(float(proba[fall]), 4)
        return Verdict(label, float(proba[k]), details)


def make(name: str | None = None, model_path: Path | str | None = None):
    selected = default_name() if name is None else name
    if selected == "rf":
        return RandomForestDetector(model_path)
    if selected == "threshold":
        return ThresholdDetector()
    raise ValueError(f"unknown detector {selected!r}; choose rf or threshold")
