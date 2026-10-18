"""Classifier evaluation on held-out folds, with session boundaries when available."""
from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.base import clone
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from backend.detection.detector import predict_labels

LABELS = ["walk", "sit-down", "fall"]


def metrics(y_true, y_pred) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    fall = LABELS.index("fall")
    tp = int(cm[fall, fall])
    fn = int(cm[fall].sum() - tp)
    fp = int(cm[:, fall].sum() - tp)
    return {"labels": LABELS, "confusion": cm.tolist(),
            "accuracy": float(np.trace(cm) / max(cm.sum(), 1)),
            "fall_recall": tp / (tp + fn) if tp + fn else None,
            "fall_precision": tp / (tp + fp) if tp + fp else None,
            "true_falls_detected": tp, "missed_falls": fn, "false_fall_predictions": fp,
            "nonfall_count": int(cm.sum() - cm[fall].sum()),
            "fall_count": int(cm[fall].sum())}


def out_of_fold(model, X, y, sessions, mode="auto", max_folds=5, seed=0) -> dict:
    """Fit separate forests on training folds; never score a clip on its own forest."""
    y = np.asarray(y)
    classes = np.unique(y)
    known = [bool(s) for s in sessions]
    if mode == "auto":
        if any(known) and not all(known):
            raise ValueError("mixed session metadata: assign verified session IDs or explicitly use --cv clip")
        mode = "session" if all(known) else "clip"
    if mode == "session":
        if not all(known):
            raise ValueError("--cv session requires session_id on every usable training recording")
        groups = np.asarray(sessions)
        folds = min(max_folds, min(len(set(groups[y == label])) for label in classes))
        if folds < 2:
            raise ValueError("need at least two training sessions containing each class for session validation")
        splitter = StratifiedGroupKFold(folds, shuffle=True, random_state=seed)
        splits = list(splitter.split(X, y, groups))
    else:
        folds = min(max_folds, min(Counter(y).values()))
        if folds < 2:
            raise ValueError("need at least two usable training recordings per class for clip validation")
        splits = list(StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y))
    probabilities = np.zeros((len(y), len(classes)))
    fold_ids = np.zeros(len(y), dtype=int)
    for fold, (train, valid) in enumerate(splits, 1):
        if set(y[train]) != set(classes):
            raise ValueError("a validation fold leaves a class out of training; collect more balanced sessions")
        forest = clone(model).fit(X[train], y[train])
        probabilities[valid] = forest.predict_proba(X[valid])[:, [list(forest.classes_).index(c) for c in classes]]
        fold_ids[valid] = fold
    return {"probabilities": probabilities, "classes": classes, "fold_ids": fold_ids,
            "mode": mode, "folds": folds}


def score(y, probabilities, classes, threshold) -> dict:
    return metrics(y, predict_labels(probabilities, classes, threshold))


def threshold_comparison(y, probabilities, classes) -> list[dict]:
    return [{"fall_threshold": threshold, **score(y, probabilities, classes, threshold)}
            for threshold in (None, 0.2, 0.35, 0.5, 0.65, 0.8)]
