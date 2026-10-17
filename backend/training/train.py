from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from backend import config
from backend.detection import features
from backend.detection.detector import MODEL_PATH
from backend.training import recordings

RESULTS_PATH = config.DATA_DIR / "results.json"
LABELS = list(recordings.EVENT_LABELS)


def featurize(recs: list[recordings.Recording]) -> tuple[np.ndarray, list[str], list[str]]:
    X, y, missed = [], [], []
    for rec in recs:
        inputs = rec.decision_inputs()
        if inputs is None:
            missed.append(rec.log_path.name)
            continue
        X.append(features.extract(*inputs))
        y.append(rec.label)
    return np.array(X).reshape(-1, len(features.NAMES)), y, missed


def report(y_true: list[str], y_pred: list[str], how: str) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    acc = float(np.trace(cm) / max(cm.sum(), 1))
    fall = LABELS.index("fall")
    recall = float(cm[fall, fall] / max(cm[fall].sum(), 1))
    precision = float(cm[fall, fall] / max(cm[:, fall].sum(), 1))
    print(f"\nScored by {how}:  accuracy {acc:.0%}   fall recall {recall:.0%}   "
          f"fall precision {precision:.0%}")
    print("rows = what it was, columns = what the model said")
    print(" " * 10 + "".join(f"{l:>10}" for l in LABELS))
    for i, l in enumerate(LABELS):
        print(f"{l:>10}" + "".join(f"{n:>10}" for n in cm[i]))
    return {"how": how, "labels": LABELS, "confusion": cm.tolist(), "accuracy": acc,
            "fall_recall": recall, "fall_precision": precision}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spot", help="only use Recordings made at this spot")
    ap.add_argument("--keep-flagged", action="store_true",
                    help="also train on clips record.py flagged (moved before GO, etc.)")
    ap.add_argument("--root", type=Path, default=recordings.RECORDINGS, help=argparse.SUPPRESS)
    ap.add_argument("--model", type=Path, default=MODEL_PATH, help=argparse.SUPPRESS)
    ap.add_argument("--results", type=Path, default=RESULTS_PATH, help=argparse.SUPPRESS)
    args = ap.parse_args()

    def pick(split: str) -> list[recordings.Recording]:
        recs = [r for r in recordings.load(split, args.root) if r.label in LABELS]
        recs = [r for r in recs if not args.spot or r.meta.get("spot") == args.spot]
        if args.keep_flagged:
            return recs
        flagged = [r for r in recs if r.problems()]
        if flagged:
            counts = Counter(r.label for r in flagged)
            print(f"{split}: skipping {len(flagged)} flagged clips {dict(counts)} (--keep-flagged to use them)")
        return [r for r in recs if r not in flagged]

    train, test = pick("train"), pick("test")
    X, y, missed = featurize(train)
    print(f"train: {dict(Counter(y))}" + (f"  (no Motion Event in {len(missed)}: {missed})" if missed else ""))
    if len(set(y)) < 2:
        raise SystemExit("Need Recordings of at least two labels in data/recordings/train/.")

    print("\nmedian feature per label (use these to hand-tune config.py too):")
    print(f"{'':>17}" + "".join(f"{l:>10}" for l in LABELS if l in y))
    for j, name in enumerate(features.NAMES):
        row = [np.median(X[[k for k, l in enumerate(y) if l == lab], j]) for lab in LABELS if lab in y]
        print(f"{name:>17}" + "".join(f"{v:>10.2f}" for v in row))

    model = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                   min_samples_leaf=2, random_state=0)
    if test:
        Xt, yt, missed_t = featurize(test)
        model.fit(X, y)
        results = report(yt, list(model.predict(Xt)), f"test set ({len(yt)} Recordings)")
        if missed_t:
            print(f"(test Recordings with no Motion Event: {missed_t})")
    else:
        folds = min(5, min(Counter(y).values()))
        if folds < 2:
            raise SystemExit("Need at least 2 Recordings per label for cross-validation.")
        pred = cross_val_predict(model, X, y, cv=StratifiedKFold(folds, shuffle=True, random_state=0))
        results = report(y, list(pred), f"{folds}-fold cross-validation on train (no test set yet)")
        model.fit(X, y)

    ranked = sorted(zip(model.feature_importances_, features.NAMES), reverse=True)
    print("\nmost useful features: " + ", ".join(f"{n} {w:.2f}" for w, n in ranked[:5]))

    empties = [r for r in recordings.load("train", args.root) + recordings.load("test", args.root)
               if r.label == "empty"]
    if empties:
        triggered = sum(r.event() is not None for r in empties)
        print(f"empty-room Recordings that produced a Motion Event: {triggered}/{len(empties)}")
        results["empty_triggered"] = [triggered, len(empties)]

    joblib.dump({"model": model, "features": features.NAMES, "labels": model.classes_.tolist(),
                 "trained_at": datetime.now().isoformat(timespec="seconds"),
                 "counts": dict(Counter(y))}, args.model)
    results["trained_at"] = datetime.now().isoformat(timespec="seconds")
    results["train_counts"] = dict(Counter(y))
    args.results.write_text(json.dumps(results, indent=1))
    print(f"\nsaved {args.model} and {args.results}")


if __name__ == "__main__":
    main()
