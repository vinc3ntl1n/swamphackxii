"""Train a reusable forest and evaluate event classification without training/test overlap."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn

from sklearn.ensemble import RandomForestClassifier

from backend import config
from backend.detection import features
from backend.detection.detector import MODEL_PATH, predict_labels
from backend.training import contract, evaluation, recordings
from backend.training.validate import audit

RESULTS_PATH = config.DATA_DIR / "results.json"
LABELS = list(recordings.EVENT_LABELS)


def featurize(recs: list[recordings.Recording]) -> tuple[np.ndarray, list[str], list[str]]:
    X, y, missed = [], [], []
    for rec in recs:
        inputs = rec.decision_inputs()
        if inputs is None:
            missed.append(rec.log_path.name)
            continue
        x = features.extract(*inputs)
        if not np.isfinite(x).all():
            raise ValueError(f"{rec.log_path.name}: nonfinite features")
        X.append(x)
        y.append(rec.label)
    return np.array(X).reshape(-1, len(features.NAMES)), y, missed


def report(y_true, y_pred, how: str) -> dict:
    result = {"how": how, **evaluation.metrics(y_true, y_pred)}
    def percent(value):
        return "n/a" if value is None else f"{value:.0%}"
    print(f"\nScored by {how}: accuracy {result['accuracy']:.0%}   "
          f"fall recall {percent(result['fall_recall'])}   fall precision {percent(result['fall_precision'])}")
    print("rows = actual label, columns = predicted label")
    print(" " * 10 + "".join(f"{label:>10}" for label in LABELS))
    for label, row in zip(LABELS, result["confusion"]):
        print(f"{label:>10}" + "".join(f"{n:>10}" for n in row))
    return result


def prediction_rows(recs, X, y, probabilities, classes, threshold, fold_ids=None) -> list[dict]:
    predictions = predict_labels(probabilities, classes, threshold)
    rows = []
    for i, rec in enumerate(recs):
        rows.append({"file": rec.log_path.name, "label": y[i], "prediction": str(predictions[i]),
                     "session_id": rec.meta.get("session_id"),
                     "fold": int(fold_ids[i]) if fold_ids is not None else None,
                     "features": dict(zip(features.NAMES, X[i].tolist())),
                     "probabilities": dict(zip(classes.tolist(), probabilities[i].tolist()))})
    return rows


def positive_int(value):
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return n


def positive_float(value):
    n = float(value)
    if not np.isfinite(n) or n <= 0:
        raise argparse.ArgumentTypeError("must be positive and finite")
    return n


def probability(value):
    n = float(value)
    if not np.isfinite(n) or not 0 <= n <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spot", help="only use recordings made at this spot")
    ap.add_argument("--keep-flagged", action="store_true", help="also use motion-quality flagged clips")
    ap.add_argument("--evaluate-test", action="store_true",
                    help="score the fixed settings on held-out test recordings (use after validation decisions)")
    ap.add_argument("--fall-threshold", type=probability,
                    help="predict fall at or above this probability; default: highest probability wins")
    ap.add_argument("--n-estimators", type=positive_int, default=300)
    ap.add_argument("--max-depth", type=positive_int, help="default: unrestricted depth")
    ap.add_argument("--min-samples-leaf", type=positive_int, default=2)
    ap.add_argument("--fall-weight", type=positive_float,
                    help="explicit fall class weight; other classes get 1 (default: balanced weights)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cv", choices=("auto", "session", "clip"), default="auto",
                    help="auto groups by session when all IDs exist; legacy data uses clip folds")
    ap.add_argument("--folds", type=positive_int, default=5)
    ap.add_argument("--root", type=Path, default=recordings.RECORDINGS)
    ap.add_argument("--model", type=Path, default=MODEL_PATH)
    ap.add_argument("--results", type=Path, default=RESULTS_PATH)
    args = ap.parse_args()
    if args.folds < 2:
        ap.error("--folds must be at least 2")

    invalid = [r for r in audit(args.root, quality=False) if r["errors"]]
    if invalid:
        for result in invalid:
            print(f"{result['file']}: {'; '.join(result['errors'])}")
        raise SystemExit("Recording contract validation failed. Run python -m backend.training.validate.")

    all_recs = [r for r in recordings.load(root=args.root)
                if not args.spot or r.meta.get("spot") == args.spot]
    exclusions = []
    def pick(split):
        chosen = []
        for rec in all_recs:
            if rec.meta["split"] != split or rec.label not in LABELS:
                continue
            problems = rec.problems()
            if problems and not args.keep_flagged:
                exclusions.append({"file": rec.log_path.name, "split": split,
                                   "label": rec.label, "reasons": problems})
            else:
                chosen.append(rec)
        return chosen

    train = pick("train")
    test = pick("test") if args.evaluate_test else []
    X, y, missed = featurize(train)
    usable = [r for r in train if r.log_path.name not in missed]
    Xt, yt, missed_t = featurize(test)
    usable_t = [r for r in test if r.log_path.name not in missed_t]
    for split, recs, omitted in (("train", train, missed), ("test", test, missed_t)):
        for rec in recs:
            if rec.log_path.name in omitted:
                exclusions.append({"file": rec.log_path.name, "split": split, "label": rec.label,
                                   "reasons": ["no Motion Event; omitted from classifier scoring"]})
    print(f"train: {dict(Counter(y))}; {len(exclusions)} action clips excluded across train/test")
    if "fall" not in y or len(set(y)) < 2:
        raise SystemExit("Need usable fall recordings and at least one non-fall class in train/.")
    test_present = args.evaluate_test and any(r.meta["split"] == "test" and r.label in LABELS for r in all_recs)
    if args.evaluate_test and not test_present:
        raise SystemExit("--evaluate-test requires action recordings in test/.")
    if test_present and not yt:
        raise SystemExit("Test recordings exist but none are usable; fix them before reporting test scores.")

    weights = ({label: args.fall_weight if label == "fall" else 1.0 for label in set(y)}
               if args.fall_weight is not None else "balanced")
    model = RandomForestClassifier(n_estimators=args.n_estimators, class_weight=weights,
                                   max_depth=args.max_depth, min_samples_leaf=args.min_samples_leaf,
                                   random_state=args.seed)
    warnings = []
    sessions = [r.meta.get("session_id") for r in usable]
    validation = None
    try:
        oof = evaluation.out_of_fold(model, X, y, sessions, args.cv, args.folds, args.seed)
    except ValueError as exc:
        if not test_present:
            raise SystemExit(f"Cannot validate: {exc}. Collect more sessions or explicitly use --cv clip.")
        warnings.append(f"Training-fold validation unavailable: {exc}; no threshold comparison generated.")
    else:
        how = f"{oof['folds']}-fold {oof['mode']} cross-validation on train"
        pred = predict_labels(oof["probabilities"], oof["classes"], args.fall_threshold)
        validation = report(y, pred, how)
        validation.update(mode=oof["mode"], folds=oof["folds"],
                          predictions=prediction_rows(usable, X, y, oof["probabilities"], oof["classes"],
                                                      args.fall_threshold, oof["fold_ids"]),
                          threshold_comparison=evaluation.threshold_comparison(
                              y, oof["probabilities"], oof["classes"]))
        if oof["mode"] == "clip":
            warnings.append("Clip folds may share performers/setups/sessions; this is not session-held-out performance.")
        print("\nTraining-fold threshold comparison (not selected automatically):")
        for row in validation["threshold_comparison"]:
            print(f"  {str(row['fall_threshold']):>5}: detected {row['true_falls_detected']}/{row['fall_count']} falls; "
                  f"{row['false_fall_predictions']} false fall predictions")

    model.fit(X, y)
    test_results = None
    if yt:
        proba_t = model.predict_proba(Xt)
        test_results = report(yt, predict_labels(proba_t, model.classes_, args.fall_threshold),
                              f"held-out test set ({len(yt)} usable recordings)")
        test_results["predictions"] = prediction_rows(usable_t, Xt, yt, proba_t, model.classes_, args.fall_threshold)
    scored = test_results if test_results is not None else validation
    results = {key: value for key, value in scored.items()
               if key not in ("predictions", "threshold_comparison")}
    results.update(validation=validation, test=test_results,
                   scope="event classifier only; motion gating and live alert flow are not scored",
                   exclusions=exclusions, fall_threshold=args.fall_threshold,
                   feature_importances=dict(zip(features.NAMES, model.feature_importances_.tolist())))
    observations = []
    for rec in usable + usable_t:
        event = rec.event()
        ts = rec._run()[event.link][1]
        if ts[-1] < event.end + config.STILLNESS_S:
            observations.append(rec.log_path.name)
    if observations:
        warnings.append(f"{len(observations)} usable recordings have incomplete post-event observation time.")
    results["incomplete_observation"] = observations
    unseen = sorted(set(yt) - set(y))
    if unseen:
        warnings.append(f"Test labels absent from training: {unseen}; the forest cannot predict these classes.")
    empties = [r for r in all_recs if r.label == "empty"
               and (r.meta["split"] == "train" or args.evaluate_test)]
    if empties:
        results["empty_triggered"] = [sum(r.event() is not None for r in empties), len(empties)]
    results["warnings"] = warnings
    for warning in warnings:
        print(f"WARNING: {warning}")

    manifest = [{"file": str(r.log_path.relative_to(args.root)), "label": r.label,
                 "session_id": r.meta.get("session_id"), "setup_id": r.meta.get("setup_id"),
                 "performer": r.meta.get("performer"), "spot": r.meta.get("spot"),
                 "raw_sha256": contract.sha256(r.log_path)} for r in usable]
    code = contract.capture_context("replay")
    provenance = {"trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  "train_counts": dict(Counter(y)), "features": features.NAMES,
                  "fall_threshold": args.fall_threshold, "motion_settings": contract.motion_settings(),
                  "parameters": model.get_params(), "training_recordings": manifest,
                  "library_versions": {"scikit-learn": sklearn.__version__, "numpy": np.__version__},
                  "training_code": {key: code[key] for key in ("git_commit", "git_dirty")}}
    results.update(provenance)
    bundle = {**provenance, "bundle_version": 1, "model": model,
              "labels": model.classes_.tolist(), "counts": dict(Counter(y)),
              "evaluation_summary": {"how": results["how"], "accuracy": results["accuracy"],
                                     "fall_recall": results["fall_recall"],
                                     "fall_precision": results["fall_precision"],
                                     "scope": results["scope"], "warnings": warnings}}
    args.model.parent.mkdir(parents=True, exist_ok=True)
    args.results.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.model)
    args.results.write_text(json.dumps(results, indent=2, allow_nan=False))
    print(f"\nsaved {args.model} and {args.results}")


if __name__ == "__main__":
    main()
