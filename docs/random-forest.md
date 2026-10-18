# Random Forest training and prediction

Train offline after syncing recordings, then load the saved forest once when
starting the live detector. Live predictions do not retrain it.

```bash
python -m backend.training.validate --report /tmp/csi-validation.json
python -m backend.training.train
python -m backend.run --detector rf
```

Training writes `data/model.joblib` and `data/results.json`. Use `--model` and
`--results` to keep experiments separate from the model used by the demo:

```bash
python -m backend.training.train --cv session \
  --model data/experiments/baseline.joblib \
  --results data/experiments/baseline.json
```

The defaults retain the existing forest: 300 trees, balanced class weights,
minimum two samples per leaf, unrestricted depth, random seed 0, six features,
and the label with the highest probability. Motion gating and alert timing
remain part of the existing live pipeline.

## Read the evaluation first

The trainer always attempts out-of-fold validation on **training** recordings.
Every usable training clip is scored by a separate forest that excluded that
clip's validation fold. When session IDs exist, each session stays in one fold.

`--cv auto` uses session folds when all usable training clips have IDs. Entirely
legacy data falls back to shuffled stratified clip folds with a warning. Mixed
known/unknown sessions require verified IDs or an explicit `--cv clip` choice.
Clip folds can share performers and setups and give optimistic estimates.

Session validation requires at least two sessions containing each class. Start
separate collection sessions, record all action classes in each, and reserve a
separate session for test. Fold count is capped by available class/session counts.
If there are insufficient training sessions but test scoring is explicitly
requested with usable test recordings,
the trainer can fit and report the fixed decision rule on test; it warns that
training-fold validation/threshold comparison is unavailable. Without either
usable validation or test data, it stops rather than reporting training accuracy.

Test scoring is opt-in with `--evaluate-test`. By default, training reports
training-fold validation only, even when test files exist. After settling the
settings, request the final evaluation explicitly:

```bash
python -m backend.training.train --evaluate-test
```

With that flag, the final model is fit on all usable **training** clips and
evaluated on test exactly with the supplied settings. Test is never
used to fit trees or generate the threshold comparison. A test set whose action
clips are all excluded is an error, not a fallback to cross-validation.

The results file contains:

- Confusion matrices, accuracy, fall recall/precision, and counts of missed
  falls and false fall predictions. Undefined precision/recall is null.
- Training-fold probabilities, predictions, features, and fold IDs per clip.
- A separate test evaluation and per-clip predictions when test data exists.
- Excluded action clips and their reasons; warnings about incomplete post-event
  windows. These clips are not included in the classifier metric denominators.
- Threshold comparisons using training-fold probabilities only.
- Feature importances, training file checksums, forest parameters, preprocessing
  settings, code provenance, and library versions.

These are **event classifier** metrics. They do not measure the live pipeline's
motion gates, resumed-movement rejection, countdown latency, or false alarms per
hour. Empty-room event counts remain a separate motion-detector check. RF vote
probabilities have not been calibrated as real-world fall likelihoods.

## Tune one decision at a time

The report compares argmax with fall thresholds 0.2, 0.35, 0.5, 0.65, and 0.8;
it never automatically chooses one. A threshold rule predicts fall when
`P(fall) >= threshold`; otherwise it picks the strongest non-fall class. This
can promote a fall below the competing class or suppress a weak winning fall.
Argmax and a 0.5 fall threshold are different rules in a three-class model.

After reviewing validation results, make an explicit experimental run:

```bash
python -m backend.training.train --fall-threshold 0.35 \
  --model data/experiments/threshold-035.joblib \
  --results data/experiments/threshold-035.json
```

`0.35` is an example to evaluate, not a validated deployment threshold. Lower
thresholds trade more false fall predictions for fewer missed falls. Do not
repeatedly choose settings based on test scores; keep test untouched until
validation decisions are settled.

Optional controls: `--n-estimators`, `--max-depth`, `--min-samples-leaf`,
`--fall-weight`, `--seed`, and `--folds`. `--fall-weight 3` assigns weight 3 to
falls and 1 to each other class; it replaces the default balanced weights.
No grid search or automatic hyperparameter selection is performed. Keep one
baseline and compare small, deliberate experiments before adding features.

## Loading an approved model

The bundle includes the threshold used during evaluation. The live RF detector
loads that threshold automatically and uses the same decision function as
offline evaluation. Legacy bundles without a threshold continue to use argmax.
The returned confidence is the selected class's vote probability; verdict
details separately expose `fall_probability`.

Feature names/order and saved preprocessing settings must match the running
code. Retrain after changing either. Compare reports before replacing the
default model, keep the previous artifact for rollback, and restart the live
process to load a replacement. To try an experiment without replacing the default:

```bash
python -m backend.run --detector rf --model data/experiments/threshold-035.joblib
```

The live command opens the dashboard by default. Use `--no-view` for terminal
output only, or `--view-port 8001` if the recording dashboard is using port 8000.
The selected artifact is used in both modes and by the dashboard's simulation.
Keep the room still during the initial live calibration; recording metadata is
automatically preloaded only for replay inputs.

The dashboard's recording replay adds a synthetic tail so its stillness window
can finish. This is a demo playback behavior, not additional measured evidence.
Training validation and the raw-clip experiments do not add that synthetic data.

Only load trusted local joblib artifacts.
