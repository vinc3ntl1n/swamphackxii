# Random Forest training and prediction

Train offline after syncing recordings, then load the saved forest once when
starting the live detector. Live predictions do not retrain it.

Live detection defaults to RF. The model must be a trusted local file; prediction
does not access R2 or a database. Set `SWAMP_RF_MODEL` in the repository's `.env`
or pass `--model` to select the copied artifact. Paths from another teammate's
machine need a local copy rather than the same absolute pathname.

```bash
# .env (or export these variables in the shell)
SWAMP_DETECTOR=rf
SWAMP_RF_MODEL=data/experiments/batch-20261017-233929/baseline.joblib
```

`SWAMP_DETECTOR=threshold` or `--detector threshold` explicitly selects the
heuristic. A missing, malformed or incompatible requested RF stops startup;
it never silently selects the heuristic.

The shared baseline and threshold-035 artifacts were saved with scikit-learn
1.6.1 and NumPy 2.1.3. Create a separate compatible environment (these pinned
versions do not support Python 3.14):

```bash
python3.12 -m venv .venv-rf
source .venv-rf/bin/activate
python -m pip install -r requirements-rf.txt
python -m backend.run --model data/experiments/batch-20261017-233929/baseline.joblib
```

Startup rejects declared library-version mismatches and scikit-learn's
inconsistent-version pickle warning. Versionless legacy bundles remain readable
when scikit-learn can load them without a compatibility warning. Re-evaluate a
retrained model before substituting it for the shared baseline.

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
Verdict details include all six feature values in order (`peak`, `static_change`,
`duration_s`, `mean_energy`, `time_to_peak_s`, `stop_s`) and `class_probabilities`.
These probabilities are model scores, not calibrated real-world risk estimates.

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

## Diagnosing live misses

Each live run saves `data/sessions/<timestamp>.log` and a matching
`<timestamp>.events.jsonl`. The terminal prints both paths. Raw logs retain every
valid row read from serial, including packets skipped by live detection. Training
capture remains lossless. To choose the diagnostics path, add
`--diagnostics /tmp/rf-events.jsonl`; replay writes diagnostics only when this flag
is supplied.

Live detection keeps at most 256 queued packets and discards packets older than
0.5 seconds. Known capture loss clears affected motion windows and pending
evidence before another packet is classified. A receiver without valid frames
for 2 seconds cannot stay stuck in `moving` and block a healthy receiver. Live
receiver suspension requires calibration when it returns; short gaps first warm
fresh motion windows. Missing packets never count as stillness. Confirmed
countdowns continue during signal loss.

The JSONL records explain each stage:

- `run_started`: selected artifact, checksum, saved threshold and feature order.
- `capture_health`: queue depth, packet age, dropped counts/reasons/links.
- `runtime`: receiver health, processing time, pending candidate and alert state.
- `detection`: motion starts/ends, `waiting_for_quiet`, `candidate_started`,
  `resumed_movement`, packet gaps, missing post-event observations, stale links,
  or `classified` with the six RF inputs, probabilities and inference time.

The capture reader drains serial bytes in chunks and reassembles complete CSV
lines, avoiding a Python read for each byte. Partial lines have a size limit;
reconnects never join bytes from two connections. `capture_health.links` reports
bytes waiting in the serial driver, bytes/lines received, valid and malformed
CSI rows, read timeouts, and connection counts per receiver. These distinguish
quiet signal from missing or corrupt acquisition.

For the ESP32-S3 25-column format, capture also preserves the sender packet ID
and receiver microsecond timestamp. `links.<link>.source` compares each
receiver's clock progression with monotonic laptop receipt time, handles
32-bit wrap and restart, and reports `source_lag_ms`, `source_step_ms`,
`host_step_ms`, `skipped_ids` and `clock_event`. Source IDs can skip before USB;
their gaps are diagnostic and do not automatically cancel events. A known clock
restart or serial reconnect requires fresh calibration on that receiver.

`source_lag_ms` measures increased delay relative to the best arrival offset
seen during this connection. It is **not absolute latency**: receiver/laptop
clocks are unsynchronized, initial buffered data may already be old, and clock
drift can contribute. `source_age_known` remains false. Source time delivered
more than 0.5 seconds faster than host time is separately flagged as
`source_compressed`; affected queued evidence is discarded. Excessive relative
source delay plus Python queue/lock age is discarded as `source_backlog` or
`stale_capture`. Raw valid CSI rows remain saved even when detection skips them.
Legacy CSVs without device timing retain host-based freshness checks.

Runtime `capture_age_ms` uses monotonic receipt age for live frames, so a laptop
wall-clock change cannot make an old packet appear fresh. The separate
`relative_source_age_ms` includes the estimated source delay. Neither changes
the host timestamps or amplitude values used by the existing RF features.
The dashboard shows `warming` until fresh motion-energy windows exist and
`invalid` for corrupt or all-zero usable CSI. Zero CSI is never treated as a
quiet-room observation.

For a missed fall, first check whether motion started, then whether a candidate
was created, then whether `classified` occurred. A missing classification with
a rejection reason is a pipeline issue; a non-fall RF verdict has its model
inputs and scores recorded. An event with no motion start can be investigated
in the raw CSI and runtime energy samples. Packet age measures time since the
laptop read the serial row; it cannot measure bytes already buffered in the
receiver or USB driver.

These fixes do not change the saved forest or its prediction rule. The existing
stillness delay and rejection when movement resumes still apply and are now
visible in diagnostics. Replaying recordings verifies integration, but a live
receiver test is still needed to check USB behavior and room conditions.

Only load trusted local joblib artifacts.
