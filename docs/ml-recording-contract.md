# CSI recording contract

The ML unit is a pair of files with the same globally unique stem:

```text
data/recordings/{train|test}/{recording_id}.log.gz
data/recordings/{train|test}/{recording_id}.json
```

R2 sync uses these same paths as object keys. This contract does not change
sync or introduce a database dependency. Treat raw recordings as immutable.
The files transfer separately: validate after syncing before training.

## Raw CSI

The gzip archive contains UTF-8 rows:

```text
<Unix receive time in seconds>\t<A|B>\t<original CSI_DATA CSV line>\n
```

Timestamps use the laptop receive clock, are nondecreasing, and may repeat.
Preserve the entire CSI_DATA CSV, including its quoted final integer array.
The parser reads RSSI at field index 3, the invalid-first-word flag in the
penultimate field, and interleaved imaginary/real integers in the final field.
The array must contain exactly twice the receiver calibration's `size` values.
Current hardware produces 384 integers / 192 amplitude channels. `mask` selects
usable channels; it is not a fixed 166-channel requirement.

## Metadata version 1

| Field | Contract |
| --- | --- |
| `schema_version` | Integer `1` |
| `recording_id` | Filename stem, generated with a UUID |
| `session_id` | Collection group shared across action batches; never split across train/test |
| `setup_id` | Explicit board placement identifier, or null if unspecified |
| `label` | Human ground truth: `fall`, `sit-down`, `walk`, `empty` |
| `split` | `train` or `test`, matching the parent directory |
| `performer`, `spot` | Nonempty performer and location identifiers |
| `go_t` | Unix receive-clock seconds at the action cue |
| `pre_s`, `post_s` | Nonnegative requested durations before/after GO, positive sum |
| `packets` | Total saved rows across receivers |
| `rate_hz` | Receiver-to-rate map: saved rows / (`pre_s` + `post_s`), rounded to one decimal |
| `recorded_at` | Save time in ISO 8601 with timezone offset |
| `calibration` | Receiver-to-calibration map; snapshot used for this recording |
| `raw_sha256` | SHA-256 of the exact compressed `.log.gz` bytes |
| `capture` | Source (`live`/`replay`), Git commit and dirty status, configured receiver/sender identities, motion settings |
| `quality` | Recorder adds current quality flags and validator version after analyzing the saved clip |

Each calibration requires `measure` (currently `mean-shift-0.5`), positive integer
`size`, a nonempty list of unique zero-based `mask` indices below `size`, a finite
positive `baseline`, and `noisy_frac` in [0, 1]. Every receiver must have packets.

`capture.motion_settings` snapshots the window, motion thresholds, quiet time,
event limits, and pre/post feature observation settings. Version 1 recordings
with settings different from the current code fail validation: consciously
restore the collection settings or implement an explicit migration/replay path
before mixing preprocessing versions. Do not edit metadata merely to pass.
Git dirty status means a commit alone does not fully reproduce capture code;
prefer a clean committed revision for collection. Hardware IDs record configured
identities, not device discovery results. Replay recordings are explicitly marked.

## Collection and validation

```bash
python -m backend.training.record --session-id collection-01 --setup-id room-a-layout-01
python -m backend.training.validate --report /tmp/csi-validation.json
python -m backend.training.train
```

One dashboard launch shares a generated session ID across its action batches.
Use an explicit ID to continue that collection group after restarting. Change
setup ID when board placement changes, and start a separate session for test
collection. The optional CLI settings do not require dashboard UI changes.

Validation exits nonzero for missing pairs, malformed metadata/CSI, bad checksum,
inconsistent packet counts/rates, missing calibration, unsupported settings, or
session leakage across train/test. An empty dataset also gives a nonzero CLI exit.
Training performs structural validation before loading any recordings, even with
`--keep-flagged`; that option only overrides existing motion quality filters.

The full validator also replays motion extraction and reports quality warnings,
missing events, selected receiver, event duration, truncated events, and available
post-event time. These warnings do not reject the pair. They require review for
collection quality and end-to-end evaluation; they are not Random Forest scores.
Quality flags are recomputed rather than trusting a stale saved `quality` field.
This command does not train a forest or run the full live alert state machine.

## Legacy recordings

Unversioned metadata remains readable. Scalar rates and unwrapped calibration
refer to receiver A; two-column logs also default to A. Naive local timestamps
and missing provenance produce warnings. Missing IDs are not fabricated, and
legacy sessions cannot be checked for split leakage. Extra historical fields
such as `calibration_original` are preserved and ignored by validation.

Full observation time is not guaranteed by clip duration alone. The validator
warns when the selected event ends too late to observe `STILLNESS_S` afterward;
the existing trainer still uses its clipped post-event window. Resolve this
with collection timing and pipeline evaluation before interpreting model scores.
