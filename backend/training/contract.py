"""Recording metadata and structural validation shared by recording and training."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import subprocess
import zlib
from collections import Counter
from datetime import datetime
from pathlib import Path

from backend import config
from backend.detection.motion import MEASURE

SCHEMA_VERSION = 1
VALIDATOR_VERSION = 1
MOTION_SETTINGS = (
    "WINDOW_S", "T_ON", "T_OFF", "QUIET_S", "PRE_EVENT_S", "MAX_EVENT_S",
    "STILLNESS_S", "AFTER_S", "BEFORE_GAP_S",
)
LABELS = ("fall", "sit-down", "walk", "empty")


def motion_settings() -> dict:
    return {name: getattr(config, name) for name in MOTION_SETTINGS}


def capture_context(source: str) -> dict:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=config.DATA_DIR.parent,
            stderr=subprocess.DEVNULL, text=True, timeout=2,
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=config.DATA_DIR.parent, stderr=subprocess.DEVNULL, text=True, timeout=2,
        ).strip())
    except (OSError, subprocess.SubprocessError):
        revision, dirty = None, None
    return {"source": source, "git_commit": revision, "git_dirty": dirty,
            "motion_settings": motion_settings(),
            "receivers": dict(config.RECEIVERS), "sender": config.SENDER_BOARD}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _number(value, minimum: float | None = None) -> bool:
    return (type(value) in (int, float) and math.isfinite(value)
            and (minimum is None or value >= minimum))


def validate_metadata(meta, split: str, stem: str) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
    if not isinstance(meta, dict):
        return ["metadata must be a JSON object"], warnings
    version = meta.get("schema_version")
    if version is None:
        warnings.append("legacy metadata: session/setup/provenance may be unavailable")
    elif type(version) is not int or version != SCHEMA_VERSION:
        errors.append(f"unsupported schema_version: {version!r}")
    if meta.get("label") not in LABELS:
        errors.append("invalid label")
    if meta.get("split") != split or split not in ("train", "test"):
        errors.append("split must be train/test and match the directory")
    for key in ("performer", "spot"):
        if not isinstance(meta.get(key), str) or not meta[key].strip():
            errors.append(f"{key} must be a nonempty string")
    for key in ("go_t", "pre_s", "post_s"):
        if not _number(meta.get(key), 0):
            errors.append(f"{key} must be finite and nonnegative")
    if _number(meta.get("pre_s"), 0) and _number(meta.get("post_s"), 0):
        if meta["pre_s"] + meta["post_s"] <= 0:
            errors.append("recording duration must be positive")
    if type(meta.get("packets")) is not int or meta["packets"] <= 0:
        errors.append("packets must be a positive integer")
    try:
        recorded = datetime.fromisoformat(meta["recorded_at"])
        if recorded.utcoffset() is None:
            (errors if version == SCHEMA_VERSION else warnings).append("recorded_at has no timezone")
    except (ValueError, TypeError, KeyError):
        errors.append("recorded_at must be an ISO timestamp")

    cal = meta.get("calibration")
    if not isinstance(cal, dict) or not cal:
        errors.append("calibration must be a nonempty object")
        cal = {}
    if "measure" in cal:
        if version == SCHEMA_VERSION:
            errors.append("version 1 calibration must be keyed by receiver")
        cal = {"A": cal}
    for link, entry in cal.items():
        prefix = f"calibration[{link}]"
        if link not in ("A", "B") or not isinstance(entry, dict):
            errors.append(f"{prefix}: invalid receiver or calibration")
            continue
        if entry.get("measure") != MEASURE:
            errors.append(f"{prefix}: unsupported motion measure")
        size, mask = entry.get("size"), entry.get("mask")
        if type(size) is not int or size <= 0:
            errors.append(f"{prefix}: size must be a positive integer")
        elif (not isinstance(mask, list) or not mask
              or any(type(i) is not int or not 0 <= i < size for i in mask)
              or len(set(mask)) != len(mask)):
            errors.append(f"{prefix}: mask must contain unique valid subcarrier indices")
        if not _number(entry.get("baseline"), 0) or entry.get("baseline") == 0:
            errors.append(f"{prefix}: baseline must be positive and finite")
        if not _number(entry.get("noisy_frac"), 0) or entry.get("noisy_frac", 2) > 1:
            errors.append(f"{prefix}: noisy_frac must be between 0 and 1")
    rates = meta.get("rate_hz")
    if not isinstance(rates, dict):
        if version == SCHEMA_VERSION:
            errors.append("version 1 rate_hz must be keyed by receiver")
        rates = {"A": rates}
    if set(rates) != set(cal) or any(not _number(v, 0) for v in rates.values()):
        errors.append("rate_hz must contain finite nonnegative rates for calibrated receivers")

    if version == SCHEMA_VERSION:
        for key in ("recording_id", "session_id"):
            if not isinstance(meta.get(key), str) or not meta[key].strip():
                errors.append(f"{key} must be a nonempty string")
        if meta.get("recording_id") != stem:
            errors.append("recording_id must match the filename stem")
        if meta.get("setup_id") is None:
            warnings.append("setup_id is unspecified; set --setup-id before collecting data")
        elif not isinstance(meta["setup_id"], str) or not meta["setup_id"].strip():
            errors.append("setup_id must be null or a nonempty string")
        checksum = meta.get("raw_sha256")
        if (not isinstance(checksum, str) or len(checksum) != 64
                or any(c not in "0123456789abcdef" for c in checksum)):
            errors.append("raw_sha256 must be a lowercase SHA-256 digest")
        capture = meta.get("capture")
        if not isinstance(capture, dict):
            errors.append("capture provenance is required")
        else:
            if capture.get("source") not in ("live", "replay"):
                errors.append("capture.source must be live or replay")
            if capture.get("motion_settings") != motion_settings():
                errors.append("capture motion settings differ from current preprocessing settings")
            devices = capture.get("receivers")
            if (not isinstance(devices, dict) or not set(cal) <= set(devices)
                    or any(not isinstance(v, str) or not v for v in devices.values())):
                errors.append("capture.receivers must identify every calibrated receiver")
            if not isinstance(capture.get("sender"), str) or not capture["sender"]:
                errors.append("capture.sender is required")
            if not capture.get("git_commit") or capture.get("git_dirty"):
                warnings.append("capture source is unversioned or has local changes")
    return errors, warnings


def validate_pair(meta_path: Path) -> dict:
    """Validate bytes and structure without dropping malformed rows like the live parser."""
    result = {"file": str(meta_path), "errors": [], "warnings": [], "counts": {}}
    errors, warnings = result["errors"], result["warnings"]
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, ValueError) as exc:
        errors.append(f"cannot read metadata: {exc}")
        return result
    found, cautions = validate_metadata(meta, meta_path.parent.name, meta_path.stem)
    errors.extend(found)
    warnings.extend(cautions)
    if errors:
        return result
    result.update(label=meta["label"], session_id=meta.get("session_id"))
    log_path = meta_path.with_name(meta_path.stem + ".log.gz")
    if not log_path.exists():
        errors.append("missing .log.gz partner")
        return result
    try:
        if meta.get("raw_sha256") and sha256(log_path) != meta["raw_sha256"]:
            errors.append("raw_sha256 mismatch")
        cal = meta["calibration"]
        cal = {"A": cal} if "measure" in cal else cal
        counts, previous = Counter(), -math.inf
        duration = meta["pre_s"] + meta["post_s"]
        with gzip.open(log_path, "rt") as stream:
            for n, row in enumerate(stream, 1):
                try:
                    parts = row.rstrip("\n").split("\t", 2)
                    if len(parts) == 2 and meta.get("schema_version") is None:
                        parts.insert(1, "A")
                    t_text, link, payload = parts
                    t = float(t_text)
                    if not math.isfinite(t) or t < previous:
                        raise ValueError("timestamp must be finite and nondecreasing")
                    previous = t
                    if not meta["go_t"] - meta["pre_s"] - 1e-5 <= t <= meta["go_t"] + meta["post_s"] + 1e-5:
                        raise ValueError("timestamp outside declared clip window")
                    if link not in cal:
                        raise ValueError("receiver has no calibration")
                    fields = next(csv.reader([payload], strict=True))
                    if len(fields) < 6 or fields[0] != "CSI_DATA":
                        raise ValueError("invalid CSI_DATA CSV")
                    int(fields[3])  # RSSI
                    if int(fields[-2]) not in (0, 1):
                        raise ValueError("invalid first-word flag")
                    if not fields[-1].startswith("[") or not fields[-1].endswith("]"):
                        raise ValueError("CSI array must be bracketed")
                    iq = [int(v) for v in fields[-1][1:-1].split(",")]
                    if len(iq) != 2 * cal[link]["size"]:
                        raise ValueError("I/Q length does not match calibration size")
                    counts[link] += 1
                except (ValueError, IndexError, StopIteration, csv.Error) as exc:
                    errors.append(f"row {n}: {exc}")
                    if len(errors) >= 20:
                        errors.append("stopped after 20 errors")
                        break
        result["counts"] = dict(counts)
        if not errors:
            if sum(counts.values()) != meta["packets"]:
                errors.append("packets does not match saved row count")
            rates = meta["rate_hz"]
            rates = rates if isinstance(rates, dict) else {"A": rates}
            for link in cal:
                if not counts[link]:
                    errors.append(f"no packets for calibrated receiver {link}")
                if not math.isclose(rates[link], counts[link] / duration, abs_tol=0.051, rel_tol=0):
                    errors.append(f"rate_hz[{link}] does not match row count / clip duration")
    except (OSError, EOFError, UnicodeError, zlib.error) as exc:
        errors.append(f"cannot read CSI archive: {exc}")
    return result
