"""Audit local CSI/metadata pairs before training; never modifies recordings."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend import config
from backend.training import contract, recordings


def audit(root: Path, *, quality: bool = True) -> list[dict]:
    results = []
    sessions: dict[str, set[str]] = {}
    for path in sorted(root.glob("*/*.json")):
        result = contract.validate_pair(path)
        results.append(result)
        if result["errors"]:
            continue
        meta = json.loads(path.read_text())
        if meta.get("session_id"):
            sessions.setdefault(meta["session_id"], set()).add(meta["split"])
        if quality:
            rec = recordings.Recording(path.with_name(path.stem + ".log.gz"), meta)
            result["quality_flags"] = rec.problems()
            result["warnings"].extend(result["quality_flags"])
            event = rec.event()
            if event is None:
                if rec.label != "empty":
                    result["warnings"].append("no Motion Event; classifier training will omit this clip")
            else:
                ts = rec._run()[event.link][1]
                result["selected_link"] = event.link
                result["event_duration_s"] = event.duration
                result["post_event_available_s"] = float(ts[-1] - event.end)
                if ts[-1] < event.end + config.STILLNESS_S:
                    result["warnings"].append("incomplete post-event observation window")
                if event.cut_short:
                    result["warnings"].append("event cut short; live pipeline bypasses the forest")
    for result in results:
        if len(sessions.get(result.get("session_id"), set())) > 1:
            result["errors"].append("session_id occurs in both train and test (evaluation leakage)")
    for path in sorted(root.glob("*/*.log.gz")):
        meta = path.with_name(path.name.removesuffix(".log.gz") + ".json")
        if not meta.exists():
            results.append({"file": str(path), "errors": ["missing .json partner"], "warnings": []})
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=recordings.RECORDINGS)
    parser.add_argument("--report", type=Path, help="write a per-recording JSON report")
    args = parser.parse_args()
    results = audit(args.root)
    for result in results:
        status = "ERROR" if result["errors"] else "WARN" if result["warnings"] else "OK"
        print(f"{status}: {result['file']}")
        for message in result["errors"] + result["warnings"]:
            print(f"  {message}")
    failed = sum(bool(r["errors"]) for r in results)
    print(f"{len(results)} files audited; {failed} invalid")
    if args.report:
        args.report.write_text(json.dumps({"validator_version": contract.VALIDATOR_VERSION,
                                          "recordings": results}, indent=2))
    if not results:
        print("No recordings found. Sync your data or pass --root.")
    raise SystemExit(1 if failed or not results else 0)


if __name__ == "__main__":
    main()
