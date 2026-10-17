from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

from backend.detection import features
from backend.training import recordings
from backend.capture.csi import parse_line
from backend.detection.detector import ThresholdDetector
from backend.detection.motion import MEASURE, MotionTracker

PRE_S, POST_S = 1.5, 8.5
SOUNDS = Path("/System/Library/Sounds")
MAX_NOISY_FRAC = 0.2


def beep(name: str) -> None:
    sound = SOUNDS / f"{name}.aiff"
    if sound.exists():
        subprocess.Popen(["afplay", str(sound)], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    else:
        print("\a", end="", flush=True)


class Reader(threading.Thread):
    def __init__(self, lines) -> None:
        super().__init__(daemon=True)
        self.lines = lines
        self.buf: deque[tuple[float, str]] = deque()
        self.lock = threading.Lock()

    def run(self) -> None:
        for t, line in self.lines:
            with self.lock:
                self.buf.append((t, line))
                while t - self.buf[0][0] > 60:
                    self.buf.popleft()

    def clock(self) -> float:
        with self.lock:
            return self.buf[-1][0] if self.buf else 0.0

    def between(self, a: float, b: float) -> list[tuple[float, str]]:
        with self.lock:
            return [(t, line) for t, line in self.buf if a <= t <= b]

    def wait_until(self, target: float) -> None:
        last_change, last = time.monotonic(), self.clock()
        while self.clock() < target:
            time.sleep(0.02)
            if self.clock() != last:
                last_change, last = time.monotonic(), self.clock()
            elif time.monotonic() - last_change > 3:
                print("  (no packets for 3 s: is the SEND board powered?)", flush=True)
                last_change = time.monotonic()


def calibrate(reader: Reader) -> dict:
    tracker, fed_until = MotionTracker(), 0.0
    while tracker.calibration() is None:
        reader.wait_until(fed_until + 0.5)
        for t, line in reader.between(fed_until + 1e-9, float("inf")):
            frame = parse_line(line, t)
            if frame is not None:
                tracker.update(frame)
            fed_until = t
    return tracker.calibration()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("label", choices=recordings.LABELS)
    ap.add_argument("--n", type=int, default=10, help="how many clips to record")
    ap.add_argument("--split", choices=("train", "test"), default="train")
    ap.add_argument("--gap", type=float, default=8.0, help="seconds to get in position")
    ap.add_argument("--performer", default="vincent")
    ap.add_argument("--spot", default="test-room", help="where this was recorded")
    ap.add_argument("--port", help="receiver serial port")
    ap.add_argument("--recalibrate", action="store_true",
                    help="learn the calm room again (do this after moving the boards)")
    ap.add_argument("--replay", help=argparse.SUPPRESS)
    ap.add_argument("--root", type=Path, default=recordings.RECORDINGS, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.replay:
        from backend.capture.replay import replay_lines
        lines = replay_lines(args.replay, 1.0)
    else:
        from backend.capture.live import find_port, live_lines
        lines = live_lines(args.port or find_port())
    reader = Reader(lines)
    reader.start()

    cal_path = args.root.parent / recordings.CALIBRATION_FILE
    saved = json.loads(cal_path.read_text()) if cal_path.exists() else None
    if saved and saved["calibration"].get("measure") != MEASURE:
        saved = None
    if saved and not args.recalibrate:
        cal = saved["calibration"]
        print(f"Using the calm-room calibration from {saved['created_at']} "
              "(pass --recalibrate if you moved the boards).", flush=True)
    else:
        while True:
            print("Calibrating: everyone out of the area and nobody moving, for 10 s...", flush=True)
            cal = calibrate(reader)
            if cal["noisy_frac"] <= MAX_NOISY_FRAC:
                break
            print(f"  Something moved during {cal['noisy_frac']:.0%} of it. Trying again...", flush=True)
        cal_path.write_text(json.dumps(
            {"created_at": datetime.now().isoformat(timespec="minutes"), "calibration": cal}))
        print(f"Calibrated, saved to {cal_path}", flush=True)

    detector = ThresholdDetector()
    bad = 0
    for i in range(1, args.n + 1):
        print(f"\nClip {i}/{args.n} [{args.label}]: get in position...", flush=True)
        reader.wait_until(reader.clock() + max(0.0, args.gap - 3))
        for word in ("3", "2", "1"):
            beep("Tink")
            print(f"  {word}...", flush=True)
            reader.wait_until(reader.clock() + 1)
        beep("Glass")
        print("  GO!", flush=True)
        go_t = reader.clock()
        reader.wait_until(go_t + POST_S)

        clip = reader.between(go_t - PRE_S, go_t + POST_S)
        meta = {
            "label": args.label, "split": args.split, "performer": args.performer,
            "spot": args.spot, "go_t": go_t, "pre_s": PRE_S, "post_s": POST_S,
            "packets": len(clip), "rate_hz": round(len(clip) / (PRE_S + POST_S), 1),
            "recorded_at": datetime.now().isoformat(timespec="seconds"),
            "calibration": cal,
        }
        path = recordings.save(clip, meta, args.root)
        rec = recordings.Recording(path, meta)
        inputs = rec.decision_inputs()
        if inputs is None:
            seen = "no Motion Event detected"
        else:
            v = detector.classify(features.extract(*inputs))
            seen = f"rules say {v.label} " + ", ".join(f"{k} {x}" for k, x in v.details.items())
        problems = rec.problems()
        beep("Basso" if problems else "Pop")
        print(f"  saved {path.name} ({meta['rate_hz']} Hz): {seen}", flush=True)
        if problems:
            bad += 1
            print(f"  !! {'; '.join(problems)}. Training will skip this one; do an extra.", flush=True)

    print(f"\nDone: {args.n} x {args.label} in {args.root / args.split}, "
          f"{args.n - bad} clean, {bad} flagged")


if __name__ == "__main__":
    main()
