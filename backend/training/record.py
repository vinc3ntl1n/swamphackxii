from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Iterator

from backend import config
from backend.detection import features
from backend.training import recordings
from backend.capture.csi import DEFAULT_LINK, parse_line
from backend.detection.detector import ThresholdDetector
from backend.detection.motion import MEASURE, MotionTracker

PRE_S, POST_S = 1.5, 8.5
SOUNDS = Path("/System/Library/Sounds")
MAX_NOISY_FRAC = 0.2

Lines = Iterator[tuple[float, str, str]]


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
        self.buf: deque[tuple[float, str, str]] = deque()
        self.lock = threading.Lock()
        self.ended = False

    def run(self) -> None:
        try:
            for t, link, line in self.lines:
                with self.lock:
                    self.buf.append((t, link, line))
                    while t - self.buf[0][0] > 60:
                        self.buf.popleft()
        finally:
            self.ended = True

    def clock(self) -> float:
        with self.lock:
            return self.buf[-1][0] if self.buf else 0.0

    def between(self, a: float, b: float) -> list[tuple[float, str, str]]:
        with self.lock:
            return [row for row in self.buf if a <= row[0] <= b]

    def links(self) -> list[str]:
        with self.lock:
            return sorted({link for _, link, _ in self.buf})

    def wait_first(self) -> None:
        self.wait_until(1.0)

    def wait_until(self, target: float) -> None:
        last_change, last = time.monotonic(), self.clock()
        while self.clock() < target:
            if self.ended:
                sys.exit("The packet stream ended before the session finished.")
            time.sleep(0.02)
            if self.clock() != last:
                last_change, last = time.monotonic(), self.clock()
            elif time.monotonic() - last_change > 3:
                print("  (no packets for 3 s: is the SEND board powered?)", flush=True)
                last_change = time.monotonic()


class View:
    def __init__(self, port: int | None, cal: dict | None, label: str, n: int, link: str) -> None:
        self.hub = self.feed = None
        self.base = {"label": label, "n": n}
        self.link = link
        self.tracker = MotionTracker(link)
        self.pending = cal
        self.failed = False
        self.t = 0.0
        if port is None:
            return
        try:
            from backend import server
            hub = server.Hub()
            url = server.serve(hub, port)
        except Exception as e:
            print(f"  (live view off: {e!r}; recording anyway)", flush=True)
            return
        if url is None:
            print(f"  (live view off: port {port} is busy; recording anyway)", flush=True)
            return
        print(f"Live view at {url}", flush=True)
        server.open_browser(url)
        self.hub, self.feed = hub, server.SignalFeed(hub)

    def watch(self, lines: Lines) -> Lines:
        return lines if self.hub is None else self._watch(lines)

    def _watch(self, lines: Lines) -> Lines:
        for t, link, line in lines:
            if not self.failed and link == self.link:
                self.t = t
                try:
                    self._take(t, line)
                except Exception as e:
                    self._fail(e)
            yield t, link, line

    def _take(self, t: float, line: str) -> None:
        if self.pending is not None:
            cal, self.pending = recordings.per_link(self.pending), None
            if self.link in cal:
                self.tracker.load_calibration(cal[self.link])
        frame = parse_line(line, t, self.link)
        if frame is None:
            return
        info, _ = self.tracker.update(frame)
        if info.amp is None and self.tracker.size is not None:
            return
        self.feed.add(info, frame.rssi, frame.amp if info.amp is None else info.amp)

    def calibrated(self, cal: dict) -> None:
        self.pending = cal

    def publish(self, msg: dict) -> None:
        if self.hub is None or self.failed:
            return
        try:
            self.hub.publish(msg)
        except Exception as e:
            self._fail(e)

    def record(self, phase: str, t: float, **fields) -> None:
        self.publish({"type": "record", "phase": phase, "t": t, **self.base, **fields})

    def close(self) -> None:
        if self.hub is not None:
            self.hub.close(1.0)

    def _fail(self, e: Exception) -> None:
        if self.failed:
            return
        self.failed = True
        print(f"  (live view stopped: {e!r}; recording continues)", flush=True)
        with contextlib.suppress(Exception):
            self.hub.publish({"type": "record", "phase": "failed", "t": self.t, **self.base, "clip": 0,
                              "error": repr(e)})


def calibrate(reader: Reader, links: list[str]) -> dict[str, dict]:
    trackers, fed_until = {link: MotionTracker(link) for link in links}, 0.0
    while any(tracker.calibration() is None for tracker in trackers.values()):
        reader.wait_until(fed_until + 0.5)
        for t, link, line in reader.between(fed_until + 1e-9, float("inf")):
            frame = parse_line(line, t, link)
            if frame is not None and link in trackers:
                trackers[link].update(frame)
            fed_until = t
    return {link: tracker.calibration() for link, tracker in trackers.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("label", choices=recordings.LABELS)
    ap.add_argument("--n", type=int, default=10, help="how many clips to record")
    ap.add_argument("--split", choices=("train", "test"), default="train")
    ap.add_argument("--gap", type=float, default=8.0, help="seconds to get in position")
    ap.add_argument("--performer", default="vincent")
    ap.add_argument("--spot", default="test-room", help="where this was recorded")
    ap.add_argument("--port", help="receiver A serial port")
    ap.add_argument("--port-b", help="receiver B serial port")
    ap.add_argument("--recalibrate", action="store_true",
                    help="learn the calm room again (do this after moving the boards)")
    ap.add_argument("--no-view", action="store_true", help="don't serve or open the live page")
    ap.add_argument("--view-port", type=int, default=8000, help="port for the live page")
    ap.add_argument("--replay", help=argparse.SUPPRESS)
    ap.add_argument("--root", type=Path, default=recordings.RECORDINGS, help=argparse.SUPPRESS)
    args = ap.parse_args()

    cal_path = args.root.parent / recordings.CALIBRATION_FILE
    saved = json.loads(cal_path.read_text()) if cal_path.exists() else None
    saved_cals = {link: cal for link, cal in recordings.per_link(saved["calibration"]).items()
                  if cal.get("measure") == MEASURE} if saved else {}

    if args.replay:
        from backend.capture.replay import replay_lines
        lines = replay_lines(args.replay, 1.0)
    else:
        from backend.capture.live import live_lines, receivers
        ports = receivers(args.port, args.port_b)
        lines = live_lines(ports)
    view_link = DEFAULT_LINK if args.replay or DEFAULT_LINK in ports else min(ports)
    view = View(None if args.no_view else args.view_port,
                saved_cals or None, args.label, args.n, view_link)
    reader = Reader(view.watch(lines))
    reader.start()
    view.publish({"type": "source", "mode": "REPLAY" if args.replay else "LIVE"})
    view.publish({"type": "config", "t_on": config.T_ON, "t_off": config.T_OFF,
                  "pre_s": PRE_S, "post_s": POST_S, "calibrate_s": config.CALIBRATE_S})

    detector = ThresholdDetector()
    recorded, bad, finished = 0, 0, False
    try:
        reader.wait_first()
        reader.wait_until(reader.clock() + 1.0)
        links = reader.links()
        if not args.replay and links != list(ports):
            silent = ", ".join(link for link in ports if link not in links)
            print(f"Receiver {silent} is plugged in but sends nothing; recording {', '.join(links)} only.",
                  flush=True)
        print(f"Recording link{'s' if len(links) > 1 else ''} {', '.join(links)}.", flush=True)
        if not args.recalibrate and all(link in saved_cals for link in links):
            cal = {link: saved_cals[link] for link in links}
            print(f"Using the calm-room calibration from {saved['created_at']} "
                  "(pass --recalibrate if you moved the boards).", flush=True)
        else:
            if not args.recalibrate:
                new = [link for link in links if link not in saved_cals]
                print(f"No saved calibration for link{'s' if len(new) > 1 else ''} {', '.join(new)} yet.",
                      flush=True)
            while True:
                print("Calibrating: everyone out of the area and nobody moving, for 10 s...", flush=True)
                view.record("calibrating", reader.clock(), clip=0)
                cal = calibrate(reader, links)
                noisy = {link: c["noisy_frac"] for link, c in cal.items() if c["noisy_frac"] > MAX_NOISY_FRAC}
                if not noisy:
                    break
                worst = ", ".join(f"{f:.0%} on link {link}" for link, f in noisy.items())
                print(f"  Something moved during {worst}. Trying again...", flush=True)
            cal_path.write_text(json.dumps(
                {"created_at": datetime.now().isoformat(timespec="minutes"), "calibration": cal}))
            view.calibrated(cal)
            print(f"Calibrated, saved to {cal_path}", flush=True)

        for i in range(1, args.n + 1):
            print(f"\nClip {i}/{args.n} [{args.label}]: get in position...", flush=True)
            start = reader.clock()
            until = start + max(0.0, args.gap - 3)
            view.record("position", start, clip=i, until=until)
            reader.wait_until(until)
            for word in ("3", "2", "1"):
                t = reader.clock()
                view.record("count", t, clip=i, word=word)
                beep("Tink")
                print(f"  {word}...", flush=True)
                reader.wait_until(t + 1)
            beep("Glass")
            print("  GO!", flush=True)
            go_t = reader.clock()
            view.record("go", go_t, clip=i)
            reader.wait_until(go_t + POST_S)

            clip = reader.between(go_t - PRE_S, go_t + POST_S)
            meta = {
                "label": args.label, "split": args.split, "performer": args.performer,
                "spot": args.spot, "go_t": go_t, "pre_s": PRE_S, "post_s": POST_S,
                "packets": len(clip),
                "rate_hz": {link: round(sum(row[1] == link for row in clip) / (PRE_S + POST_S), 1)
                            for link in links},
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
            rates = ", ".join(f"{link} {hz} Hz" for link, hz in meta["rate_hz"].items())
            print(f"  saved {path.name} ({rates}): {seen}", flush=True)
            if problems:
                bad += 1
                print(f"  !! {'; '.join(problems)}. Training will skip this one; do an extra.", flush=True)
            recorded += 1
            view.record("saved", reader.clock(), clip=i, go_t=go_t, file=path.name,
                        seen=seen, problems=problems)

        print(f"\nDone: {args.n} x {args.label} in {args.root / args.split}, "
              f"{args.n - bad} clean, {bad} flagged")
        finished = True
    except KeyboardInterrupt:
        print()
    finally:
        view.record("done", reader.clock(), clip=recorded, clean=recorded - bad, flagged=bad,
                    stopped=not finished)
        view.close()


if __name__ == "__main__":
    main()
