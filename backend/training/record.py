from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import shutil
import subprocess
import tempfile
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Iterator
from uuid import uuid4

from backend import config
from backend.detection import features
from backend.training import contract, recordings
from backend.capture.csi import DEFAULT_LINK, parse_line
from backend.detection.detector import ThresholdDetector
from backend.detection.motion import MEASURE, MotionTracker

PRE_S, POST_S, HOLD_S = 1.5, 8.5, 2.0
SOUNDS = Path("/System/Library/Sounds")
MAX_NOISY_FRAC = 0.2
COUNT_WORDS = {"3": "Three", "2": "Two", "1": "One"}
GO_WORDS = {"fall": "Go! Fall.", "sit-down": "Go! Sit down.", "walk": "Go! Walk.", "empty": "Recording. Nobody move."}
HOLD_WORDS = {"fall": "Stay down. Hold still.", "sit-down": "Stay seated. Hold still."}

Lines = Iterator[tuple[float, str, str]]


class SessionStopped(Exception):
    pass


def beep(name: str) -> None:
    sound = SOUNDS / f"{name}.aiff"
    if sound.exists():
        subprocess.Popen(["afplay", str(sound)], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    else:
        print("\a", end="", flush=True)


class Speaker:
    def __init__(self, on: bool) -> None:
        self.on = on and shutil.which("say") is not None
        self.proc: subprocess.Popen | None = None
        self.text = ""
        self.ready: dict[str, str] = {}
        self.lock = threading.Lock()
        self.dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True) if self.on else None

    def prepare(self, texts: list[str]) -> None:
        if self.on and shutil.which("afplay") is not None:
            threading.Thread(target=self._render, args=(texts,), daemon=True).start()

    def _render(self, texts: list[str]) -> None:
        for i, text in enumerate(texts):
            path = Path(self.dir.name) / f"{i}.aiff"
            if subprocess.run(["say", "-o", str(path), text], capture_output=True).returncode == 0:
                self.ready[text] = str(path)

    def say(self, text: str, cut: bool = True) -> None:
        if not self.on:
            return
        with self.lock:
            if self.proc is not None and self.proc.poll() is None:
                self.proc.terminate()
                if not cut:
                    text = f"{self.text} {text}"
            path = self.ready.get(text)
            self.text = text
            self.proc = subprocess.Popen(["afplay", path] if path else ["say", text],
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def close(self) -> None:
        if self.dir is not None:
            self.dir.cleanup()


class Reader(threading.Thread):
    def __init__(self, lines, stop: threading.Event | None = None) -> None:
        super().__init__(daemon=True)
        self.lines = lines
        self.buf: deque[tuple[float, str, str]] = deque()
        self.lock = threading.Lock()
        self.ended = False
        self.count = 0
        self.stop = stop if stop is not None else threading.Event()

    def run(self) -> None:
        try:
            for t, link, line in self.lines:
                if self.stop.is_set():
                    break
                with self.lock:
                    self.buf.append((t, link, line))
                    self.count += 1
                    while t - self.buf[0][0] > 60:
                        self.buf.popleft()
        finally:
            self.lines.close()
            self.ended = True

    def close(self) -> None:
        self.stop.set()
        self.join(4)

    def clock(self) -> float:
        with self.lock:
            return self.buf[-1][0] if self.buf else 0.0

    def between(self, a: float, b: float) -> list[tuple[float, str, str]]:
        with self.lock:
            return [row for row in self.buf if a <= row[0] <= b]

    def since(self, k: int) -> tuple[list[tuple[float, str, str]], int]:
        with self.lock:
            skip = max(0, k - (self.count - len(self.buf)))
            return list(itertools.islice(self.buf, skip, None)), self.count

    def links(self) -> list[str]:
        with self.lock:
            return sorted({link for _, link, _ in self.buf})

    def wait_first(self) -> None:
        self.wait_until(1.0)

    def wait_until(self, target: float, cut: threading.Event | None = None) -> bool:
        last_change, last = time.monotonic(), self.clock()
        if self.stop.is_set():
            raise SessionStopped()
        while self.clock() < target:
            if self.stop.is_set():
                raise SessionStopped()
            if cut is not None and cut.is_set():
                return False
            if self.ended:
                raise RuntimeError("The packet stream ended before the session finished.")
            time.sleep(0.02)
            if self.clock() != last:
                last_change, last = time.monotonic(), self.clock()
            elif time.monotonic() - last_change > 3:
                print("  (no packets for 3 s: is the SEND board powered?)", flush=True)
                last_change = time.monotonic()
        return True


class View:
    def __init__(self, port: int | None, cal: dict | None, label: str, n: int, link: str, voice: Speaker,
                 hub=None) -> None:
        self.hub = self.feed = None
        self.base = {"label": label, "n": n}
        self.link = link
        self.tracker = MotionTracker(link)
        self.pending = cal
        self.failed = False
        self.t = 0.0
        self.lock = threading.Lock()
        self.asked = threading.Event()
        self.busy = self.ended = self.closing = False
        self.voice = voice
        if hub is not None:
            from backend import server
            self.hub, self.feed = hub, server.SignalFeed(hub)
            hub.actions["calibrate"] = self.ask
            return
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
        hub.actions["calibrate"] = self.ask

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

    def ask(self) -> bool:
        with self.lock:
            if self.busy or self.ended or self.failed or self.closing:
                return False
            if not self.asked.is_set():
                self.asked.set()
                self.record("calibrate_queued", self.t)
                self.voice.say("Recalibrating before the next clip.")
            return True

    def last_clip(self) -> bool:
        with self.lock:
            self.closing = not self.asked.is_set()
            return self.closing

    def calibrating(self, t: float) -> None:
        with self.lock:
            self.busy = True
            self.asked.clear()
            self.record("calibrating", t, clip=0)

    def calibrated(self, cal: dict, t: float) -> None:
        with self.lock:
            self.pending = cal
            self.busy = False
            self.record("calibrated", t, clip=0, links=sorted(cal))

    def end(self, t: float, **fields) -> None:
        with self.lock:
            self.ended = True
            self.record("done", t, **fields)

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
        self.voice.close()
        if self.hub is not None:
            self.feed.flush()
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
    trackers, fed_until, seen = {link: MotionTracker(link) for link in links}, reader.clock(), reader.count
    while any(tracker.calibration() is None for tracker in trackers.values()):
        reader.wait_until(fed_until + 0.5)
        rows, seen = reader.since(seen)
        for t, link, line in rows:
            frame = parse_line(line, t, link)
            if frame is not None and link in trackers:
                trackers[link].update(frame)
            fed_until = t
    return {link: tracker.calibration() for link, tracker in trackers.items()}


def calibrate_and_save(view: View, reader: Reader, links: list[str], cal_path: Path) -> dict[str, dict]:
    while True:
        print(f"Calibrating in {config.CALIBRATE_SETTLE_S:g} s: everyone out of the area and nobody moving, "
              f"for {config.CALIBRATE_S:g} s...", flush=True)
        view.calibrating(reader.clock())
        view.voice.say("Calibrating. Everyone out, nobody move.", cut=False)
        reader.wait_until(reader.clock() + config.CALIBRATE_SETTLE_S)
        cal = calibrate(reader, links)
        noisy = {link: c["noisy_frac"] for link, c in cal.items() if c["noisy_frac"] > MAX_NOISY_FRAC}
        if not noisy:
            break
        worst = ", ".join(f"{f:.0%} on link {link}" for link, f in noisy.items())
        print(f"  Something moved during {worst}. Trying again...", flush=True)
    cal_path.write_text(json.dumps(
        {"created_at": datetime.now().isoformat(timespec="minutes"), "calibration": cal}))
    view.calibrated(cal, reader.clock())
    view.voice.say("Calibrated.", cut=False)
    print(f"Calibrated, saved to {cal_path}", flush=True)
    return cal


def session_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--port", help="receiver A serial port")
    ap.add_argument("--port-b", help="receiver B serial port")
    ap.add_argument("--no-view", action="store_true", help="don't serve or open the live page")
    ap.add_argument("--view-port", type=int, default=8000, help="port for the live page")
    ap.add_argument("--no-voice", action="store_true", help="beep instead of speaking the cues")
    ap.add_argument("--replay", help=argparse.SUPPRESS)
    ap.add_argument("--root", type=Path, default=recordings.RECORDINGS, help=argparse.SUPPRESS)


def open_session(args: argparse.Namespace, label: str, n: int,
                 cal: dict | None, hub=None,
                 stop: threading.Event | None = None) -> tuple[View, Reader, dict[str, str] | None]:
    stop = stop if stop is not None else threading.Event()
    ports = None
    if args.replay:
        from backend.capture.replay import replay_lines
        lines = replay_lines(args.replay, 1.0)
    else:
        from backend.capture.live import live_lines, receivers
        ports = receivers(args.port, args.port_b)
        lines = live_lines(ports, stop=stop)
    view_link = DEFAULT_LINK if args.replay or DEFAULT_LINK in ports else min(ports)
    view = View(None if args.no_view else args.view_port, cal, label, n, view_link, Speaker(not args.no_voice), hub)
    reader = Reader(view.watch(lines), stop)
    reader.start()
    view.publish({"type": "source", "mode": "REPLAY" if args.replay else "LIVE"})
    view.publish({"type": "config", "t_on": config.T_ON, "t_off": config.T_OFF,
                  "pre_s": PRE_S, "post_s": POST_S, "calibrate_s": config.CALIBRATE_S,
                  "calibrate_settle_s": config.CALIBRATE_SETTLE_S})
    return view, reader, ports


def heard_links(reader: Reader, ports: dict[str, str] | None, verb: str) -> list[str]:
    reader.wait_first()
    reader.wait_until(reader.clock() + 1.0)
    links = reader.links()
    if ports is not None and links != list(ports):
        silent = ", ".join(link for link in ports if link not in links)
        print(f"Receiver {silent} is plugged in but sends nothing; {verb.lower()} {', '.join(links)} only.",
              flush=True)
    print(f"{verb} link{'s' if len(links) > 1 else ''} {', '.join(links)}.", flush=True)
    return links


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("label", choices=recordings.LABELS, nargs="?", help="start this recording batch immediately")
    ap.add_argument("--n", type=int, default=10, help="how many clips to record")
    ap.add_argument("--split", choices=("train", "test"), default="train")
    ap.add_argument("--gap", type=float, default=8.0, help="seconds to get in position")
    ap.add_argument("--performer", default="vincent")
    ap.add_argument("--spot", default="test-room", help="where this was recorded")
    ap.add_argument("--session-id", default=uuid4().hex,
                    help="collection session shared across action batches; keep in one evaluation split")
    ap.add_argument("--setup-id", help="board placement identifier; change it when the boards move")
    ap.add_argument("--recalibrate", action="store_true",
                    help="learn the calm room again (do this after moving the boards)")
    session_args(ap)
    args = ap.parse_args()

    if not args.no_view:
        from backend.training.dashboard import run
        run(args)
    elif args.label is None:
        ap.error("label is required with --no-view")
    else:
        run_recording(args)


def run_recording(args: argparse.Namespace, hub=None, stop: threading.Event | None = None) -> None:
    session_id = getattr(args, "session_id", None) or uuid4().hex
    capture = contract.capture_context("replay" if args.replay else "live")
    cal_path = args.root.parent / recordings.CALIBRATION_FILE
    cal_path.parent.mkdir(parents=True, exist_ok=True)
    saved = json.loads(cal_path.read_text()) if cal_path.exists() else None
    saved_cals = {link: cal for link, cal in recordings.per_link(saved["calibration"]).items()
                  if cal.get("measure") == MEASURE} if saved else {}

    view, reader, ports = open_session(args, args.label, args.n, saved_cals or None, hub, stop)
    view.voice.prepare([*COUNT_WORDS.values(), GO_WORDS[args.label]])

    detector = ThresholdDetector()
    recorded, bad, finished = 0, 0, False
    try:
        links = heard_links(reader, ports, "Recording")
        if not args.recalibrate and all(link in saved_cals for link in links):
            cal = {link: saved_cals[link] for link in links}
            print(f"Using the calm-room calibration from {saved['created_at']} "
                  "(pass --recalibrate if you moved the boards).", flush=True)
        else:
            if not args.recalibrate:
                new = [link for link in links if link not in saved_cals]
                print(f"No saved calibration for link{'s' if len(new) > 1 else ''} {', '.join(new)} yet.",
                      flush=True)
            cal = calibrate_and_save(view, reader, links, cal_path)

        for i in range(1, args.n + 1):
            while True:
                if view.asked.is_set():
                    print("\nRecalibrating, as asked on the live page.", flush=True)
                    cal = calibrate_and_save(view, reader, links, cal_path)
                print(f"\nClip {i}/{args.n} [{args.label}]: get in position...", flush=True)
                view.voice.say(f"Clip {i} of {args.n}. Get in position.", cut=False)
                start = reader.clock()
                until = start + max(0.0, args.gap - 3)
                view.record("position", start, clip=i, until=until)
                if reader.wait_until(until, view.asked) and (i < args.n or view.last_clip()):
                    break
            for word in ("3", "2", "1"):
                t = reader.clock()
                view.record("count", t, clip=i, word=word)
                if view.voice.on:
                    view.voice.say(COUNT_WORDS[word])
                else:
                    beep("Tink")
                print(f"  {word}...", flush=True)
                reader.wait_until(t + 1)
            if view.voice.on:
                view.voice.say(GO_WORDS[args.label])
            else:
                beep("Glass")
            print("  GO!", flush=True)
            go_t = reader.clock()
            view.record("go", go_t, clip=i)
            if args.label in HOLD_WORDS:
                reader.wait_until(go_t + HOLD_S)
                view.voice.say(HOLD_WORDS[args.label])
            reader.wait_until(go_t + POST_S)

            clip = reader.between(go_t - PRE_S, go_t + POST_S)
            meta = {
                "label": args.label, "split": args.split, "performer": args.performer,
                "spot": args.spot, "go_t": go_t, "pre_s": PRE_S, "post_s": POST_S,
                "packets": len(clip),
                "rate_hz": {link: round(sum(row[1] == link for row in clip) / (PRE_S + POST_S), 1)
                            for link in links},
                "recorded_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "session_id": session_id, "setup_id": getattr(args, "setup_id", None),
                "capture": capture,
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
            meta["quality"] = {"validator_version": contract.VALIDATOR_VERSION,
                               "problems": problems}
            path.with_name(path.name.removesuffix(".log.gz") + ".json").write_text(
                json.dumps(meta, indent=1))
            beep("Basso" if problems else "Pop")
            view.voice.say(f"{problems[0].split(' (')[0].capitalize()}. Do an extra." if problems else "Saved.")
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
        view.voice.say(f"Done. {args.n - bad} clean, {bad} flagged.", cut=False)
        finished = True
    except (KeyboardInterrupt, SessionStopped):
        print()
    finally:
        reader.close()
        view.end(reader.clock(), clip=recorded, clean=recorded - bad, flagged=bad, stopped=not finished)
        view.close()


if __name__ == "__main__":
    main()
