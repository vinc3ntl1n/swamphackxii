from __future__ import annotations

import json
import os
import queue
import threading
import time
import traceback
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Iterator

from backend import config, server
from backend.capture.csi import Frame
from backend.detection.detector import Verdict
from backend.detection.fallflow import ALERTED, COUNTDOWN, FALL_CANDIDATE, IDLE, Sink
from backend.detection.motion import FrameInfo, MotionEvent, MotionTracker
from backend.detection.pipeline import Pipeline
from backend.training import recordings

SIM_FALL = Path(__file__).resolve().parent / "assets" / "sim_fall.log.gz"
HISTORY = config.DATA_DIR / "history.jsonl"
ENV_FILE = config.DATA_DIR.parent / ".env"
TAIL_S = 1.0
HOLD_S = config.STILLNESS_S + 1.5
TICK_S = 0.05
STALL_S = 0.3
STATE_EVERY_S = 0.5
SIM_FEED_AFTER_S = 2.0
DELIVERY_WAIT_S = 10.0
HELP_COOLDOWN_S = 3.0
HISTORY_SHOWN = 300
RESOLVE_REASONS = ("false_alarm", "handled")

Result = tuple[MotionEvent, Verdict]


def load_env(path: Path = ENV_FILE) -> None:
    """Read KEY=value lines from .env without overriding the real environment."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and key and not key.startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def clock(t: float) -> str:
    return datetime.fromtimestamp(t).strftime("%H:%M:%S")


class History:
    """Falls, Check-ins, answers and Alerts, kept in data/history.jsonl across runs."""

    def __init__(self, hub: server.Hub, mode: str, path: Path = HISTORY) -> None:
        self.hub, self.mode, self.path = hub, mode, path
        self.sim = False
        self.sim_falls: set[int] = set()
        self.last_fall = self.last_notice = 0
        if not path.exists():
            return
        lines = path.read_text().splitlines()
        for i, line in enumerate(lines):
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if isinstance(entry.get("fall"), int):
                self.last_fall = max(self.last_fall, entry["fall"])
            if entry.get("what") == "notify" and isinstance(entry.get("id"), int):
                self.last_notice = max(self.last_notice, entry["id"])
            if i >= len(lines) - HISTORY_SHOWN:
                self.hub.publish({"type": "history", **entry})

    def add(self, t: float, what: str, **fields) -> None:
        fall = fields.get("fall")
        sim = fall in self.sim_falls if isinstance(fall, int) else self.sim
        entry = {"t": t, "what": what, "mode": "sim" if sim else self.mode, **fields}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError as e:
            print(f"  (history not saved: {e})", flush=True)
        self.hub.publish({"type": "history", **entry})


class Notices:
    """Messages for the Resident or Caregiver, sent by the messaging sidecar (see docs/adr/0001)."""

    def __init__(self, hub: server.Hub, history: History) -> None:
        self.hub, self.history = hub, history
        self.next_id = history.last_notice + 1
        self.pending: dict[int, tuple[float, int]] = {}

    def send(self, t: float, to: str, kind: str, fall: int, text: str) -> None:
        nid, self.next_id = self.next_id, self.next_id + 1
        self.pending[nid] = (time.monotonic(), fall)
        msg = {"id": nid, "to": to, "kind": kind, "fall": fall, "text": text, "status": "sending"}
        self.hub.publish({"type": "notify", "t": t, **msg})
        self.history.add(t, "notify", **msg)

    def delivered(self, t: float, nid: int, status: str) -> bool:
        pending = self.pending.pop(nid, None)
        if pending is None:
            return False
        self._status(t, nid, pending[1], status)
        return True

    def overdue(self, t: float) -> None:
        now = time.monotonic()
        for nid, (sent, fall) in list(self.pending.items()):
            if now - sent > DELIVERY_WAIT_S:
                del self.pending[nid]
                self._status(t, nid, fall, "not_delivered")

    def _status(self, t: float, nid: int, fall: int, status: str) -> None:
        self.hub.publish({"type": "delivery", "t": t, "id": nid, "fall": fall, "status": status})
        self.history.add(t, "delivery", id=nid, fall=fall, status=status)


class WebSink:
    def __init__(self, hub: server.Hub, history: History, notices: Notices) -> None:
        self.hub, self.history, self.notices = hub, history, notices
        self.fall = history.last_fall
        self.fall_t: float | None = None
        self.alert_t: float | None = None

    def log(self, t: float, text: str) -> None:
        self.hub.publish({"type": "log", "t": t, "text": text})

    def alarm(self, on: bool) -> None:
        self.hub.publish({"type": "alarm", "on": on})

    def ask_resident(self, t: float) -> None:
        self.fall += 1
        self.fall_t, self.alert_t = t, None
        if self.history.sim:
            self.history.sim_falls.add(self.fall)
        self.notices.send(t, "resident", "check_in", self.fall,
                          "Fall detected. Are you OK? Reply 1 = I'm OK, 2 = I need help.")

    def alert_caregiver(self, t: float) -> None:
        self.alert_t = t
        self.history.add(t, "alert", fall=self.fall)
        self.notices.send(t, "caregiver", "alert", self.fall,
                          f"ALERT: the Resident may have fallen at {clock(self.fall_t or t)} "
                          "and did not answer the Check-in.")

    def event(self, t: float, event: MotionEvent, verdict: Verdict, links: str, sim: bool) -> None:
        details = {k: v for k, v in verdict.details.items() if k != "links"}
        if event.cut_short:
            why = "cut_short"
        elif "peak" in details:
            why = "classified"
        else:
            why = "moved_again"
        msg = {
            "label": verdict.label, "confidence": round(float(verdict.confidence), 2), "links": links,
            "details": details, "why": why, "start": event.start, "end": event.end, "sim": sim,
        }
        self.hub.publish({"type": "event", "t": t, **msg})
        if verdict.label == "fall" and why == "classified":
            self.history.add(t, "confirmed_fall", fall=self.fall, links=links, details=details)


class Tee:
    """Fans out to every sink; a failing sink is reported, never allowed to stop detection."""

    def __init__(self, *sinks: Sink) -> None:
        self.sinks = sinks

    def log(self, t: float, text: str) -> None:
        self._each("log", t, text)

    def alarm(self, on: bool) -> None:
        self._each("alarm", on)

    def ask_resident(self, t: float) -> None:
        self._each("ask_resident", t)

    def alert_caregiver(self, t: float) -> None:
        self._each("alert_caregiver", t)

    def _each(self, name: str, *args) -> None:
        for sink in self.sinks:
            try:
                getattr(sink, name)(*args)
            except Exception:
                print(f"  ({type(sink).__name__}.{name} failed; detection continues)", flush=True)
                traceback.print_exc()


class Clock:
    """Packet time, carried on by the wall clock while no packets arrive."""

    def __init__(self) -> None:
        self.t: float | None = None
        self.wall = 0.0

    def saw(self, t: float) -> None:
        if self.t is None or t >= self.t:
            self.t, self.wall = t, time.monotonic()

    def now(self) -> float:
        return time.time() if self.t is None else self.t + (time.monotonic() - self.wall)

    def stalled(self) -> bool:
        return self.t is None or time.monotonic() - self.wall > STALL_S


def recording_calibration(path: str | Path) -> dict[str, dict] | None:
    path = Path(path)
    meta_path = path.with_name(path.name.removesuffix(".gz").removesuffix(".log") + ".json")
    if not meta_path.exists():
        return None
    cal = json.loads(meta_path.read_text()).get("calibration")
    return recordings.per_link(cal) if cal else None


def preload(pipe: Pipeline, cals: dict[str, dict]) -> None:
    for link, cal in sorted(cals.items()):
        tracker = pipe.trackers[link] = MotionTracker(link)
        tracker.load_calibration(cal)


def repeat_tail(tail: list[Frame], seconds: float) -> Iterator[Frame]:
    """Copies of the last still second, so a Stillness Window can finish after a short clip."""
    if len(tail) < 2 or seconds <= 0:
        return
    span = tail[-1].t - tail[0].t
    period = span + span / (len(tail) - 1)
    k = 1
    while True:
        for f in tail:
            t = f.t + k * period
            if t - tail[-1].t > seconds:
                return
            yield Frame(t=t, rssi=f.rssi, amp=f.amp, link=f.link)
        k += 1


def held(frames: Iterator[Frame], seconds: float, speed: float) -> Iterator[Frame]:
    tail: deque[Frame] = deque()
    for f in frames:
        tail.append(f)
        while f.t - tail[0].t > TAIL_S:
            tail.popleft()
        yield f
    if not tail or seconds <= 0:
        return
    print(f"Replay ended: holding its last second for {seconds:.0f} s so the Stillness Window can finish",
          flush=True)
    wall, t0 = time.monotonic(), tail[-1].t
    for f in repeat_tail(list(tail), seconds):
        if speed > 0:
            delay = (f.t - t0) / speed - (time.monotonic() - wall)
            if delay > 0:
                time.sleep(delay)
        yield f


class SimFall:
    """Plays a fall Recording through its own trackers into the shared FallFlow."""

    def __init__(self, path: Path, main: Pipeline, sink: Sink, now: float) -> None:
        rec = recordings.Recording(path, json.loads(path.with_name(
            path.name.removesuffix(".log.gz") + ".json").read_text()))
        frames = rec.frames()
        if not frames:
            raise ValueError(f"{path.name} has no packets")
        shift = now + 0.05 - frames[0].t
        frames = [Frame(t=f.t + shift, rssi=f.rssi, amp=f.amp, link=f.link) for f in frames]
        tail = [f for f in frames if f.t > frames[-1].t - TAIL_S]
        self.frames = deque([*frames, *repeat_tail(tail, HOLD_S)])
        self.pipe = Pipeline(sink, main.detector)
        self.pipe.flow = main.flow
        preload(self.pipe, recordings.per_link(rec.meta["calibration"]))
        self.name = path.name

    def due(self, now: float) -> Iterator[Frame]:
        while self.frames and self.frames[0].t <= now:
            yield self.frames.popleft()

    @property
    def done(self) -> bool:
        return not self.frames


class Dashboard:
    """Shares the pipeline with the frame loop under one lock, and runs browser and sidecar commands."""

    def __init__(self, hub: server.Hub, pipe: Pipeline, sink: Sink, web: WebSink,
                 sim_path: Path = SIM_FALL) -> None:
        self.hub, self.pipe, self.sink, self.web = hub, pipe, sink, web
        self.sim_path = sim_path
        self.lock = threading.RLock()
        self.commands: queue.Queue[dict] = queue.Queue()
        self.clock = Clock()
        self.feeds: dict[str, server.SignalFeed] = {}
        self.live_seen: dict[str, float] = {}
        self.sim: SimFall | None = None
        self.sent: tuple | None = None
        self.sent_at = 0.0
        self.failed = False
        self.help_at = -1e9

    def on_command(self, msg: dict) -> None:
        self.commands.put(msg)

    def start(self) -> None:
        threading.Thread(target=self._run, name="dashboard-ticker", daemon=True).start()

    def process(self, frame: Frame) -> tuple[FrameInfo, Result | None]:
        with self.lock:
            self.clock.saw(frame.t)
            self.live_seen[frame.link] = time.monotonic()
            info, result = self.pipe.process(frame)
            self._view(lambda: self._show(self.pipe, frame, info, result, sim=False))
            return info, result

    def ended(self) -> None:
        with self.lock:
            self._view(self._flush)

    def _run(self) -> None:
        while True:
            time.sleep(TICK_S)
            with self.lock:
                try:
                    self._tick()
                except Exception:
                    traceback.print_exc()

    def _tick(self) -> None:
        while True:
            try:
                msg = self.commands.get_nowait()
            except queue.Empty:
                break
            self._command(msg)
        now = self.clock.now()
        if self.sim is not None:
            for frame in self.sim.due(now):
                info, result = self.sim.pipe.process(frame)
                self._view(lambda: self._show(self.sim.pipe, frame, info, result, sim=True))
            if self.sim.done:
                self._sim(None)
        if self.clock.stalled():
            self.pipe.flow.tick(now)
        self.web.notices.overdue(now)
        self._view(lambda: self._state(now))

    def _sim(self, sim: SimFall | None) -> None:
        self.sim = sim
        self.web.history.sim = sim is not None

    def _command(self, msg: dict) -> None:
        cmd, now, flow, web = msg.get("cmd"), self.clock.now(), self.pipe.flow, self.web
        if cmd in ("ok", "help"):
            self._answer(now, cmd, msg.get("fall"))
        elif cmd == "cancel":
            if flow.state == COUNTDOWN:
                web.history.add(now, "countdown_cancelled", fall=web.fall, by="caregiver")
                self._sim(None)
            flow.cancel(now, "the Caregiver")
        elif cmd == "resolve":
            reason = msg.get("reason")
            if flow.state != ALERTED or reason not in RESOLVE_REASONS:
                return
            web.history.add(now, "resolved", fall=web.fall, reason=reason)
            self._sim(None)
            flow.reset(now)
        elif cmd == "call":
            web.history.add(now, "called", fall=web.fall)
            self.sink.log(now, "Caregiver called the Resident")
        elif cmd == "delivered":
            nid, status = msg.get("id"), msg.get("status")
            if isinstance(nid, int) and status in ("sent", "failed") and web.notices.delivered(now, nid, status):
                if status == "failed":
                    self.sink.log(now, f"Message {nid} failed to send")
        elif cmd == "simulate":
            if self.sim is not None or flow.state != IDLE:
                busy = "a Simulated Fall is playing" if self.sim is not None else f"already {flow.state}"
                self.sink.log(now, f"Simulate fall ignored: {busy}")
                return
            try:
                sim = SimFall(self.sim_path, self.pipe, self.sink, now)
            except (OSError, ValueError, KeyError) as e:
                self.sink.log(now, f"Simulate fall failed: {e}")
                return
            self._sim(sim)
            self.sink.log(now, f"Simulated Fall: playing {sim.name}")

    def _answer(self, now: float, cmd: str, fall) -> None:
        flow, web = self.pipe.flow, self.web
        open_fall = flow.state in (COUNTDOWN, ALERTED)
        if not open_fall or (fall is not None and fall != web.fall):
            web.history.add(now, "stale_answer", fall=fall, answer=cmd)
            self.sink.log(now, f"Stale Answer ignored: {cmd} for fall {fall if fall is not None else '?'}")
            return
        if cmd == "help":
            if time.monotonic() - self.help_at < HELP_COOLDOWN_S:
                self.sink.log(now, f"Help Request ignored: another one came in under {HELP_COOLDOWN_S:g} s ago")
                return
            self.help_at = time.monotonic()
            web.history.add(now, "help_request", fall=web.fall)
            self.sink.log(now, "Help Request from the Resident")
            escalate = getattr(flow, "escalate", None)
            if flow.state == COUNTDOWN and escalate is not None:
                escalate(now)
            else:
                web.notices.send(now, "caregiver", "help", web.fall,
                                 f"The Resident asked for help (fall at {clock(web.fall_t or now)}).")
            return
        if flow.state == COUNTDOWN:
            web.history.add(now, "countdown_cancelled", fall=web.fall, by="resident")
            self._sim(None)
            flow.cancel(now, "the Resident (I'm OK)")
            return
        late = round(now - (web.alert_t or now))
        web.history.add(now, "late_answer", fall=web.fall, late_s=late)
        self.sink.log(now, f"Late Answer: the Resident is OK ({late} s after the Alert)")
        web.notices.send(now, "caregiver", "all_clear", web.fall,
                         f"All clear: the Resident answered they're OK, {late} s after the Alert.")
        self._sim(None)
        flow.reset(now)

    def _show(self, pipe: Pipeline, frame: Frame, info: FrameInfo, result: Result | None, sim: bool) -> None:
        tracker = pipe.trackers[frame.link]
        if not (info.amp is None and tracker.size is not None):
            quiet = time.monotonic() - self.live_seen.get(frame.link, -1e9) > SIM_FEED_AFTER_S
            if not sim or quiet:
                feed = self.feeds.get(frame.link)
                if feed is None:
                    feed = self.feeds[frame.link] = server.SignalFeed(self.hub, frame.link)
                feed.add(info, frame.rssi, frame.amp if info.amp is None else info.amp)
        if result is not None:
            event, verdict = result
            links = verdict.details.get("links") or (event.link if event.cut_short else pipe.links or event.link)
            self.web.event(info.t, event, verdict, links, sim)
        self._state(info.t)

    def _state(self, now: float) -> None:
        flow = self.pipe.flow
        key = (flow.state, flow.started, flow.deadline, self.sim is not None)
        timed = flow.state in (FALL_CANDIDATE, COUNTDOWN)
        if key == self.sent and not (timed and time.monotonic() - self.sent_at >= STATE_EVERY_S):
            return
        self.sent, self.sent_at = key, time.monotonic()
        self.hub.publish({
            "type": "state", "state": flow.state,
            "t": flow.started if flow.started is not None else now,
            "deadline": flow.deadline, "now": now, "sim": self.sim is not None, "fall": self.web.fall,
        })

    def _flush(self) -> None:
        for feed in self.feeds.values():
            feed.flush()

    def _view(self, fn) -> None:
        if self.failed:
            return
        try:
            fn()
        except Exception as e:
            self.failed = True
            print(f"  (dashboard stopped: {e!r}; detection continues)", flush=True)
