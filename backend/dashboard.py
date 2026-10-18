from __future__ import annotations

import json
import queue
import threading
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Iterator

from backend import config, server
from backend.capture.csi import Frame
from backend.detection.detector import Verdict
from backend.detection.fallflow import COUNTDOWN, FALL_CANDIDATE, IDLE, Sink
from backend.detection.motion import FrameInfo, MotionEvent, MotionTracker
from backend.detection.pipeline import Pipeline
from backend.training import recordings

SIM_FALL = recordings.RECORDINGS / "train" / "fall_20261017-021504.log.gz"
TAIL_S = 1.0
HOLD_S = config.STILLNESS_S + 1.5
TICK_S = 0.05
STALL_S = 0.3
STATE_EVERY_S = 0.5
SIM_FEED_AFTER_S = 2.0

Result = tuple[MotionEvent, Verdict]


class WebSink:
    def __init__(self, hub: server.Hub) -> None:
        self.hub = hub

    def log(self, t: float, text: str) -> None:
        self.hub.publish({"type": "log", "t": t, "text": text})

    def alarm(self, on: bool) -> None:
        self.hub.publish({"type": "alarm", "on": on})

    def ask_resident(self, t: float) -> None:
        self.hub.publish({"type": "alert", "t": t, "kind": "ask_resident", "status": "sent"})

    def alert_caregiver(self, t: float) -> None:
        self.hub.publish({"type": "alert", "t": t, "kind": "caregiver", "status": "sent"})

    def event(self, t: float, event: MotionEvent, verdict: Verdict, links: str, sim: bool) -> None:
        details = {k: v for k, v in verdict.details.items() if k != "links"}
        if event.cut_short:
            why = "cut_short"
        elif "peak" in details:
            why = "classified"
        else:
            why = "moved_again"
        self.hub.publish({
            "type": "event", "t": t, "label": verdict.label,
            "confidence": round(float(verdict.confidence), 2), "links": links,
            "details": details, "why": why, "start": event.start, "end": event.end, "sim": sim,
        })


class Tee:
    def __init__(self, *sinks: Sink) -> None:
        self.sinks = sinks

    def log(self, t: float, text: str) -> None:
        for sink in self.sinks:
            sink.log(t, text)

    def alarm(self, on: bool) -> None:
        for sink in self.sinks:
            sink.alarm(on)

    def ask_resident(self, t: float) -> None:
        for sink in self.sinks:
            sink.ask_resident(t)

    def alert_caregiver(self, t: float) -> None:
        for sink in self.sinks:
            sink.alert_caregiver(t)


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
    if not tail:
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
    """Shares the pipeline with the frame loop under one lock, and runs browser commands."""

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
                self.sim = None
        if self.clock.stalled():
            self.pipe.flow.tick(now)
        self._view(lambda: self._state(now))

    def _command(self, msg: dict) -> None:
        cmd, now, flow = msg.get("cmd"), self.clock.now(), self.pipe.flow
        if cmd == "cancel":
            flow.cancel(now, "the dashboard (I'm OK)")
        elif cmd == "reset":
            self.sim = None
            flow.reset(now)
        elif cmd == "simulate":
            if self.sim is not None or flow.state != IDLE:
                busy = "a simulated fall is playing" if self.sim is not None else f"already {flow.state}"
                self.sink.log(now, f"Simulate fall ignored: {busy}")
                return
            try:
                self.sim = SimFall(self.sim_path, self.pipe, self.sink, now)
            except (OSError, ValueError, KeyError) as e:
                self.sink.log(now, f"Simulate fall failed: {e}")
                return
            self.sink.log(now, f"Simulated fall: playing {self.sim.name}")

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
            "deadline": flow.deadline, "now": now, "sim": self.sim is not None,
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
