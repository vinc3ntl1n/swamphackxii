"""Keep receiver health and confirmed countdowns running in terminal live mode."""
from __future__ import annotations

import threading
import time
import traceback

from backend import config, server
from backend.capture.csi import Frame
from backend.capture.live import LIVE_MAX_AGE_S
from backend.dashboard import Clock, SensorStatus, TICK_S
from backend.detection.motion import FrameInfo, STALE
from backend.detection.pipeline import Pipeline
from backend.diagnostics import EventLog, RuntimeMonitor


class TerminalRuntime:
    """Use the dashboard's live timing rules without starting its web interface.

    Packet absence invalidates pending sensor evidence. Wall ticks expire lost
    receivers and advance confirmed countdowns without manufacturing CSI frames.
    """

    def __init__(self, pipe: Pipeline, links=None, *, diagnostics: EventLog | None = None,
                 live_age: bool = False) -> None:
        self.pipe = pipe
        self.live_age = live_age
        self.monitor = RuntimeMonitor(diagnostics, live=live_age) if diagnostics else None
        self.lock = threading.RLock()
        self.clock = Clock()
        self.sensors = SensorStatus(server.Hub(), pipe, "LIVE",
                                    config.RECEIVERS if links is None else links)
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        with self.lock:
            if self.thread is not None and self.thread.is_alive():
                return
            self.stop_event.clear()
            self.sensors.refresh(self.clock.now(), force=True)
            self.thread = threading.Thread(target=self._run, name="terminal-live-ticker", daemon=True)
            self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(1.0)

    def process(self, frame: Frame):
        started = time.perf_counter()
        with self.lock:
            age_s = frame.capture_age_s() if self.live_age else 0.0
            if self.live_age and frame.relative_age_s() > LIVE_MAX_AGE_S:
                self.pipe.capture_gap(time.time(), [frame.link], reason="stale_capture")
                self.sensors.refresh(time.time())
                tracker = self.pipe.trackers.get(frame.link)
                info = FrameInfo(frame.t, tracker.level if tracker else STALE,
                                 None, None, "stale_capture")
                if self.monitor:
                    self.monitor.observe(self.pipe, frame, (time.perf_counter() - started) * 1000)
                return info, None
            self.clock.saw(frame.t, age_s)
            self.sensors.refresh(frame.t)
            info, result = self.sensors.process(frame)
            self.sensors.add(frame, info, age_s=age_s)
            if self.monitor:
                self.monitor.observe(self.pipe, frame, (time.perf_counter() - started) * 1000)
            return info, result

    def capture_health(self, health: dict) -> None:
        """Apply known capture loss before the next frame can reach the RF."""
        with self.lock:
            for link in health.get("reset_links", ()):
                self.pipe.suspend_link(link, health["capture_wall_t"])
            if health.get("dropped_since_report", 0):
                self.pipe.capture_gap(health["last_drop"]["capture_t"], health.get("dropped_links"))
            if health.get("idle"):
                now = health["capture_wall_t"]
                self.sensors.refresh(now)
                self.pipe.tick(now)
                if self.monitor:
                    self.monitor.tick(self.pipe, now)

    def tick(self) -> None:
        """Advance live health and fall flow without creating motion evidence."""
        with self.lock:
            now = self.clock.now()
            self.sensors.refresh(now)
            if self.clock.stalled():
                self.pipe.tick(now)
            if self.monitor:
                self.monitor.tick(self.pipe, now)

    def _run(self) -> None:
        while not self.stop_event.wait(TICK_S):
            try:
                self.tick()
            except Exception:
                traceback.print_exc()
