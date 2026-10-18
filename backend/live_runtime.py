"""Keep receiver health and confirmed countdowns running in terminal live mode."""
from __future__ import annotations

import threading
import traceback

from backend import config, server
from backend.capture.csi import Frame
from backend.dashboard import Clock, SensorStatus, TICK_S
from backend.detection.pipeline import Pipeline


class TerminalRuntime:
    """Use the dashboard's live timing rules without starting its web interface.

    Packet absence invalidates pending sensor evidence. The wall ticker only
    advances an already confirmed countdown; it never manufactures CSI frames.
    """

    def __init__(self, pipe: Pipeline, links=None) -> None:
        self.pipe = pipe
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
        with self.lock:
            self.clock.saw(frame.t)
            self.sensors.refresh(frame.t)
            info, result = self.sensors.process(frame)
            self.sensors.add(frame, info)
            return info, result

    def tick(self) -> None:
        """Advance live health and fall flow without creating motion evidence."""
        with self.lock:
            now = self.clock.now()
            self.sensors.refresh(now)
            if self.clock.stalled():
                self.pipe.flow.tick(now)

    def _run(self) -> None:
        while not self.stop_event.wait(TICK_S):
            try:
                self.tick()
            except Exception:
                traceback.print_exc()
