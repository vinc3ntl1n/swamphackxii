"""Optional local runtime evidence; recording failures never stop detection."""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
from pathlib import Path

DIAGNOSTIC_PREFIX = "Detection diagnostic: "
SAMPLE_S = 1.0
SLOW_FRAME_MS = 50.0


class EventLog:
    """Append complete JSON lines under one lock, failing visibly at startup only."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8")
        self._lock = threading.Lock()
        self._closed = False
        self._disabled = False
        self._warned = False

    def _disable(self, error: Exception) -> None:
        self._disabled = True
        if not self._warned:
            self._warned = True
            print(f"Runtime diagnostics disabled ({self.path}): {error}", file=sys.stderr, flush=True)

    def write(self, kind: str, event_t: float | None = None, **payload) -> None:
        with self._lock:
            if self._closed or self._disabled:
                return
            try:
                row = {**payload, "type": kind, "wall_time": time.time()}
                if event_t is not None:
                    row["event_t"] = event_t
                self._file.write(json.dumps(row, allow_nan=False) + "\n")
                self._file.flush()
            except Exception as error:
                self._disable(error)

    def capture_health(self, payload: dict) -> None:
        fields = dict(payload)
        kind = fields.pop("type", "capture_health")
        event_t = fields.pop("event_t", None)
        self.write(kind, event_t, **fields)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._file.close()
            except Exception as error:
                self._disable(error)


class DiagnosticSink:
    """Keep the normal Sink behavior and persist the evidence it emits."""

    def __init__(self, sink, log: EventLog) -> None:
        self.sink, self.logfile = sink, log

    def log(self, t: float, text: str) -> None:
        fields = None
        if text.startswith(DIAGNOSTIC_PREFIX):
            try:
                candidate = json.loads(text[len(DIAGNOSTIC_PREFIX):])
                if isinstance(candidate, dict):
                    fields = candidate
            except (TypeError, ValueError):
                pass
        if fields is None:
            self.logfile.write("log", t, text=text)
        else:
            fields = {k: v for k, v in fields.items() if k not in ("type", "event_t", "text")}
            self.logfile.write("detection", t, text=text, **fields)
        self.sink.log(t, text)

    def alarm(self, on: bool) -> None:
        self.logfile.write("alarm", on=on)
        self.sink.alarm(on)

    def ask_resident(self, t: float) -> None:
        self.logfile.write("ask_resident", t)
        self.sink.ask_resident(t)

    def alert_caregiver(self, t: float) -> None:
        self.logfile.write("alert_caregiver", t)
        self.sink.alert_caregiver(t)


class CaptureLogHandler(logging.Handler):
    """Forward capture JSON records without installing a global logging policy."""

    def __init__(self, log: EventLog) -> None:
        super().__init__()
        self.logfile = log

    def emit(self, record: logging.LogRecord) -> None:
        try:
            text = record.getMessage()
            try:
                fields = json.loads(text)
            except (TypeError, ValueError):
                fields = None
            if isinstance(fields, dict):
                self.logfile.capture_health(fields)
            else:
                self.logfile.write("capture_log", level=record.levelname, text=text)
        except Exception:
            # Diagnostic handlers must never disrupt the serial reader.
            return


class RuntimeMonitor:
    """Sample processing cost and pipeline state without per-packet log volume."""

    def __init__(self, log: EventLog, live: bool) -> None:
        self.logfile, self.live = log, live
        self._lock = threading.Lock()
        self._last_sample: float | None = None
        self._last_event_t: float | None = None
        self._last_frame_t: float | None = None
        self._processing_ms: float | None = None
        self._max_processing_ms: float | None = None
        self._frames = 0
        self._slow_frames = 0

    def observe(self, pipe, frame, processing_ms: float) -> None:
        with self._lock:
            self._last_frame_t = self._last_event_t = frame.t
            self._processing_ms = processing_ms
            self._max_processing_ms = max(self._max_processing_ms or 0.0, processing_ms)
            self._frames += 1
            self._slow_frames += processing_ms > SLOW_FRAME_MS
            self._sample(pipe, frame.t)

    def tick(self, pipe, event_t: float | None = None) -> None:
        with self._lock:
            self._last_event_t = event_t if event_t is not None else self._last_event_t
            self._sample(pipe, self._last_event_t)

    def _sample(self, pipe, event_t: float | None) -> None:
        now = time.monotonic()
        if self._last_sample is not None and now - self._last_sample < SAMPLE_S:
            return
        try:
            pending = pipe.pending
            fields = {
                "live": self.live,
                "receiver_health": pipe.health(event_t) if event_t is not None else {},
                "flow": {"state": pipe.flow.state, "started": pipe.flow.started,
                         "deadline": pipe.flow.deadline},
                "pending": ({"link": pending.link, "start": pending.start, "end": pending.end,
                             "deadline": pipe.deadline, "after_frames": len(pipe.after)}
                            if pending is not None else None),
                "frames": self._frames,
                "processing_ms": self._processing_ms,
                "max_processing_ms": self._max_processing_ms,
                "slow_frames": self._slow_frames,
            }
            if self.live:
                fields["capture_age_ms"] = (max(0.0, time.time() - self._last_frame_t) * 1000
                                            if self._last_frame_t is not None else None)
            self.logfile.write("runtime", event_t, **fields)
        except Exception as error:
            self.logfile.write("diagnostic_error", event_t, error=str(error))
        self._last_sample = now
        self._processing_ms = self._max_processing_ms = None
        self._frames = self._slow_frames = 0
