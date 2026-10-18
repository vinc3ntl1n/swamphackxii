"""Receiver clock evidence without changing the timestamps used by the RF.

Device and host clocks have an unknown offset. The smallest observed arrival
offset supplies a relative lag reference, not absolute packet age. In
particular, a stream already buffered at startup can look fresh by this measure;
compressed device-time progress provides a separate catch-up warning.
"""

from __future__ import annotations

import math
from collections import deque

from backend.capture.csi import Frame

_MASK = (1 << 32) - 1
_HALF = 1 << 31
_WRAP_HIGH = 3 * (1 << 30)
_WRAP_LOW = 1 << 30


def _wrap(previous: int, current: int) -> bool:
    return previous >= _WRAP_HIGH and current < _WRAP_LOW


class SourceTiming:
    """Track one receiver; never compare independently clocked A/B timestamps.

    Counter gaps are diagnostic only: they can originate before USB capture.
    Clock drift can contribute to relative lag over a long run, so these fields
    are transport evidence rather than a synchronized clock reconstruction.
    """

    def __init__(self, max_lag_s: float = 0.5) -> None:
        if not math.isfinite(max_lag_s) or max_lag_s <= 0:
            raise ValueError("max_lag_s must be finite and positive")
        self.max_lag_s = max_lag_s
        self.skipped_ids = 0
        self._pending_reset: str | None = None
        self._clear_clock()

    def _clear_clock(self) -> None:
        self._last_source: int | None = None
        self._last_id: int | None = None
        self._last_host: float | None = None
        self._first_host: float | None = None
        self._source_elapsed = 0.0
        self._best_offset = 0.0
        self._window: deque[tuple[float, float]] = deque(maxlen=2048)
        self._compressed = False

    def reset(self, reason: str = "connection_reopened") -> None:
        """Start a new clock generation; expose the reset on its first valid row."""
        self._clear_clock()
        self._pending_reset = reason

    def observe(self, frame: Frame, received_mono: float) -> dict:
        if not math.isfinite(received_mono):
            raise ValueError("received_mono must be finite")
        source = frame.source_timestamp_us
        packet_id = frame.source_id
        result = {
            "source_id": packet_id, "source_timestamp_us": source,
            "source_available": source is not None,
            "source_age_known": False,
            "source_lag_s": None, "source_lag_ms": None,
            "source_step_ms": None, "host_step_ms": None,
            "source_clock_elapsed_s": None, "host_elapsed_s": None,
            "skipped_ids": self.skipped_ids,
            "clock_event": None, "reset_reason": None,
            "source_clock_reset": False, "source_compressed": False,
            "timing_discontinuity": False, "stale": False,
        }
        if source is None:
            return result

        source &= _MASK
        packet_id = packet_id & _MASK if packet_id is not None else None
        event = None
        reset_reason = self._pending_reset
        self._pending_reset = None
        step_s = host_step_s = None
        if self._last_source is not None:
            host_step_s = received_mono - self._last_host
            source_step = (source - self._last_source) & _MASK
            timestamp_wrap = source < self._last_source and _wrap(self._last_source, source)
            if host_step_s < 0:
                reset_reason = "host_clock_reset"
            elif source_step >= _HALF or (source < self._last_source and not timestamp_wrap):
                reset_reason = "source_timestamp_reset"
            elif (packet_id is not None and self._last_id is not None
                  and packet_id < self._last_id and not _wrap(self._last_id, packet_id)):
                reset_reason = "source_id_reset"
            else:
                step_s = source_step / 1_000_000
                if timestamp_wrap:
                    event = "wrap"
                if packet_id is not None and self._last_id is not None:
                    id_step = (packet_id - self._last_id) & _MASK
                    if 1 < id_step < _HALF:
                        self.skipped_ids += id_step - 1
        if reset_reason is not None:
            self._clear_clock()
            step_s = host_step_s = None
            event = "reset"
        if self._first_host is None:
            self._first_host = received_mono
        elif step_s is not None:
            self._source_elapsed += step_s

        host_elapsed = received_mono - self._first_host
        offset = host_elapsed - self._source_elapsed
        self._best_offset = min(self._best_offset, offset)
        lag_s = max(0.0, offset - self._best_offset)

        # A source-time span delivered too quickly is an independent timing
        # discontinuity, even when startup buffering defeats the offset test.
        # Keep two seconds of host history; require >0.5 s compression rather
        # than treating normal USB batching or packet-ID gaps as loss.
        horizon_s = max(2.0, 2 * self.max_lag_s)
        while self._window and received_mono - self._window[0][1] > horizon_s:
            self._window.popleft()
        compressed = any(
            self._source_elapsed - old_source >= self.max_lag_s
            and (self._source_elapsed - old_source) - (received_mono - old_host) > self.max_lag_s
            for old_source, old_host in self._window
        )
        compression_started = compressed and not self._compressed
        if compression_started and event is None:
            event = "compressed_catchup"
        self._window.append((self._source_elapsed, received_mono))
        self._compressed = compressed
        self._last_source, self._last_id, self._last_host = source, packet_id, received_mono

        result.update(
            source_lag_s=lag_s, source_lag_ms=round(lag_s * 1000, 3),
            source_step_ms=round(step_s * 1000, 3) if step_s is not None else None,
            host_step_ms=round(host_step_s * 1000, 3) if host_step_s is not None else None,
            source_clock_elapsed_s=self._source_elapsed, host_elapsed_s=host_elapsed,
            skipped_ids=self.skipped_ids, clock_event=event, reset_reason=reset_reason,
            source_clock_reset=event == "reset", source_compressed=compressed,
            timing_discontinuity=event == "reset" or compression_started,
            stale=lag_s > self.max_lag_s,
        )
        return result
