from __future__ import annotations

from typing import Protocol

from backend import config

IDLE, FALL_CANDIDATE, COUNTDOWN, ALERTED = "IDLE", "FALL_CANDIDATE", "COUNTDOWN", "ALERTED"


class Sink(Protocol):
    def log(self, t: float, text: str) -> None: ...
    def alarm(self, on: bool) -> None: ...
    def ask_resident(self, t: float) -> None: ...
    def alert_caregiver(self, t: float) -> None: ...


class FallFlow:
    def __init__(self, sink: Sink) -> None:
        self.sink = sink
        self.state = IDLE
        self.deadline: float | None = None
        self.started: float | None = None

    def remaining(self, t: float) -> float | None:
        return max(0.0, self.deadline - t) if self.deadline is not None else None

    def candidate(self, t: float, deadline: float) -> bool:
        if self.state != IDLE:
            self.sink.log(t, f"Movement ignored: already {self.state} (reset first)")
            return False
        self._go(FALL_CANDIDATE, t, deadline)
        return True

    def not_a_fall(self, t: float) -> None:
        if self.state == FALL_CANDIDATE:
            self._go(IDLE, t)

    def confirm(self, t: float) -> None:
        if self.state != FALL_CANDIDATE:
            return
        self.sink.log(t, "Confirmed Fall: Countdown started")
        self._go(COUNTDOWN, t, deadline=t + config.COUNTDOWN_S)
        self.sink.alarm(True)
        self.sink.ask_resident(t)

    def tick(self, t: float) -> None:
        if self.state == COUNTDOWN and t >= self.deadline:
            self.sink.alarm(False)
            self.sink.alert_caregiver(t)
            self.sink.log(t, "Alert sent to Caregiver")
            self._go(ALERTED, t)

    def escalate(self, t: float) -> bool:
        if self.state != COUNTDOWN:
            return False
        self.sink.alarm(False)
        self.sink.log(t, "Help Request: Caregiver alerted early")
        self._go(ALERTED, t)
        return True

    def cancel(self, t: float, source: str) -> None:
        if self.state != COUNTDOWN:
            return
        self.sink.alarm(False)
        self.sink.log(t, f"Countdown cancelled by {source}")
        self._go(IDLE, t)

    def reset(self, t: float) -> None:
        if self.state == IDLE:
            return
        self.sink.alarm(False)
        self.sink.log(t, "Reset")
        self._go(IDLE, t)

    def _go(self, state: str, t: float, deadline: float | None = None) -> None:
        self.state, self.started, self.deadline = state, t, deadline
