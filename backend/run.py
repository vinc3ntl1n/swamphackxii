from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from backend import config
from backend.detection import detector
from backend.detection.motion import CALIBRATING
from backend.detection.pipeline import Pipeline


class ConsoleSink:
    def __init__(self) -> None:
        self.t0: float | None = None

    def _stamp(self, t: float) -> str:
        self.t0 = self.t0 if self.t0 is not None else t
        return f"[{t - self.t0:7.1f}s]"

    def log(self, t: float, text: str) -> None:
        print(f"{self._stamp(t)} {text}", flush=True)

    def alarm(self, on: bool) -> None:
        print(f"          ALARM {'ON' if on else 'OFF'}", flush=True)

    def ask_resident(self, t: float) -> None:
        print(f"{self._stamp(t)} (would push 'Are you OK?' to the Resident)", flush=True)

    def alert_caregiver(self, t: float) -> None:
        print(f"{self._stamp(t)} (would push the Alert to the Caregiver)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", help="receiver A serial port (live mode)")
    ap.add_argument("--port-b", help="receiver B serial port (live mode)")
    ap.add_argument("--replay", help="a session log or Recording to play back instead")
    ap.add_argument("--speed", type=float, default=1.0, help="replay speed, 0 = no waiting")
    ap.add_argument("--detector", choices=("threshold", "rf"), default=config.DETECTOR)
    ap.add_argument("--model", type=Path, default=detector.MODEL_PATH,
                    help="saved forest to load with --detector rf")
    args = ap.parse_args()

    if args.replay:
        from backend.capture.replay import replay_frames
        frames = replay_frames(args.replay, args.speed)
        print(f"REPLAY {args.replay}")
    else:
        from backend.capture.live import SESSIONS, live_frames, receivers
        ports = receivers(args.port, args.port_b)
        SESSIONS.mkdir(parents=True, exist_ok=True)
        log_path = SESSIONS / f"{datetime.now():%Y%m%d-%H%M%S}.log"
        frames = live_frames(ports, log_path)
        print(f"LIVE {', '.join(f'{link} {port}' for link, port in ports.items())}, saving to {log_path}")

    sink = ConsoleSink()
    pipe = Pipeline(sink, detector.make(args.detector, args.model))
    print(f"Detector: {pipe.detector.name}")
    print("Calibrating: keep the area between the boards still for 10 s...")
    last_level: dict[str, str] = {}
    try:
        for frame in frames:
            if sink.t0 is None:
                sink.t0 = frame.t
            info, _ = pipe.process(frame)
            if info.level != last_level.get(frame.link, CALIBRATING) and info.level != CALIBRATING:
                energy = f"{info.energy:.1f}x" if info.energy is not None else "--"
                name = f" {frame.link}" if len(pipe.trackers) > 1 else ""
                print(f"{sink._stamp(info.t)}   level{name}: {info.level} ({energy})", flush=True)
            last_level[frame.link] = info.level
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
