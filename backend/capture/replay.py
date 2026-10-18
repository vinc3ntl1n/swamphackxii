from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterator

from backend.capture.csi import Frame, parse_line, read_log
from backend.capture.meter import Meter


def replay_lines(path: str | Path, speed: float = 1.0) -> Iterator[tuple[float, str, str]]:
    start_wall = start_t = None
    for t, link, line in read_log(path):
        if speed > 0:
            if start_t is None:
                start_wall, start_t = time.monotonic(), t
            delay = (t - start_t) / speed - (time.monotonic() - start_wall)
            if delay > 0:
                time.sleep(delay)
        yield t, link, line


def replay_frames(path: str | Path, speed: float = 1.0) -> Iterator[Frame]:
    for t, link, line in replay_lines(path, speed):
        frame = parse_line(line, t, link)
        if frame is not None:
            yield frame


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="a data/sessions/*.log or a Recording .log.gz")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed, 0 = no waiting")
    args = ap.parse_args()

    print(f"REPLAY {args.path}")
    meters: dict[str, Meter] = {}
    try:
        for frame in replay_frames(args.path, args.speed):
            if frame.link not in meters:
                meters[frame.link] = Meter(frame.link)
            meters[frame.link].add(frame)
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
