from __future__ import annotations

import argparse
import threading

from backend.training import record, recordings


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Learn the calm room again for every receiver that sends packets, save it, and exit.")
    record.session_args(ap)
    args = ap.parse_args()
    run_calibration(args)


def run_calibration(args: argparse.Namespace, hub=None, stop: threading.Event | None = None) -> None:
    cal_path = args.root.parent / recordings.CALIBRATION_FILE
    cal_path.parent.mkdir(parents=True, exist_ok=True)
    view, reader, ports = record.open_session(args, "", 0, None, hub, stop)
    finished = False
    try:
        links = record.heard_links(reader, ports, "Calibrating")
        record.calibrate_and_save(view, reader, links, cal_path)
        finished = True
    except (KeyboardInterrupt, record.SessionStopped):
        print()
    finally:
        reader.close()
        view.end(reader.clock(), clip=0, stopped=not finished)
        view.close()


if __name__ == "__main__":
    main()
