from __future__ import annotations

import argparse

from backend.training import record, recordings


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Learn the calm room again for every receiver that sends packets, save it, and exit.")
    record.session_args(ap)
    args = ap.parse_args()

    cal_path = args.root.parent / recordings.CALIBRATION_FILE
    view, reader, ports = record.open_session(args, "", 0, None)
    finished = False
    try:
        links = record.heard_links(reader, ports, "Calibrating")
        record.calibrate_and_save(view, reader, links, cal_path)
        finished = True
    except KeyboardInterrupt:
        print()
    finally:
        view.end(reader.clock(), clip=0, stopped=not finished)
        view.close()


if __name__ == "__main__":
    main()
