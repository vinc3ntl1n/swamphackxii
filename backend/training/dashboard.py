from __future__ import annotations

import argparse
import threading
import uuid

from backend import server
from backend.training import record
from backend.training.calibrate import run_calibration


class Dashboard:
    def __init__(self, args: argparse.Namespace, hub: server.Hub) -> None:
        self.args, self.hub = args, hub
        self.lock = threading.Lock()
        self.worker: threading.Thread | None = None
        self.stop = threading.Event()
        self.closed = False
        self.settings = {key: getattr(args, key) for key in ("n", "split", "gap", "performer", "spot")}
        self.settings["label"] = args.label or "fall"
        self.state = {"id": "", "status": "idle", "error": None, **self.settings}
        hub.actions.update(start=self.start, stop=self.stop_session, calibrate=self.calibrate)
        self.publish()

    def publish(self) -> None:
        self.hub.publish({"type": "session", **self.state})

    def start(self, settings: dict) -> bool:
        return self._start(settings, calibration=False)

    def calibrate(self) -> bool:
        return self._start(self.settings, calibration=True)

    def _start(self, settings: dict, calibration: bool) -> bool:
        with self.lock:
            if self.closed or self.worker is not None:
                return False
            args = argparse.Namespace(**{**vars(self.args), **settings, "no_view": True})
            self.settings = dict(settings)
            self.stop = threading.Event()
            self.hub.reset_records()
            self.state = {"id": uuid.uuid4().hex, "status": "running", "error": None,
                          "calibration": calibration, **self.settings}
            self.publish()
            self.worker = threading.Thread(target=self._work, args=(args, calibration),
                                           name="training-session", daemon=True)
            self.worker.start()
            return True

    def _work(self, args: argparse.Namespace, calibration: bool) -> None:
        error = None
        try:
            if calibration:
                run_calibration(args, self.hub, self.stop)
            else:
                record.run_recording(args, self.hub, self.stop)
        except (Exception, SystemExit) as exc:
            error = str(exc) or type(exc).__name__
            print(f"Session failed: {error}", flush=True)
        finally:
            with self.lock:
                self.hub.actions["calibrate"] = self.calibrate
                self.worker = None
                self.state.update(status="idle", error=error)
                self.publish()

    def stop_session(self) -> bool:
        with self.lock:
            if self.worker is None or self.state["status"] == "stopping":
                return False
            self.state["status"] = "stopping"
            self.stop.set()
            self.publish()
            return True

    def close(self) -> None:
        with self.lock:
            self.closed = True
            self.stop.set()
            worker = self.worker
        if worker is not None:
            worker.join(6)


def run(args: argparse.Namespace) -> None:
    hub = server.Hub()
    dashboard = Dashboard(args, hub)
    url = server.serve(hub, args.view_port)
    if url is None:
        raise SystemExit(f"Dashboard port {args.view_port} is unavailable; choose another with --view-port.")
    print(f"Training dashboard at {url}\nPress Ctrl-C to close the dashboard.", flush=True)
    server.open_browser(url)
    if args.label is not None:
        dashboard.start(dashboard.settings)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        print()
    finally:
        dashboard.close()
        hub.close()
