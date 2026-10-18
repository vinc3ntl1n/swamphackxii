from __future__ import annotations

import argparse
import hashlib
import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path

from backend import config
from backend.detection import detector
from backend.detection import features
from backend.detection.motion import CALIBRATING
from backend.detection.pipeline import Pipeline
from backend.diagnostics import CaptureLogHandler, DiagnosticSink, EventLog, RuntimeMonitor


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
    from backend.dashboard import load_env
    load_env()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", help="receiver A serial port (live mode)")
    ap.add_argument("--port-b", help="receiver B serial port (live mode)")
    ap.add_argument("--replay", help="a session log or Recording to play back instead")
    ap.add_argument("--speed", type=float, default=1.0, help="replay speed, 0 = no waiting")
    ap.add_argument("--detector", choices=("threshold", "rf"), default=detector.default_name())
    ap.add_argument("--model", type=Path, default=detector.default_model_path(),
                    help="saved forest to load with --detector rf")
    ap.add_argument("--diagnostics", type=Path,
                    help="JSONL diagnostics (live default: beside the raw session log)")
    ap.add_argument("--view", action=argparse.BooleanOptionalAction, default=True,
                    help="serve the live dashboard and open it in the browser (--no-view turns it off)")
    ap.add_argument("--view-port", type=int, default=8000, help="port for the dashboard")
    ap.add_argument("--sim-fall", help="fall Recording that the dashboard's Simulate fall button plays")
    ap.add_argument("--history", help=argparse.SUPPRESS)
    args = ap.parse_args()

    # Load exactly once, before opening receivers or starting the web server.
    try:
        selected_detector = detector.make(args.detector, args.model)
    except (OSError, ValueError, RuntimeError) as exc:
        ap.exit(1, f"Detector startup failed: {exc}\n")
    cals, frames, ports, log_path = None, None, {}, None
    if args.replay and not os.path.exists(args.replay):
        ap.exit(1, f"No such file: {args.replay}\n"
                   "Recordings live in R2 now: run python -m backend.sync pull first, "
                   "or replay backend/assets/sim_fall.log.gz.\n")
    if args.replay:
        from backend.capture.replay import replay_frames
        from backend.dashboard import HOLD_S, held, recording_calibration
        cals = recording_calibration(args.replay)
        frames = held(replay_frames(args.replay, args.speed), HOLD_S if cals else 0.0, args.speed)
        print(f"REPLAY {args.replay}")
    else:
        from backend.capture.live import SESSIONS, receivers
        ports = receivers(args.port, args.port_b)
        SESSIONS.mkdir(parents=True, exist_ok=True)
        log_path = SESSIONS / f"{datetime.now():%Y%m%d-%H%M%S-%f}.log"
        print(f"LIVE {', '.join(f'{link} {port}' for link, port in ports.items())}, saving to {log_path}")

    diagnostics_path = args.diagnostics or (log_path.with_suffix(".events.jsonl") if log_path else None)
    events = EventLog(diagnostics_path) if diagnostics_path else None
    try:
        _run(args, frames, cals, ports, log_path, events, selected_detector)
    finally:
        if events:
            events.close()


def _run(args, frames, cals, ports, log_path, events: EventLog | None, selected_detector) -> None:
    console = ConsoleSink()
    sink = DiagnosticSink(console, events) if events else console
    view, runtime = None, None
    stop = threading.Event()
    capture_logger, handler, old_level = None, None, None
    try:
        view = start_view(args, sink, selected_detector, diagnostics=events) if args.view else None
        pipe = view.pipe if view else Pipeline(sink, selected_detector)
        monitor = RuntimeMonitor(events, live=False) if events and not view and args.replay else None
        print(f"Detector: {pipe.detector.name}")
        if events:
            events.write("run_started", mode="replay" if args.replay else "live",
                         raw_log=str(log_path) if log_path else args.replay, receivers=ports,
                         detector=pipe.detector.name, features=features.NAMES,
                         model=str(args.model) if args.detector == "rf" else None,
                         model_sha256=(hashlib.sha256(args.model.read_bytes()).hexdigest()
                                       if args.detector == "rf" else None),
                         trained_at=getattr(pipe.detector, "trained_at", None),
                         fall_threshold=getattr(pipe.detector, "fall_threshold", None),
                         timing={"motion_on": config.T_ON, "motion_off": config.T_OFF,
                                 "window_s": config.WINDOW_S, "quiet_s": config.QUIET_S,
                                 "stillness_s": config.STILLNESS_S,
                                 "after_s": config.AFTER_S})
            print(f"Diagnostics: {events.path}", flush=True)
        if cals:
            from backend.dashboard import preload
            preload(pipe, cals)
            print(f"Using the Recording's calibration for link{'s' if len(cals) > 1 else ''} {', '.join(sorted(cals))}")
        else:
            print("Calibrating: step away for 3 s, then keep the area between the boards still for 10 s...")
        if not args.replay:
            from backend.capture import live
            if not view:
                from backend.live_runtime import TerminalRuntime
                runtime = TerminalRuntime(pipe, diagnostics=events, live_age=True)
                runtime.start()

            def capture_health(health: dict) -> None:
                if view:
                    view.capture_health(health)
                else:
                    runtime.capture_health(health)

            if events:
                capture_logger = live.LOGGER
                old_level = capture_logger.level
                handler = CaptureLogHandler(events)
                capture_logger.addHandler(handler)
                capture_logger.setLevel(logging.INFO)
            frames = live.live_frames(ports, log_path, stop=stop, health_callback=capture_health)
        process = view.process if view else runtime.process if runtime else pipe.process
        last_level: dict[str, str] = {}
        for frame in frames:
            if console.t0 is None:
                console.t0 = frame.t
            started = time.perf_counter()
            info, _ = process(frame)
            if monitor:
                monitor.observe(pipe, frame, (time.perf_counter() - started) * 1000)
            if info.level != last_level.get(frame.link, CALIBRATING) and info.level != CALIBRATING:
                energy = f"{info.energy:.1f}x" if info.energy is not None else "--"
                name = f" {frame.link}" if len(pipe.trackers) > 1 else ""
                print(f"{console._stamp(info.t)}   level{name}: {info.level} ({energy})", flush=True)
            last_level[frame.link] = info.level
        if view:
            view.ended()
            view.hub.publish({"type": "source", "mode": "REPLAY" if args.replay else "LIVE", "ended": True})
            print("Stream ended. The dashboard stays up (Simulate fall still works); Ctrl-C to quit.", flush=True)
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print()
    finally:
        stop.set()
        if runtime is not None:
            runtime.stop()
        if view:
            view.stop()
        close_frames = getattr(frames, "close", None)
        if close_frames is not None:
            close_frames()
        if view:
            view.hub.close(1.0)
        if capture_logger is not None:
            capture_logger.removeHandler(handler)
            capture_logger.setLevel(old_level)
        if events:
            events.write("run_stopped")


def start_view(args, console, loaded_detector=None, *, diagnostics: EventLog | None = None):
    from pathlib import Path
    from backend import server, store
    from backend.dashboard import HISTORY, SIM_FALL, Dashboard, History, Notices, Tee, WebSink, load_env

    load_env()
    hub = server.Hub()
    tiger = store.from_env()
    history = History(hub, "replay" if args.replay else "live", Path(args.history) if args.history else HISTORY,
                      tiger)
    web = WebSink(hub, history, Notices(hub, history))
    sink = Tee(console, web)
    pipe = Pipeline(sink, loaded_detector if loaded_detector is not None else detector.make(args.detector, args.model))
    source = "REPLAY" if args.replay else "LIVE"
    view = Dashboard(hub, pipe, sink, web, Path(args.sim_fall) if args.sim_fall else SIM_FALL,
                     source=source, diagnostics=diagnostics, live=not args.replay)
    hub.publish({"type": "detector", **pipe.detector.metadata()})
    hub.publish({"type": "source", "mode": "REPLAY" if args.replay else "LIVE"})
    hub.publish({"type": "config", "t_on": config.T_ON, "t_off": config.T_OFF,
                 "calibrate_s": config.CALIBRATE_S, "t_fall": config.T_FALL,
                 "t_static": config.T_STATIC, "sit_max_s": config.SIT_MAX_S,
                 "stillness_s": config.STILLNESS_S, "countdown_s": config.COUNTDOWN_S,
                 "detector": pipe.detector.name, "links": sorted(config.RECEIVERS),
                 "resident_phone": os.environ.get("RESIDENT_PHONE", ""),
                 "resident_name": os.environ.get("RESIDENT_NAME", ""), "trends": tiger is not None})
    try:
        url = server.serve(hub, args.view_port, view.on_command, tiger.trends if tiger else None)
    except Exception as e:
        print(f"  (dashboard off: {e!r}; detecting anyway)", flush=True)
        return None
    if url is None:
        print(f"  (dashboard off: port {args.view_port} is busy; detecting anyway)", flush=True)
        return None
    print(f"Dashboard at {url}dashboard", flush=True)
    server.open_browser(f"{url}dashboard")
    view.start()
    return view


if __name__ == "__main__":
    main()
