from __future__ import annotations

import asyncio
import atexit
import contextlib
import json
import math
import socket
import threading
import time
import webbrowser
from collections import deque
from pathlib import Path
from typing import Callable

import numpy as np
import uvicorn
from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.detection.motion import FrameInfo

WEB = Path(__file__).resolve().parent.parent / "web"
HOST = "127.0.0.1"
STICKY = ("source", "config", "state", "alarm")
KEPT = ("record", "event", "log", "notify", "delivery", "history")
KEPT_MAX = 2000
BACKLOG_S = 60.0
QUEUE_SIZE = 256
STEP_S = 0.05
HZ_S = 2.0


class Hub:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.seq = 0
        self.sticky: dict[str, tuple[int, str]] = {}
        self.kept: deque[tuple[int, str]] = deque(maxlen=KEPT_MAX)
        self.backlog: deque[tuple[int, float, str]] = deque()
        self.clients: dict[asyncio.Queue[str], int] = {}
        self.loop: asyncio.AbstractEventLoop | None = None
        self.actions: dict[str, Callable[[], bool]] = {}

    def publish(self, msg: dict) -> None:
        text = json.dumps(msg, separators=(",", ":"), allow_nan=False)
        kind, now = msg.get("type"), time.monotonic()
        with self.lock:
            self.seq += 1
            if kind in STICKY:
                self.sticky[kind] = (self.seq, text)
            elif kind in KEPT:
                self.kept.append((self.seq, text))
            else:
                self.backlog.append((self.seq, now, text))
                while now - self.backlog[0][1] > BACKLOG_S:
                    self.backlog.popleft()
            if self.loop is not None:
                with contextlib.suppress(RuntimeError):
                    self.loop.call_soon_threadsafe(self._fanout, self.seq, text)

    def join(self) -> tuple[asyncio.Queue[str], list[str]]:
        queue: asyncio.Queue[str] = asyncio.Queue(QUEUE_SIZE)
        with self.lock:
            self.loop = asyncio.get_running_loop()
            old = [*self.sticky.values(), *self.kept, *((s, x) for s, _, x in self.backlog)]
            self.clients[queue] = self.seq
        return queue, [text for _, text in sorted(old)]

    def leave(self, queue: asyncio.Queue[str]) -> None:
        self.clients.pop(queue, None)

    def close(self, timeout: float = 1.0) -> None:
        loop = self.loop
        if loop is None or loop.is_closed():
            return
        with contextlib.suppress(Exception):
            asyncio.run_coroutine_threadsafe(self._drain(), loop).result(timeout)

    async def _drain(self) -> None:
        while any(not queue.empty() for queue in self.clients):
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)

    def _fanout(self, seq: int, text: str) -> None:
        for queue, since in self.clients.items():
            if seq <= since:
                continue
            if queue.full():
                queue.get_nowait()
            queue.put_nowait(text)


class SignalFeed:
    def __init__(self, hub: Hub, link: str | None = None) -> None:
        self.hub = hub
        self.link = link
        self.slot: int | None = None
        self.sum: np.ndarray | None = None
        self.count = 0
        self.t = 0.0
        self.energy: float | None = None
        self.level = ""
        self.rssi = 0
        self.times: deque[float] = deque()

    def add(self, info: FrameInfo, rssi: int, row: np.ndarray | None = None) -> None:
        row = info.amp if row is None else row
        if row is None:
            return
        slot = math.floor(info.t / STEP_S)
        if self.count and (slot != self.slot or row.size != self.sum.size):
            self.flush()
        if not self.count:
            self.slot, self.sum, self.energy = slot, np.zeros(row.size), None
        self.sum += row
        self.count += 1
        self.t, self.level, self.rssi = info.t, info.level, int(rssi)
        if info.energy is not None and math.isfinite(info.energy):
            self.energy = max(self.energy or 0.0, float(info.energy))
        self.times.append(info.t)
        while info.t - self.times[0] > HZ_S:
            self.times.popleft()

    def flush(self) -> None:
        if not self.count:
            return
        row = np.round(self.sum / self.count, 1)
        values = row.tolist()
        if not np.isfinite(row).all():
            values = [v if math.isfinite(v) else None for v in values]
        span = self.times[-1] - self.times[0]
        hz = round((len(self.times) - 1) / span, 1) if span >= 0.5 else None
        self.count = 0
        msg = {
            "type": "frame", "t": self.t,
            "energy": None if self.energy is None else round(self.energy, 2),
            "level": self.level, "row": values, "rssi": self.rssi, "hz": hz,
        }
        if self.link is not None:
            msg["link"] = self.link
        self.hub.publish(msg)


async def _send(client: WebSocket, queue: asyncio.Queue[str], snapshot: list[str]) -> None:
    with contextlib.suppress(WebSocketDisconnect):
        for text in snapshot:
            await client.send_text(text)
        while True:
            await client.send_text(await queue.get())


Commands = Callable[[dict], None]


def _command(on_command: Commands | None, text: str | None) -> None:
    if on_command is None or not text:
        return
    with contextlib.suppress(ValueError):
        msg = json.loads(text)
        if isinstance(msg, dict):
            on_command(msg)


def make_app(hub: Hub, on_command: Commands | None = None) -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def no_cache(request: Request, call_next) -> Response:
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-cache"
        return response

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(WEB / "record.html")

    @app.get("/dashboard")
    async def dashboard() -> FileResponse:
        return FileResponse(WEB / "dashboard.html")

    @app.post("/calibrate")
    async def calibrate(request: Request) -> Response:
        port = request.scope["server"][1]
        if request.headers.get("host") not in (f"127.0.0.1:{port}", f"localhost:{port}"):
            return Response(status_code=403)
        if request.headers.get("content-type", "").split(";")[0].strip().lower() != "application/json":
            return Response(status_code=415)
        action = hub.actions.get("calibrate")
        return Response(status_code=202 if action is not None and action() else 409)

    @app.websocket("/ws")
    async def ws(client: WebSocket) -> None:
        await client.accept()
        queue, snapshot = hub.join()
        sender = asyncio.create_task(_send(client, queue, snapshot))
        try:
            while (msg := await client.receive())["type"] != "websocket.disconnect":
                _command(on_command, msg.get("text"))
        finally:
            sender.cancel()
            hub.leave(queue)

    app.mount("/", StaticFiles(directory=WEB), name="web")
    return app


def _taken(port: int) -> bool:
    with contextlib.suppress(OSError), socket.create_connection((HOST, port), timeout=0.2):
        return True
    return False


def serve(hub: Hub, port: int, on_command: Commands | None = None) -> str | None:
    if port and _taken(port):
        return None
    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((HOST, port))
        sock.listen()
    except OSError:
        sock.close()
        return None
    url = f"http://{HOST}:{sock.getsockname()[1]}/"
    server = uvicorn.Server(uvicorn.Config(make_app(hub, on_command), log_level="warning", lifespan="off"))
    thread = threading.Thread(target=server.run, args=([sock],), name="view-server", daemon=True)
    thread.start()
    while not server.started and thread.is_alive():
        time.sleep(0.01)
    if not server.started:
        return None

    def stop() -> None:
        server.should_exit = server.force_exit = True
        thread.join(2)

    atexit.register(stop)
    return url


def open_browser(url: str) -> None:
    threading.Thread(target=webbrowser.open, args=(url,), daemon=True).start()
