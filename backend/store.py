"""Copies History entries and Motion Events into Tiger Data, and reads Trends back (see docs/adr/0004).

Tiger only gets copies: data/history.jsonl still decides what the dashboard replays, and nothing in the
Fall Flow waits for Tiger. Without TIGER_DATABASE_URL none of this runs.

    python -m backend.store backfill [history.jsonl]   loads past History into Tiger (safe to re-run)
"""
from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from backend import config

QUEUE_MAX = 1000
RETRY_S = 30.0
CONNECT_TIMEOUT_S = 5
TREND_DAYS = 14
TREND_HOURS = 24
CLOSES = ("countdown_cancelled", "late_answer", "resolved")
ALERTED = ("late_answer", "handled", "false_alarm")

HISTORY_SQL = (
    "INSERT INTO history (time, room, what, fall, notice, mode, outcome, entry) "
    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING"
)
MOTION_SQL = (
    "INSERT INTO motion_events (time, room, mode, label, why, links, confidence, peak, room_change, duration_s) "
    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING"
)


def schema(tz: str) -> list[str]:
    """Idempotent setup: two hypertables, a continuous aggregate behind each Trend, and compression."""
    return [
        "CREATE EXTENSION IF NOT EXISTS timescaledb",
        # The natural key makes re-running the backfill harmless: rows already there are skipped.
        """CREATE TABLE IF NOT EXISTS history (
            time timestamptz NOT NULL, room text NOT NULL, what text NOT NULL,
            fall integer NOT NULL, notice integer NOT NULL, mode text NOT NULL,
            outcome text, entry jsonb NOT NULL,
            PRIMARY KEY (room, time, what, fall, notice))""",
        "SELECT create_hypertable('history', 'time', if_not_exists => TRUE)",
        """CREATE TABLE IF NOT EXISTS motion_events (
            time timestamptz NOT NULL, room text NOT NULL, mode text NOT NULL,
            label text NOT NULL, why text NOT NULL, links text, confidence real,
            peak real, room_change real, duration_s real,
            PRIMARY KEY (room, time))""",
        "SELECT create_hypertable('motion_events', 'time', chunk_time_interval => INTERVAL '1 day', "
        "if_not_exists => TRUE)",
        f"""CREATE MATERIALIZED VIEW IF NOT EXISTS falls_daily
            WITH (timescaledb.continuous, timescaledb.materialized_only = false) AS
            SELECT time_bucket(INTERVAL '1 day', time, '{tz}') AS day, room, mode, what, outcome,
                   count(*) AS n
            FROM history
            WHERE what IN ('confirmed_fall', 'countdown_cancelled', 'late_answer', 'resolved')
            GROUP BY day, room, mode, what, outcome
            WITH NO DATA""",
        "SELECT add_continuous_aggregate_policy('falls_daily', start_offset => INTERVAL '30 days', "
        "end_offset => INTERVAL '1 hour', schedule_interval => INTERVAL '10 minutes', if_not_exists => TRUE)",
        """CREATE MATERIALIZED VIEW IF NOT EXISTS motion_hourly
            WITH (timescaledb.continuous, timescaledb.materialized_only = false) AS
            SELECT time_bucket(INTERVAL '1 hour', time) AS hour, room, mode,
                   count(*) AS n, count(room_change) AS measured, avg(room_change) AS room_change
            FROM motion_events
            GROUP BY hour, room, mode
            WITH NO DATA""",
        "SELECT add_continuous_aggregate_policy('motion_hourly', start_offset => INTERVAL '3 days', "
        "end_offset => INTERVAL '1 hour', schedule_interval => INTERVAL '10 minutes', if_not_exists => TRUE)",
    ]


COMPRESSION = [
    "ALTER TABLE motion_events SET (timescaledb.compress, timescaledb.compress_segmentby = 'room, mode', "
    "timescaledb.compress_orderby = 'time DESC')",
    "SELECT add_compression_policy('motion_events', INTERVAL '7 days', if_not_exists => TRUE)",
]


def setup(conn, tz: str = config.TIMEZONE) -> None:
    for sql in schema(tz):
        conn.execute(sql)
    enabled = conn.execute("SELECT compression_enabled FROM timescaledb_information.hypertables "
                           "WHERE hypertable_name = 'motion_events'").fetchone()
    # Changing compression settings fails once chunks are compressed, so it is only set the first time.
    if not (enabled and enabled[0]):
        for sql in COMPRESSION:
            conn.execute(sql)


def when(seconds: float) -> datetime:
    return datetime.fromtimestamp(seconds, timezone.utc)


def outcome(entry: dict) -> str | None:
    """How a fall ended, on the entry that closed it (same names as Fall Summary metrics)."""
    what = entry.get("what")
    if what == "countdown_cancelled":
        return "resident_ok" if entry.get("by") == "resident" else "caregiver_cancelled"
    if what == "late_answer":
        return "late_answer"
    if what == "resolved":
        return entry.get("reason")
    return None


def history_row(entry: dict, room: str) -> tuple:
    from psycopg.types.json import Jsonb

    fall, notice = entry.get("fall"), entry.get("id")
    return (
        when(entry.get("wall") or entry["t"]), room, entry["what"],
        fall if isinstance(fall, int) else -1,
        notice if entry["what"] in ("notify", "delivery") and isinstance(notice, int) else -1,
        entry.get("mode", "live"), outcome(entry), Jsonb(entry),
    )


def motion_row(wall: float, room: str, mode: str, msg: dict) -> tuple:
    details = msg.get("details") or {}
    return (
        when(wall), room, mode, msg["label"], msg["why"], msg.get("links"), msg.get("confidence"),
        details.get("peak"), details.get("static_change"), details.get("duration_s"),
    )


def trends(conn, room: str, now: float, tz: str = config.TIMEZONE) -> dict:
    """Confirmed Falls per local day and Motion Events per hour, real runs only (no Simulated Falls)."""
    zone = ZoneInfo(tz)
    today = datetime.fromtimestamp(now, zone).date()
    days = [today - timedelta(days=k) for k in range(TREND_DAYS - 1, -1, -1)]
    first = datetime.combine(days[0], datetime.min.time(), zone)
    per_day = {d: {"falls": 0, "ok": 0, "cancelled": 0, "alerted": 0} for d in days}
    rows = conn.execute(
        "SELECT day, what, outcome, sum(n) FROM falls_daily "
        "WHERE room = %s AND mode <> 'sim' AND day >= %s GROUP BY day, what, outcome",
        (room, first)).fetchall()
    for day, what, how, n in rows:
        counts = per_day.get(day.astimezone(zone).date())
        if counts is None:
            continue
        if what == "confirmed_fall":
            counts["falls"] += int(n)
        elif how == "resident_ok":
            counts["ok"] += int(n)
        elif how == "caregiver_cancelled":
            counts["cancelled"] += int(n)
        elif how in ALERTED:
            counts["alerted"] += int(n)
    for counts in per_day.values():
        counts["open"] = max(counts["falls"] - counts["ok"] - counts["cancelled"] - counts["alerted"], 0)

    top = when(now).replace(minute=0, second=0, microsecond=0)
    hours = [top - timedelta(hours=k) for k in range(TREND_HOURS - 1, -1, -1)]
    per_hour = {h: {"events": 0, "room_change": None} for h in hours}
    rows = conn.execute(
        "SELECT hour, sum(n), sum(room_change * measured) / nullif(sum(measured), 0) FROM motion_hourly "
        "WHERE room = %s AND mode <> 'sim' AND hour >= %s GROUP BY hour",
        (room, hours[0])).fetchall()
    for hour, n, change in rows:
        slot = per_hour.get(hour.astimezone(timezone.utc))
        if slot is not None:
            slot["events"] = int(n)
            slot["room_change"] = None if change is None else round(float(change), 3)
    return {
        "days": [{"day": d.isoformat(), **per_day[d]} for d in days],
        "hours": [{"hour": h.timestamp(), **per_hour[h]} for h in hours],
    }


class Store:
    """Copies rows to Tiger from a background thread; when Tiger is unreachable, rows are dropped."""

    def __init__(self, url: str, room: str = config.ROOM, connect=None) -> None:
        self.url, self.room = url, room
        self.connect = connect or self._connect
        self.rows: queue.Queue[tuple[str, tuple] | None] = queue.Queue(maxsize=QUEUE_MAX)
        self.warned = False
        self.thread = threading.Thread(target=self._run, name="tiger-writer", daemon=True)
        self.thread.start()

    def _connect(self):
        import psycopg

        return psycopg.connect(self.url, autocommit=True, connect_timeout=CONNECT_TIMEOUT_S)

    def history(self, entry: dict) -> None:
        self._put(HISTORY_SQL, history_row(entry, self.room))

    def motion(self, wall: float, mode: str, msg: dict) -> None:
        self._put(MOTION_SQL, motion_row(wall, self.room, mode, msg))

    def trends(self) -> dict:
        with self.connect() as conn:
            return trends(conn, self.room, time.time())

    def close(self, timeout: float = 5.0) -> None:
        """Writes what is queued, then stops the writer."""
        self.rows.put(None)
        self.thread.join(timeout)

    def _put(self, sql: str, row: tuple) -> None:
        try:
            self.rows.put_nowait((sql, row))
        except queue.Full:
            self._warn("queue full")

    def _warn(self, why: str) -> None:
        if not self.warned:
            self.warned = True
            print(f"  (Tiger not reachable: {why}; dropping rows until it is)", flush=True)

    def _run(self) -> None:
        conn, retry_at = None, 0.0
        while (item := self.rows.get()) is not None:
            if conn is None and time.monotonic() < retry_at:
                continue
            try:
                if conn is None:
                    conn = self.connect()
                    setup(conn)
                conn.execute(*item)
                if self.warned:
                    self.warned = False
                    print("  (Tiger reachable again)", flush=True)
            except Exception as e:
                self._warn(f"{type(e).__name__}: {str(e).strip().splitlines()[0] if str(e).strip() else ''}")
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass
                conn, retry_at = None, time.monotonic() + RETRY_S
        if conn is not None:
            conn.close()


def from_env() -> Store | None:
    url = os.environ.get("TIGER_DATABASE_URL", "").strip()
    return Store(url) if url else None


def backfill(path: Path, url: str, room: str = config.ROOM) -> tuple[int, int]:
    """Loads History entries from the file; returns (entries read, rows that were new)."""
    import psycopg

    entries = []
    for line in path.read_text().splitlines():
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if isinstance(entry, dict) and "what" in entry and "t" in entry:
            entries.append(entry)
    with psycopg.connect(url, autocommit=True, connect_timeout=CONNECT_TIMEOUT_S) as conn:
        setup(conn)
        count = "SELECT count(*) FROM history WHERE room = %s"
        before = conn.execute(count, (room,)).fetchone()[0]
        with conn.cursor() as cur:
            cur.executemany(HISTORY_SQL, [history_row(e, room) for e in entries])
        after = conn.execute(count, (room,)).fetchone()[0]
        # Backfilled rows can land in already-materialized days, so those are refreshed now. Today is left
        # out: once a day is materialized, rows added to it later stay hidden until the next policy run.
        today = datetime.combine(datetime.now(ZoneInfo(config.TIMEZONE)).date(), datetime.min.time(),
                                 ZoneInfo(config.TIMEZONE))
        conn.execute("CALL refresh_continuous_aggregate('falls_daily', NULL, %s)", (today,))
    return len(entries), after - before


def main(argv: list[str]) -> None:
    from backend.dashboard import HISTORY, load_env

    if not argv or argv[0] != "backfill":
        sys.exit(__doc__)
    load_env()
    url = os.environ.get("TIGER_DATABASE_URL", "").strip()
    if not url:
        sys.exit("TIGER_DATABASE_URL is not set (put it in .env)")
    path = Path(argv[1]) if len(argv) > 1 else HISTORY
    read, new = backfill(path, url)
    print(f"Read {read} History entries from {path}; {new} were new in Tiger.")


if __name__ == "__main__":
    main(sys.argv[1:])
