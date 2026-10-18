"""Tiger Data copies (see docs/adr/0004): rows, the background writer, /trends, and the dashboard hooks.

Set TIGER_TEST_URL to a throwaway TimescaleDB to also run the round trip against a real database.
"""
import contextlib
import io
import os
import tempfile
import time
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from backend import dashboard, server, store

T = 1_792_000_000.0
EVENT = {"label": "walk", "why": "classified", "links": "A", "confidence": 0.9,
         "details": {"peak": 9.1, "static_change": 0.12, "duration_s": 1.2}}


class FakeConn:
    def __init__(self, fail: bool = False) -> None:
        self.fail, self.rows, self.closed = fail, [], False

    def execute(self, sql, params=None):
        if self.fail:
            raise OSError("connection refused")
        self.rows.append((sql, params))
        return self

    def fetchone(self):
        return (True,)

    def close(self):
        self.closed = True


class RecordingStore:
    def __init__(self) -> None:
        self.entries, self.events = [], []

    def history(self, entry):
        self.entries.append(entry)

    def motion(self, wall, mode, msg):
        self.events.append((mode, msg))


class RowTests(unittest.TestCase):
    def test_outcome_names_match_fall_summaries(self):
        self.assertEqual(store.outcome({"what": "countdown_cancelled", "by": "resident"}), "resident_ok")
        self.assertEqual(store.outcome({"what": "countdown_cancelled", "by": "caregiver"}), "caregiver_cancelled")
        self.assertEqual(store.outcome({"what": "late_answer"}), "late_answer")
        self.assertEqual(store.outcome({"what": "resolved", "reason": "false_alarm"}), "false_alarm")
        self.assertIsNone(store.outcome({"what": "alert"}))

    def test_history_row_key_uses_wall_clock_and_fills_missing_parts(self):
        row = store.history_row({"t": T, "wall": T + 50, "what": "notify", "mode": "sim", "id": 4, "fall": 2}, "home")
        self.assertEqual(row[:7], (store.when(T + 50), "home", "notify", 2, 4, "sim", None))
        row = store.history_row({"t": T, "what": "stale_answer", "fall": None, "answer": "ok"}, "home")
        self.assertEqual(row[:6], (store.when(T), "home", "stale_answer", -1, -1, "live"))

    def test_motion_row_takes_room_change_from_static_change(self):
        row = store.motion_row(T, "home", "live", EVENT)
        self.assertEqual(row, (store.when(T), "home", "live", "walk", "classified", "A", 0.9, 9.1, 0.12, 1.2))


class WriterTests(unittest.TestCase):
    def test_rows_are_written_in_order_after_setup(self):
        conn = FakeConn()
        s = store.Store("x", connect=lambda: conn)
        s.history({"t": T, "what": "alert", "fall": 1})
        s.motion(T, "live", EVENT)
        s.close()
        inserts = [sql for sql, _ in conn.rows if sql.startswith("INSERT")]
        self.assertEqual(inserts, [store.HISTORY_SQL, store.MOTION_SQL])
        self.assertTrue(any("create_hypertable('history'" in sql for sql, _ in conn.rows))
        self.assertTrue(conn.closed)

    def test_unreachable_tiger_drops_rows_and_warns_once(self):
        tries = []

        def connect():
            tries.append(1)
            raise OSError("connection refused")

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            s = store.Store("x", connect=connect)
            for k in range(5):
                s.history({"t": T + k, "what": "alert", "fall": 1})
            s.close()
        self.assertEqual(len(tries), 1)  # no retry before RETRY_S, so the rest are dropped at once
        self.assertEqual(out.getvalue().count("Tiger not reachable"), 1)

    def test_full_queue_drops_without_blocking(self):
        s = store.Store.__new__(store.Store)
        s.rows, s.warned, s.room = store.queue.Queue(maxsize=1), False, "home"
        with contextlib.redirect_stdout(io.StringIO()):
            s.history({"t": T, "what": "alert", "fall": 1})
            s.history({"t": T, "what": "alert", "fall": 2})
        self.assertTrue(s.warned)
        self.assertEqual(s.rows.qsize(), 1)

    def test_from_env_is_off_without_url(self):
        old = os.environ.pop("TIGER_DATABASE_URL", None)
        try:
            self.assertIsNone(store.from_env())
        finally:
            if old is not None:
                os.environ["TIGER_DATABASE_URL"] = old


class TrendsEndpointTests(unittest.TestCase):
    def get(self, trends):
        with TestClient(server.make_app(server.Hub(), trends=trends)) as client:
            return client.get("/trends")

    def test_off_without_tiger(self):
        self.assertEqual(self.get(None).status_code, 404)

    def test_unavailable_when_tiger_fails(self):
        def fail():
            raise OSError("down")

        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(self.get(fail).status_code, 503)

    def test_returns_trends(self):
        r = self.get(lambda: {"days": [], "hours": []})
        self.assertEqual((r.status_code, r.json()), (200, {"days": [], "hours": []}))


class DashboardHookTests(unittest.TestCase):
    def test_history_entries_and_motion_events_are_copied(self):
        with tempfile.TemporaryDirectory() as tmp:
            copies = RecordingStore()
            history = dashboard.History(server.Hub(), "live", Path(tmp) / "history.jsonl", copies)
            web = dashboard.WebSink(history.hub, history, dashboard.Notices(history.hub, history))
            history.add(T, "alert", fall=1)
            event = type("E", (), {"cut_short": False, "start": T, "end": T + 1})()
            verdict = type("V", (), {"label": "walk", "confidence": 0.8, "details": {"peak": 3.0}})()
            web.event(T, event, verdict, "A", sim=True)
            web.event(T, event, verdict, "A", sim=False)
        self.assertEqual([e["what"] for e in copies.entries], ["alert"])
        self.assertEqual([mode for mode, _ in copies.events], ["sim", "live"])

    def test_history_works_without_tiger(self):
        with tempfile.TemporaryDirectory() as tmp:
            history = dashboard.History(server.Hub(), "live", Path(tmp) / "history.jsonl")
            history.add(T, "alert", fall=1)
            self.assertEqual(len(history.entries(1)), 1)


@unittest.skipUnless(os.environ.get("TIGER_TEST_URL"), "set TIGER_TEST_URL to a throwaway TimescaleDB")
class RealTigerTests(unittest.TestCase):
    def test_backfill_is_safe_to_rerun_and_trends_leave_out_simulated_falls(self):
        url, room, now = os.environ["TIGER_TEST_URL"], f"test-{time.time_ns()}", time.time()
        lines = [
            {"t": now, "wall": now, "what": "confirmed_fall", "mode": "live", "fall": 1},
            {"t": now + 5, "wall": now + 5, "what": "countdown_cancelled", "mode": "live", "fall": 1, "by": "resident"},
            {"t": now, "wall": now, "what": "confirmed_fall", "mode": "sim", "fall": 2},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history.jsonl"
            path.write_text("\n".join(store.json.dumps(e) for e in lines) + "\nnot json\n")
            self.assertEqual(store.backfill(path, url, room), (3, 3))
            self.assertEqual(store.backfill(path, url, room), (3, 0))
        s = store.Store(url, room)
        s.motion(now, "live", EVENT)
        s.motion(now, "sim", {**EVENT, "label": "fall"})
        s.close()
        t = s.trends()
        self.assertEqual(t["days"][-1], {"day": t["days"][-1]["day"], "falls": 1, "ok": 1,
                                         "cancelled": 0, "alerted": 0, "open": 0})
        self.assertEqual((t["hours"][-1]["events"], t["hours"][-1]["room_change"]), (1, 0.12))


if __name__ == "__main__":
    unittest.main()
