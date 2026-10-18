"""Fall Summaries: metrics per closing path, the template fallback, one per fall, and replay."""
import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend import dashboard, server, summary
from backend.detection.pipeline import Pipeline

T = 1_792_000_000.0


def fall(*extra):
    return [
        {"t": T, "what": "notify", "id": 1, "to": "resident", "kind": "check_in", "fall": 3, "status": "sending"},
        {"t": T + 0.2, "what": "confirmed_fall", "fall": 3, "links": "A",
         "details": {"peak": 24.97, "static_change": 0.58, "duration_s": 3.09}},
        {"t": T + 1, "what": "delivery", "id": 1, "fall": 3, "status": "sent"},
        *extra,
    ]


ALERT = [
    {"t": T + 10.2, "what": "alert", "fall": 3},
    {"t": T + 10.2, "what": "notify", "id": 2, "to": "caregiver", "kind": "alert", "fall": 3, "status": "sending"},
    {"t": T + 11, "what": "delivery", "id": 2, "fall": 3, "status": "not_delivered"},
]


class MetricsTests(unittest.TestCase):
    def test_resident_cancels(self):
        m = summary.metrics(fall({"t": T + 6.2, "what": "countdown_cancelled", "fall": 3, "by": "resident"}))
        self.assertEqual(m["outcome"], "resident_ok")
        self.assertEqual((m["answer"], m["answer_after_s"]), ("ok", 6))
        self.assertEqual(m["check_in"], "sent")
        self.assertIsNone(m["alert_after_s"])
        self.assertEqual((m["links"], m["peak"], m["room_change"]), ("A", 24.97, 0.58))
        self.assertIn("answered the Check-in 6 s later", summary.template(m))

    def test_caregiver_cancels(self):
        m = summary.metrics(fall({"t": T + 4.2, "what": "countdown_cancelled", "fall": 3, "by": "caregiver"}))
        self.assertEqual(m["outcome"], "caregiver_cancelled")
        self.assertIsNone(m["answer"])
        self.assertIn("You cancelled the Countdown 4 s after", summary.template(m))

    def test_late_answer(self):
        m = summary.metrics(fall(*ALERT, {"t": T + 40.2, "what": "late_answer", "fall": 3, "late_s": 30}))
        self.assertEqual(m["outcome"], "late_answer")
        self.assertEqual((m["alert_after_s"], m["alert"]), (10, "not_delivered"))
        self.assertEqual(m["closed_after_s"], 40)
        self.assertIn("the Alert text could not be delivered", summary.template(m))

    def test_help_then_handled(self):
        m = summary.metrics(fall({"t": T + 5.2, "what": "help_request", "fall": 3}, *ALERT,
                                 {"t": T + 30.2, "what": "called", "fall": 3},
                                 {"t": T + 200.2, "what": "resolved", "fall": 3, "reason": "handled"}))
        self.assertEqual((m["outcome"], m["answer"], m["help_request"], m["called"]), ("handled", "help", True, True))
        text = summary.template(m)
        self.assertIn("asked for help 5 s after", text)
        self.assertIn("marked it handled 3 min after", text)

    def test_replayed_fall_shows_when_it_really_happened(self):
        entries = fall({"t": T + 6.2, "what": "countdown_cancelled", "fall": 3, "by": "resident"})
        entries[1]["wall"] = T + 86_400
        m = summary.metrics(entries)
        self.assertEqual(m["fall_t"], T + 86_400)
        self.assertEqual(m["answer_after_s"], 6)

    def test_gemini_never_sees_names_numbers_or_signal(self):
        m = summary.metrics(fall({"t": T + 6.2, "what": "countdown_cancelled", "fall": 3, "by": "resident"}))
        sent = json.dumps(summary.facts(m))
        for hidden in ("links", "peak", "room_change", "text", "+1"):
            self.assertNotIn(hidden, sent)


class WriteTests(unittest.TestCase):
    def setUp(self):
        self.m = summary.metrics(fall({"t": T + 6.2, "what": "countdown_cancelled", "fall": 3, "by": "resident"}))

    def test_template_without_key(self):
        with patch.dict(os.environ, {"RESIDENT_NAME": "Rose"}, clear=True), \
                patch.object(summary, "ask_gemini") as ask:
            text, source = summary.write(self.m)
        ask.assert_not_called()
        self.assertEqual(source, "template")
        self.assertIn("Rose answered", text)

    def test_gemini_text_gets_the_name(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "k", "RESIDENT_NAME": "Rose"}, clear=True), \
                patch.object(summary, "ask_gemini", return_value="The Resident said they were OK."):
            self.assertEqual(summary.write(self.m), ("Rose said they were OK.", "gemini"))

    def test_next_model_when_first_is_busy(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "k"}, clear=True), \
                patch.object(summary, "ask_gemini", side_effect=[RuntimeError("503"), "Fine."]) as ask, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(summary.write(self.m), ("Fine.", "gemini"))
        self.assertEqual([c.args[2] for c in ask.call_args_list], list(summary.config.GEMINI_MODELS))

    def test_template_when_gemini_fails(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "k"}, clear=True), \
                patch.object(summary, "ask_gemini", side_effect=TimeoutError("slow")), \
                contextlib.redirect_stdout(io.StringIO()):
            text, source = summary.write(self.m)
        self.assertEqual(source, "template")
        self.assertIn("The Resident answered", text)


class DashboardTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name) / "history.jsonl"
        env = patch.dict(os.environ, {}, clear=True)
        env.start()
        self.addCleanup(env.stop)

    def make(self):
        hub = server.Hub()
        history = dashboard.History(hub, "live", self.path)
        web = dashboard.WebSink(hub, history, dashboard.Notices(hub, history),
                                dashboard.Summaries(hub, history, background=False))
        pipe = Pipeline(web)
        return hub, dashboard.Dashboard(hub, pipe, web, web)

    def confirm(self, view):
        t = view.clock.now()
        view.pipe.flow.candidate(t, t + 4)
        view.pipe.flow.confirm(t)
        return t

    def summaries(self):
        entries = [json.loads(line) for line in self.path.read_text().splitlines()]
        return [e for e in entries if e["what"] == "summary"]

    def test_one_summary_when_resident_says_ok(self):
        _, view = self.make()
        self.confirm(view)
        view._command({"cmd": "ok"})
        view.web.summaries.close(view.clock.now(), view.web.fall)
        [s] = self.summaries()
        self.assertEqual((s["fall"], s["source"], s["metrics"]["outcome"]), (1, "template", "resident_ok"))

    def test_summary_after_alert_is_resolved(self):
        _, view = self.make()
        t = self.confirm(view)
        view.pipe.flow.tick(t + 11)
        self.assertEqual(self.summaries(), [])
        view._command({"cmd": "resolve", "reason": "false_alarm"})
        [s] = self.summaries()
        self.assertEqual(s["metrics"]["outcome"], "false_alarm")

    def test_summaries_replay_after_restart_even_when_old(self):
        _, view = self.make()
        self.confirm(view)
        view._command({"cmd": "cancel"})
        with self.path.open("a") as f:
            for i in range(dashboard.HISTORY_SHOWN + 5):
                f.write(json.dumps({"t": T + i, "what": "called", "mode": "live"}) + "\n")
        hub, view = self.make()
        replayed = [json.loads(text) for _, text in hub.kept]
        self.assertEqual([e["fall"] for e in replayed if e.get("what") == "summary"], [1])
        self.assertIn(1, view.web.history.summarized)
        view.web.summaries.close(T, 1)
        self.assertEqual(len(self.summaries()), 1)


if __name__ == "__main__":
    unittest.main()
