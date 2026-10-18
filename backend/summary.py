"""Fall Summaries: a few plain sentences about one fall, written once it closes (see docs/adr/0003).

The numbers come from the fall's history entries; Gemini only puts them into words. Without a key,
or when Gemini fails or is slow, the same numbers fill a template instead.
"""
from __future__ import annotations

import json
import os
from datetime import datetime

from backend import config

CLOSES = ("countdown_cancelled", "late_answer", "resolved")
INSTRUCTIONS = (
    "You write a short summary of one possible fall for a Caregiver, from the JSON facts given. "
    "Write at most 3 plain sentences, no lists or formatting. Call the person who fell \"the Resident\" "
    "and address the Caregiver as \"you\". Use only the facts given: never guess at injuries, causes or "
    "what happened in the room, and give no medical advice. Times are local, durations are in seconds."
)
OUTCOMES = {
    "resident_ok": "the Resident cancelled the Countdown by answering they were OK",
    "caregiver_cancelled": "the Caregiver cancelled the Countdown",
    "late_answer": "the Resident answered they were OK after the Alert",
    "handled": "the Caregiver marked the Alert handled",
    "false_alarm": "the Caregiver marked the Alert a false alarm",
}
UNDELIVERED = ("failed", "not_delivered")


def metrics(entries: list[dict]) -> dict:
    """What happened in one fall, from its history entries in order."""
    first = next((e for e in entries if e.get("what") == "confirmed_fall"), None)
    notices = {e.get("id"): e for e in entries if e.get("what") == "notify"}
    status = {e.get("id"): e.get("status") for e in entries if e.get("what") == "delivery"}
    start = first or min(entries, key=lambda e: e["t"])
    fall_t = start["t"]

    def after(entry: dict | None) -> int | None:
        return None if entry is None else round(entry["t"] - fall_t)

    def sent(kind: str) -> str | None:
        nid = next((i for i, e in notices.items() if e.get("kind") == kind), None)
        return None if nid is None else status.get(nid, "pending")

    def find(*whats: str, **match) -> dict | None:
        return next((e for e in entries if e.get("what") in whats
                     and all(e.get(k) == v for k, v in match.items())), None)

    help_request = find("help_request")
    answer = find("help_request", "late_answer") or find("countdown_cancelled", by="resident")
    alert = find("alert")
    close = next((e for e in reversed(entries) if e.get("what") in CLOSES), None)
    if close is None:
        outcome = None
    elif close["what"] == "countdown_cancelled":
        outcome = "resident_ok" if close.get("by") == "resident" else "caregiver_cancelled"
    elif close["what"] == "late_answer":
        outcome = "late_answer"
    else:
        outcome = close.get("reason")
    details = (first or {}).get("details", {})
    return {
        "fall_t": start.get("wall", fall_t),
        "check_in": sent("check_in"),
        "answer": None if answer is None else ("help" if answer is help_request else "ok"),
        "answer_after_s": after(answer),
        "help_request": help_request is not None,
        "alert_after_s": after(alert),
        "alert": sent("alert"),
        "called": find("called") is not None,
        "outcome": outcome,
        "closed_after_s": after(close),
        "links": (first or {}).get("links"),
        "peak": details.get("peak"),
        "room_change": details.get("static_change"),
    }


def facts(m: dict) -> dict:
    """The part of the metrics Gemini sees: no names, numbers or message texts, and no signal details."""
    keep = ("check_in", "answer", "answer_after_s", "help_request", "alert_after_s", "alert",
            "called", "closed_after_s")
    return {"fall_time": when(m["fall_t"]), "outcome": OUTCOMES.get(m["outcome"], m["outcome"]),
            **{k: m[k] for k in keep}}


def when(t: float) -> str:
    return datetime.fromtimestamp(t).strftime("%-I:%M %p")


def span(s: int | None) -> str:
    if s is None:
        return "?"
    return f"{s} s" if s < 90 else f"{round(s / 60)} min"


def template(m: dict) -> str:
    out = [f"A possible fall was detected at {when(m['fall_t'])}."]
    if m["check_in"] in UNDELIVERED:
        out.append("The Check-in text could not be delivered.")
    outcome = m["outcome"]
    if outcome == "resident_ok":
        out.append(f"The Resident answered the Check-in {span(m['answer_after_s'])} later to say they were OK, "
                   "so no Alert was sent.")
    elif outcome == "caregiver_cancelled":
        out.append(f"You cancelled the Countdown {span(m['closed_after_s'])} after the fall, so no Alert was sent.")
    else:
        if m["help_request"]:
            out.append(f"The Resident asked for help {span(m['answer_after_s'])} after the fall.")
        alerted = (f"You were alerted {span(m['alert_after_s'])} after the fall"
                   if m["alert_after_s"] is not None else "You were alerted")
        if m["alert"] in UNDELIVERED:
            alerted += ", but the Alert text could not be delivered"
        if outcome == "late_answer":
            out.append(f"{alerted}, and the Resident answered they were OK "
                       f"{span(m['closed_after_s'])} after the fall.")
        elif outcome in ("handled", "false_alarm"):
            how = "handled" if outcome == "handled" else "a false alarm"
            out.append(f"{alerted}, and you marked it {how} {span(m['closed_after_s'])} after the fall.")
        else:
            out.append(f"{alerted}.")
    return " ".join(out)


def ask_gemini(m: dict, key: str, model: str) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=int(config.SUMMARY_TIMEOUT_S * 1000)))
    reply = client.models.generate_content(
        model=model, contents=json.dumps(facts(m)),
        config=types.GenerateContentConfig(system_instruction=INSTRUCTIONS, temperature=0.2,
                                           max_output_tokens=1024),
    )
    text = " ".join((reply.text or "").split())
    if not text:
        raise ValueError("Gemini returned no text")
    return text[:config.SUMMARY_MAX_CHARS]


def named(text: str, name: str) -> str:
    if not name:
        return text
    return text.replace("The Resident", name).replace("the Resident", name)


def write(m: dict) -> tuple[str, str]:
    """The summary text and where it came from: "gemini", or "template" when Gemini can't answer."""
    key = os.environ.get("GEMINI_API_KEY", "")
    name = os.environ.get("RESIDENT_NAME", "")
    models = [os.environ["GEMINI_MODEL"]] if os.environ.get("GEMINI_MODEL") else list(config.GEMINI_MODELS)
    for model in models if key else []:
        try:
            return named(ask_gemini(m, key, model), name), "gemini"
        except Exception as e:
            print(f"  (Gemini summary with {model} failed: {e!r})", flush=True)
    if key:
        print("  (using the template summary)", flush=True)
    return named(template(m), name), "template"
