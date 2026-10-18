# Fall Summaries are written by Gemini only after a fall closes

The Caregiver wants to know what happened in a fall without reading every row of Recent activity. A Fall Summary puts it in a few plain sentences. It is written once, when the fall closes: the Countdown cancelled, a Late Answer, or a Resolved Alert. Python works out the numbers from the fall's history entries (Check-in delivery, how long until the Resident answered, whether the Alert went out and was delivered, how and when it closed). Gemini only puts those numbers into words, on a background thread with a 10 s timeout. The summary is saved once in `data/history.jsonl` and shown in the Caregiver view's "Latest fall" card and in the Dev view's History tab.

The Check-in, the Alert and the All-clear stay fixed templates. A model call can take seconds, fail or have no network, and those messages have to go out at once.

## Consequences

- Without `GEMINI_API_KEY`, or when Gemini fails or is slow, the same numbers fill a template, tagged `"source":"template"`, so the card is never empty and the demo works offline.
- Gemini sees only the times, durations and outcome. It never sees the Resident's name, phone numbers, message texts or the signal details. It writes "the Resident", and the backend puts `RESIDENT_NAME` in afterwards.
- A fall still open when the backend stops never closes, so it gets no summary.
- Summaries are not texted. The Caregiver already got the Alert, and after a cancelled Countdown they don't need a text.

## Considered Options

- Writing the Alert itself with Gemini: rejected because it adds latency and a point of failure to the one message that must not wait.
- Calling Gemini from the browser: rejected because it would put the API key in the page.
- Calling Gemini from the messaging sidecar: rejected because the dashboard would not get the summary.
