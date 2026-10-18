# Messaging sidecar

Sends the Check-in to the Resident and the Alert, All-clear and Help Request to the Caregiver as texts, and turns the Resident's replies into the dashboard's `ok` and `help` commands. It is a separate Node process that talks to `run.py` over the same `/ws` the dashboard uses ([docs/adr/0001](../docs/adr/0001-messaging-sidecar-over-ws.md)).

## Try it without Photon

```bash
python -m backend.run --replay backend/assets/sim_fall.log.gz   # terminal 1
node messaging/sidecar.mjs --console                            # terminal 2
```

Texts print in terminal 2. When the Check-in appears, type `1` and Enter: the dashboard shows "said they're OK" and the countdown stops. Type `2` for a Help Request. Anything else is passed to the Caregiver. Needs Node 22 (built-in WebSocket).

## Connect Photon

Fill in the two functions at the top of `sidecar.mjs`; nothing else needs to change:

- `sendText(phone, text)`: send one iMessage/SMS. Throw if it fails, so the dashboard shows "couldn't send".
- `listenForReplies(onReply)`: call `onReply(phone, text)` for every incoming message. Replies from numbers other than `RESIDENT_PHONE` are ignored.

Put the numbers in `.env` at the repo root (gitignored):

```
RESIDENT_PHONE=+15551234567
CAREGIVER_PHONE=+15557654321
RESIDENT_NAME=Rose
```

Use three separate accounts: the Mac running the sidecar sends, and the Resident and the Caregiver each have their own phone. With `imessage-local`, that Mac must be signed into Messages, and the terminal may need Full Disk Access.

## What it does on the wire

- On connect, the dashboard first replays recent messages, then sends `{"type":"live"}`. Anything before `live` was already handled or timed out, so it is skipped.
- For each `{"type":"notify","status":"sending","id":…,"to":"resident"|"caregiver","kind":…,"text":…}`, it texts `text` to `to` and answers `{"cmd":"delivered","id":…,"status":"sent"|"failed"}`. Without an answer within 10 s the dashboard shows "not delivered".
- A reply of `1`, `ok`, `okay`, `yes`, `fine`, `I'm OK` or 👍 sends `{"cmd":"ok"}`; `2` or `help` sends `{"cmd":"help"}`. The dashboard decides which fall that answers, and ignores Stale Answers.
