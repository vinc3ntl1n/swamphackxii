# Messaging sidecar

Sends the Check-in to the Resident and the Alert, All-clear and Help Request to the Caregiver as texts, and turns the Resident's replies into the dashboard's `ok` and `help` commands. It is a separate Node process that talks to `run.py` over the same `/ws` the dashboard uses ([docs/adr/0001](../docs/adr/0001-messaging-sidecar-over-ws.md)).

## Try it without Photon

```bash
python -m backend.run --replay backend/assets/sim_fall.log.gz   # terminal 1
node messaging/sidecar.mjs --console                            # terminal 2
```

Texts print in terminal 2. When the Check-in appears, type `1` and Enter: the dashboard shows "said they're OK" and the countdown stops. Type `2` for a Help Request. Anything else is passed to the Caregiver. Needs Node 22 (built-in WebSocket).

## Real texts with Photon

The sidecar sends through Photon's Spectrum SDK. By default it uses `@spectrum-ts/imessage-local`, which drives this Mac's Messages app: free, no Photon account.

1. On the Mac that will send (signed into Messages with its own Apple ID):
   ```bash
   cd messaging && npm install
   ```
2. Give the terminal app you run it from (Terminal, iTerm, VS Code) **Full Disk Access** in System Settings → Privacy & Security, then restart that app. Replies are read from `~/Library/Messages/chat.db`. Without access the sidecar stops with "unable to open database file".
3. Put the numbers in `.env` at the repo root (gitignored), in `+1…` form:
   ```
   RESIDENT_PHONE=+15551234567
   CAREGIVER_PHONE=+15557654321
   RESIDENT_NAME=Rose
   ```
4. Run it:
   ```bash
   python -m backend.run --replay backend/assets/sim_fall.log.gz   # terminal 1
   node messaging/sidecar.mjs                                      # terminal 2
   ```
   The first send may make macOS ask whether the terminal may control Messages; allow it.

Use three separate accounts: the Mac running the sidecar sends, and the Resident and the Caregiver each have their own phone. Messages the Mac sends itself are never read as replies.

"sent" means Messages accepted the text, not that it reached the phone. If the number has no iMessage, Messages may fall back to SMS (needs an iPhone paired for Text Message Forwarding) or fail silently.

To switch to a Photon cloud line later, set `PHOTON_PROVIDER=cloud`, `PHOTON_PROJECT_ID` and `PHOTON_PROJECT_SECRET` in `.env` and run `npm install @spectrum-ts/imessage` in `messaging/`. That path is written but not yet tested.

## What it does on the wire

- On connect, the dashboard first replays recent messages, then sends `{"type":"live"}`. Anything before `live` was already handled or timed out, so it is skipped.
- For each `{"type":"notify","status":"sending","id":…,"to":"resident"|"caregiver","kind":…,"text":…}`, it texts `text` to `to` and answers `{"cmd":"delivered","id":…,"status":"sent"|"failed"}`. Without an answer within 10 s the dashboard shows "not delivered".
- A reply of `1`, `ok`, `okay`, `yes`, `fine`, `I'm OK` or 👍 sends `{"cmd":"ok"}`; `2` or `help` sends `{"cmd":"help"}`. The dashboard decides which fall that answers, and ignores Stale Answers.
