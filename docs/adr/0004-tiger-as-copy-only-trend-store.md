# Tiger Data only receives copies; Trends are the one thing that reads it

We use Tiger Data (hosted TimescaleDB) to show Trends: Confirmed Falls per day and Motion Events per hour, across many runs. Tiger only receives copies of the data. `data/history.jsonl` still decides what the dashboard replays at startup, and the live view still comes from the WebSocket. The live demo must not depend on a network database, so nothing in the Fall Flow waits for Tiger or reads from it.

## Consequences

- Without `TIGER_DATABASE_URL` in `.env`, nothing is written to Tiger, the Trends card and popup are hidden, and the app behaves as it did before.
- Writes go through a small in-memory queue on a background thread. When Tiger cannot be reached, rows are dropped, one warning is logged, and nothing is retried.
- Each History row has a natural key (room, time, kind of entry, fall number), and inserts that hit an existing key are skipped. The backfill script that loads `history.jsonl` into Tiger can be re-run at any time to catch up, and it never duplicates live writes. Motion Events from the time Tiger was down are lost, because they are not saved anywhere else.
- Two hypertables (History entries and Motion Events). Each Trend comes from its own continuous aggregate. Motion Events older than 7 days are compressed, and nothing is deleted.
- Simulated Falls are stored with a flag and left out of Trends.
- The browser reads Trends only through the backend's `GET /trends`, so it never connects to Tiger or sees its credentials. The Dev view's Trends card shows both charts, including the average Room Change line. The Caregiver view's "Past two weeks" popup leaves out Room Change.
- Raw signal frames from a Session are never stored in Tiger.

## Considered Options

- Tiger as the main store, replacing `history.jsonl`: rejected because a dropped connection during the demo would empty the dashboard.
- Writing to Tiger without anything reading from it: rejected because it shows no real use of Tiger.
- Trends always visible in the Caregiver view: rejected because they would crowd the calm Caregiver screen.
