# Camera-Free Fall Detection

SwampHacks XII project. Detects falls in a room from how a person's movement disturbs Wi-Fi signals, then warns a caregiver. No camera, nothing to wear.

## How it works

Three Seeed XIAO ESP32-S3 boards make two links across the room. The sender broadcasts Wi-Fi packets about 100 times a second. Two receivers, A and B, each measure the **Channel State Information (CSI)** of every packet: how strongly each of its frequency subcarriers arrived (about 166 usable ones). A human body absorbs and reflects 2.4 GHz Wi-Fi, so when someone moves near the line between the sender and a receiver, that receiver's numbers change. The two lines fan out from the sender in a V, so which link reacts tells which part of the room the movement was in.

Both receivers stream CSI to a laptop over USB, where the Python backend:

1. **Parses** each `CSI_DATA` line into an amplitude per subcarrier, tagged with the link it came from (`capture/csi.py`).
2. **Measures motion energy** for each link separately: how much the average signal shifted over the last 0.5 s, divided by that link's empty-room baseline, learned during a 10 s calibration (`detection/motion.py`).
3. **Cuts Motion Events**: a burst of movement on a link lasts from when its energy crosses `T_ON` until it stays below `T_OFF` for 1 s. Bursts on both links that overlap make one Motion Event, which ends once every link is quiet. The link with the strongest burst supplies the numbers for the next steps, and the log lists which links moved (`links A`, `links B` or `links A+B`) (`detection/pipeline.py`).
4. **Waits through a 4 s Stillness Window.** If the person moves again on any link, it wasn't a fall.
5. **Classifies** the event as walk, sit-down, or fall (`detection/features.py`, `detection/detector.py`). The key signals are how hard the signal moved (`peak`) and how different the room looks before vs. after (`static_change`): a body lying on the floor changes the room far more than one sitting in a chair.
6. **Runs the fall flow** (`detection/fallflow.py`): a confirmed fall starts a 10 s countdown with an alarm. If nobody cancels, the caregiver is alerted.

```
              ┌──► receiver A XIAO ──USB──┐
sender XIAO ──┤ Wi-Fi                     ├──► csi.py ► motion.py ► features.py ► detector.py ► fallflow.py
              └──► receiver B XIAO ──USB──┘                                                          │
                                                                                       alarm / cancel button / alert
```

Two detectors share the same interface:

- **RandomForestDetector**: the default, a saved scikit-learn model trained on labeled Recordings by `train.py`. A missing or incompatible artifact stops startup with setup instructions.
- **ThresholdDetector**: hand-tuned rules from `config.py`, selected explicitly with `--detector threshold`.

## Hardware

| Item | Role |
|---|---|
| XIAO ESP32-S3 + external antenna | Sender, running Espressif's [esp-csi](https://github.com/espressif/esp-csi) `csi_send` example. It only needs power, so a USB power bank works. |
| XIAO ESP32-S3 + external antenna | Receiver A, running `csi_recv`, plugged into the laptop |
| XIAO ESP32-S3 + external antenna | Receiver B, running `csi_recv`, plugged into the laptop |
| Arduino + push button + buzzer | In-room cancel button and alarm (`arduino/cancel_button`) |

Put the boards at waist height with each receiver 3–4 m from the sender, so the two lines open 60–90° apart in a V from the sender. Each link is most sensitive close to its line, so run line A over the spot where a person would land and line B past the chair, and walk paths that cross both.

To tell the boards apart over USB: a receiver prints `CSI_DATA` lines, and the sender's boot log says `Project name: csi_send`.

**Arduino wiring:** button between pin 2 and GND (internal pull-up), buzzer + to pin 8 and − to GND. Serial at 9600 baud: it sends `CANCEL` when the button is pressed, and turns the alarm on/off on `A1` / `A0`.

### Flashing the boards

```bash
# ESP-IDF v5.5.x and esp-csi
git clone -b v5.5.1 --recursive https://github.com/espressif/esp-idf.git ~/esp/esp-idf
cd ~/esp/esp-idf && ./install.sh esp32s3 && . ./export.sh
git clone https://github.com/espressif/esp-csi.git ~/esp/esp-csi

cd ~/esp/esp-csi/examples/get-started/csi_send
idf.py set-target esp32s3 && idf.py -p /dev/cu.usbmodem<SENDER> flash

cd ../csi_recv
idf.py set-target esp32s3 && idf.py menuconfig
idf.py -p /dev/cu.usbmodem<RECEIVER_A> flash
idf.py -p /dev/cu.usbmodem<RECEIVER_B> flash
```

In `menuconfig`, set **Component config → ESP System Settings → Channel for console output → USB Serial/JTAG Controller**. The XIAO's USB-C port is the chip's built-in USB, not a UART, so without this no CSI reaches the laptop.

Before building `csi_recv`, delete the `esp_wifi_set_mac(WIFI_IF_STA, CONFIG_CSI_SEND_MAC)` line at the end of `wifi_init()` in `main/app_main.c`. It gives the receiver the sender's MAC address, and the receiver then drops the sender's packets. Both receivers run the same build.

## Running

```bash
python3.12 -m venv .venv-rf && source .venv-rf/bin/activate
pip install -r requirements-rf.txt
# Copy a trusted forest to data/model.joblib, or train from local recordings:
python -m backend.training.train
python -m backend.run
```

The shared RF artifacts require scikit-learn 1.6.1 and NumPy 2.1.3. The existing Python 3.14 `.venv` cannot install those NumPy wheels; use the separate Python 3.12 environment above. To load another local artifact, pass `--model /local/path/baseline.joblib` or set `SWAMP_RF_MODEL` in `.env`. Set `SWAMP_DETECTOR=threshold` only when intentionally using the heuristic. Prediction loads the forest once and does not access R2. See [RF runtime setup](docs/random-forest.md).

Run everything from the repo root with `python -m`. Each command finds the receivers by their USB serial numbers in `config.RECEIVERS`, so it doesn't matter which port each one is plugged into. The sender's serial number is in `config.SENDER_BOARD`, so it is skipped if it's on USB for power. If one receiver is missing, the command says so and runs on the other. `--port` (receiver A) and `--port-b` (receiver B) override the lookup.

| Command | What it does |
|---|---|
| `python -m backend.capture.live` | Check capture: prints packet rate, subcarriers, RSSI and a motion bar for each receiver, once a second. Walking across one link's line should move only that link's bar. Saves the raw stream to `data/sessions/`. |
| `python -m backend.run` | Full detection pipeline live, in the terminal and on the dashboard at `http://127.0.0.1:8000/dashboard` (see below). Stay still for the first 10 s while it calibrates. `--no-view` keeps it terminal-only, `--view-port 8001` moves it. |
| `python -m backend.run --replay data/sessions/<file>.log --speed 0` | Same, on a saved session (`--speed 0` = as fast as possible). A Recording (`data/recordings/*/*.log.gz`) replays with the calibration saved next to it, since the clip is shorter than the 10 s calibration. |
| `python -m backend.capture.replay <log>` | Play a session log or Recording through the capture meter. |
| `python -m backend.training.record` | Open the persistent training dashboard at `http://127.0.0.1:8000/`. Choose Fall, Sit Down, Walk or Empty Room, batch size, and Training or Test, then start recording. Run subsequent batches from the same page. |
| `python -m backend.training.record fall --n 10` | Record 10 labeled clips: wait for 3-2-1-GO, move, hold still until "saved". Labels: `fall`, `sit-down`, `walk`, `empty`. Add `--split test` for the test set, `--recalibrate` after moving the boards. Also opens the live record page at `http://127.0.0.1:8000/` (see below); `--no-view` turns it off, `--view-port 8001` moves it. The laptop also speaks the cues (see below); `--no-voice` goes back to the beeps. |
| `python -m backend.training.calibrate` | Learn the empty room again for every receiver that sends packets, save it to `data/calibration.json` and exit. After the announcement there are 3 s to leave, then everyone stays out of the area and nobody moves for 10 s; if something moved, it tries again. Shows the live page and speaks like `record.py` (`--no-view`, `--view-port`, `--no-voice`). |
| `python -m backend.training.train` | Train the random forest using training-fold validation, grouping by session when IDs are available. Add `--evaluate-test` for final test scoring. Writes `data/model.joblib` and `data/results.json`; see the [RF workflow](docs/random-forest.md). |
| `python -m backend.run --detector rf` | Run live RF detection with the dashboard. Add `--model <path>` to use an experimental forest, or `--no-view` for terminal output only. |
| `python -m backend.sync push` / `pull` | Upload `data/` to the R2 bucket, or download it (see Data storage above). |

All tunable numbers (thresholds, timings) are in `backend/config.py`. Phone numbers stay out of git, in `.env` (see the dashboard section below).

Session logs and Recordings store one packet per line: `<laptop time>\t<link>\t<CSI_DATA line>`. Files from before the second receiver have no link column and read as link A, so the old Recordings still train as before. A Recording's `.json` keeps one calibration per link, and its `rate_hz` gives packets per second per link. `calibration.json` also keeps one calibration per link. `record.py` recalibrates when a receiver has none saved yet.

### Data storage

Nothing in `data/` is in git, so the public repo never contains training data. Recordings, `calibration.json`, `results.json` and `model.joblib` live in the private Cloudflare R2 bucket `swamphacksdata` (`config.R2_BUCKET`), and each laptop keeps a copy in `data/`. Session logs in `data/sessions/` stay on the laptop that made them.

To set up a laptop, get the R2 keys from a teammate privately (or make your own R2 API token with Object Read & Write on the bucket) and put them in a `.env` file in the repo root. `.env` is in `.gitignore`; never commit the keys or post them anywhere public.

```
R2_ENDPOINT=https://<account id>.r2.cloudflarestorage.com
R2_ACCESS_KEY_ID=...
R2_SECRET_ACCESS_KEY=...
```

Then download the data with `python -m backend.sync pull`. Day to day:

1. `python -m backend.sync pull` before you start, to get what others recorded.
2. Record, calibrate or train as usual. Everything is saved to `data/` first.
3. `python -m backend.sync push` when you're done, so the bucket has it.

Both commands compare each file's MD5 with the bucket and only copy files that are missing or different. A pull overwrites a local `calibration.json` or `model.joblib` that wasn't pushed yet, so push before you pull if you recalibrated or retrained. Nothing is ever deleted: to drop a bad Recording, delete it locally and in the R2 dashboard.

### The live record page

Run `python -m backend.training.record` once to open the training dashboard. Its test buttons choose **Fall**, **Sit Down**, **Walk** or **Empty Room**. Set the number of clips, choose **Training** or **Test**, and press **Start recording**. Position time includes the 3-2-1 count; Session details contains the performer and room saved with each clip. **Stop** ends the current batch, keeps completed clips, and discards the unfinished clip. The page stays open for the next batch, and **Recalibrate** also works between batches. Receiver errors appear on the page so the session can be retried after reconnecting the boards. Only one batch or calibration runs at a time.

Passing a label, such as `python -m backend.training.record fall --n 10`, opens the dashboard and starts that batch immediately. `--no-view` keeps the single-batch terminal workflow. If the dashboard port is busy, choose another with `--view-port 8001`. Ctrl-C in its original terminal closes the dashboard.

During recording, the page shows when to move and what the Wi-Fi signal is doing, so the person performing the clips can follow it from 3–4 m away. Reloading the page rebuilds the current batch and its last minute of signal.

The **left half** is for the performer:

- A huge phase word with a progress bar. `GET IN POSITION` (with seconds until the count), then `3`, `2`, `1` with **HOLD STILL** under them, then `GO!` with the action in violet, a color used for nothing else. About 2 s after GO it switches to `HOLD STILL / STAY DOWN` (fall), `HOLD STILL / STAY SEATED` (sit-down) or `KEEP WALKING` (walk) until the clip window ends. Empty-room clips never show `GO!`: they show `NOBODY MOVES` for the whole window. Then `✓ SAVED` in green, or the problem (`⚠ MOVED BEFORE GO`, `⚠ MOVED AT THE END`, …) in amber, with "do an extra". After the last clip that result stays up, with `DONE` and the session totals under it. Long cues wrap onto two lines so they stay big.
- A red **WAIT FOR GO** banner when the live motion energy crosses `T_ON` during the 3-2-1 count. It clears below `T_OFF`. This is the same energy rule `record.py` later uses to flag "moved before GO". A third of the existing movement clips were flagged that way, because people start moving on "1".
- The label's instruction, a timeline strip of the current clip (position, 3, 2, 1, GO, hold, saved) with a cursor and the recorded window under it, the last result, and the list of saved clips with each clip's energy around GO.
- A **Recalibrate** button, for when the boards moved or the room changed. During a batch it queues one recalibration, and every open page says "Recalibrating before the next clip". `record.py` does it at the start of the next clip, before `GET IN POSITION`. If a clip is still in its get-in-position wait, it cuts the wait short, recalibrates, then starts that clip's wait again. It never interrupts 3-2-1, GO or the recording. Once the last clip's 3-2-1 has started there is no clip left to recalibrate before, so the press is refused; recalibrate after the batch finishes. The header says `RECALIBRATING` and the page shows `CALIBRATING` with "everyone out of the area, nobody moves" and a progress bar. Learning starts 3 s after the announcement (`CALIBRATE_SETTLE_S`), so whoever is still on the mat has time to leave. When it's done the page goes straight to the next `GET IN POSITION`, and the clips after it use and save the new calibration. The button is greyed out while a recalibration is queued or running and while the page is disconnected.

The **right half** is the live signal, all on one time axis (newest at the right, last 20 s):

- **Waveform**: 8 subcarriers drawn over time like a seismograph. Each lane shows how far that subcarrier is from its own median over the last 20 s, in dB, through a soft `tanh(d / 8 dB)` scale. An empty room gives nearly flat lines, and a person moving gives big swings. When someone blocks the path, the lanes dip together. The lanes are row positions 63, 88, 94, 108, 133, 141, 147 and 156 of the 166 usable subcarriers. Across the 58 labeled clips they react most to movement while staying the least alike. With any other row width the lanes are spread evenly.
- **Subcarrier heatmap**: time across, subcarrier up. *Amplitude* colors the raw level (0–110). *Change* colors the dB difference from each subcarrier's 20 s median, so movement and the Room Change after a fall stand out (red = signal dropped).
- **Motion energy** with dashed `T_ON` / `T_OFF` lines.
- **CSI shape**: amplitude across subcarriers for the last 0.5 s, fading.
- Markers for 3, 2, 1 and GO run down through the time charts, along with the window `[GO − 1.5 s, GO + 8.5 s]` that goes into the Recording. The window is shaded on the waveform and energy charts and drawn as edge lines on the heatmap, so its colors stay true. Chips show LIVE/REPLAY, packets per second, RSSI and the Motion Level. A warning appears when no packets have arrived for 2 s.

With two receivers, the page shows link A's signal, or link B's if A isn't plugged in. The Recording still saves both links.

`calibrate.py` shows the same page without the Recalibrate button: `CALIBRATING`, then `✓ CALIBRATED` with the links it saved.

### Spoken cues

The performer doesn't have to look at the screen: the laptop speaks each phase with macOS `say`. "Clip 3 of 10. Get in position.", then "Three", "Two", "One" on the count, then "Go! Fall." ("Go! Sit down.", "Go! Walk.", or "Recording. Nobody move." for empty-room clips). About 2 s after GO it says "Stay down. Hold still." for falls and "Stay seated. Hold still." for sit-downs. After the Pop/Basso it says "Saved." or the first problem and "Do an extra.", e.g. "Moved before go. Do an extra.", and at the end "Done. 8 clean, 2 flagged." Calibration says "Calibrating. Everyone out, nobody move." and "Calibrated.", and a Recalibrate press says "Recalibrating before the next clip."

The spoken count and GO replace the Tink/Glass beeps. A new phrase stops the one still playing, so the numbers never queue up late. A plain `say` starts sounding about 0.36 s after it is launched, so the count words and GO are rendered to audio files when the session starts and played with `afplay`, which starts in about 0.08 s. `--no-voice`, or a machine without `say`, gives the old beeps. Packet times (`go_t` and the marks) and the recorded window are the same either way.

The page's clock is the packet time, not the computer's, so the marks line up with the signal in replays too. `backend/server.py` and `web/signal.js` don't depend on `record.py`, so the dashboard can reuse them. The server sends JSON over `/ws`:

```json
{"type":"source","mode":"LIVE"}
{"type":"config","t_on":7.0,"t_off":5.5,"pre_s":1.5,"post_s":8.5,"calibrate_s":10.0}
{"type":"frame","t":1792217330.1,"energy":1.2,"level":"no motion","row":[...166 amplitudes...],"rssi":-47,"hz":79.4}
{"type":"record","phase":"position","t":...,"clip":1,"n":10,"label":"fall","until":...}
```

`frame` comes 20 times a second: the rows of every packet in a 50 ms slot, averaged. `record` phases are `calibrate_queued`, `calibrating`, `calibrated` (`links`), `position`, `count` (`word`), `go`, `saved` (`file`, `seen`, `problems`, `go_t`) and `done` (`clean`, `flagged`, `stopped`). `failed` (`error`) means the page itself broke while recording carried on. A page that connects late first gets the latest `source` and `config`, every `record` so far, and the last 60 s of `frame`s. A `calibrate.py` session sends `n` 0.

The Recalibrate button sends `POST /calibrate` with `Content-Type: application/json`. The server answers 202 when the session queued it, 409 when no session can take it (calibrating, the last clip already counting, done, or none registered), 415 without the JSON content type, so a plain form on another site can't trigger it, and 403 unless the `Host` header is `127.0.0.1:<port>` or `localhost:<port>`, so a DNS-rebinding page can't either. The server only listens on 127.0.0.1. It calls whatever handler the session put in `hub.actions["calibrate"]`, so it still knows nothing about `record.py`.

`record.py` waits for the first packet before it starts, so a slow serial start no longer eats clip 1's get-in-position time. If the packet stream ends (a replay running out), it stops with a message instead of hanging. Stop releases both receiver ports before another batch can start. Packets of an unexpected size are left out of the page once a calibration is loaded, so one stray packet can't reset the charts. If the page itself breaks, the terminal says so once, the page shows `VIEW STOPPED`, and the recording carries on.

### The dashboard

`backend.run` serves `web/dashboard.html` at `/dashboard` (the record page stays at `/`) and opens it. Words in capitals are defined in [GLOSSARY.md](GLOSSARY.md). A toggle in the header switches between three views (`/dashboard?view=resident` or `?view=dev` opens one directly), and the header holds **Simulate fall**, which plays a real fall Recording (`--sim-fall`, default `backend/assets/sim_fall.log.gz`, a copy of `fall_20261017-021504` kept in git so it works without pulling data from R2) through its own trackers into the same Fall Flow, so the detector, the Stillness Window and the Countdown all run for real without hardware.

**Caregiver view** (the default): liquid-glass cards ([liquid-glass-react](https://github.com/rdev/liquid-glass-react), MIT, rendered as small React islands by `web/glass.js`; React 19 and the library are vendored in `web/vendor/` and loaded with an import map, so there is still no build step and it works offline; the edge refraction shows in Chrome, Safari and Firefox get the plain glass) in a pixel-and-CAD theme: DotGothic16 for headings and DM Sans for text (both OFL, vendored in `web/vendor/fonts/`), shades of green on a faint drafting grid, and `web/gator_side.png`, a side render of the CAD gator, as the logo. The cards sit a slowly turning field of colored dots that moves aside under the cursor, line icons and no emoji. The centrepiece is the 3D gator enclosure itself (`hardware/enclosure/web/gator_case.glb`, served at `/models/` and shown with the vendored `web/vendor/model-viewer.min.js`, so it works offline): it spins slowly when all is well, faster while a fall is checked or counted down, and stops facing you when the Caregiver is alerted, on a stage that glows green, amber or red. Its upper jaw (a separate part in the file) chomps now and then when all is well, chomps quickly while a fall is checked or counted down, and gapes wide when the Caregiver is alerted; it turns around the back of the jaw, where the printed jaw sits on its pegs. model-viewer has no public API for moving one part, so this uses its internal scene, pinned by the vendored 4.0.0 build. You can drag it to turn it. If WebGL or the model is unavailable, a drawn gator stands in. A Simulated Fall looks like a real one there; only the Dev view tags it. One big card says what is going on in plain words ("All is well", "Possible fall", "Rose may have fallen" with a countdown ring, "Rose needs help") with the buttons that apply right then: **Cancel the alert**, **I've handled it**, **It was a false alarm**, and **Call them** when `RESIDENT_PHONE` is set. Beside it, **Movement in the room** (motion energy averaged over the Links) and **Recent activity** (falls, Check-ins with their delivery status, answers and Alerts, across runs). Set `RESIDENT_NAME` in `.env` to use the Resident's name instead of "they".

**Resident view**, a fallback for when the texts can't get through, with the same glass, dots and 3D gator: "All good" until a Confirmed Fall (it says nothing while a fall is only suspected, so it never coaches anyone to stay down), then "Are you OK?" with the seconds left, **I'm OK** and **I need help**. After the Alert it says "Your caregiver has been told" and **I'm OK** still works as a Late Answer.

**Dev view**, everything for the team and the judges' technical questions:

- A Fall Flow badge (`ALL QUIET`, `FALL SUSPECTED`, `FALL DETECTED`, `CAREGIVER ALERTED`, with a `SIMULATED` tag) and a Motion Level badge per Link, with the Caregiver's buttons.
- A top-down **room map**: the sender at the point of the V, a line to each receiver. A line turns blue while its Link is moving, amber while a fall is suspected on it, and red for the Link(s) of the Confirmed Fall.
- **Why it's a fall**: the last fall's `peak` and Room Change against `T_FALL` and `T_STATIC`. Before any fall it shows why the last Motion Event was not one.
- The waveform (Auto follows the Link that moved last), motion energy for each Link with Motion Events shaded, and a log with **Live** and **History** tabs.

Every view sounds the Alarm during the Countdown, with a mute toggle. Browsers only allow sound after a click on the page, so the toggle says so until then.

When a replay of a Recording ends, its last second is repeated for 5.5 s so the Stillness Window can finish (Recordings stop 8.5 s after GO; see [docs/adr/0002](docs/adr/0002-pad-recording-replays.md)), and the dashboard stays up until Ctrl-C.

Everything in History is appended to `data/history.jsonl`, tagged `live`, `replay` or `sim`. Phone numbers and the Resident's name go in a `.env` file at the repo root (`RESIDENT_PHONE=...`, `CAREGIVER_PHONE=...`, `RESIDENT_NAME=...`). Both files are gitignored.

#### Messages and commands

On top of the record page's messages, `run.py` sends `frame`s with a `"link"` field (one feed per Link) and:

```json
{"type":"state","state":"COUNTDOWN","t":...,"deadline":...,"now":...,"sim":false,"fall":7}
{"type":"event","t":...,"label":"fall","confidence":1.0,"links":"A","details":{"peak":24.97,"static_change":0.58,"duration_s":3.09},"why":"classified","start":...,"end":...,"sim":false}
{"type":"log","t":...,"text":"Confirmed Fall: Countdown started"}
{"type":"alarm","on":true}
{"type":"notify","t":...,"id":12,"to":"resident","kind":"check_in","fall":7,"text":"Fall detected. Are you OK? Reply 1 = I'm OK, 2 = I need help.","status":"sending"}
{"type":"delivery","t":...,"id":12,"fall":7,"status":"sent"}
{"type":"history","t":...,"what":"late_answer","mode":"live","fall":7,"late_s":30}
```

`why` is `classified`, `moved_again` (moved during the Stillness Window) or `cut_short` (kept moving for 8 s). RF `event.details` includes all six ordered feature values, `class_probabilities`, and `fall_probability`; confidence is the selected class's model score. These are uncalibrated model scores, not real-world risk estimates. The Dev view shows the saved decision rule and scores rather than applying the heuristic's thresholds to an RF verdict.

The server also sends a sticky `detector` snapshot (`name`, `ready`, `decision_rule`, `fall_threshold`, `trained_at`, `model_path`) and a separate sticky `sensor` snapshot for each receiver every 0.5 s. Sensor snapshots include `health` (`waiting`, `calibrating`, `ready`, `stale`, `invalid`, `ended`), packet counts, packet age, energy, RSSI, rate, and calibration progress/channel counts/baseline. Live startup allows three seconds to step away before collecting ten seconds of still-room calibration. **Dev → Recalibrate sensors** repeats this between events without restarting; its control is disabled during an active fall, simulation, or calibration. `calibration.phase` is `settling`, `learning`, or `ready`, with `settle_remaining_s` during the step-away interval. Movement during calibration raises the denominator used for motion energy and can make subsequent falls appear too weak; recalibrate after changing the room or moving the boards.

After two seconds without usable live frames, the pipeline discards affected unfinished candidates and requires ten seconds of fresh calibration when that receiver returns. Signal loss never means stillness, and a confirmed countdown continues. Replays preserve their recorded calibration. Simulation frames carry `sim: true` and never mark real sensors connected. The bundled early example may receive a non-fall RF verdict; `--sim-fall <local-recording.log.gz>` chooses another demonstration input, using the same forest and decision rule.

`notify` `kind` is `check_in` (to the Resident), `alert`, `all_clear` or `help` (to the Caregiver). `delivery` `status` is `sent`, `failed`, or `not_delivered` when nothing reported back within 10 s. `state` is re-sent every 0.5 s during a Countdown, and a page that connects late gets the latest `state`, `alarm`, `detector`, and per-link `sensor`, plus every `event`, `log`, `notify`, `delivery` and `history` so far.

Pages and the messaging sidecar send commands back over the same `/ws`:

| Command | From | Does |
|---|---|---|
| `{"cmd":"ok","fall":7}` | Resident | During the Countdown, cancels it. After the Alert, a Late Answer: ends the Alert and sends the Caregiver an All-clear. |
| `{"cmd":"help","fall":7}` | Resident | A Help Request: forwarded to the Caregiver at once (the Countdown keeps running until the Fall Flow gets an `escalate()`). |
| `{"cmd":"cancel"}` | Caregiver | Cancels the Countdown. |
| `{"cmd":"resolve","reason":"handled"\|"false_alarm"}` | Caregiver | Closes an Alert. |
| `{"cmd":"call"}` | Caregiver | Logs that the Caregiver called the Resident. |
| `{"cmd":"delivered","id":12,"status":"sent"\|"failed"}` | sidecar | Reports what happened to a `notify`. |
| `{"cmd":"simulate"}` | demo | Plays a Simulated Fall. |

`fall` is optional on `ok` and `help`; without it the answer counts for the fall that is open now. An answer while no fall is open, or for another fall, is a Stale Answer and changes nothing.

#### The messaging sidecar

A starter is in [`messaging/`](messaging/README.md): `node messaging/sidecar.mjs --console` runs the whole loop in a terminal (texts print there, type `1` or `2` to answer as the Resident), and only `sendText()` and `listenForReplies()` are left to connect to Photon. After replaying recent messages to a new connection, the server sends `{"type":"live"}`, so the sidecar never re-sends an old Check-in. Texts go out through [Photon](https://photon.codes/) (iMessage), from a small Node process that connects to `ws://127.0.0.1:8000/ws` like a browser does ([docs/adr/0001](docs/adr/0001-messaging-sidecar-over-ws.md)). It sends each `notify` whose `status` is `sending` to the Resident's or Caregiver's number by `to`, answers with `delivered`, and turns the Resident's replies into commands: `1`, `ok`, `okay`, `yes`, `fine` or 👍 become `ok`, `2` or `help` become `help`, and anything else is forwarded to the Caregiver. Use three separate accounts: the demo laptop sends, and the Resident and the Caregiver each have their own phone. A reply from the sending account itself may be read as its own message.

## Layout

```
backend/
  config.py            all tunable numbers and the data folder location
  run.py               terminal front end for the pipeline (and the dashboard with --view)
  server.py            live page server: WebSocket hub, 20 Hz signal feed, static files
  sync.py              copy data/ to and from the R2 bucket
  dashboard.py         dashboard glue: WebSink, commands, history, Simulate fall
  capture/             getting CSI from the board into Frames
    csi.py             parse CSI lines, read logs
    live.py            read the receiver over USB, save sessions
    replay.py          play saved logs back with original timing
    meter.py           console capture check
  detection/           deciding what happened
    motion.py          motion energy, calibration, Motion Event cutting
    features.py        numbers describing a Motion Event
    detector.py        threshold rules and random forest
    fallflow.py        candidate → countdown → alert state machine
    pipeline.py        wires motion, detector and fall flow together
  training/            collecting labeled data and training the model
    dashboard.py       persistent recording dashboard and batch controls
    recordings.py      save/load labeled Recordings
    record.py          record labeled clips
    calibrate.py       learn the empty room again
    train.py           train and evaluate the random forest
web/
  record.html          live record page (timing cues + signal)
  dashboard.html       live dashboard: Caregiver, Resident and Dev views
  dashboard.css        dashboard layout, on top of style.css
  signal.js            waveform, heatmap, energy and CSI-shape charts, reusable for the dashboard
  style.css            colors (light and dark) and layout
messaging/
  sidecar.mjs          texts Check-ins and Alerts, turns replies into ok/help (Photon goes here)
arduino/
  cancel_button/       cancel button + buzzer sketch
data/                  not in git; kept in R2 with sync.py
  recordings/          labeled clips (.log.gz + .json metadata), train/ and test/
  calibration.json     empty-room baseline used by record.py, written by record.py and calibrate.py
  results.json         latest training scores
  sessions/            raw logs of every live run (local only, not synced)
  model.joblib         trained model
```

## Status

Working: capture, replay, motion detection, both detectors, the fall flow, recording (with the live record page), training and the live dashboard. Not built yet: the Photon messaging sidecar (the dashboard already publishes what it should send and accepts its replies) and the Python side of the Arduino link.

Both receivers are read together, with motion, calibration and Motion Events per link, in the terminal pipeline, the capture check, replay and `record.py`. `train.py` takes features from the strongest link in each two-link clip, the same way the live pipeline does. The dashboard shows both links; the record page still shows one. The detector was tuned and trained on link A clips only, so it hasn't been checked on falls seen by link B yet.
