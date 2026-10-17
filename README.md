# Camera-Free Fall Detection

SwampHacks XII project. Detects falls in a room from how a person's movement disturbs Wi-Fi signals, then warns a caregiver. No camera, nothing to wear.

## How it works

Two Seeed XIAO ESP32-S3 boards sit 3–4 m apart. One broadcasts Wi-Fi packets about 100 times a second. The other measures the **Channel State Information (CSI)** of each packet: how strongly each of its frequency subcarriers arrived (about 166 usable ones). A human body absorbs and reflects 2.4 GHz Wi-Fi, so when someone moves between the boards, those numbers change.

The receiver streams CSI to a laptop over USB, where the Python backend:

1. **Parses** each `CSI_DATA` line into an amplitude per subcarrier (`capture/csi.py`).
2. **Measures motion energy**: how much the average signal shifted over the last 0.5 s, divided by an empty-room baseline learned during a 10 s calibration (`detection/motion.py`).
3. **Cuts Motion Events**: a burst of movement from when energy crosses `T_ON` until it stays below `T_OFF` for 1 s.
4. **Waits through a 4 s Stillness Window.** If the person moves again, it wasn't a fall.
5. **Classifies** the event as walk, sit-down, or fall (`detection/features.py`, `detection/detector.py`). The key signals are how hard the signal moved (`peak`) and how different the room looks before vs. after (`static_change`): a body lying on the floor changes the room far more than one sitting in a chair.
6. **Runs the fall flow** (`detection/fallflow.py`): a confirmed fall starts a 10 s countdown with an alarm. If nobody cancels, the caregiver is alerted.

```
sender XIAO ──Wi-Fi──► receiver XIAO ──USB──► csi.py ► motion.py ► features.py ► detector.py ► fallflow.py
                                                                                              │
                                                                         alarm / cancel button / alert
```

Two detectors share the same interface:

- **ThresholdDetector**: hand-tuned rules from `config.py`. The default and the demo fallback.
- **RandomForestDetector**: scikit-learn model trained on labeled Recordings by `train.py`.

## Hardware

| Item | Role |
|---|---|
| XIAO ESP32-S3 + external antenna | Sender, running Espressif's [esp-csi](https://github.com/espressif/esp-csi) `csi_send` example |
| XIAO ESP32-S3 + external antenna | Receiver, running `csi_recv`, plugged into the laptop |
| Arduino + push button + buzzer | In-room cancel button and alarm (`arduino/cancel_button`) |

Place the boards 3–4 m apart at waist height so the direct path crosses where a person would land.

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
idf.py -p /dev/cu.usbmodem<RECEIVER> flash
```

In `menuconfig`, set **Component config → ESP System Settings → Channel for console output → USB Serial/JTAG Controller**. The XIAO's USB-C port is the chip's built-in USB, not a UART, so without this no CSI reaches the laptop.

## Running

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Run everything from the repo root with `python -m`. Each command auto-detects the receiver board, or takes `--port`.

| Command | What it does |
|---|---|
| `python -m backend.capture.live` | Check capture: prints packet rate, subcarriers, RSSI and a motion bar. Saves the raw stream to `data/sessions/`. |
| `python -m backend.run` | Full detection pipeline live in the terminal. Stay still for the first 10 s while it calibrates. |
| `python -m backend.run --replay data/sessions/<file>.log --speed 0` | Same, on a saved session (`--speed 0` = as fast as possible). |
| `python -m backend.capture.replay <log>` | Play a session log or Recording through the capture meter. |
| `python -m backend.training.record fall --n 10` | Record 10 labeled clips: wait for 3-2-1-GO, move, hold still until "saved". Labels: `fall`, `sit-down`, `walk`, `empty`. Add `--split test` for the test set, `--recalibrate` after moving the boards. |
| `python -m backend.training.train` | Train the random forest on `data/recordings/train/`, score it on `data/recordings/test/` (or cross-validation if there's no test set), write `data/model.joblib` and `data/results.json`. |
| `python -m backend.run --detector rf` | Use the trained model instead of the threshold rules. |

All tunable numbers (thresholds, timings, ntfy topic) are in `backend/config.py`.

## Layout

```
backend/
  config.py            all tunable numbers and the data folder location
  run.py               terminal front end for the pipeline
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
    recordings.py      save/load labeled Recordings
    record.py          record labeled clips
    train.py           train and evaluate the random forest
arduino/
  cancel_button/       cancel button + buzzer sketch
data/
  recordings/          labeled clips (.log.gz + .json metadata), train/ and test/
  calibration.json     empty-room baseline used by record.py
  results.json         latest training scores
  sessions/            raw logs of every live run (not in git)
  model.joblib         trained model (not in git, rebuild with train.py)
```

## Status

Working: capture, replay, motion detection, both detectors, the fall flow, recording and training. Not built yet: the web dashboard, ntfy phone alerts, and the Python side of the Arduino link. The pipeline currently prints the alarm and alerts to the console.
