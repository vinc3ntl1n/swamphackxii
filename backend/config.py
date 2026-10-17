from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SERIAL_PORT = None
ARDUINO_PORT = None

CALIBRATE_S = 10.0
WINDOW_S = 0.5
T_ON = 7.0
T_OFF = 5.5
QUIET_S = 1.0
PRE_EVENT_S = 1.5
MAX_EVENT_S = 8.0

DETECTOR = "threshold"

STILLNESS_S = 4.0
AFTER_S = 1.5
BEFORE_GAP_S = 0.5

T_FALL = 14.0
T_STATIC = 0.30
SIT_MAX_S = 3.5

COUNTDOWN_S = 10.0

NTFY_SERVER = "https://ntfy.sh"
NTFY_PREFIX = "swampfall-ef86646d"
