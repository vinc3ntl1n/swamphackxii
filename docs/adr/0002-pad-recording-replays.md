# Replays of Recordings are padded with their last still second

Recordings stop 8.5 s after GO, which is often before a fall's 4 s Stillness Window ends, so replaying one never reached a verdict. When a replay of a Recording (a log with a saved calibration next to it) runs out, its last second of packets is repeated for 5.5 s. This is synthetic data on purpose: it models the person staying where they are. Session logs and live streams are never padded, because a stalled stream means missing data, not stillness.
