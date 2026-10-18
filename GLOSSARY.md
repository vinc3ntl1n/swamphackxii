# Camera-Free Fall Detection

Detects falls in a room from how a person's movement disturbs Wi-Fi signals between three boards, then warns a Caregiver.

## People

**Resident**:
The person living in the monitored room, whose falls the system detects.
_Avoid_: patient, user, elder

**Caregiver**:
The person who is warned when the Resident has fallen and does not answer.
_Avoid_: family, contact, guardian

## Sensing

**Link**:
The Wi-Fi path from the sender to one receiver (A or B). The two Links fan out from the sender in a V.
_Avoid_: channel, board, sensor

**Calibration**:
The empty-room baseline of one Link, learned while nobody moves.
_Avoid_: baseline (as a noun for the whole thing), training

**Motion Level**:
How much one Link's signal is moving right now: calibrating, no motion, or moving. Each Link has its own.
_Avoid_: activity, state

**Motion Event**:
One burst of movement, from when motion starts on any Link until every Link is quiet again.
_Avoid_: event (alone), movement, clip

**Room Change**:
How different the room's signal looks after a Motion Event compared with before it; a body on the floor changes it far more than one in a chair.
_Avoid_: static change (in speech), delta

## Fall Flow

**Fall Flow**:
The room-wide sequence a possible fall goes through: idle, Fall Candidate, Countdown, Alerted. There is one per room, unlike Motion Level.
_Avoid_: state machine, alarm state

**Stillness Window**:
The few seconds after a Motion Event during which any new movement means it was not a fall.
_Avoid_: wait, timeout

**Fall Candidate**:
A Motion Event that has ended and is waiting out its Stillness Window before it is classified.
_Avoid_: possible fall, pending fall

**Confirmed Fall**:
A Fall Candidate that stayed still and was classified as a fall; it starts the Countdown.
_Avoid_: detected fall, positive

**Countdown**:
The seconds after a Confirmed Fall during which the Resident can say they are OK before the Caregiver gets an Alert.
_Avoid_: timer, grace period

**Check-in**:
The "Are you OK?" prompt sent to the Resident when the Countdown starts.
_Avoid_: alert, ask, notification

**Alert**:
The warning sent to the Caregiver when a Countdown runs out without the Resident cancelling it.
_Avoid_: alarm, notification, check-in

**Help Request**:
The Resident saying they need help in answer to a Check-in; it ends the Countdown and sends the Alert at once.
_Avoid_: SOS, panic, emergency

**Late Answer**:
The Resident saying they are OK after the Alert has already gone out; it ends the Alert.
_Avoid_: late cancel, undo

**Stale Answer**:
A Resident's "I'm OK" that belongs to an earlier Confirmed Fall than the current one; it changes nothing.
_Avoid_: old answer, duplicate

**All-clear**:
The follow-up sent to the Caregiver after a Late Answer, saying the Resident is OK.
_Avoid_: cancellation, stand-down, second alert

**Resolved Alert**:
An Alert the Caregiver has closed, with a reason: false alarm or handled.
_Avoid_: reset, dismissed, cleared

**Alarm**:
The sound in the room during the Countdown.
_Avoid_: alert, siren

**Simulated Fall**:
A recorded real fall played through the whole detection path on demand, standing in for a live fall during a demo.
_Avoid_: fake fall, test fall, mock

## Data

**Recording**:
A labeled clip of a few seconds around one performed action (fall, sit-down, walk, empty), saved for training.
_Avoid_: sample, clip file

**Session**:
The raw log of everything the receivers sent during one live run.
_Avoid_: recording, capture
