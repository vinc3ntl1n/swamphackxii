"""Receiver health and downstream countdowns use real capture state."""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from backend import dashboard, server
from backend.capture.csi import Frame
from backend.detection.detector import ThresholdDetector, Verdict
from backend.detection.fallflow import COUNTDOWN, FALL_CANDIDATE, IDLE
from backend.detection.motion import MEASURE, MotionEvent
from backend.detection.pipeline import Pipeline


class LiveDashboardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.hub = server.Hub()
        history = dashboard.History(self.hub, 'live', Path(self.temp.name) / 'history.jsonl')
        self.web = dashboard.WebSink(self.hub, history, dashboard.Notices(self.hub, history))
        self.pipe = Pipeline(self.web, ThresholdDetector())
        self.view = dashboard.Dashboard(self.hub, self.pipe, self.web, self.web)
        dashboard.preload(self.pipe, {'A': {
            'measure': MEASURE, 'size': 4, 'mask': [0, 1, 2, 3], 'baseline': .01,
        }})
        self.wall = 100.0
        self.time = patch('backend.dashboard.time.monotonic', side_effect=lambda: self.wall)
        self.time.start()
        self.addCleanup(self.time.stop)

    def sensor(self, link='A'):
        return json.loads(self.hub.sticky[f'sensor:{link}'][1])

    def frame(self, t=10.0, width=4):
        return Frame(t=t, rssi=-48, amp=np.full(width, 20.), link='A')

    def candidate(self):
        event = MotionEvent(9., 10., np.array([9., 10.]), np.array([8., 2.]),
                            np.full((2, 4), 20.), False, 'A')
        self.pipe.pending, self.pipe.links, self.pipe.deadline = event, 'A', 14.
        self.pipe.flow.candidate(10., 14.)

    def test_health_is_per_receiver_and_survives_reconnect(self):
        self.view.sensors.refresh(10.)
        self.assertEqual(self.sensor('A')['health'], 'waiting')
        self.assertEqual(self.sensor('B')['health'], 'waiting')
        self.view.process(self.frame())
        self.view.sensors.refresh(10., force=True)
        status = self.sensor()
        self.assertEqual(status['health'], 'ready')
        self.assertEqual(status['rssi'], -48)
        self.assertEqual(status['calibration']['usable_channels'], 4)
        self.assertEqual(status['packets_received'], 1)
        self.wall += 3.
        self.view._tick()
        self.assertEqual(self.sensor()['health'], 'stale')

        async def snapshot():
            queue, old = self.hub.join()
            self.hub.leave(queue)
            return [json.loads(msg) for msg in old]
        sensors = [msg for msg in asyncio.run(snapshot()) if msg['type'] == 'sensor']
        self.assertEqual(len(sensors), 2)
        self.assertEqual({msg['link']: msg['health'] for msg in sensors},
                         {'A': 'stale', 'B': 'waiting'})

    def test_missing_packets_discard_candidate_and_return_requires_calibration(self):
        self.view.process(self.frame())
        self.candidate()
        self.assertEqual(self.pipe.flow.state, FALL_CANDIDATE)
        self.wall += 3.
        self.view._tick()
        self.assertIsNone(self.pipe.pending)
        self.assertEqual(self.pipe.flow.state, IDLE)
        self.assertIsNone(self.pipe.trackers['A'].baseline)
        self.view.process(self.frame(t=13.))
        self.view.sensors.refresh(13., force=True)
        self.assertEqual(self.sensor()['health'], 'calibrating')
        self.assertEqual(self.sensor()['calibration']['progress'], 0.)

    def test_wrong_channel_width_does_not_refresh_valid_signal(self):
        self.view.process(self.frame())
        self.candidate()
        self.wall += 1.
        self.view.process(self.frame(11., width=6))
        self.wall += 1.1
        self.view.sensors.refresh(12.1, force=True)
        status = self.sensor()
        self.assertEqual(status['health'], 'invalid')
        self.assertEqual(status['invalid_packets'], 1)
        self.assertEqual(status['last_frame_t'], 10.)
        self.assertIsNone(self.pipe.pending)

    def test_confirmed_countdown_continues_without_packets_and_can_cancel(self):
        self.view.process(self.frame())
        self.candidate()
        self.pipe.flow.confirm(10.)
        self.assertEqual(self.pipe.flow.state, COUNTDOWN)
        self.wall += 3.
        self.view._tick()
        self.assertEqual(self.pipe.flow.state, COUNTDOWN)
        self.view.on_command({'cmd': 'ok', 'fall': self.web.fall})
        self.view._tick()
        self.assertEqual(self.pipe.flow.state, IDLE)
        self.assertEqual(self.web.fall, 1)

    def test_countdown_alerts_once_when_no_more_packets_arrive(self):
        self.view.process(self.frame())
        self.candidate()
        self.pipe.flow.confirm(10.)
        self.wall += 11.
        self.view._tick()
        self.view._tick()
        notices = [json.loads(msg) for _, msg in self.hub.kept
                   if json.loads(msg).get('type') == 'notify']
        self.assertEqual([msg['kind'] for msg in notices], ['check_in', 'alert'])

    def test_simulation_does_not_claim_real_receivers_are_connected(self):
        self.view.sensors.refresh(10.)
        self.view._command({'cmd': 'simulate'})
        self.assertIs(self.view.sim.pipe.detector, self.pipe.detector)
        frame = next(self.view.sim.due(self.view.clock.now() + 1))
        info, result = self.view.sim.pipe.process(frame)
        self.view._show(self.view.sim.pipe, frame, info, result, sim=True)
        self.view._flush()
        self.assertEqual(self.sensor()['health'], 'waiting')
        frames = [json.loads(msg) for _, _, msg in self.hub.backlog
                  if json.loads(msg).get('type') == 'frame']
        self.assertTrue(frames[-1]['sim'])

    def test_replay_stall_preserves_recorded_calibration(self):
        self.view.sensors.source = 'REPLAY'
        self.view.process(self.frame())
        self.wall += 3.
        self.view._tick()
        self.assertIsNotNone(self.pipe.trackers['A'].baseline)
        self.view.ended()
        self.assertEqual(self.sensor()['health'], 'ended')

    def test_web_verdict_keeps_selected_class_probability(self):
        self.candidate()
        probability = .7906843162
        verdict = Verdict('fall', probability, {'peak': 15.04, 'fall_probability': probability,
                          'class_probabilities': {'fall': probability, 'walk': 1 - probability}})
        self.web.event(14., self.pipe.pending, verdict, 'A', False)
        event = next(json.loads(msg) for _, msg in self.hub.kept if json.loads(msg)['type'] == 'event')
        self.assertEqual(event['confidence'], probability)
        self.assertEqual(event['confidence'], event['details']['class_probabilities']['fall'])


if __name__ == '__main__':
    unittest.main()
