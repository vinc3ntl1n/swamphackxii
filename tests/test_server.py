import unittest

from fastapi.testclient import TestClient

from backend import server


class ServerTests(unittest.TestCase):
    def test_websocket_forwards_commands_to_dashboard(self):
        hub = server.Hub()
        received = []

        def on_command(command):
            received.append(command)
            hub.publish({"type": "command_received", "command": command})

        command = {"type": "cancel"}
        with TestClient(server.make_app(hub, on_command)) as client:
            with client.websocket_connect("/ws") as websocket:
                self.assertEqual(websocket.receive_json(), {"type": "live"})
                websocket.send_json(command)
                self.assertEqual(websocket.receive_json(), {
                    "type": "command_received", "command": command,
                })
        self.assertEqual(received, [command])

    def test_training_websocket_works_without_command_handler(self):
        with TestClient(server.make_app(server.Hub())) as client:
            with client.websocket_connect("/ws") as websocket:
                self.assertEqual(websocket.receive_json(), {"type": "live"})
                websocket.send_json({"type": "cancel"})


if __name__ == "__main__":
    unittest.main()
