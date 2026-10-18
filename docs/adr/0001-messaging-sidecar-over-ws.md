# Messaging runs in a Node sidecar that speaks the dashboard's /ws protocol

Check-ins, Alerts and All-clears reach people through Photon (iMessage), whose SDK is TypeScript-only while the backend is Python. Rather than embed Node in Python or open a webhook port, the sender is a separate Node process that connects to `/ws` like any browser: it sends texts when it sees `notify` messages, reports `delivered` back, and turns the Resident's replies into the same `ok` / `help` commands the Resident view sends. Nothing is exposed beyond localhost, and the Fall Flow never knows which channel answered.

## Considered Options

- ntfy push with an action button: rejected for Photon, which reaches the Resident as a normal text.
- Opening the server to the LAN or a tunnel for a webhook: rejected, since hackathon Wi-Fi often isolates devices and it would expose the demo controls.
