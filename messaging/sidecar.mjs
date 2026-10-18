// Messaging sidecar: texts the Check-ins, Alerts and All-clears the dashboard asks for, and turns
// the Resident's replies into the dashboard's ok / help commands (docs/adr/0001).
//
//   node messaging/sidecar.mjs --console      try it without Photon: texts print here, type 1 or 2 to reply
//   node messaging/sidecar.mjs                real texts, once sendText() and listenForReplies() use Photon
//
// Options: --url ws://127.0.0.1:8000/ws (match run.py's --view-port). Phone numbers come from .env:
// RESIDENT_PHONE and CAREGIVER_PHONE.
import { existsSync, readFileSync } from "node:fs";
import readline from "node:readline";

const args = process.argv.slice(2);
const CONSOLE = args.includes("--console");
const URL_ARG = args.indexOf("--url");
const WS_URL = URL_ARG >= 0 ? args[URL_ARG + 1] : "ws://127.0.0.1:8000/ws";
const RETRY_MS = 2000;

loadEnv(new URL("../.env", import.meta.url));
const PHONES = { resident: process.env.RESIDENT_PHONE || "", caregiver: process.env.CAREGIVER_PHONE || "" };

// ---------------------------------------------------------------------------------------------
// The two Photon pieces. Everything else in this file is the dashboard side and already works.

// Send one text. Throw if it couldn't be sent; the dashboard then shows "couldn't send".
async function sendText(phone, text) {
  if (CONSOLE) {
    console.log(`\n  [text to ${phone || "(no number)"}] ${text}\n`);
    return;
  }
  throw new Error("sendText is not connected to Photon yet (see messaging/README.md)");
}

// Call onReply(phone, text) for every message that arrives. Runs once at startup.
async function listenForReplies(onReply) {
  if (CONSOLE) {
    const rl = readline.createInterface({ input: process.stdin });
    console.log("Console mode: type the Resident's reply (1 = I'm OK, 2 = I need help) and press Enter.");
    rl.on("line", (line) => onReply(PHONES.resident, line));
    return;
  }
  console.log("listenForReplies is not connected to Photon yet; replies won't reach the dashboard.");
}
// ---------------------------------------------------------------------------------------------

const OK_WORDS = /^\s*(1|ok|okay|yes|yep|fine|i'?m ok|im ok|i am ok|👍)\b/i;
const HELP_WORDS = /^\s*(2|help)\b/i;
const digits = (phone) => (phone || "").replace(/\D/g, "").slice(-10);

let ws = null;
let live = false;

function connect() {
  ws = new WebSocket(WS_URL);
  live = false;
  let skipped = 0;
  ws.onopen = () => console.log(`Connected to ${WS_URL}`);
  ws.onmessage = (event) => {
    let msg;
    try {
      msg = JSON.parse(event.data);
    } catch {
      return;
    }
    if (msg.type === "live") {
      live = true;
      if (skipped) console.log(`Skipped ${skipped} message(s) from before this connection.`);
      return;
    }
    if (msg.type !== "notify" || msg.status !== "sending") return;
    if (!live) {
      skipped += 1; // replayed from before we connected: already handled or timed out
      return;
    }
    deliver(msg);
  };
  ws.onclose = () => {
    console.log(`Dashboard not reachable at ${WS_URL}; retrying...`);
    setTimeout(connect, RETRY_MS);
  };
  ws.onerror = () => {};
}

async function deliver(msg) {
  const phone = PHONES[msg.to];
  let status = "sent";
  if (!phone && !CONSOLE) {
    console.log(`No ${msg.to === "resident" ? "RESIDENT_PHONE" : "CAREGIVER_PHONE"} in .env; can't send ${msg.kind}.`);
    status = "failed";
  } else {
    try {
      await sendText(phone, msg.text);
      console.log(`Sent ${msg.kind} #${msg.id} to the ${msg.to}.`);
    } catch (e) {
      console.log(`Couldn't send ${msg.kind} #${msg.id}: ${e.message}`);
      status = "failed";
    }
  }
  command({ cmd: "delivered", id: msg.id, status });
}

function command(msg) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(msg));
  else console.log("Not connected to the dashboard; dropped", msg);
}

async function onReply(phone, text) {
  if (!CONSOLE && digits(phone) !== digits(PHONES.resident)) return;
  if (OK_WORDS.test(text)) {
    console.log("Resident says OK.");
    command({ cmd: "ok" });
  } else if (HELP_WORDS.test(text)) {
    console.log("Resident asked for help.");
    command({ cmd: "help" });
  } else if (text.trim()) {
    console.log("Passing the Resident's reply to the Caregiver.");
    try {
      await sendText(PHONES.caregiver, `The Resident replied: "${text.trim()}"`);
    } catch (e) {
      console.log(`Couldn't pass it on: ${e.message}`);
    }
  }
}

function loadEnv(url) {
  if (!existsSync(url)) return;
  for (const line of readFileSync(url, "utf8").split("\n")) {
    const m = line.match(/^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$/);
    if (m && !(m[1] in process.env)) process.env[m[1]] = m[2].replace(/^['"]|['"]$/g, "");
  }
}

connect();
listenForReplies(onReply);
