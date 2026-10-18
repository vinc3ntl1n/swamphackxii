// Read-only setup checks. No texts are sent and no message contents are printed.
import { existsSync, readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { messagingErrorHint, validateMessagesSchema } from "./imessage-compat.mjs";

export async function checkMessagesAccess(IMessageSDK) {
  const sdk = new IMessageSDK();
  try {
    validateMessagesSchema(sdk);
  } finally {
    await sdk.close();
  }
}

function dashboardAvailable(url) {
  return new Promise((resolve) => {
    const ws = new WebSocket(url);
    const finish = (ready) => {
      clearTimeout(timer);
      ws.close();
      resolve(ready);
    };
    const timer = setTimeout(() => finish(false), 3000);
    ws.onmessage = ({ data }) => {
      try {
        if (JSON.parse(data).type === "live") finish(true);
      } catch {}
    };
    ws.onerror = () => finish(false);
    ws.onclose = () => { clearTimeout(timer); resolve(false); };
  });
}

async function main() {
  let failures = 0;
  const fail = (message) => { failures += 1; console.log(`FAIL: ${message}`); };
  console.log("Messaging setup check — no texts will be sent.\n");
  if (Number(process.versions.node.split(".")[0]) < 22 || typeof WebSocket !== "function") {
    fail("Node 22 or newer with built-in WebSocket is required.");
    process.exitCode = 1;
    return;
  }
  console.log(`OK: Node ${process.versions.node}`);

  const envFile = new URL("../.env", import.meta.url);
  if (existsSync(envFile)) {
    for (const line of readFileSync(envFile, "utf8").split("\n")) {
      const match = line.match(/^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$/);
      if (match && !(match[1] in process.env)) {
        process.env[match[1]] = match[2].replace(/^['"]|['"]$/g, "");
      }
    }
    console.log("OK: Found .env at the repository root.");
  } else {
    console.log("NOTE: No .env file; checking exported environment variables.");
  }
  for (const key of ["RESIDENT_PHONE", "CAREGIVER_PHONE"]) {
    if (/^\+[1-9]\d{7,14}$/.test(process.env[key] || "")) {
      console.log(`OK: ${key} is set and has an international phone-number format.`);
    } else {
      fail(`Set ${key} in .env to a real phone number in +1… form.`);
    }
  }
  if (process.env.RESIDENT_PHONE && process.env.RESIDENT_PHONE === process.env.CAREGIVER_PHONE) {
    console.log("NOTE: Resident and caregiver currently use the same number.");
  }

  const provider = process.env.PHOTON_PROVIDER || "imessage-local";
  try {
    await import("@spectrum-ts/core");
    if (provider === "imessage-local") {
      await import("@spectrum-ts/imessage-local");
      console.log("OK: Spectrum and the local iMessage provider are installed.");
      if (process.platform !== "darwin") {
        fail("Local iMessage requires macOS.");
      } else {
        const { IMessageSDK } = await import("@photon-ai/imessage-kit");
        try {
          await checkMessagesAccess(IMessageSDK);
          console.log("OK: This terminal can open and query the Messages database.");
        } catch (error) {
          fail(`Messages database: ${error.message}`);
          const hint = messagingErrorHint(error);
          if (hint) console.log(`  ${hint}`);
        }
      }
    } else if (provider === "cloud") {
      await import("@spectrum-ts/imessage");
      for (const key of ["PHOTON_PROJECT_ID", "PHOTON_PROJECT_SECRET"]) {
        if (!process.env[key]) fail(`Set ${key} for the cloud provider.`);
      }
      console.log("NOTE: Cloud credentials and delivery are not tested by this check.");
    } else {
      fail(`Unknown PHOTON_PROVIDER: ${provider}`);
    }
  } catch (error) {
    fail(`Messaging dependencies: ${error.message}`);
    console.log(`  Run from the repository root: ${provider === "cloud" ? "npm install --prefix messaging @spectrum-ts/imessage" : "npm ci --prefix messaging"}`);
  }

  const urlIndex = process.argv.indexOf("--url");
  const url = urlIndex < 0 ? "ws://127.0.0.1:8000/ws" : process.argv[urlIndex + 1];
  try {
    if (await dashboardAvailable(url)) console.log(`OK: Dashboard is reachable at ${url}`);
    else console.log("NOTE: Dashboard is not reachable yet. Start the Python backend in another terminal.");
  } catch (error) {
    fail(`Dashboard URL: ${error.message}`);
  }
  console.log(failures ? "\nFix the FAIL items above, then run this check again." : "\nSetup checks passed. Start node messaging/sidecar.mjs; message delivery still needs a live test.");
  process.exitCode = failures ? 1 : 0;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  await main();
}
