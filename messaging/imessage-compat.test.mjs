import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import Database from "better-sqlite3";
import { IMessageSDK } from "@photon-ai/imessage-kit";
import { checkMessagesAccess } from "./check.mjs";
import { installMessagesCompatibility, messagingErrorHint, validateMessagesSchema } from "./imessage-compat.mjs";

const macOnly = { skip: process.platform !== "darwin" };

async function fixture(t, missing = {}) {
  const directory = mkdtempSync(join(tmpdir(), "imessage-compat-"));
  const path = join(directory, "fixture.db");
  const writer = new Database(path);
  writer.exec("CREATE TABLE handle (id TEXT); CREATE TABLE chat_message_join (message_id INTEGER, chat_id INTEGER); CREATE TABLE message_attachment_join (message_id INTEGER, attachment_id INTEGER)");
  const sdk = new IMessageSDK({ databasePath: path });
  t.after(async () => { await sdk.close(); writer.close(); rmSync(directory, { recursive: true }); });
  const queries = sdk.database.queries;
  const columns = { message: new Set(), chat: new Set(), attachment: new Set() };
  for (const query of [queries.buildMessageQuery({ limit: 1 }), queries.buildChatQuery({}), queries.buildAttachmentQuery([1])]) {
    for (const [, table, column] of query.sql.matchAll(/\b(message|chat|attachment)\.([a-z_]\w*)/gi)) {
      if (column.toLowerCase() !== "rowid") columns[table].add(column);
    }
  }
  for (const [table, names] of Object.entries(columns)) {
    for (const column of missing[table] || []) names.delete(column);
    writer.exec(`CREATE TABLE ${table} (${[...names].map(name => `"${name}" BLOB`).join(", ")})`);
  }
  const insert = (table, row) => writer.prepare(`INSERT INTO ${table} (${Object.keys(row).map(name => `"${name}"`).join(", ")}) VALUES (${Object.keys(row).map(() => "?").join(", ")})`).run(...Object.values(row));
  insert("handle", { id: "+15555550101" });
  insert("chat", { guid: "iMessage;-;+15555550101", chat_identifier: "+15555550101", style: 45 });
  const message = { guid: "fixture-1", text: "1", date: 1_000_000_000, handle_id: 1, other_handle: null,
    is_from_me: 0, is_read: 0, item_type: 0, service: "iMessage", cache_has_attachments: 1 };
  if (columns.message.has("date_recovered")) message.date_recovered = 2_000_000_000;
  insert("message", message);
  insert("chat_message_join", { message_id: 1, chat_id: 1 });
  insert("attachment", { guid: "fixture-attachment", created_date: 1_000_000_000, filename: "/tmp/fixture.txt", hide_attachment: 0 });
  insert("message_attachment_join", { message_id: 1, attachment_id: 1 });
  return { sdk, writer, path };
}

test("older schema reproduces date_recovered failure, then reads replies and attachments", macOnly, async t => {
  const { sdk, writer } = await fixture(t, {
    message: ["date_recovered", "share_status", "schedule_state"],
    attachment: ["emoji_image_short_description"],
    chat: ["is_deleting_incoming_messages"],
  });
  const raw = sdk.database.queries.buildMessageQuery({ limit: 1 });
  assert.throws(() => sdk.database.all(raw.sql, raw.params), /no such column: message.date_recovered/);
  validateMessagesSchema(sdk);
  const [message] = await sdk.getMessages({ limit: 1 });
  assert.equal(message.text, "1");
  assert.equal(message.participant, "+15555550101");
  assert.equal(message.isFromMe, false);
  assert.equal(message.kind, "text");
  assert.equal(message.recoveredAt, null);
  assert.equal(message.attachments.length, 1);
  assert.equal((await sdk.listChats({}))[0].kind, "dm");
  // Exercise the same query path the live watcher uses for new replies.
  assert.equal((await sdk.database.getMessagesSinceRowId(0, { limit: 100 }))[0].text, "1");
  assert.equal(writer.pragma("table_info(message)").some(row => row.name === "date_recovered"), false);
});

test("newer schema preserves real optional-field values", macOnly, async t => {
  const { sdk } = await fixture(t);
  const before = sdk.database.queries.buildMessageQuery({ limit: 1 });
  validateMessagesSchema(sdk);
  assert.deepEqual(sdk.database.queries.buildMessageQuery({ limit: 1 }), before);
  const [message] = await sdk.getMessages({ limit: 1 });
  assert.equal(message.recoveredAt.getTime(), Date.UTC(2001, 0, 1) + 2000);
});

test("missing required fields fail rather than dropping reply identity", macOnly, async t => {
  const { sdk } = await fixture(t);
  sdk.database.all("SELECT * FROM message LIMIT 0");
  const all = sdk.database.all.bind(sdk.database);
  sdk.database.all = (sql, params) => {
    const rows = all(sql, params);
    return sql === "PRAGMA table_info(message)" ? rows.filter(row => row.name !== "guid") : rows;
  };
  assert.throws(() => validateMessagesSchema(sdk), /missing required fields message.guid/);
});

test("diagnostic executes only schema checks and explicit zero-row SELECTs", macOnly, async t => {
  const { sdk } = await fixture(t, { message: ["date_recovered"] });
  const sqls = [];
  const all = sdk.database.all.bind(sdk.database);
  sdk.database.all = (sql, params) => {
    sqls.push(sql);
    const rows = all(sql, params);
    if (!sql.startsWith("PRAGMA")) assert.deepEqual(rows, []);
    return rows;
  };
  let closed = false;
  class DiagnosticSDK {
    constructor() { this.database = sdk.database; }
    async getMessages() { assert.fail("Diagnostics must not query message history"); }
    async close() { closed = true; }
  }
  await checkMessagesAccess(DiagnosticSDK);
  assert.equal(closed, true);
  assert.ok(sqls.filter(sql => !sql.startsWith("PRAGMA")).every(sql => /LIMIT 0$/.test(sql)));
});

test("diagnostic closes the SDK after a schema failure", async () => {
  let closed = false;
  class SDK {
    async close() { closed = true; }
  }
  await assert.rejects(checkMessagesAccess(SDK), /Unsupported iMessage SDK internals/);
  assert.equal(closed, true);
});

test("compatibility reaches SDK instances created inside the provider", macOnly, async t => {
  const { sdk } = await fixture(t, { message: ["date_recovered"] });
  installMessagesCompatibility(IMessageSDK);
  installMessagesCompatibility(IMessageSDK);
  assert.equal((await sdk.getMessages({ limit: 1 }))[0].text, "1");
});

test("live reply watcher starts against an older schema", macOnly, async t => {
  const { sdk } = await fixture(t, { message: ["date_recovered", "schedule_state"] });
  installMessagesCompatibility(IMessageSDK);
  let resolveBatch, rejectBatch;
  const batch = new Promise((resolve, reject) => { resolveBatch = resolve; rejectBatch = reject; });
  const read = sdk.database.getMessagesSinceRowId.bind(sdk.database);
  sdk.database.getMessagesSinceRowId = async (...args) => {
    const messages = await read(...args);
    resolveBatch(messages);
    return messages;
  };
  const timer = setTimeout(() => rejectBatch(new Error("Watcher did not query the fixture")), 2000);
  try {
    await sdk.startWatching({ onError: rejectBatch });
    assert.deepEqual(await batch, []);
  } finally {
    clearTimeout(timer);
    await sdk.stopWatching();
  }
});

test("schema errors get compatibility advice; access errors get permission advice", () => {
  const hint = messagingErrorHint(new Error("Query failed: no such column: message.date_recovered"));
  assert.match(hint, /compatibility/);
  assert.doesNotMatch(hint, /Full Disk Access/);
  assert.match(messagingErrorHint(new Error("Failed to open database: unable to open database file")), /Full Disk Access/);
  assert.equal(messagingErrorHint(new Error("Unknown failure")), null);
});
