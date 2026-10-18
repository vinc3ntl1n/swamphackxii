// Compatibility for imessage-kit 3.0.0's fixed macOS 26 SELECT lists.
// Only optional projected fields change; Messages' database is never modified.
const prepared = new WeakSet();
const installed = new WeakSet();
const REQUIRED = {
  message: ["guid", "date", "text", "attributedBody", "handle_id", "other_handle", "is_from_me", "is_read", "item_type"],
  chat: ["guid", "chat_identifier", "style"],
  attachment: ["guid", "created_date", "filename", "hide_attachment"],
};

function adaptProjections(sql, columns) {
  return sql.replace(/\bSELECT\b([\s\S]*?)\bFROM\b/gi, (select) =>
    select.replace(/\b(message|chat|attachment)\.([a-z_]\w*)(\s+AS\s+\w+)?/gi,
      (reference, table, column, alias) => {
        if (column.toLowerCase() === "rowid" || columns[table.toLowerCase()].has(column.toLowerCase())) return reference;
        return `NULL${alias || ` AS ${column}`}`;
      }));
}

export function validateMessagesSchema(sdk) {
  if (prepared.has(sdk)) return;
  const reader = sdk.database;
  if (!reader?.all || !reader.queries?.buildMessageQuery || !reader.queries?.buildAttachmentQuery) {
    throw new Error("Unsupported iMessage SDK internals; install the versions in messaging/package-lock.json.");
  }
  const columns = {};
  for (const [table, required] of Object.entries(REQUIRED)) {
    columns[table] = new Set(reader.all(`PRAGMA table_info(${table})`).map(row => row.name.toLowerCase()));
    const missing = required.filter(column => !columns[table].has(column.toLowerCase()));
    if (missing.length) throw new Error(`Unsupported Messages schema: missing required fields ${missing.map(column => `${table}.${column}`).join(", ")}`);
  }

  // Each reader gets its own builders; the SDK's shared defaults stay intact.
  const original = reader.queries;
  const queries = { ...original };
  for (const name of ["buildMessageQuery", "buildChatQuery", "buildAttachmentQuery"]) {
    if (typeof original[name] !== "function") continue;
    queries[name] = (...args) => {
      const query = original[name](...args);
      return { ...query, sql: adaptProjections(query.sql, columns) };
    };
  }
  reader.queries = queries;

  // The SDK treats limit: 0 as unlimited. An outer SQL LIMIT 0 validates the
  // actual joins and optional fields without returning any message history.
  for (const query of [queries.buildMessageQuery({ limit: 1 }), queries.buildAttachmentQuery([0])]) {
    reader.all(`SELECT * FROM (${query.sql}) LIMIT 0`, query.params);
  }
  prepared.add(sdk);
}

export function installMessagesCompatibility(IMessageSDK) {
  if (installed.has(IMessageSDK)) return;
  for (const name of ["getMessages", "listChats", "startWatching"]) {
    const original = IMessageSDK.prototype[name];
    if (typeof original !== "function") throw new Error(`Unsupported iMessage SDK: missing ${name}`);
    IMessageSDK.prototype[name] = async function (...args) {
      validateMessagesSchema(this);
      return original.apply(this, args);
    };
  }
  installed.add(IMessageSDK);
}

export function messagingErrorHint(error) {
  const message = error.message || "";
  if (error.code === "ERR_MODULE_NOT_FOUND") return "Install the messaging dependencies from the repo root: npm ci --prefix messaging";
  if (/no such (?:column|table)|unsupported (?:Messages schema|iMessage SDK)/i.test(message)) {
    return "This is a Messages database / SDK compatibility error. Paste the error above so the compatibility adapter can be checked.";
  }
  if (/unable to open database|cannot open database|permission denied|operation not permitted|not authorized/i.test(message)) {
    return "Open Messages and sign into iMessage. Enable Full Disk Access for the app running this command in System Settings → Privacy & Security, then quit that app completely and reopen it.";
  }
  return null;
}
