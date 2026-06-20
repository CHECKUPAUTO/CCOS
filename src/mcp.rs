//! # MCP server — expose CCOS memory as Model Context Protocol tools
//!
//! A dependency-free [Model Context Protocol](https://modelcontextprotocol.io)
//! server over **stdio JSON-RPC 2.0**, so any MCP-compatible agent (Claude, a
//! local agent on the Jetson, …) can use CCOS as its working memory natively. The
//! memory lives in an [`AgentSession`], so the whole interaction is event-sourced
//! and replayable.
//!
//! Tools: `ingest`, `recall`, `signal_failure`, `page_fault`, `stats`, `verify`,
//! plus the time-travel pair `timeline` / `recall_what_if`. It also exposes two
//! read-only **resources** — `ccos://session/context` (the current
//! self-bounding working set, linearised for direct injection into a system
//! prompt) and `ccos://session/timeline` (the cognitive journal).
//!
//! Run with `ccos mcp [workspace.ccos]`. With a path, the session reloads that
//! checkpoint on start and re-checkpoints after every memory-changing call, so
//! the memory survives restarts; without one it stays purely in-process.
//! Point your MCP client's stdio transport at it.

use crate::agent_session::AgentSession;
use crate::external_memory::{ExternalMemory, Recall, RecallWindow};
use serde_json::{json, Value};

/// MCP protocol revision we speak (echoed back to the client when offered).
pub const PROTOCOL_VERSION: &str = "2024-11-05";

/// The tool catalogue advertised by `tools/list`, with JSON-Schema inputs.
fn tool_specs() -> Value {
    json!([
        {
            "name": "ingest",
            "description": "Ingest (or update) a source file into the causal memory graph.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "uri": {"type": "string", "description": "file path, e.g. src/db.rs"},
                    "source": {"type": "string"}
                },
                "required": ["uri", "source"]
            }
        },
        {
            "name": "recall",
            "description": "Recall a bounded, causally-coherent context window.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "strategy": {"type": "string", "enum": ["around", "task", "working_set"]},
                    "anchor": {"type": "string", "description": "node id / file uri for 'around'"},
                    "text": {"type": "string", "description": "free-text task for 'task'"},
                    "budget": {"type": "integer", "description": "token budget (default 2048)"}
                }
            }
        },
        {
            "name": "signal_failure",
            "description": "Mark a node as failing and propagate the pressure across the graph.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "node": {"type": "string"},
                    "depth": {"type": "integer", "description": "propagation depth (default 3)"}
                },
                "required": ["node"]
            }
        },
        {
            "name": "page_fault",
            "description": "Feed cargo-test/compiler output back in: parse the faulting files, inject pressure, recall a refreshed window.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output": {"type": "string", "description": "cargo test / panic / backtrace text"},
                    "budget": {"type": "integer"}
                },
                "required": ["output"]
            }
        },
        {
            "name": "stats",
            "description": "Memory counts (nodes/edges/events/files).",
            "inputSchema": {"type": "object", "properties": {}}
        },
        {
            "name": "verify",
            "description": "Verify the tamper-evident hash chain.",
            "inputSchema": {"type": "object", "properties": {}}
        },
        {
            "name": "timeline",
            "description": "The event-sourced cognitive timeline: every recorded operation (ingest / signal_failure / recall / page_fault), in order.",
            "inputSchema": {"type": "object", "properties": {}}
        },
        {
            "name": "recall_what_if",
            "description": "Time-travel debugging: rewind to a past step and re-run a recall under (possibly) different parameters — a deterministic replay of what the agent's window would have been.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "step": {"type": "integer", "description": "timeline step to rewind to (0 = before any op)"},
                    "strategy": {"type": "string", "enum": ["around", "task", "working_set"]},
                    "anchor": {"type": "string"},
                    "text": {"type": "string"},
                    "budget": {"type": "integer"}
                },
                "required": ["step"]
            }
        }
    ])
}

/// The read-only resources advertised by `resources/list`.
fn resource_specs() -> Value {
    json!([
        {
            "uri": "ccos://session/context",
            "name": "CCOS working-set context",
            "description": "The current causally-scored, token-bounded working set, linearised for direct injection into a system prompt. Reflects accumulated failure pressure and recency; self-bounds at the causal region (no K to tune). Budget via CCOS_MCP_CONTEXT_BUDGET (default 2048 tokens).",
            "mimeType": "text/plain"
        },
        {
            "uri": "ccos://session/timeline",
            "name": "CCOS cognitive timeline",
            "description": "The event-sourced journal of every memory operation this session (audit / replay).",
            "mimeType": "text/plain"
        }
    ])
}

/// Wrap a payload string as MCP tool-call content.
fn content(text: String) -> Value {
    json!({ "content": [{ "type": "text", "text": text }] })
}

/// Read a string argument (empty when absent).
fn str_arg(args: &Value, k: &str) -> String {
    args.get(k)
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string()
}

/// Build a [`Recall`] strategy from `{strategy, anchor, text}` arguments. Shared
/// by `recall` and the time-travel `recall_what_if`.
fn recall_from_args(args: &Value) -> Recall {
    match args
        .get("strategy")
        .and_then(Value::as_str)
        .unwrap_or("working_set")
    {
        "around" => Recall::around(str_arg(args, "anchor")),
        "task" => Recall::task(str_arg(args, "text")),
        _ => Recall::working_set(),
    }
}

/// Execute a `tools/call`.
fn call_tool(session: &mut AgentSession, params: &Value) -> Result<Value, (i64, String)> {
    let name = params.get("name").and_then(Value::as_str).unwrap_or("");
    let args = params
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| json!({}));
    let budget = args.get("budget").and_then(Value::as_u64).unwrap_or(2048) as usize;

    let text = match name {
        "ingest" => {
            let uri = str_arg(&args, "uri");
            if uri.is_empty() {
                return Err((-32602, "ingest requires 'uri' and 'source'".into()));
            }
            serde_json::to_string(&session.ingest(&uri, &str_arg(&args, "source")))
                .unwrap_or_default()
        }
        "signal_failure" => {
            let depth = args.get("depth").and_then(Value::as_u64).unwrap_or(3) as u32;
            match session.signal_failure(&str_arg(&args, "node"), depth) {
                Ok(n) => json!({ "affected": n }).to_string(),
                Err(e) => {
                    return Ok(json!({
                        "content": [{ "type": "text", "text": e.to_string() }],
                        "isError": true
                    }))
                }
            }
        }
        "recall" => serde_json::to_string(&session.recall(recall_from_args(&args), budget))
            .unwrap_or_default(),
        "page_fault" => {
            serde_json::to_string(&session.page_fault(&str_arg(&args, "output"), budget))
                .unwrap_or_default()
        }
        "stats" => serde_json::to_string(&session.memory().stats()).unwrap_or_default(),
        "verify" => serde_json::to_string(&session.memory().verify()).unwrap_or_default(),
        "timeline" => json!({ "timeline": session.timeline() }).to_string(),
        "recall_what_if" => {
            let step = args.get("step").and_then(Value::as_u64).unwrap_or(0) as usize;
            let window = session.recall_what_if(step, &recall_from_args(&args), budget);
            serde_json::to_string(&window).unwrap_or_default()
        }
        other => return Err((-32602, format!("unknown tool: {other}"))),
    };
    Ok(content(text))
}

/// Linearise a recalled window into a single text blob a host can drop straight
/// into a system prompt (the auto-calibrated context chain).
fn linearize_window(win: &RecallWindow) -> String {
    let mut out = format!(
        "// CCOS context — {} ({} items, ~{} tokens)\n",
        win.strategy,
        win.items.len(),
        win.tokens
    );
    for it in &win.items {
        out.push_str(&format!(
            "\n// {} [{}] score={:.3}\n{}\n",
            it.uri, it.kind, it.score, it.content
        ));
    }
    out
}

/// Execute a `resources/read`.
fn read_resource(session: &AgentSession, params: &Value) -> Result<Value, (i64, String)> {
    let uri = params.get("uri").and_then(Value::as_str).unwrap_or("");
    let text = match uri {
        "ccos://session/context" => {
            // Budget tunable at launch without a flag.
            let budget = std::env::var("CCOS_MCP_CONTEXT_BUDGET")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(2048usize);
            // Anchor on the workspace signal: if something is failing, inject the
            // causal *region* of that problem (far more useful on a real codebase
            // than the global working set, which a `use`-heavy repo fills with the
            // hottest file); otherwise fall back to the global working set.
            let mem = session.memory();
            let window = match mem.hottest_failure_node() {
                Some(anchor) => mem.recall(&Recall::around(anchor), budget),
                None => mem.recall(&Recall::working_set(), budget),
            };
            linearize_window(&window)
        }
        "ccos://session/timeline" => session.timeline().join("\n"),
        other => return Err((-32602, format!("unknown resource: {other}"))),
    };
    Ok(json!({ "contents": [{ "uri": uri, "mimeType": "text/plain", "text": text }] }))
}

/// Handle one JSON-RPC message. Returns `Some(response)` for a request, `None`
/// for a notification (which gets no reply).
pub fn handle(session: &mut AgentSession, msg: &Value) -> Option<Value> {
    let id = msg.get("id").cloned();
    let method = msg.get("method").and_then(Value::as_str).unwrap_or("");
    // Notifications carry no id and expect no response.
    id.as_ref()?;
    let id = id.unwrap();

    let result: Result<Value, (i64, String)> = match method {
        "initialize" => {
            let pv = msg
                .get("params")
                .and_then(|p| p.get("protocolVersion"))
                .and_then(Value::as_str)
                .unwrap_or(PROTOCOL_VERSION)
                .to_string();
            Ok(json!({
                "protocolVersion": pv,
                "capabilities": { "tools": {}, "resources": {} },
                "serverInfo": { "name": "ccos-memory", "version": env!("CARGO_PKG_VERSION") }
            }))
        }
        "tools/list" => Ok(json!({ "tools": tool_specs() })),
        "tools/call" => call_tool(session, msg.get("params").unwrap_or(&Value::Null)),
        "resources/list" => Ok(json!({ "resources": resource_specs() })),
        "resources/read" => read_resource(session, msg.get("params").unwrap_or(&Value::Null)),
        "ping" => Ok(json!({})),
        _ => Err((-32601, format!("method not found: {method}"))),
    };

    Some(match result {
        Ok(r) => json!({ "jsonrpc": "2.0", "id": id, "result": r }),
        Err((code, message)) => {
            json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } })
        }
    })
}

/// Run the stdio JSON-RPC loop on a fresh **in-memory** session (nothing is
/// persisted). See [`serve_workspace`] for the persistent variant.
pub fn serve() {
    serve_session(AgentSession::new());
}

/// Run the stdio loop, optionally persisting to (and reloading from) a workspace
/// checkpoint. With `Some(path)` the session loads that checkpoint on start and
/// re-checkpoints after every memory-changing call (and once more at EOF), so
/// the causal memory survives restarts; with `None` it behaves like [`serve`].
pub fn serve_workspace(
    workspace: Option<std::path::PathBuf>,
) -> Result<(), crate::external_memory::MemoryError> {
    let session = match workspace {
        Some(p) => AgentSession::open(p)?,
        None => AgentSession::new(),
    };
    serve_session(session);
    Ok(())
}

/// The shared stdio JSON-RPC loop until EOF. One JSON message per line; a
/// best-effort checkpoint follows every state-changing tool call.
fn serve_session(mut session: AgentSession) {
    use std::io::{BufRead, Write};
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let reply = match serde_json::from_str::<Value>(line) {
            Ok(msg) => {
                let mutated = is_mutating_call(&msg);
                let resp = handle(&mut session, &msg);
                if mutated {
                    persist(&mut session);
                }
                resp
            }
            Err(_) => Some(json!({
                "jsonrpc": "2.0", "id": Value::Null,
                "error": { "code": -32700, "message": "parse error" }
            })),
        };
        if let Some(resp) = reply {
            let mut out = stdout.lock();
            let _ = writeln!(out, "{resp}");
            let _ = out.flush();
        }
    }
    persist(&mut session); // final checkpoint at close (no-op when no path is bound)
}

/// True iff the message is a `tools/call` to a state-changing tool.
fn is_mutating_call(msg: &Value) -> bool {
    if msg.get("method").and_then(Value::as_str) != Some("tools/call") {
        return false;
    }
    let name = msg
        .get("params")
        .and_then(|p| p.get("name"))
        .and_then(Value::as_str)
        .unwrap_or("");
    matches!(name, "ingest" | "signal_failure" | "page_fault")
}

/// Checkpoint the session, best-effort: silent when no path is bound, a stderr
/// line on a real IO/serialisation error (stdout is reserved for JSON-RPC).
fn persist(session: &mut AgentSession) {
    use crate::external_memory::MemoryError;
    match session.checkpoint() {
        Ok(()) | Err(MemoryError::NoPath) => {}
        Err(e) => eprintln!("ccos mcp: checkpoint failed: {e}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(id: i64, method: &str, params: Value) -> Value {
        json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params })
    }

    #[test]
    fn initialize_echoes_protocol_and_names_the_server() {
        let mut s = AgentSession::new();
        let r = handle(
            &mut s,
            &req(1, "initialize", json!({ "protocolVersion": "2025-01-01" })),
        )
        .unwrap();
        assert_eq!(r["result"]["protocolVersion"], "2025-01-01");
        assert_eq!(r["result"]["serverInfo"]["name"], "ccos-memory");
        assert!(r["result"]["capabilities"]["tools"].is_object());
    }

    #[test]
    fn tools_list_advertises_the_catalogue() {
        let mut s = AgentSession::new();
        let r = handle(&mut s, &req(2, "tools/list", Value::Null)).unwrap();
        let names: Vec<&str> = r["result"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["name"].as_str().unwrap())
            .collect();
        for n in [
            "ingest",
            "recall",
            "signal_failure",
            "page_fault",
            "stats",
            "verify",
            "timeline",
            "recall_what_if",
        ] {
            assert!(names.contains(&n), "missing tool {n}");
        }
    }

    #[test]
    fn notification_gets_no_response() {
        let mut s = AgentSession::new();
        let n = json!({ "jsonrpc": "2.0", "method": "notifications/initialized" });
        assert!(handle(&mut s, &n).is_none());
    }

    #[test]
    fn ingest_then_recall_round_trips_through_tools() {
        let mut s = AgentSession::new();
        handle(
            &mut s,
            &req(
                1,
                "tools/call",
                json!({
                    "name": "ingest",
                    "arguments": { "uri": "src/a.rs", "source": "pub fn a() {}\n" }
                }),
            ),
        )
        .unwrap();
        let r = handle(
            &mut s,
            &req(
                2,
                "tools/call",
                json!({
                    "name": "recall",
                    "arguments": { "strategy": "working_set", "budget": 1000 }
                }),
            ),
        )
        .unwrap();
        let text = r["result"]["content"][0]["text"].as_str().unwrap();
        assert!(
            text.contains("file:src/a.rs"),
            "recall returns the ingested file: {text}"
        );
    }

    #[test]
    fn unknown_method_is_a_jsonrpc_error() {
        let mut s = AgentSession::new();
        let r = handle(&mut s, &req(9, "frobnicate", Value::Null)).unwrap();
        assert_eq!(r["error"]["code"], -32601);
    }

    fn ingest(s: &mut AgentSession, id: i64, uri: &str, src: &str) {
        handle(
            s,
            &req(
                id,
                "tools/call",
                json!({ "name": "ingest", "arguments": { "uri": uri, "source": src } }),
            ),
        )
        .unwrap();
    }

    #[test]
    fn time_travel_what_if_replays_a_past_step() {
        let mut s = AgentSession::new();
        ingest(&mut s, 1, "src/db.rs", "pub fn q() {}\n");
        ingest(
            &mut s,
            2,
            "src/api.rs",
            "use crate::db;\npub fn h() { db::q() }\n",
        );
        // Rewind to step 1 (only db.rs ingested): the window must predate api.rs.
        let r = handle(
            &mut s,
            &req(
                3,
                "tools/call",
                json!({
                    "name": "recall_what_if",
                    "arguments": { "step": 1, "strategy": "working_set", "budget": 4000 }
                }),
            ),
        )
        .unwrap();
        let text = r["result"]["content"][0]["text"].as_str().unwrap();
        assert!(
            text.contains("file:src/db.rs"),
            "what-if sees db.rs: {text}"
        );
        assert!(
            !text.contains("file:src/api.rs"),
            "step-1 replay predates api.rs: {text}"
        );
    }

    #[test]
    fn initialize_advertises_resources() {
        let mut s = AgentSession::new();
        let r = handle(&mut s, &req(1, "initialize", json!({}))).unwrap();
        assert!(r["result"]["capabilities"]["resources"].is_object());
    }

    #[test]
    fn resources_list_and_read_the_context_window() {
        let mut s = AgentSession::new();
        ingest(&mut s, 1, "src/a.rs", "pub fn alpha() {}\n");

        let list = handle(&mut s, &req(2, "resources/list", Value::Null)).unwrap();
        let uris: Vec<&str> = list["result"]["resources"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r["uri"].as_str().unwrap())
            .collect();
        assert!(uris.contains(&"ccos://session/context"));

        let read = handle(
            &mut s,
            &req(
                3,
                "resources/read",
                json!({ "uri": "ccos://session/context" }),
            ),
        )
        .unwrap();
        let text = read["result"]["contents"][0]["text"].as_str().unwrap();
        assert!(
            text.contains("file:src/a.rs"),
            "context resource linearises the working set: {text}"
        );
    }

    #[test]
    fn context_resource_anchors_on_the_active_failure() {
        let mut s = AgentSession::new();
        ingest(&mut s, 1, "src/db.rs", "pub fn q() {}\n");
        ingest(
            &mut s,
            2,
            "src/api.rs",
            "use crate::db;\npub fn h() { db::q() }\n",
        );
        // A failure on db.rs → the injected context should be db.rs's causal region.
        handle(
            &mut s,
            &req(
                3,
                "tools/call",
                json!({ "name": "signal_failure", "arguments": { "node": "file:src/db.rs" } }),
            ),
        )
        .unwrap();
        let read = handle(
            &mut s,
            &req(
                4,
                "resources/read",
                json!({ "uri": "ccos://session/context" }),
            ),
        )
        .unwrap();
        let text = read["result"]["contents"][0]["text"].as_str().unwrap();
        assert!(
            text.contains("file:src/db.rs"),
            "context anchors on the failing file: {text}"
        );
    }

    #[test]
    fn unknown_resource_is_a_jsonrpc_error() {
        let mut s = AgentSession::new();
        let r = handle(
            &mut s,
            &req(1, "resources/read", json!({ "uri": "ccos://session/nope" })),
        )
        .unwrap();
        assert_eq!(r["error"]["code"], -32602);
    }

    #[test]
    fn only_state_changing_tools_trigger_a_checkpoint() {
        let mutating = |name: &str| {
            is_mutating_call(&json!({
                "method": "tools/call", "params": { "name": name }
            }))
        };
        assert!(mutating("ingest"));
        assert!(mutating("signal_failure"));
        assert!(mutating("page_fault"));
        assert!(!mutating("recall"));
        assert!(!mutating("stats"));
        assert!(!mutating("recall_what_if"));
        // Non-tools/call messages never checkpoint.
        assert!(!is_mutating_call(&json!({ "method": "resources/read" })));
    }
}
