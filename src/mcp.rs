//! # MCP server — expose CCOS memory as Model Context Protocol tools
//!
//! A dependency-free [Model Context Protocol](https://modelcontextprotocol.io)
//! server over **stdio JSON-RPC 2.0**, so any MCP-compatible agent (Claude, a
//! local agent on the Jetson, …) can use CCOS as its working memory natively. The
//! memory lives in an [`AgentSession`], so the whole interaction is event-sourced
//! and replayable.
//!
//! Tools: `ingest`, `recall`, `signal_failure`, `page_fault`, `stats`, `verify`.
//! Run with `ccos mcp` and point your MCP client's stdio transport at it.

use crate::agent_session::AgentSession;
use crate::external_memory::{ExternalMemory, Recall};
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
        }
    ])
}

/// Wrap a payload string as MCP tool-call content.
fn content(text: String) -> Value {
    json!({ "content": [{ "type": "text", "text": text }] })
}

/// Execute a `tools/call`.
fn call_tool(session: &mut AgentSession, params: &Value) -> Result<Value, (i64, String)> {
    let name = params.get("name").and_then(Value::as_str).unwrap_or("");
    let args = params
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| json!({}));
    let s = |k: &str| {
        args.get(k)
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string()
    };
    let budget = args.get("budget").and_then(Value::as_u64).unwrap_or(2048) as usize;

    let text = match name {
        "ingest" => {
            let uri = s("uri");
            if uri.is_empty() {
                return Err((-32602, "ingest requires 'uri' and 'source'".into()));
            }
            serde_json::to_string(&session.ingest(&uri, &s("source"))).unwrap_or_default()
        }
        "signal_failure" => {
            let depth = args.get("depth").and_then(Value::as_u64).unwrap_or(3) as u32;
            match session.signal_failure(&s("node"), depth) {
                Ok(n) => json!({ "affected": n }).to_string(),
                Err(e) => {
                    return Ok(json!({
                        "content": [{ "type": "text", "text": e.to_string() }],
                        "isError": true
                    }))
                }
            }
        }
        "recall" => {
            let recall = match args
                .get("strategy")
                .and_then(Value::as_str)
                .unwrap_or("working_set")
            {
                "around" => Recall::around(s("anchor")),
                "task" => Recall::task(s("text")),
                _ => Recall::working_set(),
            };
            serde_json::to_string(&session.recall(recall, budget)).unwrap_or_default()
        }
        "page_fault" => {
            serde_json::to_string(&session.page_fault(&s("output"), budget)).unwrap_or_default()
        }
        "stats" => serde_json::to_string(&session.memory().stats()).unwrap_or_default(),
        "verify" => serde_json::to_string(&session.memory().verify()).unwrap_or_default(),
        other => return Err((-32602, format!("unknown tool: {other}"))),
    };
    Ok(content(text))
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
                "capabilities": { "tools": {} },
                "serverInfo": { "name": "ccos-memory", "version": env!("CARGO_PKG_VERSION") }
            }))
        }
        "tools/list" => Ok(json!({ "tools": tool_specs() })),
        "tools/call" => call_tool(session, msg.get("params").unwrap_or(&Value::Null)),
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

/// Run the stdio JSON-RPC loop until EOF. One JSON message per line.
pub fn serve() {
    use std::io::{BufRead, Write};
    let mut session = AgentSession::new();
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let reply = match serde_json::from_str::<Value>(line) {
            Ok(msg) => handle(&mut session, &msg),
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
}
