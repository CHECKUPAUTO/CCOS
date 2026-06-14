use ccos::adversarial::{AdversarialEngine, AdversarialMode};
use ccos::consensus::ConsensusEngine;
use ccos::distributed_event_log::DistributedEventLog;
use ccos::event_log::{EventLog, EventPayload, EventReplayer, EventType};
use ccos::guard::{GuardConfig, GuardLayer};
use ccos::incremental::IncrementalGraphEngine;
use ccos::llm::{LlmClient, LlmConfig};
use ccos::memory::{MemoryGraph, NodeId};
use ccos::parser::ASTParser;
use ccos::persist::KernelSnapshot;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use uuid::Uuid;

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let command = args.get(1).map(String::as_str).unwrap_or("demo");
    let rest = &args[args.len().min(2)..];

    let code = match command {
        "-h" | "--help" | "help" => {
            print_help();
            0
        }
        "-V" | "--version" | "version" => {
            println!("ccos {}", env!("CARGO_PKG_VERSION"));
            0
        }
        "demo" => {
            run_demo().await;
            0
        }
        "analyze" => run_analyze(&AnalyzeOpts::parse(rest)),
        "verify" => run_verify(rest.first().map(String::as_str)),
        "replay" => run_replay(rest.first().map(String::as_str)),
        "chaos" => run_chaos(&ChaosOpts::parse(rest)),
        other => {
            eprintln!("ccos: unknown command '{other}'\n");
            print_help();
            2
        }
    };
    std::process::exit(code);
}

/// Built-in end-to-end demonstration of every kernel subsystem on a small
/// synthetic workspace (parsing, LLM + guard, incremental delta, failure
/// propagation, context selection, deterministic replay, paging).
async fn run_demo() {
    println!("╔══════════════════════════════════════════════╗");
    println!("║  CCOS — Causal Context Operating System     ║");
    println!("║  Kernel v{} | Rust 2021                   ║", env!("CARGO_PKG_VERSION"));
    println!("╚══════════════════════════════════════════════╝\n");

    // ── Initialization ─────────────────────────────────────────────
    let session_id = Uuid::new_v4().to_string();
    println!("[INIT] Session ID: {}", session_id);

    let mut event_log = EventLog::new(session_id.clone());
    let mut memory_graph = MemoryGraph::new(0.2, 80);
    let mut incremental_engine = IncrementalGraphEngine::new();
    let _guard = GuardLayer::new(GuardConfig::default());

    let llm_config = LlmConfig {
        endpoint: std::env::var("OLLAMA_ENDPOINT")
            .unwrap_or_else(|_| "http://localhost:11434".into()),
        model: std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "codellama".into()),
        ..Default::default()
    };
    let llm_client = LlmClient::new(llm_config.clone());
    let mut dist_log = DistributedEventLog::new();
    dist_log.append("kernel_initialized".into(), "kernel".into());

    // ── Event Sourcing: Log initialization ────────────────────────
    event_log.append(
        EventType::CycleStart,
        EventPayload::CycleEvent {
            cycle_number: 0,
            action: "kernel_initialized".into(),
        },
    );

    // ── Workspace Simulation ──────────────────────────────────────
    println!("\n─── PHASE 1: Workspace Ingestion & AST Parsing ───\n");

    let mut workspace: HashMap<String, String> = HashMap::new();
    workspace.insert(
        "src/lib.rs".into(),
        r#"mod auth;
mod database;
mod api;

use std::collections::HashMap;
use tokio::runtime::Runtime;
use serde::{Serialize, Deserialize};

pub struct AppState {
    pub db: Database,
    pub auth: AuthService,
    pub config: AppConfig,
}

pub struct AppConfig {
    pub port: u16,
    pub host: String,
}

pub fn init_app(config: AppConfig) -> AppState {
    let db = Database::connect(&config);
    let auth = AuthService::new();
    AppState { db, auth, config }
}"#
        .into(),
    );

    workspace.insert(
        "src/auth.rs".into(),
        r#"use sha2::{Sha256, Digest};
use std::collections::HashMap;

pub struct AuthService {
    sessions: HashMap<String, Session>,
}

pub struct Session {
    pub user_id: String,
    pub token: String,
    pub expires_at: u64,
}

impl AuthService {
    pub fn new() -> Self {
        Self {
            sessions: HashMap::new(),
        }
    }

    pub fn authenticate(&self, token: &str) -> Option<&Session> {
        self.sessions.get(token)
    }

    pub fn create_session(&mut self, user_id: &str) -> Session {
        let token = Self::generate_token(user_id);
        let session = Session {
            user_id: user_id.to_string(),
            token: token.clone(),
            expires_at: 0,
        };
        self.sessions.insert(token.clone(), session.clone());
        session
    }

    fn generate_token(user_id: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(user_id.as_bytes());
        format!("{:x}", hasher.finalize())
    }
}"#
        .into(),
    );

    workspace.insert(
        "src/database.rs".into(),
        r#"use std::sync::Mutex;
use std::collections::HashMap;

pub struct Database {
    store: Mutex<HashMap<String, Vec<u8>>>,
    connected: bool,
}

impl Database {
    pub fn connect(config: &crate::AppConfig) -> Self {
        Self {
            store: Mutex::new(HashMap::new()),
            connected: true,
        }
    }

    pub fn insert(&self, key: &str, value: Vec<u8>) {
        if let Ok(mut store) = self.store.lock() {
            store.insert(key.to_string(), value);
        }
    }

    pub fn get(&self, key: &str) -> Option<Vec<u8>> {
        self.store.lock().ok()?.get(key).cloned()
    }

    pub fn is_connected(&self) -> bool {
        self.connected
    }
}"#
        .into(),
    );

    workspace.insert(
        "src/api.rs".into(),
        r#"use crate::AppState;
use serde_json::Value;

pub async fn handle_request(state: &AppState, path: &str, body: &str) -> Result<Value, String> {
    match path {
        "/health" => Ok(serde_json::json!({"status": "ok"})),
        "/auth/login" => {
            let session = state.auth.create_session("user_1");
            Ok(serde_json::json!({"token": session.token}))
        }
        "/data" => {
            let data = state.db.get("default");
            Ok(serde_json::json!({"data": data}))
        }
        _ => Err("not found".into()),
    }
}"#
        .into(),
    );

    // ── Phase 1: Initial parsing ──────────────────────────────────
    for (file_path, source_code) in &workspace {
        event_log.append(
            EventType::Parsing,
            EventPayload::Parsing {
                file_path: file_path.clone(),
                file_hash: compute_file_hash(source_code),
                modules_found: 0,
                uses_found: 0,
                symbols_found: 0,
            },
        );

        let parse_result = incremental_engine.register_file(file_path, source_code);
        let parser = ASTParser::new();
        parser.update_memory_graph(&parse_result, &mut memory_graph);

        let _event_id = event_log.append(
            EventType::GraphMutation,
            EventPayload::GraphMutation {
                node_id: format!("file:{}", file_path),
                operation: "ingest".into(),
                nodes_before: memory_graph.node_count().saturating_sub(parse_result.generated_nodes),
                nodes_after: memory_graph.node_count(),
                edges_before: memory_graph.edge_count().saturating_sub(parse_result.generated_edges),
                edges_after: memory_graph.edge_count(),
            },
        );

        println!(
            "  [PARSE] {} → {} modules, {} uses, {} symbols (nodes: {}, edges: {})",
            file_path,
            parse_result.modules.len(),
            parse_result.use_statements.len(),
            parse_result.symbols.len(),
            memory_graph.node_count(),
            memory_graph.edge_count(),
        );
    }

    // ── Phase 2: LLM Interaction ──────────────────────────────────
    println!("\n─── PHASE 2: LLM Integration & Guard Layer ───\n");

    let prompt = format!(
        r#"Analyze the following Rust codebase graph:
Nodes: {}
Edges: {}
Top modules: Analyze the dependency structure and identify potential issues."#,
        memory_graph.node_count(),
        memory_graph.edge_count()
    );

    event_log.append(
        EventType::LlmCall,
        EventPayload::LlmCallRequest {
            model: llm_config.model.clone(),
            prompt_hash: compute_file_hash(&prompt),
            input_tokens: prompt.len(),
        },
    );

    let validated = llm_client
        .query(&prompt, Some("You are a code analysis assistant. Respond with valid JSON only."))
        .await;

    let _guard_event = event_log.append(
        EventType::GuardCheck,
        EventPayload::GuardCheck {
            input_hash: validated.response_hash.clone(),
            passed: validated.guard_passed,
            score: validated.reliability_score,
            warnings: validated.guard_warnings.clone(),
        },
    );

    event_log.append(
        EventType::LlmResponse,
        EventPayload::LlmCallResponse {
            model: validated.model.clone(),
            response_hash: validated.response_hash.clone(),
            output_tokens: validated.sanitized_output.len(),
            latency_ms: validated.latency_ms,
            guard_passed: validated.guard_passed,
            reliability_score: validated.reliability_score,
        },
    );

    let guard_status = if validated.guard_passed {
        "PASSED"
    } else {
        "BLOCKED"
    };
    println!(
        "  [LLM] Query → {} | Guard: {} (score: {:.2}) | Latency: {}ms | Fallback: {}",
        validated.model,
        guard_status,
        validated.reliability_score,
        validated.latency_ms,
        validated.is_fallback
    );
    println!("  [LLM] Output: {:.120}...", validated.sanitized_output);
    dist_log.append(
        format!("llm_guard:{}", validated.guard_passed),
        "llm".into(),
    );

    // Multi-model consensus (illustrative — offline models all fall back, so
    // they agree on the deterministic fallback output).
    let consensus_models = ["codellama", "mistral"];
    let votes = llm_client
        .query_models(&prompt, Some("Respond with valid JSON only."), &consensus_models)
        .await;
    let consensus = ConsensusEngine::with_threshold(0.5).resolve_weighted(&votes);
    println!(
        "  [CONSENSUS] {} models, agreement {:.2} → reached: {} (models: {:?})",
        consensus.total_votes,
        consensus.agreement_ratio,
        consensus.consensus_reached,
        consensus.models_in_agreement,
    );
    dist_log.append(
        format!("consensus_reached:{}", consensus.consensus_reached),
        "consensus".into(),
    );

    // ── Phase 3: Incremental Update Simulation ────────────────────
    println!("\n─── PHASE 3: Incremental Graph Update (O(Δ)) ───\n");
    memory_graph.tick(); // advance the recency clock between cycles

    // Simulate a code change: modify api.rs
    let modified_api = r#"use crate::AppState;
use serde_json::Value;

pub async fn handle_request(state: &AppState, path: &str, body: &str) -> Result<Value, String> {
    match path {
        "/health" => Ok(serde_json::json!({"status": "ok", "version": "1.0"})),
        "/auth/login" => {
            let session = state.auth.create_session("user_1");
            Ok(serde_json::json!({"token": session.token, "expires": 3600}))
        }
        "/data" => {
            let data = state.db.get("default");
            Ok(serde_json::json!({"data": data}))
        }
        "/metrics" => {
            Ok(serde_json::json!({"cpu": 0.42, "mem": 0.73}))
        }
        _ => Err("not found".into()),
    }
}"#;

    let old_source = workspace.get("src/api.rs").map(|s| s.as_str());
    let delta = incremental_engine.process_delta(
        "src/api.rs",
        old_source,
        modified_api,
        &mut memory_graph,
    );

    workspace.insert("src/api.rs".into(), modified_api.into());

    event_log.append(
        EventType::GraphMutation,
        EventPayload::GraphMutation {
            node_id: "file:src/api.rs".into(),
            operation: format!("{:?}", delta.operation).to_lowercase(),
            nodes_before: memory_graph.node_count().saturating_sub(delta.nodes_added),
            nodes_after: memory_graph.node_count(),
            edges_before: memory_graph.edge_count().saturating_sub(delta.edges_added),
            edges_after: memory_graph.edge_count(),
        },
    );

    println!(
        "  [DELTA] api.rs → {:?} | Δnodes: +{}/-{} | Δedges: +{}/-{}",
        delta.operation,
        delta.nodes_added,
        delta.nodes_removed,
        delta.edges_added,
        delta.edges_removed,
    );

    // ── Phase 4: Failure Propagation ──────────────────────────────
    println!("\n─── PHASE 4: Failure Propagation ───\n");

    let failure_node = NodeId::from("file:src/database.rs");
    memory_graph.set_failure_relevance(&failure_node, 0.95);

    event_log.append(
        EventType::FailureDetection,
        EventPayload::FailureDetection {
            node_id: failure_node.to_string(),
            failure_type: "connection_lost".into(),
            severity: 0.95,
        },
    );

    let affected: Vec<String> = memory_graph
        .edges
        .iter()
        .filter(|e| e.source == failure_node)
        .map(|e| e.target.to_string())
        .collect();

    memory_graph.propagate_failure(&failure_node, 0, 3);

    event_log.append(
        EventType::FailurePropagation,
        EventPayload::FailurePropagation {
            origin_node_id: failure_node.to_string(),
            affected_nodes: affected.clone(),
            depth: 3,
        },
    );

    println!(
        "  [FAILURE] Origin: {} | Severity: 0.95 | Affected: {:?}",
        failure_node, affected
    );

    // Display updated scores
    let scores = memory_graph.get_node_scores();
    println!("\n  Memory Graph Scores (top 8):");
    for (id, score) in scores.iter().take(8) {
        let node = memory_graph.nodes.get(id);
        let fr = node.map(|n| n.failure_relevance).unwrap_or(0.0);
        let rec = node.map(|n| n.recency).unwrap_or(0.0);
        println!(
            "    {} → score: {:.4} (failure: {:.2}, recency: {:.2})",
            id, score, fr, rec
        );
    }

    // ── Phase 5: Context Window Selection ─────────────────────────
    println!("\n─── PHASE 5: VRAM-like Context Selection ───\n");

    let context = memory_graph.select_context_window(2048);
    println!(
        "  [VRAM] Selected {} nodes for context window (2048 token budget):",
        context.len()
    );
    for node in &context {
        let score = memory_graph.compute_node_score(node);
        println!(
            "    {} ({:?}) — score: {:.3} | recency: {:.2}",
            node.label, node.node_type, score, node.recency
        );
    }

    // ── Phase 6: Event Sourcing Replay ────────────────────────────
    println!("\n─── PHASE 6: Deterministic Replay ───\n");

    let replay_event_id = event_log.append(
        EventType::ReplayStart,
        EventPayload::ReplayEvent {
            original_event_id: "none".into(),
            replayed_at: 0,
        },
    );

    let mut replayer = ccos::event_log::EventReplayer::new();
    match event_log.replay_deterministic(&mut replayer) {
        Ok(count) => {
            println!("  [REPLAY] Replayed {} events successfully", count);
            println!(
                "  [REPLAY] Stats: {} llm, {} parse, {} graph, {} failures",
                replayer.statistics.llm_calls,
                replayer.statistics.parsing_events,
                replayer.statistics.graph_mutations,
                replayer.statistics.failures
            );
        }
        Err(e) => {
            println!("  [REPLAY] Error: {}", e);
        }
    }

    event_log.append(
        EventType::ReplayEnd,
        EventPayload::ReplayEvent {
            original_event_id: replay_event_id,
            replayed_at: 1,
        },
    );

    // ── Phase 7: Paging Enforcement ───────────────────────────────
    println!("\n─── PHASE 7: Paging Threshold Enforcement ───\n");
    let before_paging = memory_graph.node_count();
    memory_graph.max_in_memory_nodes = 15;
    memory_graph.enforce_paging();
    println!(
        "  [PAGING] Nodes: {} → {} (threshold: {})",
        before_paging,
        memory_graph.node_count(),
        memory_graph.paging_threshold
    );

    // ── Final Snapshot ────────────────────────────────────────────
    event_log.append(
        EventType::Snapshot,
        EventPayload::Snapshot {
            nodes_count: memory_graph.node_count(),
            edges_count: memory_graph.edge_count(),
            total_events: event_log.event_count(),
        },
    );

    event_log.append(
        EventType::CycleEnd,
        EventPayload::CycleEvent {
            cycle_number: 1,
            action: "cycle_complete".into(),
        },
    );

    // ── Phase 8: Tamper-evident Distributed Log ───────────────────
    println!("\n─── PHASE 8: Hash-Chained Log Integrity ───\n");
    dist_log.append("cycle_complete".into(), "kernel".into());
    let integrity = dist_log.verify_integrity();
    println!(
        "  [CHAIN] {} links | valid: {} | errors: {}",
        integrity.verified_events,
        integrity.valid,
        integrity.errors.len()
    );

    // ── Summary ──────────────────────────────────────────────────
    println!("\n╔══════════════════════════════════════════════╗");
    println!("║  CCOS CYCLE COMPLETE                         ║");
    println!("╠══════════════════════════════════════════════╣");
    println!(
        "║  Session:      {:<30}║",
        &session_id[..30.min(session_id.len())]
    );
    println!("║  Total Events: {:<30}║", event_log.event_count());
    println!(
        "║  Graph Nodes:  {:<30}║",
        memory_graph.node_count()
    );
    println!(
        "║  Graph Edges:  {:<30}║",
        memory_graph.edge_count()
    );
    println!(
        "║  Mutations:    {:<30}║",
        incremental_engine.total_mutations()
    );
    println!("║  Guard Status: {:<30}║", guard_status);
    println!("╚══════════════════════════════════════════════╝");

}

/// Options for `ccos analyze`.
struct AnalyzeOpts {
    path: String,
    json: bool,
    cycles: bool,
    out: Option<String>,
    dot: Option<String>,
}

impl AnalyzeOpts {
    fn parse(args: &[String]) -> Self {
        let mut opts = Self {
            path: ".".to_string(),
            json: false,
            cycles: false,
            out: None,
            dot: None,
        };
        let mut i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "--json" => opts.json = true,
                "--cycles" => opts.cycles = true,
                "--out" => {
                    i += 1;
                    opts.out = args.get(i).cloned();
                }
                "--dot" => {
                    i += 1;
                    opts.dot = args.get(i).cloned();
                }
                s if !s.starts_with("--") => opts.path = s.to_string(),
                other => eprintln!("ccos: ignoring unknown flag '{other}'"),
            }
            i += 1;
        }
        opts
    }
}

/// `ccos analyze <path> [--json] [--cycles] [--out FILE]` — ingest every `.rs`
/// file under `path` into the causal memory graph and print (or export) a
/// structural report. Returns a process exit code (0 on success).
fn run_analyze(opts: &AnalyzeOpts) -> i32 {
    let root = Path::new(&opts.path);
    if !root.exists() {
        eprintln!("ccos: path '{}' does not exist", opts.path);
        return 1;
    }

    let human = !opts.json;
    if human {
        println!("╔══════════════════════════════════════════════╗");
        println!("║  CCOS analyze — {:<29}║", truncate(&opts.path, 29));
        println!("╚══════════════════════════════════════════════╝\n");
    }

    let mut files: Vec<PathBuf> = Vec::new();
    if root.is_dir() {
        collect_rs_files(root, &mut files);
    } else if root.extension().and_then(|e| e.to_str()) == Some("rs") {
        files.push(root.to_path_buf());
    }
    files.sort();

    if files.is_empty() {
        eprintln!("ccos: no .rs files found under '{}'", opts.path);
        return 1;
    }

    let mut graph = MemoryGraph::new(0.2, 5000);
    let mut engine = IncrementalGraphEngine::new();
    let mut event_log = EventLog::new(Uuid::new_v4().to_string());
    let mut dist_log = DistributedEventLog::new();

    let mut read_errors = 0usize;
    for file in &files {
        let source = match std::fs::read_to_string(file) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("  [SKIP] {}: {}", file.display(), e);
                read_errors += 1;
                continue;
            }
        };
        let path_str = file.to_string_lossy().to_string();
        let file_hash = compute_file_hash(&source);
        let delta = engine.process_delta(&path_str, None, &source, &mut graph);
        let (m, u, s) = engine
            .get_state(&path_str)
            .map(|st| (st.modules_count, st.uses_count, st.symbols_count))
            .unwrap_or((0, 0, 0));
        event_log.append(
            EventType::Parsing,
            EventPayload::Parsing {
                file_path: path_str.clone(),
                file_hash: file_hash.clone(),
                modules_found: m,
                uses_found: u,
                symbols_found: s,
            },
        );
        dist_log.append(file_hash, "parser".into());
        if human {
            println!(
                "  [PARSE] {:<40} Δnodes:+{:<4} Δedges:+{}",
                truncate(&path_str, 40),
                delta.nodes_added,
                delta.edges_added
            );
        }
    }

    // Integrity: the graph must never hold edges to evicted/absent nodes.
    let dangling = graph.prune_dangling_edges();
    let cycles = if opts.cycles || opts.json {
        graph.find_cycles()
    } else {
        Vec::new()
    };
    let orphans = graph.orphan_nodes().len();

    if let Some(dot_path) = &opts.dot {
        match std::fs::write(dot_path, graph.to_dot()) {
            Ok(()) => eprintln!("[DOT] graph written to {dot_path}"),
            Err(e) => eprintln!("ccos: failed to write DOT to {dot_path}: {e}"),
        }
    }

    if opts.json {
        let top: Vec<_> = graph
            .get_node_scores()
            .iter()
            .take(15)
            .map(|(id, s)| serde_json::json!({ "id": id.0, "score": s }))
            .collect();
        let types: Vec<_> = graph
            .node_type_counts()
            .into_iter()
            .map(|(t, c)| serde_json::json!({ "type": t, "count": c }))
            .collect();
        let report = serde_json::json!({
            "path": opts.path,
            "files_ingested": files.len() - read_errors,
            "nodes": graph.node_count(),
            "edges": graph.edge_count(),
            "dangling_edges": dangling,
            "orphan_nodes": orphans,
            "dependency_cycles": cycles.len(),
            "node_types": types,
            "top_nodes": top,
        });
        println!("{}", serde_json::to_string_pretty(&report).unwrap());
    } else {
        println!("\n─── Graph Summary ───");
        println!("  Files ingested:  {}", files.len() - read_errors);
        println!("  Graph nodes:     {}", graph.node_count());
        println!("  Graph edges:     {}", graph.edge_count());
        println!("  Mutations:       {}", engine.total_mutations());
        println!("  Events logged:   {}", event_log.event_count());
        println!("  Dangling edges:  {dangling} (must be 0)");
        println!("  Orphan nodes:    {orphans}");

        println!("\n─── Node types ───");
        for (ty, count) in graph.node_type_counts() {
            println!("    {ty:<16} {count}");
        }

        if opts.cycles {
            println!("\n─── Dependency cycles ({}) ───", cycles.len());
            for cycle in cycles.iter().take(5) {
                let path: Vec<&str> = cycle.iter().map(|n| n.0.as_str()).collect();
                println!("    {} → {}", path.join(" → "), path.first().unwrap_or(&""));
            }
        }

        println!("\n─── Top 10 nodes by causal score ───");
        for (id, score) in graph.get_node_scores().iter().take(10) {
            println!("    {:<46} {:.4}", truncate(&id.0, 46), score);
        }

        let context = graph.select_context_window(2048);
        println!(
            "\n─── Context window (2048 tokens → {} nodes) ───",
            context.len()
        );
        for node in context.iter().take(10) {
            println!("    {:<40} ({:?})", truncate(&node.label, 40), node.node_type);
        }
    }

    if let Some(out) = &opts.out {
        let snapshot = KernelSnapshot::new(graph, event_log, dist_log);
        match snapshot.save(out) {
            Ok(()) => eprintln!("\n[SAVE] snapshot written to {out}"),
            Err(e) => {
                eprintln!("ccos: failed to save snapshot to {out}: {e}");
                return 1;
            }
        }
    }

    if dangling != 0 {
        return 1;
    }
    0
}

/// `ccos verify <snapshot.json>` — re-check a saved snapshot's integrity: the
/// hash chain must validate and the graph must hold no dangling edges.
fn run_verify(file: Option<&str>) -> i32 {
    let Some(file) = file else {
        eprintln!("usage: ccos verify <snapshot.json>");
        return 2;
    };
    let snapshot = match KernelSnapshot::load(file) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("ccos: cannot load '{file}': {e}");
            return 1;
        }
    };

    let integrity = snapshot.dist_log.verify_integrity();
    let mut graph = snapshot.graph.clone();
    let dangling = graph.prune_dangling_edges();

    println!("╔══════════════════════════════════════════════╗");
    println!("║  CCOS verify — {:<30}║", truncate(file, 30));
    println!("╚══════════════════════════════════════════════╝\n");
    println!("  Snapshot version:  {}", snapshot.version);
    println!(
        "  Graph nodes/edges: {}/{}",
        snapshot.graph.node_count(),
        snapshot.graph.edge_count()
    );
    println!("  Dangling edges:    {dangling} (must be 0)");
    println!("  Event-log events:  {}", snapshot.event_log.event_count());
    println!(
        "  Hash-chain links:  {} | valid: {}",
        integrity.verified_events, integrity.valid
    );
    for err in integrity.errors.iter().take(10) {
        println!("    ! {err}");
    }

    if integrity.valid && dangling == 0 {
        println!("\n  ✓ snapshot verified");
        0
    } else {
        println!("\n  ✗ verification FAILED");
        1
    }
}

/// `ccos replay <snapshot.json>` — deterministically replay a saved event log
/// and print the reconstructed statistics, then re-verify the hash chain.
fn run_replay(file: Option<&str>) -> i32 {
    let Some(file) = file else {
        eprintln!("usage: ccos replay <snapshot.json>");
        return 2;
    };
    let snapshot = match KernelSnapshot::load(file) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("ccos: cannot load '{file}': {e}");
            return 1;
        }
    };

    println!("╔══════════════════════════════════════════════╗");
    println!("║  CCOS replay — {:<30}║", truncate(file, 30));
    println!("╚══════════════════════════════════════════════╝\n");

    let mut replayer = EventReplayer::new();
    match snapshot.event_log.replay_deterministic(&mut replayer) {
        Ok(n) => {
            let s = &replayer.statistics;
            println!("  Replayed {n} events");
            println!(
                "  Stats: {} llm · {} parse · {} graph · {} guard · {} failures · {} cycles",
                s.llm_calls, s.parsing_events, s.graph_mutations, s.guard_checks, s.failures, s.cycles
            );
        }
        Err(e) => {
            eprintln!("ccos: replay error: {e}");
            return 1;
        }
    }

    let integrity = snapshot.dist_log.verify_integrity();
    println!("  Hash-chain valid: {}", integrity.valid);
    if integrity.valid {
        0
    } else {
        1
    }
}

/// Options for `ccos chaos`.
struct ChaosOpts {
    iters: usize,
}

impl ChaosOpts {
    fn parse(args: &[String]) -> Self {
        let mut iters = 1000usize;
        let mut i = 0;
        while i < args.len() {
            if args[i] == "--iters" {
                i += 1;
                if let Some(n) = args.get(i).and_then(|v| v.parse().ok()) {
                    iters = n;
                }
            }
            i += 1;
        }
        Self { iters }
    }
}

/// `ccos chaos [--iters N]` — drive adversarial payloads (JSON corruption,
/// hallucination, prompt injection, timeouts) through the guard and assert its
/// core invariant: the guard must *never* emit non-JSON output.
fn run_chaos(opts: &ChaosOpts) -> i32 {
    println!("╔══════════════════════════════════════════════╗");
    println!("║  CCOS chaos — {:>5} iterations                ║", opts.iters);
    println!("╚══════════════════════════════════════════════╝\n");

    let guard = GuardLayer::new(GuardConfig::default());
    let modes = [
        AdversarialMode::JsonCorruption,
        AdversarialMode::Hallucination,
        AdversarialMode::PromptInjection,
        AdversarialMode::TimeoutSimulation,
    ];

    let (mut passed, mut blocked, mut invalid_outputs) = (0u64, 0u64, 0u64);
    for i in 0..opts.iters {
        let mut engine = AdversarialEngine::with_corruption_rate(modes[i % modes.len()].clone(), 0.9);
        let corrupted = engine.corrupt("{\"action\": \"analyze\", \"ok\": true}");
        let result = guard.validate_and_sanitize(&corrupted);
        if result.passed {
            passed += 1;
        } else {
            blocked += 1;
        }
        if serde_json::from_str::<serde_json::Value>(&result.sanitized_output).is_err() {
            invalid_outputs += 1;
        }
    }

    println!("  Iterations:            {}", opts.iters);
    println!("  Guard passed:          {passed}");
    println!("  Guard blocked:         {blocked}");
    println!("  Invalid guard outputs: {invalid_outputs} (must be 0)");

    if invalid_outputs == 0 {
        println!("\n  ✓ guard never emitted invalid JSON under chaos");
        0
    } else {
        println!("\n  ✗ guard emitted invalid JSON — safety invariant violated");
        1
    }
}

/// Recursively collect `.rs` files, skipping `target/`, VCS and hidden dirs.
fn collect_rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if path.is_dir() {
            if name == "target" || name == ".git" || name.starts_with('.') {
                continue;
            }
            collect_rs_files(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("…{}", &s[s.len().saturating_sub(max - 1)..])
    }
}

fn print_help() {
    println!(
        "CCOS — Causal Context Operating System (v{})\n\n\
USAGE:\n\
    ccos [COMMAND]\n\n\
COMMANDS:\n\
    demo                       Run the built-in end-to-end kernel demo (default)\n\
    analyze <path> [flags]     Ingest all .rs files under <path> and report\n\
        --json                 Emit the report as JSON instead of text\n\
        --cycles               Detect and list dependency cycles\n\
        --dot <file>           Export the causal graph as Graphviz DOT\n\
        --out <file>           Save a full kernel snapshot (graph + logs) to <file>\n\
    verify <snapshot.json>     Re-check a saved snapshot's hash chain & integrity\n\
    replay <snapshot.json>     Deterministically replay a saved event log\n\
    chaos [--iters N]          Fuzz the guard with adversarial payloads\n\
    help, --help               Show this help\n\
    version, --version         Show the version\n\n\
ENVIRONMENT (demo only):\n\
    OLLAMA_ENDPOINT            LLM endpoint (default http://localhost:11434)\n\
    OLLAMA_MODEL               Model name (default codellama)\n\n\
EXAMPLES:\n\
    ccos analyze src --cycles\n\
    ccos analyze src --out run.json && ccos verify run.json && ccos replay run.json\n\
    ccos chaos --iters 5000\n",
        env!("CARGO_PKG_VERSION")
    );
}

fn compute_file_hash(content: &str) -> String {
    use sha2::Digest;
    use sha2::Sha256;
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    format!("{:x}", hasher.finalize())
}
