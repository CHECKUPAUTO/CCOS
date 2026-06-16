mod commands_demo;
mod commands_runtime;

use ccos::adversarial::{AdversarialEngine, AdversarialMode};
use ccos::distributed_event_log::DistributedEventLog;
use ccos::event_log::{EventLog, EventPayload, EventReplayer, EventType, GraphReconstructor};
use ccos::guard::{GuardConfig, GuardLayer};
use ccos::incremental::IncrementalGraphEngine;
use ccos::memory::{MemoryGraph, NodeId};
use ccos::persist::KernelSnapshot;
use ccos::query;
use ccos::util::sha256_hex;
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
            commands_demo::run_demo().await;
            0
        }
        "analyze" => run_analyze(&AnalyzeOpts::parse(rest)),
        "verify" => run_verify(rest.first().map(String::as_str)),
        "replay" => run_replay(rest.first().map(String::as_str)),
        "diff" => run_diff(
            rest.first().map(String::as_str),
            rest.get(1).map(String::as_str),
        ),
        "failure" => run_failure(&FailureOpts::parse(rest)),
        "chaos" => run_chaos(&ChaosOpts::parse(rest)),
        "top" => run_top(&TopOpts::parse(rest)),
        "blame" => run_blame(&BlameOpts::parse(rest)),
        "export" => run_export(&ExportOpts::parse(rest)),
        // ── CCOS v0.3 — Autonomous Context Runtime ──────────────────
        "scan" => commands_runtime::run_scan(rest).await,
        "agents" => commands_runtime::run_agents(rest).await,
        "benchmark" => commands_runtime::run_benchmark(rest),
        "runtime" => commands_runtime::run_runtime(rest).await,
        other => {
            eprintln!("ccos: unknown command '{other}'\n");
            print_help();
            2
        }
    };
    std::process::exit(code);
}

/// Options for `ccos analyze`.
struct AnalyzeOpts {
    path: String,
    json: bool,
    cycles: bool,
    out: Option<String>,
    dot: Option<String>,
    max_nodes: usize,
    budget: usize,
}

impl AnalyzeOpts {
    fn parse(args: &[String]) -> Self {
        let mut opts = Self {
            path: ".".to_string(),
            json: false,
            cycles: false,
            out: None,
            dot: None,
            max_nodes: 5000,
            budget: 2048,
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
                "--max-nodes" => {
                    i += 1;
                    if let Some(n) = args.get(i).and_then(|v| v.parse().ok()) {
                        opts.max_nodes = n;
                    }
                }
                "--budget" => {
                    i += 1;
                    if let Some(n) = args.get(i).and_then(|v| v.parse().ok()) {
                        opts.budget = n;
                    }
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

    let mut graph = MemoryGraph::new(0.2, opts.max_nodes);
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
        let file_hash = sha256_hex(&source);
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

        let context = graph.select_context_window(opts.budget);
        println!(
            "\n─── Context window ({} tokens → {} nodes) ───",
            opts.budget,
            context.len()
        );
        for node in context.iter().take(10) {
            println!(
                "    {:<40} ({:?})",
                truncate(&node.label, 40),
                node.node_type
            );
        }
    }

    if let Some(out) = &opts.out {
        // Record the graph as replayable events so `ccos replay` can rebuild it
        // from the log alone (event-sourcing round-trip), then snapshot.
        event_log.record_graph(&graph);
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
    let log_integrity = snapshot.event_log.verify_integrity();
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
        "  Dist-log chain:    {} links | valid: {}",
        integrity.verified_events, integrity.valid
    );
    for err in integrity.errors.iter().take(10) {
        println!("    ! {err}");
    }
    println!(
        "  Event-log chain:   {} verified | valid: {}",
        log_integrity.verified_events, log_integrity.valid
    );
    for err in log_integrity.errors.iter().take(10) {
        println!("    ! {err}");
    }

    if integrity.valid && log_integrity.valid && dangling == 0 {
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
                s.llm_calls,
                s.parsing_events,
                s.graph_mutations,
                s.guard_checks,
                s.failures,
                s.cycles
            );
        }
        Err(e) => {
            eprintln!("ccos: replay error: {e}");
            return 1;
        }
    }

    // Rebuild the graph purely from the log and check it matches the snapshot.
    let mut recon = GraphReconstructor::new();
    let _ = snapshot.event_log.replay_deterministic(&mut recon);
    if recon.nodes_built > 0 {
        let matches = recon.graph.node_count() == snapshot.graph.node_count()
            && recon.graph.edge_count() == snapshot.graph.edge_count();
        println!(
            "  Reconstructed graph: {} nodes / {} edges (matches snapshot: {})",
            recon.graph.node_count(),
            recon.graph.edge_count(),
            matches
        );
    }

    let integrity = snapshot.dist_log.verify_integrity();
    let log_integrity = snapshot.event_log.verify_integrity();
    println!(
        "  Hash-chain valid: {} (dist-log) · {} (event-log, {} links)",
        integrity.valid, log_integrity.valid, log_integrity.verified_events
    );
    if integrity.valid && log_integrity.valid {
        0
    } else {
        1
    }
}

/// `ccos diff <a.json> <b.json>` — structural difference between two saved
/// snapshots: nodes/edges added & removed, plus the biggest causal-score movers.
fn run_diff(a: Option<&str>, b: Option<&str>) -> i32 {
    let (Some(a_path), Some(b_path)) = (a, b) else {
        eprintln!("usage: ccos diff <old-snapshot.json> <new-snapshot.json>");
        return 2;
    };
    let load = |p: &str| KernelSnapshot::load(p).map_err(|e| format!("cannot load '{p}': {e}"));
    let (snap_a, snap_b) = match (load(a_path), load(b_path)) {
        (Ok(a), Ok(b)) => (a, b),
        (Err(e), _) | (_, Err(e)) => {
            eprintln!("ccos: {e}");
            return 1;
        }
    };

    let d = snap_a.graph.diff(&snap_b.graph);

    println!("╔══════════════════════════════════════════════╗");
    println!("║  CCOS diff                                   ║");
    println!("╚══════════════════════════════════════════════╝\n");
    println!("  {}  →  {}", truncate(a_path, 18), truncate(b_path, 18));
    println!(
        "  Nodes:  +{} / -{}  ({} common)",
        d.nodes_added.len(),
        d.nodes_removed.len(),
        d.common_nodes
    );
    println!("  Edges:  +{} / -{}", d.edges_added, d.edges_removed);

    if !d.nodes_added.is_empty() {
        println!("\n  Added nodes:");
        for id in d.nodes_added.iter().take(10) {
            println!("    + {}", truncate(&id.0, 50));
        }
    }
    if !d.nodes_removed.is_empty() {
        println!("\n  Removed nodes:");
        for id in d.nodes_removed.iter().take(10) {
            println!("    - {}", truncate(&id.0, 50));
        }
    }

    // Causal-score drift among nodes present in both snapshots.
    let mut movers: Vec<(String, f64)> = Vec::new();
    for (id, node_b) in &snap_b.graph.nodes {
        if let Some(node_a) = snap_a.graph.nodes.get(id) {
            let drift =
                snap_b.graph.compute_node_score(node_b) - snap_a.graph.compute_node_score(node_a);
            if drift.abs() > 1e-9 {
                movers.push((id.0.clone(), drift));
            }
        }
    }
    movers.sort_by(|x, y| {
        y.1.abs()
            .partial_cmp(&x.1.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| x.0.cmp(&y.0))
    });
    if !movers.is_empty() {
        println!("\n  Top causal-score movers:");
        for (id, drift) in movers.iter().take(10) {
            println!("    {:+.4}  {}", drift, truncate(id, 44));
        }
    }
    0
}

/// Options for `ccos failure`.
struct FailureOpts {
    snapshot: Option<String>,
    node: Option<String>,
    depth: u32,
}

impl FailureOpts {
    fn parse(args: &[String]) -> Self {
        let (mut snapshot, mut node, mut depth) = (None, None, 3u32);
        let mut positional = 0;
        let mut i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "--depth" => {
                    i += 1;
                    if let Some(n) = args.get(i).and_then(|v| v.parse().ok()) {
                        depth = n;
                    }
                }
                s if !s.starts_with("--") => {
                    if positional == 0 {
                        snapshot = Some(s.to_string());
                    } else {
                        node = Some(s.to_string());
                    }
                    positional += 1;
                }
                other => eprintln!("ccos: ignoring unknown flag '{other}'"),
            }
            i += 1;
        }
        Self {
            snapshot,
            node,
            depth,
        }
    }
}

/// `ccos failure <snapshot.json> <node-id> [--depth N]` — inject a fault at a
/// node and propagate it across the causal graph, reporting the affected
/// neighborhood ranked by resulting failure relevance.
fn run_failure(opts: &FailureOpts) -> i32 {
    let (Some(file), Some(node_id)) = (opts.snapshot.as_deref(), opts.node.as_deref()) else {
        eprintln!("usage: ccos failure <snapshot.json> <node-id> [--depth N]");
        return 2;
    };
    let snapshot = match KernelSnapshot::load(file) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("ccos: cannot load '{file}': {e}");
            return 1;
        }
    };
    let mut graph = snapshot.graph;
    let origin = NodeId(node_id.to_string());
    if !graph.nodes.contains_key(&origin) {
        eprintln!(
            "ccos: node '{node_id}' not found ({} nodes). List ids with `ccos analyze <path> --json`.",
            graph.node_count()
        );
        return 1;
    }

    println!("╔══════════════════════════════════════════════╗");
    println!("║  CCOS failure propagation                    ║");
    println!("╚══════════════════════════════════════════════╝\n");

    graph.set_failure_relevance(&origin, 0.95);
    graph.propagate_failure(&origin, 0, opts.depth);

    let mut affected: Vec<(String, f64)> = graph
        .nodes
        .iter()
        .filter(|(id, n)| **id != origin && n.failure_relevance > 0.0)
        .map(|(id, n)| (id.0.clone(), n.failure_relevance))
        .collect();
    affected.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });

    println!("  Origin:   {}", truncate(node_id, 40));
    println!("  Severity: 0.95   depth: {}", opts.depth);
    println!("  Affected: {} nodes", affected.len());
    if !affected.is_empty() {
        println!("\n  Causal neighborhood (by failure relevance):");
        for (id, fr) in affected.iter().take(15) {
            println!("    {:.3}  {}", fr, truncate(id, 46));
        }
    }
    0
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
    println!(
        "║  CCOS chaos — {:>5} iterations                ║",
        opts.iters
    );
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
        let mut engine =
            AdversarialEngine::with_corruption_rate(modes[i % modes.len()].clone(), 0.9);
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

/// Ingest every `.rs` file under `path` into a fresh memory graph (the same way
/// `analyze` does, minus the event log and reporting). Shared by `top`.
fn build_graph_from_path(path: &str, max_nodes: usize) -> Result<MemoryGraph, String> {
    let root = Path::new(path);
    if !root.exists() {
        return Err(format!("path '{path}' does not exist"));
    }
    let mut files: Vec<PathBuf> = Vec::new();
    if root.is_dir() {
        collect_rs_files(root, &mut files);
    } else if root.extension().and_then(|e| e.to_str()) == Some("rs") {
        files.push(root.to_path_buf());
    }
    files.sort();
    if files.is_empty() {
        return Err(format!("no .rs files found under '{path}'"));
    }

    let mut graph = MemoryGraph::new(0.2, max_nodes);
    let mut engine = IncrementalGraphEngine::new();
    for file in &files {
        if let Ok(source) = std::fs::read_to_string(file) {
            let path_str = file.to_string_lossy().to_string();
            engine.process_delta(&path_str, None, &source, &mut graph);
        }
    }
    graph.prune_dangling_edges();
    Ok(graph)
}

/// Options for `ccos top`.
struct TopOpts {
    path: String,
    limit: usize,
    json: bool,
    max_nodes: usize,
}

impl TopOpts {
    fn parse(args: &[String]) -> Self {
        let mut opts = Self {
            path: ".".to_string(),
            limit: 20,
            json: false,
            max_nodes: 5000,
        };
        let mut i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "--json" => opts.json = true,
                "--limit" => {
                    i += 1;
                    if let Some(n) = args.get(i).and_then(|v| v.parse().ok()) {
                        opts.limit = n;
                    }
                }
                "--max-nodes" => {
                    i += 1;
                    if let Some(n) = args.get(i).and_then(|v| v.parse().ok()) {
                        opts.max_nodes = n;
                    }
                }
                s if !s.starts_with("--") => opts.path = s.to_string(),
                other => eprintln!("ccos: ignoring unknown flag '{other}'"),
            }
            i += 1;
        }
        opts
    }
}

/// `ccos top <path> [--limit N] [--json]` — ingest `path` and print the hottest
/// nodes by causal score: the working set the kernel would page in first.
fn run_top(opts: &TopOpts) -> i32 {
    let graph = match build_graph_from_path(&opts.path, opts.max_nodes) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("ccos: {e}");
            return 1;
        }
    };
    let hot = query::hot_set(&graph, opts.limit);

    if opts.json {
        let rows: Vec<_> = hot
            .iter()
            .map(|(id, s)| serde_json::json!({ "id": id.0, "score": s }))
            .collect();
        let report = serde_json::json!({
            "path": opts.path,
            "nodes": graph.node_count(),
            "edges": graph.edge_count(),
            "top": rows,
        });
        println!("{}", serde_json::to_string_pretty(&report).unwrap());
        return 0;
    }

    println!("╔══════════════════════════════════════════════╗");
    println!("║  CCOS top — {:<33}║", truncate(&opts.path, 33));
    println!("╚══════════════════════════════════════════════╝\n");
    println!(
        "  {} nodes / {} edges — top {}:\n",
        graph.node_count(),
        graph.edge_count(),
        hot.len()
    );
    println!("    {:>7}  {:<8}  NODE", "SCORE", "TYPE");
    for (id, score) in &hot {
        let ty = graph
            .nodes
            .get(id)
            .map(|n| format!("{:?}", n.node_type))
            .unwrap_or_else(|| "?".into());
        println!(
            "    {:>7.4}  {:<8}  {}",
            score,
            truncate(&ty, 8),
            truncate(&id.0, 44)
        );
    }
    0
}

/// Options for `ccos blame`.
struct BlameOpts {
    snapshot: Option<String>,
    node: Option<String>,
    depth: u32,
    json: bool,
}

impl BlameOpts {
    fn parse(args: &[String]) -> Self {
        let (mut snapshot, mut node, mut depth, mut json) = (None, None, 3u32, false);
        let mut positional = 0;
        let mut i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "--json" => json = true,
                "--depth" => {
                    i += 1;
                    if let Some(n) = args.get(i).and_then(|v| v.parse().ok()) {
                        depth = n;
                    }
                }
                s if !s.starts_with("--") => {
                    if positional == 0 {
                        snapshot = Some(s.to_string());
                    } else {
                        node = Some(s.to_string());
                    }
                    positional += 1;
                }
                other => eprintln!("ccos: ignoring unknown flag '{other}'"),
            }
            i += 1;
        }
        Self {
            snapshot,
            node,
            depth,
            json,
        }
    }
}

/// `ccos blame <snapshot.json> <node-id> [--depth N]` — show a node's upstream
/// causes (what it rests on) and downstream blast radius (what breaks with it).
fn run_blame(opts: &BlameOpts) -> i32 {
    let (Some(file), Some(node_id)) = (opts.snapshot.as_deref(), opts.node.as_deref()) else {
        eprintln!("usage: ccos blame <snapshot.json> <node-id> [--depth N]");
        return 2;
    };
    let snapshot = match KernelSnapshot::load(file) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("ccos: cannot load '{file}': {e}");
            return 1;
        }
    };
    let graph = snapshot.graph;
    let origin = NodeId(node_id.to_string());
    if !graph.nodes.contains_key(&origin) {
        eprintln!(
            "ccos: node '{node_id}' not found ({} nodes). List ids with `ccos analyze <path> --json`.",
            graph.node_count()
        );
        return 1;
    }

    let causes = query::source_set(&graph, &origin, opts.depth);
    let impact = query::impact_set(&graph, &origin, opts.depth);

    if opts.json {
        let to_rows = |v: &[query::Reached]| {
            v.iter()
                .map(|r| {
                    serde_json::json!({ "id": r.id.0, "distance": r.distance, "score": r.score })
                })
                .collect::<Vec<_>>()
        };
        let report = serde_json::json!({
            "node": node_id,
            "depth": opts.depth,
            "causes": to_rows(&causes),
            "impact": to_rows(&impact),
        });
        println!("{}", serde_json::to_string_pretty(&report).unwrap());
        return 0;
    }

    println!("╔══════════════════════════════════════════════╗");
    println!("║  CCOS blame                                  ║");
    println!("╚══════════════════════════════════════════════╝\n");
    println!("  Node:  {}", truncate(node_id, 40));
    println!("  Depth: {}\n", opts.depth);

    println!(
        "  ── Causes (upstream — what it rests on): {} ──",
        causes.len()
    );
    for r in causes.iter().take(15) {
        println!(
            "    d{}  {:.4}  {}",
            r.distance,
            r.score,
            truncate(&r.id.0, 42)
        );
    }
    println!(
        "\n  ── Blast radius (downstream — what breaks with it): {} ──",
        impact.len()
    );
    for r in impact.iter().take(15) {
        println!(
            "    d{}  {:.4}  {}",
            r.distance,
            r.score,
            truncate(&r.id.0, 42)
        );
    }
    0
}

/// Options for `ccos export`.
struct ExportOpts {
    snapshot: Option<String>,
    out: String,
}

impl ExportOpts {
    fn parse(args: &[String]) -> Self {
        let (mut snapshot, mut out) = (None, "ccos.graphml".to_string());
        let mut i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "--out" => {
                    i += 1;
                    if let Some(v) = args.get(i) {
                        out = v.clone();
                    }
                }
                // `--format graphml` is accepted for forward-compatibility; GraphML
                // is currently the only target.
                "--format" => {
                    i += 1;
                    if let Some(fmt) = args.get(i) {
                        if fmt != "graphml" {
                            eprintln!("ccos: unknown export format '{fmt}', using graphml");
                        }
                    }
                }
                s if !s.starts_with("--") => snapshot = Some(s.to_string()),
                other => eprintln!("ccos: ignoring unknown flag '{other}'"),
            }
            i += 1;
        }
        Self { snapshot, out }
    }
}

/// `ccos export <snapshot.json> [--out FILE]` — export the snapshot's causal
/// graph as GraphML for Gephi / yEd / Cytoscape / networkx.
fn run_export(opts: &ExportOpts) -> i32 {
    let Some(file) = opts.snapshot.as_deref() else {
        eprintln!("usage: ccos export <snapshot.json> [--out FILE]");
        return 2;
    };
    let snapshot = match KernelSnapshot::load(file) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("ccos: cannot load '{file}': {e}");
            return 1;
        }
    };
    let graphml = query::to_graphml(&snapshot.graph);
    match std::fs::write(&opts.out, graphml) {
        Ok(()) => {
            println!(
                "[EXPORT] {} nodes / {} edges → {} (GraphML)",
                snapshot.graph.node_count(),
                snapshot.graph.edge_count(),
                opts.out
            );
            0
        }
        Err(e) => {
            eprintln!("ccos: failed to write '{}': {e}", opts.out);
            1
        }
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
        --max-nodes <N>        Paging cap (default 5000)\n\
        --budget <N>           Context-window token budget (default 2048)\n\
    verify <snapshot.json>     Re-check a saved snapshot's hash chain & integrity\n\
    replay <snapshot.json>     Deterministically replay a saved event log\n\
    diff <a.json> <b.json>     Structural diff between two snapshots (+ score drift)\n\
    failure <snap> <node-id>   Inject a fault at a node and propagate it (--depth N)\n\
    chaos [--iters N]          Fuzz the guard with adversarial payloads\n\
\n\
  Inspection & export:\n\
    top <path> [--limit N]     Show the hottest nodes by causal score (--json)\n\
    blame <snap> <node-id>     Causes (upstream) + blast radius (downstream) (--depth N, --json)\n\
    export <snap> [--out F]    Export the causal graph as GraphML (default ccos.graphml)\n\
\n\
  CCOS v0.3 — Autonomous Context Runtime:\n\
    scan <path>                Scan a real workspace and ingest the delta\n\
    agents <path>              Run Coder/Reviewer/Security agents over a workspace\n\
    benchmark [--cycles N]     Run the cycle benchmark → benchmark_report.json\n\
                               (also: --cap N, --out FILE)\n\
    runtime <path> [--state D] Scan → schedule → agents → persist (capstone)\n\
\n\
    help, --help               Show this help\n\
    version, --version         Show the version\n\n\
ENVIRONMENT (demo only):\n\
    OLLAMA_ENDPOINT            LLM endpoint (default http://localhost:11434)\n\
    OLLAMA_MODEL               Model name (default codellama)\n\n\
EXAMPLES:\n\
    ccos analyze src --cycles\n\
    ccos analyze src --out run.json && ccos verify run.json && ccos replay run.json\n\
    ccos top src --limit 15\n\
    ccos blame run.json file:src/memory.rs --depth 4\n\
    ccos export run.json --out graph.graphml\n\
    ccos runtime src --state data\n\
    ccos benchmark --cycles 100000\n",
        env!("CARGO_PKG_VERSION")
    );
}
