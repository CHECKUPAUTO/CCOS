# CCOS Architecture & Developer Guide

A practical map of the codebase for contributors. For the conceptual write-up see
[`PAPER.md`](PAPER.md); for the roadmap see [`../ROADMAP.md`](../ROADMAP.md).

## Module map

| Module                      | Responsibility | Key types |
| --------------------------- | -------------- | --------- |
| `parser`                    | Line-based extraction of modules / `use` / symbols from Rust source | `ASTParser`, `ParseResult`, `Symbol`, `SymbolKind` |
| `memory`                    | The causal graph: scoring, paging, failure propagation, analytics | `MemoryGraph`, `GraphNode`, `GraphEdge`, `NodeId`, `NodeType`, `EdgeType` |
| `incremental`               | `O(Δ)` graph maintenance on file edits | `IncrementalGraphEngine`, `DeltaMutation`, `MutationOp`, `FileState` |
| `event_log`                 | Append-only event log, deterministic replay & graph reconstruction | `EventLog`, `TraceEvent`, `EventType`, `EventPayload`, `EventReplayer`, `GraphReconstructor` |
| `distributed_event_log`     | Tamper-evident hash-chained log | `DistributedEventLog`, `HashChainLink`, `IntegrityReport` |
| `llm`                       | Async Ollama-style client + retries + fallback | `LlmClient`, `LlmConfig`, `ValidatedResponse` |
| `guard`                     | Validate/sanitize model output | `GuardLayer`, `GuardConfig`, `GuardResult` |
| `consensus`                 | Majority / confidence-weighted voting | `ConsensusEngine`, `LlmVote`, `ConsensusResult` |
| `adversarial`               | Fault injection for hardening | `AdversarialEngine`, `AdversarialMode` |
| `persist`                   | Save/load full kernel state | `KernelSnapshot` |
| `main` (bin)                | CLI dispatch + the demo | — |

## Core data structures

```
MemoryGraph
├── nodes: HashMap<NodeId, GraphNode>     // the working set
├── edges: Vec<GraphEdge>                 // invariant: edges ⊆ nodes × nodes
├── paging_threshold: f64
├── max_in_memory_nodes: usize            // paging cap
└── clock: u64                            // advanced by tick(); drives recency decay

GraphNode { id, label, content, node_type,
            base_importance, failure_relevance, recency,
            access_count, created_at, last_accessed }
```

Node ids are namespaced strings: `file:<path>`, `mod:<path>:<name>`,
`use:<path>:<path>`, `sym:<path>:<name>`, `dep:<root>`. The `incremental` engine
relies on these prefixes to evict exactly one file's subgraph.

## Invariants (and where they live)

| Invariant | Enforced by |
| --------- | ----------- |
| `edges ⊆ nodes × nodes` (no dangling edges) | `MemoryGraph::add_edge` (rejects absent endpoints), `prune_dangling_edges`, `enforce_paging` |
| Node count ≤ `max_in_memory_nodes` | `enforce_paging` (called from `upsert_node`) |
| Deterministic eviction/ordering | total order *(score, NodeId)* in `enforce_paging`, `get_node_scores`, `select_context_window` |
| Guard output is always valid JSON | `GuardLayer::validate_and_sanitize` → `fallback_response` |
| Hash chain is tamper-evident | `DistributedEventLog::compute_link_hash` / `verify_integrity` |

Regression coverage: `tests/graph_invariants.rs` (dangling-free, bounded,
deterministic), `tests/long_term_stability.rs` (10k cycles),
`tests/snapshot_differential.rs` (reproducible hashes),
`tests/llm_adversarial_test.rs` + `tests/ccos_adversarial_suite.rs` (guard).

## Control flow

### `ccos analyze <path>`

```
collect_rs_files → for each file:
    IncrementalGraphEngine::process_delta(file, None, source, &mut graph)
        → ASTParser::parse_source → update_memory_graph (upsert nodes, add edges)
        → enforce_paging (deterministic eviction)
    EventLog::append(Parsing …); DistributedEventLog::append(hash)
prune_dangling_edges → report (scores, types, cycles, orphans) → [--dot|--out]
```

### `ccos demo`

A scripted single cycle touching every subsystem: parse → LLM+guard → consensus
→ incremental delta → failure propagation → context selection → replay → paging →
hash-chain integrity.

## How to extend

- **New node/edge type** — add a variant to `NodeType` / `EdgeType` (both are
  `serde`-tagged enums); update `MemoryGraph::to_dot` color/label maps.
- **New event** — add a variant to `EventPayload`; handle it in
  `EventReplayer::handle_event` and (if it should be hashed in tests) in
  `tests/snapshot_differential.rs::event_log_hash`.
- **New CLI command** — add a match arm in `main()`, an `OptsX::parse`, and a
  `run_x(...) -> i32`; document it in `print_help`, `README.md` and this file.
- **New invariant** — add a `MemoryGraph` method that checks/repairs it and a
  test in `tests/graph_invariants.rs`.

## Build, test, lint

```bash
cargo build --all-targets
cargo test                    # 118 unit + integration tests
cargo clippy --all-targets    # warning-clean (CI denies warnings)
cargo doc --open              # rendered module docs
```

Heavier chaos/stress harnesses live in [`../scripts/`](../scripts/).
