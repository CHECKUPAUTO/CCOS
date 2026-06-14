# CCOS — Causal Context Operating System

> An experimental kernel that manages an LLM's working context the way an OS
> manages memory: parse code into a **causal graph**, score and **page** nodes
> in/out of a bounded context window, and record every transition in a
> **deterministically replayable** event log.

CCOS is a research prototype written in Rust (edition 2021). It is **not** a
production system — see [Status & limitations](#status--limitations).

---

## Why

Coding agents drown in context. CCOS reframes context management as an operating
-system problem:

| OS concept            | CCOS analogue                                             |
| --------------------- | -------------------------------------------------------- |
| Pages / working set   | Graph nodes (files, modules, symbols, imports)           |
| RAM ↔ VRAM paging     | `select_context_window()` + `enforce_paging()`           |
| Process scheduling    | Causal scoring (importance · failure · recency · access) |
| Write-ahead log       | Append-only `EventLog` + hash-chained distributed log    |
| Fault handling        | Failure detection → weighted propagation across edges    |
| Syscall validation    | `GuardLayer` over every LLM response                     |

## Architecture

```
            ┌─────────────┐   register/Δ   ┌──────────────────────────┐
 .rs files →│   parser    │───────────────▶│  IncrementalGraphEngine   │
            └─────────────┘                └────────────┬─────────────┘
                                                         │ O(Δ) mutations
                                                         ▼
   ┌─────────┐  validate   ┌─────────┐          ┌──────────────────┐
   │   llm   │────────────▶│  guard  │          │   MemoryGraph    │
   └─────────┘  sanitize   └─────────┘          │  scoring/paging/  │
        ▲                                        │  failure-propag.  │
        │                                        └────────┬─────────┘
   consensus / adversarial (multi-model + fault injection)│ snapshots
                                                          ▼
                              ┌──────────────────────────────────────┐
                              │ EventLog  +  DistributedEventLog       │
                              │ (deterministic + hash-chained replay)  │
                              └──────────────────────────────────────┘
```

Module reference: run `cargo doc --open` (each module has rustdoc), or see
[`src/lib.rs`](src/lib.rs).

## Build

Requires a recent stable Rust toolchain.

```bash
cargo build --release
```

## Usage (CLI)

```
ccos [COMMAND]

COMMANDS:
    demo             Run the built-in end-to-end kernel demo (default)
    analyze <path>   Ingest all .rs files under <path> and print a graph report
    help, --help     Show this help
    version          Show the version
```

### `ccos demo`

Runs all subsystems on a small synthetic workspace: parsing → LLM + guard →
incremental delta → failure propagation → context selection → deterministic
replay → paging. The LLM call targets an [Ollama](https://ollama.com)-style
endpoint and falls back to a deterministic stub when none is reachable:

```bash
OLLAMA_ENDPOINT=http://localhost:11434 OLLAMA_MODEL=codellama cargo run -- demo
```

### `ccos analyze <path>`

Ingests real `.rs` files into the causal graph and prints a structural report
(node/edge counts, top nodes by causal score, the selected context window).
CCOS can analyze its own source tree:

```bash
cargo run -- analyze src
```

```
─── Graph Summary ───
  Files ingested:  11
  Graph nodes:     308
  Graph edges:     338
  Dangling edges:  0 (must be 0)

─── Top 10 nodes by causal score ───
    dep:std                                        0.5104
    dep:serde                                      0.4790
    ...
```

## Testing

```bash
cargo test          # 111 unit + integration tests
cargo clippy --all-targets   # lint-clean
```

Heavier stress/chaos harnesses live in [`scripts/`](scripts/) (multi-day chaos,
100k-cycle stress, replay-consistency, memory-pressure, graph fuzzing).

### Key invariants under test

- **No dangling edges**: the graph always satisfies `edges ⊆ nodes × nodes`,
  even under aggressive paging (`tests/graph_invariants.rs`).
- **Bounded growth**: node *and* edge counts stay bounded over 10k+ mutation
  cycles — no linear/quadratic creep (`tests/long_term_stability.rs`).
- **Deterministic eviction**: identical builds evict identically, so snapshot
  hashes and replays are reproducible (`tests/snapshot_differential.rs`).
- **Guard safety**: every guard output is valid JSON; injection/hallucination
  payloads are rejected (`tests/llm_adversarial_test.rs`,
  `tests/ccos_adversarial_suite.rs`).
- **Tamper-evidence**: the hash-chained log detects any mutation
  (`src/distributed_event_log.rs`).

## Status & limitations

This is a prototype. Known gaps (tracked in [`ROADMAP.md`](ROADMAP.md)):

- The parser is a **line-based heuristic**, not a real Rust AST (no `syn`); it
  misses multi-line declarations and nested-module bodies.
- `consensus`, `adversarial`, and `distributed_event_log` are implemented and
  tested but **not yet wired into the kernel runtime** (`main.rs`).
- `GuardConfig::max_nesting_depth` is defined but not enforced.
- No persistence: graph/log state lives only in memory for the run.

## License

Unlicensed research prototype. Add a license before any external use.
