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
    demo                       Run the built-in end-to-end kernel demo (default)
    analyze <path> [flags]     Ingest all .rs files under <path> and report
        --json                 Emit the report as JSON instead of text
        --cycles               Detect and list dependency cycles
        --out <file>           Save a full kernel snapshot (graph + logs) to <file>
    verify <snapshot.json>     Re-check a saved snapshot's hash chain & integrity
    replay <snapshot.json>     Deterministically replay a saved event log
    chaos [--iters N]          Fuzz the guard with adversarial payloads
    help, --help               Show this help
    version, --version         Show the version
```

### `ccos demo`

Runs all subsystems on a small synthetic workspace: parsing → LLM + guard →
multi-model consensus → incremental delta → failure propagation → context
selection → deterministic replay → paging → hash-chain integrity. The LLM call
targets an [Ollama](https://ollama.com)-style endpoint and falls back to a
deterministic stub when none is reachable:

```bash
OLLAMA_ENDPOINT=http://localhost:11434 OLLAMA_MODEL=codellama cargo run -- demo
```

### `ccos analyze <path>`

Ingests real `.rs` files into the causal graph and prints a structural report
(node/edge counts, node-type histogram, optional dependency cycles, top nodes by
causal score, the selected context window). CCOS can analyze its own source tree:

```bash
cargo run -- analyze src --cycles          # human-readable report + cycles
cargo run -- analyze src --json            # machine-readable JSON
```

### Save → verify → replay

`analyze --out` persists a full snapshot (graph + event log + hash chain) that
the other commands consume:

```bash
cargo run -- analyze src --out run.json
cargo run -- verify run.json     # hash chain valid? dangling edges? → exit 0/1
cargo run -- replay run.json     # deterministic event-log replay + stats
```

### `ccos chaos`

Drives adversarial payloads (corruption, hallucination, injection, timeouts)
through the guard and asserts it **never** emits invalid JSON:

```bash
cargo run -- chaos --iters 5000
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

## Design paper

A full write-up of the architecture, algorithms (causal scoring, failure
propagation, deterministic paging, hash-chained log, consensus) and the
audit-driven evaluation is in [`docs/PAPER.md`](docs/PAPER.md).

## Status & limitations

This is a prototype. Known gaps (tracked in [`ROADMAP.md`](ROADMAP.md)):

- The parser is a **line-based heuristic**, not a real Rust AST (no `syn`); it
  misses multi-line declarations and nested-module bodies. *(top future-work item)*
- Replay reconstructs **statistics**, not yet full graph state (snapshots persist
  state directly via `--out`).
- The multi-model `consensus` path only does real work against a live
  Ollama-style endpoint; offline runs fall back deterministically.

Recently addressed (see `ROADMAP.md` → *Done*): unbounded edge leak, guard
prefix-bypass, non-deterministic eviction, `max_nesting_depth` enforcement,
persistence (`save`/`verify`/`replay`), and wiring `consensus` /
`distributed_event_log` / `adversarial` into the CLI.

## License

Unlicensed research prototype. Add a license before any external use.
