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
| Write-ahead log       | Append-only, **hash-chained** `EventLog` + distributed log |
| Fault handling        | Failure detection → weighted propagation across edges    |
| Syscall validation    | `GuardLayer` over every LLM response                     |

## What the research found — honestly

CCOS began as a bet that organising memory by **causal regions** would retrieve
long-horizon context *better than RAG*. We built a validation harness to test that
on real bugs — and the bet did not pay off:

- On **70 real bug-fix commits** across `fd`, `bat`, `hyperfine`, causal selection
  **ties a plain lexical TF-IDF retriever** at putting a fix's files in the window,
  and **loses at a tight budget**. On real code a fix's files share vocabulary, so
  lexical similarity finds them too.
- A crash-trace pivot (seed CCOS from a panic backtrace) is **beaten by
  RAG-over-the-error-message** — Rust error messages name the cause.
- End-to-end (Phase 4: 30B model + compiler-in-the-loop), CCOS and RAG **resolve
  equally** (2/10 real `fd` bugs) — **but CCOS does so on 6.9× fewer context
  tokens** (776 vs 5366), and **4–9× fewer across 51 fixes from 3 crates**
  (model-free): it stops at the causal working set instead of padding a top-k to
  budget, and **self-calibrates with no k to tune**. **Efficiency is the one axis
  where CCOS wins** (the baseline fills the budget by construction — a tuned-k RAG
  would be sparser too; the point is CCOS bounds itself). See
  [`cargo run --example time_travel`](examples/time_travel.rs) for the
  time-travel debugging demo.

We report this rather than bury it (see [`scripts/causal_validation/`](scripts/causal_validation/)
and the [paper](docs/paper/)). It relocates CCOS's value:

> **Not a better retriever — a _frugal, deterministic, replayable, auditable_ agent
> memory.** It reaches the same result on a fraction of the context budget, and
> every cognitive operation is event-sourced and hash-chained, so you can **rewind
> an agent's exact context state** to any step and **replay it under different
> parameters** (_time-travel debugging_, `agent_session`) — a capability a
> probabilistic RAG/framework stack structurally lacks.

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
        --dot <file>           Export the causal graph as Graphviz DOT
        --out <file>           Save a full kernel snapshot (graph + logs) to <file>
    verify <snapshot.json>     Re-check a saved snapshot's hash chain & integrity
    replay <snapshot.json>     Deterministically replay a saved event log
    diff <a.json> <b.json>     Structural diff between two snapshots (+ score drift)
    failure <snap> <node-id>   Inject a fault at a node and propagate it (--depth N)
    chaos [--iters N]          Fuzz the guard with adversarial payloads

  Inspection & export:
    top <path> [--limit N]     Show the hottest nodes by causal score (--json)
    blame <snap> <node-id>     Causes (upstream) + blast radius (downstream) (--depth N)
    export <snap> [--out F]    Export the causal graph as GraphML (default ccos.graphml)

  Context Region Engine (spatial memory):
    regions <path>             Cluster the causal graph into context regions (--json)
        --activate <node-id>   Hydrate the context window for a node's region
        --metrics <node-id>    Flat-vs-region locality comparison (--radius N)
    experiment [--tasks N]     Hypothesis test: regional memory vs RAG/GraphRAG (--json)
    eval [--tasks N]           Real-LLM eval (set OPENAI_API_KEY or OLLAMA_ENDPOINT)

  CCOS v0.3 — Autonomous Context Runtime:
    scan <path>                Scan a real workspace and ingest the delta
    agents <path>              Run Coder/Reviewer/Security agents over a workspace
    benchmark [--cycles N]     Run the cycle benchmark → benchmark_report.json
    runtime <path> [--state D] Scan → schedule → agents → persist (capstone)

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
cargo run -- analyze src --dot ccos.dot    # Graphviz export → render with:
dot -Tsvg ccos.dot -o ccos.svg             #   (requires graphviz)
```

### Save → verify → replay

`analyze --out` persists a full snapshot (graph + event log + hash chain) that
the other commands consume:

```bash
cargo run -- analyze src --out run.json
cargo run -- verify run.json     # hash chain valid? dangling edges? → exit 0/1
cargo run -- replay run.json     # deterministic event-log replay + stats
```

### `ccos diff` & `ccos failure`

Inspect how a codebase evolves and how faults ripple through it:

```bash
cargo run -- analyze src   --out a.json
cargo run -- analyze tests --out b.json
cargo run -- diff a.json b.json          # nodes/edges added·removed + score movers

# Inject a fault at a node and watch it propagate across causal edges:
cargo run -- failure a.json file:src/memory.rs --depth 2
```

### `ccos chaos`

Drives adversarial payloads (corruption, hallucination, injection, timeouts)
through the guard and asserts it **never** emits invalid JSON:

```bash
cargo run -- chaos --iters 5000
```

### Inspect the causal graph — `top`, `blame`, `export`

Treat the graph like a running system: see what's *hot*, trace cause/impact, and
export it for external graph tools.

```bash
# `top` — the hottest nodes by causal score (the working set paged in first):
cargo run -- top src --limit 15
cargo run -- top src --json                 # machine-readable

# `blame` — a node's upstream causes and downstream blast radius:
cargo run -- analyze src --out run.json
cargo run -- blame run.json file:src/memory.rs --depth 4

# `export` — the causal graph as GraphML (Gephi / yEd / Cytoscape / networkx):
cargo run -- export run.json --out graph.graphml
```

`blame` follows the same edge direction as failure propagation: **causes** are
upstream (`target → source`, what the node rests on) and the **blast radius** is
downstream (`source → target`, what breaks if the node fails). See
[`docs/USAGE.md`](docs/USAGE.md) for a full command reference and walkthrough.

### Context Region Engine — `regions`

CCOS v0.3 lifts the 1-D scored graph into a **spatial map of causal regions**: an
agent no longer loads files, it hydrates a *region* (a causally coherent cluster
of files, dependencies and faults) with a temperature, a density and a dynamic
admission policy.

```bash
cargo run -- regions src                                   # cluster → region map
cargo run -- regions src --activate file:src/memory.rs     # hydrate a context window
cargo run -- regions src --metrics sym:src/memory.rs:MemoryGraph --json  # flat vs region
bash scripts/region_benchmark.sh src                       # full locality benchmark
cargo run -- experiment --tasks 800                        # hypothesis test (oracle, vs RAG/GraphRAG)
OPENAI_API_KEY=… OPENAI_BASE_URL=https://api.deepseek.com OPENAI_MODEL=… \
  cargo run -- eval --tasks 40                              # real-LLM eval (needs a model + host allowlisted)
```

On CCOS's own tree, region selection covers **97%** of a task's causal
neighbourhood (vs 35% for flat top-score selection) at **≈48% fewer tokens**, with
regions that are **95.5%** internally connected. In a deterministic, LLM-free
simulation (`ccos experiment`), **lexical RAG solves 0%** of cross-file causal
tasks while structure-aware methods solve 100%; and under a **misleading query**,
every lexically-seeded method (RAG, GraphRAG, *and* an ablation of CCOS that
trusts the query) collapses to **0%** while only the **workspace-anchored region
survives** — isolating the *anchor* as CCOS's differentiator. See
[`docs/context_regions.md`](docs/context_regions.md) and the research paper in
[`docs/paper/`](docs/paper/).

### CCOS v0.3 — Autonomous Context Runtime

v0.3 scans a real workspace, pages its context (HOT/WARM/COLD), runs specialized
agents, and persists the runtime so it resumes after a restart. The `runtime`
command wires all of it together:

```bash
cargo run -- scan src                      # async FS scan → causal graph
cargo run -- agents src                    # Coder/Reviewer/Security over the code
cargo run -- benchmark --cycles 100000     # → benchmark_report.json
cargo run -- runtime src --state data      # scan → schedule → agents → persist
```

See [`CCOS_v0.3_REPORT.md`](CCOS_v0.3_REPORT.md) for the full v0.3 report.

### Agent memory interface — `ccos memory` / `ccos mcp`

Use CCOS as an agent's **external working memory** — ingest source, signal
failures, recall a bounded causal window, verify the hash chain — over two
transports on the same documented façade:

```bash
# Any language, via stdio JSON-Lines (checkpoint-backed):
printf '%s\n' '{"op":"ingest","uri":"src/db.rs","source":"pub fn query() {}"}' \
              '{"op":"recall","strategy":"around","anchor":"file:src/db.rs","budget":2048}' \
  | cargo run -- memory --path workspace.ccos

# Any MCP-compatible agent, via stdio JSON-RPC 2.0 (live event-sourced session):
cargo run -- mcp        # point a client's stdio transport at this: {"command":"ccos","args":["mcp"]}
```

`ccos mcp` speaks the standard MCP handshake and advertises six tools (`ingest`,
`recall`, `signal_failure`, `page_fault`, `stats`, `verify`) — dependency-free,
backed by the event-sourced session, so it stays replayable. Full contract, tool
schemas and a client-config snippet: [`docs/MEMORY_INTERFACE.md`](docs/MEMORY_INTERFACE.md).

## Testing

```bash
cargo test          # 212 unit + integration tests
cargo clippy --all-targets   # lint-clean
cargo test -- --ignored      # opt-in: 1,000,000-cycle long-stability run
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
- **Tamper-evidence**: both the primary `EventLog` (canonical hash chain) and
  the `DistributedEventLog` detect any payload mutation, reorder, insertion or
  deletion (`src/event_log.rs`, `src/distributed_event_log.rs`).

## Documentation

- [`docs/USAGE.md`](docs/USAGE.md) — **command reference & walkthroughs**: every
  command with example invocations and output, an end-to-end "analyze a real
  project" tour, the snapshot/replay workflow, and a troubleshooting FAQ.
- [`docs/MEMORY_INTERFACE.md`](docs/MEMORY_INTERFACE.md) — the **external-memory
  interface**: the documented façade an agent uses to treat CCOS as working memory,
  plus its two transports (`ccos memory` stdio JSON, and the `ccos mcp` MCP server).
- [`docs/context_regions.md`](docs/context_regions.md) — the **Context Region
  Engine** (v0.3): spatial memory model, formal region definition, dynamic
  admission policy, determinism, and measured locality.
- [`docs/paper/`](docs/paper/) — **research paper** (arXiv LaTeX): *Causal Context
  Regions* — formal model, determinism proof, measured locality, and a falsifiable
  comparison protocol vs RAG / GraphRAG / MemGPT / LangGraph.
- [`docs/PAPER.md`](docs/PAPER.md) — design paper: architecture, algorithms
  (causal scoring, failure propagation, deterministic paging, hash-chained log,
  consensus) and the audit-driven evaluation.
- [`docs/BIBLIOGRAPHY.md`](docs/BIBLIOGRAPHY.md) — annotated reading list (~60
  verified papers across 12 themes) mapping the research literature to CCOS's
  modules.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — developer guide: module map,
  data structures, invariants, control flow, and how to extend the kernel.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — local dev setup, the CI jobs, coding
  conventions, and how to add a node type / event / CLI command.
- [`CHANGELOG.md`](CHANGELOG.md) — notable changes per version.
- [`CCOS_v0.3_REPORT.md`](CCOS_v0.3_REPORT.md) — v0.3 Autonomous Context Runtime
  report: new modules, tests, performance, and limitations.
- `cargo doc --open` — rendered API docs (every module has rustdoc).

## Status & limitations

This is a prototype. Known gaps (tracked in [`ROADMAP.md`](ROADMAP.md)):

- The parser defaults to a **line-based heuristic** (zero dependencies). Build
  with `--features syn-parser` for a real `syn` AST that resolves nested-module
  bodies, multi-line signatures, grouped `use` and impl methods; the heuristic
  stays as the fallback when the feature is off or a file does not parse. The
  heuristic strips `//` and inline `/* … */` comments but does not track
  multi-line block comments.
- Edges capture containment/dependency, **not** call graphs or data flow.
- The multi-model `consensus` path only does real work against a live
  Ollama-style endpoint; offline runs fall back deterministically.

Recently addressed (see `ROADMAP.md` → *Done*): unbounded edge leak, guard
prefix-bypass, non-deterministic eviction, `max_nesting_depth` enforcement,
persistence (`save`/`verify`/`replay`), and wiring `consensus` /
`distributed_event_log` / `adversarial` into the CLI.

## License

Unlicensed research prototype. Add a license before any external use.
