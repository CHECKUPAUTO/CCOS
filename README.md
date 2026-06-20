# CCOS — Causal Context Operating System

> A local, deterministic **cognitive MMU** for LLM coding agents: it keeps the
> *right* code in the agent's context window, and makes the agent's attention
> **auditable** when a long-horizon session drifts.

CCOS treats an agent's working memory the way a CPU's MMU treats RAM. It maps the
side effects of a coding session — files read, compiler/test failures, panics —
into a causal graph, pages that graph against a token budget, and records every
transition in a deterministic, replayable, hash-chained log. It exposes a
self-bounding, linearised context window a host can inject into its prompt, plus a
post-mortem debugger to rewind to exactly where the agent's attention went off the
rails.

**What it is, honestly.** CCOS is *not* a better retriever — on real bugs a lexical
baseline matches it at putting a fix's files in the window. Its value is the axis a
probabilistic RAG/framework stack structurally lacks: it reaches the same result on a
fraction of the context budget (it self-bounds at the causal region, with no top-k to
tune), and every step is deterministic, replayable and auditable. The research story
behind that conclusion — the original hypothesis, the bug-mining harness, the
measurements against RAG/GraphRAG — lives in [`docs/paper/`](docs/paper/) (six
languages). CCOS is a research prototype in Rust (edition 2021); see
[Status & limitations](#status--limitations).

## The cognitive-MMU cycle

```
  [Host / IDE] ◄──── (linearised, bounded context) ──────┐
       │                                                  │
       ▼  (optional PostToolUse hook — docs/SELF_ANALYSIS) │
  [page fault / ingest]                                   │
       │                                                  │
       ▼                                                  │
  [CCOS kernel] ──► [causal graph + scoring / paging] ────┘
       │
       ▼  (on every state change)
  [storage] ──► workspace.ccos       (snapshot, shared with `ccos memory`)
            └─► workspace.ccos.oplog  (compacted op-log → time-travel)
```

## Capabilities

### 1. Demand paging by causal pressure

- **Self-calibration.** CCOS assembles a token-bounded working set from node
  activation in the causal graph — typically **700–1600 context tokens** where
  budget-filling lexical RAG uses **~5000–6000** on the same single-file fixes (a
  measured **4–9× reduction**). It stops at the causal region instead of padding a
  top-k to budget; there is no `k` to tune. Efficiency is the *one* axis CCOS wins —
  a carefully tuned-k RAG would also be sparser; the point is that CCOS bounds
  *itself*.
- **Context page fault.** Feed `cargo test` / panic output back in: CCOS parses the
  faulting source locations from the trace, injects failure pressure on those files,
  and re-pages a refreshed window for the next attempt. (It targets the files the
  trace names — often the *symptom* site; the post-mortem tools below let you check
  whether the deep cause actually entered the window.)

### 2. Transactional, replayable storage

- **Hybrid event-sourcing.** A structural snapshot (`.ccos`) plus an operation log
  (`.oplog`), persisted **durably** on every change (`fsync` + atomic rename, so a
  crash never leaves a half-written file); the snapshot format is shared with the
  `ccos memory` transport.
- **Deterministic compaction.** Older ops fold into the baseline past a threshold
  (`CCOS_OPLOG_MAX` / `CCOS_OPLOG_KEEP`), keeping the op-log bounded for long-running
  sessions (e.g. on a Jetson) while preserving **absolute step indices** — so
  time-travel stays index-stable across a compaction.
- **Cross-restart resilience.** Reopen a workspace and the cognitive timeline is
  restored: replay and time-travel span restarts (up to the compaction floor), even
  after the daemon was killed. A stale log that no longer reproduces the snapshot
  self-heals to the snapshot — the memory is never corrupted.

### 3. Standard MCP transport

- **Stdio JSON-RPC server.** Native, synchronous, zero-network integration with any
  MCP-compatible host (e.g. Claude Code). Eight tools: `ingest`, `recall`,
  `signal_failure`, `page_fault`, `stats`, `verify`, `timeline`, `recall_what_if`.
- **Dynamic resources.** `ccos://session/context` exposes the self-bounding working
  set for the host to drop into its system prompt; `ccos://session/timeline` exposes
  the cognitive journal.

### 4. Post-mortem debugging (`ccos postmortem`)

- **Time-travel REPL.** Step a cursor backward/forward through the agent's recorded
  memory; recall the window as it stood at any past step (deterministic replay).
- **Diff vs energy views.** Contrast which *files* entered/left the working set
  (`diff A B`) against the node-level *causal-heat migration* through the graph
  (`energy A B`) — drift the file view misses when the file set is stable.
- **Eviction watchpoint (`missing <node>`).** Find the exact step a node was squeezed
  out of the budgeted window by competing pressure, with the triggering op and the
  token gap — e.g. `·●●●●●○○●●` reads "in context until a failure made a neighbour hot
  and evicted the real cause, then a page-fault pulled it back".

## Quickstart — give your agent a memory

```bash
cargo build --release          # → ./target/release/ccos
```

Wire CCOS into an MCP-compatible host. The repo ships a project [`.mcp.json`](.mcp.json):

```json
{
  "mcpServers": {
    "ccos": { "command": "./target/release/ccos", "args": ["mcp", "workspace.ccos"] }
  }
}
```

The agent now has the CCOS tools and the `ccos://session/context` resource; memory
persists in `workspace.ccos` (+ a `.oplog` timeline) across restarts. When a run
drifts, debug it post-mortem:

```bash
printf '%s\n' 'timeline' 'missing src/db.rs 40' 'energy 4 9' 'quit' \
  | ccos postmortem workspace.ccos
```

Agent-memory contract and MCP tool schemas:
[`docs/MEMORY_INTERFACE.md`](docs/MEMORY_INTERFACE.md). Wiring CCOS to feed an agent's
own runs automatically (the transparent PostToolUse "hardware intercept") and the
post-mortem protocol: [`docs/SELF_ANALYSIS.md`](docs/SELF_ANALYSIS.md).

## Inspect the causal graph (CLI)

CCOS can also be driven directly to analyse a codebase's causal structure — it can
analyse its own source tree:

```bash
ccos analyze src --cycles                 # structural report (+ dependency cycles)
ccos analyze src --out run.json           # persist a snapshot (graph + hash-chained log)
ccos verify run.json                      # hash chain valid? dangling edges? → exit 0/1
ccos replay run.json                      # deterministic event-log replay + stats

ccos top src --limit 15                   # the hottest nodes by causal score
ccos blame run.json file:src/memory.rs --depth 4   # upstream causes + downstream blast radius
ccos failure run.json file:src/memory.rs --depth 2 # inject a fault and watch it propagate
ccos regions src --activate file:src/memory.rs     # cluster into causal regions, hydrate one
```

See [`docs/USAGE.md`](docs/USAGE.md) for every command with examples, and `ccos --help`
for the full list.

## Architecture

```
            ┌─────────────┐   register/Δ   ┌──────────────────────────┐
 .rs files →│   parser    │───────────────▶│  IncrementalGraphEngine   │
            └─────────────┘                └────────────┬─────────────┘
                                                         │ O(Δ) mutations
                                                         ▼
   ┌─────────┐  recall/page  ┌──────────────────┐   ┌──────────────────┐
   │  agent  │◀─────────────▶│ external_memory / │──▶│   MemoryGraph    │
   │  (MCP)  │   page_fault  │  agent_session    │   │  scoring/paging/  │
   └─────────┘               └─────────┬────────┘   │  failure-propag.  │
                                       │ checkpoint  └────────┬─────────┘
                                       ▼                      │ snapshots
                              ┌──────────────────────────────▼───────────┐
                              │ EventLog + DistributedEventLog + .oplog    │
                              │ (deterministic + hash-chained replay)      │
                              └────────────────────────────────────────────┘
```

Module reference: `cargo doc --open` (every module has rustdoc), or
[`src/lib.rs`](src/lib.rs) and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Testing

```bash
cargo test                     # 267 unit, integration & doc tests
cargo clippy --all-targets --all-features   # lint-clean (-D warnings in CI)
cargo test -- --ignored        # opt-in: 1,000,000-cycle long-stability run
```

Key invariants under test: no dangling edges (`edges ⊆ nodes × nodes`) even under
aggressive paging; bounded node/edge growth over 10k+ mutation cycles; deterministic
eviction (reproducible snapshot hashes and replays); and tamper-evidence (both
hash-chained logs detect any mutation, reorder, insertion or deletion).

## Documentation

- [`docs/USAGE.md`](docs/USAGE.md) — **command reference & walkthroughs**.
- [`docs/MEMORY_INTERFACE.md`](docs/MEMORY_INTERFACE.md) — the **external-memory
  interface**: the façade an agent programs against, and the `ccos memory` / `ccos mcp`
  transports.
- [`docs/SELF_ANALYSIS.md`](docs/SELF_ANALYSIS.md) — **dogfooding**: wire CCOS into a
  coding agent and debug its drifts post-mortem.
- [`docs/paper/`](docs/paper/) — the **research paper** (English + fr/es/zh/ko/ar): the
  formal model, the determinism + replay theorem, and the honest negative result vs
  RAG / GraphRAG.
- [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) — **bare-metal notes**: durable
  checkpoints, the Jetson reproducible-measurement script, and the honest triage of
  which low-level knobs actually matter for a <1%-of-the-loop kernel.
- [`docs/context_regions.md`](docs/context_regions.md), [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md),
  [`CONTRIBUTING.md`](CONTRIBUTING.md), [`CHANGELOG.md`](CHANGELOG.md),
  [`ROADMAP.md`](ROADMAP.md), [`docs/BIBLIOGRAPHY.md`](docs/BIBLIOGRAPHY.md).

## Status & limitations

A research prototype, not a production system. Known gaps (tracked in
[`ROADMAP.md`](ROADMAP.md)):

- The parser defaults to a **line-based heuristic** (zero dependencies); build with
  `--features syn-parser` for a real `syn` AST. Edges capture containment/dependency,
  **not** call graphs or data flow — so the causal graph is structural, not semantic.
- CCOS makes drift **auditable and debuggable**; it does **not** claim to prevent
  drift or to improve task success (the paper is explicit on this — the demonstrated
  win is efficiency and auditability, not resolution).
- The agent self-feed hook is a best-effort heuristic intercept, not a ground-truth
  tracer; use one writer per `workspace.ccos`.

## License

Unlicensed research prototype. Add a license before any external use.
