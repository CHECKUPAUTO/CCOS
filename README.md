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

**What's genuinely new.** Many systems page code into a context window; the
distinctive contribution of CCOS is to treat the agent's **working memory itself**
as a transactional subsystem — *deterministic, hash-chained, replayable bit-for-bit,
and post-mortem debuggable*. To our knowledge it is the **first to make an agent's
attention a "flight recorder"**: you can rewind to the exact step its representation
of the project corrupted, replay that window under different parameters, and a
`missing <node>` watchpoint names the **precise moment the real cause was evicted**
from the budgeted window. (Every other axis — paging, causal graphs, frugal
retrieval — has prior art; *this* deterministic, replayable, attention-level
debugger is the part a probabilistic RAG/agent stack structurally lacks.)

**What it is, honestly.** CCOS's measured advantage is **coverage of the right context,
frugally**. When you work on a real source file, its causal recall puts that file's
cross-file dependencies into a tight (2048-token) window **81–100 %** of the time, where
naively opening the file truncated to the same budget gets **0–2 %** — and cross-file
dependencies are everywhere, so this is the *everyday* case, not a corner one (measured
model-free over `syn`, `serde_json` and this repo — `scripts/ccos_context_value.py`). On the
*narrow* slice of **multi-file bugs** (the cause sits in a file a budget would truncate away —
only ~1–2 % of real fixes), that coverage advantage becomes a **resolution** one: a capable
local model fixes the root cause where an equal-budget file dump cannot. CCOS is *not* a
better retriever in the RAG sense (a tuned top-k baseline can also be sparse); its structural
wins are self-bounding (no `k` to tune) plus **deterministic, replayable, auditable**. The
full research story — the original hypothesis, the bug-mining harness, the honest negative
result vs RAG/GraphRAG — lives in [`docs/paper/`](docs/paper/) (six languages); the field
measurements behind the numbers above are in
[`docs/FIELD_CAMPAIGN_H.md`](docs/FIELD_CAMPAIGN_H.md). CCOS is a research prototype in Rust
(edition 2021); see [Status & limitations](#status--limitations).

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

- **Self-calibration.** CCOS assembles a token-bounded working set from causal-graph
  activation and **stops at the causal region** — there is no `k` to tune. Measured over
  real crates (`syn`, `serde_json`, this repo): for a file you're working on, its cross-file
  dependencies land in a 2048-token window **81–100 %** of the time, vs **0–2 %** for naively
  opening the file at the same budget. On a *big* file — where opening it truncates every
  dependency — that gap is **79–100 % vs 0 %**. Three measured fixes get it there at a fixed
  budget regardless of the anchor's size: symbol-span granularity (no node carries a whole
  file), degree-aware failure propagation (a hub distributes pressure instead of flooding),
  and anchor-proximity ranking — see [`docs/FIELD_CAMPAIGN_H.md`](docs/FIELD_CAMPAIGN_H.md).
- **Context page fault.** Feed `cargo test` / panic output back in: CCOS parses the faulting
  source locations from the trace, injects failure pressure on those files, and re-pages a
  refreshed window. The propagation reaches the cross-file *cause* (up to ~3 hops), not just
  the symptom the trace names — the post-mortem tools below let you verify which nodes the
  window actually held at each step.

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
- **Field-data export.** `ccos postmortem <workspace> --json` dumps the session
  record (stats / integrity / timeline / working set) for archiving or fleet
  collection (`scripts/fleet_collect.sh`); a copied workspace replays bit-for-bit
  off-site. See [`docs/SELF_ANALYSIS.md`](docs/SELF_ANALYSIS.md).

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
cargo test                     # 285 unit, integration & doc tests
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
- [`docs/COMPETITIVE.md`](docs/COMPETITIVE.md) — **honest competitive read**: what a
  source-code reading of Headroom (the closest competitor) actually shows — where it is
  stronger (compression, RAG memory) and the one axis it does not occupy (a replayable,
  auditable, post-mortem-debuggable working memory).
- [`docs/context_regions.md`](docs/context_regions.md), [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md),
  [`CONTRIBUTING.md`](CONTRIBUTING.md), [`CHANGELOG.md`](CHANGELOG.md),
  [`ROADMAP.md`](ROADMAP.md), [`docs/BIBLIOGRAPHY.md`](docs/BIBLIOGRAPHY.md).

## Status & limitations

A research prototype, not a production system. Known gaps (tracked in
[`ROADMAP.md`](ROADMAP.md)):

- The parser defaults to a **line-based heuristic** (zero dependencies); build with
  `--features syn-parser` for a real `syn` AST. Edges capture containment/dependency,
  **not** call graphs or data flow — so the causal graph is structural, not semantic.
- CCOS's broad, proven wins are **coverage** (the right context, frugally) and
  **auditability**. On the *narrow* slice of multi-file bugs it also improves
  **resolution** (a capable local model fixes the root cause where an equal-budget dump
  can't — measured across two model families); on single-file bugs it's at parity, and
  the right context is **necessary but not sufficient** — a weak model (≤~3B), or even a
  strong one, can still misuse it (see `docs/FIELD_CAMPAIGN_H.md`). It does **not** claim
  to prevent drift.
- The agent self-feed hook is a best-effort heuristic intercept, not a ground-truth
  tracer; use one writer per `workspace.ccos`.




## License

Dual-licensed: [PolyForm Noncommercial 1.0.0](LICENSE.md) for noncommercial and personal use; commercial license required for any commercial use. See [LICENSING.md](LICENSING.md).
