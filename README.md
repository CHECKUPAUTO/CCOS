# CCOS — Causal Context Operating System

> **An external working memory for coding agents.** CCOS keeps the *right* code in
> an LLM agent's context the way a CPU's MMU manages memory — a causal graph of the
> codebase, paged against a token budget — and records every step in a
> deterministic, replayable, hash-chained log you can rewind and audit.

CCOS gives a coding agent a memory that is **frugal, deterministic, replayable, and
auditable**, plus the tooling to debug the agent's attention when a run goes wrong:

- **A memory server any agent plugs into.** `ccos mcp` speaks the
  [Model Context Protocol](https://modelcontextprotocol.io) over stdio, so an
  MCP-compatible agent (Claude Code, a local agent on a Jetson) uses CCOS as native
  working memory: ingest code, signal failures, recall a bounded causal window —
  persisted across restarts.
- **Time-travel sessions.** Every cognitive operation is event-sourced, so you can
  rewind the agent's exact context to any past step and replay it under different
  parameters (a larger budget, a different anchor).
- **A post-mortem debugger.** `ccos postmortem` is "GDB for the agent's memory":
  walk the cognitive timeline and watch the working set drift — down to the exact
  step the real cause was evicted from the budgeted window (`missing`, the eviction
  watchpoint) and the node-level pressure that pushed it out (`energy`).
- **Transparent self-instrumentation.** A drop-in hook feeds an agent's own file
  reads and failed test runs into CCOS automatically, so its drifts are reproducible
  after the fact ([`docs/SELF_ANALYSIS.md`](docs/SELF_ANALYSIS.md)).

CCOS is **not a retriever and not a RAG replacement** — a lexical baseline matches it
at putting a fix's files in the window. Its value is the axis a probabilistic
RAG/framework stack structurally lacks: it reaches the same result on a fraction of
the context budget (it self-bounds at the causal region, with no top-k to tune), and
every step is deterministic, replayable, and tamper-evident. The research story
behind that conclusion — the original hypothesis, the bug-mining harness, the
measurements against RAG/GraphRAG — lives in the paper, not here:
[`docs/paper/`](docs/paper/).

| OS / MMU concept     | CCOS analogue                                                |
| -------------------- | ------------------------------------------------------------ |
| Pages / working set  | Graph nodes (files, modules, symbols, imports)               |
| Paging to a budget   | causal `recall` / `select_context_window` + `enforce_paging` |
| Page fault           | feed a compiler/test failure back in → re-page the region    |
| Scheduling priority  | causal score (importance · failure · recency · access)       |
| Write-ahead log      | append-only, **hash-chained** event log (replay + audit)     |
| Fault propagation    | failure pressure weighted across causal edges                |

CCOS is a research prototype in Rust (edition 2021) — see
[Status & limitations](#status--limitations).

---

## Quickstart — give your agent a memory

```bash
cargo build --release          # → ./target/release/ccos
```

Plug CCOS into an MCP-compatible agent. The repo ships a project
[`.mcp.json`](.mcp.json):

```json
{ "mcpServers": { "ccos": { "command": "./target/release/ccos", "args": ["mcp", "workspace.ccos"] } } }
```

The agent now has CCOS tools (`ingest`, `recall`, `signal_failure`, `page_fault`,
`timeline`, `recall_what_if`, `stats`, `verify`) and the `ccos://session/context`
resource — its self-bounding working set, ready to inject into a system prompt.
Memory persists in `workspace.ccos` (plus a `.oplog` timeline) across restarts.

When a run drifts, debug it post-mortem:

```bash
printf '%s\n' 'timeline' 'missing src/db.rs 40' 'energy 4 9' 'quit' \
  | ccos postmortem workspace.ccos
```

## Agent memory — `ccos mcp` / `ccos memory`

Use CCOS as an agent's external working memory — ingest source, signal failures,
recall a bounded causal window, verify the hash chain — over two transports on the
same documented façade:

```bash
# Any MCP-compatible agent, via stdio JSON-RPC 2.0 (event-sourced, persistent):
ccos mcp workspace.ccos

# Any language, via stdio JSON-Lines (checkpoint-backed):
printf '%s\n' '{"op":"ingest","uri":"src/db.rs","source":"pub fn query() {}"}' \
              '{"op":"recall","strategy":"around","anchor":"file:src/db.rs","budget":2048}' \
  | ccos memory --path workspace.ccos
```

`ccos mcp` speaks the standard MCP handshake and advertises eight tools and two
resources (`ccos://session/context`, the self-bounding working set, and
`ccos://session/timeline`). Pass a `workspace.ccos` to persist across restarts; the
op-log compacts so a long-running session stays bounded. Full contract, tool schemas
and a client-config snippet: [`docs/MEMORY_INTERFACE.md`](docs/MEMORY_INTERFACE.md).

## Post-mortem debugging — `ccos postmortem`

A "GDB for the agent's memory": walk a recorded (or persisted) cognitive timeline by
hand and watch the working set drift as failures propagate. With a workspace path it
loads the persisted op-log (even after a crashed run); with none it walks a built-in
session that drifts.

```bash
ccos postmortem workspace.ccos     # then: timeline / goto N / recall / diff / energy / missing
```

A cursor time-travels the timeline (`goto`, `next`/`prev`, `recall`/`around`/`task`
as of the cursor), and three drift views explain what happened:

- `diff A B` — which **files** entered/left the working set between two steps.
- `energy A B` — node-level **Δscore + failure-pressure**, the migration of causal
  "heat" through the AST as failures propagate (visible even when the file set is
  stable).
- `missing <node> [budget]` — an **eviction watchpoint**: the exact step a node drops
  out of the budgeted window, the triggering op, and the token gap. A status strip
  reads at a glance — `·●●●●●○○●●`: in context until a failure made a neighbour hot
  and squeezed the real cause out, then a page-fault pulled it back.

Every command reconstructs state deterministically via replay — exact and
side-effect free.

## Self-analysis — dogfood CCOS on your own agent

Wire CCOS into a coding agent so its runs feed a causal memory you can then debug.
The repo ships the `.mcp.json` above (the agent queries its memory) and
[`scripts/ccos_self_feed.py`](scripts/ccos_self_feed.py), a PostToolUse hook — a
transparent "hardware intercept" that turns every source file read into an `ingest`
and every failed `cargo test/build` into a `page_fault`, with zero cognitive overhead.
Methodology and the post-mortem protocol: [`docs/SELF_ANALYSIS.md`](docs/SELF_ANALYSIS.md).

## Inspect the causal graph (CLI)

CCOS can also be driven directly to analyse a codebase's causal structure. It can
analyse its own source tree:

```bash
ccos analyze src --cycles                 # structural report (+ dependency cycles)
ccos analyze src --out run.json           # persist a snapshot (graph + hash-chained log)
ccos verify run.json                      # hash chain valid? dangling edges? → exit 0/1
ccos replay run.json                      # deterministic event-log replay + stats

ccos top src --limit 15                   # the hottest nodes by causal score (the working set)
ccos blame run.json file:src/memory.rs --depth 4   # upstream causes + downstream blast radius
ccos failure run.json file:src/memory.rs --depth 2 # inject a fault and watch it propagate
ccos export run.json --out graph.graphml  # GraphML for Gephi / yEd / Cytoscape / networkx
ccos regions src --activate file:src/memory.rs     # cluster into causal regions, hydrate one
```

See [`docs/USAGE.md`](docs/USAGE.md) for every command with examples and output, and
`ccos --help` for the full list (including the v0.3 autonomous runtime: `scan`,
`agents`, `runtime`).

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

Module reference: `cargo doc --open` (each module has rustdoc), or
[`src/lib.rs`](src/lib.rs) and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Testing

```bash
cargo test                     # 267 unit, integration & doc tests
cargo clippy --all-targets     # lint-clean
cargo test -- --ignored        # opt-in: 1,000,000-cycle long-stability run
```

Heavier stress/chaos harnesses live in [`scripts/`](scripts/). Key invariants under
test:

- **No dangling edges**: `edges ⊆ nodes × nodes`, even under aggressive paging.
- **Bounded growth**: node and edge counts stay bounded over 10k+ mutation cycles.
- **Deterministic eviction**: identical builds evict identically, so snapshot hashes
  and replays are reproducible.
- **Tamper-evidence**: both hash-chained logs detect any payload mutation, reorder,
  insertion or deletion.

## Documentation

- [`docs/USAGE.md`](docs/USAGE.md) — **command reference & walkthroughs**: every
  command with example invocations and output.
- [`docs/MEMORY_INTERFACE.md`](docs/MEMORY_INTERFACE.md) — the **external-memory
  interface**: the façade an agent uses, plus its `ccos memory` and `ccos mcp`
  transports.
- [`docs/SELF_ANALYSIS.md`](docs/SELF_ANALYSIS.md) — **dogfooding**: wire CCOS into a
  coding agent and debug its drifts post-mortem.
- [`docs/paper/`](docs/paper/) — **research paper**: the formal model, the
  determinism argument, and the falsifiable comparison protocol vs RAG / GraphRAG /
  MemGPT / LangGraph — the *why* and the *what-it-was-meant-to-be*.
- [`docs/context_regions.md`](docs/context_regions.md) — the Context Region Engine:
  spatial memory model, region definition, admission policy, measured locality.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — developer guide: module map, data
  structures, invariants, and how to extend the kernel.
- [`CONTRIBUTING.md`](CONTRIBUTING.md), [`CHANGELOG.md`](CHANGELOG.md),
  [`ROADMAP.md`](ROADMAP.md), [`docs/BIBLIOGRAPHY.md`](docs/BIBLIOGRAPHY.md).

## Status & limitations

A research prototype, not a production system. Known gaps (tracked in
[`ROADMAP.md`](ROADMAP.md)):

- The parser defaults to a **line-based heuristic** (zero dependencies). Build with
  `--features syn-parser` for a real `syn` AST (nested modules, multi-line signatures,
  grouped `use`, impl methods); the heuristic stays as the fallback.
- Edges capture containment/dependency, **not** call graphs or data flow.
- The agent self-feed hook is a best-effort heuristic intercept, not a ground-truth
  tracer; use one writer per `workspace.ccos`.

## License

Unlicensed research prototype. Add a license before any external use.
