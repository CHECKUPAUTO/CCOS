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
- **Hybrid entry fusion.** A free-text recall (`Recall::Hybrid`) resolves its entry node by
  **reciprocal-rank fusion** of three signals — lexical token overlap, semantic INT4-TF-IDF
  cosine, and the causal **active-failure focus** — before causal expansion. No cross-signal
  score calibration (RRF ranks, it doesn't add scores): a node strong on any one axis can
  surface, while consensus across several wins. The causal vote is *sparse* — it speaks only
  for what's failing — so it abstains on a quiet graph and pulls the active problem region in
  once you signal a failure. Deterministic; reachable via the MCP `recall` tool
  (`strategy:"hybrid"`).
- **Opt-in learned embedder (`learned-embed`).** The semantic signal defaults to
  deterministic INT4 TF-IDF (the measured baseline). Build with `--features learned-embed`
  and it's distilled into a **learned latent-semantic (LSA) projection** — the top singular
  vectors of the corpus's own term co-occurrence, via a fixed Jacobi sweep — so a query term
  that only *co-occurs* with a file's terms still matches it (synonymy raw TF-IDF can't see).
  **Zero new dependencies, fully deterministic** (the replay invariant holds); the default
  build is byte-identical. (Honestly: it's a *linear* distillation, not a neural model — it
  helps when there are enough files to truncate, and it's opt-in because the eigensolve costs.)
- **Non-destructive eviction (the "swap").** When the resident set exceeds its cap, CCOS
  **demotes** the coldest nodes — with their edges — into a **COLD tier** instead of dropping
  them, so the working memory is *unbounded-backed*: the resident window stays small
  (frugality), the backing store grows into available RAM, and **nothing is lost**. Any node
  pages back on demand (`page_in`); on the read paths the tier is **transparent** — a
  `signal_failure` or a `page_fault` resurrects a demoted faulting file, and a **recall around**
  a demoted node pages it (and its cold neighbours) back automatically. Opt in to
  `attach_cold_spill(dir, budget)` and the coldest COLD **content** spills to a
  content-addressed on-disk store (SHA-256, **deduplicated**, **hash-verified** on read —
  a tampered blob is a cold-miss, not a silent restore), so the resident *and* cold content
  footprint is **RAM-bounded** while the backing store on disk is **unbounded**; it faults
  back transparently. And at the deepest tier, opt in to `set_cold_content_budget(bytes)`:
  once the backing store itself must stay frugal, the coldest content is **compacted** —
  code skeletonised, prose summarised — to a causal summary (lossy, but **observable** via
  `cold_compacted`, **never silently dropped**). And once even the per-entry *metadata*
  must shrink — measured to be the COLD tier's dominant *resident* cost, ~2.8× the spilled
  content — opt in to `set_cold_resident_budget(bytes)`: the coldest entries are
  **deep-spilled**, archived whole to the same store and kept in RAM as a compact husk (a
  stub plus the neighbour ids paging needs), so the COLD tier's *resident* footprint is
  bounded too, not just its content — losslessly (faults back hash-verified on `page_in`),
  and by shrinking edges to ids rather than the bridge-edge blow-up a lossy contraction
  would inflict on hubs. The cognitive-MMU promise made literal —
  "infinite" working memory as a *direction*, expressed concretely as **frugality × available
  RAM** (`MemoryStats.cold` / `cold_spilled` / `cold_compacted` surface the tier).
  Deterministic; lossless and off by default — spill and compaction are opt-in modes.

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
- **Self-improving retrieval (trains on the log).** The replayable log isn't just for
  debugging — it's *training data*. A retrieval **reward** is read straight off it: for
  each recorded recall, was the node the agent engaged *next* (a failure / page-fault)
  in the window that recall would have produced? `tune_recall_weights` then learns the
  scoring weights that maximise that hit rate by **deterministic coordinate ascent,
  evaluated by replay** (same log ⇒ same weights), and adopting them records an
  `Op::Retune` so the learned policy is **auditable and reproduced on replay** —
  `replay == live` holds. Better retrieval that *reinforces* the moat: nobody else has a
  deterministic, replayable causal log to train on. (The reward is an honest proxy and
  the optimiser is greedy — it's an offline tune, not a hot path.)

### 3. Standard MCP transport

- **Stdio JSON-RPC server.** Native, synchronous, zero-network integration with any
  MCP-compatible host (e.g. Claude Code). Nine tools: `ingest`, `recall`,
  `signal_failure`, `page_fault`, `stats`, `verify`, `timeline`, `recall_what_if`,
  `ccos_retrieve` (fetch the original of a compressed item).
- **Dynamic resources.** `ccos://session/context` exposes the self-bounding working
  set, **reversibly compressed** by default (`CCOS_COMPRESS_CONTEXT=0` to disable for
  A/B), for the host to drop into its system prompt; `ccos://session/timeline` exposes
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

### 5. Reversible context compression (the CCOS ↔ headroom angle)

CCOS historically *selected* the right nodes but never *re-encoded* their content
— it paged a header or a symbol span verbatim into the prompt. The compressor
module (`src/compressor.rs`) adds a real compression pass **downstream of the
causal MMU**, without sacrificing the determinism / replay / auditability
guarantees:

- **Three deterministic compressors.** [`CausalCrusher`] collapses JSON
  (columnar arrays, null-drop, string back-refs); [`CausalAST`] skeletonizes
  code (strips comments / blank lines / `use` imports, collapses long
  signature runs, renames `_`-temporaries to `$n`); [`CausalSumm`] is a
  TextRank extractive summarizer **biased by the causal score** so sentences
  touching high-pressure nodes surface first — the angle headroom's TextRank
  lacks. No ML model, no stochastic step: everything is seed-stable and
  total-order tie-broken, so the hash-chain replay and the `postmortem`
  time-travel debugger remain bit-reproducible.
- **Reversibility (CCR store).** Every compressed item carries a 12-char
  `ccr_ref` (a truncated SHA-256 of the original); the host LLM calls the
  `ccos_retrieve` MCP tool to fetch the full text on demand — the CCOS
  equivalent of headroom's `headroom_retrieve`. Nothing is ever lost.
- **Cross-item near-duplicate suppression.** A distilled MinHash (64 hashes,
  3-char shingles) estimates Jaccard similarity over the *compressed* forms;
  near-dup items are replaced by a one-line `// ~dup of <uri>` placeholder
  (their original stays retrievable). The causal graph dedups cross-file; this
  dedups *within* a window.
- **Budget feedback loop** — CCOS's unique advantage over a pure compressor:
  when compression shrinks the window below the token budget, the freed space
  is *re-spent* on more causal nodes (a second recall pass with a grown
  effective budget), so the host gets **strictly more causal signal at the
  same emitted-token cost**. Measured on this repo's source: on `around
  external_memory` at an 8192-token budget the loop recalls **+15 causal nodes**
  vs a single compressed pass, while staying under budget.
- **Auto-tuner.** `CausalCompressor::auto_tune(sample)` runs a deterministic
  coordinate-descent over the knobs (dedup threshold, AST v2 on/off, signature
  collapse point, summary length, min-chars) to minimise the compressed-token
  count on a representative sample — bootstrap the config on a new corpus
  without hand-tuning.

**Measured on this repo's own source** (38 Rust files; run
`cargo run --example bench_compress --release`):

| recall                | budget | raw tokens | compressed | reduction | shrink |
| --------------------- | -----: | ---------: | ---------: | --------: | -----: |
| working_set           |   2048 |        895 |        595 |      34 % | 1.50×  |
| working_set           |   8192 |       6783 |       3450 |      49 % | 1.97×  |
| around parser         |   4096 |       4096 |       3291 |      20 % | 1.24×  |
| around external_mem   |   8192 |       8192 |       5563 |      32 % | 1.47×  |

CausalAST-led compression delivers ≈20–50 % on real Rust code — the deterministic floor
of headroom's 47–92 % range (headroom's upper band comes from its trained
`Kompress-base` model, which CCOS does not ship by choice: it would break the
deterministic-replay invariant). The budget feedback loop then *adds* causal
nodes on top of the compression gain. Zero new dependencies; the module reuses
only `serde_json` (already a CCOS dep) and std. The SCIRUST counterparts the
algorithms were distilled from live in `scirust-nlp-advanced`
(`bloom`, `lsh`, `trie`, `huffman`, `similarity`, `keyword`).

### 6. Input hardening — deterministic de-obfuscation + an injection signal

The context an agent reads is an attack surface. CCOS de-obfuscates ingested
text **at the boundary**, deterministically and auditably — the same axis as the
rest of the system, applied to security.

- **Unicode de-obfuscation (`src/sanitizer.rs`).** Hidden-character attacks that
  a human reviewer cannot see but a model still tokenises are **surfaced as
  explicit visible literals** (`[U+202E RLO]`, `[U+200B ZWSP]`,
  `[U+E0048 TAG:H]`) rather than silently stripped: **Trojan-Source** bidi
  overrides (CVE-2021-42574), zero-width formatting, **Unicode-Tags ASCII
  smuggling** (decoded back to the ASCII it shadows), and raw controls. This
  closes the hidden-character class *completely* — the category-**Cf** vectors
  that `guard.rs`'s output-side `is_control()` strip is blind to. It runs
  default-on in `ingest_source` (clean source is borrowed unchanged, zero copy);
  findings ride back in `IngestReport.anomalies` and the event-log hash is taken
  over the cleaned form, so **replay reproduces the de-obfuscated state**.
- **Injection *signal* (`src/hashing_tokenizer.rs` + `src/injection_classifier.rs`).**
  A deterministic feature-hashing tokenizer → a linear log-space
  (multinomial-Naive-Bayes) score `W·X + b`, with weights **locked into an
  immutable, SHA-256-verified blob** and a **forensic** per-feature
  decomposition of every decision. Held-out red-team
  (`cargo run --example injection_redteam`): **precision 0.868, recall 0.933,
  F1 0.900**. We label it a *signal, not a shield* — and the forensic output
  shows exactly why (false positives on benign trigger-word mentions; false
  negatives on novel paraphrase, the structural blind spot of any
  bag-of-features model). No character pass — and no bag-of-words model — solves
  prompt injection; privilege separation in the host remains the real mitigation.

```bash
ccos sanitize path/to/file.rs            # de-obfuscate + score (human / --json)
ccos sanitize --strict path/to/file.rs   # non-zero exit on danger (CI / pre-commit)
```

The signal also rides the live path: every `ingest` (CLI, façade, or MCP) returns
an `injection_score` / `injection_flagged` alongside the de-obfuscation `anomalies`.

See [`docs/SECURITY.md`](docs/SECURITY.md) for the full threat model and the
honest scope (what it does *not* cover: homoglyphs, semantic paraphrase).

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
cargo test                     # 364 unit, integration & doc tests (default features)
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
- [`docs/SECURITY.md`](docs/SECURITY.md) — **input hardening**: the deterministic
  Unicode de-obfuscation pass and the injection *signal*, with the threat model
  and the honest scope (and the measured red-team numbers).
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
