# CCOS — Improvement Plan

Prioritized roadmap from the audit. Effort: S/M/L.

## ✅ Done — audit pass 1 (correctness)

- **Fixed critical edge leak.** `enforce_paging()` ran re-entrantly inside
  `upsert_node()`, so `add_edge()` attached edges to just-evicted nodes; dangling
  edges grew `O(cycles)` (9,000+ edges for 200 nodes), breaking the `O(Δ)` promise
  and the stability budget. `add_edge` now rejects dangling endpoints,
  `enforce_paging` prunes defensively, and `tests/graph_invariants.rs` locks the
  invariant. (10k-cycle slowdown: 11× → 1.08×.)
- **Deterministic eviction/ordering** — tie-break on `NodeId` across paging,
  context selection and score listing → reproducible replays/snapshots.
- **Hardened the guard** — `is_valid_json` now requires the *whole* payload to
  parse (was accepting any valid prefix → trailing injection slipped through).
  Replaced tautological adversarial assertions with real safety checks.
- **Real CLI** (`demo`, `analyze`, `help`, `version`) replacing the hard-coded
  demo that ended in `process::exit(0)`.
- **Docs**: `README.md`, crate-level rustdoc, this roadmap. Zero clippy warnings.

## ✅ Done — audit pass 2 (capability)

- **Enforced `GuardConfig::max_nesting_depth`** (was a defined-but-unused config
  field) + tests. *(was P0.2)*
- **Persistence** — `persist::KernelSnapshot` (graph + event log + hash chain)
  with `save`/`load`; new `ccos analyze --out`, `ccos verify`, `ccos replay`
  commands. *(was P2.7)*
- **Multi-model consensus wired in** — `LlmClient::query_models` +
  `ConsensusEngine` in the demo. *(was P1.4)*
- **Adversarial chaos mode** — `ccos chaos [--iters N]` drives fault injection
  through the guard and asserts it never emits invalid JSON. *(was P1.5)*
- **Hash-chained log integrated** — the demo and `analyze` now build a
  `DistributedEventLog`; `verify` checks its integrity. *(was P1.6, partial)*
- **Recency clock wired** — the demo calls `MemoryGraph::tick()` between cycles
  so recency actually decays. *(was P2.8)*
- **Richer `analyze`** — `--json` export, `--cycles` dependency-cycle detection,
  node-type histogram. *(was P2.9)*
- **Fallible `LlmClient::try_new`** alongside the panicking `new`. *(was P3.12)*
- **CI** — `.github/workflows/ci.yml` runs `build`, `clippy -D warnings`,
  `test`, and a CLI smoke test (`analyze → verify → replay → chaos`).
- **Full graph reconstruction from the event log** — `EventLog::record_graph`
  emits `NodeUpserted`/`EdgeAdded` events; `GraphReconstructor` rebuilds an
  identical graph from the log alone (`replay` reports `matches snapshot: true`).
  Closes the event-sourcing loop. *(was P0.3 / P0.2)*
- **Graphviz export** (`analyze --dot`) and **orphan-node** reporting.

## ✅ Done — v0.3 (Autonomous Context Runtime)

- **Context scheduler** (`scheduler.rs`) — HOT/WARM/COLD paging by token budget
  and priority; `allocate`/`evict`/`optimize`, no node lost.
- **Real workspace scanner** (`workspace.rs`) — async `tokio::fs` scan with
  add/modify/remove delta detection feeding only Δ to the engine.
- **Multi-agent layer** (`agents.rs`) — Coder/Reviewer/Security agents, guarded
  + logged + deterministic.
- **Persistent runtime** (`persistence.rs`) — directory-based save/load/restore
  with verify-on-restore.
- **Benchmark framework** (`benchmark.rs`) — cycle benchmark → JSON report
  (100k stress in CI; 1M opt-in).
- **CLI** — `scan`, `agents`, `benchmark`, `runtime` (capstone).
- **Quality** — `main.rs` split into `commands_demo`/`commands_runtime`
  (1206 → 679 lines); `util::sha256_hex` DRY consolidation; dead code removed;
  config flags (`--max-nodes`, `--budget`); property tests; criterion benches.
  *(was P2.4, P2.5, P3.8)*

---

## ✅ Done — v0.3 Context Region Engine (spatial memory)

- **Context regions** (`context_region`, `region_engine`) — the 1-D scored graph
  is lifted into a spatial map: nodes are embedded in a 3-D context space
  (structural / causal / temporal) and clustered into regions (connected
  components of the cross-file causal-link graph) with a temperature and causal
  density. Regions are hydrated as `ContextWindow`s instead of loading files.
- **Dynamic admission policy** (`context_policy`) — the static `0.6` threshold
  becomes a function of token pressure, task complexity, region temperature and
  density.
- **Event sourcing + deterministic replay** — `RegionCreated/Activated/Merged/
  Evicted/ContextWindowGenerated` events; `replay_from` reconstructs regions
  bit-for-bit from a rebuilt graph (proof + 10k-cycle no-drift test).
- **Locality metrics** (`region_metrics`) + `scripts/region_benchmark.sh`: region
  selection covers 97% of a task's causal neighbourhood vs 35% flat, ≈48% fewer
  tokens; regions 95.5% internally connected.
- **Docs**: `docs/context_regions.md` + an arXiv research paper in `docs/paper/`
  (formal model, determinism theorem, falsifiable comparison protocol vs
  RAG/GraphRAG/MemGPT/LangGraph). `ccos regions` CLI.

---

## Audit pass 3 — full-codebase review (2026-06-24)

A four-axis read of the whole tree (correctness/determinism, architecture/redundancy,
honesty code↔docs↔paper, tests/API). **Fixed in this pass:**

- **`truncate()` UTF-8 panic** (`main.rs`) — byte-sliced a multi-byte char on any
  non-ASCII id/message; now cuts on a char boundary (+ regression test).
- **Non-reproducible distributed-log chain** — `compute_link_hash` hashed the
  wall-clock timestamp; now excludes it (mirrors `EventLog`), so the chain is
  replay-reproducible (+ test). *On-disk chain hashes change — a format bump.*
- **`parser.rs` slice OOB** under `--features syn-parser` (a span line past EOF) —
  clamped into the slice (+ test).
- **`bench_compress` panicked** (hard-coded `/root/CCOS`) — derives the corpus from
  `CARGO_MANIFEST_DIR`; README §5 table re-measured on the real 38-file corpus.
- **Test/file counts reconciled** — docs disagreed (156/202/212/285/288); the real
  default-feature `cargo test` count is **364**, fixed across README, PITCH,
  ARCHITECTURE, PAPER.md and all six `.tex`; "33 Rust files" → 38.
- **Honesty** — `mcp.rs` docstring now lists all 9 tools; `COMPETITIVE.md` no longer
  implies *live* semantic recall; `lib.rs` documents which recent modules are in-tree
  but **not yet on the live path**.

**Remaining (decisions / larger work, not bugs):**

### A — Wire in or retire the unwired recent modules
- `embeddings` (semantic recall) and `eviction_policy` (learned paging) are built and
  tested but unreachable from the live recall/ingest path. **Decide:** wire in (a
  `Recall::Semantic`; blend the policy into `enforce_paging`) or remove. (M)
- `injection_classifier` runs only in `ccos sanitize`. Decide whether to surface an
  injection score in `IngestReport` or keep it CLI-only. (S)

### B — Collapse duplicated abstractions
- One snapshot type: merge `persist::KernelSnapshot` and `persistence::RuntimeState`
  (field-identical payloads). (M)
- One event chain: collapse `distributed_event_log` onto `event_log`'s chain, or
  document why two exist. (M)
- One context selector: designate `CcosMemory::recall` canonical; demote
  `select_context_window` / `hot_set` / `hot_context` / `activate_region`. (M)
- One snapshot-error type (unify on `MemoryError`). (S)

### C — Encapsulation & API
- Make `MemoryGraph.{nodes,edges}` (and `EventLog.events`, `LinearModel` fields)
  `pub(crate)` + accessors — the only thing letting a caller break the
  `edges ⊆ nodes²` invariant. (M)
- `lib.rs` re-exports / prelude for the core types; `#[non_exhaustive]` on the error +
  event enums and `Recall`, pre-1.0. (S)
- Cache the recall-time region clustering instead of rebuilding it per `around`/`task`
  call. (M)

### D — Test coverage
- The CLI binary + the 9 `Opts::parse` have zero coverage; add compressor
  reversibility-under-eviction, `persist` disk save→load hash-stability, MCP
  parse-error envelopes, and an equal-score eviction-order tie-break test. (M)

### E — Hygiene
- Extract `main.rs` (2.3 KLoC) into per-domain command modules. (S)
- Consolidate the 3 license files; document the three-log taxonomy in `lib.rs`. (S)
- Port the security subsection + originality framing into the zh/ko/ar papers (en/fr/es
  have them) — *deferred by the maintainer; test counts already corrected.* (S)

---

## Remaining

### P0 — Correctness

- ✅ **`syn`-based AST parser** — *done.* Behind the `syn-parser` feature, the
  parser builds a real Rust AST (nested-module bodies, multi-line signatures,
  grouped `use`, impl methods), with the zero-dependency line-based heuristic as
  the fallback (used when the feature is off or a file does not parse as valid
  Rust). CI lints and tests both paths. See `src/parser.rs::syn_ast`.

### P1 — Depth

- ✅ **Canonical hash-chained log** — *done.* The primary `EventLog` is now
  tamper-evident: every `append` links the event into a SHA-256 chain over its
  replayable content (sequence + type + payload, excluding the non-deterministic
  `id`/`timestamp` so the chain stays reproducible). `EventLog::verify_integrity`
  detects any payload tamper, reorder, insertion or deletion, and `ccos verify` /
  `ccos replay` check it on every run. See `src/event_log.rs`.
3. **Semantic edges.** (L) Call-graph and data-flow edges, not just
   containment/dependency — richer causal propagation.

### P2 — Ergonomics

4. **Configurable scoring/paging/guard** via CLI flags or a config file instead
   of magic constants. (S)
5. **Benchmarks.** (S) `criterion` benches for `process_delta` to guard the
   `O(Δ)` claim against regressions.
6. **`analyze` extras.** (S) dead-symbol detection, per-file failure simulation,
   GraphML export to complement the existing Graphviz/DOT output.

### P3 — Hygiene

8. **Property tests.** (S) `proptest` for parser round-trips and graph invariants
   (dangling-free, bounded) under random edit sequences.
9. **Result-returning CLI commands** end-to-end (thread `Result` instead of
   ad-hoc exit codes). (S)

---

### Suggested order

~~`P0.1 (syn)`~~ ✅ → ~~`P1.2 (canonical log)`~~ ✅ → **`P2.5 (benches)`** (next) →
`P1.3 (semantic edges)` → polish. P2.4 and P3.7 are quick wins
that can land anytime.
