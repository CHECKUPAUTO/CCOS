# CCOS — Improvement Plan

Prioritized roadmap from the audit. Effort: S/M/L.

## 🚀 Direction — unbounded working memory (frugality × available RAM)

"Infinite" working memory is a **direction**, not a literal claim. CCOS keeps the
resident window tiny (*frugality*) and uses the machine's RAM as the backing
store, so the effective working memory is **as large as RAM allows relative to a
small window** — and that ratio is how far we stretch toward the asymptote. The
cognitive MMU, made real: page, don't drop.

- ✅ **Slice 1 — non-destructive eviction → a COLD tier (the swap).** Eviction now
  *demotes* a node (with its incident edges) into a COLD tier instead of dropping
  it; the resident set stays bounded by `max_in_memory_nodes`, the backing store
  grows into RAM, and `MemoryGraph::page_in` brings anything back. A failure on a
  demoted node resurrects it (a page fault) rather than erroring. Observable via
  `MemoryStats.cold`. Deterministic (sorted demotion, BTreeMap COLD store).
- ✅ **Slice 2 — page-fault from COLD on the read paths.** `page_fault` resurrects
  cold faulting files (via the cold-aware `signal_failure`); a `recall` *around* a
  demoted node pages it **and its cold neighbours** (`MemoryGraph::cold_neighbours`)
  back via `CcosMemory::ensure_resident`, wired into `AgentSession::recall*` and
  **reproduced on replay** so `replay == live`. `set_max_resident` exposes the
  frugal-window knob.
- ✅ **Slice 3 — spill COLD content to disk → RAM-bounded content, disk-unbounded.**
  An opt-in `attach_cold_spill(dir, inline_budget)` flushes the coldest COLD
  *content* blobs to a content-addressed on-disk store (SHA-256 keys, the CCR
  addressing scheme) once resident COLD content passes a byte budget, leaving a
  hash **stub** in RAM; `page_in` faults them back, **hash-verified** (a tampered
  or missing blob is a cold-miss, never a silent empty restore). Identical content
  is **deduplicated**; spill is lossless and deterministic (coldest-first). Off by
  default ⇒ byte-identical serialization, so the replay/snapshot invariants are
  untouched. `MemoryStats.cold_spilled{,_bytes}` surface it. **Honest scope:** only
  the unbounded *content* moves to disk — per-cold-node *metadata* (small) still
  grows in RAM (that's slice 4); blobs are stored verbatim (dedup, no codec yet);
  a snapshot taken with spill on needs its `dir` re-attached to restore (sidecar). (M)
- **Slice 4 — bound + compact COLD** (the final tier): summarise the coldest
  entries (CausalSumm) and bound the in-RAM metadata so the backing store itself
  stays frugal. (M)

## 🎯 Direction — better retrieval
- **Slice A — hybrid entry fusion** (lexical ⊕ semantic ⊕ causal, reciprocal-rank)
  before causal expansion. (M)
- **Slice B — opt-in learned embedder** behind a feature flag; INT4 TF-IDF stays
  the deterministic default (keep the replay invariant). (L)
- **Slice C — self-improving retrieval from the replayable log**: learn recall
  weights from which recalls actually *helped* — the auditable history as training
  data. The CCOS-native gem (better retrieval *and* reinforces the moat). (L)

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

### A — Wire in the unwired recent modules — ✅ done
- ✅ `embeddings` → a `Recall::Semantic` strategy (INT4 TF-IDF cosine entry), exposed
  via the MCP `recall` tool + `ccos memory`. *(Follow-up: cache the per-call store —
  see C.)*
- ✅ `eviction_policy` → blended into `MemoryGraph::enforce_paging` (untrained ⇒
  identical to the greedy; `train_eviction_policy` fits it offline). *(Follow-up:
  derive training transitions from the live op-log instead of an external feed.)*
- ✅ `injection_classifier` → `IngestReport.injection_score` / `injection_flagged`
  via a shared detector, on the live ingest/MCP path.

### B — Collapse duplicated abstractions
- ✅ One snapshot type: `persistence::RuntimeState` is now a type alias for
  `persist::KernelSnapshot` (the duplicate payload struct is gone); the
  integrity check is shared via `KernelSnapshot::verify_integrity` (used by the
  runtime restore and a new `KernelSnapshot::load_verified`).
- One event chain: collapse `distributed_event_log` onto `event_log`'s chain, or
  document why two exist. (M)
- One context selector: designate `CcosMemory::recall` canonical; demote
  `select_context_window` / `hot_set` / `hot_context` / `activate_region`. (M)
- One snapshot-error type (unify on `MemoryError`). (S)

### C — Encapsulation & API
- ✅ `MemoryGraph.{nodes,edges}` are now `pub(crate)` + read accessors (`node`,
  `node_mut`, `node_ids`, `node_entries`, `node_values`, `contains_node`, `edges`)
  — external callers can no longer push a dangling edge or orphan a node and break
  the `edges ⊆ nodes²` invariant. *(Still `pub` and could get the same treatment:
  `EventLog.events`, the `DistributedEventLog` fields, the `LinearModel` fields.)*
- ✅ `lib.rs` re-exports the core entry types at the crate root (`ccos::CcosMemory`,
  `Recall`, `MemoryGraph`, `AgentSession`, `KernelSnapshot`, …) with a crate-level
  doc-test; `#[non_exhaustive]` on the three **error** enums (`MemoryError`,
  `PersistenceError`, `ModelError`). The event/`Recall`/`NodeType`/`EdgeType` enums
  are **deliberately left exhaustive** — CCOS is its own only consumer, so the
  compiler's exhaustiveness check (catch a new variant nobody handled) is worth
  more here than cross-crate add-without-break.
- Cache the recall-time region clustering instead of rebuilding it per `around`/`task`
  call. (M)

### D — Test coverage
- ✅ The CLI now has coverage: a black-box `tests/cli.rs` (version/help, unknown
  command → exit 2, `analyze → verify → replay` round-trip, `sanitize --strict`
  on a Trojan-Source bidi override) driven via `CARGO_BIN_EXE_ccos`, plus unit
  tests for the option parsers (`analyze`/`top`/`chaos`/`blame`/`focus`, covering
  every distinct parse pattern — value flags, positionals, two-positionals, and
  the `--workspace` optional-arg branch). The remaining parsers reuse these
  patterns.
- ✅ Compressor reversibility-under-eviction — the test found and fixed a real
  (latent) bug: CCR eviction could drop a ref the current window had just handed
  back. See the CHANGELOG "Fixed" entry.
- Still TODO: `persist` disk save→load hash-stability, MCP parse-error envelopes,
  and an equal-score eviction-order tie-break test. (M)

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
