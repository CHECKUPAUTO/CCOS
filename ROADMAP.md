# CCOS — Improvement Plan

Prioritized roadmap from the audit. Items are grouped by priority; each notes
the rationale and rough effort (S/M/L).

## ✅ Done in this audit pass

- **Fixed critical edge leak.** `enforce_paging()` ran re-entrantly inside
  `upsert_node()`, so `add_edge()` could attach edges to just-evicted nodes.
  Dangling edges grew `O(cycles)` (9,000+ edges for 200 nodes), breaking the
  `O(Δ)` promise and failing the stability budget. `add_edge` now rejects
  dangling endpoints, `enforce_paging` prunes defensively, and a regression
  suite (`tests/graph_invariants.rs`) locks the invariant.
- **Deterministic eviction/ordering.** Tie-broke paging, context selection and
  score listing on `NodeId` so replays/snapshots are reproducible.
- **Hardened the guard.** `is_valid_json` now requires the *whole* payload to
  parse (was accepting any valid prefix → trailing injection/hallucination
  slipped through). Replaced tautological (`x || !x`) adversarial assertions
  with real safety checks.
- **Added a real CLI** (`demo`, `analyze <path>`, `help`, `version`) — replacing
  the hard-coded demo that ended in `process::exit(0)`.
- **Docs**: `README.md`, crate-level rustdoc, this roadmap. Zero clippy warnings.

---

## P0 — Correctness & integrity

1. **Replace the heuristic parser with a real AST (`syn`).** (L)
   The current line-based parser misses multi-line signatures, nested module
   bodies, attributes and `use` groups (`use a::{b, c}`). `syn` gives correct
   structure and unlocks call-graph edges. Keep the line parser behind a feature
   flag as a zero-dep fallback.
2. **Enforce `GuardConfig::max_nesting_depth`.** (S)
   The field exists but is never applied; deeply nested JSON is accepted up to
   serde's default limit. Count depth during validation and reject beyond the
   configured bound.
3. **True snapshot/replay reconstruction.** (M)
   `EventLog::take_snapshot` currently maps `len → len` (a placeholder). Make a
   snapshot capture enough state (or deltas) to *rebuild* the graph, and add a
   test that replays the log into a graph byte-identical to the live one.

## P1 — Wire in the unused subsystems

4. **Integrate `consensus` + `llm` multi-model.** (M)
   Add `LlmClient::query_many()` and resolve via `ConsensusEngine` in the demo /
   a new `analyze --consensus` path. Today `consensus` is library-only.
5. **Integrate `adversarial` into a fuzz/chaos test mode.** (S)
   Drive `analyze`/`demo` through `AdversarialEngine` to continuously exercise
   the guard, instead of only in isolated tests.
6. **Adopt `DistributedEventLog` as the canonical log.** (M)
   Fold the hash-chain/tamper-evidence into the main `EventLog` (or have the
   kernel write both) so integrity verification covers real runs.

## P2 — Capability & ergonomics

7. **Persistence.** (M) Serialize graph + event log to disk
   (`ccos save/load`); enable cross-session replay and incremental re-analysis.
8. **Recency clock wired to real cycles.** (S)
   `MemoryGraph::tick()` (recency decay) is implemented but never called by the
   kernel; call it per cycle so recency actually decays over a session.
9. **Richer `analyze` output.** (S) `--json` event-log export, dependency
   cycles, dead-symbol detection, per-file failure simulation.
10. **CI + bench.** (S) GitHub Actions for `fmt`/`clippy -D warnings`/`test`;
    add `criterion` benches for `process_delta` to guard the `O(Δ)` claim.

## P3 — Hygiene

11. **Config surface.** Make paging thresholds, scoring weights and guard limits
    configurable via CLI flags / a config file instead of magic constants.
12. **Error handling.** Replace remaining `expect` in `LlmClient::new` with a
    fallible constructor; return `Result` from CLI commands.
13. **Property tests.** Add `proptest` for parser round-trips and graph
    invariants (dangling-free, bounded) under random edit sequences.

---

### Suggested order

`P0.1 (syn)` → `P0.2 / P0.3` → `P1.4–6 (wire subsystems)` → `P2.7 (persistence)`
→ `P2.10 (CI/bench)` → polish. P0.2, P2.8 and P3.12 are quick wins that can land
immediately.
