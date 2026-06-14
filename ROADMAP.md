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

---

## Remaining

### P0 — Correctness

1. **`syn`-based AST parser.** (L) The line-based parser misses multi-line
   signatures, nested-module bodies, grouped `use` and macros. Put it behind a
   feature flag with the heuristic parser as a zero-dep fallback. *(top item)*
2. **Full graph reconstruction from the event log.** (M) Replay currently folds
   the log into statistics only; make `GraphMutation`/`Parsing` events carry
   enough to rebuild a graph byte-identical to the live one, and assert it.

### P1 — Depth

3. **Canonical hash-chained log.** (M) Fold tamper-evidence into the primary
   `EventLog` (or mirror every kernel event), so integrity covers *all* runs, not
   just snapshots.
4. **Semantic edges.** (L) Call-graph and data-flow edges, not just
   containment/dependency — richer causal propagation.

### P2 — Ergonomics

5. **Configurable scoring/paging/guard** via CLI flags or a config file instead
   of magic constants. (S)
6. **Benchmarks.** (S) `criterion` benches for `process_delta` to guard the
   `O(Δ)` claim against regressions.
7. **`analyze` extras.** (S) dead-symbol detection, per-file failure simulation,
   DOT/GraphML export for visualization.

### P3 — Hygiene

8. **Property tests.** (S) `proptest` for parser round-trips and graph invariants
   (dangling-free, bounded) under random edit sequences.
9. **Result-returning CLI commands** end-to-end (thread `Result` instead of
   ad-hoc exit codes). (S)

---

### Suggested order

`P0.1 (syn)` → `P0.2 (replay reconstruction)` → `P1.3 (canonical log)` →
`P2.6 (benches)` → `P1.4 (semantic edges)` → polish. P2.5 and P3.8 are quick wins
that can land anytime.
