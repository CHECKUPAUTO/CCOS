# Batch resolution — deferring the whole-graph passes turns O(N²) ingestion into O(N)

> Reproduce: `cargo run --release --example ingest_profile`

The ingestion profiler (`docs/` companion to `examples/ingest_profile.rs`) localised the cost of
turning source into the causal graph. Parsing (the `syn` AST) is cheap; the cost is the three
**whole-graph resolve passes** — `link_module_imports` (imports → `DependsOn`/`Contains`),
`resolve_symbol_calls` (`Calls`), `resolve_data_flow` (`DataFlow`). The first fix, **B1**, made
`add_edge` dedup O(1) instead of an O(E) linear scan, dropping ingestion from ~cubic to ~quadratic.
The remaining quadratic is structural: **per-file ingestion re-runs all three whole-graph passes
after every file**, so a batch of N files pays N resolutions of an O(N) graph.

This note records **B2-batch**: the passes are *order-independent pure functions of the final node +
pending-ref set*, so running them **once at the batch boundary** instead of per file collapses the
cost to a single O(N) pass — and what it revealed about resolution **semantics**.

## The measurement

`examples/ingest_profile.rs`, synthetic corpus (20 fns/file, cross-file `use` + call + shared const),
no paging. "Per-file" re-runs the three passes after each file (the historical `ingest_source`
pattern); "B2-batch" ingests every file first, then resolves once:

```
# Scaling — whole-graph resolve re-run per file (the real ingest pattern)
  files   resolve total(ms)   ratio when files ×2
    150             790.3     —
    300            3160.7    4.00      ← ×4 per doubling  ⇒ O(N²)
    600           15596.0    4.93

# B2-batch — defer, then resolve ONCE after all files
  files   batch resolve(ms)   ratio when files ×2
    150              13.4     —
    300              32.5    2.43      ← ×~2.5 per doubling ⇒ O(N)
    600              89.5    2.75
```

At 600 files the batch resolves in **89.5 ms vs 15,596 ms — ~174× faster**, and the scaling is linear
(~×2.5 per doubling) instead of quadratic (~×4.9). The win grows with N, exactly as the
O(N²)→O(N) shape predicts.

`CcosMemory::ingest_deferred` + `CcosMemory::resolve` expose this directly: `ingest_deferred` records a
file and marks resolution pending (no passes); `resolve` runs the three passes **once** and clears the
flag (idempotent and near-free when clean). The eager `CcosMemory::ingest_source` is unchanged — it is
`ingest_deferred` + `resolve`, so a single-file ingest still leaves a fully-resolved graph (every
`&self` reader — `recall`, serialise — sees resolved edges, the contract the whole test suite relies
on). A `debug_assert` in `recall`/`to_json`/`checkpoint` turns any future "deferred ingest, then read
without resolve" into a loud failure.

## The honest subtlety — eager and batch are *not* always the same graph

Deferring resolution is **not** a pure speedup: it changes *when* resolution sees the graph, and the
two answers can differ. Resolution is **resolve-uniquely-or-skip** but **add-only** (it never removes
an edge). Consider a name that is globally-unique when a caller is ingested, then made ambiguous by a
later file:

```
src/a.rs       pub fn target() -> i32 { 1 }      // defines target
src/caller.rs  pub fn run() -> i32 { target() }  // calls target — unique *now*
src/b.rs       pub fn target() -> i32 { 2 }      // a SECOND target — now ambiguous
```

- **Eager (per-file, incremental):** at `caller.rs`, `target` is globally-unique, so
  `resolve_symbol_calls` adds `run → a::target` (`Calls`). At `b.rs` the name is ambiguous, so it adds
  nothing — but **the earlier edge stays**. Final graph: `run → a::target` exists. The edge is an
  artefact of *ingestion order* (a.rs happened to arrive before b.rs).
- **Batch (resolve once, final-state):** the single pass sees two `target`s, the call is ambiguous,
  resolve-uniquely-or-skip declines. Final graph: **no** `run → target` edge.

This is verified in `external_memory.rs`:
`eager_keeps_stale_edge_that_batch_drops_under_late_ambiguity`. The batch answer is the **cleaner**
one — the call genuinely *is* ambiguous in the complete program, and inventing a particular target
from arrival order is exactly what resolve-uniquely-or-skip exists to avoid. The batch path is also
**order-independent** (`deferred_batch_is_order_independent`): any ingest order yields the identical
graph, because it is a pure function of the final state. On an **unambiguous** corpus (the common
case) the two paths are identical (`deferred_batch_equals_eager_on_unambiguous_corpus`).

## Why the replayable path stays eager (for now)

CCOS's sacred invariant is `replay == live`: replaying the op log must reconstruct the identical
graph. Live ingestion via `AgentSession::ingest` is **eager** (incremental), and `replay_to` replays
those ops **eagerly**, op by op — so replay reproduces live's exact incremental sequence, *including*
any order-dependent stale edge. If the replay path were switched to batch (final-state) while live
stayed eager, the two would diverge under late-arriving ambiguity and break `replay == live` — a
violation the current property tests would *not* catch, because their generators don't produce
colliding names. So this change keeps the `AgentSession` path **eager and unchanged**; the batch
primitive is for callers without a replay log (one-shot analysis, bulk loaders, the profiler).

## The follow-up — order-independent resolution (makes eager ≡ batch everywhere)

The clean way to get the batch speedup on *every* path (including replay) is to make resolution itself
**order-independent**: prune the resolution-owned edges and rebuild from the final state on each run,
so eager and batch always agree and replay can batch safely. The edge ownership is already mapped and
is clean enough to make this tractable:

| Edge type           | Created by                                   | Resolution-owned?                    |
|---------------------|----------------------------------------------|--------------------------------------|
| `Calls`             | `resolve_symbol_calls` only                  | **yes** (all)                        |
| `DataFlow`          | `resolve_data_flow` only                     | **yes** (all)                        |
| `DependsOn`         | parser (`file:→use:`, `use:→dep:`) + imports | only `file:→file:` (import edges)    |
| `Contains`          | parser (`file:→mod:/sym:`) + module hierarchy| only `file:→file:` (hierarchy edges) |
| `Supports`/`Contradicts` | assertion path                          | no                                   |

So a prune step removes `Calls`, `DataFlow`, and the `file:→file:` `DependsOn`/`Contains` edges (then
the three passes rebuild), leaving every parser/assertion edge untouched. That also *fixes* the stale
edge above (eager would then drop it too). It is a behaviour change to a long-stable subsystem, so it
is deferred to its own focused, full-suite-validated change (B2-full), rather than riding along with
this perf work.

**Bottom line:** measure first. The bottleneck was algorithmic (per-file whole-graph re-resolution),
not cache layout — DOD/SoA would have shaved a constant factor off the wrong thing. Deferring the
passes to the batch boundary is ~174× at 600 files, and surfaced a real semantic distinction
(incremental vs final-state resolution) that we now measure, test, and document instead of papering
over.
