# CCOS: A Causal-Context Operating System for LLM Coding Agents

**A research prototype treating LLM working-context as a paged, causally-scored, deterministically-replayable resource.**

---

## Abstract

Large-language-model (LLM) coding agents are bottlenecked not by reasoning but
by **context management**: deciding which fragments of a large codebase to hold
in a bounded prompt window, when to evict them, and how to recover when a
retrieved fact turns out to be wrong. We present **CCOS** (Causal Context
Operating System), a kernel that reframes this problem using operating-system
abstractions. Source code is parsed into a **causal memory graph** whose nodes
(files, modules, symbols, imports) are scored by a blend of intrinsic
importance, *failure relevance*, recency and access frequency. A bounded
**context window** is paged in and out of this graph the way an OS pages memory,
and every state transition is recorded in an append-only, **deterministically
replayable** event log backed by a tamper-evident hash chain. We describe the
architecture, the core algorithms, and an audit-driven hardening pass that
eliminated an unbounded edge-leak (reducing a 10,000-cycle workload from 9,036
edges and an 11× slowdown to ~30 edges and a 1.08× slowdown), closed a guard
bypass, and made eviction deterministic. The prototype is 6 KLoC of Rust, passes
156 tests with zero linter warnings, and can analyze its own source tree.

---

## 1. Introduction

A coding agent operating over a repository of thousands of files cannot fit the
repository into a single prompt. It must continuously answer three questions:

1. **What is relevant now?** — which symbols/files belong in the prompt for the
   current task.
2. **What do we evict?** — when the budget is exceeded, which context to drop.
3. **What just broke?** — when an action fails (a test, a build, a wrong import),
   which context becomes *more* relevant because it is causally implicated.

Conventional retrieval-augmented pipelines treat this as a similarity-search
problem. CCOS instead treats it as **resource management**, and borrows the
mature vocabulary of operating systems:

| OS concept              | CCOS analogue                                                |
| ----------------------- | ------------------------------------------------------------ |
| Page / working set      | Graph node (file, module, symbol, import)                    |
| RAM ↔ VRAM paging        | `select_context_window()` + `enforce_paging()`               |
| Scheduler priority      | Causal score (importance · failure · recency · access)       |
| Page fault → high prio  | Failure detection → weighted propagation across causal edges |
| Write-ahead log         | Append-only `EventLog`                                       |
| Merkle / audit log      | Hash-chained `DistributedEventLog`                           |
| Syscall validation      | `GuardLayer` over every model response                       |
| N-version programming   | Multi-model `ConsensusEngine`                                |

This paper documents the design, the algorithms, and the measured behavior of
the prototype after a correctness-focused audit.

## 2. Design principles

- **P1 — Causality over similarity.** Relevance is propagated along *dependency
  and containment edges*, not just embedding distance. A failed node raises the
  priority of its causal neighbors.
- **P2 — Boundedness.** Memory (nodes *and* edges) must stay bounded under
  arbitrarily long sessions. The kernel maintains the invariant
  `edges ⊆ nodes × nodes` at all times.
- **P3 — Determinism.** Given the same event log, replay must reconstruct
  identical state. This rules out reliance on hash-map iteration order; all
  ordering decisions are total and reproducible.
- **P4 — Defense in depth.** Every LLM response passes a guard that guarantees a
  valid, bounded, sanitized output — or a deterministic fallback.
- **P5 — Auditability.** Every transition is logged; the log is tamper-evident.

## 3. Architecture

```
            ┌─────────────┐   register/Δ   ┌──────────────────────────┐
 .rs files →│   parser    │───────────────▶│  IncrementalGraphEngine   │
            └─────────────┘                └────────────┬─────────────┘
                                                         │ O(Δ) mutations
   ┌─────────┐  validate   ┌─────────┐                  ▼
   │   llm   │────────────▶│  guard  │          ┌──────────────────┐
   └─────────┘  sanitize   └─────────┘          │   MemoryGraph    │
        ▲                                        │ scoring/paging/   │
   consensus / adversarial                       │ failure-propag.   │
        │                                        └────────┬─────────┘
        ▼                                                 │ snapshots
  ConsensusEngine                ┌──────────────────────────────────────┐
                                 │ EventLog + DistributedEventLog         │
                                 │ (deterministic + hash-chained replay)  │
                                 └──────────────────────────────────────┘
```

Nine library modules (`parser`, `memory`, `incremental`, `event_log`,
`distributed_event_log`, `llm`, `guard`, `consensus`, `adversarial`) plus a
`persist` layer compose the kernel; a CLI (`demo`, `analyze`, `verify`,
`replay`, `chaos`) drives them.

## 4. Core algorithms

### 4.1 Causal scoring

Each node *n* carries `base_importance`, `failure_relevance`, `recency` and
`access_count`. Its score is a bounded linear blend:

```
score(n) = clamp₀¹( 0.15·base(n)
                  + 0.50·failure(n)
                  + 0.30·recency(n)
                  + 0.05·ln(max(1, access(n))) )
```

The dominant term is **failure relevance** (weight 0.50): a node implicated in a
fault is the strongest candidate for retention. Recency (0.30) decays
multiplicatively each kernel tick (`recency ← max(0.01, 0.95·recency)`), giving
an exponential forgetting curve. The logarithmic access term rewards
frequently-touched hubs without letting them dominate.

### 4.2 Failure detection and propagation

When a node fails, its `failure_relevance` is set and the signal propagates
breadth-first along outgoing edges with geometric attenuation by depth:

```
prop(t) = base_value · weight(s→t) · 0.8ᵈᵉᵖᵗʰ
failure(t) ← clamp₀¹( failure(t) + prop(t) )
```

Propagation is bounded by `max_depth`, so a fault influences a *causal
neighborhood* rather than the whole graph. The propagated nodes also have their
recency refreshed, pulling them toward the working set.

### 4.3 Incremental O(Δ) graph maintenance

On a file edit, the `IncrementalGraphEngine` compares content hashes to classify
the change (`FileAdded` / `FileModified` / `FileRemoved` / `NoChange`). For a
modification it evicts only that file's subgraph (nodes keyed by the
`file:`/`mod:`/`use:`/`sym:` + path prefix) and re-ingests it — cost proportional
to the *change*, not the repository. This is the property that lets the kernel
keep up with an editing agent.

### 4.4 Deterministic paging and eviction

When the node count exceeds `max_in_memory_nodes`, the kernel evicts the
lowest-scored nodes. The audit revealed two problems here, both fixed (§6):

- **Eviction was non-deterministic** when scores tied (it depended on hash-map
  order), violating P3. Eviction now sorts by *(score, NodeId)* — a **total
  order** — so identical builds evict identically.
- **Eviction left dangling edges.** Paging ran *re-entrantly* inside node
  insertion, so an edge could be attached to a node that had just been evicted,
  and such edges were never reclaimed. The fix (§6) restores invariant P2.

### 4.5 The guard layer

Every model response is validated and sanitized: control characters are
stripped, output is length-bounded, and the payload must parse as a **single,
whole** JSON value within a configured nesting depth. Failing any check, the
guard substitutes a deterministic, always-valid fallback. The guard's contract
is a safety invariant: **its output is always valid JSON**. §6 describes a
bypass (prefix-acceptance) that this audit closed, and §5/§6 the chaos harness
that empirically stresses the invariant.

### 4.6 Event sourcing and deterministic replay

State is derived from an append-only log of typed events
(`Parsing`, `GraphMutation`, `NodeUpserted`, `EdgeAdded`, `LlmCall`,
`GuardCheck`, `FailureDetection`, `FailurePropagation`, `Snapshot`,
`CycleEvent`, …). A `ReplayHandler` either folds the log into summary statistics
(`EventReplayer`) or — via `record_graph` + `GraphReconstructor` — **rebuilds the
graph itself**, faithfully, from `NodeUpserted`/`EdgeAdded` events. Because all
kernel ordering is total (P3), replay is reproducible. This closes the
event-sourcing loop: state is fully derivable from the log.

### 4.7 Tamper-evident distributed log

In parallel, a `DistributedEventLog` chains each event to its predecessor:

```
hashᵢ = SHA256( idᵢ ‖ payloadᵢ ‖ tsᵢ ‖ hashᵢ₋₁ ),   hash₋₁ = "GENESIS"
```

Any mutation to a past event invalidates every subsequent link, which
`verify_integrity()` detects. This gives an auditable, append-only history
suitable for multi-agent or untrusted-transport settings.

### 4.8 Multi-model consensus

For high-stakes queries the kernel can fan a prompt across several models and
resolve their (guarded) outputs by majority or **confidence-weighted** vote:

```
ratio = Σ_{v ∈ winning} conf(v)  /  Σ_{v} conf(v)
reached = ratio ≥ threshold
```

This is N-version programming for hallucination resistance.

### 4.9 Adversarial hardening

An `AdversarialEngine` injects four fault classes — JSON corruption,
hallucination, prompt injection, and timeout/empty responses — to continuously
exercise the guard and the graph. It powers both the test suite and the `ccos
chaos` command.

## 5. Implementation

CCOS is ~6,000 lines of Rust (edition 2021), dependency-light (`tokio`,
`reqwest`, `serde`, `sha2`, `chrono`, `uuid`, `rand`). The parser is a
**line-based heuristic** rather than a full `syn` AST — a deliberate trade-off
for zero parse-dependencies, at the cost of missing multi-line declarations
(see §7). The CLI exposes:

```
ccos demo                                  end-to-end subsystem demo
ccos analyze <path> [--json|--cycles|--dot FILE|--out FILE]   ingest real .rs files
ccos verify  <snapshot.json>               re-check hash chain + integrity
ccos replay  <snapshot.json>               deterministic event-log replay + reconstruction
ccos diff    <a.json> <b.json>             structural diff + causal-score drift
ccos failure <snapshot.json> <node-id>     inject a fault and propagate it
ccos chaos   [--iters N]                   fuzz the guard adversarially
```

## 6. Evaluation

### 6.1 The edge-leak fix (boundedness, P2)

The headline finding of the audit. A 10,000-cycle mutation workload was run
before and after the fix:

| Metric (10k cycles)         | Before     | After    |
| --------------------------- | ---------- | -------- |
| Nodes (paging cap 200)      | 200        | 200      |
| **Edges**                   | **9,036**  | **~30**  |
| Dangling edges              | 9,036 (100%) | **0**  |
| Per-cycle time (final)      | 3.26 ms    | 0.29 ms  |
| First-tenth → last-tenth    | **11.05×** | **1.08×**|
| Wall-clock                  | 17.5 s     | 2.9 s    |

The root cause was re-entrant paging (§4.4): edges were attached to nodes that
paging had just evicted, and were never reclaimed, so the edge set grew `O(cycles)`
while the node set stayed capped. The fix (a) makes `add_edge` reject endpoints
that are absent, (b) prunes defensively after paging, and (c) is guarded by a
regression suite asserting zero dangling edges and bounded growth.

### 6.2 Determinism (P3)

With *(score, NodeId)* tie-breaking, building the same graph twice under paging
pressure produces an **identical surviving node set** and identical snapshot
hashes (`tests/graph_invariants.rs::eviction_is_deterministic`,
`tests/snapshot_differential.rs`).

### 6.3 Guard safety under chaos (P4)

`ccos chaos --iters 2000` drives 2,000 adversarial payloads through the guard:

```
  Iterations:            2000
  Guard passed:          207
  Guard blocked:         1793
  Invalid guard outputs: 0     ← safety invariant holds
```

Across all four fault classes, the guard **never** emitted invalid JSON. The
audit also closed a bypass where the guard accepted any valid *prefix*
(e.g. `{"ok":1} <injected text>`); validation now requires the whole payload.

### 6.4 Tamper-evidence (P5)

Mutating any stored hash or payload is detected by `verify_integrity()`
(`distributed_event_log::tests::test_chain_detects_tampering`); `ccos verify`
exposes this for saved snapshots.

### 6.5 Self-hosting

`ccos analyze src` ingests CCOS's own source tree into a ~350-node / ~400-edge
graph with **zero dangling edges**, ranking `dep:std`, `dep:serde`, `dep:ccos`
as the highest-scored (most-referenced) nodes — a sanity check that the causal
scoring surfaces genuine hubs.

### 6.6 Event-sourcing round-trip

`ccos analyze src --out run.json` records the graph as `NodeUpserted`/`EdgeAdded`
events; `ccos replay run.json` then **rebuilds the graph from the log alone** and
reports `matches snapshot: true` — an identical node/edge set — confirming state
is fully derivable from the event stream (`GraphReconstructor`).

### 6.7 Test posture

156 unit + integration tests pass; `cargo clippy --all-targets` is warning-clean.
Stress harnesses (10k-cycle stability, snapshot differential, replay
consistency, adversarial suite) run in CI-friendly time.

## 7. Limitations

- **Heuristic parser.** Line-based extraction misses multi-line signatures,
  nested-module bodies, grouped `use` imports and macro expansions. A `syn`-based
  AST is the top future-work item.
- **No semantic edges.** Edges capture containment/dependency, not call graphs or
  data flow.
- **Consensus/adversarial/distributed-log** are wired into the CLI but the LLM
  path is only exercised against an Ollama-style endpoint; offline runs fall back
  deterministically.
- **In-memory only at runtime** (persistence is explicit via `--out`/`verify`).

## 8. Related work

CCOS draws on **virtual memory & paging** (Denning's working-set model),
**event sourcing / CQRS** and write-ahead logging, **Merkle/hash chains** for
tamper-evidence, **N-version programming** for fault tolerance, and the recent
line of work on **memory-augmented and retrieval-augmented agents**. Its novelty
is the synthesis: a *causal* scoring function with failure propagation, made
deterministic and auditable end-to-end.

## 9. Future work

In priority order (tracked in `ROADMAP.md`): (1) `syn`-based AST parsing;
(2) call-graph / data-flow (semantic) edges; (3) folding tamper-evidence into the
primary log so every run is auditable; (4) configurable scoring weights and
benchmarking of the O(Δ) claim with `criterion`; (5) property-based testing of
the graph invariants under random edit sequences.

## 10. Conclusion

By treating LLM context as a managed OS resource — paged, causally scored,
deterministically replayable and guarded — CCOS turns ad-hoc context juggling
into a system with explicit invariants. The audit reported here shows the value
of those invariants: a single re-entrancy bug had silently broken boundedness
and the O(Δ) guarantee, and making the invariants *testable* both surfaced and
fixed it. The prototype is small, fast, and self-hosting, and provides a concrete
substrate for further research on causal context management.

---

### Reproducibility

```bash
cargo test                     # 156 tests
cargo run -- analyze src --cycles
cargo run -- analyze src --out run.json
cargo run -- verify run.json && cargo run -- replay run.json
cargo run -- chaos --iters 2000
```

*This document describes a research prototype; numbers are from local runs and
will vary by machine.*
