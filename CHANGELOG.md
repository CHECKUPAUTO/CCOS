# Changelog

All notable changes to CCOS are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project aims to
adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **External memory interface** (`external_memory` module) — a single, documented
  façade (`ExternalMemory` trait + `CcosMemory`) an agent uses to treat CCOS as
  its external working memory, unifying the kernel's separate pieces (causal
  graph, incremental parser, hash-chained logs, causal queries, region engine)
  behind a handful of verbs: `ingest_source`, `signal_failure`, `recall`
  (`WorkingSet` / `Around` region-anchored / `Task` lexical), `verify`, `stats`,
  `checkpoint` (+ inherent `open`, `impact`/`causes`, `tick`). Deterministic
  recall, tamper-evident persistence that round-trips, all result types
  `Serialize` (ready for a server/CLI layer). Reference guide in
  [`docs/MEMORY_INTERFACE.md`](docs/MEMORY_INTERFACE.md); 5 tests + a doctest.
- **`ccos eval --model M`** + live progress — override the active provider's model
  from the CLI (defaults to a local Ollama server if no provider env is set), and
  a live `[scenario] i/N tasks…` counter on stderr so long cloud-model runs no
  longer look hung.
- **Anthropic reasoning-model support in `ccos eval`** — read the `text` content
  block past a `<thinking>` block, larger `max_tokens`, no `temperature`; the
  grader also strips inline `<think>…</think>` blocks. Lets reasoning models
  (deepseek-v4-pro, qwen3.x, …) be graded on their final answer.
- **Causal-validation harness** (`scripts/causal_validation/`) — a closed-loop,
  LLM-free harness that tests CCOS's failure-propagation claim against the
  repository's **own Git history**. Phase 1 mines fix commits, reconstructs the
  pre-fix world in a throwaway worktree, and injects the fault at a changed file;
  Phase 2 scores `R_cov = |F_target ∩ WorkingSet_K| / |F_target|` per node budget
  `K` (arithmetic + geometric mean). Has a `--dry-run`; standard-library only.
  First run (on this thin history) honestly reports `R_cov ≈ 0.30`, flat across
  `K` — only the seed file is recovered — which localises a real limitation
  (failure pressure flows downstream only) and gives Phase 3 a concrete objective.
- **Tunable scoring weights** — the causal-score coefficients and the
  failure-propagation decay are now a `ScoringWeights` value on `MemoryGraph`
  (defaults reproduce the shipped constants exactly, regression-tested), settable
  via `set_scoring_weights` or the environment (`CCOS_W_BASE`, `CCOS_W_FAILURE`,
  `CCOS_W_RECENCY`, `CCOS_W_ACCESS`, `CCOS_FAILURE_DECAY`). `ccos analyze` and
  `ccos failure` honour them, so a hyperparameter search needs no recompile.
- **`ccos failure --max-nodes K --json`** — re-pages the graph to the bounded
  **WorkingSet_K** after fault injection and emits it (plus the affected set and
  the weights used) as JSON: the measurement hook the validation harness drives.
- **Anthropic Messages provider** for `ccos eval` — the real-LLM harness now also
  speaks `/v1/messages` (`ANTHROPIC_API_KEY` + optional `ANTHROPIC_BASE_URL` /
  `ANTHROPIC_MODEL`), so it can drive any Anthropic-compatible endpoint (e.g.
  DeepSeek at `https://api.deepseek.com/anthropic`, model `deepseek-v4-pro`).
- **Context Region Engine** (CCOS v0.3) — a spatial memory model above the causal
  graph. New modules `context_region`, `region_engine`, `context_policy`,
  `region_metrics`: nodes are embedded in a 3-D context space and clustered into
  **regions** (connected components of the cross-file causal-link graph) with a
  temperature and causal density; a region is hydrated as a `ContextWindow` under
  a **dynamic admission policy**. Five new event types
  (`RegionCreated/Activated/Merged/Evicted/ContextWindowGenerated`) keep it
  event-sourced and deterministically replayable (`replay_from`). New `ccos
  regions` CLI (cluster / activate / metrics), `scripts/region_benchmark.sh`,
  `docs/context_regions.md`, and an arXiv research paper in `docs/paper/`.
  Measured: region selection covers 97% of a task's causal neighbourhood vs 35%
  flat at ≈48% fewer tokens; regions 95.5% internally connected.
- **Hypothesis harness** (`experiment` module + `ccos experiment` CLI) — a
  deterministic, LLM-free simulation testing the *necessary condition* of the
  research thesis on modular synthetic repos with cross-file causal tasks of
  growing diameter, six strategies (RAG-dense/hybrid, GraphRAG-1hop/BFS,
  CCOS-from-query, CCOS-region), under an explicit success oracle, across two
  scenarios. **Clean query:** lexical RAG solves 0% while structure-aware methods
  (graph-BFS, CCOS) solve 100% — the lever is causal *structure*, not CCOS per se.
  **Noisy query** (a decoy out-scores the target lexically): every lexically-seeded
  method collapses to 0% — including graph-BFS and the `ccos-from-query` ablation —
  while only `ccos-region`, anchored on the workspace signal, survives at 100%. The
  ablation isolates the differentiator: the *anchor source*, not the region
  machinery. Folded into the paper (`docs/paper/` §8, two-scenario table).
- **Real-LLM evaluation harness** (`eval` module + `ccos eval` CLI) — tests the
  *sufficient* condition: auto-gradable multi-file "arithmetic causal chain" tasks
  whose answer requires the distant cause, six strategies assembling a budgeted
  window, sent to any OpenAI-compatible or Ollama endpoint. Reports task success,
  model-independent **oracle coverage**, and symbol-hallucination per diameter.
  Runs offline against a no-model stub (reproducing the coverage result on real
  file text) so the pipeline is CI-checked; real success numbers await a reachable
  model. Paper §9 updated (harness implemented; results pending a model).
- **Canonical tamper-evident `EventLog`** (ROADMAP P1.2): every appended event is
  linked into a SHA-256 hash chain over its replayable content (sequence + type +
  payload), so integrity now covers *all* runs, not just persisted snapshots.
  `EventLog::verify_integrity` detects payload tampering, reordering, insertion or
  deletion; `ccos verify` and `ccos replay` check it. The chain excludes the
  non-deterministic `id`/`timestamp`, so logs stay reproducible.
- **Optional `syn`-based AST parser** behind the `syn-parser` feature (ROADMAP
  P0.1): accurate parsing of nested-module bodies, multi-line signatures, grouped
  `use` and impl methods, with the zero-dependency line-based parser as the
  fallback (used when the feature is off or a file does not parse). CI lints
  (`--all-features`) and tests both paths.
- **Graph inspection commands** backed by a new read-only `query` module:
  - `ccos top <path> [--limit N] [--json]` — the hottest nodes by causal score
    (the working set the kernel would page in first).
  - `ccos blame <snapshot> <node-id> [--depth N] [--json]` — a node's upstream
    **causes** and downstream **blast radius**, walked deterministically in each
    edge direction.
  - `ccos export <snapshot> [--out FILE]` — export the causal graph as
    **GraphML** for Gephi / yEd / Cytoscape / networkx (deterministic, id-sorted).
- `query` module API: `impact_set`, `source_set`, `walk`, `hot_set`,
  `to_graphml`, plus `Reached` and `Direction` types (unit-tested).
- New docs: [`docs/USAGE.md`](docs/USAGE.md) (full command reference, end-to-end
  walkthrough, troubleshooting FAQ), [`CONTRIBUTING.md`](CONTRIBUTING.md), and
  this changelog.
- Annotated research **bibliography** ([`docs/BIBLIOGRAPHY.md`](docs/BIBLIOGRAPHY.md))
  — ~60 web-verified papers across 12 themes, each mapped to a CCOS module
  (context paging, causal graph, agents, guard/consensus/adversarial, hash-chained
  log & failure propagation).

### Changed

- The CI pipeline is **consolidated into a single cached job** (Format → Clippy
  `--all-features` → tests on both parser paths → Docs → CLI smoke) to keep
  GitHub Actions minute usage low on the private repo; `cargo audit` moved to a
  **weekly** `audit.yml` (and on-demand) instead of every push. Uses only
  GitHub-authored actions (`actions/checkout`, `actions/cache`).
- `README.md` and `docs/ARCHITECTURE.md` updated for the `query` module and the
  new commands.

### Fixed

- **Parser:** `strip_comments` now also removes inline `/* … */` block comments
  (string-aware), so symbols hidden in block comments are no longer extracted as
  real nodes. Multi-line block comments remain a known limitation of the
  line-based parser.

## [0.3.0] — Autonomous Context Runtime

### Added

- `scan`, `agents`, `benchmark` and `runtime` commands.
- New modules: `scheduler` (HOT/WARM/COLD context paging), `workspace` (async
  real-filesystem delta scanner), `agents` (Coder/Reviewer/Security behind an
  `Agent` trait), `persistence` (durable runtime state with integrity verify),
  and `benchmark` (cycle harness → JSON report).
- See [`CCOS_v0.3_REPORT.md`](CCOS_v0.3_REPORT.md) for the full report.

## [0.2.0] — Causal Kernel

### Added

- Causal memory graph with scoring, deterministic paging and failure
  propagation; incremental `O(Δ)` updates; append-only `EventLog` with
  deterministic replay and graph reconstruction; hash-chained
  `DistributedEventLog`; `GuardLayer`; multi-model `consensus`; `adversarial`
  fault injection; single-file `persist` snapshots.
- CLI: `demo`, `analyze`, `verify`, `replay`, `diff`, `failure`, `chaos`.

### Fixed

- Unbounded edge leak, guard prefix-bypass, non-deterministic eviction, and
  `max_nesting_depth` enforcement (see [`ROADMAP.md`](ROADMAP.md) → *Done*).

[Unreleased]: https://github.com/CHECKUPAUTO/CCOS/compare/v0.3.0...HEAD
