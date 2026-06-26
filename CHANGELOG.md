# Changelog

All notable changes to CCOS are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project aims to
adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **The real `syn` AST parser is now the default ingestion path** (was opt-in behind
  the `syn-parser` feature). On real code the old line heuristic is **36.5% wrong**
  structurally — import recall only 66.9% (grouped `use a::{b,c}` collapsed, so a third
  of the cross-file dependency edges were invisible) plus 145 hallucinated symbols
  (local consts promoted to top-level) — see `docs/MEASUREMENT_ast.md`. `syn` /
  `proc-macro2` are already in the dependency tree via serde, so defaulting to the AST
  pulls **no new dependency**; `--no-default-features` keeps the zero-extra-dependency
  heuristic, retained as the fallback for non-Rust / unparseable input. Still no async
  runtime and no TLS in the default build.

### Added

- **Data-flow semantic edges — `EdgeType::DataFlow` (ROADMAP P1.3, the second half of "semantic
  edges").** The `syn` AST captures in-body references to module-level `static`/`const` items
  (Slice 1: bare `SCREAMING_SNAKE` value paths — the Rust convention, which precisely excludes
  PascalCase types and snake_case fns/locals). A deterministic whole-graph pass
  (`MemoryGraph::resolve_data_flow`, run after call resolution) links each `reader → item` with a
  `DataFlow` edge when **exactly one** resident `static`/`const` of that name exists graph-wide
  (**resolve-uniquely-or-skip**, so a wrong edge is never invented) — the shared-global-state
  channel that call and import edges miss (a function reads a global defined in a file it never
  imports by name). The graph node carries `NodeType` not `SymbolKind`, so the parser marks the
  data-symbols at ingest; the references live in a transient `#[serde(skip)]` field (only the edges
  persist, rebuilt on the replay re-ingest → `replay == live` holds). Off on the heuristic path.
  A **scope guard** excludes locally-bound names (parameters, `let`s, fn-local `const`/`static`)
  from capture, so a local never mislinks to a same-named global — closing the cardinal false-edge
  an adversarial review found. Slice 1 covers bare references resolved global-unique; **Slice 2**
  (below) adds qualified `m::CONST`. Same-module disambiguation, write/read direction, and the rare
  residual (a bare `SCREAMING`-cased `use`-imported enum variant coinciding with a global const)
  remain later slices.

- **Data-flow Slice 2 — qualified `m::CONST` references.** In-body value paths whose *last* segment
  is `SCREAMING_SNAKE` (`config::MAX_RETRIES`, `crate::limits::MAX`, `self::FOO`) are now captured
  with their full `::`-path and resolved through a shared `resolve_qualified` helper — the *same*
  machinery qualified calls use, but against a **data-symbol-only** index, so a qualified ref can
  only ever land on a `static`/`const`, never a fn. **Resolve-uniquely-or-skip**: the module prefix
  is pinned to a defining file (crate-rooted, or an alias expanded through the file's imports), with
  no fallback to the bare global index — an unresolvable/ambiguous qualified ref adds no edge. The
  local-binding scope guard extends to qualified paths (a locally-bound head segment is skipped).

- **`data_flow_crux` measurement** (`examples/data_flow_crux.rs`, `docs/MEASUREMENT_data_flow_crux.md`).
  The data-flow analogue of the call/import crux: a reader names the const it reads (partial lexical
  signal), but two **co-readers** of the same global share only that one concept — swamped by their
  disjoint domain vocabulary, a true co-reader typically ranks below an unrelated decoy (lexical
  recall@1 ≈25 %, MRR ≈0.49). The data-flow graph recovers the shared-state link by construction —
  the cross-vocabulary channel a vector retriever cannot see.

- **Call-graph polish — renamed-import aliases & cross-impl-block self-calls.** Two precision gains,
  both resolve-uniquely-or-skip and deterministic: (1) `use a::b as c` now binds the local alias `c`
  to target `a::b` (top-level, in groups, nested groups), so a call `c()` / `c::X` rewrites onto the
  real target and never mislinks to a same-named sibling; (2) `self.method()` / `Self::method` now
  resolves across **all** impl blocks of a type — a `BTreeMap<type, methods>` unions every inherent
  and trait impl, so a self-call reaches a method defined in a *different* block of the same type,
  while a blanket `impl<T> .. for T` (type-variable Self) and two distinct types sharing a method
  name are strictly kept from cross-linking.

- **Spectral primitive — deterministic eigenvector centrality (`src/spectral.rs`, #13 first slice).**
  `eigenvector_centrality` computes the textbook `A x = λ x` ranking by power iteration on the
  **symmetrized**, `A + I`-shifted adjacency (the shift defeats the bipartite oscillation a DAG-like
  code graph would otherwise cause), L2-normalized, processed in sorted node order for byte-identical
  runs. Dependency-free and pure (read-only, not wired into scoring/CLI) — a clean brick complementary
  to the damped `MemoryGraph::eigencentrality`. Spectral regions, the temporal tensor, and any
  `scirust` fusion are deliberately deferred to a later design pass.

- **Call-graph semantic edges — `EdgeType::Calls` (ROADMAP P1.3, Slice 1).** The `syn` AST
  now extracts in-body function-call sites; a deterministic whole-graph pass
  (`MemoryGraph::resolve_symbol_calls`) resolves each `caller → callee` via a strict
  import-scoped → same-module → global-unique ladder (**resolve-uniquely-or-skip**, so a wrong
  edge is never invented) and adds a `Calls` edge — the fn→fn structure import edges miss.
  Slices 1–3 cover bare (`foo()`), qualified (`crate::m::foo()`, and `alias::foo()` expanded
  through the file's imports), and **`self.method()` / `Self::assoc()`** calls (resolved in the
  caller's own module, never via imports); arbitrary `x.bar()` (unknown receiver) stays deferred.
  Off on the heuristic path; call-sites held in a transient field so only the edges persist
  (snapshots unchanged, `replay == live` holds). Measured (`docs/MEASUREMENT_call_crux.md`,
  adversarially reviewed): a vector retriever recovers **direct** calls (it names the callee,
  recall@1 75 %) but collapses on **transitive** 2-hop calls (recall@1 0 %), which the call
  graph reaches by traversal — the call-level analogue of the import crux.

- **Node lifecycle state (`NodeState`: `Stable` / `Working` / `Orphan`).** Separates a
  node's *health/attention* from graph *topology* so it can't pollute the structural
  signal — a per-node enum field (not a tensor dimension; a node's state is single-valued).
  `Orphan` is excluded from the centrality calc and evicted first regardless of recency;
  `Working` is pinned resident as the current focus even as recency decays. Off by default
  (`Stable`) ⇒ centrality, score and snapshot are byte-identical until a state is set;
  `set_node_state` invalidates the centrality caches. See `docs/MEASUREMENT_node_lifecycle.md`
  (pillar in-degree 12→6 once dead dependents are excluded; real-work retention 1/6→6/6 when
  freshly-edited dead code is labeled). Companion to the off-by-default **eigenvector
  centrality** mode (`CentralityMode::Eigenvector`) added earlier in the series.

- **COLD entry-count bound — an on-disk husk index (slice 5c, "Lever 2"; the
  `O(1)`-resident COLD tier).** Slices 3–5b bounded each COLD entry's *size*; this
  bounds their *count*. The deep-spill tier no longer keeps one `BTreeMap` node per
  husk in RAM — husks live in a hand-rolled, dependency-free LSM-lite
  (`src/cold_index.rs`): immutable sorted segments with a sparse resident index, a
  memtable + flush, tombstone deletes + compaction, and a bounded LRU read cache, each
  verified standalone by a model-check property test before wiring. `MemoryGraph`'s
  resident `cold_deep` map is gone; `cold_neighbours` is answered `O(degree)` by a
  keyed on-disk **reverse-adjacency** index (`<dir>.radj`), and `flush_cold_tier`
  durabilises the indices at checkpoint. Measured (`examples/cold_count.rs`): **≈2 B
  per husk resident** (vs 146 B fully resident), 1 GiB at **~537 M husks**. Lossless
  round-trip, no-leak GC and crash recovery are property-/model-checked;
  dependency-free (`std` only); `replay == live` is untouched (the event log is the
  source of truth, the cold tier a rebuildable cache). See
  `docs/DESIGN_cold_entry_count.md`.
- **Natural-language queries match code identifiers (subword tokenization).** The
  TF-IDF tokenizer now splits each token on `snake_case` and `camelCase` boundaries,
  so `connection_pool_acquire` yields `connection`, `pool`, … — a query like
  "connection pool acquire" shared *zero* tokens with it before, making the semantic
  signal zero. Measured (`examples/identifier_recall.rs`): 6/6 NL queries recall their
  identifier-named target at rank ≤2 (overlap 0 → 3/3); on the `lsa_rerank` corpus the
  topic target's mean rank improves 11.8 → 2.0. Deterministic.
- **LSA re-ranking stage for recall (`set_lsa_rerank`, opt-in).** Wires the LSA
  embedder where #39 measured it earns its keep — *re-ranking* the recalled region
  (recall@k≥5), not entry selection (recall@1=0). A node's score is multiplied by
  `1 + w·max(0, cosine)` (only ever promotes). Measured (`examples/lsa_rerank.rs`):
  target mean rank 11.8 → 2.1; the honest limiter is entry selection (synonyms score
  ≈0), which re-ranking can't repair. Deterministic, `replay == live` untouched.

### Changed

- **Spill stubs hold a raw `[u8; 32]` hash, not a 64-char hex `String`** — −56 B and
  one fewer heap allocation per COLD spill/husk stub (serialized form unchanged via
  serde-hex). **Snapshots are byte-canonical** — the resident `nodes` `HashMap` now
  serializes in sorted key order, so identical state ⇒ byte-identical snapshot, not
  merely identical *sorted* hash. Both verified by property tests.

### Fixed

- **A COLD spill blob leaked on page-in.** When `page_in` faulted a blob back and
  dropped its last reference (content folded inline, or a husk removed), the on-disk
  blob became unreferenced but was never reclaimed — a slow disk leak no later
  `remove` could find. Caught by a new cross-tier hardening property test (lossless
  round-trip + no orphaned blobs under random op streams); `page_in` now reclaims the
  dropped blob in both paths. The headline `replay == live` invariant is now also
  **fuzzed** (byte-identical full-graph hash over random op streams), as is snapshot
  round-trip and the on-disk index's model.
- **An on-disk lossless codec for the spill store (LZSS, dependency-free).** Spill
  blobs are LZSS-compressed on write and verified on read (the key is the original
  content's SHA-256, so dedup is unchanged and any codec bug is a recoverable
  cold-miss); a `proptest` round-trip pins `decompress(compress(x)) == x`. Closes the
  "no codec yet" gap.

- **Structural-centrality scoring term** (from a design discussion — the one idea in
  that conversation CCOS's score didn't already have). `compute_node_score` gains a
  `w_centrality · ln(1 + in_degree)` term: a hub (a shared module / interface many
  nodes depend on) is structurally more important than a leaf, independent of recency.
  **Off by default** (`w_centrality = 0.0`, `skip_serializing_if` elides it) ⇒ the
  score is byte-identical to before and replay/snapshots are unchanged. In-degree is
  computed via a cache keyed on `edges.len()` (edges are append/`retain`-only) and is
  only built when the term is enabled, so the default path pays nothing.
  `CCOS_W_CENTRALITY` overrides it, and the log-tuner
  (`AgentSession::tune_recall_weights`) now learns it too (absolute candidates, since
  a multiplicative move can't escape 0). Deterministic.
- **COLD-tier deep-spill — bound the per-entry *resident* metadata, losslessly**
  (slices 5 & 5b; measure-then-fix, see `docs/MEASUREMENT_cold_ram.md`, reproduce with
  `examples/cold_ram.rs`). A measurement first showed slice 3 left the COLD tier's
  dominant RAM cost as per-entry **metadata** — ~2.8× the spilled content, ~60% of it
  edges — and that lossy edge-contraction is the *wrong* lever (it inflates that edge
  cost on hubs). So `set_cold_resident_budget(Some(b))` drives resident COLD metadata
  toward `b` by **deep-spilling** the coldest entries: each is archived *whole* to the
  content-addressed store and represented in RAM only by a compact `DeepHusk`
  (body-blob stub + the neighbour **ids** that `cold_neighbours`/region paging need),
  held in a separate `cold_deep` map. Because the husk is far smaller than a full
  `ColdNode`, *every* entry shrinks when spilled and the budget is actually reached —
  resident COLD metadata **halves (−50%)** on the 120K-node fixture (slice 5's
  full-husk first cut had stalled at ~11% against the `size_of::<ColdNode>()` floor).
  **Lossless** (the node faults back, hash-verified, on `page_in`; a missing/tampered
  body is a cold-miss, never a half-restore), **deterministic** (coldest-first), and
  **off by default** (`cold_deep` is `serde`-elided when empty and the budget is a
  runtime knob ⇒ byte-identical default snapshot/replay). Deep husks are *terminal*
  (excluded from further spill/compaction). Shrinks edges to ids — never adds bridge
  edges — so hubs get cheaper, not the O(degree²) blow-up contraction would cause.
  Observable via `cold_deep_spilled_count` / `is_deep_spilled`. **Honest scope:** this
  bounds the per-entry resident *size*; bounding the entry *count* (an on-disk husk
  index) remains future work.

### Performance

- **Per-recall caches make recall up to ~5700× faster at scale** (the perf pass —
  measure-then-fix; see `docs/MEASUREMENT_latency.md`, reproduce with
  `examples/recall_latency.rs`). A latency benchmark showed recall was super-linear
  in corpus size because every query recall rebuilt derived structures from scratch:
  `around`/`task` re-ran the whole **region clustering** (`initialize_regions`), and
  `semantic`/`hybrid` additionally re-fit the **embedding store** (and the LSA
  eigensolve under `learned-embed`). `CcosMemory` now memoises both behind a **graph
  version counter** bumped on every resident-graph mutation; a cache is reused only
  at the same version, so it is **never stale** and the result is byte-identical to a
  fresh rebuild — **determinism and `replay == live` are preserved** (a new test
  asserts a post-warm ingest is visible to the next recall; the full replay suite
  still passes). At 2000 nodes: `around` 75 ms → 13 µs, `semantic` ~42×, `hybrid`
  ~21×. (The first recall after a mutation still rebuilds; the win is on the repeated
  recalls between mutations — the common pattern.)

### Added

- **Recall-strategy measurement (`examples/recall_eval.rs`) + honest findings**
  (`docs/MEASUREMENT_recall.md`). An LLM-free benchmark on a synthetic corpus with
  ground-truth relevant files, comparing working-set / lexical / semantic / hybrid
  recall at a tight budget across three task types. Result: **hybrid fusion is
  measurably the best query strategy** (overall hit-rate 58% vs lexical 17% /
  semantic 21%; it alone recovers the target in the decoy+failure case) —
  validating slice A in measurement. The **opt-in LSA embedder does *not* help and
  can hurt** in CCOS's entry-selection use (drops hybrid to 38%), so it correctly
  stays off by default; the data, not assumption, sets the recommendation.
- **Opt-in learned semantic embedder (`learned-embed` feature)** — slice B of better
  retrieval, completing the arc. A new `src/lsa.rs` distils the deterministic INT4
  TF-IDF into a learned **latent-semantic (LSA / truncated-SVD) projection**: the top
  singular vectors of the corpus's document–term matrix, found by a fixed
  cyclic-Jacobi sweep (zero new dependencies, fully deterministic). It captures
  synonymy/transitivity raw TF-IDF can't — a query term that only *co-occurs* with a
  document's terms still matches it. `CausalEmbeddings::fit_and_embed_lsa` stores the
  projected vectors and `embed_query` projects queries the same way;
  `build_embeddings` uses it only under `--features learned-embed`, so the **default
  build stays raw INT4 TF-IDF, byte-identical and replayable** (the embedder's
  `projection` field is `skip_serializing_if = None`). *Honest scope:* LSA is a
  linear distillation, not a neural model; it helps most with enough documents to
  truncate; the eigensolve adds per-build cost, hence opt-in.
- **Self-improving retrieval from the replayable log** (slice C of better retrieval —
  the CCOS-native gem). A retrieval **reward** is read straight off the hash-chained
  timeline: for each recorded recall, was the node the agent engaged *next* (a
  failure signal / page-fault) present — at file granularity — in the window that
  recall would have produced? `AgentSession::retrieval_hit_rate` reports it;
  `tune_recall_weights` learns the `ScoringWeights` that maximise it by
  **deterministic coordinate ascent, evaluated by replay** (same log ⇒ same
  weights); `adopt_tuned_recall_weights` applies them **and records an `Op::Retune`**,
  so the learned policy is auditable and **reproduced on replay** — `replay == live`
  still holds. This is retrieval that trains on CCOS's own moat: the deterministic,
  replayable causal history. *Honest scope:* the reward is a proxy (the next failing
  node = the context recall should have surfaced); the optimiser is greedy (a local
  optimum) over the four scoring weights; evaluation is one replay per candidate, so
  it is an offline/maintenance call, not a hot path.
- **Hybrid entry fusion for recall** (slice A of better retrieval). A new
  `Recall::Hybrid(text)` resolves a free-text task's entry node by
  **reciprocal-rank fusion** of three independent rankings — lexical token
  overlap, semantic INT4-TF-IDF cosine, and the causal **active-failure focus** —
  before the usual causal-region expansion. RRF compares ranks (no cross-signal
  score calibration), so a node strong on any one axis can still surface while a
  node decent across several wins; `K = 60`. The causal vote is **sparse** — it
  ranks only nodes under failure pressure, so it abstains on a quiet graph (no
  spurious id-ordered bias) and speaks for the active problem region once a
  failure is signalled (the CCOS-native attention signal). Deterministic; wired
  through `recall()`, the MCP `recall` tool (`strategy:"hybrid"`), and the runtime
  recall CLI. `Recall::hybrid(text)` constructs it.
- **Compact the coldest COLD tail → a frugal backing store** (slice 4 of unbounded
  working memory, the deepest tier). A new, opt-in
  `CcosMemory::set_cold_content_budget(Some(bytes))` keeps total COLD *content*
  (inline + spilled) toward `bytes` by **lossily compacting** the coldest entries —
  routed by kind, code is skeletonised / prose summarised / JSON crushed
  (`CausalAst` / `CausalSumm` / `CausalCrusher`, reused as pure functions), and the
  full original is discarded. Deterministic (coldest-first by causal score), and
  **observable**: `is_compacted` and `MemoryStats.cold_compacted` report the lossy
  tier. This is where "infinite working memory as a *direction*" bottoms out — at
  the floor frugality wins, and CCOS compacts to a summary, **never silently
  drops**. **Off by default** ⇒ COLD stays lossless and serialization byte-identical
  (the `ColdNode.compacted` flag is `skip_serializing_if = false`; the budget is
  `serde(skip)`). *Honest scope:* this bounds the cold **content** footprint, not
  the entry **count** (the in-RAM stub map is still O(N) — an on-disk index is
  future work); compaction is lossy and, like spill, an operational mode layered on
  the deterministic default path, not part of replay.
- **Spill COLD content to disk → RAM-bounded content, disk-unbounded** (slice 3
  of unbounded working memory). A new, opt-in
  `CcosMemory::attach_cold_spill(dir, inline_budget)` flushes the coldest COLD
  *content* blobs to a content-addressed on-disk store (SHA-256 keys — the same
  addressing as the CCR store) once resident COLD content exceeds `inline_budget`
  bytes, dropping the blob from RAM and leaving a hash **stub**. `page_in` faults
  it back **hash-verified**: a tampered, truncated, or missing blob is a cold-miss,
  never a silent empty restore — so disk spill *extends* the integrity story.
  Identical content is **deduplicated**; the flush is lossless and deterministic
  (coldest-first by causal score, ties on id). **Off by default** ⇒ no spill,
  byte-identical serialization, replay/snapshot invariants untouched (the new
  `spill` stub is `skip_serializing_if = None`; the store handle is `serde(skip)`).
  `MemoryStats.cold_spilled` / `cold_spilled_bytes` surface it (via `ccos stats` /
  the MCP `stats` tool). *Honest scope:* only the unbounded **content** moves to
  disk — per-cold-node metadata still grows in RAM (slice 4); blobs are stored
  verbatim (dedup, no compression codec yet); a snapshot taken with spill active
  references blobs by hash and needs the `dir` re-attached to restore (a sidecar,
  like a swapfile).
- **Page-fault from the COLD tier on the read paths** (slice 2 of unbounded
  working memory). A `page_fault` now resurrects cold *faulting* files (its
  per-file `signal_failure` is cold-aware), and a `recall` **around** a demoted
  node pages it — and its cold neighbours (`MemoryGraph::cold_neighbours`) — back
  into the resident graph via the new `CcosMemory::ensure_resident`, wired into
  `AgentSession::recall` / `recall_compressed` / `recall_compressed_with_feedback`.
  The page-in is a deterministic, **replayed** side effect (`Op::Recall` reproduces
  it), so `replay == live` holds. New `CcosMemory::set_max_resident` configures the
  frugal resident-window size.
- **Non-destructive eviction → a COLD tier (the "swap").** First slice of the
  *unbounded working memory* direction (frugality × available RAM). Evicting a
  node from the resident graph now **demotes** it — with its incident edges — into
  a COLD tier instead of dropping it: the resident set stays capped by
  `max_in_memory_nodes`, the backing store grows into RAM, and any node can be
  paged back (`MemoryGraph::page_in`). A `signal_failure` on a demoted node
  **resurrects it from COLD** (a page fault) instead of erroring. `MemoryStats.cold`
  surfaces the tier (via `ccos stats` / the MCP `stats` tool). Deterministic
  (sorted demotion, `BTreeMap` COLD store); snapshots stay reproducible. See
  ROADMAP for the arc (disk-spill + compaction next).
- **Wired the recent modules onto the live path.** Three capabilities that were
  in-tree but unreachable from the live recall/ingest core are now connected:
  (1) **semantic recall** — a new `Recall::Semantic` strategy resolves a
  free-text task to its entry node by INT4 TF-IDF cosine (`embeddings`), exposed
  via the MCP `recall` tool and `ccos memory`; (2) **injection signal at ingest**
  — every `IngestReport` now carries `injection_score` / `injection_flagged` from
  a shared `InjectionDetector`, so the signal is recorded on the live path, not
  only in `ccos sanitize`; (3) **learned eviction** — `MemoryGraph::enforce_paging`
  now consults `EvictionPolicy`, blending its learned keep/evict preference into
  the eviction order. The policy is **untrained by default**, in which case paging
  is byte-identical to the deterministic greedy (never worse); `train_eviction_policy`
  fits it offline. All three preserve determinism/replay; each has a wiring test.
- **Input hardening — deterministic Unicode de-obfuscation + an injection
  signal** (`sanitizer`, `hashing_tokenizer`, `injection_classifier` modules).
  Hidden-character injection vectors — Trojan-Source bidi overrides
  (CVE-2021-42574), zero-width formatting, Unicode-Tags ASCII smuggling — are
  surfaced as explicit, auditable literals (`[U+202E RLO]`) at ingest
  (default-on in `ingest_source`; clean source is borrowed unchanged, zero
  copy), with findings in `IngestReport.anomalies` and the event-log hash taken
  over the cleaned form so a replay reproduces it. A deterministic
  feature-hashing tokenizer feeds a linear log-space (multinomial-Naive-Bayes)
  injection **signal** whose weights are locked in an immutable,
  SHA-256-verified blob, with a forensic per-feature explanation of every score;
  a held-out red-team measures F1 0.90 (precision 0.87, recall 0.93). Labelled a
  *signal, not a shield* by design (evaded by paraphrase; no character pass
  solves semantic injection). New `ccos sanitize` CLI command and the
  `train_injection` / `injection_redteam` examples. See
  [`docs/SECURITY.md`](docs/SECURITY.md).
- **Reversible context compression pipeline** (`compressor` module) — the real
  *compression* pass CCOS historically lacked, sitting downstream of the causal
  MMU's selection so the graph, the scoring, the paging and the hash-chain
  replay are untouched. Three deterministic compressors: `CausalCrusher`
  (columnar JSON collapse + null-drop + string back-refs), `CausalAST`
  (skeletonizes code — strips comments / blank lines / `use` imports, collapses
  long signature runs, renames `_`-temporaries to `$n`), `CausalSumm` (TextRank
  extractive summary **biased by the causal score**). No ML model, no
  stochastic step: everything is seed-stable and total-order tie-broken, so
  the replay / `postmortem` invariants hold. Measured on this repo's source:
  30–50 % token reduction on real Rust code (run `cargo run --example
  bench_compress --release`). Zero new dependencies.
- **CCR store + `ccos_retrieve` MCP tool** — every compressed item carries a
  12-char `ccr_ref` (truncated SHA-256 of the original); the host LLM calls
  `ccos_retrieve` to fetch the full text on demand (the CCOS equivalent of
  headroom's `headroom_retrieve`). Nothing is ever lost. `RecallItem` gains an
  optional `ccr_ref` field (serde-skipped when absent, so old snapshots still
  load).
- **Cross-item near-duplicate suppression** — a distilled MinHash (64 hashes,
  3-char shingles, FNV-1a + double-hashing, seed-stable) estimates Jaccard
  similarity over the *compressed* forms within a window; near-dup items are
  replaced by `// ~dup of <uri>` (their original stays retrievable). The causal
  graph dedups cross-file; this dedups *within* a window.
- **Budget feedback loop** (`CcosMemory::recall_compressed_with_feedback` /
  `AgentSession::recall_compressed_with_feedback`) — when compression shrinks
  the window below the token budget, the freed space is *re-spent* on more
  causal nodes (a second recall pass with a grown effective budget), so the
  host gets strictly more causal signal at the same emitted-token cost.
  Monotonic and bounded (max 3 rounds); stops at convergence. Measured: +11
  causal nodes on a 4096-token task recall vs a single compressed pass, while
  staying under budget.
- **`CausalAST` v2 knobs** — `enable_ast_v2` drops pure `use` lines (the causal
  graph already encodes the dependency) and `ast_signature_collapse_after`
  collapses a run of >N one-line `fn` signatures into the first N + `// (+M
  more signatures)`. `pub use` re-exports are kept.
- **Auto-tuner** (`CausalCompressor::auto_tune`) — deterministic coordinate
  descent over the config knobs (dedup threshold/on, AST v2/collapse, summary
  length, prose on, min-chars) to minimise the compressed-token count on a
  representative sample. `eval_config` is public for external benchmarks.
- **`ccos://session/context` compressed by default** — the resource now runs
  through `recall_compressed` unless `CCOS_COMPRESS_CONTEXT=0` (A/B escape
  hatch). The linearised form appends `// ccr_ref=… (call ccos_retrieve for
  full)` so the host knows the handle.
- **SCIRUST counterparts** — the algorithms were distilled from
  `scirust-nlp-advanced`, which gains four new modules: `bloom` (Bloom filter),
  `lsh` (MinHash-LSH band-and-bucket), `trie` (byte-radix shared-prefix
  compaction), `huffman` (canonical reversible entropy coding).
- **Causal embeddings** (`embeddings` module) — a zero-dependency TF-IDF
  embedder with a hashed vocabulary (128-dim default) whose vectors are
  **INT4-quantized** (distilled from SCIRUST's `elastic_kv_cache.rs` SLHAv2
  scheme: grouped absmax symmetric INT4, cosine error < 0.01). The
  [`CausalEmbeddings`] store is ~4× smaller than `f32` and powers a
  [`CcosMemory::semantic_entry`] for `Recall::Task` that down-weights
  ubiquitous tokens via IDF (catches "connection pool" → `db.rs` where a
  raw lexical overlap is distracted by the ubiquitous `fn`). Deterministic:
  the hashed vocab + `BTreeMap` store serialize bit-stable.
- **RL eviction policy** (`eviction_policy` module) — a tabular Q-learning
  agent (distilled from SCIRUST's `scirust-rl-algo::TabularQLearner`) that
  learns when to evict a node from the paging window based on a 4-bucket
  state (score / recency / failure-pressure / size). 162 cells max, serializes
  as a `BTreeMap`, bit-reproducible. **Advisory**: [`should_evict`] returns
  `false` when untrained, so the deterministic greedy stays the authority
  until the policy has learned a preference — turning it on is never worse
  than the status quo. Training is offline (`fit` over a replayed timeline
  with reward shaping for keep/evict decisions).

### Fixed

- **Audit pass 4 (hardening the unbounded-memory + retrieval slices).** Four
  adversarial auditors (one each for determinism, `replay == live`, default-path
  byte-identity, and resource bounds) confirmed the crown invariants hold on the
  default path, and surfaced three real issues now fixed:
  - **Spill-blob garbage collection (was an unbounded disk leak).** The on-disk
    spill store (`ColdSpill`) had no deletion path, so re-ingesting, removing, or
    compacting a previously-spilled node orphaned its blob forever. Added
    `ColdSpill::remove` and a **dedup-safe** `release_blob_if_orphan` (a blob is
    deleted only once no COLD entry still references its hash), wired into
    `upsert_node`, `remove_node`, and compaction. (Off by default — only matters
    when a spill store is attached.)
  - **Compaction floor no longer busy-loops.** An un-shrinkable cold entry was
    re-selected (and its blob re-read from disk) on every ingest while the tier
    stayed over budget. Such entries are now parked with a new `ColdNode.at_floor`
    flag and excluded from future compaction candidates (a fresh ingest drops the
    shadow, so the flag never goes stale). `skip_serializing_if` keeps the default
    serialization byte-identical.
  - **LSA corpus order pinned for determinism.** `build_embeddings` now sorts
    nodes by id before fitting, so the `learned-embed` LSA Gram-matrix f64 sum is
    independent of `HashMap` iteration order (the one place determinism rested on
    float-associativity rather than a fixed order). The default TF-IDF path was
    already order-free.

  Deferred to a perf pass (documented, not regressions): per-ingest `O(cold)`
  budget re-scans, per-recall `cold_neighbours` scan, and the per-recall
  embedding-store rebuild — all to be addressed with incremental counters/indices
  and a cached, dirty-invalidated embedding store.

### Changed

- **Unified the two snapshot types.** `persistence::RuntimeState` was a
  field-for-field duplicate of `persist::KernelSnapshot`; it is now a type alias
  for it (one state type, two on-disk layouts — single-file vs three-file
  directory). The load-time integrity check (both hash chains valid + no dangling
  edges) moved into the shared `KernelSnapshot::verify_integrity`, now also
  reachable via `KernelSnapshot::load_verified` and reused by the runtime restore.
  No caller changes (audit pass 3, section B).
- **Encapsulated `MemoryGraph.{nodes,edges}`** (now `pub(crate)`). External
  callers go through read accessors — `node`, `node_mut`, `node_ids`,
  `node_entries`, `node_values`, `contains_node`, `edges()` — instead of touching
  the maps directly, so the `edges ⊆ nodes²` invariant can no longer be broken
  from outside the crate (audit pass 3, section C). Internal behaviour is
  unchanged; a minor breaking change for any external consumer that read the
  fields.
- **Repositioned, honestly.** Measurements refute "causal regions retrieve better
  than RAG": on 70 real bug-fix commits causal selection ties (and at a tight
  budget loses to) a lexical TF-IDF retriever, and the crash-trace pivot is beaten
  by RAG-over-the-error-message. End-to-end (Phase 4, 30B + compiler-in-the-loop)
  CCOS and RAG resolve equally (2/10), **but CCOS uses 6.9× fewer context tokens
  (776 vs 5366)** — efficiency, not retrieval quality, is its measured advantage.
  CCOS's contribution is relocated from *retrieval* to a **frugal, deterministic,
  replayable, auditable** agent memory. README and the paper (title, abstract,
  contributions, time-travel section, Phase-4 efficiency result, conclusion)
  rewritten accordingly.

### Added

- **Deeper page-fault propagation.** A page-fault now injects failure pressure to
  depth **3** (was 2), configurable via `CCOS_PAGE_FAULT_DEPTH` — a Jetson field run
  showed depth 2 left a 3-hop-deep cause un-pressurised (the symptom got hot, the
  cause stayed cold and was evicted under a tight budget). The depth is recorded in
  the op-log so replay reproduces the exact pressure (old logs default to the
  historical depth of 2); determinism preserved.
- **Field-data collection.** `ccos postmortem <workspace> --json` dumps an
  analytics-ready field record (version, stats, hash-chain integrity, timeline,
  compaction floor, current working set) and exits — the non-interactive way to
  archive a session (e.g. on a cron, before compaction folds older steps away).
  `scripts/fleet_collect.sh` pulls workspaces from a fleet over `rsync` and writes a
  `session.json` per node (local-first; integrity is verified offline). Because the
  timeline replays bit-for-bit, a copied workspace reproduces the field run off-site.
  See [`docs/SELF_ANALYSIS.md`](docs/SELF_ANALYSIS.md) → *Collecting field data*.
- **Durable checkpoints + bare-metal notes.** Snapshots (`.ccos`) and the op-log
  (`.oplog`) are now written **durably and atomically** (`util::write_durable`: temp
  + `fsync` + atomic rename + directory `fsync`), so a power loss or killed daemon
  can't leave a truncated file — hardening the "replayable after a crash" guarantee
  (a plain `std::fs::write` only reaches the page cache). On by default. Adds
  `scripts/jetson_repro_env.sh` (pin a Jetson to max clocks for reproducible
  measurement — `nvpmodel`/`jetson_clocks`, no `nvidia-smi`/NUMA on Tegra), an
  optional `mimalloc` allocator feature and a `target-cpu=native` build note for
  bare-metal A/B benchmarking, and [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) — an
  honest triage (the kernel is <1% of an agent loop, so most low-level knobs don't
  move the needle; what matters is durability and reproducible measurement).
- **Self-analysis dogfood loop** (`.mcp.json`, `scripts/ccos_self_feed.py`,
  `docs/SELF_ANALYSIS.md`) — wires CCOS into a coding agent (Claude Code) as its
  causal memory. A project `.mcp.json` registers `ccos mcp` so the agent gets the
  memory tools natively (Mode A), and a **PostToolUse hook** is the transparent
  "hardware intercept" (Mode B): every source file the agent reads/writes becomes an
  `ingest` and every failed `cargo test/build` becomes a `page_fault`, with zero
  cognitive overhead — so `workspace.ccos` + `.oplog` accumulate a replayable record
  you then debug with `ccos postmortem`. Verified end-to-end: simulated tool events
  feed the memory and the session is walkable post-mortem.
- **MCP server** (`ccos mcp`, `mcp` module) — exposes the external-memory façade
  as [Model Context Protocol](https://modelcontextprotocol.io) tools over **stdio
  JSON-RPC 2.0**, so any MCP-compatible agent (Claude, a local agent on the Jetson)
  can use CCOS as native working memory. Dependency-free (`serde_json` only); speaks
  the standard `initialize` / `tools/list` / `tools/call` / `resources/list` /
  `resources/read` / `ping` handshake. Advertises **eight tools** (`ingest`,
  `recall`, `signal_failure`, `page_fault`, `stats`, `verify`, plus the time-travel
  pair `timeline` / `recall_what_if` — rewind to a past step and re-run a recall) and
  **two resources** (`ccos://session/context`, the self-bounding working set
  linearised for direct system-prompt injection, and `ccos://session/timeline`),
  backed by an event-sourced `AgentSession`. Optional **persistence**: `ccos mcp
  [workspace.ccos]` (or `CCOS_MCP_WORKSPACE`) reloads the checkpoint on start and
  re-checkpoints after every memory-changing call — the same snapshot format as
  `ccos memory`, so the two transports share one workspace. The **cognitive timeline
  persists too** in a `<workspace>.oplog` sidecar (the op-log plus its replay
  baseline), so `timeline` / `recall_what_if` time-travel spans the whole recorded
  history **across restarts**; a stale sidecar that no longer reproduces the snapshot
  self-heals to the snapshot (the memory is never corrupted by a stale log). The
  op-log **compacts** to stay bounded for a long-running daemon — older ops fold into
  the baseline past `CCOS_OPLOG_MAX` (default 512), keeping the last `CCOS_OPLOG_KEEP`
  (default 128) replayable; compaction is index-stable and never touches the live
  memory (only deep historical rewind is traded away). Point a client's stdio
  transport at it: `{"command":"ccos","args":["mcp","workspace.ccos"]}`. See
  [`MEMORY_INTERFACE.md`](docs/MEMORY_INTERFACE.md#serving-over-mcp-ccos-mcp).
- **Interactive post-mortem debugger** (`ccos postmortem [workspace.ccos]`,
  `postmortem` module) — a "GDB for the agent's memory": load a persisted timeline
  (`<workspace>.oplog`, even after a crashed run) or a built-in drifting session and
  walk it by hand. A REPL cursor time-travels the cognitive timeline (`timeline`,
  `goto`/`next`/`prev`, `recall`/`around`/`task` at the cursor) and two drift views
  surface how the working set moved: `diff A B` (files that entered/left) and
  `energy A B` (node-level Δscore + failure-pressure — the migration of causal heat
  through the AST as failures propagate, visible even when the file set is stable).
  `missing <node> [budget]` is an **eviction watchpoint**: it finds the first step a
  node drops out of the budgeted window, with the triggering op, the token gap, and a
  status strip (`·●●●●●○○●●`); it reports cleanly against the compaction floor when
  the eviction lies in folded history. Every command reconstructs state
  deterministically via `recall_what_if`/replay, so it is exact and side-effect free.
- **Time-travel debugging demo** (`examples/time_travel.rs`, `cargo run --example
  time_travel`) — an agent session that drifts (a tight-budget recall evicts the
  cause two hops away), then is debugged by rewinding to the exact recall and
  replaying it under a larger budget; `replay_to` reconstructs the state exactly.
- **Robust efficiency number** — `phase4_eval.py` prints a context-efficiency
  report (works in `--dry-run`, no model). Across 51 single-file fixes from
  `fd`/`bat`/`hyperfine`, CCOS assembles 700–1600 context tokens vs RAG's
  budget-filling ~6000 — a **4.1–9.1× reduction** (it self-bounds at the causal
  region; the baseline fills the budget by construction).
- **Event-sourced agent session** (`agent_session` module) — `AgentSession`
  records every cognitive op (ingest / failure / recall / page-fault) as a
  timeline; `replay_to(step)` reconstructs the exact state, and
  `recall_what_if(step, q, b)` re-runs a recall under different parameters:
  **time-travel debugging** for an agent's context, the capability a probabilistic
  retrieval stack lacks.
- **Context page fault** (`AgentSession::page_fault`) — feed `cargo test` /
  compiler output back in: parse the faulting locations (`trace`), inject failure
  pressure, recall a refreshed window — the MMU "demand paging on a fault" step,
  logged and replayable. `scripts/phase4_eval.py` now uses it as a
  **compiler-in-the-loop** retry (patch → test → page-fault → enriched context →
  retry, `--max-attempts`).
- **`ccos trace`** + **module-hierarchy linking** — parse `cargo test` / panic /
  backtrace (stdin) into the crash's source files (`trace` module); and
  `link_module_imports` now adds parent→sub-module edges so sub-modules reached
  only via a re-export aren't orphaned. (Both from the crash-trace pivot PoC, whose
  verdict was that RAG-over-the-error-message still wins.)
- **Phase-4 prototype** (`scripts/phase4_eval.py`) — the *sufficient*-condition
  harness: for a real single-file fix it builds the agent's context two ways at an
  equal token budget (CCOS causal region vs lexical-RAG top files), asks a model
  to rewrite the buggy file, applies it, and runs `cargo test`, comparing CCOS vs
  RAG resolved-rate. Validated in `--dry-run` offline; the model (Ollama) + test
  grading run on a machine with a toolchain (the Jetson). Dry-run already shows a
  caveat: CCOS's region is often *just the target file* (sparse cross-file edges),
  so it gives a thinner context than RAG at equal budget — the verdict hinges on
  whether targeted-thin beats broad-lexical for the model.
- **Thesis check in the validation harness** — measures seed↔target lexical
  similarity per scenario and reports Δ(CCOS−RAG) for far vs near seeds. On the
  available data (fd, n=8) it is *unsupportive*: CCOS does worse, not better, when
  the seed is lexically far from its targets (corr +0.45, thesis predicts −).
- **Bidirectional failure propagation** — `MemoryGraph::propagate_failure_bidirectional`
  / `ccos failure --bidirectional` spread failure pressure to *upstream causes*
  (callers/importers) as well as downstream dependencies, and `ccos analyze` now
  links cross-file imports into the snapshot it writes. Measured on the
  causal-validation harness across three mature crates (`fd`, `bat`, `hyperfine`;
  70 mined fix commits), at a sufficient budget (`K≥50`) `R_cov` reaches
  **0.85–1.0** (recovering the large majority of the files each fix touched), up
  from `0.50–0.84` downstream-only, while diluting to `0.19–0.28` at a tight
  `K=20` — an honest, systematic trade-off (see
  `scripts/causal_validation/README.md`).
- **Lexical-RAG baseline in the harness** (TF-IDF cosine, same file budget) — and
  the honest result it gives: causal selection has **no net coverage advantage**
  over lexical similarity on these real repos (CCOS/RAG ties at `K≥50`; RAG is
  clearly better at `K=20`). On real bugs a fix's files are lexically similar to
  each other, so TF-IDF finds them too; the high `R_cov` is the *necessary*
  condition, not a CCOS win. Reported, not buried. Also: crate-aware import
  resolution (multi-crate workspaces + absolute paths).
- **Cross-file import linking** — `MemoryGraph::link_module_imports()` resolves
  intra-crate imports (`use:<file>:<path>` nodes) into `file→file` dependency
  edges by mapping each file to its module path and longest-prefix-matching the
  import. The kernel previously connected causally-related files only through
  shared `dep:` hubs, so failure propagation and region recall could not reach a
  fix's cross-file cause; now they do (opt-in, idempotent; called by the external
  memory façade on ingest). On a `db→repo→api` workspace, `recall(Around api.rs)`
  returns the cause `db.rs` and excludes unrelated files, and injected failure
  attenuates along the chain (0.85 → 0.78 → 0.65) above the 0.375 noise floor.
- **Agent-loop demo** (`scripts/agent_demo.py`) — a runnable, stdlib-only demo of
  CCOS as an agent's external memory: a bug whose cause is two lexically-distant
  files away is recalled by the causal region (not by a top-k/lexical retriever).
  Runs offline; uses a local Ollama model for the fix step if `OLLAMA_ENDPOINT` is
  set.
- **External memory interface** (`external_memory` module) — a single, documented
  façade (`ExternalMemory` trait + `CcosMemory`) an agent uses to treat CCOS as
  its external working memory, unifying the kernel's separate pieces (causal
  graph, incremental parser, hash-chained logs, causal queries, region engine)
  behind a handful of verbs: `ingest_source`, `signal_failure`, `recall`
  (`WorkingSet` / `Around` region-anchored / `Task` lexical), `verify`, `stats`,
  `checkpoint` (+ inherent `open`, `impact`/`causes`, `tick`). Deterministic
  recall, tamper-evident persistence that round-trips, all result types
  `Serialize`. Also exposed as **`ccos memory`** — a stdio JSON-Lines command
  (one request per line → one JSON response) so any language can use CCOS as
  memory via a subprocess, no server required. Reference guide in
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

- **Compressor CCR reversibility under eviction.** `store` evicted the
  lowest-hash entry as soon as the store passed `ccr_capacity`, so a single
  recall window with more compressed items than the capacity could evict refs it
  had *just handed back* — breaking the "nothing is lost, call `ccos_retrieve`"
  guarantee (latent: the default capacity is 4096, larger than any real window).
  Eviction is now deferred to *after* an item/window is produced
  (`enforce_ccr_capacity`) and never drops a live ref — the cap is a floor
  against older entries, lifted when the current window exceeds it. Regression
  test: `compress_window_keeps_every_ref_retrievable_below_capacity`.
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
