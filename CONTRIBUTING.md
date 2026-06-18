# Contributing to CCOS

Thanks for your interest! CCOS is a research prototype, but it holds itself to a
clean bar: deterministic behaviour, enforced invariants, and a green CI. This
guide gets you productive quickly.

- [Local setup](#local-setup)
- [The development loop](#the-development-loop)
- [What CI checks](#what-ci-checks)
- [Coding conventions](#coding-conventions)
- [Non-negotiable invariants](#non-negotiable-invariants)
- [Extension recipes](#extension-recipes)
- [Branches & commits](#branches--commits)
- [Pull-request checklist](#pull-request-checklist)

## Local setup

You only need a recent **stable** Rust toolchain (CI tracks current stable):

```bash
rustup update stable
rustup component add rustfmt clippy
git clone <repo> && cd CCOS
cargo build --all-targets
cargo test
```

No system dependencies, no network, no GPU. Tests run fully offline (the LLM
client falls back deterministically when no Ollama endpoint is reachable).

## The development loop

Run these four before every push — they mirror CI exactly, so if they pass
locally CI will too:

```bash
cargo fmt --all                          # format (CI runs --check)
cargo clippy --all-targets -- -D warnings  # lint; warnings are errors
cargo test                               # 202 unit + integration tests
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps   # docs must build clean
```

Useful extras:

```bash
cargo test --lib query::                 # one module's unit tests
cargo test -- --ignored                  # opt-in 1,000,000-cycle stability run
cargo bench                              # criterion delta benchmark
cargo run -- analyze src --cycles        # exercise the CLI on our own source
```

## What CI checks

`.github/workflows/ci.yml` runs a **single consolidated job** (one checkout, one
toolchain, one cached `target/` so dependencies compile once) to keep Actions
minute usage low on this private repo. It uses **only** GitHub-authored actions
(`actions/checkout`, `actions/cache`) plus the runner's `rustup`, so no
third-party action can be blocked by org policy. Steps run cheapest-first
(fail-fast):

| Step | Command | Blocking |
| --- | ------- | -------- |
| **Format** | `cargo fmt --all --check` | yes |
| **Clippy** | `cargo clippy --all-targets --all-features --locked -- -D warnings` | yes |
| **Test** | `cargo test` **and** `cargo test --features syn-parser` | yes |
| **Docs** | `cargo doc --no-deps` with `RUSTDOCFLAGS=-D warnings` | yes |
| **CLI smoke** | `analyze → verify → replay → top → blame → export → chaos` | yes |

A broken command fails the smoke step even without a dedicated test. The slower
`cargo audit` lives in `.github/workflows/audit.yml`, which runs **weekly** (and
on demand via *Run workflow*) rather than on every push.

## Coding conventions

- **Format with `rustfmt`** (default settings). No manual alignment that the
  formatter would undo.
- **Zero clippy warnings.** Prefer iterators, avoid needless clones/allocations,
  and don't `#[allow(...)]` without a one-line reason.
- **Document public items.** Every public module/type/fn needs a rustdoc line;
  `cargo doc -D warnings` is enforced, so intra-doc links must resolve and
  must not be redundant (`[`Type`]`, not `[`Type`](crate::m::Type)` when the
  short form resolves).
- **Avoid panics on the library path.** Return `Result`/`Option`; reserve
  `unwrap`/`expect` for tests or genuinely-impossible cases (with a comment).
- **Keep determinism.** Any time you iterate a `HashMap`/`HashSet` to produce
  output, impose a total order before emitting (sort by a stable key, ties
  broken by `NodeId`). See `MemoryGraph::get_node_scores` /
  `enforce_paging` for the pattern.

## Non-negotiable invariants

These are enforced in code and covered by tests; don't regress them.

| Invariant | Lives in |
| --------- | -------- |
| `edges ⊆ nodes × nodes` (no dangling edges) | `MemoryGraph::add_edge`, `prune_dangling_edges`, `enforce_paging` |
| Node count ≤ `max_in_memory_nodes` | `enforce_paging` |
| Deterministic eviction / ordering | total order *(score, NodeId)* across `memory` |
| Guard output is always valid JSON | `GuardLayer::validate_and_sanitize` → `fallback_response` |
| Hash chain is tamper-evident | `EventLog::verify_integrity` (primary log) + `DistributedEventLog::verify_integrity` |

Regression coverage lives in `tests/graph_invariants.rs`,
`tests/snapshot_differential.rs`, `tests/long_term_stability.rs`,
`tests/llm_adversarial_test.rs`, and `tests/property_invariants.rs` (proptest).

## Extension recipes

- **New node/edge type** — add a variant to `NodeType`/`EdgeType` (serde-tagged
  enums) and update the color/label maps in `MemoryGraph::to_dot` and
  `query::to_graphml`.
- **New event** — add a variant to `event_log::EventPayload`, handle it in
  `EventReplayer::handle_event`, and (if it should be hashed) in
  `tests/snapshot_differential.rs::event_log_hash`.
- **New CLI command** — add a match arm in `main()`, an `OptsX::parse(&[String])`,
  and a `run_x(...) -> i32`; document it in `print_help`, `README.md`,
  `docs/USAGE.md`, and the smoke run in `ci.yml`.
- **New graph query** — add a pure function to `query` with unit tests; reuse it
  from a command rather than reimplementing traversal.
- **New invariant** — add a `MemoryGraph` method that checks/repairs it plus a
  test in `tests/graph_invariants.rs`.

## Branches & commits

- Work on a feature branch; keep `main` green.
- Write imperative, scoped commit subjects (`parser: strip inline block
  comments`), with a body explaining *why* when it isn't obvious.
- Keep formatting-only churn out of logic commits.

## Pull-request checklist

- [ ] `cargo fmt --all --check` clean
- [ ] `cargo clippy --all-targets -- -D warnings` clean
- [ ] `cargo test` green (add/extend tests for new behaviour)
- [ ] `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps` clean
- [ ] Docs updated (`README.md` / `docs/USAGE.md` / `docs/ARCHITECTURE.md`) for
      any user-facing or structural change
- [ ] `CHANGELOG.md` updated under *Unreleased*
- [ ] No new invariant regressions; determinism preserved
