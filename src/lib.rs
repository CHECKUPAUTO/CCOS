//! # CCOS — Causal Context Operating System
//!
//! CCOS is an experimental "kernel" that treats an LLM's working context like a
//! virtual-memory system: source code is parsed into a **causal memory graph**,
//! nodes are scored by importance / failure-relevance / recency, and a bounded
//! "context window" is paged in and out much like RAM ↔ VRAM. Every state
//! transition is recorded in an append-only **event log** so a session can be
//! replayed deterministically.
//!
//! ## Modules
//!
//! - [`parser`] — dependency-light line-based AST extraction (modules, `use`
//!   statements, symbols) from Rust source.
//! - [`memory`] — the causal [`memory::MemoryGraph`]: scoring, failure
//!   propagation, deterministic eviction/paging and context-window selection.
//! - [`incremental`] — `O(Δ)` graph updates: only the changed file's subgraph is
//!   re-evaluated on each edit ([`incremental::IncrementalGraphEngine`]).
//! - [`event_log`] — append-only [`event_log::EventLog`] with deterministic
//!   replay over typed [`event_log::EventPayload`] records.
//! - [`distributed_event_log`] — hash-chained, tamper-evident event log with an
//!   integrity verifier.
//! - [`llm`] — async client for an Ollama-style `/api/generate` endpoint with
//!   retries and a deterministic offline fallback.
//! - [`guard`] — validation/sanitization layer that rejects malformed model
//!   output and substitutes a safe, valid-JSON fallback.
//! - [`consensus`] — majority and confidence-weighted multi-model voting.
//! - [`adversarial`] — fault injector (JSON corruption, hallucination, prompt
//!   injection, timeouts) used to harden the guard and the graph.
//! - [`persist`] — save/load a full [`persist::KernelSnapshot`] (graph + both
//!   logs) to JSON for cross-session replay and verification.
//! - [`query`] — read-only causal queries (impact/cause walks, hot set, GraphML
//!   export) behind the `top`, `blame` and `export` subcommands.
//! - [`region_engine`] — the **Context Region Engine** (v0.3): clusters the
//!   graph into spatial [`region_engine::ContextRegionEngine`] regions that are
//!   hydrated as context windows, with a dynamic [`context_policy`] admission
//!   policy and deterministic replay. See [`context_region`], [`region_metrics`].
//!
//! ## Invariants
//!
//! The memory graph maintains `edges ⊆ nodes × nodes` at all times (see
//! [`memory::MemoryGraph::add_edge`] and
//! [`memory::MemoryGraph::prune_dangling_edges`]), and eviction order is
//! deterministic so replays and snapshot hashes are reproducible regardless of
//! `HashMap` iteration order.

pub mod adversarial;
pub mod consensus;
pub mod distributed_event_log;
pub mod event_log;
pub mod guard;
pub mod incremental;
pub mod llm;
pub mod memory;
pub mod parser;
pub mod persist;
pub mod query;
pub mod util;

// ── CCOS v0.3 — Autonomous Context Runtime ──────────────────────────
pub mod agents;
pub mod benchmark;
pub mod persistence;
pub mod scheduler;
pub mod workspace;

// ── CCOS v0.3 — Context Region Engine (spatial memory) ──────────────
pub mod context_policy;
pub mod context_region;
pub mod experiment;
pub mod region_engine;
pub mod region_metrics;
