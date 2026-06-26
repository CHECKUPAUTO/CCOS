//! # Agent session — an event-sourced cognitive timeline (time-travel debugging)
//!
//! This is the capability that a RAG stack (LangGraph, LlamaIndex, …) structurally
//! lacks, and the one our measurements point CCOS toward: not *better retrieval*
//! (a lexical baseline matches that), but a **deterministic, replayable, auditable**
//! record of how an agent's working memory evolved.
//!
//! An [`AgentSession`] wraps a [`CcosMemory`] and records every cognitive operation
//! (ingest, failure signal, recall) as an ordered timeline. Because every operation
//! is deterministic, the state after the first `n` operations can be reconstructed
//! exactly with [`replay_to`](AgentSession::replay_to) — you can *rewind the agent's
//! mind*. And with [`recall_what_if`](AgentSession::recall_what_if) you can replay to
//! a point and re-run a recall under **different parameters** (e.g. a larger token
//! budget) to see whether the agent would have made a better decision — the
//! "time-travel debugger" for an LLM agent's context.
//!
//! ## Example
//!
//! ```
//! use ccos::agent_session::AgentSession;
//! use ccos::external_memory::{ExternalMemory, Recall};
//!
//! let mut s = AgentSession::new();
//! s.ingest("src/db.rs", "pub fn timeout() -> i64 { 30 }\n");
//! s.ingest("src/api.rs", "use crate::db;\npub fn handle() -> i64 { db::timeout() }\n");
//! s.signal_failure("file:src/api.rs", 3).ok();
//!
//! // A tight budget recalls a thin window…
//! let tight = s.recall(Recall::around("file:src/api.rs"), 30);
//! // …rewind to that step and replay with a larger budget — the what-if.
//! let roomy = s.recall_what_if(s.len() - 1, &Recall::around("file:src/api.rs"), 4000);
//! assert!(roomy.items.len() >= tight.items.len());
//!
//! // Replay reconstructs the exact state (determinism / auditability).
//! assert_eq!(s.replay_to(s.len()).stats().nodes, s.memory().stats().nodes);
//! ```

use crate::compressor::{CausalCompressor, CcrRef};
use crate::external_memory::{
    CcosMemory, ExternalMemory, IngestReport, MemoryError, Recall, RecallWindow,
};
use crate::memory::ScoringWeights;
use crate::trace::parse_cargo_test_output;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// One recorded cognitive operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
enum Op {
    Ingest {
        uri: String,
        source: String,
    },
    Failure {
        node: String,
        depth: u32,
    },
    Recall {
        recall: Recall,
        budget: usize,
    },
    /// A compiler/test failure fed back in: pressure injected on the faulting
    /// files (the "page fault"), then a refreshed recall around them. The
    /// propagation `depth` is recorded so replay reproduces the exact pressure
    /// (old logs predate it and default to the historical depth of 2).
    PageFault {
        files: Vec<String>,
        #[serde(default = "legacy_page_fault_depth")]
        depth: u32,
    },
    /// The causal scoring weights were **retuned** from the replayable log (slice
    /// C). Logged so the change is auditable *and* reproduced on replay — the
    /// learned policy is part of the timeline, not a side channel, so
    /// `replay == live` still holds.
    Retune {
        weights: ScoringWeights,
    },
    /// An explicit **dual-evidence assertion** (Q-Pages): `evidence` supports (`supports = true`)
    /// or contradicts (`false`) `claim`, with `weight`. Logged so the agent-asserted belief surface
    /// is part of the replayable timeline — `replay == live` holds for contradictions too, not just
    /// for ingested structure. Appended last so old logs (which never contain it) stay readable.
    Assert {
        evidence: String,
        claim: String,
        supports: bool,
        weight: f64,
    },
}

/// Failure-propagation depth a `page_fault` injects. Configurable via
/// `CCOS_PAGE_FAULT_DEPTH` (default 3): deeper reaches a cause further down the
/// dependency chain from the symptom, at the cost of less focus. A field run
/// showed depth 2 left a 3-hop cause un-pressurised, hence the bump.
fn page_fault_depth() -> u32 {
    std::env::var("CCOS_PAGE_FAULT_DEPTH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3)
}

/// The depth `PageFault` ops used before it was recorded — for replaying old logs.
fn legacy_page_fault_depth() -> u32 {
    2
}

/// Whether a recall window holds the engaged node `uri` at **file granularity**:
/// a window item for the same source file — its file node *or* any of its symbol
/// nodes — counts as the relevant context having been surfaced. Used by the
/// log-tuned retrieval reward (slice C).
fn window_holds(win: &RecallWindow, uri: &str) -> bool {
    let want = engaged_path(uri);
    win.items.iter().any(|it| engaged_path(&it.uri) == want)
}

/// The source path a node uri refers to, stripped of any `kind:` prefix and
/// `:symbol` suffix — so `sym:src/x.rs:foo`, `file:src/x.rs`, and a bare
/// `src/x.rs` all reduce to `src/x.rs`.
fn engaged_path(uri: &str) -> &str {
    let rest = ["file:", "sym:", "mod:", "use:", "dep:"]
        .iter()
        .find_map(|p| uri.strip_prefix(p))
        .unwrap_or(uri);
    rest.split(':').next().unwrap_or(rest)
}

/// The on-disk timeline sidecar (`<workspace>.oplog`): the replay **baseline**
/// the session started from, the ordered op-log, and the number of older ops
/// already folded into the baseline by compaction. Persisting these lets the
/// whole cognitive timeline — not just the memory snapshot — survive a restart,
/// so [`replay_to`](AgentSession::replay_to) / time-travel work across runs.
#[derive(Serialize, Deserialize)]
struct PersistedTimeline {
    baseline: Option<String>,
    ops: Vec<Op>,
    /// Ops folded into `baseline` by compaction (absent in pre-compaction logs).
    #[serde(default)]
    folded: usize,
}

/// An event-sourced agent memory session: every operation is recorded so the
/// state is replayable and auditable. See the [module docs](crate::agent_session).
pub struct AgentSession {
    live: CcosMemory,
    ops: Vec<Op>,
    /// JSON snapshot of the memory the session's timeline replays *on top of*. A
    /// freshly [`new`](Self::new) session has none (the baseline is empty); a
    /// session [`open`](Self::open)ed from a checkpoint carries either the state
    /// it was seeded from (so [`replay_to`](Self::replay_to) layers the timeline
    /// on top of it) or `None` when the op-log reproduces the memory from empty.
    /// Compaction also folds older ops into this baseline.
    baseline: Option<String>,
    /// Number of older ops folded into `baseline` by [`compact`](Self::compact).
    /// Timeline indices stay absolute: logical length is `folded + ops.len()`, and
    /// a `replay_to(step)` for `step <= folded` collapses to the baseline (the
    /// compaction floor). Keeps the op-log bounded for a long-running daemon.
    folded: usize,
    /// Where the timeline sidecar lives (`<workspace>.oplog`); `None` for an
    /// in-memory [`new`](Self::new) session.
    oplog_path: Option<PathBuf>,
    /// The reversible context-compression pipeline (CCR store). Owns the
    /// originals cache so the host LLM can call `ccos_retrieve` across calls.
    /// See [`crate::compressor`].
    compressor: CausalCompressor,
}

impl Default for AgentSession {
    fn default() -> Self {
        Self::new()
    }
}

impl AgentSession {
    /// A fresh in-memory session with no checkpoint path (nothing is persisted).
    pub fn new() -> Self {
        AgentSession {
            live: CcosMemory::new(),
            ops: Vec::new(),
            baseline: None,
            folded: 0,
            oplog_path: None,
            compressor: CausalCompressor::new(),
        }
    }

    /// Open a **persistent** session backed by `path`: load the causal memory
    /// from the checkpoint if it exists (otherwise start empty), bind the path
    /// for [`checkpoint`](Self::checkpoint), and restore the cognitive timeline
    /// from the sidecar (`<path>.oplog`) when present. The memory snapshot is the
    /// same form `ccos memory` reads/writes, so both transports can share one
    /// `workspace.ccos`; the sidecar adds the op-log on top.
    ///
    /// With a sidecar, [`replay_to`](Self::replay_to) / time-travel span the whole
    /// recorded history across restarts. The op-log is trusted only if it
    /// reproduces the loaded memory exactly; on a mismatch (e.g. the snapshot was
    /// mutated out-of-band by `ccos memory`) the snapshot wins and the timeline
    /// restarts from it — the memory is never corrupted by a stale log.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, MemoryError> {
        // Resolve a directory workspace to its state file so the snapshot and the
        // op-log sidecar land together (see CcosMemory::open).
        let file = crate::external_memory::workspace_file(path.as_ref());
        let live = CcosMemory::open(&file)?;
        let oplog_path = oplog_sidecar(&file);

        // Restore the timeline from the sidecar if one is there and parses; a
        // missing/corrupt sidecar just means "no recorded history yet".
        let restored = std::fs::read_to_string(&oplog_path)
            .ok()
            .and_then(|s| serde_json::from_str::<PersistedTimeline>(&s).ok());

        let mut session = match restored {
            Some(t) => AgentSession {
                live,
                ops: t.ops,
                baseline: t.baseline,
                folded: t.folded,
                oplog_path: Some(oplog_path),
                compressor: CausalCompressor::new(),
            },
            None => {
                let baseline = Some(live.to_json()?);
                AgentSession {
                    live,
                    ops: Vec::new(),
                    baseline,
                    folded: 0,
                    oplog_path: Some(oplog_path),
                    compressor: CausalCompressor::new(),
                }
            }
        };

        // Consistency guard: a restored op-log must reproduce the loaded memory.
        // If not, trust the snapshot (authoritative state) and reset the timeline.
        if !session.ops.is_empty() && !session.timeline_reproduces_memory() {
            session.baseline = Some(session.live.to_json()?);
            session.ops.clear();
            session.folded = 0;
        }
        Ok(session)
    }

    /// Persist the live memory **and** the cognitive timeline. First **compacts**
    /// the op-log if it has grown past the threshold (so a long-running daemon
    /// stays bounded), then writes the memory snapshot to the bound checkpoint path
    /// (shared with `ccos memory`) and the baseline + op-log to the `<path>.oplog`
    /// sidecar. Returns [`MemoryError::NoPath`] for an in-memory `new` session (the
    /// caller can treat that as a no-op).
    pub fn checkpoint(&mut self) -> Result<(), MemoryError> {
        self.compact_if_needed();
        self.live.checkpoint()?;
        if let Some(p) = &self.oplog_path {
            let timeline = PersistedTimeline {
                baseline: self.baseline.clone(),
                ops: self.ops.clone(),
                folded: self.folded,
            };
            crate::util::write_durable(p, serde_json::to_string(&timeline)?.as_bytes())?;
        }
        Ok(())
    }

    /// Fold all but the most recent `keep` operations into the baseline snapshot,
    /// bounding the op-log. The live memory is unchanged — only the *representation*
    /// of the timeline (a newer baseline plus a shorter tail). Recent rewind depth
    /// (the last `keep` ops) is preserved; [`replay_to`](Self::replay_to) below the
    /// new floor collapses to the baseline. A no-op when `ops.len() <= keep`.
    pub fn compact(&mut self, keep: usize) {
        if self.ops.len() <= keep {
            return;
        }
        let fold = self.ops.len() - keep;
        let floor = self.replay_to(self.folded + fold);
        if let Ok(snapshot) = floor.to_json() {
            self.baseline = Some(snapshot);
            self.ops.drain(0..fold);
            self.folded += fold;
        }
    }

    /// Compact when the op-log exceeds `CCOS_OPLOG_MAX` (default 512), folding down
    /// to `CCOS_OPLOG_KEEP` (default 128) retained ops. `CCOS_OPLOG_MAX=0` disables.
    fn compact_if_needed(&mut self) {
        let env = |k: &str, d: usize| {
            std::env::var(k)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(d)
        };
        let (max, keep) = (env("CCOS_OPLOG_MAX", 512), env("CCOS_OPLOG_KEEP", 128));
        if max > 0 && self.ops.len() > max {
            self.compact(keep);
        }
    }

    /// Whether replaying the recorded op-log on top of the baseline reproduces the
    /// live memory's structure (nodes/edges/files) — the invariant `open` checks
    /// before trusting a restored timeline.
    fn timeline_reproduces_memory(&self) -> bool {
        let replayed = self.replay_to(self.len());
        let (a, b) = (replayed.stats(), self.live.stats());
        a.nodes == b.nodes && a.edges == b.edges && a.files == b.files
    }

    /// Record and apply an ingest.
    pub fn ingest(&mut self, uri: &str, source: &str) -> IngestReport {
        self.ops.push(Op::Ingest {
            uri: uri.to_string(),
            source: source.to_string(),
        });
        self.live.ingest_source(uri, source)
    }

    /// Record and apply a **support** assertion — `evidence` is evidence *for* `claim` (the
    /// affirmative surface of the claim's [Q-Page](crate::memory::MemoryGraph::qbelief)). Logged as
    /// an `Op::Assert` so it replays identically. Returns whether a new edge was added.
    pub fn assert_support(&mut self, evidence: &str, claim: &str, weight: f64) -> bool {
        self.ops.push(Op::Assert {
            evidence: evidence.to_string(),
            claim: claim.to_string(),
            supports: true,
            weight,
        });
        self.live.assert_support(evidence, claim, weight)
    }

    /// Record and apply a **contradiction** assertion — `evidence` is evidence *against* `claim`
    /// (the negative surface). The dual of [`assert_support`](Self::assert_support); logged as an
    /// `Op::Assert` so `replay == live` holds for the contested-knowledge channel too.
    pub fn assert_contradiction(&mut self, evidence: &str, claim: &str, weight: f64) -> bool {
        self.ops.push(Op::Assert {
            evidence: evidence.to_string(),
            claim: claim.to_string(),
            supports: false,
            weight,
        });
        self.live.assert_contradiction(evidence, claim, weight)
    }

    /// Bring `uri` up to date against a persisted workspace **without** logging a
    /// redundant op when it is unchanged — for read-side tools (`ccos focus`) that
    /// re-scan a tree each run. Records (and applies) an `Ingest` op only when the
    /// content actually changed; returns whether it re-ingested.
    pub fn sync(&mut self, uri: &str, source: &str) -> bool {
        if self.live.file_unchanged(uri, source) {
            return false;
        }
        self.ingest(uri, source);
        true
    }

    /// Record and apply a failure signal.
    pub fn signal_failure(&mut self, node: &str, depth: u32) -> Result<usize, MemoryError> {
        self.ops.push(Op::Failure {
            node: node.to_string(),
            depth,
        });
        self.live.signal_failure(node, depth)
    }

    /// Record and run a recall. Read-only for *selection*, with one deterministic
    /// side effect: a recall *around* a node that was demoted to the COLD tier
    /// pages it (and its cold neighbours) back first — a page fault on the read
    /// path. The op is logged and the page-in is reproduced on replay, so the
    /// timeline stays complete and `replay == live`.
    pub fn recall(&mut self, recall: Recall, budget: usize) -> RecallWindow {
        // Page fault on the read path: a recall *around* a demoted node pages it
        // (and its cold neighbours) back from the COLD tier first, so the cold
        // tier is transparent to a recalling agent.
        if let Recall::Around(uri) = &recall {
            self.live.ensure_resident(uri);
        }
        let window = self.live.recall(&recall, budget);
        self.ops.push(Op::Recall { recall, budget });
        window
    }

    /// Record and run a **compressed** recall: same selection as [`Self::recall`],
    /// then each item's content is passed through the [`CausalCompressor`] and
    /// the original is cached in the session's CCR store. The host LLM can call
    /// [`retrieve_original`](Self::retrieve_original) (exposed as the
    /// `ccos_retrieve` MCP tool) to fetch any original back — the CCOS
    /// equivalent of headroom's reversible `headroom_retrieve`. This is the
    /// real *compression* pass CCOS historically lacked; it is a pure
    /// post-selection transform, so the causal graph, the scoring, the paging
    /// and the hash-chain replay invariants are untouched.
    pub fn recall_compressed(&mut self, recall: Recall, budget: usize) -> RecallWindow {
        if let Recall::Around(uri) = &recall {
            self.live.ensure_resident(uri);
        }
        let window = self
            .live
            .recall_compressed(&recall, budget, &mut self.compressor);
        self.ops.push(Op::Recall { recall, budget });
        window
    }

    /// Retrieve an original content blob from the CCR store (backend for the
    /// `ccos_retrieve` MCP tool). `None` when the ref is unknown or has been
    /// evicted by the store's capacity cap.
    pub fn retrieve_original(&self, ccr: &CcrRef) -> Option<&str> {
        self.compressor.retrieve(ccr)
    }

    /// **Budget feedback loop** — compression-aware recall that re-spends the
    /// tokens freed by compression on more causal nodes (up to `max_rounds`
    /// passes). See [`CcosMemory::recall_compressed_with_feedback`]. This is
    /// the CCOS differentiator vs headroom: headroom compresses a fixed
    /// selection; CCOS grows the selection with the freed space, so the host
    /// gets strictly more causal signal at the same emitted-token cost.
    pub fn recall_compressed_with_feedback(
        &mut self,
        recall: Recall,
        budget: usize,
        max_rounds: usize,
    ) -> RecallWindow {
        if let Recall::Around(uri) = &recall {
            self.live.ensure_resident(uri);
        }
        let window = self.live.recall_compressed_with_feedback(
            &recall,
            budget,
            &mut self.compressor,
            max_rounds,
        );
        self.ops.push(Op::Recall { recall, budget });
        window
    }

    /// Read-only access to the compression pipeline's last-run per-item stats
    /// (algorithm, tokens before/after, ratio). Useful for a `ccos stats`-style
    /// compression dashboard.
    pub fn last_compression_stats(&self) -> &[crate::compressor::CompressionStat] {
        &self.compressor.last_stats
    }

    /// **Context page fault**: feed `cargo test` / compiler output back into the
    /// session. The faulting source locations are parsed out (a direct symptom→
    /// cause signal), failure pressure is injected on those in-memory files, and a
    /// refreshed window is recalled around the fault — the compiler-in-the-loop
    /// step. It is logged like any other op, so the whole correction loop replays.
    /// Returns the refreshed context window for the agent's next attempt.
    pub fn page_fault(&mut self, compiler_output: &str, budget: usize) -> RecallWindow {
        let files = parse_cargo_test_output(compiler_output).files();
        let depth = page_fault_depth();
        for f in &files {
            let _ = self.live.signal_failure(&format!("file:{f}"), depth);
        }
        let recall = files
            .first()
            .map(|f| Recall::around(format!("file:{f}")))
            .unwrap_or(Recall::WorkingSet);
        let window = self.live.recall(&recall, budget);
        self.ops.push(Op::PageFault { files, depth });
        window
    }

    /// Logical timeline length: ops recorded **plus** those already folded into the
    /// baseline by compaction. Stays stable across a compaction (the floor rises,
    /// the tail shrinks), so absolute `step` indices keep their meaning.
    pub fn len(&self) -> usize {
        self.folded + self.ops.len()
    }

    /// Whether nothing has been recorded yet.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The compaction floor: the logical step at/below which history has been folded
    /// into the baseline and is no longer separable (0 when nothing is compacted).
    pub fn floor(&self) -> usize {
        self.folded
    }

    /// Read-only access to the live memory.
    pub fn memory(&self) -> &CcosMemory {
        &self.live
    }

    /// Deterministically reconstruct the memory state after the first `step`
    /// (logical) operations, replaying the mutating ones on top of the session's
    /// baseline (empty for a [`new`](Self::new) session, the loaded checkpoint or a
    /// compaction floor for an [`open`](Self::open)ed one). `step` is clamped to
    /// [`len`](Self::len); a `step` at or below the compaction floor returns the
    /// baseline (older history has been folded in and is no longer separable).
    pub fn replay_to(&self, step: usize) -> CcosMemory {
        let mut m = match &self.baseline {
            Some(snapshot) => CcosMemory::from_json(snapshot).unwrap_or_default(),
            None => CcosMemory::new(),
        };
        // Map the logical step onto the retained tail (everything <= folded is the
        // baseline already).
        let tail = step.min(self.len()).saturating_sub(self.folded);
        for op in self.ops.iter().take(tail) {
            match op {
                Op::Ingest { uri, source } => {
                    m.ingest_source(uri, source);
                }
                Op::Failure { node, depth } => {
                    let _ = m.signal_failure(node, *depth);
                }
                Op::PageFault { files, depth } => {
                    for f in files {
                        let _ = m.signal_failure(&format!("file:{f}"), *depth);
                    }
                }
                Op::Recall { recall, .. } => {
                    // A recall is read-only for *selection*, but a recall *around*
                    // a demoted node pages it (and its cold neighbours) back from
                    // the COLD tier — a side effect on the resident/cold partition.
                    // Reproduce it so replay matches live (deterministic page-in).
                    if let Recall::Around(uri) = recall {
                        m.ensure_resident(uri);
                    }
                }
                Op::Retune { weights } => {
                    // Reproduce the learned-weights change so replay == live.
                    m.set_scoring_weights(*weights);
                }
                Op::Assert {
                    evidence,
                    claim,
                    supports,
                    weight,
                } => {
                    // Re-apply the dual-evidence assertion so the belief surfaces (and the
                    // QBelief derived from them) reconstruct identically — replay == live.
                    if *supports {
                        m.assert_support(evidence, claim, *weight);
                    } else {
                        m.assert_contradiction(evidence, claim, *weight);
                    }
                }
            }
        }
        m
    }

    /// **Time-travel what-if**: rewind to just before (logical) operation `step`,
    /// then run a recall with (possibly) different parameters — does the agent get
    /// a better window? `step` is clamped to the timeline length.
    pub fn recall_what_if(&self, step: usize, recall: &Recall, budget: usize) -> RecallWindow {
        self.replay_to(step).recall(recall, budget)
    }

    // ── slice C: self-improving retrieval from the replayable log ─────────────

    /// Replay the timeline under candidate scoring `weights` and score how often
    /// recall **helped**: for each recorded recall, was the node the agent engaged
    /// *next* (a failure signal or page-fault) present — at file granularity — in
    /// the window that recall would have produced under `weights`? Returns the hit
    /// rate over all such recall→engagement pairs, or `None` when the log has no
    /// judged pair (nothing to learn from yet).
    ///
    /// This is **counterfactual** evaluation on the hash-chained log: the same log
    /// yields the same number, deterministically. The reward is an honest *proxy* —
    /// it assumes the agent's next failing/faulting node is the context recall
    /// should have surfaced. Cost is one full replay per call (each recall op is
    /// re-run), so it is an offline/maintenance operation, not a hot path.
    pub fn retrieval_reward(&self, weights: &ScoringWeights) -> Option<f64> {
        let mut m = match &self.baseline {
            Some(snapshot) => CcosMemory::from_json(snapshot).unwrap_or_default(),
            None => CcosMemory::new(),
        };
        m.set_scoring_weights(*weights);
        let mut pending: Option<RecallWindow> = None;
        let (mut hits, mut total) = (0usize, 0usize);
        for op in &self.ops {
            match op {
                Op::Ingest { uri, source } => {
                    m.ingest_source(uri, source);
                }
                Op::Recall { recall, budget } => {
                    if let Recall::Around(uri) = recall {
                        m.ensure_resident(uri);
                    }
                    pending = Some(m.recall(recall, *budget));
                }
                Op::Failure { node, depth } => {
                    if let Some(win) = pending.take() {
                        total += 1;
                        if window_holds(&win, node) {
                            hits += 1;
                        }
                    }
                    let _ = m.signal_failure(node, *depth);
                }
                Op::PageFault { files, depth } => {
                    if let Some(win) = pending.take() {
                        total += 1;
                        if files.iter().any(|f| window_holds(&win, f)) {
                            hits += 1;
                        }
                    }
                    for f in files {
                        let _ = m.signal_failure(&format!("file:{f}"), *depth);
                    }
                }
                Op::Retune { .. } => {
                    // Hold `weights` fixed across the whole evaluation; a recorded
                    // retune is what we are *measuring against*, not applying here.
                }
                Op::Assert {
                    evidence,
                    claim,
                    supports,
                    weight,
                } => {
                    // Re-apply so the replayed graph matches the timeline; belief edges do not
                    // affect the recall-hit metric, but the state must stay consistent.
                    if *supports {
                        m.assert_support(evidence, claim, *weight);
                    } else {
                        m.assert_contradiction(evidence, claim, *weight);
                    }
                }
            }
        }
        (total > 0).then_some(hits as f64 / total as f64)
    }

    /// The retrieval hit rate under the **current** scoring weights — the baseline
    /// the tuner improves on. `None` when the log has no judged recall yet.
    pub fn retrieval_hit_rate(&self) -> Option<f64> {
        self.retrieval_reward(&self.live.scoring_weights())
    }

    /// Learn scoring weights that maximise the retrieval reward over the
    /// replayable log, by **deterministic coordinate ascent** from the current
    /// weights (a fixed multiplicative grid, fixed dimension order, strict
    /// improvement only — so the same log always yields the same weights). Pure: it
    /// does not mutate the session. Returns the current weights unchanged when
    /// there is nothing to learn from.
    pub fn tune_recall_weights(&self) -> ScoringWeights {
        let mut best = self.live.scoring_weights();
        let Some(mut best_reward) = self.retrieval_reward(&best) else {
            return best;
        };
        const FACTORS: [f64; 6] = [0.25, 0.5, 0.75, 1.5, 2.0, 4.0];
        // Absolute candidates for the centrality weight: it starts at 0, which a
        // multiplicative move can never escape, so it is tried as a set of fixed
        // values (relative to the base weight scale).
        const CENTRALITY: [f64; 4] = [0.05, 0.15, 0.3, 0.5];
        for _pass in 0..2 {
            for dim in 0..4 {
                for &f in &FACTORS {
                    let mut cand = best;
                    match dim {
                        0 => cand.w_base *= f,
                        1 => cand.w_failure *= f,
                        2 => cand.w_recency *= f,
                        _ => cand.w_access *= f,
                    }
                    if let Some(r) = self.retrieval_reward(&cand) {
                        if r > best_reward {
                            best = cand;
                            best_reward = r;
                        }
                    }
                }
            }
            // Structural centrality (absolute candidates, plus 0 = off).
            for &c in CENTRALITY.iter().chain(std::iter::once(&0.0)) {
                let mut cand = best;
                cand.w_centrality = c;
                if let Some(r) = self.retrieval_reward(&cand) {
                    if r > best_reward {
                        best = cand;
                        best_reward = r;
                    }
                }
            }
        }
        best
    }

    /// Learn weights from the log ([`tune_recall_weights`](Self::tune_recall_weights))
    /// and **adopt** them when they strictly improve the measured hit rate: apply
    /// them to the live memory *and* record a retune op so the change is
    /// auditable and reproduced on replay (`replay == live` holds). Returns `true`
    /// when weights were adopted. The CCOS-native loop — the replayable history is
    /// the training data.
    pub fn adopt_tuned_recall_weights(&mut self) -> bool {
        let before = self.retrieval_hit_rate();
        let tuned = self.tune_recall_weights();
        let after = self.retrieval_reward(&tuned);
        if after > before {
            self.live.set_scoring_weights(tuned);
            self.ops.push(Op::Retune { weights: tuned });
            true
        } else {
            false
        }
    }

    /// A human-readable journal of the cognitive timeline. If compaction has folded
    /// older ops into the baseline, a leading marker stands in for them (their
    /// details are no longer retained); the live tail follows at its absolute step.
    pub fn timeline(&self) -> Vec<String> {
        let mut out = Vec::new();
        if self.folded > 0 {
            out.push(format!(
                "t≤{}  «{} earlier operation(s) compacted into the baseline»",
                self.folded, self.folded
            ));
        }
        out.extend(self.ops.iter().enumerate().map(|(i, op)| {
            let t = self.folded + i + 1;
            match op {
                Op::Ingest { uri, .. } => format!("t={t}  Ingest({uri})"),
                Op::Failure { node, depth } => {
                    format!("t={t}  SignalFailure({node}, depth={depth})")
                }
                Op::Recall { recall, budget } => {
                    format!("t={t}  Recall({recall:?}, budget={budget})")
                }
                Op::PageFault { files, depth } => {
                    format!("t={t}  PageFault(files={files:?}, depth={depth})")
                }
                Op::Retune { .. } => format!("t={t}  Retune(scoring weights from log)"),
                Op::Assert {
                    evidence,
                    claim,
                    supports,
                    weight,
                } => format!(
                    "t={t}  Assert({evidence} {} {claim}, w={weight})",
                    if *supports { "supports" } else { "contradicts" }
                ),
            }
        }));
        out
    }
}

/// The timeline sidecar path for a workspace: `<path>.oplog`.
fn oplog_sidecar(path: &Path) -> PathBuf {
    let mut s = path.as_os_str().to_os_string();
    s.push(".oplog");
    PathBuf::from(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chain_session() -> AgentSession {
        let mut s = AgentSession::new();
        s.ingest("src/db.rs", "pub fn timeout() -> i64 { 30 }\n");
        s.ingest(
            "src/repo.rs",
            "use crate::db;\npub fn fetch() -> i64 { db::timeout() }\n",
        );
        s.ingest(
            "src/api.rs",
            "use crate::repo;\npub fn handle() -> i64 { repo::fetch() }\n",
        );
        s.signal_failure("file:src/api.rs", 3).unwrap();
        s
    }

    // ── slice C: self-improving retrieval from the replayable log ─────────────

    /// A session whose log has a recall that *misses* the node engaged next under
    /// the default weights, but is fixable by re-weighting: the engaged file `d`
    /// has a high access count (re-ingested), while a failing competitor `f` wins
    /// the tight default window.
    fn learnable_session() -> AgentSession {
        let mut s = AgentSession::new();
        for _ in 0..40 {
            s.ingest("src/d.rs", "pub fn delta() -> u32 { 0 }\n"); // raise d's access
        }
        s.ingest("src/f.rs", "pub fn frank() -> u32 { 1 }\n");
        s.signal_failure("file:src/f.rs", 0).unwrap(); // f is the hot default-window winner
        s.recall(Recall::working_set(), 8); // tight budget → ~1 node
        s.signal_failure("file:src/d.rs", 0).unwrap(); // the agent actually engages d
        s
    }

    #[test]
    fn retrieval_reward_scores_recall_hits_and_misses() {
        // HIT: recall around A, then fail on A → A is in its own region window.
        let mut s = AgentSession::new();
        s.ingest("src/a.rs", "pub fn alpha() -> u32 { 1 }\n");
        s.ingest("src/z.rs", "pub fn zeta() -> u32 { 2 }\n");
        s.recall(Recall::around("file:src/a.rs"), 4096);
        s.signal_failure("file:src/a.rs", 1).unwrap();
        assert_eq!(
            s.retrieval_hit_rate(),
            Some(1.0),
            "recall around A then fail on A is a hit"
        );

        // MISS: recall around A, then fail on the disconnected Z.
        let mut s = AgentSession::new();
        s.ingest("src/a.rs", "pub fn alpha() -> u32 { 1 }\n");
        s.ingest("src/z.rs", "pub fn zeta() -> u32 { 2 }\n");
        s.recall(Recall::around("file:src/a.rs"), 4096);
        s.signal_failure("file:src/z.rs", 1).unwrap();
        assert_eq!(
            s.retrieval_hit_rate(),
            Some(0.0),
            "recall around A then fail on disconnected Z is a miss"
        );

        // No judged pair yet → nothing to learn from.
        let mut s = AgentSession::new();
        s.ingest("src/a.rs", "pub fn alpha() -> u32 { 1 }\n");
        assert_eq!(s.retrieval_hit_rate(), None);
    }

    #[test]
    fn tune_recall_weights_is_deterministic() {
        let a = learnable_session().tune_recall_weights();
        let b = learnable_session().tune_recall_weights();
        assert_eq!(a.w_base, b.w_base);
        assert_eq!(a.w_failure, b.w_failure);
        assert_eq!(a.w_recency, b.w_recency);
        assert_eq!(a.w_access, b.w_access);
    }

    #[test]
    fn tuning_improves_retrieval_on_a_learnable_log() {
        let s = learnable_session();
        let before = s.retrieval_hit_rate();
        let tuned = s.tune_recall_weights();
        let after = s.retrieval_reward(&tuned);
        assert!(
            after > before,
            "log-tuned weights improve the measured hit rate: {before:?} -> {after:?}"
        );
    }

    #[test]
    fn adopting_tuned_weights_is_logged_and_replay_matches_live() {
        let mut s = learnable_session();
        let adopted = s.adopt_tuned_recall_weights();
        assert!(
            adopted,
            "the learnable log should yield an improving retune"
        );
        // The retune was recorded as an op, so replaying the whole timeline
        // reproduces the live scoring weights — replay == live still holds.
        let replayed = s.replay_to(s.len());
        let live = s.live.scoring_weights();
        let rep = replayed.scoring_weights();
        assert_eq!(live.w_base, rep.w_base);
        assert_eq!(live.w_failure, rep.w_failure);
        assert_eq!(live.w_recency, rep.w_recency);
        assert_eq!(live.w_access, rep.w_access);
    }

    #[test]
    fn replay_is_deterministic() {
        let s = chain_session();
        let replayed = s.replay_to(s.len());
        // Same structure as the live memory.
        assert_eq!(replayed.stats().nodes, s.memory().stats().nodes);
        assert_eq!(replayed.stats().edges, s.memory().stats().edges);
        // And replaying twice gives identical node counts (determinism).
        assert_eq!(replayed.stats().nodes, s.replay_to(s.len()).stats().nodes);
    }

    #[test]
    fn assertions_replay_identically_belief_and_edges() {
        // Explicit dual-evidence assertions on a claim must replay byte-for-byte: the same edges
        // AND the same *derived* QBelief — replay == live for the contested-knowledge surface, not
        // just for ingested structure.
        let mut s = AgentSession::new();
        s.ingest("src/claim.rs", "pub fn claim() {}");
        s.assert_support("ev:a", "file:src/claim.rs", 1.0);
        s.assert_support("ev:b", "file:src/claim.rs", 1.0);
        s.assert_contradiction("ev:x", "file:src/claim.rs", 1.0);

        let claim = crate::memory::NodeId("file:src/claim.rs".to_string());
        let live_q = s.memory().graph().qbelief(&claim);
        let replayed = s.replay_to(s.len());

        assert_eq!(
            replayed.graph().edges().len(),
            s.memory().graph().edges().len(),
            "replay reconstructs the same edge set"
        );
        assert_eq!(
            replayed.graph().qbelief(&claim),
            live_q,
            "derived belief reconstructs identically on replay (replay == live)"
        );
        assert_eq!(
            (live_q.support, live_q.contradiction),
            (2.0, 1.0),
            "two support + one contradiction asserted"
        );
    }

    #[test]
    fn replay_to_earlier_step_has_fewer_files() {
        let s = chain_session();
        // After 1 op only db.rs is ingested; after 3, all three files.
        assert_eq!(s.replay_to(1).stats().files, 1);
        assert!(s.replay_to(3).stats().files >= 3);
    }

    #[test]
    fn what_if_larger_budget_widens_the_window() {
        let mut s = chain_session();
        let tight = s.recall(Recall::around("file:src/api.rs"), 20);
        let step = s.len() - 1; // the recall we just made
        let roomy = s.recall_what_if(step, &Recall::around("file:src/api.rs"), 8000);
        assert!(
            roomy.tokens >= tight.tokens && roomy.items.len() >= tight.items.len(),
            "a larger budget at the same step yields a window at least as large"
        );
    }

    #[test]
    fn timeline_records_every_operation() {
        let mut s = chain_session();
        s.recall(Recall::working_set(), 1000);
        let tl = s.timeline();
        assert_eq!(tl.len(), 5); // 3 ingest + 1 failure + 1 recall
        assert!(tl[0].contains("Ingest(src/db.rs)"));
        assert!(tl[3].contains("SignalFailure"));
        assert!(tl[4].contains("Recall"));
    }

    #[test]
    fn open_persists_and_reloads_across_sessions() {
        let path = tmp("ccos-sess-reload");
        cleanup(&path);
        {
            let mut s = AgentSession::open(&path).unwrap();
            s.ingest("src/db.rs", "pub fn q() -> i64 { 1 }\n");
            s.ingest(
                "src/api.rs",
                "use crate::db;\npub fn h() -> i64 { db::q() }\n",
            );
            s.checkpoint().unwrap();
        }
        // A brand-new session reloads the causal memory straight from disk.
        let s2 = AgentSession::open(&path).unwrap();
        assert!(s2.memory().stats().files >= 2, "files survived the reload");
        assert!(
            s2.memory().verify().valid,
            "hash chain still verifies after reload"
        );
        cleanup(&path);
    }

    /// A session built entirely through `AgentSession` keeps its **whole** timeline
    /// across a restart: the op-log makes `replay_to` span the full history.
    #[test]
    fn timeline_spans_full_history_across_restart() {
        let path = tmp("ccos-sess-history");
        cleanup(&path);
        {
            let mut s = AgentSession::open(&path).unwrap();
            s.ingest("src/db.rs", "pub fn q() {}\n");
            s.ingest("src/api.rs", "use crate::db;\npub fn h() { db::q() }\n");
            s.checkpoint().unwrap();
        }
        let s2 = AgentSession::open(&path).unwrap();
        // The recorded timeline came back, not just an empty post-reload tail.
        assert_eq!(s2.len(), 2, "both ingests were restored");
        assert!(s2.timeline()[0].contains("Ingest(src/db.rs)"));
        // Full cross-restart replay: floor is empty, top reproduces the memory.
        assert_eq!(s2.replay_to(0).stats().files, 0, "replay floor is empty");
        assert_eq!(s2.replay_to(s2.len()).stats().files, 2);
        assert_eq!(
            s2.replay_to(s2.len()).stats().nodes,
            s2.memory().stats().nodes,
            "replayed history matches the loaded memory"
        );
        cleanup(&path);
    }

    /// Seeding from a `ccos memory` snapshot (a checkpoint with no op-log): the
    /// timeline starts empty and layers on the seed as its replay baseline, then
    /// becomes durable once checkpointed.
    #[test]
    fn timeline_layers_on_a_ccos_memory_seed() {
        let path = tmp("ccos-sess-seed");
        cleanup(&path);
        // Simulate `ccos memory`: write a snapshot only (no sidecar).
        {
            let mut mem = CcosMemory::open(&path).unwrap();
            mem.ingest_source("src/seed.rs", "pub fn s() {}\n");
            mem.checkpoint().unwrap();
        }
        let mut s = AgentSession::open(&path).unwrap();
        let seed_files = s.memory().stats().files;
        assert!(seed_files >= 1);
        assert!(
            s.is_empty(),
            "no recorded timeline yet (seeded from snapshot)"
        );
        s.ingest("src/new.rs", "pub fn n() {}\n");
        // Replay floor is the seed, not empty.
        assert_eq!(s.replay_to(0).stats().files, seed_files);
        assert_eq!(s.replay_to(s.len()).stats().files, seed_files + 1);
        s.checkpoint().unwrap();
        // The seeded baseline + op survive a reopen.
        let s2 = AgentSession::open(&path).unwrap();
        assert_eq!(s2.len(), 1);
        assert_eq!(s2.replay_to(0).stats().files, seed_files);
        assert_eq!(s2.replay_to(s2.len()).stats().files, seed_files + 1);
        cleanup(&path);
    }

    /// If the snapshot is mutated out-of-band (e.g. by `ccos memory`) so the op-log
    /// no longer reproduces it, the snapshot wins and the timeline resets — the
    /// memory is never corrupted by a stale log.
    #[test]
    fn stale_oplog_self_heals_to_the_snapshot() {
        let path = tmp("ccos-sess-heal");
        cleanup(&path);
        {
            let mut s = AgentSession::open(&path).unwrap();
            s.ingest("src/a.rs", "pub fn a() {}\n");
            s.checkpoint().unwrap();
        }
        // Out-of-band mutation of the snapshot only (the sidecar is left stale).
        {
            let mut mem = CcosMemory::open(&path).unwrap();
            mem.ingest_source("src/b.rs", "pub fn b() {}\n");
            mem.checkpoint().unwrap();
        }
        let s = AgentSession::open(&path).unwrap();
        assert_eq!(
            s.memory().stats().files,
            2,
            "authoritative snapshot (a + b) wins"
        );
        assert!(s.is_empty(), "stale timeline was reset");
        assert!(s.memory().verify().valid);
        cleanup(&path);
    }

    #[test]
    fn new_session_has_no_checkpoint_path() {
        let mut s = AgentSession::new();
        assert!(matches!(s.checkpoint(), Err(MemoryError::NoPath)));
    }

    /// Field regression: a launcher created `workspace.ccos` as a *directory*, so
    /// opening it used to fail with "Is a directory". A directory workspace must
    /// place its state file inside it and round-trip.
    #[test]
    fn open_accepts_a_directory_workspace() {
        let dir = tmp("ccos-ws-dir");
        let _ = std::fs::remove_file(&dir);
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        {
            let mut s = AgentSession::open(&dir).unwrap(); // a DIRECTORY, not a file
            s.ingest("src/db.rs", "pub fn q() {}\n");
            s.ingest("src/api.rs", "use crate::db;\npub fn h() { db::q() }\n");
            s.checkpoint().unwrap();
        }
        assert!(
            dir.join("workspace.ccos").is_file(),
            "state file lives inside the dir"
        );
        assert!(
            dir.join("workspace.ccos.oplog").is_file(),
            "oplog lives inside the dir"
        );
        let s2 = AgentSession::open(&dir).unwrap();
        assert!(
            s2.memory().stats().files >= 2,
            "reloads from the directory workspace"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Compaction bounds the op-log by folding older ops into the baseline, while
    /// keeping logical indices, recent rewind depth, and the live memory intact.
    #[test]
    fn compaction_bounds_the_log_and_preserves_recent_replay() {
        let mut s = AgentSession::new();
        for i in 0..10 {
            s.ingest(&format!("src/f{i}.rs"), &format!("pub fn f{i}() {{}}\n"));
        }
        let (full_files, len_before, at7) = (
            s.memory().stats().files,
            s.len(),
            s.replay_to(7).stats().files,
        );

        s.compact(4); // fold the oldest 6, keep the last 4

        assert_eq!(s.len(), len_before, "logical length is stable");
        assert_eq!(s.ops.len(), 4, "op-log bounded to the retained tail");
        assert_eq!(s.folded, 6);
        assert_eq!(
            s.memory().stats().files,
            full_files,
            "live memory untouched"
        );
        // Replays in the retained tail are exact; the full replay matches live.
        assert_eq!(s.replay_to(7).stats().files, at7);
        assert_eq!(s.replay_to(s.len()).stats().nodes, s.memory().stats().nodes);
        // A step below the floor collapses to the floor (the folded baseline).
        assert_eq!(s.replay_to(2).stats().files, s.replay_to(6).stats().files);
        assert!(
            s.timeline()[0].contains("compacted"),
            "journal notes the fold"
        );
    }

    /// A compacted timeline (folded count + tail) survives a restart.
    #[test]
    fn compaction_survives_restart() {
        let path = tmp("ccos-sess-compact");
        cleanup(&path);
        {
            let mut s = AgentSession::open(&path).unwrap();
            for i in 0..8 {
                s.ingest(&format!("src/f{i}.rs"), &format!("pub fn f{i}() {{}}\n"));
            }
            s.compact(3); // folded = 5, tail = 3
            assert_eq!(s.len(), 8);
            s.checkpoint().unwrap();
        }
        let s2 = AgentSession::open(&path).unwrap();
        assert_eq!(s2.len(), 8, "logical length survives compaction + restart");
        assert_eq!(s2.memory().stats().files, 8);
        assert!(s2.memory().verify().valid);
        assert_eq!(
            s2.replay_to(s2.len()).stats().nodes,
            s2.memory().stats().nodes,
            "restored timeline still reproduces the memory"
        );
        assert!(s2.timeline()[0].contains("compacted"));
        cleanup(&path);
    }

    /// A unique temp path for a persistence test.
    fn tmp(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("{tag}-{}.json", std::process::id()))
    }

    /// Remove a workspace and its `.oplog` sidecar.
    fn cleanup(path: &std::path::Path) {
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(super::oplog_sidecar(path));
    }

    #[test]
    fn sync_reingests_only_changed_files() {
        let mut s = AgentSession::new();
        // First sight of a file → re-ingest (records an op).
        assert!(s.sync("src/a.rs", "pub fn a() {}\n"));
        let after_first = s.len();
        // Same content again → no-op, no new op recorded.
        assert!(!s.sync("src/a.rs", "pub fn a() {}\n"));
        assert_eq!(s.len(), after_first, "unchanged file logs no op");
        // Changed content → re-ingest.
        assert!(s.sync("src/a.rs", "pub fn a() -> i32 { 1 }\n"));
        assert_eq!(s.len(), after_first + 1);
    }

    #[test]
    fn page_fault_pulls_the_faulting_file_in_and_replays() {
        let mut s = chain_session();
        // A test failure pointing at the deep cause db.rs.
        let err = "thread 'main' panicked at src/db.rs:1:14:\nattempt to add with overflow\n";
        let win = s.page_fault(err, 8000);
        assert!(
            win.items.iter().any(|i| i.uri == "file:src/db.rs"),
            "page fault recalls the faulting file"
        );
        // The whole correction loop replays: reconstructed state matches live.
        let replayed = s.replay_to(s.len());
        assert_eq!(replayed.stats().nodes, s.memory().stats().nodes);
        assert!(s.timeline().last().unwrap().contains("PageFault"));
    }

    #[test]
    fn recall_around_pages_a_demoted_anchor_back_from_cold() {
        use crate::memory::NodeId;
        let mut s = AgentSession::new();
        s.ingest("src/db.rs", "pub fn query() -> i64 { 0 }\n");
        s.ingest(
            "src/api.rs",
            "use crate::db;\npub fn h() -> i64 { db::query() }\n",
        );
        // Shrink the frugal resident window so some nodes demote to COLD.
        s.live.set_max_resident(1);
        assert!(
            s.live.graph().cold_count() > 0,
            "some nodes demoted to COLD"
        );

        let cold_id = s.live.graph().cold_ids().next().unwrap().0.clone();
        assert!(s.live.graph().is_cold(&NodeId(cold_id.clone())));

        // A recall *around* the demoted node pages it back from COLD first
        // (the page fault on the read path) — transparent to the agent.
        let _ = s.recall(Recall::around(&cold_id), 4096);
        assert!(
            s.live.graph().contains_node(&NodeId(cold_id.clone())),
            "the cold anchor is paged back in by recall"
        );
        assert!(!s.live.graph().is_cold(&NodeId(cold_id)), "no longer cold");
    }

    #[test]
    fn page_fault_resurrects_a_demoted_faulting_file() {
        use crate::memory::NodeId;
        let mut s = AgentSession::new();
        s.ingest("src/db.rs", "pub fn query() -> i64 { 0 }\n");
        s.ingest(
            "src/api.rs",
            "use crate::db;\npub fn h() -> i64 { db::query() }\n",
        );
        s.live.set_max_resident(1);
        assert!(s.live.graph().cold_count() > 0);

        // A crash trace blaming db.rs pages the (demoted) file back in before recall.
        let win = s.page_fault("thread 'main' panicked at src/db.rs:1:1:\nboom\n", 4096);
        assert!(
            s.live
                .graph()
                .contains_node(&NodeId("file:src/db.rs".into())),
            "the faulting file is resident after the page fault"
        );
        assert!(win.items.iter().any(|i| i.uri == "file:src/db.rs"));
    }

    /// A field finding: depth-2 propagation left a 3-hop cause un-pressurised. With
    /// the default depth (3) a page-fault on the symptom reaches a 3-hop-deep cause,
    /// and the recorded depth replays the exact same pressure (determinism).
    #[test]
    fn page_fault_reaches_a_three_hop_cause_and_replays_the_depth() {
        let mut s = AgentSession::new();
        s.ingest("src/a.rs", "use crate::b;\npub fn f() -> i64 { b::g() }\n");
        s.ingest("src/b.rs", "use crate::c;\npub fn g() -> i64 { c::h() }\n");
        s.ingest("src/c.rs", "use crate::d;\npub fn h() -> i64 { d::i() }\n");
        s.ingest("src/d.rs", "pub fn i() -> i64 { 0 }\n");
        // Panic at the entry a.rs (the symptom); the cause d.rs is 3 hops down.
        s.page_fault("thread 'main' panicked at src/a.rs:1:1:\nboom\n", 8000);

        let fr = |m: &CcosMemory| {
            m.graph()
                .nodes
                .iter()
                .find(|(id, _)| id.0 == "file:src/d.rs")
                .map(|(_, n)| n.failure_relevance)
                .unwrap_or(0.0)
        };
        assert!(
            fr(s.memory()) > 0.0,
            "depth-3 page-fault pressurises the 3-hop cause d.rs"
        );
        // Replay uses the recorded depth → reproduces the same pressure on d.rs.
        assert!((fr(&s.replay_to(s.len())) - fr(s.memory())).abs() < 1e-9);
    }
}
