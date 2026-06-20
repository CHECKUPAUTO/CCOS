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

use crate::external_memory::{
    CcosMemory, ExternalMemory, IngestReport, MemoryError, Recall, RecallWindow,
};
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
    /// files (the "page fault"), then a refreshed recall around them.
    PageFault {
        files: Vec<String>,
    },
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
        let path = path.as_ref();
        let live = CcosMemory::open(path)?;
        let oplog_path = oplog_sidecar(path);

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
            },
            None => {
                let baseline = Some(live.to_json()?);
                AgentSession {
                    live,
                    ops: Vec::new(),
                    baseline,
                    folded: 0,
                    oplog_path: Some(oplog_path),
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

    /// Record and apply a failure signal.
    pub fn signal_failure(&mut self, node: &str, depth: u32) -> Result<usize, MemoryError> {
        self.ops.push(Op::Failure {
            node: node.to_string(),
            depth,
        });
        self.live.signal_failure(node, depth)
    }

    /// Record and run a recall (read-only, but logged so the timeline is complete).
    pub fn recall(&mut self, recall: Recall, budget: usize) -> RecallWindow {
        let window = self.live.recall(&recall, budget);
        self.ops.push(Op::Recall { recall, budget });
        window
    }

    /// **Context page fault**: feed `cargo test` / compiler output back into the
    /// session. The faulting source locations are parsed out (a direct symptom→
    /// cause signal), failure pressure is injected on those in-memory files, and a
    /// refreshed window is recalled around the fault — the compiler-in-the-loop
    /// step. It is logged like any other op, so the whole correction loop replays.
    /// Returns the refreshed context window for the agent's next attempt.
    pub fn page_fault(&mut self, compiler_output: &str, budget: usize) -> RecallWindow {
        let files = parse_cargo_test_output(compiler_output).files();
        for f in &files {
            let _ = self.live.signal_failure(&format!("file:{f}"), 2);
        }
        let recall = files
            .first()
            .map(|f| Recall::around(format!("file:{f}")))
            .unwrap_or(Recall::WorkingSet);
        let window = self.live.recall(&recall, budget);
        self.ops.push(Op::PageFault { files });
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
                Op::PageFault { files } => {
                    for f in files {
                        let _ = m.signal_failure(&format!("file:{f}"), 2);
                    }
                }
                Op::Recall { .. } => {} // read-only: no state change
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
                Op::PageFault { files } => format!("t={t}  PageFault(files={files:?})"),
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
}
