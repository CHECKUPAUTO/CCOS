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
use std::path::Path;

/// One recorded cognitive operation.
#[derive(Debug, Clone)]
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

/// An event-sourced agent memory session: every operation is recorded so the
/// state is replayable and auditable. See the [module docs](crate::agent_session).
pub struct AgentSession {
    live: CcosMemory,
    ops: Vec<Op>,
    /// JSON snapshot of the memory the session was *opened* from, if any. A
    /// freshly [`new`](Self::new) session has none (the baseline is empty); a
    /// session [`open`](Self::open)ed from a checkpoint carries the loaded state
    /// here so [`replay_to`](Self::replay_to) layers this session's timeline on
    /// top of it instead of on an empty graph.
    baseline: Option<String>,
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
        }
    }

    /// Open a **persistent** session backed by `path`: load the causal memory
    /// from the checkpoint if it exists (otherwise start empty), bind the path
    /// for [`checkpoint`](Self::checkpoint), and keep the loaded state as the
    /// replay baseline. The on-disk form is the same snapshot `ccos memory`
    /// reads/writes, so both transports can share one `workspace.ccos`.
    ///
    /// The cognitive timeline ([`replay_to`](Self::replay_to) / time-travel)
    /// starts fresh from the reload point, layered on top of the loaded baseline:
    /// `replay_to(0)` reconstructs the loaded state, not an empty graph.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, MemoryError> {
        let live = CcosMemory::open(path)?;
        let baseline = Some(live.to_json()?);
        Ok(AgentSession {
            live,
            ops: Vec::new(),
            baseline,
        })
    }

    /// Persist the live memory to the bound checkpoint path. Returns
    /// [`MemoryError::NoPath`] for an in-memory [`new`](Self::new) session (the
    /// caller can treat that as a no-op).
    pub fn checkpoint(&self) -> Result<(), MemoryError> {
        self.live.checkpoint()
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

    /// Number of recorded operations (timeline length).
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Whether nothing has been recorded yet.
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Read-only access to the live memory.
    pub fn memory(&self) -> &CcosMemory {
        &self.live
    }

    /// Deterministically reconstruct the memory state after the first `step`
    /// operations (replaying only the mutating ones) on top of the session's
    /// baseline (empty for a [`new`](Self::new) session, the loaded checkpoint
    /// for an [`open`](Self::open)ed one). `step >= len()` replays all.
    pub fn replay_to(&self, step: usize) -> CcosMemory {
        let mut m = match &self.baseline {
            Some(snapshot) => CcosMemory::from_json(snapshot).unwrap_or_default(),
            None => CcosMemory::new(),
        };
        for op in self.ops.iter().take(step) {
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

    /// **Time-travel what-if**: rewind to just before operation `step`, then run a
    /// recall with (possibly) different parameters — does the agent get a better
    /// window? `step` is clamped to the timeline length.
    pub fn recall_what_if(&self, step: usize, recall: &Recall, budget: usize) -> RecallWindow {
        self.replay_to(step.min(self.ops.len()))
            .recall(recall, budget)
    }

    /// A human-readable journal of the cognitive timeline.
    pub fn timeline(&self) -> Vec<String> {
        self.ops
            .iter()
            .enumerate()
            .map(|(i, op)| {
                let t = i + 1;
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
            })
            .collect()
    }
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
        let path =
            std::env::temp_dir().join(format!("ccos-sess-reload-{}.json", std::process::id()));
        let _ = std::fs::remove_file(&path);
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
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn replay_after_open_layers_on_the_loaded_baseline() {
        let path = std::env::temp_dir().join(format!("ccos-sess-base-{}.json", std::process::id()));
        let _ = std::fs::remove_file(&path);
        {
            let mut s = AgentSession::open(&path).unwrap();
            s.ingest("src/db.rs", "pub fn q() {}\n");
            s.checkpoint().unwrap();
        }
        let mut s2 = AgentSession::open(&path).unwrap();
        let baseline_files = s2.memory().stats().files; // db.rs, loaded from disk
        assert!(baseline_files >= 1);
        s2.ingest("src/api.rs", "use crate::db;\npub fn h() { db::q() }\n");
        // replay_to(0) reconstructs the loaded baseline, not an empty graph.
        assert_eq!(
            s2.replay_to(0).stats().files,
            baseline_files,
            "replay floor is the loaded checkpoint"
        );
        // replay_to(len) = baseline + this session's ingest.
        assert_eq!(s2.replay_to(s2.len()).stats().files, baseline_files + 1);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn new_session_has_no_checkpoint_path() {
        let s = AgentSession::new();
        assert!(matches!(s.checkpoint(), Err(MemoryError::NoPath)));
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
