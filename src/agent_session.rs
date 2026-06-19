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

/// One recorded cognitive operation.
#[derive(Debug, Clone)]
enum Op {
    Ingest { uri: String, source: String },
    Failure { node: String, depth: u32 },
    Recall { recall: Recall, budget: usize },
}

/// An event-sourced agent memory session: every operation is recorded so the
/// state is replayable and auditable. See the [module docs](crate::agent_session).
pub struct AgentSession {
    live: CcosMemory,
    ops: Vec<Op>,
}

impl Default for AgentSession {
    fn default() -> Self {
        Self::new()
    }
}

impl AgentSession {
    /// A fresh in-memory session.
    pub fn new() -> Self {
        AgentSession {
            live: CcosMemory::new(),
            ops: Vec::new(),
        }
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
    /// operations (replaying only the mutating ones). `step >= len()` replays all.
    pub fn replay_to(&self, step: usize) -> CcosMemory {
        let mut m = CcosMemory::new();
        for op in self.ops.iter().take(step) {
            match op {
                Op::Ingest { uri, source } => {
                    m.ingest_source(uri, source);
                }
                Op::Failure { node, depth } => {
                    let _ = m.signal_failure(node, *depth);
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
}
