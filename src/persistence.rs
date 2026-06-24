//! # Persistent State Engine (CCOS v0.3)
//!
//! Durably stores the full runtime so CCOS can resume *exactly* where it left
//! off after a shutdown or reboot. State is written to a directory as three
//! files:
//!
//! - `graph.snapshot`  — the causal [`MemoryGraph`]
//! - `events.log`      — the append-only [`EventLog`]
//! - `memory.snapshot` — the hash-chained [`DistributedEventLog`]
//!
//! [`PersistentRuntime::restore_runtime`] reloads and *verifies* the state
//! (hash chain valid, no dangling edges) before handing it back.

use crate::distributed_event_log::DistributedEventLog;
use crate::event_log::EventLog;
use crate::memory::MemoryGraph;
use crate::persist::KernelSnapshot;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Typed error for persistence operations.
#[derive(Debug)]
pub enum PersistenceError {
    Io(String),
    Serde(String),
    Integrity(String),
}

impl std::fmt::Display for PersistenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PersistenceError::Io(e) => write!(f, "persistence I/O error: {e}"),
            PersistenceError::Serde(e) => write!(f, "persistence (de)serialization error: {e}"),
            PersistenceError::Integrity(e) => write!(f, "persistence integrity error: {e}"),
        }
    }
}

impl std::error::Error for PersistenceError {}

impl From<std::io::Error> for PersistenceError {
    fn from(e: std::io::Error) -> Self {
        PersistenceError::Io(e.to_string())
    }
}
impl From<serde_json::Error> for PersistenceError {
    fn from(e: serde_json::Error) -> Self {
        PersistenceError::Serde(e.to_string())
    }
}

/// The complete runtime state that survives a restart.
///
/// This is the **same payload**, field-for-field, as
/// [`crate::persist::KernelSnapshot`] (they were duplicated), so `RuntimeState`
/// is now a type alias for it: one state type, two on-disk layouts —
/// [`PersistentRuntime`] stores it as a three-file directory, while
/// [`KernelSnapshot::save`]/[`load`](KernelSnapshot::load) store it as a single
/// file. The integrity check is shared via [`KernelSnapshot::verify_integrity`].
pub type RuntimeState = KernelSnapshot;

/// Reads/writes [`RuntimeState`] under a directory.
#[derive(Debug, Clone)]
pub struct PersistentRuntime {
    pub dir: PathBuf,
}

impl PersistentRuntime {
    /// Use `dir` as the state directory (e.g. `"data"`).
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    pub fn graph_path(&self) -> PathBuf {
        self.dir.join("graph.snapshot")
    }
    pub fn events_path(&self) -> PathBuf {
        self.dir.join("events.log")
    }
    pub fn memory_path(&self) -> PathBuf {
        self.dir.join("memory.snapshot")
    }

    /// True if a previously-saved state exists in the directory.
    pub fn exists(&self) -> bool {
        self.graph_path().exists() && self.events_path().exists() && self.memory_path().exists()
    }

    /// Persist `state` to the directory (creating it if needed).
    pub fn save_state(&self, state: &RuntimeState) -> Result<(), PersistenceError> {
        std::fs::create_dir_all(&self.dir)?;
        write_json(&self.graph_path(), &state.graph)?;
        write_json(&self.events_path(), &state.event_log)?;
        write_json(&self.memory_path(), &state.dist_log)?;
        Ok(())
    }

    /// Load state from the directory (no validation).
    pub fn load_state(&self) -> Result<RuntimeState, PersistenceError> {
        let graph: MemoryGraph = read_json(&self.graph_path())?;
        let event_log: EventLog = read_json(&self.events_path())?;
        let dist_log: DistributedEventLog = read_json(&self.memory_path())?;
        Ok(KernelSnapshot::new(graph, event_log, dist_log))
    }

    /// Load **and verify** the runtime: the hash chain must validate and the
    /// graph must hold no dangling edges. Returns the ready-to-run state.
    pub fn restore_runtime(&self) -> Result<RuntimeState, PersistenceError> {
        let state = self.load_state()?;
        state
            .verify_integrity()
            .map_err(PersistenceError::Integrity)?;
        Ok(state)
    }
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<(), PersistenceError> {
    let json = serde_json::to_string(value)?;
    std::fs::write(path, json)?;
    Ok(())
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, PersistenceError> {
    let data = std::fs::read_to_string(path)?;
    Ok(serde_json::from_str(&data)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event_log::{EventPayload, EventType};
    use crate::memory::{EdgeType, NodeType};

    fn temp_dir(tag: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "ccos_persist_{}_{}_{}",
            tag,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        p
    }

    fn sample_state() -> RuntimeState {
        let mut graph = MemoryGraph::default();
        for i in 0..8 {
            graph.upsert_node(
                format!("n{i}").into(),
                format!("L{i}"),
                "content".into(),
                NodeType::Module,
            );
        }
        graph.add_edge("n0".into(), "n1".into(), 0.9, EdgeType::DependsOn);
        graph.add_edge("n1".into(), "n2".into(), 0.8, EdgeType::Contains);

        let mut event_log = EventLog::new("persist-runtime".into());
        for c in 0..5 {
            event_log.append(
                EventType::CycleEnd,
                EventPayload::CycleEvent {
                    cycle_number: c,
                    action: format!("cycle_{c}"),
                },
            );
        }

        let mut dist_log = DistributedEventLog::new();
        for c in 0..5 {
            dist_log.append(format!("event_{c}"), "kernel".into());
        }

        RuntimeState::new(graph, event_log, dist_log)
    }

    #[test]
    fn save_then_restore_reproduces_state() {
        let dir = temp_dir("roundtrip");
        let runtime = PersistentRuntime::new(&dir);

        // 1. Create state.
        let before = sample_state();
        let (n0, e0, ev0, chain0) = (
            before.graph.node_count(),
            before.graph.edge_count(),
            before.event_log.event_count(),
            before.dist_log.event_count(),
        );

        // 2. Persist and simulate shutdown (drop the in-memory state).
        runtime.save_state(&before).unwrap();
        drop(before);
        assert!(runtime.exists());

        // 3. Reload on a fresh runtime handle.
        let after = PersistentRuntime::new(&dir).restore_runtime().unwrap();

        // 4. Compare before/after.
        assert_eq!(after.graph.node_count(), n0);
        assert_eq!(after.graph.edge_count(), e0);
        assert_eq!(after.event_log.event_count(), ev0);
        assert_eq!(after.dist_log.event_count(), chain0);
        assert!(after.dist_log.verify_integrity().valid);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn restore_detects_tampered_chain() {
        let dir = temp_dir("tamper");
        let runtime = PersistentRuntime::new(&dir);
        let mut state = sample_state();
        // Corrupt a hash-chain link before saving.
        state.dist_log.hash_chain[2].hash = "deadbeef".repeat(8);
        runtime.save_state(&state).unwrap();

        let restored = runtime.restore_runtime();
        assert!(
            matches!(restored, Err(PersistenceError::Integrity(_))),
            "tampered chain must be rejected on restore"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_missing_state_errors_cleanly() {
        let dir = temp_dir("missing");
        let runtime = PersistentRuntime::new(&dir);
        assert!(!runtime.exists());
        assert!(matches!(runtime.load_state(), Err(PersistenceError::Io(_))));
    }
}
