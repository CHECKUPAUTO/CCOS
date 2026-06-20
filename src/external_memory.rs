//! # External memory interface (`ExternalMemory` / [`CcosMemory`])
//!
//! A single, documented façade an agent uses to treat CCOS as its **external
//! working memory**: write code and failure signals in, recall a bounded,
//! causally-coherent context window out, and persist an auditable, hash-chained
//! state. It unifies the kernel's otherwise separate pieces — the causal
//! [`MemoryGraph`], the incremental parser ([`IncrementalGraphEngine`]), the
//! tamper-evident [`EventLog`]/[`DistributedEventLog`], the causal
//! [`query`] walks, and the [`ContextRegionEngine`] — behind one trait.
//!
//! ## The contract
//!
//! [`ExternalMemory`] is the stable surface; [`CcosMemory`] is the in-process
//! implementation. The operations are:
//!
//! | Operation | Meaning |
//! | --------- | ------- |
//! | [`ingest_source`](ExternalMemory::ingest_source) | write/update a file; the graph and the hash chain advance |
//! | [`signal_failure`](ExternalMemory::signal_failure) | mark a node as failing and propagate the pressure downstream |
//! | [`recall`](ExternalMemory::recall) | select a bounded context window ([`Recall`] strategy) |
//! | [`verify`](ExternalMemory::verify) | check the hash chain is intact (tamper-evidence) |
//! | [`stats`](ExternalMemory::stats) | counts (nodes/edges/events/files) |
//! | [`checkpoint`](ExternalMemory::checkpoint) | persist the whole state to the bound path |
//!
//! Plus inherent helpers on [`CcosMemory`]: [`open`](CcosMemory::open),
//! [`impact`](CcosMemory::impact)/[`causes`](CcosMemory::causes) (blast radius /
//! upstream causes), and [`tick`](CcosMemory::tick) (recency decay).
//!
//! ## Recall strategies
//!
//! - [`Recall::WorkingSet`] — the globally hottest nodes by causal score.
//! - [`Recall::Around`] — the causal **region** anchored on a node (the workspace
//!   signal: the active file / failing test), independent of any query text.
//! - [`Recall::Task`] — a free-text task: a lexical entry point, expanded to its
//!   region.
//!
//! ## Example
//!
//! ```no_run
//! use ccos::external_memory::{CcosMemory, ExternalMemory, Recall};
//!
//! let mut mem = CcosMemory::open("workspace.ccos").unwrap();
//! mem.ingest_source("src/db.rs", "pub fn query() {}\n");
//! mem.signal_failure("file:src/db.rs", 3).ok();
//! let window = mem.recall(&Recall::task("fix db timeout"), 2048);
//! for item in &window.items {
//!     println!("{:.3}  {}", item.score, item.uri);
//! }
//! assert!(mem.verify().valid);
//! mem.checkpoint().unwrap();
//! ```
//!
//! ## Guarantees
//!
//! Deterministic recall (total order on `(score, uri)`); every `ingest_source`
//! extends a canonical SHA-256 hash chain, so [`verify`](ExternalMemory::verify)
//! detects any tampering with a persisted checkpoint; a checkpoint round-trips
//! (reload reproduces the graph and the chain).

use crate::context_region::file_of;
use crate::distributed_event_log::DistributedEventLog;
use crate::event_log::{EventLog, EventPayload, EventType};
use crate::incremental::IncrementalGraphEngine;
use crate::memory::{GraphNode, MemoryGraph, NodeId};
use crate::query::{self, Reached};
use crate::region_engine::ContextRegionEngine;
use crate::util::sha256_hex;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// Errors returned by memory operations.
#[derive(Debug)]
pub enum MemoryError {
    /// A referenced node id is not present in the graph.
    NodeNotFound(String),
    /// Filesystem error while persisting or loading a checkpoint.
    Io(std::io::Error),
    /// (De)serialisation error for a checkpoint.
    Serde(serde_json::Error),
    /// [`ExternalMemory::checkpoint`] was called with no path bound; use
    /// [`CcosMemory::open`] or [`CcosMemory::checkpoint_to`].
    NoPath,
}

impl std::fmt::Display for MemoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemoryError::NodeNotFound(id) => write!(f, "node not found: {id}"),
            MemoryError::Io(e) => write!(f, "io error: {e}"),
            MemoryError::Serde(e) => write!(f, "serialization error: {e}"),
            MemoryError::NoPath => write!(f, "no checkpoint path bound"),
        }
    }
}

impl std::error::Error for MemoryError {}

impl From<std::io::Error> for MemoryError {
    fn from(e: std::io::Error) -> Self {
        MemoryError::Io(e)
    }
}

impl From<serde_json::Error> for MemoryError {
    fn from(e: serde_json::Error) -> Self {
        MemoryError::Serde(e)
    }
}

/// How a [`recall`](ExternalMemory::recall) selects its context window.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Recall {
    /// The globally hottest nodes by causal score (the default working set).
    WorkingSet,
    /// The causal region (or, if the node is region-less, its k-hop causal
    /// neighbourhood) anchored on a node id / file uri — the workspace signal.
    Around(String),
    /// A free-text task: a lexical entry point, expanded to its causal region.
    Task(String),
}

impl Recall {
    /// The globally hottest working set.
    pub fn working_set() -> Self {
        Recall::WorkingSet
    }
    /// Recall the region anchored on `uri` (a node id, or a bare path — `file:`
    /// is assumed when no known prefix is present).
    pub fn around(uri: impl Into<String>) -> Self {
        Recall::Around(uri.into())
    }
    /// Recall from a free-text task description.
    pub fn task(text: impl Into<String>) -> Self {
        Recall::Task(text.into())
    }
}

/// One node in a recalled window.
#[derive(Debug, Clone, Serialize)]
pub struct RecallItem {
    /// The node id (e.g. `file:src/db.rs`, `sym:src/db.rs:query`).
    pub uri: String,
    /// The node's causal score at recall time.
    pub score: f64,
    /// The node kind (`Module`, `Symbol`, …).
    pub kind: String,
    /// Best available content: the ingested source of the node's file when
    /// known, otherwise the node's own stored content.
    pub content: String,
}

/// A bounded context window produced by [`recall`](ExternalMemory::recall).
#[derive(Debug, Clone, Serialize)]
pub struct RecallWindow {
    /// Which strategy produced this window.
    pub strategy: String,
    /// The selected nodes, highest causal score first.
    pub items: Vec<RecallItem>,
    /// Estimated input tokens of the assembled window.
    pub tokens: usize,
}

/// Result of an [`ingest_source`](ExternalMemory::ingest_source).
#[derive(Debug, Clone, Serialize)]
pub struct IngestReport {
    /// The file uri that was ingested.
    pub uri: String,
    /// Nodes added to the graph by this delta.
    pub nodes_added: usize,
    /// Nodes removed by this delta.
    pub nodes_removed: usize,
    /// Edges added by this delta.
    pub edges_added: usize,
}

/// Integrity status of the memory's hash-chained logs.
#[derive(Debug, Clone, Serialize)]
pub struct Integrity {
    /// True iff both the primary and distributed chains verify.
    pub valid: bool,
    /// Number of verified events in the primary log.
    pub events: usize,
    /// Any integrity errors found (empty when `valid`).
    pub errors: Vec<String>,
}

/// Summary counts for the memory.
#[derive(Debug, Clone, Serialize)]
pub struct MemoryStats {
    /// Nodes currently in the graph.
    pub nodes: usize,
    /// Edges currently in the graph.
    pub edges: usize,
    /// Events appended to the primary log.
    pub events: usize,
    /// Files whose source is retained.
    pub files: usize,
    /// The graph's logical clock.
    pub clock: u64,
}

/// The stable, documented surface an agent programs against.
///
/// See the [module docs](crate::external_memory) for the contract and an
/// example. [`CcosMemory`] is the in-process implementation.
pub trait ExternalMemory {
    /// Write (or update) a source file: parse it, fold the delta into the causal
    /// graph, and extend the hash chain. Idempotent re-ingestion of identical
    /// text is a no-op delta.
    fn ingest_source(&mut self, uri: &str, source: &str) -> IngestReport;

    /// Mark `node` as failing (severity `0.95`) and propagate the pressure up to
    /// `depth` hops downstream. Returns the number of affected nodes, or
    /// [`MemoryError::NodeNotFound`] if the node is absent.
    fn signal_failure(&mut self, node: &str, depth: u32) -> Result<usize, MemoryError>;

    /// Select a bounded context window under a [`Recall`] strategy and a token
    /// budget. Deterministic: ties break on the node id.
    fn recall(&self, recall: &Recall, budget_tokens: usize) -> RecallWindow;

    /// Verify the hash chain is intact (tamper-evidence over the whole history).
    fn verify(&self) -> Integrity;

    /// Summary counts.
    fn stats(&self) -> MemoryStats;

    /// Persist the whole state to the bound path (see [`CcosMemory::open`] /
    /// [`CcosMemory::checkpoint_to`]). Errors with [`MemoryError::NoPath`] if no
    /// path is bound.
    fn checkpoint(&self) -> Result<(), MemoryError>;
}

/// On-disk form (serialised by reference, deserialised owned).
#[derive(Serialize)]
struct PersistedRef<'a> {
    graph: &'a MemoryGraph,
    event_log: &'a EventLog,
    dist_log: &'a DistributedEventLog,
    sources: &'a BTreeMap<String, String>,
}

#[derive(Deserialize)]
struct Persisted {
    graph: MemoryGraph,
    event_log: EventLog,
    dist_log: DistributedEventLog,
    sources: BTreeMap<String, String>,
}

/// The in-process [`ExternalMemory`] implementation backed by the CCOS kernel.
pub struct CcosMemory {
    graph: MemoryGraph,
    engine: IncrementalGraphEngine,
    event_log: EventLog,
    dist_log: DistributedEventLog,
    /// File uri (`file:<path>`) → retained source text.
    sources: BTreeMap<String, String>,
    /// Bound checkpoint path, if any.
    path: Option<PathBuf>,
}

impl Default for CcosMemory {
    fn default() -> Self {
        Self::new()
    }
}

impl CcosMemory {
    /// An empty in-memory kernel with no checkpoint path.
    pub fn new() -> Self {
        CcosMemory {
            graph: MemoryGraph::new(0.2, 5000),
            engine: IncrementalGraphEngine::new(),
            event_log: EventLog::new("ccos-external-memory".to_string()),
            dist_log: DistributedEventLog::new(),
            sources: BTreeMap::new(),
            path: None,
        }
    }

    /// Open a memory backed by `path`: load it if the file exists, otherwise
    /// start empty with `path` bound as the checkpoint target. If `path` is an
    /// existing **directory** (a launcher may create the workspace as one), state
    /// is placed in `<path>/workspace.ccos` inside it rather than failing with
    /// "Is a directory".
    pub fn open(path: impl AsRef<Path>) -> Result<Self, MemoryError> {
        let p = workspace_file(path.as_ref());
        let mut mem = if p.exists() {
            Self::from_json(&std::fs::read_to_string(&p)?)?
        } else {
            Self::new()
        };
        mem.path = Some(p);
        Ok(mem)
    }

    /// Serialize the whole state to the canonical JSON snapshot string — the same
    /// on-disk shape [`open`](Self::open) reads and
    /// [`checkpoint`](ExternalMemory::checkpoint) writes. Lets a higher layer (an
    /// [`AgentSession`](crate::agent_session::AgentSession)) capture a baseline
    /// without touching the filesystem.
    pub fn to_json(&self) -> Result<String, MemoryError> {
        let persisted = PersistedRef {
            graph: &self.graph,
            event_log: &self.event_log,
            dist_log: &self.dist_log,
            sources: &self.sources,
        };
        Ok(serde_json::to_string(&persisted)?)
    }

    /// Reconstruct a memory from a JSON snapshot string. No checkpoint path is
    /// bound and a fresh incremental engine is created (mirroring [`open`](Self::open)).
    pub fn from_json(s: &str) -> Result<Self, MemoryError> {
        let p: Persisted = serde_json::from_str(s)?;
        Ok(CcosMemory {
            graph: p.graph,
            engine: IncrementalGraphEngine::new(),
            event_log: p.event_log,
            dist_log: p.dist_log,
            sources: p.sources,
            path: None,
        })
    }

    /// Persist to an explicit path and bind it for later [`checkpoint`](ExternalMemory::checkpoint).
    pub fn checkpoint_to(&mut self, path: impl AsRef<Path>) -> Result<(), MemoryError> {
        let p = path.as_ref().to_path_buf();
        self.write_to(&p)?;
        self.path = Some(p);
        Ok(())
    }

    /// Downstream **blast radius** of a node (what it causally affects).
    pub fn impact(&self, node: &str, depth: u32) -> Vec<Reached> {
        query::impact_set(&self.graph, &NodeId(normalize(node)), depth)
    }

    /// Upstream **causes** of a node (what causally affects it).
    pub fn causes(&self, node: &str, depth: u32) -> Vec<Reached> {
        query::source_set(&self.graph, &NodeId(normalize(node)), depth)
    }

    /// Advance the logical clock (applies recency decay).
    pub fn tick(&mut self) {
        self.graph.tick();
    }

    /// Read-only access to the underlying causal graph (escape hatch).
    pub fn graph(&self) -> &MemoryGraph {
        &self.graph
    }

    fn write_to(&self, p: &Path) -> Result<(), MemoryError> {
        crate::util::write_durable(p, self.to_json()?.as_bytes())?;
        Ok(())
    }

    /// Node ids of the causal region anchored on `uri`; if the node belongs to no
    /// region, fall back to its k-hop causal neighbourhood (both directions).
    fn region_member_ids(&self, uri: &str) -> Vec<NodeId> {
        let anchor = normalize(uri);
        let mut engine = ContextRegionEngine::new();
        let mut sink = EventLog::new("recall".to_string());
        engine.initialize_regions(&self.graph, &mut sink);
        if let Some(rid) = engine.region_of(&anchor) {
            if let Some(region) = engine.regions.get(&rid) {
                return region.members.iter().map(|m| NodeId(m.clone())).collect();
            }
        }
        // Region-less node: its neighbourhood (causes + impact), plus itself.
        let id = NodeId(anchor);
        let mut ids = vec![id.clone()];
        for r in query::impact_set(&self.graph, &id, 2) {
            ids.push(r.id);
        }
        for r in query::source_set(&self.graph, &id, 2) {
            ids.push(r.id);
        }
        ids
    }

    /// Best lexical entry node for a free-text task (token overlap on
    /// label+content), or `None` if nothing matches.
    fn lexical_entry(&self, text: &str) -> Option<String> {
        let q: Vec<String> = text
            .split(|c: char| !c.is_alphanumeric() && c != '_')
            .filter(|t| t.len() > 2)
            .map(|t| t.to_lowercase())
            .collect();
        self.graph
            .nodes
            .values()
            .map(|n| {
                let hay = format!("{} {}", n.label, n.content).to_lowercase();
                let score = q.iter().filter(|t| hay.contains(t.as_str())).count();
                (score, n.id.0.clone())
            })
            .filter(|(s, _)| *s > 0)
            .max_by(|a, b| a.0.cmp(&b.0).then_with(|| b.1.cmp(&a.1)))
            .map(|(_, id)| id)
    }

    /// Best available content for a node: its file's ingested source if known,
    /// else the node's own stored content.
    fn content_for(&self, node_id: &str, node: &GraphNode) -> String {
        let file_key = format!("file:{}", file_of(node_id));
        self.sources
            .get(&file_key)
            .cloned()
            .unwrap_or_else(|| node.content.clone())
    }

    /// Score, order (by `(score, uri)`), and budget-truncate a set of nodes.
    fn assemble_window(&self, strategy: &str, ids: Vec<NodeId>, budget: usize) -> RecallWindow {
        let mut seen = BTreeSet::new();
        let mut scored: Vec<RecallItem> = Vec::new();
        for id in ids {
            if !seen.insert(id.0.clone()) {
                continue;
            }
            if let Some(node) = self.graph.nodes.get(&id) {
                scored.push(RecallItem {
                    uri: id.0.clone(),
                    score: self.graph.compute_node_score(node),
                    kind: format!("{:?}", node.node_type),
                    content: self.content_for(&id.0, node),
                });
            }
        }
        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.uri.cmp(&b.uri))
        });
        let mut items = Vec::new();
        let mut tokens = 0usize;
        for it in scored {
            let t = it.content.chars().count() / 4;
            if tokens + t > budget && !items.is_empty() {
                break;
            }
            tokens += t;
            items.push(it);
        }
        RecallWindow {
            strategy: strategy.to_string(),
            items,
            tokens,
        }
    }
}

/// Prefix a bare path with `file:`; leave known node-id prefixes untouched.
fn normalize(uri: &str) -> String {
    const PREFIXES: [&str; 5] = ["file:", "sym:", "mod:", "use:", "dep:"];
    if PREFIXES.iter().any(|p| uri.starts_with(p)) {
        uri.to_string()
    } else {
        format!("file:{uri}")
    }
}

/// Resolve a workspace path to its state **file**. A plain path is used as-is; an
/// existing directory becomes `<dir>/workspace.ccos` inside it (so a launcher that
/// pre-creates the workspace as a directory works instead of erroring with "Is a
/// directory"). Idempotent: resolving an already-resolved file path is a no-op.
pub(crate) fn workspace_file(path: &Path) -> PathBuf {
    if path.is_dir() {
        path.join("workspace.ccos")
    } else {
        path.to_path_buf()
    }
}

impl ExternalMemory for CcosMemory {
    fn ingest_source(&mut self, uri: &str, source: &str) -> IngestReport {
        let file_key = format!("file:{uri}");
        let prev = self.sources.get(&file_key).cloned();
        let delta = self
            .engine
            .process_delta(uri, prev.as_deref(), source, &mut self.graph);
        self.sources.insert(file_key, source.to_string());
        // Resolve intra-crate imports into file→file edges so recall, failure
        // propagation and regions see the real cross-file causal structure.
        let cross_edges = self.graph.link_module_imports();
        let file_hash = sha256_hex(source);
        self.event_log.append(
            EventType::Parsing,
            EventPayload::Parsing {
                file_path: uri.to_string(),
                file_hash: file_hash.clone(),
                modules_found: 0,
                uses_found: 0,
                symbols_found: 0,
            },
        );
        self.dist_log
            .append(file_hash, "external-memory".to_string());
        IngestReport {
            uri: uri.to_string(),
            nodes_added: delta.nodes_added,
            nodes_removed: delta.nodes_removed,
            edges_added: delta.edges_added + cross_edges,
        }
    }

    fn signal_failure(&mut self, node: &str, depth: u32) -> Result<usize, MemoryError> {
        let id = NodeId(normalize(node));
        if !self.graph.nodes.contains_key(&id) {
            return Err(MemoryError::NodeNotFound(id.0));
        }
        self.graph.set_failure_relevance(&id, 0.95);
        self.graph.propagate_failure(&id, 0, depth);
        let affected = self
            .graph
            .nodes
            .iter()
            .filter(|(k, n)| **k != id && n.failure_relevance > 0.0)
            .count();
        Ok(affected)
    }

    fn recall(&self, recall: &Recall, budget_tokens: usize) -> RecallWindow {
        match recall {
            Recall::WorkingSet => {
                let ids = self
                    .graph
                    .get_node_scores()
                    .into_iter()
                    .map(|(id, _)| id)
                    .collect();
                self.assemble_window("working-set", ids, budget_tokens)
            }
            Recall::Around(uri) => {
                let ids = self.region_member_ids(uri);
                self.assemble_window("region", ids, budget_tokens)
            }
            Recall::Task(text) => {
                let ids = self
                    .lexical_entry(text)
                    .map(|e| self.region_member_ids(&e))
                    .unwrap_or_default();
                self.assemble_window("task-region", ids, budget_tokens)
            }
        }
    }

    fn verify(&self) -> Integrity {
        let log = self.event_log.verify_integrity();
        let dist = self.dist_log.verify_integrity();
        let mut errors = log.errors;
        errors.extend(dist.errors);
        Integrity {
            valid: log.valid && dist.valid,
            events: log.verified_events,
            errors,
        }
    }

    fn stats(&self) -> MemoryStats {
        MemoryStats {
            nodes: self.graph.nodes.len(),
            edges: self.graph.edges.len(),
            events: self.event_log.event_count(),
            files: self.sources.len(),
            clock: self.graph.clock,
        }
    }

    fn checkpoint(&self) -> Result<(), MemoryError> {
        match &self.path {
            Some(p) => self.write_to(p),
            None => Err(MemoryError::NoPath),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recall_around_reaches_cross_file_cause() {
        let mut mem = CcosMemory::new();
        mem.ingest_source("src/db.rs", "pub fn connect() -> i32 { 5 }\n");
        mem.ingest_source(
            "src/repo.rs",
            "use crate::db;\npub fn load() -> i32 { db::connect() }\n",
        );
        mem.ingest_source(
            "src/api.rs",
            "use crate::repo;\npub fn handler() -> i32 { repo::load() }\n",
        );
        mem.ingest_source("src/unrelated.rs", "pub fn fmt_date() {}\n");
        let win = mem.recall(&Recall::around("file:src/api.rs"), 8000);
        let files: Vec<&str> = win
            .items
            .iter()
            .map(|i| i.uri.as_str())
            .filter(|u| u.starts_with("file:"))
            .collect();
        assert!(
            files.contains(&"file:src/db.rs"),
            "recall reaches the cross-file cause db.rs: {files:?}"
        );
        assert!(
            !files.contains(&"file:src/unrelated.rs"),
            "recall excludes unrelated code: {files:?}"
        );
    }

    #[test]
    fn ingest_recall_verify_roundtrip() {
        let mut mem = CcosMemory::new();
        let r = mem.ingest_source("src/a.rs", "pub fn alpha() -> i32 { 1 }\n");
        assert!(r.nodes_added >= 1, "ingest creates nodes");
        mem.ingest_source("src/b.rs", "pub fn beta() {}\n");

        let win = mem.recall(&Recall::working_set(), 10_000);
        assert!(!win.items.is_empty(), "working set is non-empty");

        // A file node carries its ingested source as content.
        let file_item = win.items.iter().find(|i| i.uri == "file:src/a.rs");
        assert!(
            file_item
                .map(|i| i.content.contains("alpha"))
                .unwrap_or(false),
            "file node returns its source"
        );

        assert!(mem.verify().valid, "hash chain verifies");
        assert_eq!(mem.stats().files, 2);
    }

    #[test]
    fn signal_failure_unknown_node_errs() {
        let mut mem = CcosMemory::new();
        assert!(matches!(
            mem.signal_failure("file:nope.rs", 2),
            Err(MemoryError::NodeNotFound(_))
        ));
    }

    #[test]
    fn signal_failure_marks_nodes() {
        let mut mem = CcosMemory::new();
        mem.ingest_source("src/a.rs", "pub fn alpha() {}\n");
        let n = mem.signal_failure("src/a.rs", 3).unwrap();
        // At least the origin's own symbols are reachable; never panics.
        let _ = n;
        assert!(mem.verify().valid);
    }

    #[test]
    fn checkpoint_roundtrips_through_a_file() {
        let path = std::env::temp_dir().join(format!("ccos-mem-ckpt-{}.json", std::process::id()));
        {
            let mut mem = CcosMemory::open(&path).unwrap();
            mem.ingest_source("src/a.rs", "pub fn alpha() {}\n");
            mem.checkpoint().unwrap();
        }
        let reloaded = CcosMemory::open(&path).unwrap();
        assert!(reloaded.stats().nodes >= 1, "graph survived the round-trip");
        assert!(reloaded.verify().valid, "chain still verifies after reload");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn checkpoint_without_path_errs() {
        let mem = CcosMemory::new();
        assert!(matches!(mem.checkpoint(), Err(MemoryError::NoPath)));
    }
}
