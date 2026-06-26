use crate::cold_index::HuskStore;
use crate::compressor::{CausalAst, CausalCrusher, CausalSumm, ContentRouter, Route};
use crate::eviction_policy::{
    bucket_pressure, bucket_recency, bucket_score, bucket_size, EvictionPolicy, PagingState, EVICT,
    KEEP,
};
use crate::util::{hex32, sha256_bytes};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap};
use std::path::PathBuf;

/// Memtable size (husks buffered before flushing to a segment) and read-cache
/// capacity for the Lever 2 [`HuskStore`]. Runtime tuning constants; generous enough
/// that the on-disk index stays cheap on small graphs.
const HUSK_BUFFER_LIMIT: usize = 4096;
const HUSK_CACHE_CAP: usize = 1024;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub String);

impl From<&str> for NodeId {
    fn from(s: &str) -> Self {
        NodeId(s.to_string())
    }
}

impl From<String> for NodeId {
    fn from(s: String) -> Self {
        NodeId(s)
    }
}

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GraphNode {
    pub id: NodeId,
    pub label: String,
    pub content: String,
    pub node_type: NodeType,
    pub base_importance: f64,
    pub failure_relevance: f64,
    pub recency: f64,
    pub access_count: u64,
    pub created_at: u64,
    pub last_accessed: u64,
    /// Lifecycle state (see [`NodeState`]). Kept **separate from topology** so a node's
    /// health/attention cannot pollute the structural centrality signal. `serde(default)` +
    /// skip-if-`Stable` keeps existing snapshots byte-identical on the default path.
    #[serde(default, skip_serializing_if = "NodeState::is_stable")]
    pub state: NodeState,
}

/// Lifecycle state of a node, kept orthogonal to graph topology so a node's *health* /
/// *attention* does not distort the structural signal centrality reads. Default `Stable`,
/// so it is off-by-default and snapshot-compatible.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum NodeState {
    /// Verified code, part of the load-bearing structure. The default.
    #[default]
    Stable,
    /// Code under active modification — "hot" (the current focus) but structurally fragile.
    /// **Pinned** in working memory (a retention boost) even as its recency decays.
    Working,
    /// Dead / unreachable code — **excluded** from the structural centrality calc, and
    /// **evicted first** even when recently touched.
    Orphan,
}

impl NodeState {
    /// `serde` skip predicate: elide the default so snapshots stay byte-identical.
    fn is_stable(&self) -> bool {
        matches!(self, NodeState::Stable)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum NodeType {
    Module,
    Symbol,
    ContextBlock,
    AnalysisResult,
    CodeRegion,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    pub source: NodeId,
    pub target: NodeId,
    pub weight: f64,
    pub edge_type: EdgeType,
    pub created_at: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EdgeType {
    DependsOn,
    Contains,
    References,
    Causes,
    RelatedTo,
    /// A `caller → callee` function-call edge, resolved from in-body call-sites by
    /// [`MemoryGraph::resolve_symbol_calls`]. Captures fn→fn structure that import edges miss
    /// (a call crosses files even when the two functions share no vocabulary). Appended last so
    /// the enum's serialized form is strictly additive — old snapshots never contain `Calls`.
    Calls,
    /// A `function → static/const` data-flow edge: the function reads (references) a module-level
    /// `static`/`const` item, resolved from in-body data-references by
    /// [`MemoryGraph::resolve_data_flow`]. Captures the shared-mutable-state channel that connects
    /// functions through globals — invisible to call and import edges. Appended last so the
    /// serialized form stays strictly additive.
    DataFlow,
}

/// Tunable coefficients of the causal score and failure-propagation decay.
///
/// A node's score is
/// `clamp(w_base·imp + w_failure·fail + w_recency·rec + w_access·ln(1+acc), 0, 1)`,
/// and injected failure pressure attenuates by `failure_decay^depth` per hop.
/// [`Default`] reproduces the constants CCOS shipped with; [`ScoringWeights::from_env`]
/// lets an external optimiser (the causal-validation harness) override them per
/// run without recompiling — the knobs Phase 3 of that harness searches over.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct ScoringWeights {
    /// Weight on a node's intrinsic importance.
    pub w_base: f64,
    /// Weight on propagated failure relevance (the dominant term by default).
    pub w_failure: f64,
    /// Weight on recency.
    pub w_recency: f64,
    /// Weight on `ln(access_count)`.
    pub w_access: f64,
    /// Weight on **structural centrality** — `ln(1 + in_degree)`, the number of
    /// incoming causal edges. A hub (a shared module / interface that many nodes
    /// depend on) is structurally more important than a leaf, independent of how
    /// recently it was touched. **Default `0.0`** (off): the score is then
    /// byte-identical to before this term existed, so snapshots/replay are
    /// unchanged. Set it (or let [`crate::agent_session::AgentSession::tune_recall_weights`]
    /// learn it) to retain hubs more strongly. `skip_serializing_if` elides it when
    /// `0.0` so the default serialized form is unchanged.
    #[serde(default, skip_serializing_if = "is_zero_f64")]
    pub w_centrality: f64,
    /// How the centrality term measures structural importance (only consulted when
    /// `w_centrality != 0`). [`CentralityMode::InDegree`] (default) is the raw incoming-
    /// edge count `ln(1 + in_degree)` — cheap and local. [`CentralityMode::Eigenvector`]
    /// is a *global*, recursive importance (damped power iteration, see
    /// [`eigencentrality`](MemoryGraph::eigencentrality)): a node is important if
    /// *important* nodes depend on it. Elided from the serialized form when default, so
    /// existing snapshots are byte-identical.
    #[serde(default, skip_serializing_if = "CentralityMode::is_default")]
    pub centrality_mode: CentralityMode,
    /// Geometric attenuation of failure pressure per propagation hop.
    pub failure_decay: f64,
    /// Out-degree at which a node starts **distributing** (rather than
    /// replicating) failure pressure across its edges. At or below this fan-out,
    /// propagation is unchanged — so sparse causal chains still reach depth; above
    /// it, a hub's emission is damped by `failure_fanout / out_degree`, so one
    /// over-connected node (e.g. a file with dozens of contained symbols) cannot
    /// flood the graph. See `docs/FIELD_CAMPAIGN_H.md` (root cause #2).
    #[serde(default = "default_failure_fanout")]
    pub failure_fanout: f64,
}

/// How [`ScoringWeights::centrality_mode`] turns graph structure into a node's
/// centrality score.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum CentralityMode {
    /// `ln(1 + in_degree)` — importance = the raw count of incoming causal edges.
    /// Local and O(1) per node (one cached in-degree pass); the shipped behaviour.
    #[default]
    InDegree,
    /// **Eigenvector centrality** — importance propagates recursively: a node is
    /// important if *important* nodes depend on it, so a hub that many *other hubs*
    /// rely on outranks a hub that only leaves rely on (which in-degree cannot tell
    /// apart). Computed by deterministic **damped power iteration** — see
    /// [`MemoryGraph::eigencentrality`] for why the damping (Katz/PageRank form) is
    /// the correct realization of `A x = λ x` on a code graph, which is largely a DAG.
    Eigenvector,
}

impl CentralityMode {
    /// `serde` skip predicate: elide the default so existing snapshots are unchanged.
    fn is_default(&self) -> bool {
        matches!(self, CentralityMode::InDegree)
    }
}

/// Default [`ScoringWeights::failure_fanout`]; also fills the field when an older
/// snapshot (written before it existed) is deserialised.
fn default_failure_fanout() -> f64 {
    6.0
}

/// `skip_serializing_if` predicate: elide an `f64` weight when it is exactly `0.0`
/// (an off-by-default term), keeping the serialized form unchanged.
fn is_zero_f64(x: &f64) -> bool {
    *x == 0.0
}

impl Default for ScoringWeights {
    fn default() -> Self {
        Self {
            w_base: 0.15,
            w_failure: 0.50,
            w_recency: 0.30,
            w_access: 0.05,
            w_centrality: 0.0,
            centrality_mode: CentralityMode::InDegree,
            failure_decay: 0.8,
            failure_fanout: default_failure_fanout(),
        }
    }
}

impl ScoringWeights {
    /// Read overrides from the environment, falling back to [`Default`] for any
    /// variable that is unset or unparsable. Recognised variables: `CCOS_W_BASE`,
    /// `CCOS_W_FAILURE`, `CCOS_W_RECENCY`, `CCOS_W_ACCESS`, `CCOS_FAILURE_DECAY`,
    /// `CCOS_FAILURE_FANOUT`.
    pub fn from_env() -> Self {
        let d = Self::default();
        let get = |key: &str, fallback: f64| -> f64 {
            std::env::var(key)
                .ok()
                .and_then(|v| v.trim().parse::<f64>().ok())
                .filter(|x| x.is_finite())
                .unwrap_or(fallback)
        };
        Self {
            w_base: get("CCOS_W_BASE", d.w_base),
            w_failure: get("CCOS_W_FAILURE", d.w_failure),
            w_recency: get("CCOS_W_RECENCY", d.w_recency),
            w_access: get("CCOS_W_ACCESS", d.w_access),
            w_centrality: get("CCOS_W_CENTRALITY", d.w_centrality),
            // `CCOS_CENTRALITY_MODE=eigenvector` switches to the global recursive
            // centrality; anything else (or unset) keeps the in-degree default.
            centrality_mode: match std::env::var("CCOS_CENTRALITY_MODE")
                .ok()
                .as_deref()
                .map(str::trim)
                .map(str::to_ascii_lowercase)
                .as_deref()
            {
                Some("eigenvector") | Some("eigen") | Some("pagerank") => {
                    CentralityMode::Eigenvector
                }
                _ => d.centrality_mode,
            },
            failure_decay: get("CCOS_FAILURE_DECAY", d.failure_decay),
            failure_fanout: get("CCOS_FAILURE_FANOUT", d.failure_fanout),
        }
    }
}

/// Serialize a `NodeId → GraphNode` map in **sorted key order** so a snapshot is
/// byte-canonical. The resident node map is a `HashMap` (O(1) lookups on the hot
/// path) whose iteration order is nondeterministic, so a plain derive lets two
/// memories with identical state serialize to different bytes (same length, shuffled
/// order). Sorting on the way out makes the stronger invariant hold — *identical
/// state ⇒ byte-identical snapshot*, not merely identical *sorted* hash.
/// Deserialization is unaffected: a `HashMap` reads from a JSON map in any order.
fn serialize_sorted_nodes<S>(
    nodes: &HashMap<NodeId, GraphNode>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    use serde::ser::SerializeMap;
    let mut keys: Vec<&NodeId> = nodes.keys().collect();
    keys.sort();
    let mut map = serializer.serialize_map(Some(keys.len()))?;
    for k in keys {
        map.serialize_entry(k, &nodes[k])?;
    }
    map.end()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryGraph {
    #[serde(serialize_with = "serialize_sorted_nodes")]
    pub(crate) nodes: HashMap<NodeId, GraphNode>,
    pub(crate) edges: Vec<GraphEdge>,
    pub paging_threshold: f64,
    pub max_in_memory_nodes: usize,
    pub clock: u64,
    /// Scoring/decay coefficients (see [`ScoringWeights`]). Serialised so a
    /// snapshot records the weights it was scored under; `serde(default)` keeps
    /// older snapshots (written before this field existed) loadable.
    #[serde(default)]
    pub scoring_weights: ScoringWeights,
    /// Learned eviction policy consulted by [`enforce_paging`](Self::enforce_paging).
    /// Default is **untrained**, in which case eviction is exactly the
    /// deterministic greedy (lowest score first) — so enabling it is never worse.
    /// Train it offline ([`EvictionPolicy::fit`]) and inject via
    /// [`set_eviction_policy`](Self::set_eviction_policy). Serialised (a `BTreeMap`
    /// Q-table) so a snapshot records the policy it paged under; `serde(default)`
    /// keeps pre-existing snapshots loadable.
    #[serde(default = "EvictionPolicy::new")]
    pub eviction_policy: EvictionPolicy,
    /// The **COLD tier** — the "swap". Nodes evicted from the resident graph by
    /// [`enforce_paging`](Self::enforce_paging) are *demoted* here instead of
    /// dropped, with the edges incident to them at demotion time, so the working
    /// memory is **non-destructive**: nothing is lost, anything can be
    /// [`page_in`](Self::page_in)ed back on demand. The resident set
    /// ([`node_count`](Self::node_count)) stays bounded by `max_in_memory_nodes`;
    /// the backing store (this map) is the unbounded "virtual memory" behind it.
    /// `BTreeMap` ⇒ deterministic iteration/serialization; `serde(default)` keeps
    /// pre-existing snapshots loadable.
    #[serde(default)]
    cold: BTreeMap<NodeId, ColdNode>,
    /// Optional on-disk **spill store** for COLD content — the "swap file". A
    /// runtime handle only (`#[serde(skip)]`): a snapshot records the spilled
    /// *stubs* (see [`ColdNode::spill`]), and the directory travels alongside,
    /// re-attached on restore via [`attach_cold_spill`](Self::attach_cold_spill).
    /// `None` (the default) ⇒ COLD content stays fully resident in RAM,
    /// byte-identical to a graph that never knew about spill, so every existing
    /// invariant (replay == live, snapshot hash) is untouched on the default path.
    #[serde(skip)]
    spill: Option<SpillConfig>,
    /// Lever 2 (slice 5c): the **authoritative** on-disk index of deep-spilled husks.
    /// The COLD tier's *entry count* is no longer `O(N)` resident — husks live in
    /// segments + a bounded cache, not one `BTreeMap` node each. Runtime handle only
    /// (`#[serde(skip)]`), opened in a sibling `<dir>.husks` directory and re-attached
    /// on restore like the spill store; deep-spilled state thus travels in that
    /// directory, not the JSON snapshot. `None` (the default, no spill attached) ⇒ no
    /// deep tier, byte-identical to a graph that never knew about it.
    #[serde(skip)]
    husk_store: Option<HuskStore>,
    /// Lever 2 brick 8: an on-disk **reverse-adjacency index** of the COLD tier —
    /// for every archived cold edge `a─b`, the keys `"a\x1fb"` and `"b\x1fa"`. Lets
    /// [`cold_neighbours`](Self::cold_neighbours) answer with one prefix scan
    /// (`O(degree)`) instead of deserializing the whole deep tier (`O(N)`). Maintained
    /// at demote (add) and page-in / removal (drop), opened in a sibling `<dir>.radj`
    /// directory, runtime-only (`#[serde(skip)]`) and backfilled from the current cold
    /// tier on attach. `None` ⇒ no spill attached, and `cold_neighbours` falls back to
    /// the resident edge scan (there is no deep tier then).
    #[serde(skip)]
    radj: Option<HuskStore>,
    /// Optional **compaction budget** for the COLD tier (slice 4). When `Some(b)`,
    /// total COLD *content* (inline + spilled) is kept toward `b` bytes by
    /// **lossily compacting** the coldest entries — code is skeletonised, prose
    /// summarised (CausalSumm/CausalAst), the full original discarded — so the
    /// *backing store itself* stays frugal. A runtime knob (`#[serde(skip)]`);
    /// `None` (default) ⇒ no compaction, COLD stays lossless. This is the deepest
    /// tier: "infinite" working memory is a *direction*, and at the bottom
    /// frugality wins — CCOS compacts to a summary (observable via
    /// [`is_compacted`](Self::is_compacted)), never silently drops.
    #[serde(skip)]
    cold_content_budget: Option<usize>,
    /// Optional **resident-metadata budget** for the COLD tier (slice 5). When
    /// `Some(b)`, the bytes the COLD tier keeps in RAM ([`cold_resident_bytes`](Self::cold_resident_bytes))
    /// are driven toward `b` by **deep-spilling** the coldest entries — moving each
    /// one's `label` and full `edges` to the on-disk store and keeping only the
    /// neighbour ids (`adj`) resident. Still **lossless** (the body
    /// faults back on [`page_in`](Self::page_in)); the measured-dominant resident
    /// cost (edges, `docs/MEASUREMENT_cold_ram.md`) is shrunk to ids, not dropped or
    /// contracted. A runtime knob (`#[serde(skip)]`); `None` (default) ⇒ no
    /// deep-spill, COLD metadata stays fully resident — byte-identical on the
    /// default path. Needs an attached spill store (the same "swap file").
    #[serde(skip)]
    cold_resident_budget: Option<usize>,
    /// Cached in-degree map for the structural-centrality score term, keyed on
    /// `edges.len()` (edges are only ever appended or `retain`-pruned, so the
    /// length changes whenever the edge set does). Runtime-only; rebuilt lazily,
    /// and only ever consulted when `scoring_weights.w_centrality != 0`.
    #[serde(skip)]
    indegree_cache: RefCell<Option<(usize, HashMap<NodeId, u32>)>>,
    /// Cached eigenvector-centrality vector for the structural-centrality score term,
    /// keyed on `(nodes.len(), edges.len())` (a cheap proxy for "the graph changed",
    /// matching [`indegree_cache`](Self::indegree_cache)). Runtime-only; rebuilt lazily
    /// by deterministic power iteration, and only ever consulted when
    /// `w_centrality != 0` *and* `centrality_mode == Eigenvector`.
    #[serde(skip)]
    eigencentrality_cache: RefCell<EigenCentralityCache>,
    /// In-body call-sites awaiting resolution into `Calls` edges, keyed by source file →
    /// `(caller, callee, line)` in source order. Populated by the parser at ingest, consumed by
    /// [`resolve_symbol_calls`](Self::resolve_symbol_calls). **Runtime-only** (`serde(skip)`): the
    /// resolved `Calls` *edges* are the durable state; this raw input is rebuilt on the replay
    /// re-ingest, so the snapshot is unchanged and `replay == live` holds. `BTreeMap` ⇒ sorted,
    /// deterministic iteration.
    #[serde(skip)]
    pending_calls: BTreeMap<String, Vec<(String, String, usize)>>,
    /// In-body `static`/`const` references awaiting resolution into `DataFlow` edges, keyed by
    /// source file → `(reader, name, line)`. Same **runtime-only** contract as `pending_calls`.
    #[serde(skip)]
    pending_data_refs: BTreeMap<String, Vec<(String, String, usize)>>,
    /// Renamed-import bindings for call resolution, keyed by source file → `(local_name,
    /// target_path)` — one entry per `use a::b as c` (`("c", "a::b")`). Consulted by
    /// [`resolve_symbol_calls`](Self::resolve_symbol_calls) to rewrite a call to an alias onto its
    /// real target. Same **runtime-only** (`serde(skip)`, rebuilt on the replay re-ingest) and
    /// deterministic-`BTreeMap` contract as `pending_calls`.
    #[serde(skip)]
    pending_aliases: BTreeMap<String, Vec<(String, String)>>,
    /// Node ids of `static`/`const` symbols (the only valid `DataFlow` targets). The graph node
    /// stores `NodeType`, not the finer `SymbolKind`, so the parser marks these at ingest. Runtime
    /// -only; rebuilt on the replay re-ingest, and filtered to still-resident nodes at resolve time.
    #[serde(skip)]
    data_symbols: std::collections::BTreeSet<NodeId>,
}

/// Cache value behind [`MemoryGraph::node_eigencentrality`]: the centrality vector keyed
/// by `(nodes.len(), edges.len())` — a cheap "did the graph change" stamp. Aliased to keep
/// the field type readable (and clippy's `type_complexity` happy).
type EigenCentralityCache = Option<((usize, usize), HashMap<NodeId, f64>)>;

/// A bound spill store plus the resident-content budget that triggers a flush.
/// Not serialised — held only while a [`MemoryGraph`] is live.
#[derive(Debug, Clone)]
struct SpillConfig {
    store: ColdSpill,
    /// Max bytes of COLD *content* kept inline (resident) before the coldest is
    /// flushed to disk. `usize::MAX` ⇒ never flush.
    inline_budget: usize,
}

/// An on-disk, content-addressed store for vacated COLD content — the unbounded
/// "swap" backing the bounded resident window. A blob is keyed by the **SHA-256**
/// of its content (the same addressing scheme as the CCR store,
/// [`crate::compressor::CcrRef`]), so identical content is **deduplicated** to a
/// single file, and a read **re-verifies** the hash: a truncated or tampered
/// blob is a detectable miss, never a silent empty restore. Spill thus *extends*
/// the integrity story rather than weakening it. Lossless and verbatim (no codec
/// yet — content-dedup is the only space win at this layer).
#[derive(Debug, Clone)]
pub struct ColdSpill {
    dir: PathBuf,
}

impl ColdSpill {
    /// Open (creating if needed) a spill directory.
    pub fn new(dir: impl Into<PathBuf>) -> std::io::Result<Self> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)?;
        Ok(Self { dir })
    }

    /// Write `content` addressed by its SHA-256, returning the raw 32-byte key. The
    /// on-disk filename is its [`hex32`] and the file holds the **LZSS-compressed**
    /// bytes (lossless; the key/integrity hash is of the *original* content, so
    /// dedup is unchanged). Idempotent — content-addressed *and* the codec is
    /// deterministic, so re-spilling the same blob is a no-op write.
    fn put(&self, content: &str) -> std::io::Result<[u8; 32]> {
        let hash = sha256_bytes(content);
        let path = self.dir.join(hex32(&hash));
        if !path.exists() {
            std::fs::write(&path, crate::lzss::compress(content.as_bytes()))?;
        }
        Ok(hash)
    }

    /// Read a blob by hash: decompress, then **verify** integrity. `None` if the
    /// file is missing, the codec stream is malformed, the bytes aren't UTF-8, or
    /// the decompressed content no longer hashes to `hash` (tampered/corrupt /
    /// codec bug) — all surfaced as a cold-miss by the caller, never a silent wrong
    /// restore.
    fn get(&self, hash: &[u8; 32]) -> Option<String> {
        let blob = std::fs::read(self.dir.join(hex32(hash))).ok()?;
        let text = String::from_utf8(crate::lzss::decompress(&blob)?).ok()?;
        (sha256_bytes(&text) == *hash).then_some(text)
    }

    /// Delete a blob by hash (best-effort; a missing file is fine). Used to
    /// reclaim a spilled original once no COLD entry references it any more — the
    /// content-addressed store's garbage collection.
    fn remove(&self, hash: &[u8; 32]) {
        let _ = std::fs::remove_file(self.dir.join(hex32(hash)));
    }
}

/// A node demoted out of the resident graph into the [`MemoryGraph`] COLD tier,
/// kept with the edges incident to it at demotion time so a later `page_in` can
/// restore its causal structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ColdNode {
    node: GraphNode,
    edges: Vec<GraphEdge>,
    /// When `Some`, this node's `content` blob has been vacated to the on-disk
    /// spill store and `node.content` is empty; it must be faulted back in
    /// (verified by hash) before the node can be paged resident. `None` ⇒
    /// content is inline (resident), the default and the only state when no
    /// spill store is attached. `skip_serializing_if` keeps the serialized form
    /// byte-identical to the pre-spill layout whenever nothing is spilled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    spill: Option<SpillRef>,
    /// When `true`, `node.content` is a **lossy** compaction (a CausalSumm summary
    /// or CausalAst skeleton) of the original, produced once the COLD compaction
    /// budget was exceeded; the full original has been discarded. Elided from the
    /// serialized form when `false` (the default) so the layout is unchanged
    /// whenever nothing is compacted.
    #[serde(default, skip_serializing_if = "is_false")]
    compacted: bool,
    /// When `true`, compaction tried this entry and could not make it any smaller
    /// (its content is already at the summary/skeleton floor), so it is excluded
    /// from future compaction candidates — otherwise every ingest would re-attempt
    /// it (and re-read its blob from disk) for no gain, and a tier of only
    /// un-shrinkable entries would busy-loop the enforcer. A fresh ingest of the
    /// node drops the whole COLD shadow, so the flag never goes stale. Elided when
    /// `false` (the default) — byte-identical serialization.
    #[serde(default, skip_serializing_if = "is_false")]
    at_floor: bool,
}

/// `skip_serializing_if` predicate: elide a `bool` field when it is `false`.
fn is_false(b: &bool) -> bool {
    !*b
}

/// A **deep-spilled** COLD entry (slice 5b): the most aggressive, still-lossless
/// tier. The *entire* `ColdNode` (node, content folded inline, edges, flags) is
/// serialized into one content-addressed blob, and all that stays resident is this
/// compact husk — the body-blob stub plus the neighbour **ids** (`adj`), so
/// [`cold_neighbours`](MemoryGraph::cold_neighbours) and region paging keep working
/// without touching disk. Replacing the full ~`size_of::<ColdNode>()` struct with
/// this husk is what bounds the per-entry resident *floor* the measurement flagged
/// (`docs/MEASUREMENT_cold_ram.md`); the node faults back, hash-verified, on
/// [`page_in`](MemoryGraph::page_in). Deep husks are **terminal** — already at the
/// floor, they are not re-scored for further spill/compaction.
/// Pack neighbour ids into one length-prefixed byte buffer (each id: a `u32` LE
/// length + its UTF-8 bytes). One allocation for the whole adjacency instead of a
/// `Vec` plus a `String` per id — slice 5c Lever 1, attacking the ~9× allocation
/// overhead the measurement found (`docs/DESIGN_cold_entry_count.md`).
fn pack_adj(ids: &[NodeId]) -> Box<[u8]> {
    let mut buf = Vec::new();
    for id in ids {
        let bytes = id.0.as_bytes();
        buf.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(bytes);
    }
    buf.into_boxed_slice()
}

/// Iterate the ids packed by [`pack_adj`] without allocating (yields borrowed
/// `&str`). Stops cleanly on a truncated or non-UTF-8 buffer.
fn unpack_adj(packed: &[u8]) -> impl Iterator<Item = &str> {
    let mut pos = 0;
    std::iter::from_fn(move || {
        let lb = packed.get(pos..pos + 4)?;
        let len = u32::from_le_bytes(lb.try_into().ok()?) as usize;
        pos += 4;
        let s = std::str::from_utf8(packed.get(pos..pos + len)?).ok()?;
        pos += len;
        Some(s)
    })
}

/// (De)serialize the packed adjacency as a plain JSON array of id strings, so a
/// deep-husk snapshot is byte-identical to the previous `Vec<NodeId>` layout while
/// RAM holds the compact packed form.
mod packed_adj {
    use super::{pack_adj, unpack_adj, NodeId};
    use serde::ser::SerializeSeq;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(adj: &[u8], s: S) -> Result<S::Ok, S::Error> {
        let ids: Vec<&str> = unpack_adj(adj).collect();
        let mut seq = s.serialize_seq(Some(ids.len()))?;
        for id in ids {
            seq.serialize_element(id)?;
        }
        seq.end()
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Box<[u8]>, D::Error> {
        let ids: Vec<String> = Vec::deserialize(d)?;
        Ok(pack_adj(&ids.into_iter().map(NodeId).collect::<Vec<_>>()))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DeepHusk {
    /// Stub of the on-disk blob holding the whole serialized `ColdNode`.
    body: SpillRef,
    /// Neighbour ids — the other endpoint of each edge the entry carried at
    /// deep-spill time — **packed** into one length-prefixed buffer (sorted + deduped
    /// at deep-spill, so deterministic). The only structure kept resident, so the
    /// cold↔cold adjacency survives without faulting the blob; see [`pack_adj`].
    #[serde(with = "packed_adj")]
    adj: Box<[u8]>,
}

/// (De)serialize a 32-byte content hash as its lowercase-hex string, so the stub
/// keeps its compact in-RAM form (a raw `[u8; 32]` — no heap allocation) while the
/// *serialized* snapshot stays the same readable, canonical hex it always was.
mod hex_hash {
    use crate::util::{from_hex32, hex32};
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(h: &[u8; 32], s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&hex32(h))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<[u8; 32], D::Error> {
        let s = String::deserialize(d)?;
        from_hex32(&s).ok_or_else(|| serde::de::Error::custom("invalid 32-byte hex hash"))
    }
}

/// A stub left in RAM after a COLD node's content is flushed to the on-disk spill
/// store: enough to fault it back (the hash key + integrity check) and to account
/// for its disk footprint (the original length) without touching the disk. The hash
/// is held raw (`[u8; 32]`, no heap) — its on-disk filename and serialized form are
/// the hex of it.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SpillRef {
    /// Raw SHA-256 of the vacated content — the on-disk key (as [`hex32`]) *and* the
    /// read-time integrity check.
    #[serde(with = "hex_hash")]
    hash: [u8; 32],
    /// Byte length of the original content (so budget/stat math need not fault
    /// the blob back in).
    len: usize,
}

impl MemoryGraph {
    pub fn new(paging_threshold: f64, max_in_memory_nodes: usize) -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
            paging_threshold,
            max_in_memory_nodes,
            clock: 0,
            scoring_weights: ScoringWeights::default(),
            eviction_policy: EvictionPolicy::new(),
            cold: BTreeMap::new(),
            spill: None,
            husk_store: None,
            radj: None,
            cold_content_budget: None,
            cold_resident_budget: None,
            indegree_cache: RefCell::new(None),
            eigencentrality_cache: RefCell::new(None),
            pending_calls: BTreeMap::new(),
            pending_data_refs: BTreeMap::new(),
            pending_aliases: BTreeMap::new(),
            data_symbols: std::collections::BTreeSet::new(),
        }
    }

    /// The paging eviction floor, overridable by `CCOS_PAGING_THRESHOLD` (else `default`). The same
    /// env-override convention as [`ScoringWeights::from_env`] — a non-finite or unparsable value
    /// falls back to the default, so a misconfigured env never destabilises paging.
    pub fn paging_threshold_from_env(default: f64) -> f64 {
        std::env::var("CCOS_PAGING_THRESHOLD")
            .ok()
            .and_then(|v| v.trim().parse::<f64>().ok())
            .filter(|x| x.is_finite() && *x >= 0.0)
            .unwrap_or(default)
    }

    /// Construct with the paging knobs read from the environment: `CCOS_PAGING_THRESHOLD` (the
    /// eviction floor) and `CCOS_MAX_RESIDENT` (the resident-node cap), each falling back to the
    /// given default. Mirrors [`ScoringWeights::from_env`] so the whole frugal-window behaviour is
    /// env-tunable (the validation harness sweeps these without recompiling), default-identical.
    pub fn new_from_env(default_threshold: f64, default_max: usize) -> Self {
        let max = std::env::var("CCOS_MAX_RESIDENT")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(default_max);
        Self::new(Self::paging_threshold_from_env(default_threshold), max)
    }

    /// Replace the eviction policy consulted by [`enforce_paging`](Self::enforce_paging).
    pub fn set_eviction_policy(&mut self, policy: EvictionPolicy) {
        self.eviction_policy = policy;
    }

    /// Train the eviction policy in place from a replay of
    /// `(state, action, reward, next_state)` paging transitions.
    pub fn train_eviction_policy<I>(&mut self, transitions: I)
    where
        I: IntoIterator<Item = (PagingState, u8, f64, PagingState)>,
    {
        self.eviction_policy.fit(transitions);
    }

    /// Replace the scoring/decay coefficients (see [`ScoringWeights`]). Set these
    /// before ingesting or re-paging so the new weights drive eviction.
    pub fn set_scoring_weights(&mut self, weights: ScoringWeights) {
        self.scoring_weights = weights;
    }

    pub fn tick(&mut self) {
        self.clock += 1;
        // Apply recency decay to all nodes
        let decay = 0.95_f64;
        for node in self.nodes.values_mut() {
            node.recency *= decay;
            if node.recency < 0.01 {
                node.recency = 0.01;
            }
        }
    }

    pub fn upsert_node(&mut self, id: NodeId, label: String, content: String, node_type: NodeType) {
        let now = self.clock;
        // A fresh ingest supersedes any demoted (COLD) shadow of this node — full
        // entry or deep husk — reclaiming its on-disk blob(s) if it was the last
        // referent.
        self.forget_cold_shadow(&id);
        match self.nodes.get_mut(&id) {
            Some(existing) => {
                existing.label = label;
                existing.content = content;
                existing.node_type = node_type;
                existing.recency = 1.0;
                existing.last_accessed = now;
                existing.access_count += 1;
            }
            None => {
                let node = GraphNode {
                    id: id.clone(),
                    label,
                    content,
                    node_type,
                    base_importance: 0.5,
                    failure_relevance: 0.0,
                    recency: 1.0,
                    access_count: 1,
                    created_at: now,
                    last_accessed: now,
                    state: NodeState::Stable,
                };
                self.nodes.insert(id, node);
            }
        }
        self.enforce_paging();
    }

    pub fn add_edge(
        &mut self,
        source: NodeId,
        target: NodeId,
        weight: f64,
        edge_type: EdgeType,
    ) -> bool {
        // Refuse to create dangling edges: both endpoints must already exist as
        // nodes. This preserves the invariant `edges ⊆ nodes × nodes`, which
        // bounds the edge set and keeps the graph consistent when paging evicts
        // a node mid-construction (otherwise edges to evicted nodes accumulate
        // forever and are never reclaimed by `retain`).
        if !self.nodes.contains_key(&source) || !self.nodes.contains_key(&target) {
            return false;
        }
        let now = self.clock;
        // Avoid duplicate edges
        let already_exists = self
            .edges
            .iter()
            .any(|e| e.source == source && e.target == target && e.edge_type == edge_type);
        if already_exists {
            return false;
        }
        self.edges.push(GraphEdge {
            source,
            target,
            weight,
            edge_type,
            created_at: now,
        });
        true
    }

    pub fn remove_node(&mut self, id: &NodeId) {
        self.nodes.remove(id);
        // Explicit removal forgets the COLD shadow too — full entry or deep husk —
        // and reclaims its on-disk blob(s) if no other entry still references them.
        self.forget_cold_shadow(id);
        self.edges.retain(|e| &e.source != id && &e.target != id);
    }

    pub fn set_failure_relevance(&mut self, id: &NodeId, relevance: f64) {
        if let Some(node) = self.nodes.get_mut(id) {
            node.failure_relevance = relevance.clamp(0.0, 1.0);
            node.recency = 1.0;
            node.last_accessed = self.clock;
        }
    }

    /// Record an access to a node that is **already resident** — refresh its
    /// recency to full and bump its access count, exactly as
    /// [`page_in`](Self::page_in) does when it faults a *cold* node back. This
    /// models a **resident hit**: the agent re-reads a node already in working
    /// memory, so a frequently-used resident node ages like a paged-in one
    /// rather than only ever decaying via [`tick`](Self::tick). Returns `true`
    /// if the node was resident (and thus touched); `false` if it is cold or
    /// absent — `touch` never resurrects a demoted node (that is
    /// [`page_in`](Self::page_in)'s job), so the COLD tier and the
    /// `replay == live` path are untouched. Deterministic: `&mut self` only
    /// updates the node's recency/access bookkeeping.
    pub fn touch(&mut self, id: &NodeId) -> bool {
        if let Some(node) = self.nodes.get_mut(id) {
            node.recency = 1.0;
            node.last_accessed = self.clock;
            node.access_count = node.access_count.saturating_add(1);
            true
        } else {
            false
        }
    }

    /// Set a node's lifecycle [`NodeState`]. State affects the structural centrality (an
    /// `Orphan` is excluded) and the eviction score (`Orphan` first, `Working` pinned), so
    /// this invalidates the centrality caches — which key only on edge/node counts — when the
    /// state actually changes. No-op if the node is absent. Deterministic.
    pub fn set_node_state(&mut self, id: &NodeId, state: NodeState) {
        let changed = match self.nodes.get_mut(id) {
            Some(node) => {
                let was = node.state;
                node.state = state;
                was != state
            }
            None => false,
        };
        if changed {
            self.indegree_cache.borrow_mut().take();
            self.eigencentrality_cache.borrow_mut().take();
        }
    }

    /// Whether `id` is a resident `Orphan` (excluded from the structural centrality signal).
    fn is_orphan(&self, id: &NodeId) -> bool {
        self.nodes
            .get(id)
            .is_some_and(|n| n.state == NodeState::Orphan)
    }

    pub fn compute_node_score(&self, node: &GraphNode) -> f64 {
        let w = &self.scoring_weights;
        let base = node.base_importance * w.w_base;
        let failure = node.failure_relevance * w.w_failure;
        let recency = node.recency * w.w_recency;
        let access = (node.access_count.max(1) as f64).ln() * w.w_access;
        // Structural-centrality term. Off by default (`w_centrality == 0`), in which case
        // this is byte-identical to the previous score and neither the in-degree map nor
        // the eigenvector vector is ever built. When on, the mode picks the signal:
        // local `ln(1 + in_degree)` (default) or global recursive eigenvector centrality.
        let centrality = if w.w_centrality != 0.0 {
            let signal = match w.centrality_mode {
                CentralityMode::InDegree => (1.0 + self.node_in_degree(&node.id) as f64).ln(),
                CentralityMode::Eigenvector => self.node_eigencentrality(&node.id),
            };
            signal * w.w_centrality
        } else {
            0.0
        };
        // Lifecycle bias (orthogonal to topology): `Working` is **pinned** (kept resident as
        // the current focus even as recency decays); `Orphan` is driven to the bottom so it is
        // **evicted first** regardless of recency. `Stable` (the default) is neutral ⇒ the
        // score is byte-identical to before this field existed.
        let state_bias = match node.state {
            NodeState::Stable => 0.0,
            NodeState::Working => 0.3,
            NodeState::Orphan => -1.0,
        };
        (base + failure + recency + access + centrality + state_bias).clamp(0.0, 1.0)
    }

    /// In-degree of `id` among the **resident** graph — the count of incoming
    /// causal edges whose target is `id`. Only edges between two resident nodes
    /// are in `self.edges` (paging archives incident edges on demote), so this is
    /// the *resident* structural signal the centrality term ([`compute_node_score`](Self::compute_node_score))
    /// scores on, not the global in-degree. Cached on `edges.len()`; deterministic.
    pub fn node_in_degree(&self, id: &NodeId) -> u32 {
        let mut cache = self.indegree_cache.borrow_mut();
        if cache.as_ref().map(|(v, _)| *v) != Some(self.edges.len()) {
            let mut m: HashMap<NodeId, u32> = HashMap::new();
            for e in &self.edges {
                // Edges incident to an Orphan don't count toward the load-bearing structure:
                // dead code depending on a node should not inflate its centrality.
                if self.is_orphan(&e.source) || self.is_orphan(&e.target) {
                    continue;
                }
                *m.entry(e.target.clone()).or_default() += 1;
            }
            *cache = Some((self.edges.len(), m));
        }
        cache.as_ref().unwrap().1.get(id).copied().unwrap_or(0)
    }

    /// **Eigenvector centrality** of the resident graph, by deterministic damped power
    /// iteration. Importance flows along causal edges into their target, so a node is
    /// central when *central* nodes depend on it — distinguishing a hub that other hubs
    /// rely on from a hub that only leaves rely on, which [`node_in_degree`](Self::node_in_degree)
    /// cannot. Returns a `NodeId → score` map normalized to `[0, 1]` (most-central = 1.0).
    ///
    /// **Why damped.** Pure eigenvector centrality solves `A x = λ x` for the principal
    /// eigenvalue, but a code dependency graph is largely a **DAG** (imports / contains
    /// flow one way); its adjacency is nilpotent (`λ_max = 0`), so the pure vector
    /// collapses onto the sinks and is ill-defined. The damped iteration
    /// `x ← (1−d)/N + d·(Aᵀ x with out-degree split)` (the Katz / PageRank form) is the
    /// eigenvector of the *damped* operator and stays well-defined and strictly positive
    /// on any directed graph — the standard, correct realization on real code.
    ///
    /// **Deterministic:** a fixed iteration count accumulating over **sorted** node ids,
    /// so it is a pure function of the graph (no `HashMap`-order or float-summation-order
    /// dependence). Like all scoring it is a read-only ranking signal and never enters the
    /// snapshot/replay hash, so its `f64`s are safe.
    pub fn eigencentrality(&self) -> HashMap<NodeId, f64> {
        // Stable ordering ⇒ deterministic accumulation order. `Orphan` nodes are excluded from
        // the structural graph (and so get centrality 0), and their edges drop out below because
        // their endpoints are absent from `index`. `n` is therefore the *non-Orphan* count.
        let mut ids: Vec<&NodeId> = self
            .nodes
            .iter()
            .filter(|(_, node)| node.state != NodeState::Orphan)
            .map(|(id, _)| id)
            .collect();
        ids.sort();
        let n = ids.len();
        if n == 0 {
            return HashMap::new();
        }
        let index: HashMap<&NodeId, usize> =
            ids.iter().enumerate().map(|(i, id)| (*id, i)).collect();
        // Resident edges as (source_idx, target_idx); out-degree splits each node's mass
        // over its out-edges (PageRank) so one over-connected node cannot flood the rest.
        let mut outdeg = vec![0u32; n];
        let mut edges: Vec<(usize, usize)> = Vec::with_capacity(self.edges.len());
        for e in &self.edges {
            if let (Some(&s), Some(&t)) = (index.get(&e.source), index.get(&e.target)) {
                edges.push((s, t));
                outdeg[s] += 1;
            }
        }
        // Canonical edge order so the floating-point accumulation below is invariant to how
        // the edges were *inserted* (import resolution adds them in HashMap order). With the
        // sorted ids above, this makes eigencentrality a pure function of the graph's
        // *structure*, not its construction order — deterministic across processes.
        edges.sort_unstable();
        const ITERS: usize = 64;
        const DAMP: f64 = 0.85;
        let base = (1.0 - DAMP) / n as f64;
        let mut x = vec![1.0 / n as f64; n];
        for _ in 0..ITERS {
            let mut next = vec![base; n];
            for &(s, t) in &edges {
                // `s` is a source ⇒ outdeg[s] ≥ 1, so this never divides by zero.
                next[t] += DAMP * x[s] / outdeg[s] as f64;
            }
            x = next;
        }
        // Normalize to [0,1] by the max so the term sits on the same scale as the rest of
        // the (clamped [0,1]) score. No edges anywhere ⇒ all equal ⇒ all 1.0.
        let max = x.iter().copied().fold(0.0_f64, f64::max);
        let inv = if max > 0.0 { 1.0 / max } else { 1.0 };
        ids.into_iter()
            .enumerate()
            .map(|(i, id)| (id.clone(), x[i] * inv))
            .collect()
    }

    /// Cached per-node eigenvector centrality for the scoring hot path (mirrors
    /// [`node_in_degree`](Self::node_in_degree)): the whole vector is recomputed only when
    /// `(nodes.len(), edges.len())` changes, then looked up per node.
    fn node_eigencentrality(&self, id: &NodeId) -> f64 {
        let key = (self.nodes.len(), self.edges.len());
        let mut cache = self.eigencentrality_cache.borrow_mut();
        if cache.as_ref().map(|(k, _)| *k) != Some(key) {
            *cache = Some((key, self.eigencentrality()));
        }
        cache.as_ref().unwrap().1.get(id).copied().unwrap_or(0.0)
    }

    pub fn select_context_window(&self, max_tokens: usize) -> Vec<&GraphNode> {
        let estimated_tokens_per_node = 128;
        let max_nodes = (max_tokens / estimated_tokens_per_node).max(1);

        struct ScoredRef<'a> {
            node: &'a GraphNode,
            score: f64,
        }
        impl<'a> PartialEq for ScoredRef<'a> {
            fn eq(&self, other: &Self) -> bool {
                self.node == other.node
            }
        }
        impl<'a> Eq for ScoredRef<'a> {}
        impl<'a> PartialOrd for ScoredRef<'a> {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }
        impl<'a> Ord for ScoredRef<'a> {
            fn cmp(&self, other: &Self) -> Ordering {
                // Order by score, breaking ties on node id so the heap pops a
                // deterministic node when scores are equal.
                self.score
                    .partial_cmp(&other.score)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| self.node.id.cmp(&other.node.id))
            }
        }

        let mut heap: BinaryHeap<ScoredRef> = BinaryHeap::new();
        for node in self.nodes.values() {
            let score = self.compute_node_score(node);
            heap.push(ScoredRef { node, score });
        }

        let mut selected: Vec<&GraphNode> = Vec::new();
        while let Some(scored) = heap.pop() {
            if selected.len() >= max_nodes {
                break;
            }
            selected.push(scored.node);
        }
        selected
    }

    pub fn enforce_paging(&mut self) {
        if self.nodes.len() <= self.max_in_memory_nodes {
            return;
        }
        let total = self.nodes.len();
        // Recency rank (0 = most recently accessed), for the eviction-policy state.
        let mut by_recency: Vec<(NodeId, f64)> = self
            .nodes
            .iter()
            .map(|(id, n)| (id.clone(), n.recency))
            .collect();
        by_recency.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        let recency_rank: HashMap<NodeId, usize> = by_recency
            .into_iter()
            .enumerate()
            .map(|(i, (id, _))| (id, i))
            .collect();

        let trained = self.eviction_policy.is_trained();
        let to_remove: Vec<NodeId> = {
            // Eviction priority = base causal score, nudged by the learned
            // policy's keep−evict preference. When the policy is untrained the
            // nudge is exactly 0, so this is byte-identical to the deterministic
            // greedy (lowest score first, ties broken by node id) and replay /
            // snapshot hashes are unchanged.
            let mut entries: Vec<(&NodeId, f64)> = self
                .nodes
                .iter()
                .map(|(id, node)| {
                    let base = self.compute_node_score(node);
                    let bias = if trained {
                        let state = PagingState {
                            score: bucket_score(base),
                            recency: bucket_recency(recency_rank[id], total),
                            pressure: bucket_pressure(node.failure_relevance),
                            size: bucket_size((node.content.len() + node.label.len()) / 4),
                        };
                        self.eviction_policy.q_value(state, KEEP)
                            - self.eviction_policy.q_value(state, EVICT)
                    } else {
                        0.0
                    };
                    (id, base + bias)
                })
                .collect();
            entries.sort_by(|a, b| {
                a.1.partial_cmp(&b.1)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| a.0.cmp(b.0))
            });
            let remove_count = self.nodes.len() - self.max_in_memory_nodes;
            entries
                .iter()
                .take(remove_count)
                .map(|(id, _)| (*id).clone())
                .collect()
        };
        for id in &to_remove {
            self.demote(id);
        }
        // Defensive: guarantee no edge survives pointing at an evicted node.
        self.prune_dangling_edges();
        // Newly-demoted content may push the COLD tier over its budgets: compact
        // the coldest tail first (shrink), then spill what remains (move to disk).
        self.enforce_cold_content_budget();
        self.enforce_cold_budget();
        self.enforce_cold_resident_budget();
    }

    /// Demote a node out of the resident graph into the COLD tier, archiving the
    /// edges incident to it so [`page_in`](Self::page_in) can later restore its
    /// causal structure. Non-destructive: the node and its links are kept, just
    /// no longer resident.
    fn demote(&mut self, id: &NodeId) {
        if let Some(node) = self.nodes.remove(id) {
            let incident: Vec<GraphEdge> = self
                .edges
                .iter()
                .filter(|e| &e.source == id || &e.target == id)
                .cloned()
                .collect();
            self.edges.retain(|e| &e.source != id && &e.target != id);
            // The other endpoint of each archived edge — recorded in the reverse index
            // (brick 8) so cold_neighbours is O(degree), not an O(N) tier scan.
            let others: Vec<NodeId> = incident
                .iter()
                .map(|e| {
                    if &e.source == id {
                        e.target.clone()
                    } else {
                        e.source.clone()
                    }
                })
                .collect();
            self.cold.insert(
                id.clone(),
                ColdNode {
                    node,
                    edges: incident,
                    spill: None,
                    compacted: false,
                    at_floor: false,
                },
            );
            for other in others {
                self.radj_add_edge(id, &other);
            }
        }
    }

    /// Restore a node from the **COLD tier** into the resident graph — a page-in
    /// (a swap). The node is marked freshly accessed; if the resident set is at
    /// capacity, the lowest-scored **other** node is demoted to make room — the
    /// just-requested node is never the one bounced back out. Any archived edge
    /// whose other endpoint is resident is re-linked. Returns `true` if the node
    /// was cold and is now resident. Deterministic (tie-break on id).
    pub fn page_in(&mut self, id: &NodeId) -> bool {
        // A deep-spilled entry lives as a compact husk in `cold_deep`: fault its
        // whole serialized node back (hash-verified) into the full COLD map first,
        // then the normal page-in path below runs unchanged. A missing / tampered /
        // undeserializable body is a cold-miss — never a silent half-restore.
        if let Some(husk) = self.deep_get(id) {
            let body_hash = husk.body.hash;
            match self
                .spill
                .as_ref()
                .and_then(|cfg| cfg.store.get(&body_hash))
                .and_then(|s| serde_json::from_str::<ColdNode>(&s).ok())
            {
                Some(node) => {
                    if let Some(hs) = self.husk_store.as_mut() {
                        let _ = hs.delete(&id.0); // the husk leaves the on-disk tier
                    }
                    self.cold.insert(id.clone(), node);
                    // The husk's body blob is now unreferenced — the node is back with
                    // its content folded inline — so reclaim it unless another husk
                    // still shares it. (Without this, page-in orphans the blob: a slow
                    // disk leak no later `remove` can find.)
                    self.release_blob_if_orphan(&body_hash);
                }
                None => return false,
            }
        }
        // Fault spilled content back from disk (verified by hash). A spilled entry
        // whose blob is missing/tampered, or whose store has been detached, is a
        // cold-miss — never a silent empty restore. (A just-restored deep husk has
        // its content folded inline, so this is a no-op for it.)
        let spilled_hash = self
            .cold
            .get(id)
            .and_then(|c| c.spill.as_ref().map(|s| s.hash));
        if let Some(hash) = spilled_hash {
            match self.spill.as_ref().and_then(|cfg| cfg.store.get(&hash)) {
                Some(text) => {
                    if let Some(entry) = self.cold.get_mut(id) {
                        entry.node.content = text;
                        entry.spill = None;
                    }
                    // The content blob is now unreferenced by this entry; reclaim it
                    // unless another cold entry shares it (content blob ⇒ cheap variant).
                    self.release_content_blob_if_orphan(&hash);
                }
                None => return false,
            }
        }
        let Some(mut cold) = self.cold.remove(id) else {
            return false;
        };
        // The node leaves the cold tier, so drop its reverse-adjacency entries.
        let others: Vec<NodeId> = cold
            .edges
            .iter()
            .map(|e| {
                if &e.source == id {
                    e.target.clone()
                } else {
                    e.source.clone()
                }
            })
            .collect();
        cold.node.recency = 1.0;
        cold.node.last_accessed = self.clock;
        cold.node.access_count = cold.node.access_count.saturating_add(1);
        self.nodes.insert(id.clone(), cold.node);
        for e in cold.edges {
            if self.nodes.contains_key(&e.source)
                && self.nodes.contains_key(&e.target)
                && !self.edges.iter().any(|x| {
                    x.source == e.source && x.target == e.target && x.edge_type == e.edge_type
                })
            {
                self.edges.push(e);
            }
        }
        for other in others {
            self.radj_del_edge(id, &other);
        }
        // Swap: while over capacity, demote the lowest-scored node *other* than
        // the one just paged in (deterministic tie-break on id).
        while self.nodes.len() > self.max_in_memory_nodes {
            let victim = self
                .nodes
                .iter()
                .filter(|(nid, _)| *nid != id)
                .min_by(|(aid, an), (bid, bn)| {
                    self.compute_node_score(an)
                        .partial_cmp(&self.compute_node_score(bn))
                        .unwrap_or(Ordering::Equal)
                        .then_with(|| aid.cmp(bid))
                })
                .map(|(nid, _)| nid.clone());
            match victim {
                Some(v) => self.demote(&v),
                None => break, // only `id` is resident and still over cap — keep it
            }
        }
        self.prune_dangling_edges();
        // Swap-demoted victims add content to the COLD tier; re-check both budgets
        // (compact the coldest tail, then spill what remains).
        self.enforce_cold_content_budget();
        self.enforce_cold_budget();
        self.enforce_cold_resident_budget();
        true
    }

    /// Number of nodes in the COLD tier (demoted but retrievable via [`page_in`](Self::page_in)) —
    /// full entries plus deep-spilled husks.
    pub fn cold_count(&self) -> usize {
        self.cold.len() + self.deep_count()
    }

    /// Whether `id` is currently demoted to the COLD tier (full entry or deep husk).
    pub fn is_cold(&self, id: &NodeId) -> bool {
        self.cold.contains_key(id) || self.deep_contains(id)
    }

    /// Live deep-spilled ids (keys only, no husk deserialization).
    fn deep_live_keys(&self) -> Vec<NodeId> {
        self.husk_store.as_ref().map_or(Vec::new(), |hs| {
            hs.live_entries()
                .unwrap_or_default()
                .into_iter()
                .map(|(k, _)| NodeId(k))
                .collect()
        })
    }

    /// The ids currently in the COLD tier, in sorted (deterministic) order — both
    /// full entries and deep-spilled husks (the latter from the on-disk index).
    /// Yields owned ids (the deep ones are reconstructed from the index, not borrowed).
    pub fn cold_ids(&self) -> impl Iterator<Item = NodeId> + '_ {
        let mut ids: Vec<NodeId> = self
            .cold
            .keys()
            .cloned()
            .chain(self.deep_live_keys())
            .collect();
        ids.sort();
        ids.dedup();
        ids.into_iter()
    }

    /// The **cold** neighbours of `id` — the other endpoints of any COLD-archived
    /// edge incident to `id` that are themselves cold, i.e. the rest of `id`'s causal
    /// region that would page in alongside it. Sorted (deterministic).
    ///
    /// With the reverse-adjacency index attached (Lever 2 brick 8), a single keyed
    /// prefix scan gives every cold edge incident to `id` in both directions —
    /// `O(degree)`, no deep-tier scan. Without a spill store there is no deep tier, so
    /// it falls back to reading the resident cold-full entries' edges directly.
    pub fn cold_neighbours(&self, id: &NodeId) -> Vec<NodeId> {
        let mut out: BTreeSet<NodeId> = BTreeSet::new();
        if let Some(radj) = self.radj.as_ref() {
            let prefix = format!("{}\u{1f}", id.0);
            for (k, _) in radj.scan_prefix(&prefix).unwrap_or_default() {
                let other = NodeId(k[prefix.len()..].to_owned());
                // Keep only neighbours that are themselves still cold.
                if other != *id && (self.cold.contains_key(&other) || self.deep_contains(&other)) {
                    out.insert(other);
                }
            }
            return out.into_iter().collect();
        }
        // No spill store ⇒ no deep tier; read the resident cold-full edges straight.
        for c in self.cold.values() {
            for e in &c.edges {
                let other = if &e.source == id {
                    &e.target
                } else if &e.target == id {
                    &e.source
                } else {
                    continue;
                };
                if other != id && self.cold.contains_key(other) {
                    out.insert(other.clone());
                }
            }
        }
        out.into_iter().collect()
    }

    /// Attach an on-disk **spill store** (the "swap file") for COLD content,
    /// flushing the coldest content to `dir` whenever resident COLD content
    /// exceeds `inline_budget` bytes. Content is addressed by SHA-256 (so a
    /// blob is deduplicated and integrity-checked on read) and faulted back by
    /// [`page_in`](Self::page_in) on demand. Re-attaching the *same* directory
    /// after a snapshot restore lets previously-spilled stubs fault back in.
    /// Applies the budget immediately. Errors only if `dir` can't be created.
    pub fn attach_cold_spill(
        &mut self,
        dir: impl Into<PathBuf>,
        inline_budget: usize,
    ) -> std::io::Result<()> {
        let dir = dir.into();
        // Lever 2 husk index, in a *sibling* `<dir>.husks` directory — kept out of the
        // spill (blob) directory so that stays purely the content-addressed blob store
        // (the "no orphaned blobs" invariant counts only blobs there).
        let mut husk_dir = dir.clone().into_os_string();
        husk_dir.push(".husks");
        let husk_store =
            HuskStore::open(PathBuf::from(husk_dir), HUSK_BUFFER_LIMIT, HUSK_CACHE_CAP)?;
        // Reverse-adjacency index (brick 8), a second sibling directory.
        let mut radj_dir = dir.clone().into_os_string();
        radj_dir.push(".radj");
        let radj = HuskStore::open(PathBuf::from(radj_dir), HUSK_BUFFER_LIMIT, HUSK_CACHE_CAP)?;
        let store = ColdSpill::new(dir)?;
        self.spill = Some(SpillConfig {
            store,
            inline_budget,
        });
        self.husk_store = Some(husk_store);
        self.radj = Some(radj);
        self.backfill_radj();
        self.enforce_cold_budget();
        self.enforce_cold_resident_budget();
        Ok(())
    }

    /// Seed the reverse-adjacency index from the COLD tier already present at attach
    /// time (entries demoted before the index existed, or restored from a snapshot),
    /// so [`cold_neighbours`](Self::cold_neighbours) is correct immediately.
    fn backfill_radj(&mut self) {
        let mut edges: Vec<(NodeId, NodeId)> = Vec::new();
        for (id, c) in &self.cold {
            for e in &c.edges {
                let other = if &e.source == id {
                    &e.target
                } else {
                    &e.source
                };
                edges.push((id.clone(), other.clone()));
            }
        }
        for (id, h) in self.deep_entries() {
            for o in unpack_adj(&h.adj) {
                edges.push((id.clone(), NodeId(o.to_owned())));
            }
        }
        for (a, b) in edges {
            self.radj_add_edge(&a, &b);
        }
    }

    /// Detach the spill store. Already-spilled entries stay stubbed and become
    /// unreachable (a cold-miss on [`page_in`](Self::page_in)) until the same
    /// directory is re-attached. Mainly for tests and controlled teardown.
    pub fn detach_cold_spill(&mut self) {
        self.spill = None;
        self.husk_store = None;
        self.radj = None;
    }

    /// Flush the COLD tier's on-disk indices (husk store + reverse adjacency) so their
    /// in-RAM write buffers reach durable segments. Call before a checkpoint so the
    /// `<dir>.husks` / `<dir>.radj` directories are consistent with the snapshot and a
    /// crash loses nothing committed since the last checkpoint. Spill *blobs* are
    /// already written durably on each put (and segments via `write_durable`), so the
    /// only volatile cold-tier state is these two memtables. A no-op without a store.
    ///
    /// Crash model: the **event log** is the source of truth — `replay == live` is
    /// reconstructed from it regardless of the cold tier, which is a rebuildable cache.
    /// This flush makes the cache itself recover to its last checkpoint instead of its
    /// last segment flush.
    pub fn flush_cold_tier(&mut self) -> std::io::Result<()> {
        if let Some(hs) = self.husk_store.as_mut() {
            hs.flush()?;
        }
        if let Some(radj) = self.radj.as_mut() {
            radj.flush()?;
        }
        Ok(())
    }

    /// Record an archived cold edge `a─b` in the reverse-adjacency index (both
    /// directions). A no-op without an attached index. Best-effort: a write miss only
    /// degrades `cold_neighbours` to (still-correct) incompleteness, never corrupts.
    fn radj_add_edge(&mut self, a: &NodeId, b: &NodeId) {
        if let Some(radj) = self.radj.as_mut() {
            let _ = radj.put(&format!("{}\u{1f}{}", a.0, b.0), Vec::new());
            let _ = radj.put(&format!("{}\u{1f}{}", b.0, a.0), Vec::new());
        }
    }

    /// Drop the reverse-adjacency entries for edge `a─b` (both directions).
    fn radj_del_edge(&mut self, a: &NodeId, b: &NodeId) {
        if let Some(radj) = self.radj.as_mut() {
            let _ = radj.delete(&format!("{}\u{1f}{}", a.0, b.0));
            let _ = radj.delete(&format!("{}\u{1f}{}", b.0, a.0));
        }
    }

    /// Whether an on-disk COLD spill store is currently attached.
    pub fn has_cold_spill(&self) -> bool {
        self.spill.is_some()
    }

    /// Bytes of COLD content currently held **inline** (resident in RAM, not yet
    /// spilled). This is the quantity the spill budget bounds.
    pub fn cold_inline_bytes(&self) -> usize {
        self.cold
            .values()
            .filter(|c| c.spill.is_none())
            .map(|c| c.node.content.len())
            .sum()
    }

    /// Number of COLD entries whose content has been spilled to disk.
    pub fn cold_spilled_count(&self) -> usize {
        self.cold.values().filter(|c| c.spill.is_some()).count()
    }

    /// Whether `id` is a COLD entry whose content currently lives on disk (spilled).
    pub fn is_spilled(&self, id: &NodeId) -> bool {
        self.cold.get(id).is_some_and(|c| c.spill.is_some())
    }

    /// Bytes of COLD content spilled to disk (sum of original content lengths;
    /// the on-disk store deduplicates identical blobs, so the actual disk
    /// footprint is ≤ this).
    pub fn cold_spilled_bytes(&self) -> usize {
        self.cold
            .values()
            .filter_map(|c| c.spill.as_ref().map(|s| s.len))
            .sum()
    }

    /// Estimated **resident** bytes the COLD tier still holds in RAM — the part
    /// that does *not* go to disk even when content is spilled: the `BTreeMap` key,
    /// the node's id/label (and any inline content), the archived edges, and the
    /// spill-hash stub. This is the O(N) footprint slice 5 bounds; a logical
    /// estimate (string lengths + struct sizes, ignoring allocator slack), honest
    /// for "how much RAM is stuck per cold entry". A **deep-spilled** entry is just
    /// a compact `DeepHusk` — body-blob hash + neighbour ids — so it contributes a
    /// small fraction of a full entry (the whole `ColdNode` struct is gone).
    pub fn cold_resident_bytes(&self) -> usize {
        let full: usize = self
            .cold
            .iter()
            .map(|(k, c)| Self::entry_resident_bytes(k, c))
            .sum();
        // Deep-spilled husks live in the on-disk index now; their resident cost is the
        // store's bounded footprint (memtable + sparse indices + cache), not one entry
        // per husk — the whole point of Lever 2.
        let deep = self
            .husk_store
            .as_ref()
            .map_or(0, HuskStore::resident_bytes);
        full + deep
    }

    /// Per-entry resident-byte estimate for a full `ColdNode`, shared by
    /// [`cold_resident_bytes`](Self::cold_resident_bytes) and the deep-spill enforcer
    /// (so the budget loop accounts exactly as the stat reports).
    fn entry_resident_bytes(key: &NodeId, c: &ColdNode) -> usize {
        // The spill stub's hash is a raw `[u8; 32]` inline in `size_of::<ColdNode>()`
        // (no heap), so it needs no separate term — only the variable-length strings
        // and archived edges do.
        let mut b = std::mem::size_of::<ColdNode>() + std::mem::size_of::<NodeId>();
        b += key.0.len() + c.node.id.0.len() + c.node.label.len() + c.node.content.len();
        for e in &c.edges {
            b += std::mem::size_of::<GraphEdge>() + e.source.0.len() + e.target.0.len();
        }
        b
    }

    /// Flush the coldest resident COLD content to the spill store until resident
    /// COLD content is within `inline_budget`. Deterministic: candidates are
    /// ordered coldest-first by causal score, ties broken on node id. A no-op
    /// without an attached store, or when already within budget. A blob that
    /// fails to write is left **inline** (kept in RAM) — spill never drops data.
    fn enforce_cold_budget(&mut self) {
        let budget = match self.spill.as_ref() {
            Some(cfg) => cfg.inline_budget,
            None => return,
        };
        let mut resident = self.cold_inline_bytes();
        if resident <= budget {
            return;
        }
        // Coldest-first candidate order (deterministic); scores computed up front
        // so the mutation loop borrows neither `self.scoring_weights` nor the map.
        let mut candidates: Vec<(NodeId, f64)> = self
            .cold
            .iter()
            .filter(|(_, c)| c.spill.is_none() && !c.node.content.is_empty())
            .map(|(id, c)| (id.clone(), self.compute_node_score(&c.node)))
            .collect();
        candidates.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        for (id, _) in candidates {
            if resident <= budget {
                break;
            }
            let content = match self.cold.get(&id) {
                Some(c) if c.spill.is_none() => c.node.content.clone(),
                _ => continue,
            };
            let written = self
                .spill
                .as_ref()
                .and_then(|cfg| cfg.store.put(&content).ok());
            if let Some(hash) = written {
                if let Some(entry) = self.cold.get_mut(&id) {
                    let len = entry.node.content.len();
                    entry.node.content = String::new();
                    entry.spill = Some(SpillRef { hash, len });
                    resident = resident.saturating_sub(len);
                }
            }
        }
    }

    /// Set the COLD **resident-metadata budget** (slice 5) — the most aggressive
    /// tier. With `Some(b)`, the bytes the COLD tier keeps in RAM
    /// ([`cold_resident_bytes`](Self::cold_resident_bytes)) are driven toward `b` by
    /// **deep-spilling** the coldest entries (label + full edges → one on-disk blob,
    /// only neighbour ids kept resident). Still **lossless** — the body faults back
    /// on [`page_in`](Self::page_in). Applies immediately. `None` (default) disables
    /// it. Needs an attached spill store; a no-op without one. Like spill/compaction
    /// it is a runtime mode layered on the deterministic default path, not a change
    /// to it (replay and the default snapshot stay byte-identical).
    pub fn set_cold_resident_budget(&mut self, budget: Option<usize>) {
        self.cold_resident_budget = budget;
        self.enforce_cold_resident_budget();
    }

    /// Read a deep-spilled husk from the on-disk [`HuskStore`] (Lever 2). `None` if no
    /// store is attached, the key is absent, or the bytes don't deserialize.
    fn deep_get(&self, id: &NodeId) -> Option<DeepHusk> {
        let bytes = self.husk_store.as_ref()?.get(&id.0).ok().flatten()?;
        serde_json::from_slice::<DeepHusk>(&bytes).ok()
    }

    /// Whether `id` has a deep-spilled husk in the on-disk store.
    fn deep_contains(&self, id: &NodeId) -> bool {
        self.husk_store
            .as_ref()
            .and_then(|hs| hs.get(&id.0).ok().flatten())
            .is_some()
    }

    /// Every live deep-spilled `(id, husk)` — a full store scan. Lever 2 brick 6:
    /// until a keyed adjacency index lands, cold-neighbour and GC sweeps enumerate
    /// the whole deep tier this way.
    fn deep_entries(&self) -> Vec<(NodeId, DeepHusk)> {
        let Some(hs) = self.husk_store.as_ref() else {
            return Vec::new();
        };
        hs.live_entries()
            .unwrap_or_default()
            .into_iter()
            .filter_map(|(k, b)| {
                serde_json::from_slice::<DeepHusk>(&b)
                    .ok()
                    .map(|h| (NodeId(k), h))
            })
            .collect()
    }

    /// Number of live deep-spilled husks.
    fn deep_count(&self) -> usize {
        self.husk_store
            .as_ref()
            .map_or(0, |hs| hs.live_entries().map_or(0, |v| v.len()))
    }

    /// Number of COLD entries deep-spilled to disk (archived whole, represented in
    /// RAM only by a compact `DeepHusk` in the on-disk index).
    pub fn cold_deep_spilled_count(&self) -> usize {
        self.deep_count()
    }

    /// Whether `id` is a deep-spilled COLD entry (its whole node on disk, only a
    /// compact husk — body-blob stub + neighbour ids — in the index).
    pub fn is_deep_spilled(&self, id: &NodeId) -> bool {
        self.deep_contains(id)
    }

    /// Deep-spill the coldest COLD entries until resident COLD metadata
    /// ([`cold_resident_bytes`](Self::cold_resident_bytes)) is within
    /// [`cold_resident_budget`](Self::cold_resident_budget). Each chosen entry is
    /// serialized **whole** (node + content folded inline + edges) into one
    /// content-addressed blob and replaced in RAM by a compact `DeepHusk` (the
    /// body-blob stub + the neighbour ids) moved into [`cold_deep`](Self::cold_deep) —
    /// shedding the full `ColdNode` struct, which is the per-entry resident floor.
    /// Deterministic: coldest-first by causal score, ties on id. **Lossless**: the
    /// whole node faults back in [`page_in`](Self::page_in). A no-op without an
    /// attached store or a budget, or when already within it. An entry whose blob
    /// fails to write (or whose spilled content can't be faulted to fold in) is left
    /// intact — deep-spill never drops data; the budget is approached best-effort,
    /// never by dropping a node and never below the compact-husk floor.
    fn enforce_cold_resident_budget(&mut self) {
        let Some(budget) = self.cold_resident_budget else {
            return;
        };
        if self.spill.is_none() {
            return;
        }
        let mut resident = self.cold_resident_bytes();
        if resident <= budget {
            return;
        }
        // Coldest-first candidates from the full COLD map (deep husks already live in
        // `cold_deep` and are terminal). Scored up front so the mutation loop borrows
        // neither the weights nor the map.
        let mut candidates: Vec<(NodeId, f64)> = self
            .cold
            .iter()
            .map(|(id, c)| (id.clone(), self.compute_node_score(&c.node)))
            .collect();
        candidates.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        for (id, _) in candidates {
            if resident <= budget {
                break;
            }
            let Some(c) = self.cold.get(&id) else {
                continue;
            };
            let before = Self::entry_resident_bytes(&id, c);
            // Neighbour ids = the other endpoint of each archived edge (each is
            // incident to this node), sorted + deduped, self-loops dropped.
            let mut adj: Vec<NodeId> = c
                .edges
                .iter()
                .map(|e| {
                    if e.source == c.node.id {
                        e.target.clone()
                    } else {
                        e.source.clone()
                    }
                })
                .filter(|n| *n != c.node.id)
                .collect();
            adj.sort();
            adj.dedup();
            let adj_bytes: usize = adj
                .iter()
                .map(|n| std::mem::size_of::<NodeId>() + n.0.len())
                .sum();
            // The full `ColdNode` struct is replaced by a compact husk (body stub +
            // ids), so this almost always shrinks; the guard still parks the rare
            // entry that wouldn't — the floor, never a drop (mirrors compaction). The
            // body hash is inline in `size_of::<DeepHusk>()` (a raw `[u8; 32]`).
            let projected =
                std::mem::size_of::<DeepHusk>() + std::mem::size_of::<NodeId>() + adj_bytes;
            if projected >= before {
                continue;
            }
            // Fault any spilled content back inline so the body blob is
            // self-contained (one blob per husk → simple GC); its old content blob is
            // then orphaned and reclaimed below.
            let mut entry = c.clone();
            let old_content_hash = entry.spill.as_ref().map(|s| s.hash);
            if let Some(hash) = &old_content_hash {
                match self.spill.as_ref().and_then(|cfg| cfg.store.get(hash)) {
                    Some(text) => {
                        entry.node.content = text;
                        entry.spill = None;
                    }
                    None => continue, // content blob missing — leave the entry intact
                }
            }
            // Serialize the whole content-inline node as the body blob.
            let Ok(serialized) = serde_json::to_string(&entry) else {
                continue;
            };
            let Some(cfg) = self.spill.as_ref() else {
                return;
            };
            let Some(body_hash) = cfg.store.put(&serialized).ok() else {
                continue;
            };
            // Commit: archive the whole entry as a husk in the on-disk index, then
            // drop the resident full entry. The index is authoritative now (Lever 2),
            // so if the store write fails we leave the entry full rather than lose it.
            let husk = DeepHusk {
                body: SpillRef {
                    hash: body_hash,
                    len: serialized.len(),
                },
                adj: pack_adj(&adj),
            };
            let Ok(bytes) = serde_json::to_vec(&husk) else {
                continue;
            };
            let stored = self
                .husk_store
                .as_mut()
                .is_some_and(|hs| hs.put(&id.0, bytes).is_ok());
            if !stored {
                continue;
            }
            self.cold.remove(&id);
            if let Some(hash) = old_content_hash {
                self.release_content_blob_if_orphan(&hash);
            }
            // The full entry's resident cost is freed; the husk's is the store's
            // bounded footprint (counted in `cold_resident_bytes`), not per-entry.
            resident = resident.saturating_sub(before);
        }
    }

    /// Set the COLD **compaction budget** (slice 4) — the deepest tier. With
    /// `Some(bytes)`, total COLD *content* (inline + spilled) is driven toward
    /// `bytes` by **lossily compacting** the coldest entries: code is
    /// skeletonised, prose summarised, the full original discarded — so the
    /// backing store itself stays frugal. Applies immediately. `None` (default)
    /// disables compaction (COLD stays lossless). Compaction is **lossy** and
    /// observable ([`is_compacted`](Self::is_compacted)); it is *not* part of
    /// replay, so — like the spill store — it is an operational mode layered on
    /// the deterministic default path, not a change to it.
    pub fn set_cold_content_budget(&mut self, budget: Option<usize>) {
        self.cold_content_budget = budget;
        self.enforce_cold_content_budget();
    }

    /// Number of COLD entries whose content has been lossily compacted.
    pub fn cold_compacted_count(&self) -> usize {
        self.cold.values().filter(|c| c.compacted).count()
    }

    /// Whether `id` is a COLD entry whose content is a lossy compaction (a
    /// summary/skeleton), i.e. its full original has been discarded.
    pub fn is_compacted(&self, id: &NodeId) -> bool {
        self.cold.get(id).is_some_and(|c| c.compacted)
    }

    /// Whether `id` is a COLD entry parked at the compaction floor — compaction
    /// tried it and could not shrink it, so it is excluded from further attempts.
    pub fn is_at_floor(&self, id: &NodeId) -> bool {
        self.cold.get(id).is_some_and(|c| c.at_floor)
    }

    /// Lossy compaction of one content blob, routed by node kind: code →
    /// skeleton, JSON → crushed, prose → extractive summary. Pure/deterministic.
    /// Returns the compact form (callers adopt it only when it is actually
    /// shorter than the original).
    fn compact_content(kind: &str, content: &str) -> String {
        match ContentRouter::classify(kind, content) {
            Route::Code => CausalAst::skeletonize(content),
            Route::Json => CausalCrusher::crush(content),
            Route::Prose => CausalSumm::summarize(content, 0, 0.0),
        }
    }

    /// Compact the coldest COLD content until total COLD content (inline +
    /// spilled) is within [`cold_content_budget`](Self::cold_content_budget).
    /// Deterministic: coldest-first by causal score, ties on id. A no-op without
    /// a budget or when already within it. **Lossy**: each compacted entry's full
    /// original is replaced by its summary/skeleton and discarded (a spilled
    /// original's on-disk blob is orphaned — reclaimed by a future GC pass). An
    /// entry that cannot be made smaller is skipped (the summary is the floor),
    /// so the budget is approached best-effort, never by dropping a node.
    /// Reclaim the on-disk spill blob for `hash` **iff** no COLD entry still
    /// references it. Safe with content-dedup (two cold nodes can share a blob):
    /// the file is deleted only once its last referent is gone. A no-op without an
    /// attached store. This is the GC that keeps the spill store from leaking
    /// orphaned blobs when a spilled node is re-ingested, removed, or compacted.
    /// Reclaim a spilled **content** blob iff no resident COLD full entry references
    /// it. A content hash is never a deep-husk *body* hash (different byte strings ⇒
    /// different SHA-256), so this deliberately skips the husk scan that
    /// [`release_blob_if_orphan`](Self::release_blob_if_orphan) does — keeping the
    /// deep-spill and compaction loops, which release content blobs, off the `O(N)`
    /// deep-tier sweep (otherwise deep-spilling N entries is `O(N²)` disk).
    fn release_content_blob_if_orphan(&self, hash: &[u8; 32]) {
        let Some(cfg) = self.spill.as_ref() else {
            return;
        };
        let referenced = self
            .cold
            .values()
            .any(|c| c.spill.as_ref().is_some_and(|s| s.hash == *hash));
        if !referenced {
            cfg.store.remove(hash);
        }
    }

    fn release_blob_if_orphan(&self, hash: &[u8; 32]) {
        let Some(cfg) = self.spill.as_ref() else {
            return;
        };
        let referenced_by_full = self
            .cold
            .values()
            .any(|c| c.spill.as_ref().is_some_and(|s| s.hash == *hash));
        // Lever 2: husks are on disk, so this scans the index — correct but O(N) per
        // release. Only body-blob releases (page-in, removal) hit it; content-blob
        // releases use the cheaper variant above.
        let referenced_by_husk = self
            .deep_entries()
            .iter()
            .any(|(_, h)| h.body.hash == *hash);
        if !referenced_by_full && !referenced_by_husk {
            cfg.store.remove(hash);
        }
    }

    /// Drop any COLD shadow of `id` — a full `ColdNode` or a deep `DeepHusk` —
    /// and reclaim its on-disk blob(s) if no other entry still references them. Used
    /// when a fresh ingest supersedes the shadow, or on explicit removal.
    fn forget_cold_shadow(&mut self, id: &NodeId) {
        // Gather the reverse-adjacency entries to drop (full entry: archived edges;
        // deep husk: neighbour ids) alongside reclaiming the blobs.
        let mut others: Vec<NodeId> = Vec::new();
        if let Some(old) = self.cold.remove(id) {
            for e in &old.edges {
                others.push(if &e.source == id {
                    e.target.clone()
                } else {
                    e.source.clone()
                });
            }
            if let Some(s) = old.spill {
                self.release_content_blob_if_orphan(&s.hash);
            }
        }
        if let Some(husk) = self.deep_get(id) {
            for o in unpack_adj(&husk.adj) {
                others.push(NodeId(o.to_owned()));
            }
            if let Some(hs) = self.husk_store.as_mut() {
                let _ = hs.delete(&id.0); // drop the husk from the on-disk index
            }
            self.release_blob_if_orphan(&husk.body.hash);
        }
        for other in others {
            self.radj_del_edge(id, &other);
        }
    }

    fn enforce_cold_content_budget(&mut self) {
        let Some(budget) = self.cold_content_budget else {
            return;
        };
        let mut total = self.cold_inline_bytes() + self.cold_spilled_bytes();
        if total <= budget {
            return;
        }
        // Coldest-first candidates: not already compacted, and not parked at the
        // compaction floor (un-shrinkable — see `ColdNode::at_floor`).
        let mut candidates: Vec<(NodeId, f64)> = self
            .cold
            .iter()
            .filter(|(_, c)| !c.compacted && !c.at_floor)
            .map(|(id, c)| (id.clone(), self.compute_node_score(&c.node)))
            .collect();
        candidates.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        for (id, _) in candidates {
            if total <= budget {
                break;
            }
            // Materialise the current content (faulting a spilled blob back in),
            // keeping the old spill hash so its blob can be reclaimed after.
            let (content, kind, old_len, old_hash) = match self.cold.get(&id) {
                Some(c) if !c.compacted => {
                    let kind = format!("{:?}", c.node.node_type);
                    match &c.spill {
                        Some(s) => match self.spill.as_ref().and_then(|cfg| cfg.store.get(&s.hash))
                        {
                            Some(text) => (text, kind, s.len, Some(s.hash)),
                            None => continue, // store detached / blob gone — leave it
                        },
                        None => (c.node.content.clone(), kind, c.node.content.len(), None),
                    }
                }
                _ => continue,
            };
            let compact = Self::compact_content(&kind, &content);
            if compact.len() >= content.len() {
                // Already at the floor: park it so later passes skip it (no repeat
                // work, no repeat disk read) rather than re-attempting every ingest.
                if let Some(entry) = self.cold.get_mut(&id) {
                    entry.at_floor = true;
                }
                continue;
            }
            let new_len = compact.len();
            if let Some(entry) = self.cold.get_mut(&id) {
                entry.node.content = compact;
                entry.spill = None; // discard the full original
                entry.compacted = true;
            }
            // The original (content) blob may now be orphaned — reclaim it if so.
            if let Some(h) = old_hash {
                self.release_content_blob_if_orphan(&h);
            }
            total = total.saturating_sub(old_len).saturating_add(new_len);
        }
    }

    /// Remove any edge whose endpoints are no longer present in the graph and
    /// return how many were pruned. With `add_edge` rejecting dangling edges
    /// this is normally a no-op, but it enforces the `edges ⊆ nodes × nodes`
    /// invariant after bulk or out-of-band mutations.
    pub fn prune_dangling_edges(&mut self) -> usize {
        let before = self.edges.len();
        let nodes = &self.nodes;
        self.edges
            .retain(|e| nodes.contains_key(&e.source) && nodes.contains_key(&e.target));
        before - self.edges.len()
    }

    pub fn propagate_failure(&mut self, origin_id: &NodeId, depth: u32, max_depth: u32) {
        if depth > max_depth {
            return;
        }
        // Get the origin's current failure relevance as the propagation base
        let base_value = self
            .nodes
            .get(origin_id)
            .map(|n| n.failure_relevance)
            .unwrap_or(0.0);

        // Find all edges where origin is the source
        let targets: Vec<(NodeId, f64)> = self
            .edges
            .iter()
            .filter(|e| &e.source == origin_id)
            .map(|e| (e.target.clone(), e.weight))
            .collect();

        let decay = self.scoring_weights.failure_decay;
        // Degree-aware damping: a node *distributes* its pressure across its
        // out-edges rather than replicating it to each. At or below `failure_fanout`
        // this is a no-op (`damp == 1`), so sparse causal chains still reach depth;
        // a hub (a file with dozens of contained symbols) is damped by
        // `fanout / out_degree`, which stops one over-connected node from flooding
        // the graph (FIELD_CAMPAIGN_H.md, root cause #2).
        let fanout = self.scoring_weights.failure_fanout.max(1.0);
        let damp = (fanout / (targets.len() as f64).max(fanout)).min(1.0);
        let floor = self.paging_threshold;
        for (target, weight) in targets {
            let propagation = base_value * weight * decay.powi(depth as i32) * damp;
            if let Some(node) = self.nodes.get_mut(&target) {
                node.failure_relevance = (node.failure_relevance + propagation).clamp(0.0, 1.0);
                node.recency = 1.0;
                node.last_accessed = self.clock;
            }
            // Stop relaying once the pressure added this hop is below the paging
            // floor: it cannot page anything in, and continuing only floods.
            if propagation > floor {
                self.propagate_failure(&target, depth + 1, max_depth);
            }
        }
    }

    /// Like [`propagate_failure`](Self::propagate_failure) but **bidirectional**:
    /// pressure flows to neighbours in *both* edge directions, so an injected
    /// fault reaches its upstream causes (callers/importers) as well as its
    /// downstream dependencies. Recursion only continues into a neighbour whose
    /// failure relevance actually grew, which (with the `depth ≤ max_depth` bound)
    /// keeps it terminating on cyclic graphs. Use when the failing node's *cause*
    /// may lie either side of it — e.g. bug-fix localisation.
    pub fn propagate_failure_bidirectional(
        &mut self,
        origin_id: &NodeId,
        depth: u32,
        max_depth: u32,
    ) {
        if depth > max_depth {
            return;
        }
        let base_value = self
            .nodes
            .get(origin_id)
            .map(|n| n.failure_relevance)
            .unwrap_or(0.0);

        // Neighbours via edges in either direction.
        let neighbours: Vec<(NodeId, f64)> = self
            .edges
            .iter()
            .filter_map(|e| {
                if &e.source == origin_id {
                    Some((e.target.clone(), e.weight))
                } else if &e.target == origin_id {
                    Some((e.source.clone(), e.weight))
                } else {
                    None
                }
            })
            .collect();

        let decay = self.scoring_weights.failure_decay;
        // Degree-aware damping (see [`propagate_failure`](Self::propagate_failure)):
        // a hub distributes pressure across its neighbours instead of replicating it.
        let fanout = self.scoring_weights.failure_fanout.max(1.0);
        let damp = (fanout / (neighbours.len() as f64).max(fanout)).min(1.0);
        let floor = self.paging_threshold;
        for (nb, weight) in neighbours {
            let propagation = base_value * weight * decay.powi(depth as i32) * damp;
            let mut grew = false;
            if let Some(node) = self.nodes.get_mut(&nb) {
                let before = node.failure_relevance;
                node.failure_relevance = (before + propagation).clamp(0.0, 1.0);
                node.recency = 1.0;
                node.last_accessed = self.clock;
                grew = node.failure_relevance > before + 1e-9;
            }
            // Continue only into neighbours that actually grew (cycle-safe) and only
            // while the added pressure can still page something in (anti-flood).
            if grew && propagation > floor {
                self.propagate_failure_bidirectional(&nb, depth + 1, max_depth);
            }
        }
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// A node by id, if present. (Read accessor — `nodes` is `pub(crate)` so
    /// external callers cannot break the `edges ⊆ nodes²` invariant by removing
    /// a node out from under its edges.)
    pub fn node(&self, id: &NodeId) -> Option<&GraphNode> {
        self.nodes.get(id)
    }

    /// A mutable node by id. Editing a node's *fields* (score, recency, failure
    /// pressure…) cannot break the structural invariant — only adding/removing
    /// nodes or edges can — so this is safe to expose where the structural
    /// mutators ([`add_edge`](Self::add_edge), [`upsert_node`](Self::upsert_node))
    /// are not.
    pub fn node_mut(&mut self, id: &NodeId) -> Option<&mut GraphNode> {
        self.nodes.get_mut(id)
    }

    /// Whether a node with this id is present.
    pub fn contains_node(&self, id: &NodeId) -> bool {
        self.nodes.contains_key(id)
    }

    /// All node ids (unordered — sort if you need determinism).
    pub fn node_ids(&self) -> impl Iterator<Item = &NodeId> + '_ {
        self.nodes.keys()
    }

    /// All `(id, node)` pairs (unordered).
    pub fn node_entries(&self) -> impl Iterator<Item = (&NodeId, &GraphNode)> + '_ {
        self.nodes.iter()
    }

    /// All nodes (unordered).
    pub fn node_values(&self) -> impl Iterator<Item = &GraphNode> + '_ {
        self.nodes.values()
    }

    /// The edges as a **read-only** slice — callers can inspect but not push a
    /// dangling edge or remove an endpoint.
    pub fn edges(&self) -> &[GraphEdge] {
        &self.edges
    }

    /// Count edges that violate `edges ⊆ nodes²` (an endpoint is missing),
    /// **without** modifying the graph — the read-only counterpart of
    /// [`prune_dangling_edges`](Self::prune_dangling_edges), used by integrity
    /// checks that must not mutate the snapshot they verify.
    pub fn dangling_edge_count(&self) -> usize {
        self.edges
            .iter()
            .filter(|e| !self.nodes.contains_key(&e.source) || !self.nodes.contains_key(&e.target))
            .count()
    }

    /// Resolve intra-crate imports into `file → file` dependency edges.
    ///
    /// The parser records imports as `use:<file>:<path>` nodes but does not link
    /// the importing file to the file that *defines* the imported module, so
    /// causally-related files end up connected only through shared `dep:` hubs.
    /// This pass maps each file to its module path (`src/a/b.rs` → `a::b`,
    /// `…/mod.rs|lib.rs|main.rs` to the parent) and, for every `use:` node,
    /// links the importer to the file whose module is the longest matching
    /// prefix of the import path. Idempotent (duplicate edges are rejected);
    /// returns the number of edges added. Opt-in — callers invoke it after
    /// ingesting a set of files (see [`crate::external_memory`]).
    pub fn link_module_imports(&mut self) -> usize {
        // Deterministic iteration: visit node ids in sorted order so the HashMap seed can't
        // change import resolution across processes (e.g. which file wins a `(crate, module)`
        // key collision via the last `insert`) — fresh ingestion must be reproducible, the
        // same property replay relies on.
        let mut sorted_ids: Vec<&NodeId> = self.nodes.keys().collect();
        sorted_ids.sort();
        // (crate, intra-module path) → file node.
        let mut index: HashMap<(String, String), NodeId> = HashMap::new();
        for id in &sorted_ids {
            if let Some(path) = id.0.strip_prefix("file:") {
                if let Some(km) = crate_and_module(path) {
                    index.insert(km, (*id).clone());
                }
            }
        }
        let mut to_add: Vec<(NodeId, NodeId, EdgeType)> = Vec::new();

        // (a) imports: importer → defining file.
        for id in &sorted_ids {
            let Some(rest) = id.0.strip_prefix("use:") else {
                continue;
            };
            let Some((file, usepath)) = rest.split_once(':') else {
                continue;
            };
            let importer = NodeId(format!("file:{file}"));
            if !self.nodes.contains_key(&importer) {
                continue;
            }
            let importer_crate = crate_and_module(file).map(|(c, _)| c).unwrap_or_default();
            if let Some(target) = resolve_use(&importer_crate, usepath, &index) {
                if target != importer {
                    to_add.push((importer, target, EdgeType::DependsOn));
                }
            }
        }

        // (b) module hierarchy: a parent module's file → its sub-module's file
        // (e.g. `filter/mod.rs → filter/owner.rs`). The parser records `pub mod x;`
        // only as a node, so without this a sub-module reached through a re-export
        // is orphaned and failure pressure can never flow into it.
        let mut entries: Vec<((String, String), NodeId)> =
            index.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        entries.sort();
        for ((krate, module), child) in &entries {
            if module.is_empty() {
                continue;
            }
            let parent_module = match module.rsplit_once("::") {
                Some((p, _)) => p.to_string(),
                None => String::new(), // top-level module → crate root (lib/main)
            };
            if let Some(parent) = index.get(&(krate.clone(), parent_module)) {
                if parent != child {
                    to_add.push((parent.clone(), child.clone(), EdgeType::Contains));
                }
            }
        }

        let mut added = 0;
        for (s, t, ty) in to_add {
            if self.add_edge(s, t, 0.85, ty) {
                added += 1;
            }
        }
        added
    }

    /// Record (replacing) this file's in-body call-sites, the input to
    /// [`resolve_symbol_calls`](Self::resolve_symbol_calls). Empty ⇒ drop the entry, so a
    /// re-ingest that removed every call leaves nothing stale. Each tuple is `(caller, callee,
    /// line)` in source order.
    pub fn set_pending_calls(&mut self, file: &str, calls: Vec<(String, String, usize)>) {
        if calls.is_empty() {
            self.pending_calls.remove(file);
        } else {
            self.pending_calls.insert(file.to_string(), calls);
        }
    }

    /// Record a file's in-body `static`/`const` references, awaiting
    /// [`resolve_data_flow`](Self::resolve_data_flow). Mirrors
    /// [`set_pending_calls`](Self::set_pending_calls).
    pub fn set_pending_data_refs(&mut self, file: &str, refs: Vec<(String, String, usize)>) {
        if refs.is_empty() {
            self.pending_data_refs.remove(file);
        } else {
            self.pending_data_refs.insert(file.to_string(), refs);
        }
    }

    /// Record (replacing) this file's renamed-import bindings, the alias input to
    /// [`resolve_symbol_calls`](Self::resolve_symbol_calls). Each tuple is `(local_name,
    /// target_path)` — `use a::b as c` ⇒ `("c", "a::b")`. Empty ⇒ drop the entry (a re-ingest that
    /// removed every rename leaves nothing stale). Mirrors [`set_pending_calls`](Self::set_pending_calls).
    pub fn set_pending_aliases(&mut self, file: &str, aliases: Vec<(String, String)>) {
        if aliases.is_empty() {
            self.pending_aliases.remove(file);
        } else {
            self.pending_aliases.insert(file.to_string(), aliases);
        }
    }

    /// Mark a symbol node as a `static`/`const` — the only valid `DataFlow` target. The parser
    /// calls this at ingest (the graph node itself does not carry `SymbolKind`).
    pub fn mark_data_symbol(&mut self, id: NodeId) {
        self.data_symbols.insert(id);
    }

    /// Resolve recorded `static`/`const` references into `reader → item` [`EdgeType::DataFlow`]
    /// edges — the shared-global-state channel call and import edges miss (a function reads a
    /// global defined in a file it never imports by name). A **whole-graph** pass, run after
    /// [`resolve_symbol_calls`](Self::resolve_symbol_calls). Resolution is **global-unique,
    /// resolve-uniquely-or-skip**: a reference to `X` links only when exactly one resident
    /// `static`/`const` named `X` exists graph-wide, so a wrong edge is never invented. Self-edges
    /// dropped. Deterministic: indices over the **sorted** `data_symbols`/`pending_data_refs`,
    /// candidate edges sorted+deduped before insertion. Returns the number of `DataFlow` edges added.
    pub fn resolve_data_flow(&mut self) -> usize {
        // Build the global-unique index from resident data symbols, then collect candidate edges —
        // in an inner scope so its borrows of `self` end before the mutating `add_edge` below.
        let mut to_add: Vec<(NodeId, NodeId)> = {
            let mut name_count: HashMap<&str, u32> = HashMap::new();
            let mut name_first: HashMap<&str, &NodeId> = HashMap::new();
            for id in &self.data_symbols {
                if !self.nodes.contains_key(id) {
                    continue; // evicted since it was marked — skip the stale entry
                }
                if let Some((_, name)) = id.0.rsplit_once(':') {
                    *name_count.entry(name).or_insert(0) += 1;
                    name_first.entry(name).or_insert(id);
                }
            }
            let mut acc: Vec<(NodeId, NodeId)> = Vec::new();
            for (file, refs) in &self.pending_data_refs {
                for (reader, name, _line) in refs {
                    let reader_sym = NodeId(format!("sym:{file}:{reader}"));
                    if !self.nodes.contains_key(&reader_sym) {
                        continue;
                    }
                    if name_count.get(name.as_str()).copied() == Some(1) {
                        if let Some(target) = name_first.get(name.as_str()) {
                            if **target != reader_sym {
                                acc.push((reader_sym.clone(), (*target).clone()));
                            }
                        }
                    }
                }
            }
            acc
        };
        to_add.sort();
        to_add.dedup();
        let mut added = 0usize;
        for (s, t) in to_add {
            if self.add_edge(s, t, 0.6, EdgeType::DataFlow) {
                added += 1;
            }
        }
        added
    }

    /// Resolve recorded call-sites into `caller → callee` [`EdgeType::Calls`] edges — the fn→fn
    /// structure import edges miss (a call crosses files even when the two functions share no
    /// vocabulary). A **whole-graph** pass (a call may target a symbol in a file ingested later);
    /// run right after [`link_module_imports`](Self::link_module_imports). Resolution is a strict
    /// precision ladder, **resolve-uniquely-or-skip** at every tier, so a wrong edge is never
    /// invented. A **`Self::method`** callee (a captured `self.m()` or `Self::assoc()`, Slice 3)
    /// resolves the method name in the caller's OWN module only — never via imports or global-unique,
    /// so a method is never mislinked to a same-named free function elsewhere. A **qualified** callee
    /// (`mod::…::name`, Slice 2) resolves its module prefix to a
    /// file — crate-rooted (`crate::m::name`) directly, or an `alias::name` by expanding the leading
    /// segment through the file's imports — then takes `sym:<file>:name`; unresolvable/ambiguous
    /// qualified paths skip without falling back. A **bare** callee (`foo()`, Slice 1) uses the
    /// ladder: (A) import-scoped — the file does `use …::foo`, its module resolving (as in
    /// `link_module_imports`) to the defining file with a real `sym:<file>:foo`; (B) same-module —
    /// `foo` defined in the caller's own file/module; (C) global-unique — exactly one `sym:*:foo`
    /// exists graph-wide. Self-edges are dropped. Deterministic: indices built over
    /// **sorted** node ids, calls iterated in sorted (file, source) order, candidate edges
    /// sorted+deduped before insertion. Returns the number of Calls edges added.
    pub fn resolve_symbol_calls(&mut self) -> usize {
        let mut sorted: Vec<&NodeId> = self.nodes.keys().collect();
        sorted.sort();
        let mut file_index: HashMap<(String, String), NodeId> = HashMap::new(); // (crate,module)->file
        let mut defs: HashMap<(String, String, String), NodeId> = HashMap::new(); // (crate,module,name)->sym
        let mut name_count: HashMap<String, u32> = HashMap::new();
        let mut name_first: HashMap<String, NodeId> = HashMap::new();
        let mut scope: HashMap<String, Vec<String>> = HashMap::new(); // file -> use paths
        for id in &sorted {
            let s = id.0.as_str();
            if let Some(path) = s.strip_prefix("file:") {
                if let Some(km) = crate_and_module(path) {
                    file_index.insert(km, (*id).clone());
                }
            } else if let Some(rest) = s.strip_prefix("sym:") {
                if let Some((file, name)) = rest.rsplit_once(':') {
                    if let Some((c, m)) = crate_and_module(file) {
                        defs.insert((c, m, name.to_string()), (*id).clone());
                    }
                    *name_count.entry(name.to_string()).or_default() += 1;
                    name_first
                        .entry(name.to_string())
                        .or_insert_with(|| (*id).clone());
                }
            } else if let Some(rest) = s.strip_prefix("use:") {
                if let Some((file, usepath)) = rest.split_once(':') {
                    scope
                        .entry(file.to_string())
                        .or_default()
                        .push(usepath.to_string());
                }
            }
        }

        // Per-file alias lookup: local name → target path (`use a::b as c` ⇒ `c` → `a::b`). Built
        // from the renamed-import bindings the parser handed over. A later call to a local name in
        // this map is rewritten onto its target before resolution (see `resolve_call`). An empty
        // (default) map for a file ⇒ no aliases, the unchanged path.
        let mut alias_index: HashMap<&str, HashMap<&str, &str>> = HashMap::new();
        for (file, aliases) in &self.pending_aliases {
            let m = alias_index.entry(file.as_str()).or_default();
            for (local, target) in aliases {
                // First binding wins (deterministic: pending lists are source-order). A duplicate
                // local name only arises in non-compiling Rust; keeping the first is harmless.
                m.entry(local.as_str()).or_insert(target.as_str());
            }
        }
        let empty_aliases: HashMap<&str, &str> = HashMap::new();

        let mut to_add: Vec<(NodeId, NodeId)> = Vec::new();
        for (file, calls) in &self.pending_calls {
            let (fcrate, fmodule) = crate_and_module(file).unwrap_or_default();
            let file_aliases = alias_index.get(file.as_str()).unwrap_or(&empty_aliases);
            for (caller, callee, _line) in calls {
                let caller_sym = NodeId(format!("sym:{file}:{caller}"));
                if !self.nodes.contains_key(&caller_sym) {
                    continue;
                }
                if let Some(t) = resolve_call(
                    file,
                    callee,
                    &fcrate,
                    &fmodule,
                    &scope,
                    &file_index,
                    &defs,
                    &name_count,
                    &name_first,
                    file_aliases,
                ) {
                    if t != caller_sym {
                        to_add.push((caller_sym, t));
                    }
                }
            }
        }
        to_add.sort();
        to_add.dedup();
        let mut added = 0;
        for (s, t) in to_add {
            if self.add_edge(s, t, 0.75, EdgeType::Calls) {
                added += 1;
            }
        }
        added
    }

    /// Detect directed dependency cycles via an iterative (stack-safe) DFS.
    /// Each returned vector lists the nodes forming one cycle, in order.
    pub fn find_cycles(&self) -> Vec<Vec<NodeId>> {
        use std::collections::BTreeMap;
        let mut adj: BTreeMap<NodeId, Vec<NodeId>> = BTreeMap::new();
        for id in self.nodes.keys() {
            adj.entry(id.clone()).or_default();
        }
        for e in &self.edges {
            if self.nodes.contains_key(&e.source) && self.nodes.contains_key(&e.target) {
                adj.get_mut(&e.source).unwrap().push(e.target.clone());
            }
        }
        for v in adj.values_mut() {
            v.sort();
            v.dedup();
        }

        // color: 0 = unvisited, 1 = on the current DFS path, 2 = done
        let mut color: HashMap<NodeId, u8> = HashMap::new();
        let mut cycles: Vec<Vec<NodeId>> = Vec::new();

        for start in adj.keys() {
            if *color.get(start).unwrap_or(&0) != 0 {
                continue;
            }
            let mut stack: Vec<(NodeId, usize)> = vec![(start.clone(), 0)];
            let mut path: Vec<NodeId> = vec![start.clone()];
            color.insert(start.clone(), 1);
            while let Some((node, idx)) = stack.last().cloned() {
                let children = &adj[&node];
                if idx < children.len() {
                    stack.last_mut().unwrap().1 += 1;
                    let child = children[idx].clone();
                    match *color.get(&child).unwrap_or(&0) {
                        0 => {
                            color.insert(child.clone(), 1);
                            stack.push((child.clone(), 0));
                            path.push(child);
                        }
                        1 => {
                            // Back edge: a cycle spans path[pos..].
                            if let Some(pos) = path.iter().position(|n| n == &child) {
                                cycles.push(path[pos..].to_vec());
                            }
                        }
                        _ => {}
                    }
                } else {
                    color.insert(node.clone(), 2);
                    stack.pop();
                    path.pop();
                }
            }
        }
        cycles
    }

    /// Structural difference against another graph (added/removed nodes & edges).
    pub fn diff(&self, other: &MemoryGraph) -> GraphDiff {
        use std::collections::HashSet;
        let a: HashSet<&NodeId> = self.nodes.keys().collect();
        let b: HashSet<&NodeId> = other.nodes.keys().collect();

        let mut nodes_added: Vec<NodeId> = b.difference(&a).map(|n| (*n).clone()).collect();
        let mut nodes_removed: Vec<NodeId> = a.difference(&b).map(|n| (*n).clone()).collect();
        nodes_added.sort();
        nodes_removed.sort();

        let edge_key = |e: &GraphEdge| {
            (
                e.source.0.clone(),
                e.target.0.clone(),
                format!("{:?}", e.edge_type),
            )
        };
        let ea: HashSet<_> = self.edges.iter().map(edge_key).collect();
        let eb: HashSet<_> = other.edges.iter().map(edge_key).collect();

        GraphDiff {
            nodes_added,
            nodes_removed,
            edges_added: eb.difference(&ea).count(),
            edges_removed: ea.difference(&eb).count(),
            common_nodes: a.intersection(&b).count(),
        }
    }

    /// Count nodes by type, sorted by descending frequency (then name).
    pub fn node_type_counts(&self) -> Vec<(String, usize)> {
        let mut counts: HashMap<String, usize> = HashMap::new();
        for node in self.nodes.values() {
            *counts.entry(format!("{:?}", node.node_type)).or_insert(0) += 1;
        }
        let mut out: Vec<(String, usize)> = counts.into_iter().collect();
        out.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        out
    }

    /// Nodes with no incident edges (neither incoming nor outgoing) — isolated
    /// fragments that no other part of the graph references.
    pub fn orphan_nodes(&self) -> Vec<&NodeId> {
        use std::collections::HashSet;
        let mut connected: HashSet<&NodeId> = HashSet::new();
        for e in &self.edges {
            connected.insert(&e.source);
            connected.insert(&e.target);
        }
        let mut orphans: Vec<&NodeId> = self
            .nodes
            .keys()
            .filter(|id| !connected.contains(id))
            .collect();
        orphans.sort();
        orphans
    }

    /// Render the graph as Graphviz DOT (deterministic node/edge order, colored
    /// by type) for visualization: `ccos analyze <path> --dot graph.dot`.
    pub fn to_dot(&self) -> String {
        fn esc(s: &str) -> String {
            s.replace('\\', "\\\\").replace('"', "\\\"")
        }
        let color = |t: &NodeType| match t {
            NodeType::Module => "#cfe2ff",
            NodeType::Symbol => "#d1e7dd",
            NodeType::ContextBlock => "#fff3cd",
            NodeType::AnalysisResult => "#f8d7da",
            NodeType::CodeRegion => "#e2e3e5",
            NodeType::Unknown => "#ffffff",
        };

        let mut out = String::from(
            "digraph ccos {\n  rankdir=LR;\n  node [shape=box, style=\"rounded,filled\", fontname=monospace];\n",
        );

        let mut ids: Vec<&NodeId> = self.nodes.keys().collect();
        ids.sort();
        for id in &ids {
            let n = &self.nodes[*id];
            out.push_str(&format!(
                "  \"{}\" [label=\"{}\", fillcolor=\"{}\"];\n",
                esc(&id.0),
                esc(&n.label),
                color(&n.node_type),
            ));
        }

        let mut edges: Vec<&GraphEdge> = self.edges.iter().collect();
        edges.sort_by(|a, b| {
            a.source
                .cmp(&b.source)
                .then_with(|| a.target.cmp(&b.target))
                .then_with(|| format!("{:?}", a.edge_type).cmp(&format!("{:?}", b.edge_type)))
        });
        for e in &edges {
            out.push_str(&format!(
                "  \"{}\" -> \"{}\" [label=\"{:?}\"];\n",
                esc(&e.source.0),
                esc(&e.target.0),
                e.edge_type,
            ));
        }
        out.push_str("}\n");
        out
    }

    /// Symbol nodes that **nothing references** — no incoming edge other than the structural
    /// `Contains` from their own file. A deliberate **heuristic**: a `pub` API, an entry point
    /// (`main`), or a trait-impl method reached only from outside the analyzed graph are false
    /// positives, so these are dead-code *candidates*, not a proof. Deterministic (sorted ids).
    pub fn dead_symbols(&self) -> Vec<NodeId> {
        let mut referenced: std::collections::HashSet<&NodeId> = std::collections::HashSet::new();
        for e in &self.edges {
            if e.edge_type != EdgeType::Contains {
                referenced.insert(&e.target);
            }
        }
        let mut dead: Vec<NodeId> = self
            .nodes
            .iter()
            .filter(|(id, n)| {
                n.node_type == NodeType::Symbol
                    && id.0.starts_with("sym:")
                    && !referenced.contains(*id)
            })
            .map(|(id, _)| id.clone())
            .collect();
        dead.sort();
        dead
    }

    pub fn get_node_scores(&self) -> Vec<(NodeId, f64)> {
        let mut scores: Vec<(NodeId, f64)> = self
            .nodes
            .iter()
            .map(|(id, node)| (id.clone(), self.compute_node_score(node)))
            .collect();
        scores.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        scores
    }
}

impl Default for MemoryGraph {
    fn default() -> Self {
        Self::new(0.2, 100)
    }
}

/// Structural difference between two graphs, produced by [`MemoryGraph::diff`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphDiff {
    pub nodes_added: Vec<NodeId>,
    pub nodes_removed: Vec<NodeId>,
    pub edges_added: usize,
    pub edges_removed: usize,
    pub common_nodes: usize,
}

/// The `::`-joined intra-crate module path of a source file below a `src/` root.
/// `a/b.rs` → `a::b`; `mod.rs`/`lib.rs`/`main.rs` → the parent (empty at the
/// crate root). Used by [`crate_and_module`].
fn module_of(after: &str) -> Option<String> {
    let p = after.strip_suffix(".rs")?;
    let segs: Vec<&str> = p.split('/').filter(|s| !s.is_empty()).collect();
    let last = *segs.last()?;
    let module = if matches!(last, "mod" | "lib" | "main") {
        segs[..segs.len() - 1].join("::")
    } else {
        segs.join("::")
    };
    Some(module)
}

/// Split a file path into `(crate, intra-module)` for import resolution, robust to
/// relative, absolute, and multi-crate-workspace layouts:
/// `src/a/b.rs` → `("", "a::b")`; `/repo/src/a.rs` → `("repo", "a")`;
/// `crates/grep-matcher/src/lib.rs` → `("grep_matcher", "")`. The crate is the
/// directory name immediately above `src/` (`-` normalised to `_`), or `""` for a
/// path that simply starts with `src/`.
fn crate_and_module(file_path: &str) -> Option<(String, String)> {
    let (krate, after) = if let Some(idx) = file_path.find("/src/") {
        let krate = file_path[..idx]
            .rsplit('/')
            .next()
            .unwrap_or("")
            .replace('-', "_");
        (krate, &file_path[idx + 5..])
    } else if let Some(after) = file_path.strip_prefix("src/") {
        (String::new(), after)
    } else {
        (String::new(), file_path)
    };
    Some((krate, module_of(after)?))
}

/// Parse a `use`/call path into `(target_crate, module_segments_after_the_crate_root)`.
/// `crate::`/`self::`/`super::` (and bare `crate`/`self`/`super`) stay in the importer's crate; a
/// leading crate name (e.g. `grep_matcher::…`) targets that crate.
fn parse_modpath(importer_crate: &str, usepath: &str) -> (String, Vec<String>) {
    let (target_crate, rest): (String, &str) = if let Some(r) = usepath
        .strip_prefix("crate::")
        .or_else(|| usepath.strip_prefix("self::"))
        .or_else(|| usepath.strip_prefix("super::"))
    {
        (importer_crate.to_string(), r)
    } else if matches!(usepath, "crate" | "self" | "super") {
        (importer_crate.to_string(), "")
    } else {
        match usepath.split_once("::") {
            Some((c, r)) => (c.to_string(), r),
            None => (usepath.to_string(), ""),
        }
    };
    let segs = rest
        .split("::")
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect();
    (target_crate, segs)
}

/// Resolve an import to the defining file's node id. `crate::`/`self::`/`super::` stay in the
/// importer's crate (requiring a real sub-module match); a leading crate name targets that crate
/// and may resolve to its root (`lib.rs`). Tries the full module path then **shortens** to ancestor
/// modules — correct for a `use`-target re-exported at an ancestor (`use a::Thing`, `Thing` living
/// in `a`'s file). External paths like `std::io` match nothing.
fn resolve_use(
    importer_crate: &str,
    usepath: &str,
    index: &HashMap<(String, String), NodeId>,
) -> Option<NodeId> {
    let (target_crate, segs) = parse_modpath(importer_crate, usepath);
    // Same-crate imports must hit a real sub-module (len ≥ 1); cross-crate imports
    // may resolve to the crate root (len 0) — depending on the crate as a whole.
    let min_len = usize::from(target_crate == importer_crate);
    (min_len..=segs.len()).rev().find_map(|len| {
        let module = segs[..len].join("::");
        index.get(&(target_crate.clone(), module)).cloned()
    })
}

/// Resolve a path to the file whose module is **exactly** that path — no ancestor shortening.
/// Used to pin a qualified call's *definition* module: unlike [`resolve_use`], a `crate::a::b` whose
/// `a::b` module has no file must NOT fall back to an existing `a` (that would attribute the call to
/// a wrong-existing symbol — the inline-submodule / ancestor-fallback / `Enum::Variant` false edge).
fn resolve_module_exact(
    importer_crate: &str,
    path: &str,
    index: &HashMap<(String, String), NodeId>,
) -> Option<NodeId> {
    let (target_crate, segs) = parse_modpath(importer_crate, path);
    index.get(&(target_crate, segs.join("::"))).cloned()
}

/// The Slice-1 call→def resolution ladder used by
/// [`MemoryGraph::resolve_symbol_calls`](MemoryGraph::resolve_symbol_calls): Tier A (import-scoped),
/// then B (same-module), then C (global-unique), each **resolve-uniquely-or-skip**. Returns the
/// definition symbol node for a bare `callee()` call in `file`, or `None` to skip. `aliases` maps a
/// renamed-import local name to its original target path (`use a::b as c` ⇒ `c` → `a::b`); a call
/// whose leading segment is such a local name is rewritten onto the target first, so it resolves
/// exactly as the original path would — never onto a same-named sibling module/symbol.
#[allow(clippy::too_many_arguments)]
fn resolve_call(
    file: &str,
    callee: &str,
    fcrate: &str,
    fmodule: &str,
    scope: &HashMap<String, Vec<String>>,
    file_index: &HashMap<(String, String), NodeId>,
    defs: &HashMap<(String, String, String), NodeId>,
    name_count: &HashMap<String, u32>,
    name_first: &HashMap<String, NodeId>,
    aliases: &HashMap<&str, &str>,
) -> Option<NodeId> {
    // Renamed import — `use a::b as c` binds local `c` to target `a::b`. If the call's LEADING
    // segment is such a local name, rewrite it onto the target (`c` ⇒ `a::b`, `c::X` ⇒ `a::b::X`)
    // and re-resolve the rewritten path as if it had been written literally. Re-resolution passes
    // an EMPTY alias map, so an alias is applied at most once (no re-aliasing, no recursion loop).
    // This keeps resolution byte-for-byte identical to the original path: a bare alias falls through
    // the same Tier A/B/C ladder, a multi-segment alias through the qualified-path logic — and the
    // resolve-uniquely-or-skip guards (exact module match, global-unique) still gate every edge, so
    // an alias whose target has no resident symbol simply yields no edge.
    if !aliases.is_empty() {
        let first = callee.split("::").next().unwrap_or(callee);
        if let Some(target) = aliases.get(first) {
            let rewritten = match callee.split_once("::") {
                Some((_, rest)) => format!("{target}::{rest}"),
                None => (*target).to_string(),
            };
            let empty: HashMap<&str, &str> = HashMap::new();
            return resolve_call(
                file, &rewritten, fcrate, fmodule, scope, file_index, defs, name_count, name_first,
                &empty,
            );
        }
    }
    // Slice 3 — `Self::method` (a captured `self.m()` or `Self::assoc()`): the receiver is the
    // enclosing impl's type, whose methods live in the caller's own file/module. Resolve the method
    // name there ONLY — never via imports (Tier A) or global-unique (Tier C), so a method is never
    // mislinked to a free function of the same name elsewhere. Unresolvable → skip.
    if let Some(method) = callee.strip_prefix("Self::") {
        return defs
            .get(&(fcrate.to_string(), fmodule.to_string(), method.to_string()))
            .cloned();
    }
    // Slice 2 — qualified path `mod::…::name`: pin the module prefix to its defining file, then take
    // the unique `sym:<file>:name`. resolve-uniquely-or-skip, so anything unresolvable or ambiguous →
    // no edge (and a qualified path never falls through to the bare same-module/global ladder below).
    if let Some((prefix, name)) = callee.rsplit_once("::") {
        // Build the prefix's absolute module path: crate-rooted directly, else expand the leading
        // segment through the file's imports (`use …::alias`). Skip if the alias matches no import,
        // or more than one (ambiguous). `self`/`super`/`Self`/`Type` and external roots (`std::…`)
        // match no import here → skipped (Slice 3 / by design).
        let abs: Option<String> = if prefix == "crate" || prefix.starts_with("crate::") {
            Some(prefix.to_string())
        } else {
            let first = prefix.split("::").next().unwrap_or(prefix);
            let rest = prefix.strip_prefix(first).unwrap_or(""); // "" or "::b::c"
            let mut hits = scope
                .get(file)
                .into_iter()
                .flatten()
                .filter(|u| u.rsplit("::").next() == Some(first));
            match (hits.next(), hits.next()) {
                (Some(use_path), None) => Some(format!("{use_path}{rest}")),
                _ => None,
            }
        };
        // Resolve the module EXACTLY (no ancestor shortening — see `resolve_module_exact`): a
        // `crate::a::b` whose `a::b` module has no file must skip, not fall back to `a`'s symbols.
        return abs
            .and_then(|a| resolve_module_exact(fcrate, &a, file_index))
            .and_then(|df| df.0.strip_prefix("file:").map(str::to_string))
            .and_then(|f| crate_and_module(&f))
            .and_then(|(c, m)| defs.get(&(c, m, name.to_string())).cloned());
    }
    // Tier A — import-scoped: an import `use <module>::callee` pins the defining module; resolve it
    // to a file and require exactly one unique `sym:<file>:callee`. (Rust-name-resolution-correct:
    // links the cross-module call without linking same-named fns in unrelated modules.)
    if let Some(paths) = scope.get(file) {
        let mut cands: std::collections::BTreeSet<NodeId> = std::collections::BTreeSet::new();
        for usepath in paths {
            if usepath.rsplit("::").next() != Some(callee) {
                continue; // this import does not bring in `callee`
            }
            let Some((module_path, _)) = usepath.rsplit_once("::") else {
                continue; // bare `use callee;` — no module path to resolve in Slice 1
            };
            if let Some(deffile_node) = resolve_use(fcrate, module_path, file_index) {
                if let Some(deffile) = deffile_node.0.strip_prefix("file:") {
                    if let Some((c, m)) = crate_and_module(deffile) {
                        if let Some(sym) = defs.get(&(c, m, callee.to_string())) {
                            cands.insert(sym.clone());
                        }
                    }
                }
            }
        }
        if cands.len() == 1 {
            return cands.into_iter().next();
        }
        if cands.len() > 1 {
            // The import scope itself is ambiguous for `callee` (only possible in non-compiling
            // Rust — duplicate imports). Honour resolve-uniquely-or-skip: do NOT fall through to a
            // same-module local guess, which would link a call the import scope already contradicts.
            return None;
        }
    }
    // Tier B — same module as the caller.
    if let Some(sym) = defs.get(&(fcrate.to_string(), fmodule.to_string(), callee.to_string())) {
        return Some(sym.clone());
    }
    // Tier C — exactly one symbol of this name graph-wide (prelude / sibling re-export).
    if name_count.get(callee).copied() == Some(1) {
        return name_first.get(callee).cloned();
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn link_module_imports_connects_cross_file_uses() {
        let mut g = MemoryGraph::new(0.0, usize::MAX);
        for f in ["src/api.rs", "src/repo.rs", "src/db.rs"] {
            g.upsert_node(
                format!("file:{f}").into(),
                f.into(),
                "".into(),
                NodeType::Module,
            );
        }
        g.upsert_node(
            "use:src/api.rs:crate::repo".into(),
            "".into(),
            "".into(),
            NodeType::Unknown,
        );
        g.upsert_node(
            "use:src/repo.rs:crate::db::connect".into(),
            "".into(),
            "".into(),
            NodeType::Unknown,
        );
        let added = g.link_module_imports();
        assert_eq!(added, 2, "two cross-file imports resolved");
        assert!(g
            .edges
            .iter()
            .any(|e| e.source.0 == "file:src/api.rs" && e.target.0 == "file:src/repo.rs"));
        assert!(g
            .edges
            .iter()
            .any(|e| e.source.0 == "file:src/repo.rs" && e.target.0 == "file:src/db.rs"));
        assert_eq!(g.link_module_imports(), 0, "idempotent");
    }

    #[test]
    fn link_module_imports_handles_multi_crate_and_absolute_paths() {
        let mut g = MemoryGraph::new(0.0, usize::MAX);
        let node = |g: &mut MemoryGraph, id: &str| {
            g.upsert_node(id.into(), "".into(), "".into(), NodeType::Module);
        };
        // Multi-crate workspace: core imports a sibling crate `util` (dir uses `-`).
        node(&mut g, "file:crates/core/src/api.rs");
        node(&mut g, "file:crates/grep-util/src/lib.rs");
        node(&mut g, "use:crates/core/src/api.rs:grep_util::helper");
        // Absolute-path mono-crate with an intra-crate import.
        node(&mut g, "file:/repo/src/a.rs");
        node(&mut g, "file:/repo/src/b.rs");
        node(&mut g, "use:/repo/src/a.rs:crate::b");
        g.link_module_imports();
        assert!(
            g.edges
                .iter()
                .any(|e| e.source.0 == "file:crates/core/src/api.rs"
                    && e.target.0 == "file:crates/grep-util/src/lib.rs"),
            "cross-crate import (grep_util ← grep-util dir) resolves to the crate root"
        );
        assert!(
            g.edges
                .iter()
                .any(|e| e.source.0 == "file:/repo/src/a.rs" && e.target.0 == "file:/repo/src/b.rs"),
            "absolute-path intra-crate import resolves"
        );
    }

    #[test]
    fn bidirectional_failure_reaches_upstream_causes() {
        let mut g = MemoryGraph::new(0.0, usize::MAX);
        g.upsert_node("a".into(), "a".into(), "".into(), NodeType::Module);
        g.upsert_node("b".into(), "b".into(), "".into(), NodeType::Module);
        // a depends on b (a → b); b is upstream of nothing, a is upstream of b.
        g.add_edge("a".into(), "b".into(), 1.0, EdgeType::DependsOn);
        // Inject at b: downstream-only propagation cannot reach a.
        g.set_failure_relevance(&"b".into(), 1.0);
        g.propagate_failure(&"b".into(), 0, 3);
        assert_eq!(g.nodes.get(&"a".into()).unwrap().failure_relevance, 0.0);
        // Bidirectional propagation reaches the upstream cause a.
        g.propagate_failure_bidirectional(&"b".into(), 0, 3);
        assert!(g.nodes.get(&"a".into()).unwrap().failure_relevance > 0.0);
    }

    #[test]
    fn test_upsert_node() {
        let mut graph = MemoryGraph::default();
        graph.upsert_node(
            "n1".into(),
            "Test".into(),
            "content".into(),
            NodeType::Module,
        );
        assert_eq!(graph.node_count(), 1);
    }

    #[test]
    fn test_add_edge() {
        let mut graph = MemoryGraph::default();
        graph.upsert_node("a".into(), "A".into(), "".into(), NodeType::Module);
        graph.upsert_node("b".into(), "B".into(), "".into(), NodeType::Module);
        graph.add_edge("a".into(), "b".into(), 0.8, EdgeType::DependsOn);
        assert_eq!(graph.edge_count(), 1);
    }

    #[test]
    fn scoring_weights_default_matches_shipped_constants() {
        // Regression guard: the tunable defaults must reproduce the original
        // hard-coded score, so existing snapshots/hashes/behaviour are unchanged.
        let w = ScoringWeights::default();
        assert_eq!(w.w_base, 0.15);
        assert_eq!(w.w_failure, 0.50);
        assert_eq!(w.w_recency, 0.30);
        assert_eq!(w.w_access, 0.05);
        assert_eq!(w.failure_decay, 0.8);
        assert_eq!(w.failure_fanout, 6.0);
    }

    #[test]
    fn scoring_weights_alter_score() {
        let mut g = MemoryGraph::new(0.0, usize::MAX);
        g.upsert_node("n".into(), "n".into(), "".into(), NodeType::Module);
        g.set_failure_relevance(&"n".into(), 1.0);
        let node = g.nodes.get(&"n".into()).unwrap().clone();
        g.set_scoring_weights(ScoringWeights {
            w_failure: 0.0,
            ..Default::default()
        });
        let low = g.compute_node_score(&node);
        g.set_scoring_weights(ScoringWeights {
            w_failure: 1.0,
            ..Default::default()
        });
        let high = g.compute_node_score(&node);
        assert!(
            high > low,
            "raising w_failure must raise a failing node's score"
        );
    }

    #[test]
    fn touch_refreshes_resident_score_and_never_resurrects_cold() {
        let mut g = MemoryGraph::new(0.2, usize::MAX);
        g.upsert_node("a".into(), "a".into(), "x".into(), NodeType::Module);
        // Let recency decay, then touch must restore it (recency → 1.0, access++).
        for _ in 0..5 {
            g.tick();
        }
        let before = g.compute_node_score(g.node(&"a".into()).unwrap());
        assert!(g.touch(&"a".into()), "a resident node is touchable");
        let after = g.compute_node_score(g.node(&"a".into()).unwrap());
        assert!(
            after > before,
            "touch must raise the score via recency/access refresh: {before} -> {after}"
        );
        // access_count: 1 at ingest, +1 from the single touch (tick does not bump it).
        assert_eq!(g.node(&"a".into()).unwrap().access_count, 2);
        // A cold (demoted) node is never paged back in by touch.
        g.max_in_memory_nodes = 0;
        g.enforce_paging();
        assert!(g.is_cold(&"a".into()));
        assert!(
            !g.touch(&"a".into()),
            "touch must not resurrect a cold node"
        );
        assert!(
            g.is_cold(&"a".into()),
            "touch must leave the COLD tier untouched"
        );
        // An absent id is simply a miss.
        assert!(!g.touch(&"missing".into()));
    }

    /// Eigenvector centrality is a pure function of the graph and handles the trivial
    /// shapes (empty, edgeless, single hub).
    #[test]
    fn eigencentrality_is_deterministic_and_handles_edge_cases() {
        // Empty graph → empty vector.
        assert!(MemoryGraph::new(0.2, usize::MAX)
            .eigencentrality()
            .is_empty());

        // Edgeless graph → every node equal (normalized to 1.0).
        let mut g = MemoryGraph::new(0.2, usize::MAX);
        for n in ["a", "b", "c"] {
            g.upsert_node(n.into(), n.into(), "".into(), NodeType::Module);
        }
        let iso = g.eigencentrality();
        assert_eq!(iso.len(), 3);
        assert!(iso.values().all(|v| (v - 1.0).abs() < 1e-9));

        // A hub depended-upon by three leaves: deterministic, and the hub leads.
        let build = || {
            let mut g = MemoryGraph::new(0.2, usize::MAX);
            for n in ["hub", "x", "y", "z"] {
                g.upsert_node(n.into(), n.into(), "".into(), NodeType::Module);
            }
            for s in ["x", "y", "z"] {
                g.add_edge(s.into(), "hub".into(), 1.0, EdgeType::DependsOn);
            }
            g.eigencentrality()
        };
        assert_eq!(
            build(),
            build(),
            "eigencentrality is a pure function of the graph"
        );
        let ec = build();
        let hub = ec.get(&"hub".into()).copied().unwrap();
        for leaf in ["x", "y", "z"] {
            assert!(
                hub > ec.get(&leaf.into()).copied().unwrap(),
                "hub outranks its leaves"
            );
        }
    }

    /// Eigencentrality must be invariant to the *order* edges were inserted (import
    /// resolution adds them in HashMap order), not merely reproducible for one fixed order —
    /// otherwise its `f64`s drift across processes. Same graph, edges added in two different
    /// orders ⇒ identical vector.
    #[test]
    fn eigencentrality_is_invariant_to_edge_insertion_order() {
        let mk = |edges: &[(&str, &str)]| {
            let mut g = MemoryGraph::new(0.2, usize::MAX);
            for n in ["a", "b", "c", "d", "hub"] {
                g.upsert_node(n.into(), n.into(), "".into(), NodeType::Module);
            }
            for &(s, t) in edges {
                g.add_edge(s.into(), t.into(), 1.0, EdgeType::DependsOn);
            }
            g.eigencentrality()
        };
        let forward = mk(&[
            ("a", "hub"),
            ("b", "hub"),
            ("c", "hub"),
            ("d", "c"),
            ("c", "b"),
        ]);
        let shuffled = mk(&[
            ("c", "b"),
            ("d", "c"),
            ("c", "hub"),
            ("a", "hub"),
            ("b", "hub"),
        ]);
        assert_eq!(
            forward, shuffled,
            "eigencentrality must not depend on edge insertion order"
        );
    }

    /// The point of eigenvector centrality: it ranks a *recursively* central node above
    /// one with higher *raw* in-degree — exactly the case in-degree gets wrong.
    #[test]
    fn eigenvector_centrality_captures_recursive_importance_indegree_misses() {
        // 3 mid-hubs, each depended-upon by 5 leaves; all 3 mids depend on `top`.
        //   in-degree:  each mid = 5, top = 3  → ranks a mid ABOVE top
        //   eigenvector: top inherits the mids' mass → ranks top ABOVE a mid
        let mut g = MemoryGraph::new(0.2, usize::MAX);
        g.upsert_node("top".into(), "top".into(), "".into(), NodeType::Module);
        for m in 0..3 {
            let mid = format!("mid{m}");
            g.upsert_node(mid.clone().into(), mid.clone(), "".into(), NodeType::Module);
            g.add_edge(mid.clone().into(), "top".into(), 1.0, EdgeType::DependsOn);
            for l in 0..5 {
                let leaf = format!("leaf{m}_{l}");
                g.upsert_node(
                    leaf.clone().into(),
                    leaf.clone(),
                    "".into(),
                    NodeType::Module,
                );
                g.add_edge(leaf.into(), mid.clone().into(), 1.0, EdgeType::DependsOn);
            }
        }
        assert!(
            g.node_in_degree(&"mid0".into()) > g.node_in_degree(&"top".into()),
            "in-degree ranks a mid (5) above top (3)"
        );
        let ec = g.eigencentrality();
        let top = ec.get(&"top".into()).copied().unwrap();
        let mid = ec.get(&"mid0".into()).copied().unwrap();
        assert!(
            top > mid,
            "eigenvector ranks top above a mid: top={top} mid={mid}"
        );

        // And the mode actually drives the score: with the eigenvector mode the
        // recursively-central node scores highest; with in-degree the raw-count mid does.
        let score = |g: &MemoryGraph, id: &str| g.compute_node_score(g.node(&id.into()).unwrap());
        g.set_scoring_weights(ScoringWeights {
            w_centrality: 0.5,
            centrality_mode: CentralityMode::Eigenvector,
            ..Default::default()
        });
        assert!(
            score(&g, "top") > score(&g, "mid0"),
            "eigenvector mode lifts top"
        );
        g.set_scoring_weights(ScoringWeights {
            w_centrality: 0.5,
            centrality_mode: CentralityMode::InDegree,
            ..Default::default()
        });
        assert!(
            score(&g, "mid0") >= score(&g, "top"),
            "in-degree mode favors the raw count"
        );
    }

    /// The default mode is elided from the snapshot (byte-identical) and round-trips when set.
    #[test]
    fn centrality_mode_serde_elides_default_and_round_trips() {
        let w = ScoringWeights::default();
        assert_eq!(w.centrality_mode, CentralityMode::InDegree);
        let json = serde_json::to_string(&w).unwrap();
        assert!(
            !json.contains("centrality_mode"),
            "default mode is elided: {json}"
        );
        let eigen = ScoringWeights {
            centrality_mode: CentralityMode::Eigenvector,
            ..Default::default()
        };
        let back: ScoringWeights =
            serde_json::from_str(&serde_json::to_string(&eigen).unwrap()).unwrap();
        assert_eq!(back.centrality_mode, CentralityMode::Eigenvector);
    }

    #[test]
    fn node_state_default_is_elided_and_round_trips() {
        let mut g = MemoryGraph::new(0.2, usize::MAX);
        g.upsert_node("a".into(), "a".into(), "x".into(), NodeType::Module);
        assert_eq!(g.node(&"a".into()).unwrap().state, NodeState::Stable);
        // Default Stable ⇒ no `state` key ⇒ snapshots stay byte-identical.
        assert!(!serde_json::to_string(&g).unwrap().contains("\"state\""));
        g.set_node_state(&"a".into(), NodeState::Working);
        let json = serde_json::to_string(&g).unwrap();
        assert!(json.contains("Working"));
        let back: MemoryGraph = serde_json::from_str(&json).unwrap();
        assert_eq!(back.node(&"a".into()).unwrap().state, NodeState::Working);
    }

    #[test]
    fn orphan_is_excluded_from_centrality_and_set_state_invalidates_cache() {
        let mut g = MemoryGraph::new(0.2, usize::MAX);
        for n in ["hub", "x", "y", "dead"] {
            g.upsert_node(n.into(), n.into(), "".into(), NodeType::Module);
        }
        for s in ["x", "y", "dead"] {
            g.add_edge(s.into(), "hub".into(), 1.0, EdgeType::DependsOn);
        }
        assert_eq!(g.node_in_degree(&"hub".into()), 3); // primes the cache
                                                        // Marking `dead` Orphan must drop it from the structural signal — and the cache,
                                                        // keyed only on edge count, must be invalidated by set_node_state.
        g.set_node_state(&"dead".into(), NodeState::Orphan);
        assert_eq!(
            g.node_in_degree(&"hub".into()),
            2,
            "an orphan dependent no longer inflates the hub's in-degree"
        );
        assert!(
            !g.eigencentrality().contains_key(&"dead".into()),
            "an orphan has no eigenvector centrality"
        );
    }

    #[test]
    fn orphan_evicted_first_and_working_pinned_regardless_of_recency() {
        let mut g = MemoryGraph::new(0.2, usize::MAX);
        for n in ["dead", "work", "stable"] {
            g.upsert_node(n.into(), n.into(), "".into(), NodeType::Module);
        }
        let score = |g: &MemoryGraph, id: &str| g.compute_node_score(g.node(&id.into()).unwrap());
        // `dead` is freshly upserted (recency 1.0) yet, once Orphan, scores below a Stable peer.
        g.set_node_state(&"dead".into(), NodeState::Orphan);
        assert!(
            score(&g, "dead") < score(&g, "stable"),
            "orphan scores below a stable peer even at full recency → evicted first"
        );
        // `work` is Working; after decay it still outscores a decayed Stable peer (pinned).
        g.set_node_state(&"work".into(), NodeState::Working);
        for _ in 0..30 {
            g.tick();
        }
        assert!(
            score(&g, "work") > score(&g, "stable"),
            "Working is pinned above a decayed Stable peer"
        );
    }

    /// Build a graph of `(file, symbol)` defs (with `file:` + `sym:` nodes), for the call tests.
    fn graph_with_defs(defs: &[(&str, &str)]) -> MemoryGraph {
        let mut g = MemoryGraph::new(0.2, usize::MAX);
        for (file, name) in defs {
            let fid = NodeId(format!("file:{file}"));
            g.upsert_node(fid.clone(), (*file).into(), "".into(), NodeType::Module);
            let sid = NodeId(format!("sym:{file}:{name}"));
            g.upsert_node(sid.clone(), (*name).into(), "".into(), NodeType::Symbol);
            g.add_edge(fid, sid, 0.6, EdgeType::Contains);
        }
        g
    }
    fn calls_of(g: &MemoryGraph) -> Vec<(String, String)> {
        let mut v: Vec<(String, String)> = g
            .edges()
            .iter()
            .filter(|e| e.edge_type == EdgeType::Calls)
            .map(|e| (e.source.0.clone(), e.target.0.clone()))
            .collect();
        v.sort();
        v
    }

    fn data_flow_of(g: &MemoryGraph) -> Vec<(String, String)> {
        let mut v: Vec<(String, String)> = g
            .edges()
            .iter()
            .filter(|e| e.edge_type == EdgeType::DataFlow)
            .map(|e| (e.source.0.clone(), e.target.0.clone()))
            .collect();
        v.sort();
        v
    }

    #[test]
    fn resolve_data_flow_links_reader_to_global_unique_static() {
        // `reader()` references `CONFIG`, a uniquely-named const in config.rs → a `DataFlow` edge,
        // the shared-state channel call/import edges miss.
        let mut g = graph_with_defs(&[("src/config.rs", "CONFIG"), ("src/api.rs", "reader")]);
        g.mark_data_symbol(NodeId("sym:src/config.rs:CONFIG".into()));
        g.set_pending_data_refs("src/api.rs", vec![("reader".into(), "CONFIG".into(), 1)]);
        g.resolve_data_flow();
        assert_eq!(
            data_flow_of(&g),
            vec![(
                "sym:src/api.rs:reader".to_string(),
                "sym:src/config.rs:CONFIG".to_string()
            )],
            "reader → CONFIG data-flow edge"
        );
    }

    #[test]
    fn resolve_data_flow_skips_ambiguous_global_name() {
        // `CONFIG` defined in two files (ambiguous) → the reference skips (resolve-uniquely-or-skip).
        let mut g = graph_with_defs(&[
            ("src/a.rs", "CONFIG"),
            ("src/b.rs", "CONFIG"),
            ("src/api.rs", "reader"),
        ]);
        g.mark_data_symbol(NodeId("sym:src/a.rs:CONFIG".into()));
        g.mark_data_symbol(NodeId("sym:src/b.rs:CONFIG".into()));
        g.set_pending_data_refs("src/api.rs", vec![("reader".into(), "CONFIG".into(), 1)]);
        g.resolve_data_flow();
        assert!(
            data_flow_of(&g).is_empty(),
            "an ambiguous global static name skips — no guessed edge"
        );
    }

    #[test]
    fn resolve_data_flow_skips_reference_to_non_data_symbol() {
        // `HELPER` exists as a symbol but is NOT a static/const (it was never marked) — e.g. a
        // SCREAMING-named function. A reference to it must not become a data-flow edge.
        let mut g = graph_with_defs(&[("src/x.rs", "HELPER"), ("src/api.rs", "reader")]);
        // deliberately do NOT mark HELPER as a data symbol
        g.set_pending_data_refs("src/api.rs", vec![("reader".into(), "HELPER".into(), 1)]);
        g.resolve_data_flow();
        assert!(
            data_flow_of(&g).is_empty(),
            "a reference resolving to a non-static/const symbol is not a data-flow edge"
        );
    }

    #[test]
    fn dead_symbols_flags_only_unreferenced() {
        // api.rs: `run` is called by nothing; `helper` is called by `run`. Only `run` is a
        // candidate — `helper` has an incoming Calls edge, and the file's Contains edges (which
        // every symbol has) do not count as references.
        let mut g = graph_with_defs(&[("src/api.rs", "run"), ("src/api.rs", "helper")]);
        g.add_edge(
            NodeId("sym:src/api.rs:run".into()),
            NodeId("sym:src/api.rs:helper".into()),
            0.75,
            EdgeType::Calls,
        );
        assert_eq!(
            g.dead_symbols(),
            vec![NodeId("sym:src/api.rs:run".into())],
            "only the unreferenced symbol is flagged"
        );
    }

    #[test]
    fn paging_from_env_falls_back_to_defaults_when_unset() {
        // With no CCOS_PAGING_THRESHOLD / CCOS_MAX_RESIDENT set, the env constructor is identical
        // to `new` with the given defaults (the env-override convention, default-identical).
        assert_eq!(MemoryGraph::paging_threshold_from_env(0.2), 0.2);
        let g = MemoryGraph::new_from_env(0.2, 123);
        assert_eq!(g.paging_threshold, 0.2);
        assert_eq!(g.max_in_memory_nodes, 123);
    }

    #[test]
    fn resolve_symbol_calls_ladder_global_unique_ambiguous_and_self() {
        // `connect` unique; `shared` defined twice (ambiguous); `run` is the caller.
        let mk = || {
            graph_with_defs(&[
                ("src/db.rs", "connect"),
                ("src/db.rs", "shared"),
                ("src/net.rs", "shared"),
                ("src/api.rs", "run"),
            ])
        };
        let mut g = mk();
        g.set_pending_calls(
            "src/api.rs",
            vec![
                ("run".into(), "connect".into(), 1), // unique → Tier C links
                ("run".into(), "shared".into(), 2),  // ambiguous → skipped
                ("run".into(), "run".into(), 3),     // self → dropped
                ("run".into(), "nope".into(), 4),    // unknown → skipped
            ],
        );
        let added = g.resolve_symbol_calls();
        assert_eq!(added, 1);
        assert_eq!(
            calls_of(&g),
            vec![(
                "sym:src/api.rs:run".to_string(),
                "sym:src/db.rs:connect".to_string()
            )],
            "only the globally-unique callee links; ambiguous/self/unknown are skipped"
        );
        // Determinism: a fresh build + resolve yields the identical edge set.
        let mut g2 = mk();
        g2.set_pending_calls(
            "src/api.rs",
            vec![
                ("run".into(), "connect".into(), 1),
                ("run".into(), "shared".into(), 2),
                ("run".into(), "run".into(), 3),
            ],
        );
        g2.resolve_symbol_calls();
        assert_eq!(calls_of(&g), calls_of(&g2));
    }

    #[test]
    fn resolve_symbol_calls_tier_a_import_disambiguates() {
        // `connect` is ambiguous globally (db + net), so Tier C would skip — but an import in
        // api.rs scopes the call to db::connect, so Tier A resolves it precisely.
        let mut g = graph_with_defs(&[
            ("src/db.rs", "connect"),
            ("src/net.rs", "connect"),
            ("src/api.rs", "run"),
        ]);
        // the `use crate::db::connect;` node the parser would have produced.
        let uid = NodeId("use:src/api.rs:crate::db::connect".into());
        g.upsert_node(uid.clone(), "use".into(), "".into(), NodeType::Module);
        g.add_edge(
            NodeId("file:src/api.rs".into()),
            uid,
            0.5,
            EdgeType::DependsOn,
        );
        g.set_pending_calls("src/api.rs", vec![("run".into(), "connect".into(), 1)]);
        g.resolve_symbol_calls();
        assert_eq!(
            calls_of(&g),
            vec![(
                "sym:src/api.rs:run".to_string(),
                "sym:src/db.rs:connect".to_string()
            )],
            "the import scopes the otherwise-ambiguous call to db::connect (Tier A)"
        );
    }

    #[test]
    fn resolve_symbol_calls_ambiguous_import_scope_skips_never_falls_to_local() {
        // api.rs imports `connect` from BOTH db and net (ambiguous Tier A) AND defines a local
        // `connect`. Tier A is populated-but-ambiguous, so we skip — never linking the same-module
        // local (resolve-uniquely-or-skip). (Only reachable on non-compiling Rust; no guessing.)
        let mut g = graph_with_defs(&[
            ("src/db.rs", "connect"),
            ("src/net.rs", "connect"),
            ("src/api.rs", "connect"), // the same-module local Tier B would otherwise grab
            ("src/api.rs", "run"),
        ]);
        for p in ["crate::db::connect", "crate::net::connect"] {
            let uid = NodeId(format!("use:src/api.rs:{p}"));
            g.upsert_node(uid.clone(), "use".into(), "".into(), NodeType::Module);
            g.add_edge(
                NodeId("file:src/api.rs".into()),
                uid,
                0.5,
                EdgeType::DependsOn,
            );
        }
        g.set_pending_calls("src/api.rs", vec![("run".into(), "connect".into(), 1)]);
        g.resolve_symbol_calls();
        assert!(
            calls_of(&g).is_empty(),
            "ambiguous import scope skips — it does not fall back to the same-module local"
        );
    }

    #[test]
    fn resolve_symbol_calls_qualified_crate_rooted() {
        // `crate::db::connect()` resolves absolutely via its module prefix — no import needed.
        let mut g = graph_with_defs(&[("src/db.rs", "connect"), ("src/api.rs", "run")]);
        g.set_pending_calls(
            "src/api.rs",
            vec![("run".into(), "crate::db::connect".into(), 1)],
        );
        g.resolve_symbol_calls();
        assert_eq!(
            calls_of(&g),
            vec![(
                "sym:src/api.rs:run".to_string(),
                "sym:src/db.rs:connect".to_string()
            )],
            "crate-rooted qualified call resolves to the module's symbol (Slice 2)"
        );
    }

    #[test]
    fn resolve_symbol_calls_qualified_via_import_alias() {
        // `use crate::db;` then `db::connect()` — the leading segment is expanded via the import.
        let mut g = graph_with_defs(&[("src/db.rs", "connect"), ("src/api.rs", "run")]);
        let uid = NodeId("use:src/api.rs:crate::db".into());
        g.upsert_node(uid.clone(), "use".into(), "".into(), NodeType::Module);
        g.add_edge(
            NodeId("file:src/api.rs".into()),
            uid,
            0.5,
            EdgeType::DependsOn,
        );
        g.set_pending_calls("src/api.rs", vec![("run".into(), "db::connect".into(), 1)]);
        g.resolve_symbol_calls();
        assert_eq!(
            calls_of(&g),
            vec![(
                "sym:src/api.rs:run".to_string(),
                "sym:src/db.rs:connect".to_string()
            )],
            "module-alias call resolves by expanding the import to an absolute path (Slice 2)"
        );
    }

    #[test]
    fn resolve_symbol_calls_qualified_external_or_unknown_type_skips() {
        // `std::mem::swap()` (external root) and `Foo::bar()` (a type with no import bringing in
        // `Foo`) expand to no local module — qualified-but-unresolvable skips, never falling back to
        // a same-file local of the same final name (`swap`/`bar` exist locally yet must NOT be
        // linked). (`Self::…` is a different case — it resolves same-module; see the Slice-3 tests.)
        let mut g = graph_with_defs(&[
            ("src/api.rs", "run"),
            ("src/api.rs", "bar"),
            ("src/api.rs", "swap"),
        ]);
        g.set_pending_calls(
            "src/api.rs",
            vec![
                ("run".into(), "std::mem::swap".into(), 1),
                ("run".into(), "Foo::bar".into(), 2),
            ],
        );
        g.resolve_symbol_calls();
        assert!(
            calls_of(&g).is_empty(),
            "qualified external/unknown-type calls skip — no import expands them, no fallback to a local"
        );
    }

    #[test]
    fn resolve_symbol_calls_qualified_no_ancestor_fallback() {
        // `crate::a::b::connect()` but only module `a` (src/a.rs) exists — no file for `a::b` (an
        // inline submodule, or not-yet-ingested). Must SKIP, never falling back to `a`'s `connect`
        // (the inline-submodule / ancestor-fallback false edge the exact-module guard closes).
        let mut g = graph_with_defs(&[("src/a.rs", "connect"), ("src/api.rs", "run")]);
        g.set_pending_calls(
            "src/api.rs",
            vec![("run".into(), "crate::a::b::connect".into(), 1)],
        );
        g.resolve_symbol_calls();
        assert!(
            calls_of(&g).is_empty(),
            "qualified call to a missing submodule skips — no ancestor fallback to a::connect"
        );
    }

    #[test]
    fn resolve_symbol_calls_qualified_variant_or_assoc_collision_skips() {
        // `crate::db::Conn::open()` — `Conn` is a type, not a module, so there is no `db::Conn`
        // file. Must SKIP, not drop the `Conn` tail and link to the free fn `open` in module `db`.
        let mut g = graph_with_defs(&[("src/db.rs", "open"), ("src/api.rs", "run")]);
        g.set_pending_calls(
            "src/api.rs",
            vec![("run".into(), "crate::db::Conn::open".into(), 1)],
        );
        g.resolve_symbol_calls();
        assert!(
            calls_of(&g).is_empty(),
            "associated/variant call skips — the type tail is not collapsed to the module's free fn"
        );
    }

    #[test]
    fn resolve_symbol_calls_self_method_resolves_same_module() {
        // `self.helper()` (captured as `Self::helper`) inside api.rs resolves to api.rs's own
        // `helper` — the enclosing impl's method lives in the caller's module.
        let mut g = graph_with_defs(&[("src/api.rs", "run"), ("src/api.rs", "helper")]);
        g.set_pending_calls("src/api.rs", vec![("run".into(), "Self::helper".into(), 1)]);
        g.resolve_symbol_calls();
        assert_eq!(
            calls_of(&g),
            vec![(
                "sym:src/api.rs:run".to_string(),
                "sym:src/api.rs:helper".to_string()
            )],
            "self.helper() resolves to the same-module method (Slice 3)"
        );
    }

    #[test]
    fn resolve_symbol_calls_self_method_never_uses_imports_or_global() {
        // api.rs has its OWN `connect` AND imports `crate::db::connect` (also a global in db.rs).
        // `self.connect()` must resolve to api.rs's own connect (same module) — NEVER the imported
        // or global db::connect. This isolates the Slice-3 same-module-only guard: if it leaked to
        // Tier A it would pick db's connect; if it fell to the qualified branch it would skip.
        let mut g = graph_with_defs(&[
            ("src/db.rs", "connect"),
            ("src/api.rs", "connect"),
            ("src/api.rs", "run"),
        ]);
        let uid = NodeId("use:src/api.rs:crate::db::connect".into());
        g.upsert_node(uid.clone(), "use".into(), "".into(), NodeType::Module);
        g.add_edge(
            NodeId("file:src/api.rs".into()),
            uid,
            0.5,
            EdgeType::DependsOn,
        );
        g.set_pending_calls(
            "src/api.rs",
            vec![("run".into(), "Self::connect".into(), 1)],
        );
        g.resolve_symbol_calls();
        assert_eq!(
            calls_of(&g),
            vec![(
                "sym:src/api.rs:run".to_string(),
                "sym:src/api.rs:connect".to_string()
            )],
            "self.connect() resolves to api's OWN connect, not the imported or global db::connect"
        );
    }

    #[test]
    fn smaller_failure_decay_reduces_distant_pressure() {
        let pressure_at_c = |decay: f64| {
            let mut g = MemoryGraph::new(0.0, usize::MAX);
            g.set_scoring_weights(ScoringWeights {
                failure_decay: decay,
                ..Default::default()
            });
            for n in ["a", "b", "c"] {
                g.upsert_node(n.into(), n.into(), "".into(), NodeType::Module);
            }
            g.add_edge("a".into(), "b".into(), 1.0, EdgeType::DependsOn);
            g.add_edge("b".into(), "c".into(), 1.0, EdgeType::DependsOn);
            g.set_failure_relevance(&"a".into(), 1.0);
            g.propagate_failure(&"a".into(), 0, 5);
            g.nodes.get(&"c".into()).unwrap().failure_relevance
        };
        assert!(
            pressure_at_c(0.9) > pressure_at_c(0.5),
            "a smaller decay must attenuate failure pressure two hops out"
        );
    }

    #[test]
    fn test_failure_propagation() {
        let mut graph = MemoryGraph::default();
        graph.upsert_node("root".into(), "R".into(), "".into(), NodeType::Module);
        graph.upsert_node("child".into(), "C".into(), "".into(), NodeType::Module);
        graph.add_edge("root".into(), "child".into(), 1.0, EdgeType::DependsOn);
        graph.set_failure_relevance(&"root".into(), 1.0);
        graph.propagate_failure(&"root".into(), 0, 3);
        let child = graph.nodes.get(&"child".into()).unwrap();
        assert!(child.failure_relevance > 0.0);
    }

    #[test]
    fn high_fanout_node_damps_pressure_but_low_fanout_does_not() {
        // Degree-aware damping (FIELD_CAMPAIGN_H.md #2): a focused node (out-degree
        // 1) passes pressure undamped; a hub (out-degree ≫ failure_fanout)
        // distributes it, so each neighbour receives strictly less.
        let mut g = MemoryGraph::new(0.0, usize::MAX);
        g.upsert_node("f".into(), "f".into(), "".into(), NodeType::Module);
        g.upsert_node("t".into(), "t".into(), "".into(), NodeType::Module);
        g.add_edge("f".into(), "t".into(), 1.0, EdgeType::DependsOn);
        g.upsert_node("h".into(), "h".into(), "".into(), NodeType::Module);
        for i in 0..20 {
            let leaf = format!("l{i}");
            g.upsert_node(
                leaf.clone().into(),
                leaf.clone(),
                "".into(),
                NodeType::Module,
            );
            g.add_edge("h".into(), leaf.into(), 1.0, EdgeType::Contains);
        }
        g.set_failure_relevance(&"f".into(), 1.0);
        g.propagate_failure(&"f".into(), 0, 1);
        g.set_failure_relevance(&"h".into(), 1.0);
        g.propagate_failure(&"h".into(), 0, 1);

        let focused = g.nodes.get(&"t".into()).unwrap().failure_relevance;
        let from_hub = g.nodes.get(&"l0".into()).unwrap().failure_relevance;
        assert!(
            from_hub < focused,
            "a high-fanout node must spread less pressure per neighbour: hub={from_hub} focused={focused}"
        );
        // 20 leaves with fanout 6 ⇒ ~6/20 of the undamped amount.
        assert!((from_hub - focused * 6.0 / 20.0).abs() < 1e-9);
    }

    #[test]
    fn degree_damping_preserves_sparse_chain_reach() {
        // a→b→c→d, every out-degree 1: damping is a no-op, so depth-3 propagation
        // still reaches the 3-hop cause d — the property that lets us keep a deep
        // default depth without flooding dense graphs.
        let mut g = MemoryGraph::new(0.0, usize::MAX);
        for n in ["a", "b", "c", "d"] {
            g.upsert_node(n.into(), n.into(), "".into(), NodeType::Module);
        }
        g.add_edge("a".into(), "b".into(), 1.0, EdgeType::DependsOn);
        g.add_edge("b".into(), "c".into(), 1.0, EdgeType::DependsOn);
        g.add_edge("c".into(), "d".into(), 1.0, EdgeType::DependsOn);
        g.set_failure_relevance(&"a".into(), 1.0);
        g.propagate_failure(&"a".into(), 0, 3);
        assert!(
            g.nodes.get(&"d".into()).unwrap().failure_relevance > 0.0,
            "the 3-hop cause must still be pressured on a sparse chain"
        );
    }

    #[test]
    fn test_paging_enforcement() {
        let mut graph = MemoryGraph::new(0.0, 5);
        for i in 0..10 {
            graph.upsert_node(
                NodeId(format!("n{}", i)),
                format!("Node{}", i),
                "x".into(),
                NodeType::Unknown,
            );
        }
        assert!(graph.node_count() <= 5);
    }

    #[test]
    fn eviction_policy_is_wired_into_paging_and_can_override_the_greedy() {
        use crate::eviction_policy::{EvictionPolicy, PagingState, KEEP};
        // Two nodes, cap 1 → exactly one is evicted. `x` scores ~0 (bucket 0),
        // `y` scores ~0.95 (bucket 2). Build at a high cap so setup doesn't
        // auto-page, then tighten the cap and page manually.
        let mut g = MemoryGraph::new(0.2, 100);
        g.upsert_node("x".into(), "x".into(), "x".into(), NodeType::Symbol);
        g.upsert_node("y".into(), "y".into(), "y".into(), NodeType::Symbol);
        {
            let x = g.nodes.get_mut(&NodeId("x".into())).unwrap();
            x.base_importance = 0.0;
            x.failure_relevance = 0.0;
            x.recency = 0.0;
            x.access_count = 1;
        }
        {
            let y = g.nodes.get_mut(&NodeId("y".into())).unwrap();
            y.base_importance = 1.0;
            y.failure_relevance = 1.0;
            y.recency = 1.0;
            y.access_count = 1;
        }
        g.max_in_memory_nodes = 1;

        // Untrained → deterministic greedy: the low-score `x` is evicted.
        let mut greedy = g.clone();
        greedy.enforce_paging();
        assert!(
            greedy.nodes.contains_key(&NodeId("y".into()))
                && !greedy.nodes.contains_key(&NodeId("x".into())),
            "greedy evicts the low-score node x"
        );

        // Train the policy to strongly KEEP x's exact state → x now outranks y,
        // so y is evicted instead. Proves the policy is consulted by enforce_paging.
        let mut trained = g.clone();
        let mut policy = EvictionPolicy::new();
        let x_state = PagingState {
            score: 0,
            recency: 1,
            pressure: 0,
            size: 0,
        };
        policy.q.insert((x_state, KEEP), 100.0);
        trained.set_eviction_policy(policy);
        trained.enforce_paging();
        assert!(
            trained.nodes.contains_key(&NodeId("x".into())),
            "the trained policy protects x"
        );
        assert!(
            !trained.nodes.contains_key(&NodeId("y".into())),
            "the trained policy evicts y instead"
        );
    }

    #[test]
    fn eviction_demotes_to_cold_and_page_in_swaps_it_back() {
        // Build at a high cap so setup does not auto-page, then tighten to 1.
        let mut g = MemoryGraph::new(0.2, 100);
        g.upsert_node("hot".into(), "hot".into(), "x".into(), NodeType::Symbol);
        g.upsert_node("cold".into(), "cold".into(), "y".into(), NodeType::Symbol);
        g.add_edge("hot".into(), "cold".into(), 0.6, EdgeType::Contains);
        {
            let h = g.nodes.get_mut(&NodeId("hot".into())).unwrap();
            h.base_importance = 1.0;
            h.recency = 1.0;
        }
        {
            let c = g.nodes.get_mut(&NodeId("cold".into())).unwrap();
            c.base_importance = 0.0;
            c.recency = 0.0;
        }
        g.max_in_memory_nodes = 1;
        g.enforce_paging();

        // Non-destructive eviction: the victim is DEMOTED to COLD, not dropped.
        assert_eq!(g.node_count(), 1, "resident set capped");
        assert!(g.node(&NodeId("hot".into())).is_some());
        assert!(g.node(&NodeId("cold".into())).is_none(), "out of resident");
        assert!(
            g.is_cold(&NodeId("cold".into())),
            "kept in COLD — nothing lost"
        );
        assert_eq!(g.cold_count(), 1);
        assert_eq!(g.edge_count(), 0, "incident edge archived on demotion");

        // Page it back in: a swap — `cold` returns resident, `hot` demotes.
        assert!(g.page_in(&NodeId("cold".into())));
        assert!(g.node(&NodeId("cold".into())).is_some(), "paged back in");
        assert!(!g.is_cold(&NodeId("cold".into())));
        assert!(
            g.is_cold(&NodeId("hot".into())),
            "the lowest-scored other node swapped out, not the requested one"
        );
        assert_eq!(g.node_count(), 1, "resident set still capped (a swap)");

        // page_in on an id that is not cold is a no-op.
        assert!(!g.page_in(&NodeId("nope".into())));
    }

    #[test]
    fn cold_neighbours_is_symmetric_across_demotion_order() {
        let mut g = MemoryGraph::new(0.2, 100);
        for id in ["a", "b", "c"] {
            g.upsert_node(id.into(), id.into(), "x".into(), NodeType::Symbol);
        }
        g.add_edge("a".into(), "b".into(), 0.6, EdgeType::DependsOn);
        // Demote everything (a target resident set of 0).
        g.max_in_memory_nodes = 0;
        g.enforce_paging();
        assert_eq!(g.node_count(), 0);
        assert_eq!(g.cold_count(), 3);
        // a–b are neighbours regardless of which was archived with the edge; c is isolated.
        assert_eq!(
            g.cold_neighbours(&NodeId("a".into())),
            vec![NodeId("b".into())]
        );
        assert_eq!(
            g.cold_neighbours(&NodeId("b".into())),
            vec![NodeId("a".into())]
        );
        assert!(g.cold_neighbours(&NodeId("c".into())).is_empty());
    }

    // ── slice 3: COLD spill to disk (RAM-bounded content, disk-unbounded) ──────

    fn spill_temp_dir(tag: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "ccos_spill_{}_{}_{}",
            tag,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        p
    }

    #[test]
    fn spill_round_trips_cold_content_losslessly() {
        // Resident cap 1 ⇒ all but one node is demoted to the COLD tier.
        let mut g = MemoryGraph::new(0.2, 1);
        let bodies: Vec<(String, String)> = (0..6)
            .map(|i| {
                (
                    format!("n{i}"),
                    format!("// node {i}\n{}\n", "payload ".repeat(50 + i)),
                )
            })
            .collect();
        for (id, body) in &bodies {
            g.upsert_node(
                id.clone().into(),
                id.clone(),
                body.clone(),
                NodeType::Symbol,
            );
        }
        assert!(
            g.cold_count() >= 4,
            "expected a populated COLD tier, got {}",
            g.cold_count()
        );
        assert!(g.cold_inline_bytes() > 0);

        // Attach a spill store with a tiny budget → coldest content flushes to disk.
        let dir = spill_temp_dir("roundtrip");
        g.attach_cold_spill(&dir, 64).unwrap();
        assert!(
            g.cold_spilled_count() > 0,
            "a 64-byte budget must force at least one spill"
        );
        assert!(
            g.cold_inline_bytes() <= 64,
            "resident COLD content must be within budget, got {}",
            g.cold_inline_bytes()
        );

        // Every body must reconstruct exactly when its node is made resident.
        for (id, body) in &bodies {
            let nid = NodeId(id.clone());
            g.page_in(&nid); // faults the blob back (hash-verified); no-op if resident
            let n = g.node(&nid).expect("node must be resident after page_in");
            assert_eq!(
                &n.content, body,
                "content for {id} must round-trip losslessly"
            );
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn spill_decisions_are_deterministic() {
        fn spilled_set(dir: &std::path::Path) -> Vec<NodeId> {
            let mut g = MemoryGraph::new(0.2, 1);
            for i in 0..6 {
                let id = format!("n{i}");
                g.upsert_node(
                    id.clone().into(),
                    id,
                    format!("body {}", "z".repeat(40 + i)),
                    NodeType::Symbol,
                );
            }
            g.attach_cold_spill(dir, 50).unwrap();
            let mut out: Vec<NodeId> = g.cold_ids().filter(|id| g.is_spilled(id)).collect();
            out.sort();
            out
        }
        let d1 = spill_temp_dir("det1");
        let d2 = spill_temp_dir("det2");
        let a = spilled_set(&d1);
        let b = spilled_set(&d2);
        assert_eq!(
            a, b,
            "identical histories must spill the identical node set"
        );
        assert!(!a.is_empty(), "the budget should have forced some spills");
        std::fs::remove_dir_all(&d1).ok();
        std::fs::remove_dir_all(&d2).ok();
    }

    #[test]
    fn spill_off_by_default_leaves_serialization_unchanged() {
        let mut g = MemoryGraph::new(0.2, 1);
        for i in 0..5 {
            let id = format!("n{i}");
            g.upsert_node(
                id.clone().into(),
                id,
                format!("content {}", "q".repeat(30)),
                NodeType::Symbol,
            );
        }
        assert!(g.cold_count() > 0);
        assert_eq!(
            g.cold_spilled_count(),
            0,
            "nothing spills without an attached store"
        );
        // With no store attached, every `spill` stub is `None` and elided, so the
        // JSON carries no new field — byte-identical to the pre-spill layout.
        let json = serde_json::to_string(&g).unwrap();
        assert!(
            !json.contains("\"spill\""),
            "the default path must not emit any spill stub"
        );
    }

    #[test]
    fn a_tampered_or_detached_spill_blob_is_a_cold_miss() {
        let mut g = MemoryGraph::new(0.2, 1);
        for i in 0..4 {
            let id = format!("n{i}");
            g.upsert_node(
                id.clone().into(),
                id,
                format!("secret-{i} {}", "p".repeat(60)),
                NodeType::Symbol,
            );
        }
        let dir = spill_temp_dir("tamper");
        g.attach_cold_spill(&dir, 0).unwrap(); // budget 0 ⇒ spill all cold content
        let spilled: Vec<NodeId> = g.cold_ids().filter(|id| g.is_spilled(id)).collect();
        assert!(spilled.len() >= 2, "expected several spilled nodes");

        // Tamper: rewrite every blob so its bytes no longer hash to its name.
        for entry in std::fs::read_dir(&dir).unwrap() {
            std::fs::write(entry.unwrap().path(), b"tampered").unwrap();
        }
        let victim = spilled[0].clone();
        assert!(
            !g.page_in(&victim),
            "a tampered blob must fail the hash check → cold-miss"
        );
        assert!(
            g.is_cold(&victim),
            "the victim stays cold, not silently restored"
        );
        assert!(
            g.node(&victim).is_none(),
            "no empty or partial node may become resident on a failed fault"
        );

        // Detaching the store makes the remaining spilled nodes unreachable until
        // the same directory is re-attached.
        g.detach_cold_spill();
        let other = spilled.iter().find(|id| **id != victim).unwrap().clone();
        assert!(!g.page_in(&other), "a detached store is a cold-miss");
        std::fs::remove_dir_all(&dir).ok();
    }

    // ── slice 4: COLD compaction (lossy, observable; the deepest tier) ─────────

    /// A deterministic multi-sentence paragraph so CausalSumm has something to
    /// extract (≥3 sentences ⇒ the summary is strictly shorter).
    fn para(tag: &str, sentences: usize) -> String {
        (0..sentences)
            .map(|i| format!("{tag} clause {i} has a handful of filler words here. "))
            .collect()
    }

    #[test]
    fn compaction_shrinks_the_coldest_tail_and_spares_warmer_cold() {
        // Resident cap 1: at equal scores `enforce_paging` demotes the *smallest*
        // id and keeps the *largest* resident, so after ingesting a,b,c,d the
        // cold tier is {a,b,c} and d is resident. Compaction is coldest-first by
        // id, so a,b compact while c (the largest cold id) is spared once the
        // budget is met.
        let mut g = MemoryGraph::new(0.2, 1);
        let a = para("a", 14);
        let b = para("b", 14);
        let c = para("c", 10);
        g.upsert_node("a".into(), "a".into(), a.clone(), NodeType::ContextBlock);
        g.upsert_node("b".into(), "b".into(), b.clone(), NodeType::ContextBlock);
        g.upsert_node("c".into(), "c".into(), c.clone(), NodeType::ContextBlock);
        g.upsert_node("d".into(), "d".into(), para("d", 6), NodeType::ContextBlock);
        assert_eq!(g.cold_count(), 3);
        assert!(
            g.is_cold(&NodeId("c".into())),
            "c is the warmest (largest-id) cold node"
        );

        let before = g.cold_inline_bytes() + g.cold_spilled_bytes();
        g.set_cold_content_budget(Some(1000));
        let after = g.cold_inline_bytes() + g.cold_spilled_bytes();

        assert!(
            g.cold_compacted_count() >= 1,
            "the budget must force compaction"
        );
        assert!(after < before, "compaction must shrink total cold content");
        assert!(after <= 1000, "compaction must reach the budget here");
        assert!(
            g.is_compacted(&NodeId("a".into())),
            "the coldest entry is compacted"
        );
        assert!(
            !g.is_compacted(&NodeId("c".into())),
            "the warmest cold node is compacted last and spared by the budget"
        );

        // Lossy contract on compacted entries; lossless on the spared one.
        let originals = [("a", &a), ("b", &b), ("c", &c)];
        let cold_ids: Vec<NodeId> = g.cold_ids().collect();
        for cid in cold_ids {
            let orig = originals.iter().find(|(id, _)| *id == cid.0).unwrap().1;
            let was_compacted = g.is_compacted(&cid);
            g.page_in(&cid);
            let content = &g.node(&cid).expect("resident after page_in").content;
            if was_compacted {
                assert!(
                    content.len() < orig.len(),
                    "compacted content must be shorter (lossy)"
                );
                assert_ne!(
                    content, orig,
                    "compacted content is a summary, not the original"
                );
            } else {
                assert_eq!(content, orig, "a spared (warm) cold node stays lossless");
            }
        }
    }

    #[test]
    fn compaction_is_off_by_default_and_lossless() {
        let mut g = MemoryGraph::new(0.2, 1);
        let bodies: Vec<(String, String)> = (0..4)
            .map(|i| (format!("n{i}"), para(&format!("n{i}"), 12)))
            .collect();
        for (id, body) in &bodies {
            g.upsert_node(
                id.clone().into(),
                id.clone(),
                body.clone(),
                NodeType::ContextBlock,
            );
        }
        assert!(g.cold_count() > 0);
        assert_eq!(g.cold_compacted_count(), 0, "no budget ⇒ no compaction");
        // No `compacted` stub on the default path ⇒ serialization unchanged.
        let json = serde_json::to_string(&g).unwrap();
        assert!(
            !json.contains("\"compacted\""),
            "default path must emit no compacted flag"
        );
        // Every cold body is still the verbatim original.
        for (id, body) in &bodies {
            let nid = NodeId(id.clone());
            if g.is_cold(&nid) {
                g.page_in(&nid);
                assert_eq!(
                    &g.node(&nid).unwrap().content,
                    body,
                    "lossless without a budget"
                );
            }
        }
    }

    #[test]
    fn compaction_decisions_are_deterministic() {
        fn compacted_set() -> Vec<NodeId> {
            let mut g = MemoryGraph::new(0.2, 1);
            g.upsert_node(
                "z_warm".into(),
                "z_warm".into(),
                para("zwarm", 10),
                NodeType::ContextBlock,
            );
            for id in ["a", "b", "c"] {
                g.upsert_node(
                    id.into(),
                    id.to_string(),
                    para(id, 14),
                    NodeType::ContextBlock,
                );
            }
            g.set_cold_content_budget(Some(900));
            let mut out: Vec<NodeId> = g.cold_ids().filter(|id| g.is_compacted(id)).collect();
            out.sort();
            out
        }
        let a = compacted_set();
        let b = compacted_set();
        assert_eq!(
            a, b,
            "identical histories must compact the identical node set"
        );
        assert!(
            !a.is_empty(),
            "the budget should have forced some compaction"
        );
    }

    // ── audit pass 4: spill-blob GC (F1) and compaction floor (F5) ────────────

    #[test]
    fn cold_resident_bytes_counts_metadata_and_edges() {
        let build = |with_edge: bool| {
            let mut g = MemoryGraph::new(0.2, usize::MAX);
            g.upsert_node(
                "file:a".into(),
                "file:a".into(),
                "A".repeat(200),
                NodeType::Module,
            );
            g.upsert_node(
                "sym:a:f".into(),
                "sym:a:f".into(),
                "b".repeat(200),
                NodeType::Symbol,
            );
            if with_edge {
                g.add_edge("file:a".into(), "sym:a:f".into(), 0.6, EdgeType::Contains);
            }
            g.max_in_memory_nodes = 0;
            g.enforce_paging();
            g
        };
        let g = build(true);
        assert!(
            g.cold_resident_bytes() >= "file:a".len() + "sym:a:f".len(),
            "counts at least the node ids"
        );
        // The archived edge contributes resident bytes (it stays in RAM).
        assert!(
            build(true).cold_resident_bytes() > build(false).cold_resident_bytes(),
            "an archived edge adds to the resident COLD footprint"
        );
    }

    #[test]
    fn removing_a_spilled_node_reclaims_its_blob() {
        let mut g = MemoryGraph::new(0.2, 1);
        for i in 0..3 {
            let id = format!("n{i}");
            g.upsert_node(
                id.clone().into(),
                id,
                format!("c{i} {}", "p".repeat(80)),
                NodeType::Symbol,
            );
        }
        let dir = spill_temp_dir("gc_remove");
        g.attach_cold_spill(&dir, 0).unwrap(); // budget 0 ⇒ spill all cold content
        let spilled: Vec<NodeId> = g.cold_ids().filter(|id| g.is_spilled(id)).collect();
        assert!(spilled.len() >= 2, "expected several spilled nodes");
        let before = std::fs::read_dir(&dir).unwrap().count();
        let victim = spilled[0].clone();
        g.remove_node(&victim);
        let after = std::fs::read_dir(&dir).unwrap().count();
        assert_eq!(
            after,
            before - 1,
            "removing a spilled node reclaims exactly its (unshared) blob: {before} -> {after}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn gc_keeps_a_blob_shared_by_another_cold_node() {
        // a and b have IDENTICAL content ⇒ one deduplicated blob. Removing a must
        // NOT delete it, because b still references it.
        let mut g = MemoryGraph::new(0.2, 1);
        let shared = format!("shared body {}", "x".repeat(80));
        g.upsert_node("a".into(), "a".into(), shared.clone(), NodeType::Symbol);
        g.upsert_node("b".into(), "b".into(), shared.clone(), NodeType::Symbol);
        g.upsert_node(
            "c".into(),
            "c".into(),
            format!("other {}", "y".repeat(80)),
            NodeType::Symbol,
        );
        let dir = spill_temp_dir("gc_dedup");
        g.attach_cold_spill(&dir, 0).unwrap();
        assert!(
            g.is_spilled(&NodeId("a".into())) && g.is_spilled(&NodeId("b".into())),
            "a and b should both be spilled (and dedup to one blob)"
        );
        g.remove_node(&NodeId("a".into()));
        // b's shared blob must have survived: it still faults back losslessly.
        assert!(
            g.page_in(&NodeId("b".into())),
            "b's shared blob must survive a's removal"
        );
        assert_eq!(g.node(&NodeId("b".into())).unwrap().content, shared);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn compaction_parks_unshrinkable_entries_at_the_floor() {
        let mut g = MemoryGraph::new(0.2, 1);
        // Tiny content that cannot summarise/skeletonise smaller.
        for i in 0..4 {
            let id = format!("n{i}");
            g.upsert_node(id.clone().into(), id, format!("x{i}"), NodeType::Symbol);
        }
        assert!(g.cold_count() >= 3);
        g.set_cold_content_budget(Some(1)); // impossibly tight ⇒ nothing shrinks
        let parked = g.cold_ids().filter(|id| g.is_at_floor(id)).count();
        assert!(
            parked > 0,
            "un-shrinkable cold entries are parked at the floor, not re-tried forever"
        );
        // Nothing was actually compacted (they couldn't shrink), and re-enforcing
        // is idempotent (the parked entries are skipped).
        assert_eq!(g.cold_compacted_count(), 0);
        g.set_cold_content_budget(Some(1));
        assert_eq!(g.cold_ids().filter(|id| g.is_at_floor(id)).count(), parked);
    }

    // ── structural-centrality scoring term (the Gemini-reflection idea) ───────

    #[test]
    fn centrality_is_off_by_default_and_elided() {
        let mut g = MemoryGraph::new(0.2, 100);
        for id in ["hub", "a", "b"] {
            g.upsert_node(id.into(), id.into(), "x".into(), NodeType::Symbol);
        }
        g.add_edge("a".into(), "hub".into(), 1.0, EdgeType::DependsOn);
        g.add_edge("b".into(), "hub".into(), 1.0, EdgeType::DependsOn);
        // With centrality off (the default), the hub (in-degree 2) scores exactly
        // like a leaf (in-degree 0) — the term contributes nothing.
        let hub = g.compute_node_score(g.node(&NodeId("hub".into())).unwrap());
        let a = g.compute_node_score(g.node(&NodeId("a".into())).unwrap());
        assert_eq!(hub, a, "centrality off ⇒ in-degree does not move the score");
        // And the default weights serialize without the new field.
        let json = serde_json::to_string(&g.scoring_weights).unwrap();
        assert!(
            !json.contains("w_centrality"),
            "an off (0.0) centrality weight is elided: {json}"
        );
    }

    #[test]
    fn centrality_boosts_a_hub_over_a_leaf_when_enabled() {
        let mut g = MemoryGraph::new(0.2, 100);
        for id in ["hub", "a", "b", "leaf"] {
            g.upsert_node(id.into(), id.into(), "x".into(), NodeType::Symbol);
        }
        g.add_edge("a".into(), "hub".into(), 1.0, EdgeType::DependsOn);
        g.add_edge("b".into(), "hub".into(), 1.0, EdgeType::DependsOn);
        g.set_scoring_weights(ScoringWeights {
            w_centrality: 0.3,
            ..ScoringWeights::default()
        });
        let hub = g.compute_node_score(g.node(&NodeId("hub".into())).unwrap());
        let leaf = g.compute_node_score(g.node(&NodeId("leaf".into())).unwrap());
        assert!(
            hub > leaf,
            "a hub (in-degree 2) outscores a leaf (in-degree 0): {hub} vs {leaf}"
        );
        // The in-degree cache must track edge changes (keyed on edges.len()): a new
        // incoming edge raises the hub's score further.
        let before = hub;
        g.upsert_node("c".into(), "c".into(), "x".into(), NodeType::Symbol);
        g.add_edge("c".into(), "hub".into(), 1.0, EdgeType::DependsOn);
        let after = g.compute_node_score(g.node(&NodeId("hub".into())).unwrap());
        assert!(
            after > before,
            "a new dependant raises centrality: {after} > {before}"
        );
    }

    #[test]
    fn test_find_cycles_detects_loop() {
        let mut graph = MemoryGraph::default();
        for id in ["a", "b", "c", "d"] {
            graph.upsert_node(id.into(), id.into(), "".into(), NodeType::Module);
        }
        // a -> b -> c -> a is a cycle; d is acyclic.
        graph.add_edge("a".into(), "b".into(), 1.0, EdgeType::DependsOn);
        graph.add_edge("b".into(), "c".into(), 1.0, EdgeType::DependsOn);
        graph.add_edge("c".into(), "a".into(), 1.0, EdgeType::DependsOn);
        graph.add_edge("c".into(), "d".into(), 1.0, EdgeType::DependsOn);

        let cycles = graph.find_cycles();
        assert!(!cycles.is_empty(), "must detect the a->b->c->a cycle");
        assert!(cycles[0].len() >= 3);
    }

    #[test]
    fn test_graph_diff() {
        let mut a = MemoryGraph::default();
        for id in ["x", "y", "z"] {
            a.upsert_node(id.into(), id.into(), "".into(), NodeType::Module);
        }
        a.add_edge("x".into(), "y".into(), 1.0, EdgeType::DependsOn);

        let mut b = MemoryGraph::default();
        for id in ["y", "z", "w"] {
            b.upsert_node(id.into(), id.into(), "".into(), NodeType::Module);
        }
        b.add_edge("y".into(), "z".into(), 1.0, EdgeType::DependsOn);

        let d = a.diff(&b);
        assert_eq!(d.nodes_added, vec![NodeId("w".into())]);
        assert_eq!(d.nodes_removed, vec![NodeId("x".into())]);
        assert_eq!(d.common_nodes, 2); // y, z
        assert_eq!(d.edges_added, 1); // y->z
        assert_eq!(d.edges_removed, 1); // x->y
    }

    #[test]
    fn test_to_dot_and_orphans() {
        let mut graph = MemoryGraph::default();
        graph.upsert_node("a".into(), "A".into(), "".into(), NodeType::Module);
        graph.upsert_node("b".into(), "B".into(), "".into(), NodeType::Symbol);
        graph.upsert_node("lonely".into(), "L".into(), "".into(), NodeType::Unknown);
        graph.add_edge("a".into(), "b".into(), 1.0, EdgeType::Contains);

        let dot = graph.to_dot();
        assert!(dot.starts_with("digraph ccos {"));
        assert!(dot.contains("\"a\" -> \"b\""));

        let orphans = graph.orphan_nodes();
        assert_eq!(orphans.len(), 1);
        assert_eq!(orphans[0].0, "lonely");
    }

    #[test]
    fn test_find_cycles_none_when_acyclic() {
        let mut graph = MemoryGraph::default();
        for id in ["a", "b", "c"] {
            graph.upsert_node(id.into(), id.into(), "".into(), NodeType::Module);
        }
        graph.add_edge("a".into(), "b".into(), 1.0, EdgeType::DependsOn);
        graph.add_edge("b".into(), "c".into(), 1.0, EdgeType::DependsOn);
        assert!(graph.find_cycles().is_empty(), "DAG must have no cycles");
    }

    #[test]
    fn test_context_selection_returns_sorted() {
        let mut graph = MemoryGraph::default();
        for i in 0..5 {
            let mut node = GraphNode {
                id: NodeId(format!("n{}", i)),
                label: format!("N{}", i),
                content: "test".into(),
                node_type: NodeType::Unknown,
                base_importance: (i as f64) * 0.1,
                failure_relevance: 0.0,
                recency: 0.5,
                access_count: 1,
                created_at: 0,
                last_accessed: 0,
                state: NodeState::Stable,
            };
            node.recency = (5 - i) as f64 * 0.1;
            graph.nodes.insert(node.id.clone(), node);
        }
        let selected = graph.select_context_window(1024);
        assert!(!selected.is_empty());
    }

    // ── slice 5: COLD deep-spill (lossless full-entry archive; resident-bounded) ──

    #[test]
    fn pack_adj_round_trips_ids() {
        // Empty, single, multi, and multibyte ids all survive the length-prefixed
        // packing (slice 5c Lever 1).
        for ids in [
            vec![],
            vec!["a"],
            vec!["file:src/x.rs", "sym:src/x.rs:fn_é"],
            vec!["", "x", "longer-id-with-dashes-0123456789"],
        ] {
            let nodes: Vec<NodeId> = ids.iter().map(|s| NodeId(s.to_string())).collect();
            let packed = pack_adj(&nodes);
            assert_eq!(
                unpack_adj(&packed).collect::<Vec<_>>(),
                ids,
                "round-trip {ids:?}"
            );
        }
        // A truncated buffer stops cleanly rather than panicking.
        assert_eq!(
            unpack_adj(&[5, 0, 0, 0, b'a']).collect::<Vec<_>>(),
            Vec::<&str>::new()
        );
    }

    /// Five nodes — a chain a→b→c→d plus a hub `h` linked to all four — demoted to
    /// COLD with a spill store attached but no content/deep pressure yet. The caller
    /// sets a resident budget to drive deep-spill.
    fn cold_region(dir: &std::path::Path) -> MemoryGraph {
        let mut g = MemoryGraph::new(0.2, 100);
        for id in ["a", "b", "c", "d", "h"] {
            g.upsert_node(
                id.into(),
                format!("label::{id}"),
                format!("content of {id} {}", "x".repeat(40)),
                NodeType::Symbol,
            );
        }
        g.add_edge("a".into(), "b".into(), 0.6, EdgeType::DependsOn);
        g.add_edge("b".into(), "c".into(), 0.6, EdgeType::DependsOn);
        g.add_edge("c".into(), "d".into(), 0.6, EdgeType::DependsOn);
        for t in ["a", "b", "c", "d"] {
            g.add_edge("h".into(), t.into(), 0.5, EdgeType::Contains);
        }
        g.attach_cold_spill(dir, usize::MAX).unwrap(); // store attached; no inline-budget spill
        g.max_in_memory_nodes = 0;
        g.enforce_paging(); // demote everything to COLD
        g
    }

    /// Ground-truth cold neighbours: scan the cold tier's actual adjacency (resident
    /// full-entry edges + deep husk ids) directly — what `cold_neighbours` returned
    /// before the reverse index. The `radj`-backed implementation must match this.
    fn reference_cold_neighbours(g: &MemoryGraph, id: &NodeId) -> Vec<NodeId> {
        let mut out: BTreeSet<NodeId> = BTreeSet::new();
        let mut consider = |a: &NodeId, b: &NodeId| {
            let other = if a == id {
                Some(b.clone())
            } else if b == id {
                Some(a.clone())
            } else {
                None
            };
            if let Some(o) = other {
                if o != *id && (g.cold.contains_key(&o) || g.deep_contains(&o)) {
                    out.insert(o);
                }
            }
        };
        for c in g.cold.values() {
            for e in &c.edges {
                consider(&e.source, &e.target);
            }
        }
        for (hid, h) in g.deep_entries() {
            for o in unpack_adj(&h.adj) {
                consider(&hid, &NodeId(o.to_owned()));
            }
        }
        out.into_iter().collect()
    }

    #[test]
    fn radj_cold_neighbours_match_ground_truth_under_random_ops() {
        // Lever 2 brick 8: the reverse-adjacency index must agree with a direct scan
        // of the cold tier after any sequence of demote / page-in / remove.
        let dir = spill_temp_dir("radj_equiv");
        let mut g = MemoryGraph::new(0.2, 6);
        g.attach_cold_spill(&dir, usize::MAX).unwrap();
        let mut seed = 0x1234_5678_9abc_def0u64;
        let mut rng = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) as u32
        };
        for step in 0..500 {
            match rng() % 6 {
                0 | 1 => {
                    let a = format!("n{}", rng() % 12);
                    g.upsert_node(
                        a.clone().into(),
                        format!("l{a}"),
                        format!("c{a} {}", "x".repeat((rng() % 30) as usize)),
                        NodeType::Symbol,
                    );
                    let b = format!("n{}", rng() % 12);
                    if a != b {
                        g.page_in(&a.clone().into());
                        g.page_in(&b.clone().into());
                        g.add_edge(a.into(), b.into(), 0.5, EdgeType::DependsOn);
                    }
                }
                2 => {
                    g.max_in_memory_nodes = (rng() % 5) as usize;
                    g.enforce_paging();
                }
                3 => g.set_cold_resident_budget(Some((rng() % 256) as usize)),
                4 => {
                    g.page_in(&format!("n{}", rng() % 12).into());
                }
                _ => g.remove_node(&format!("n{}", rng() % 12).into()),
            }
            if step % 5 == 0 {
                for id in g.cold_ids().collect::<Vec<_>>() {
                    assert_eq!(
                        g.cold_neighbours(&id),
                        reference_cold_neighbours(&g, &id),
                        "radj mismatch for {} at step {step}",
                        id.0,
                    );
                }
            }
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn deep_tier_survives_a_simulated_crash() {
        // Lever 2 brick 9: deep-spill + flush_cold_tier makes the husk and reverse-
        // adjacency indices durable, so a *fresh* graph re-attaching the same directory
        // recovers the whole deep tier — content, neighbours and all.
        let dir = spill_temp_dir("crash");
        let expected: Vec<(NodeId, Vec<NodeId>)>;
        {
            // First "process": build a cold region, deep-spill it, flush, drop the graph.
            let mut g = cold_region(&dir);
            g.set_cold_resident_budget(Some(0)); // deep-spill every entry
            expected = g
                .cold_ids()
                .collect::<Vec<_>>()
                .into_iter()
                .map(|id| {
                    let n = g.cold_neighbours(&id);
                    (id, n)
                })
                .collect();
            g.flush_cold_tier().unwrap();
            // `g` drops here: the in-RAM write buffers vanish, the directory persists.
        }

        // Second "process": a fresh graph re-attaches the same directory.
        let mut g2 = MemoryGraph::new(0.2, 100);
        g2.attach_cold_spill(&dir, usize::MAX).unwrap();
        assert!(!expected.is_empty(), "expected deep entries");

        // Adjacency and membership recover — checked before any page-in changes the
        // cold set.
        for (id, neighbours) in &expected {
            assert!(g2.is_deep_spilled(id), "{} recovered as deep-spilled", id.0);
            assert_eq!(
                &g2.cold_neighbours(id),
                neighbours,
                "reverse adjacency recovered for {}",
                id.0
            );
        }
        // Content recovers losslessly through the disk trip.
        for (id, _) in &expected {
            assert!(g2.page_in(id), "{} pages back in after the crash", id.0);
            let node = g2.node(id).expect("resident after page-in");
            assert_eq!(
                node.content,
                format!("content of {} {}", id.0, "x".repeat(40)),
                "content recovered losslessly for {}",
                id.0
            );
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn deep_tier_lives_in_the_husk_store() {
        // Lever 2 brick 6: the deep tier IS the on-disk index — every deep-spilled
        // entry is reachable via `deep_get`, counted by `deep_count`, and page-in
        // removes it from the store.
        let dir = spill_temp_dir("husk_auth");
        let mut g = cold_region(&dir);
        let total = g.cold_count();
        g.set_cold_resident_budget(Some(0)); // deep-spill every cold entry

        assert_eq!(
            g.cold_deep_spilled_count(),
            total,
            "all entries deep-spilled"
        );
        for id in g.cold_ids().collect::<Vec<_>>() {
            assert!(g.is_deep_spilled(&id), "{} is in the deep tier", id.0);
            assert!(g.deep_get(&id).is_some(), "husk readable for {}", id.0);
        }

        // Page one back in: it leaves the on-disk tier entirely.
        let some_id = g.cold_ids().next().unwrap();
        g.max_in_memory_nodes = 100;
        assert!(g.page_in(&some_id), "paged back in");
        assert!(!g.is_deep_spilled(&some_id), "left the deep tier");
        assert!(
            g.deep_get(&some_id).is_none(),
            "husk removed from the store"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn deep_spill_round_trips_through_page_in() {
        // `a` stays resident, `b` is demoted then deep-spilled. The a–b edge is
        // archived under `b`, so paging `b` back (with `a` resident) re-links it —
        // proving label, content and the full edge record all survive the disk trip.
        let dir = spill_temp_dir("deep_rt");
        let mut g = MemoryGraph::new(0.2, 100);
        g.upsert_node(
            "a".into(),
            "label::a".into(),
            "alpha body".into(),
            NodeType::Symbol,
        );
        g.upsert_node(
            "b".into(),
            "label::b".into(),
            "beta body padded ".repeat(8),
            NodeType::Symbol,
        );
        // Make `a` the warmer node so `b` is the demotion victim.
        {
            let a = g.nodes.get_mut(&NodeId("a".into())).unwrap();
            a.recency = 1.0;
            a.base_importance = 1.0;
        }
        g.add_edge("a".into(), "b".into(), 0.7, EdgeType::DependsOn);
        g.max_in_memory_nodes = 1;
        g.enforce_paging();
        assert!(g.is_cold(&NodeId("b".into())), "b demoted to COLD");

        g.attach_cold_spill(&dir, usize::MAX).unwrap();
        g.set_cold_resident_budget(Some(0)); // force deep-spill of every cold entry
        assert!(g.is_deep_spilled(&NodeId("b".into())), "b deep-spilled");
        {
            // The full ColdNode is gone from RAM — only a compact husk remains.
            assert!(
                !g.cold.contains_key(&NodeId("b".into())),
                "no full ColdNode kept for a deep entry"
            );
            let h = g.deep_get(&NodeId("b".into())).unwrap();
            assert_eq!(
                unpack_adj(&h.adj).collect::<Vec<_>>(),
                vec!["a"],
                "only the neighbour id kept (packed)"
            );
            assert!(
                h.body.len > 0,
                "whole node archived to a non-empty body blob"
            );
        }

        g.max_in_memory_nodes = 2;
        assert!(g.page_in(&NodeId("b".into())), "faults the deep body back");
        let b = g.node(&NodeId("b".into())).unwrap();
        assert_eq!(b.label, "label::b", "label restored");
        assert_eq!(b.content, "beta body padded ".repeat(8), "content restored");
        assert!(
            g.edges.iter().any(|e| e.source == NodeId("a".into())
                && e.target == NodeId("b".into())
                && e.edge_type == EdgeType::DependsOn
                && (e.weight - 0.7).abs() < 1e-9),
            "the full edge record (weight + type) round-tripped"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn deep_spill_preserves_cold_neighbours() {
        let dir = spill_temp_dir("deep_nbr");
        let mut g = cold_region(&dir);
        let ids: Vec<NodeId> = ["a", "b", "c", "d", "h"]
            .iter()
            .map(|s| NodeId(s.to_string()))
            .collect();
        let before: Vec<Vec<NodeId>> = ids.iter().map(|id| g.cold_neighbours(id)).collect();
        g.set_cold_resident_budget(Some(0)); // deep-spill the whole tier
                                             // Every entry now lives only as a husk (the compact form beats the full
                                             // ColdNode for any entry), so cold_neighbours answers purely from resident
                                             // `adj` ids — and must reproduce the original adjacency exactly.
        assert_eq!(
            g.cold_deep_spilled_count(),
            g.cold_count(),
            "all entries deep-spilled"
        );
        let after: Vec<Vec<NodeId>> = ids.iter().map(|id| g.cold_neighbours(id)).collect();
        assert_eq!(
            before, after,
            "the resident neighbour ids reproduce the cold adjacency exactly"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn deep_spill_off_by_default_leaves_serialization_unchanged() {
        let dir = spill_temp_dir("deep_off");
        let g = cold_region(&dir);
        assert_eq!(g.cold_deep_spilled_count(), 0, "no budget ⇒ no deep-spill");
        // The deep-husk map is empty and elided, so the JSON is byte-identical to a
        // graph that never knew about deep-spill — and still round-trips.
        let json = serde_json::to_string(&g).unwrap();
        assert!(
            !json.contains("cold_deep"),
            "no deep-husk map on the default path"
        );
        let _back: MemoryGraph = serde_json::from_str(&json).unwrap();
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn deep_spill_decisions_are_deterministic() {
        fn deep_set(dir: &std::path::Path) -> Vec<NodeId> {
            let mut g = cold_region(dir);
            g.set_cold_resident_budget(Some(g.cold_resident_bytes() / 3));
            let mut out: Vec<NodeId> = g.cold_ids().filter(|id| g.is_deep_spilled(id)).collect();
            out.sort();
            out
        }
        let d1 = spill_temp_dir("deep_det1");
        let d2 = spill_temp_dir("deep_det2");
        let a = deep_set(&d1);
        let b = deep_set(&d2);
        assert_eq!(a, b, "identical histories deep-spill the identical set");
        assert!(
            !a.is_empty(),
            "the budget should have forced some deep-spills"
        );
        std::fs::remove_dir_all(&d1).ok();
        std::fs::remove_dir_all(&d2).ok();
    }

    #[test]
    fn deep_spill_reduces_resident_bytes_without_dropping_nodes() {
        let dir = spill_temp_dir("deep_res");
        let mut g = cold_region(&dir);
        let before = g.cold_resident_bytes();
        let cold_before = g.cold_count();
        g.set_cold_resident_budget(Some(before / 3));
        assert!(g.cold_deep_spilled_count() > 0, "budget forced deep-spills");
        assert!(
            g.cold_resident_bytes() < before,
            "resident metadata shrank ({} → {})",
            before,
            g.cold_resident_bytes()
        );
        assert_eq!(
            g.cold_count(),
            cold_before,
            "non-destructive — no node dropped"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn deep_spill_compacts_even_a_tiny_isolated_entry() {
        // The whole point of the compact husk (slice 5b): the husk is smaller than a
        // full ColdNode struct, so *every* entry shrinks — even a 1-byte, edge-less,
        // label-less one that slice 5 had to leave at the floor. The floor is gone.
        let dir = spill_temp_dir("deep_floor");
        let mut g = MemoryGraph::new(0.2, 100);
        g.upsert_node(
            "big".into(),
            "label-big".into(),
            "B".repeat(200),
            NodeType::Symbol,
        );
        g.upsert_node("tiny".into(), String::new(), "x".into(), NodeType::Symbol);
        g.attach_cold_spill(&dir, usize::MAX).unwrap();
        g.max_in_memory_nodes = 0;
        g.enforce_paging();
        let before = g.cold_resident_bytes();
        g.set_cold_resident_budget(Some(0));
        assert!(
            g.is_deep_spilled(&NodeId("big".into())),
            "the large entry shrinks"
        );
        assert!(
            g.is_deep_spilled(&NodeId("tiny".into())),
            "and so does the tiny isolated one — no per-entry struct floor any more"
        );
        assert!(
            g.cold_resident_bytes() < before,
            "resident dropped ({} → {})",
            before,
            g.cold_resident_bytes()
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn deep_spill_gc_reclaims_body_blob_on_remove() {
        let dir = spill_temp_dir("deep_gc");
        let mut g = cold_region(&dir);
        g.set_cold_resident_budget(Some(0));
        let victim = g
            .cold_ids()
            .find(|id| g.is_deep_spilled(id))
            .expect("at least one entry deep-spilled");
        let files_before = std::fs::read_dir(&dir).unwrap().count();
        // Removing a deep-spilled node reclaims its body (and content) blobs when no
        // other entry shares them.
        g.remove_node(&victim);
        let files_after = std::fs::read_dir(&dir).unwrap().count();
        assert!(
            files_after < files_before,
            "removing a deep node reclaimed its blob(s) ({files_before} → {files_after})"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn page_in_deep_spilled_with_missing_body_is_a_cold_miss() {
        let dir = spill_temp_dir("deep_miss");
        let mut g = cold_region(&dir);
        g.set_cold_resident_budget(Some(0));
        let bid = NodeId("b".into());
        let body_hash = g.deep_get(&bid).unwrap().body.hash;
        // Delete the on-disk body blob (its filename is the hex of the hash): page_in
        // must refuse — no silent half-restore.
        std::fs::remove_file(dir.join(hex32(&body_hash))).unwrap();
        g.max_in_memory_nodes = 100;
        assert!(!g.page_in(&bid), "missing deep body ⇒ cold-miss");
        assert!(g.is_cold(&bid), "node stays cold, not half-restored");
        std::fs::remove_dir_all(&dir).ok();
    }
}
