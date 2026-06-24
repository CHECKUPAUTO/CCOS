use crate::eviction_policy::{
    bucket_pressure, bucket_recency, bucket_score, bucket_size, EvictionPolicy, PagingState, EVICT,
    KEEP,
};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

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

/// Default [`ScoringWeights::failure_fanout`]; also fills the field when an older
/// snapshot (written before it existed) is deserialised.
fn default_failure_fanout() -> f64 {
    6.0
}

impl Default for ScoringWeights {
    fn default() -> Self {
        Self {
            w_base: 0.15,
            w_failure: 0.50,
            w_recency: 0.30,
            w_access: 0.05,
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
            failure_decay: get("CCOS_FAILURE_DECAY", d.failure_decay),
            failure_fanout: get("CCOS_FAILURE_FANOUT", d.failure_fanout),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryGraph {
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
        }
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
        self.edges.retain(|e| &e.source != id && &e.target != id);
    }

    pub fn set_failure_relevance(&mut self, id: &NodeId, relevance: f64) {
        if let Some(node) = self.nodes.get_mut(id) {
            node.failure_relevance = relevance.clamp(0.0, 1.0);
            node.recency = 1.0;
            node.last_accessed = self.clock;
        }
    }

    pub fn compute_node_score(&self, node: &GraphNode) -> f64 {
        let w = &self.scoring_weights;
        let base = node.base_importance * w.w_base;
        let failure = node.failure_relevance * w.w_failure;
        let recency = node.recency * w.w_recency;
        let access = (node.access_count.max(1) as f64).ln() * w.w_access;
        (base + failure + recency + access).clamp(0.0, 1.0)
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
            self.nodes.remove(id);
            self.edges.retain(|e| &e.source != id && &e.target != id);
        }
        // Defensive: guarantee no edge survives pointing at an evicted node.
        self.prune_dangling_edges();
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
        // (crate, intra-module path) → file node.
        let mut index: HashMap<(String, String), NodeId> = HashMap::new();
        for id in self.nodes.keys() {
            if let Some(path) = id.0.strip_prefix("file:") {
                if let Some(km) = crate_and_module(path) {
                    index.insert(km, id.clone());
                }
            }
        }
        let mut to_add: Vec<(NodeId, NodeId, EdgeType)> = Vec::new();

        // (a) imports: importer → defining file.
        for id in self.nodes.keys() {
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
        let entries: Vec<((String, String), NodeId)> =
            index.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
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

/// Resolve an import to the defining file's node id. `crate::`/`self::`/`super::`
/// stay in the importer's crate (and require a real sub-module match); a leading
/// crate name (e.g. `grep_matcher::…`) targets that crate and may resolve to its
/// root (`lib.rs`). External paths like `std::io` match nothing.
fn resolve_use(
    importer_crate: &str,
    usepath: &str,
    index: &HashMap<(String, String), NodeId>,
) -> Option<NodeId> {
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
    let segs: Vec<&str> = rest.split("::").filter(|s| !s.is_empty()).collect();
    // Same-crate imports must hit a real sub-module (len ≥ 1); cross-crate imports
    // may resolve to the crate root (len 0) — depending on the crate as a whole.
    let min_len = usize::from(target_crate == importer_crate);
    (min_len..=segs.len()).rev().find_map(|len| {
        let module = segs[..len].join("::");
        index.get(&(target_crate.clone(), module)).cloned()
    })
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
            };
            node.recency = (5 - i) as f64 * 0.1;
            graph.nodes.insert(node.id.clone(), node);
        }
        let selected = graph.select_context_window(1024);
        assert!(!selected.is_empty());
    }
}
