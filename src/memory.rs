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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryGraph {
    pub nodes: HashMap<NodeId, GraphNode>,
    pub edges: Vec<GraphEdge>,
    pub paging_threshold: f64,
    pub max_in_memory_nodes: usize,
    pub clock: u64,
}

impl MemoryGraph {
    pub fn new(paging_threshold: f64, max_in_memory_nodes: usize) -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
            paging_threshold,
            max_in_memory_nodes,
            clock: 0,
        }
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
        let base = node.base_importance * 0.15;
        let failure = node.failure_relevance * 0.50;
        let recency = node.recency * 0.30;
        let access = (node.access_count.max(1) as f64).ln() * 0.05;
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
        let to_remove: Vec<NodeId> = {
            let mut entries: Vec<(&NodeId, f64)> = self
                .nodes
                .iter()
                .map(|(id, node)| (id, self.compute_node_score(node)))
                .collect();
            // Deterministic eviction: lowest score first, ties broken by node id
            // so replay and snapshot hashes are reproducible regardless of the
            // (randomized) HashMap iteration order.
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

        for (target, weight) in targets {
            let propagation = base_value * weight * 0.8_f64.powi(depth as i32);
            if let Some(node) = self.nodes.get_mut(&target) {
                node.failure_relevance = (node.failure_relevance + propagation).clamp(0.0, 1.0);
                node.recency = 1.0;
                node.last_accessed = self.clock;
            }
            self.propagate_failure(&target, depth + 1, max_depth);
        }
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
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

#[cfg(test)]
mod tests {
    use super::*;

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
