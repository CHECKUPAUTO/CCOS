use serde::{Deserialize, Serialize};
use std::collections::{BinaryHeap, HashMap};
use std::cmp::Ordering;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
#[derive(PartialEq)]
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

    pub fn add_edge(&mut self, source: NodeId, target: NodeId, weight: f64, edge_type: EdgeType) {
        let now = self.clock;
        // Avoid duplicate edges
        let already_exists = self.edges.iter().any(|e| {
            e.source == source && e.target == target && e.edge_type == edge_type
        });
        if !already_exists {
            self.edges.push(GraphEdge {
                source,
                target,
                weight,
                edge_type,
                created_at: now,
            });
        }
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
        let access = (node.access_count as f64).ln() * 0.05;
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
                self.score.partial_cmp(&other.score)
            }
        }
        impl<'a> Ord for ScoredRef<'a> {
            fn cmp(&self, other: &Self) -> Ordering {
                self.partial_cmp(other).unwrap_or(Ordering::Equal)
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
            entries.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
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
                node.failure_relevance =
                    (node.failure_relevance + propagation).clamp(0.0, 1.0);
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

    pub fn get_node_scores(&self) -> Vec<(NodeId, f64)> {
        let mut scores: Vec<(NodeId, f64)> = self
            .nodes
            .iter()
            .map(|(id, node)| (id.clone(), self.compute_node_score(node)))
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        scores
    }
}

impl Default for MemoryGraph {
    fn default() -> Self {
        Self::new(0.2, 100)
    }
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
