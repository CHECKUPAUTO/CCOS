//! Deterministic **eigenvector centrality** over the causal-memory graph.
//!
//! This is the foundational, fully-verifiable first slice of the "spectral"
//! direction (ROADMAP #13: *eigenvector centrality → spectral regions → temporal
//! tensor*). It computes the classic eigenvector-centrality ranking — the dominant
//! eigenvector of the adjacency, `A x = λ x` — by **power iteration**, as a clean,
//! self-contained, dependency-free primitive. Nothing here is wired into the CLI,
//! the scoring path, or the snapshot; it is a pure read-only ranking signal you can
//! call on any [`MemoryGraph`]. The richer pipeline
//! (spectral regions, the temporal tensor, any `scirust` fusion) is **deliberately
//! deferred** to a later design pass and adds no new dependency here.
//!
//! ## Relationship to [`MemoryGraph::eigencentrality`]
//!
//! [`MemoryGraph::eigencentrality`] is the
//! *damped, directed* Katz / PageRank realization (`x ← (1−d)/N + d·Aᵀx`), which stays
//! well-defined on a directed, largely-acyclic code graph. [`eigenvector_centrality`]
//! is the **undirected, undamped** textbook form: it symmetrizes the adjacency and runs
//! pure power iteration to the Perron eigenvector. The two are complementary signals; this
//! module keeps the textbook spectral form isolated so the deferred spectral work can build
//! on a clean `A x = λ x` brick rather than the damped operator.
//!
//! ## Method, and why these choices
//!
//! * **Symmetrization.** Each edge `s → t` is treated as an *undirected* connection:
//!   it contributes to both `A[s][t]` and `A[t][s]` (a parallel edge in either
//!   direction just adds mass). A code dependency graph is largely a **DAG**, so the
//!   *directed* adjacency has spectral radius `0` and a degenerate eigenvector — power
//!   iteration on it collapses to zero. The symmetric adjacency is a non-negative
//!   **symmetric** matrix, so by Perron–Frobenius each connected component has a unique
//!   positive dominant eigenvector, and power iteration converges to it. This is the
//!   standard realization of eigenvector centrality on an otherwise-acyclic graph.
//! * **Spectral shift (`A + I`).** Power iteration is run on `A + I` — a unit self-loop on
//!   every node — not on `A` directly. A symmetrized path / tree is **bipartite**, so its
//!   eigenvalues come in `±` pairs about `0` and plain power iteration on `A` *oscillates*
//!   between the dominant pair without converging. The shift makes every eigenvalue `λ + 1`,
//!   so the largest is strictly positive and unique and the iteration converges monotonically;
//!   crucially `A` and `A + I` share the **same eigenvectors**, so the returned ordering is
//!   exactly that of the adjacency's dominant eigenvector. (This is the same shift the
//!   deferred spectral-region work would use on the Laplacian.)
//! * **Normalization.** The vector is normalized to unit **L2** (Euclidean) norm —
//!   the conventional normalization for an eigenvector (`‖x‖₂ = 1`). It is also the
//!   quantity power iteration naturally preserves, and unlike L1 it never degenerates
//!   when a component carries little mass.
//! * **Determinism.** Node ids are processed in **sorted** order and edges are
//!   accumulated in a **canonical sorted order**, so the floating-point summation order
//!   is a pure function of the graph's *structure*, independent of `HashMap` iteration
//!   or edge-insertion order. Iteration stops at a fixed L2 **tolerance**
//!   ([`CONVERGENCE_TOL`]) or a hard **iteration cap** ([`MAX_ITERS`]), whichever comes
//!   first, so it **always terminates** with byte-identical output run-to-run. The
//!   initial vector is the **uniform** vector `1/√n` (no randomness). The Perron vector
//!   is oriented **non-negative** by a deterministic sign convention, so the sign never
//!   flips between runs.
//!
//! ## Edge cases (all well-defined, never `NaN`/panic)
//!
//! * **Empty graph** → empty map.
//! * **Edgeless graph** (nodes but no edges) → the adjacency is the zero matrix and the
//!   dominant eigenvector is undefined; we return the natural limit, the **uniform**
//!   value `1/√n` for every node (all equal), matching the convention that with no
//!   structure no node is privileged.
//! * **Isolated / dangling node** (degree 0 in a graph that *does* have edges) → score
//!   **`0.0`**: nothing feeds it, so its mass decays and renormalization on the connected
//!   structure leaves it at zero.
//! * **Disconnected components** → handled transparently: power iteration runs over the
//!   whole symmetric adjacency at once, and the single global L2 normalization ranks the
//!   component with the largest dominant eigenvalue highest (each node still gets a
//!   finite, deterministic score).
//!
//! `Orphan` nodes (dead code, [`NodeState::Orphan`]) are
//! excluded from the structural graph — consistent with how the rest of CCOS treats
//! topology — so they receive no entry in the result and their edges drop out.
//!
//! ```
//! use ccos::memory::{EdgeType, MemoryGraph, NodeType};
//! use ccos::spectral::eigenvector_centrality;
//!
//! let mut g = MemoryGraph::new(0.2, usize::MAX);
//! for n in ["hub", "a", "b", "c"] {
//!     g.upsert_node(n.into(), n.into(), String::new(), NodeType::Module);
//! }
//! for leaf in ["a", "b", "c"] {
//!     g.add_edge(leaf.into(), "hub".into(), 1.0, EdgeType::DependsOn);
//! }
//! let c = eigenvector_centrality(&g);
//! // The hub the leaves connect to is the most central node.
//! assert!(c[&"hub".into()] > c[&"a".into()]);
//! ```

use crate::memory::{MemoryGraph, NodeId, NodeState};
use std::collections::BTreeMap;

/// L2 convergence tolerance: power iteration stops once the L2 distance between two
/// successive (normalized) iterates falls below this.
pub const CONVERGENCE_TOL: f64 = 1e-12;

/// Hard cap on power-iteration steps, so the routine **always** terminates
/// deterministically even when the tolerance is never reached.
pub const MAX_ITERS: usize = 1000;

/// Eigenvector centrality of every (non-[`Orphan`](NodeState::Orphan)) node, computed
/// by deterministic **power iteration** on the **symmetrized** adjacency
/// (`A x = λ x`), L2-normalized.
///
/// Returns a [`BTreeMap`] keyed by [`NodeId`] so iteration / equality is stable and
/// results are byte-identical run-to-run. See the [module docs](crate::spectral) for
/// the symmetrization, normalization, and edge-case (empty / edgeless / isolated /
/// disconnected) choices. Pure and side-effect-free: it borrows the graph read-only and
/// never mutates it, the snapshot, or any cache.
///
/// # Examples
///
/// ```
/// use ccos::memory::MemoryGraph;
/// use ccos::spectral::eigenvector_centrality;
///
/// // An empty graph yields an empty ranking.
/// let g = MemoryGraph::new(0.2, usize::MAX);
/// assert!(eigenvector_centrality(&g).is_empty());
/// ```
pub fn eigenvector_centrality(graph: &MemoryGraph) -> BTreeMap<NodeId, f64> {
    // Stable node ordering ⇒ deterministic float accumulation. Orphan (dead) nodes are
    // excluded from the structural graph, exactly as the rest of CCOS treats topology.
    let mut ids: Vec<&NodeId> = graph
        .node_entries()
        .filter(|(_, node)| node.state != NodeState::Orphan)
        .map(|(id, _)| id)
        .collect();
    ids.sort();
    let n = ids.len();
    if n == 0 {
        return BTreeMap::new();
    }

    // id → dense index, over the sorted id list.
    let index: std::collections::HashMap<&NodeId, usize> =
        ids.iter().enumerate().map(|(i, id)| (*id, i)).collect();

    // Build the **symmetrized** adjacency as a sorted edge list of (i, j) index pairs,
    // each undirected edge contributing both orientations. Self-loops are kept (they add
    // to the node's own diagonal, the standard treatment). Edges whose endpoints are not
    // both in `index` (e.g. an endpoint is Orphan) are skipped — they are not part of the
    // structural graph.
    let mut adj: Vec<(usize, usize)> = Vec::with_capacity(graph.edges().len() * 2);
    let mut degree = vec![0u64; n];
    for e in graph.edges() {
        if let (Some(&s), Some(&t)) = (index.get(&e.source), index.get(&e.target)) {
            adj.push((s, t));
            adj.push((t, s));
            degree[s] += 1;
            degree[t] += 1;
        }
    }
    // Canonical order ⇒ the accumulation below is invariant to edge-insertion order, so
    // the result is a pure function of the graph's structure, identical across processes.
    adj.sort_unstable();

    // Edgeless graph: the adjacency is the zero matrix, whose dominant eigenvector is
    // undefined. Return the natural limit — the uniform unit-L2 vector (all nodes equal),
    // i.e. with no structure no node is privileged.
    if adj.is_empty() {
        let uniform = 1.0 / (n as f64).sqrt();
        return ids.into_iter().map(|id| (id.clone(), uniform)).collect();
    }

    // Power iteration from the uniform unit vector on **`A + I`** (a unit self-loop on
    // every node), `x` kept L2-normalized each step; stop when the iterate barely moves
    // (TOL) or at the hard cap (MAX_ITERS). The `+ I` spectral shift is essential, not
    // cosmetic: a symmetrized path/tree adjacency is **bipartite**, whose eigenvalues come
    // in ± pairs about 0, so plain power iteration on `A` *oscillates* between the dominant
    // pair and never settles. Shifting to `A + I` makes every eigenvalue `λ + 1`, so the
    // largest is strictly positive and unique (Perron) — convergence is monotone — while
    // the **eigenvectors are unchanged** (`A` and `A + I` share them), so the centrality
    // *ordering* this returns is exactly that of the adjacency's dominant eigenvector. (A
    // degree-0 node's `A + I` eigenvalue is `1`, strictly below any edged component's, so it
    // already decays toward `0`; we additionally pin it to *exactly* `0` after the loop so
    // the "isolated ⇒ 0" contract is bit-exact, not merely "vanishingly small".)
    let mut x = vec![1.0 / (n as f64).sqrt(); n];
    for _ in 0..MAX_ITERS {
        // y = (A + I)·x: seed with x (the +I diagonal), then add the symmetric adjacency
        // contributions in sorted order (⇒ deterministic summation order).
        let mut y = x.clone();
        for &(i, j) in &adj {
            y[i] += x[j];
        }
        // Re-normalize to unit L2.
        let norm = l2_norm(&y);
        if norm == 0.0 {
            // Unreachable while `x` is non-zero (the `+ I` term copies `x` into `y`), but
            // guard against a zero vector rather than ever dividing by zero.
            break;
        }
        for v in &mut y {
            *v /= norm;
        }
        // Converged once the update barely changes the vector.
        let delta = l2_distance(&x, &y);
        x = y;
        if delta < CONVERGENCE_TOL {
            break;
        }
    }

    // Pin every degree-0 (isolated / dangling) node to **exactly** `0.0`: it carries no
    // edge, so its eigenvector-centrality support is empty. Iteration already drives it to a
    // vanishing residual; zeroing makes the "isolated ⇒ 0" contract bit-exact and removes the
    // tolerance-stop residual from the result.
    for (i, &deg) in degree.iter().enumerate() {
        if deg == 0 {
            x[i] = 0.0;
        }
    }
    // Re-normalize to unit L2 over the surviving (connected) structure, so zeroing the
    // isolates does not leave the vector slightly under unit norm. `adj` is non-empty here,
    // so at least one connected node is non-zero and the norm is positive.
    let norm = l2_norm(&x);
    if norm > 0.0 {
        for v in &mut x {
            *v /= norm;
        }
    }

    // Orient the Perron eigenvector non-negative by a deterministic sign convention: the
    // sum of components is non-negative. (Power iteration on a non-negative matrix already
    // converges to the non-negative Perron vector, but pinning the sign keeps output stable
    // against any floating-point sign drift.)
    if x.iter().sum::<f64>() < 0.0 {
        for v in &mut x {
            *v = -*v;
        }
    }

    ids.into_iter()
        .enumerate()
        .map(|(i, id)| (id.clone(), x[i]))
        .collect()
}

/// L2 (Euclidean) norm of a vector.
fn l2_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// L2 distance between two equal-length vectors.
fn l2_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{EdgeType, MemoryGraph, NodeType};

    /// Build a graph directly (no parser) from node ids and directed edges.
    fn graph(nodes: &[&str], edges: &[(&str, &str)]) -> MemoryGraph {
        let mut g = MemoryGraph::new(0.2, usize::MAX);
        for n in nodes {
            g.upsert_node((*n).into(), (*n).into(), String::new(), NodeType::Module);
        }
        for &(s, t) in edges {
            g.add_edge(s.into(), t.into(), 1.0, EdgeType::DependsOn);
        }
        g
    }

    fn score(c: &BTreeMap<NodeId, f64>, id: &str) -> f64 {
        *c.get(&id.into()).expect("node present in centrality map")
    }

    /// Hand-computable graph: a path `a — b — c`. By symmetry the middle node `b`
    /// (degree 2) must dominate the two endpoints (degree 1), and the endpoints must
    /// tie. This pins the dominant-eigenvector ordering on a graph whose answer is
    /// obvious by inspection.
    #[test]
    fn path_graph_orders_middle_above_endpoints() {
        let c = eigenvector_centrality(&graph(&["a", "b", "c"], &[("a", "b"), ("b", "c")]));
        assert_eq!(c.len(), 3);
        assert!(
            score(&c, "b") > score(&c, "a"),
            "the degree-2 middle node dominates an endpoint"
        );
        assert!(
            score(&c, "b") > score(&c, "c"),
            "the degree-2 middle node dominates the other endpoint"
        );
        assert!(
            (score(&c, "a") - score(&c, "c")).abs() < 1e-9,
            "the two symmetric endpoints have equal centrality"
        );
        // The vector is L2-normalized.
        let norm: f64 = c.values().map(|v| v * v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9, "result is unit-L2 normalized");
        // All scores are finite and non-negative (Perron vector).
        assert!(c.values().all(|v| v.is_finite() && *v >= 0.0));
    }

    /// Star graph: a center connected to many leaves. The center must have strictly the
    /// highest centrality, and (by symmetry) all leaves tie below it.
    #[test]
    fn star_graph_center_is_most_central() {
        let leaves = ["l0", "l1", "l2", "l3", "l4"];
        let mut nodes = vec!["center"];
        nodes.extend_from_slice(&leaves);
        let edges: Vec<(&str, &str)> = leaves.iter().map(|l| ("center", *l)).collect();
        let c = eigenvector_centrality(&graph(&nodes, &edges));

        let center = score(&c, "center");
        for l in leaves {
            assert!(center > score(&c, l), "the star center outranks every leaf");
        }
        // Leaves are symmetric ⇒ all equal.
        let first = score(&c, "l0");
        for l in &leaves[1..] {
            assert!(
                (score(&c, l) - first).abs() < 1e-9,
                "symmetric leaves share one centrality value"
            );
        }
    }

    /// Determinism: two independent runs on the same graph produce **byte-identical**
    /// scores (same key set, same `f64` bit patterns), and the result is invariant to
    /// the *order* edges were inserted.
    #[test]
    fn two_runs_are_byte_identical() {
        let build = || {
            eigenvector_centrality(&graph(
                &["a", "b", "c", "hub"],
                &[("a", "hub"), ("b", "hub"), ("c", "hub"), ("a", "b")],
            ))
        };
        let r1 = build();
        let r2 = build();
        assert_eq!(r1.len(), r2.len());
        for (k, v) in &r1 {
            // Compare the raw bits so this is a true byte-for-byte determinism check.
            assert_eq!(v.to_bits(), r2[k].to_bits(), "score for {k} is identical");
        }

        // Same graph, edges inserted in a different order ⇒ identical result.
        let reordered = eigenvector_centrality(&graph(
            &["a", "b", "c", "hub"],
            &[("a", "b"), ("c", "hub"), ("a", "hub"), ("b", "hub")],
        ));
        for (k, v) in &r1 {
            assert_eq!(
                v.to_bits(),
                reordered[k].to_bits(),
                "result is invariant to edge-insertion order for {k}"
            );
        }
    }

    /// Empty graph → empty result.
    #[test]
    fn empty_graph_is_empty() {
        let g = MemoryGraph::new(0.2, usize::MAX);
        assert!(eigenvector_centrality(&g).is_empty());
    }

    /// Edge-case stress: an edgeless graph, a single isolated node, and a graph mixing a
    /// connected pair with two disconnected isolates. None may produce `NaN` or panic.
    #[test]
    fn isolated_and_disconnected_nodes_are_well_defined() {
        // Edgeless graph: all nodes equal (uniform unit-L2), nothing NaN.
        let edgeless = eigenvector_centrality(&graph(&["a", "b", "c"], &[]));
        assert_eq!(edgeless.len(), 3);
        let uniform = 1.0 / 3.0_f64.sqrt();
        assert!(edgeless
            .values()
            .all(|v| v.is_finite() && (v - uniform).abs() < 1e-9));

        // A single isolated node (no edges anywhere): defined as the uniform value 1/√1 = 1.
        let solo = eigenvector_centrality(&graph(&["only"], &[]));
        assert_eq!(solo.len(), 1);
        assert!((score(&solo, "only") - 1.0).abs() < 1e-9);

        // Connected pair `p — q` plus two fully-isolated nodes `x`, `y`. The isolates get
        // exactly 0 (nothing feeds them); the connected pair is symmetric and positive.
        let mixed = eigenvector_centrality(&graph(&["p", "q", "x", "y"], &[("p", "q")]));
        assert!(mixed.values().all(|v| v.is_finite()), "no NaN anywhere");
        assert_eq!(score(&mixed, "x"), 0.0, "an isolated node scores exactly 0");
        assert_eq!(score(&mixed, "y"), 0.0, "an isolated node scores exactly 0");
        assert!(score(&mixed, "p") > 0.0 && score(&mixed, "q") > 0.0);
        assert!(
            (score(&mixed, "p") - score(&mixed, "q")).abs() < 1e-9,
            "the symmetric connected pair ties"
        );
    }

    /// A directed cycle is, after symmetrization, the regular (every node degree 2)
    /// undirected ring — so every node must receive **exactly equal** centrality. This
    /// confirms the symmetrization is genuine (a purely directed reading would not be
    /// vertex-transitive on a DAG-like input) and that the routine converges cleanly.
    #[test]
    fn symmetrized_cycle_is_uniform() {
        let c = eigenvector_centrality(&graph(
            &["a", "b", "c", "d"],
            &[("a", "b"), ("b", "c"), ("c", "d"), ("d", "a")],
        ));
        let first = score(&c, "a");
        for n in ["b", "c", "d"] {
            assert!(
                (score(&c, n) - first).abs() < 1e-9,
                "every node of a symmetric ring has equal centrality"
            );
        }
        assert!(c.values().all(|v| *v > 0.0));
    }
}
