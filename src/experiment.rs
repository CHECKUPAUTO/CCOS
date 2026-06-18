//! # Hypothesis harness: regional causal memory vs. retrieval baselines
//!
//! We want to test, *without an LLM*, the mechanism on which the research
//! hypothesis rests:
//!
//! > An agent cannot solve a task whose required causal context is absent from
//! > its window. Does **regional causal memory** place that context in the
//! > window on long, multi-file tasks better than relevance-ranked retrieval?
//!
//! This is a **simulation under an explicit, falsifiable oracle**, not an LLM
//! evaluation (see the paper for the still-proposed real-LLM study). We generate
//! synthetic repositories with cross-file *causal chains*, draw tasks of growing
//! **diameter** (how far the required context spreads), and, at an equal token
//! budget, compare four selection strategies:
//!
//! - `rag-dense`   — top-budget nodes by lexical similarity to the query
//!   (classical chunk RAG);
//! - `rag-hybrid`  — similarity blended with the global causal score;
//! - `graphrag-1hop` — the best lexical hit plus its 1-hop graph neighbours
//!   (a graph-expansion baseline);
//! - `ccos-region` — the members of the target's causal region.
//!
//! Success oracle: a task is *solved* iff its required causal set is fully inside
//! the window. We report the success rate and mean coverage per diameter. The
//! generator builds the causal structure independently of any strategy, the RAG
//! baselines are given strong formulations, and we report the regimes where CCOS
//! does **not** win. Everything is seeded and deterministic.

use crate::memory::{EdgeType, MemoryGraph, NodeId, NodeType};
use crate::region_engine::ContextRegionEngine;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::Serialize;
use std::collections::{BTreeSet, VecDeque};

/// Token cost per selected node (matches the rest of CCOS).
const TOKENS_PER_NODE: usize = 128;

/// Configuration for one experiment.
#[derive(Debug, Clone)]
pub struct ExperimentConfig {
    /// RNG seed (determinism).
    pub seed: u64,
    /// Number of tasks to sample.
    pub tasks: usize,
    /// Node budget of the context window (tokens = budget × 128). Sized to hold
    /// roughly one subsystem — far less than the whole repository, so selection
    /// is a genuine constraint.
    pub budget: usize,
    /// Independent subsystems (modular causal clusters) in the repository.
    pub subsystems: usize,
    /// Files per subsystem; also the causal-chain length, so it bounds the
    /// reachable task diameter.
    pub files_per_subsystem: usize,
    /// Decoy (high-score, off-topic) symbols per file — the lure for ranked retrieval.
    pub decoys_per_file: usize,
    /// Task diameters to sweep (causal radius of the required set).
    pub diameters: Vec<u32>,
}

impl Default for ExperimentConfig {
    fn default() -> Self {
        ExperimentConfig {
            seed: 42,
            tasks: 400,
            budget: 30,
            subsystems: 8,
            files_per_subsystem: 9,
            decoys_per_file: 1,
            diameters: vec![1, 2, 3, 4],
        }
    }
}

/// Aggregated outcome for one strategy at one diameter (or overall).
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct StrategyResult {
    pub strategy: String,
    pub tasks: usize,
    pub successes: usize,
    pub success_rate: f32,
    pub mean_coverage: f32,
    pub mean_tokens: f32,
}

/// Full experiment report: per-diameter and overall.
#[derive(Debug, Clone, Serialize)]
pub struct ExperimentReport {
    pub seed: u64,
    pub budget_tokens: usize,
    pub n_tasks: usize,
    /// `(diameter, [results per strategy])`.
    pub per_diameter: Vec<(u32, Vec<StrategyResult>)>,
    pub overall: Vec<StrategyResult>,
}

const STRATEGIES: [&str; 5] = [
    "rag-dense",
    "rag-hybrid",
    "graphrag-1hop",
    "graphrag-bfs",
    "ccos-region",
];

struct Synth {
    graph: MemoryGraph,
    /// Each chain is an ordered list of symbol node ids spanning several files.
    chains: Vec<Vec<String>>,
    /// Lexical tokens per node id (file token + a unique random name token).
    tokens: std::collections::HashMap<String, BTreeSet<String>>,
}

/// Generate a synthetic repository whose *causal* structure (cross-file chains)
/// is deliberately decoupled from its *lexical* structure (random per-symbol
/// names sharing only a file token): the realistic case in which a call/data
/// dependency is causally essential yet lexically dissimilar to the query.
fn generate_repo(cfg: &ExperimentConfig, rng: &mut StdRng) -> Synth {
    let mut graph = MemoryGraph::new(0.0, usize::MAX);
    let mut tokens: std::collections::HashMap<String, BTreeSet<String>> = Default::default();
    let add = |g: &mut MemoryGraph,
               toks: &mut std::collections::HashMap<String, BTreeSet<String>>,
               id: &str,
               file_tok: &str,
               name_tok: &str| {
        g.upsert_node(id.into(), id.into(), String::new(), NodeType::Symbol);
        let mut t = BTreeSet::new();
        t.insert(file_tok.to_string());
        t.insert(name_tok.to_string());
        toks.insert(id.to_string(), t);
    };
    let rand_tok = |rng: &mut StdRng| -> String {
        let n: u32 = rng.gen();
        format!("t{n:08x}")
    };

    // Modular structure: `subsystems` independent clusters, each its own set of
    // files linked by a single cross-file causal chain. There are NO edges
    // between subsystems, so each subsystem is one bounded causal region.
    let mut chains: Vec<Vec<String>> = Vec::new();
    for s in 0..cfg.subsystems {
        let l = cfg.files_per_subsystem;
        for j in 0..l {
            let fid = format!("file:s{s}_f{j}.rs");
            let ftok = format!("s{s}_f{j}");
            add(&mut graph, &mut tokens, &fid, &ftok, &ftok);
        }
        // The causal chain: one symbol per file, each depending on the previous
        // (cross-file). Random names → a chain neighbour shares no lexical token
        // with the target, only an edge.
        let mut chain: Vec<String> = Vec::new();
        for j in 0..l {
            let ftok = format!("s{s}_f{j}");
            let name = rand_tok(rng);
            let id = format!("sym:s{s}_f{j}.rs:{name}");
            add(&mut graph, &mut tokens, &id, &ftok, &name);
            graph.add_edge(
                format!("file:{ftok}.rs").into(),
                id.clone().into(),
                0.6,
                EdgeType::Contains,
            );
            if let Some(prev) = chain.last() {
                graph.add_edge(
                    prev.clone().into(),
                    id.clone().into(),
                    0.9,
                    EdgeType::DependsOn,
                );
            }
            chain.push(id);
        }
        chains.push(chain);
        // Decoys: high access-count (high score), lexically same-file, causally
        // irrelevant — the lure for ranked retrieval.
        for j in 0..l {
            let ftok = format!("s{s}_f{j}");
            for d in 0..cfg.decoys_per_file {
                let name = rand_tok(rng);
                let id = format!("sym:s{s}_f{j}.rs:decoy_{d}_{name}");
                add(&mut graph, &mut tokens, &id, &ftok, &name);
                graph.add_edge(
                    format!("file:{ftok}.rs").into(),
                    id.clone().into(),
                    0.6,
                    EdgeType::Contains,
                );
                for _ in 0..40 {
                    graph.upsert_node(
                        id.clone().into(),
                        id.clone(),
                        String::new(),
                        NodeType::Symbol,
                    );
                }
            }
        }
    }

    Synth {
        graph,
        chains,
        tokens,
    }
}

struct Task {
    target: String,
    required: BTreeSet<String>,
    diameter: u32,
}

/// Draw tasks: pick a chain, a centre and a radius `r` (the diameter); the
/// required causal set is the `±r` window along the chain, spanning up to
/// `2r+1` files.
fn generate_tasks(synth: &Synth, cfg: &ExperimentConfig, rng: &mut StdRng) -> Vec<Task> {
    let mut tasks = Vec::new();
    let usable: Vec<&Vec<String>> = synth.chains.iter().filter(|c| c.len() >= 3).collect();
    if usable.is_empty() {
        return tasks;
    }
    for _ in 0..cfg.tasks {
        let chain = usable[rng.gen_range(0..usable.len())];
        let requested = cfg.diameters[rng.gen_range(0..cfg.diameters.len())];
        // Clamp the radius so the ±rr window fits inside the chain.
        let max_r = ((chain.len() - 1) / 2) as u32;
        let rr = requested.min(max_r).max(1) as usize;
        let i = rng.gen_range(rr..=chain.len() - 1 - rr);
        let required: BTreeSet<String> = chain[i - rr..=i + rr].iter().cloned().collect();
        tasks.push(Task {
            target: chain[i].clone(),
            required,
            diameter: rr as u32, // the actual (clamped) diameter
        });
    }
    tasks
}

/// Jaccard lexical similarity between a node and the query (the target's tokens).
fn sim(synth: &Synth, node: &str, target: &str) -> f32 {
    let empty = BTreeSet::new();
    let a = synth.tokens.get(node).unwrap_or(&empty);
    let b = synth.tokens.get(target).unwrap_or(&empty);
    let inter = a.intersection(b).count();
    let union = a.union(b).count();
    if union == 0 {
        0.0
    } else {
        inter as f32 / union as f32
    }
}

/// The node most lexically similar to the query (ties broken by smallest id).
fn best_lexical_hit(synth: &Synth, target: &str) -> String {
    synth
        .graph
        .nodes
        .keys()
        .map(|k| (sim(synth, &k.0, target), k.0.clone()))
        .max_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.1.cmp(&a.1))
        })
        .map(|(_, id)| id)
        .unwrap_or_else(|| target.to_string())
}

/// Select up to `budget` node ids under one strategy.
fn select(strategy: &str, synth: &Synth, target: &str, budget: usize) -> BTreeSet<String> {
    let g = &synth.graph;
    match strategy {
        "rag-dense" => rank_take(g, budget, |id| {
            (sim(synth, id, target) as f64, id.to_string())
        }),
        "rag-hybrid" => rank_take(g, budget, |id| {
            let s = sim(synth, id, target) as f64;
            let score = g
                .nodes
                .get(&NodeId(id.to_string()))
                .map(|n| g.compute_node_score(n))
                .unwrap_or(0.0);
            (0.5 * s + 0.5 * score, id.to_string())
        }),
        "graphrag-1hop" => {
            // Best lexical hit, then expand by one undirected hop, then fill by sim.
            let seed = best_lexical_hit(synth, target);
            let mut sel: BTreeSet<String> = BTreeSet::new();
            sel.insert(seed.clone());
            for e in &g.edges {
                if e.source.0 == seed {
                    sel.insert(e.target.0.clone());
                } else if e.target.0 == seed {
                    sel.insert(e.source.0.clone());
                }
            }
            // Truncate / fill to budget by similarity.
            if sel.len() > budget {
                let mut v: Vec<String> = sel.into_iter().collect();
                v.sort_by(|a, b| {
                    sim(synth, b, target)
                        .partial_cmp(&sim(synth, a, target))
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| a.cmp(b))
                });
                v.truncate(budget);
                v.into_iter().collect()
            } else {
                let fill = rank_take(g, budget, |id| {
                    (sim(synth, id, target) as f64, id.to_string())
                });
                for id in fill {
                    if sel.len() >= budget {
                        break;
                    }
                    sel.insert(id);
                }
                sel
            }
        }
        "graphrag-bfs" => {
            // Strong graph baseline: unbounded breadth-first expansion from the
            // best lexical hit over undirected edges, taking nodes in BFS order
            // until the budget is full. Seeded at the target (its own best hit),
            // this reaches the whole causal component.
            let seed = best_lexical_hit(synth, target);
            let mut sel: BTreeSet<String> = BTreeSet::new();
            let mut visited: BTreeSet<String> = BTreeSet::new();
            let mut queue: VecDeque<String> = VecDeque::new();
            visited.insert(seed.clone());
            queue.push_back(seed);
            while let Some(n) = queue.pop_front() {
                if sel.len() >= budget {
                    break;
                }
                sel.insert(n.clone());
                let mut nbs: Vec<String> = Vec::new();
                for e in &g.edges {
                    if e.source.0 == n {
                        nbs.push(e.target.0.clone());
                    } else if e.target.0 == n {
                        nbs.push(e.source.0.clone());
                    }
                }
                nbs.sort();
                for nb in nbs {
                    if visited.insert(nb.clone()) {
                        queue.push_back(nb);
                    }
                }
            }
            sel
        }
        "ccos-region" => {
            let mut engine = ContextRegionEngine::new();
            let mut sink = crate::event_log::EventLog::new("exp".into());
            engine.initialize_regions(g, &mut sink);
            let Some(rid) = engine.region_of(target) else {
                return BTreeSet::new();
            };
            let mut members: Vec<String> = engine.regions[&rid].members.clone();
            // Cap by causal score (chain nodes outrank decoys via dependency weight).
            members.sort_by(|a, b| {
                let sa = g
                    .nodes
                    .get(&NodeId(a.clone()))
                    .map(|n| g.compute_node_score(n))
                    .unwrap_or(0.0);
                let sb = g
                    .nodes
                    .get(&NodeId(b.clone()))
                    .map(|n| g.compute_node_score(n))
                    .unwrap_or(0.0);
                sb.partial_cmp(&sa)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.cmp(b))
            });
            members.truncate(budget);
            members.into_iter().collect()
        }
        _ => BTreeSet::new(),
    }
}

/// Rank all nodes by a `(score, tiebreak)` key (descending score, then id) and
/// take the top `budget`.
fn rank_take<F>(g: &MemoryGraph, budget: usize, key: F) -> BTreeSet<String>
where
    F: Fn(&str) -> (f64, String),
{
    let mut scored: Vec<(f64, String)> = g.nodes.keys().map(|k| key(&k.0)).collect();
    scored.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.cmp(&b.1))
    });
    scored.into_iter().take(budget).map(|(_, id)| id).collect()
}

/// For the failure-propagation flavour used by CCOS scoring: mark the target and
/// propagate along edges so chain nodes acquire failure relevance. Returns a
/// graph clone with the propagation applied (keeps tasks independent).
fn with_failure(graph: &MemoryGraph, target: &str, depth: u32) -> MemoryGraph {
    let mut g = graph.clone();
    g.set_failure_relevance(&NodeId(target.to_string()), 0.95);
    g.propagate_failure(&NodeId(target.to_string()), 0, depth);
    g
}

/// Run the full experiment.
pub fn run_experiment(cfg: &ExperimentConfig) -> ExperimentReport {
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let base = generate_repo(cfg, &mut rng);
    let tasks = generate_tasks(&base, cfg, &mut rng);

    // Accumulators: strategy → diameter → (successes, coverage_sum, token_sum, n).
    let mut acc: std::collections::HashMap<(String, u32), (usize, f32, f32, usize)> =
        Default::default();

    for task in &tasks {
        // CCOS sees the failure-propagated graph (its causal signal); the RAG
        // baselines are scored on the same graph for fairness.
        let g = with_failure(&base.graph, &task.target, task.diameter + 1);
        let synth = Synth {
            graph: g,
            chains: base.chains.clone(),
            tokens: base.tokens.clone(),
        };
        for strat in STRATEGIES {
            let sel = select(strat, &synth, &task.target, cfg.budget);
            let hit = task.required.intersection(&sel).count();
            let coverage = hit as f32 / task.required.len() as f32;
            let success = usize::from(hit == task.required.len());
            let e = acc
                .entry((strat.to_string(), task.diameter))
                .or_insert((0, 0.0, 0.0, 0));
            e.0 += success;
            e.1 += coverage;
            e.2 += (sel.len() * TOKENS_PER_NODE) as f32;
            e.3 += 1;
        }
    }

    let mut per_diameter: Vec<(u32, Vec<StrategyResult>)> = Vec::new();
    for &d in &cfg.diameters {
        let mut row = Vec::new();
        for strat in STRATEGIES {
            if let Some((succ, cov, tok, n)) = acc.get(&(strat.to_string(), d)) {
                if *n > 0 {
                    row.push(StrategyResult {
                        strategy: strat.to_string(),
                        tasks: *n,
                        successes: *succ,
                        success_rate: *succ as f32 / *n as f32,
                        mean_coverage: cov / *n as f32,
                        mean_tokens: tok / *n as f32,
                    });
                }
            }
        }
        if !row.is_empty() {
            per_diameter.push((d, row));
        }
    }

    // Overall per strategy.
    let mut overall = Vec::new();
    for strat in STRATEGIES {
        let (mut succ, mut cov, mut tok, mut n) = (0usize, 0.0f32, 0.0f32, 0usize);
        for (&d, _) in cfg.diameters.iter().zip(std::iter::repeat(())) {
            if let Some((s, c, t, k)) = acc.get(&(strat.to_string(), d)) {
                succ += s;
                cov += c;
                tok += t;
                n += k;
            }
        }
        if n > 0 {
            overall.push(StrategyResult {
                strategy: strat.to_string(),
                tasks: n,
                successes: succ,
                success_rate: succ as f32 / n as f32,
                mean_coverage: cov / n as f32,
                mean_tokens: tok / n as f32,
            });
        }
    }

    ExperimentReport {
        seed: cfg.seed,
        budget_tokens: cfg.budget * TOKENS_PER_NODE,
        n_tasks: tasks.len(),
        per_diameter,
        overall,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn experiment_is_deterministic() {
        let cfg = ExperimentConfig {
            tasks: 100,
            ..ExperimentConfig::default()
        };
        let a = run_experiment(&cfg);
        let b = run_experiment(&cfg);
        assert_eq!(a.overall, b.overall, "same seed → identical results");
    }

    #[test]
    fn all_strategies_are_evaluated() {
        let report = run_experiment(&ExperimentConfig {
            tasks: 80,
            ..ExperimentConfig::default()
        });
        assert_eq!(report.overall.len(), 5);
        for r in &report.overall {
            assert!(r.tasks > 0);
            assert!((0.0..=1.0).contains(&r.success_rate));
            assert!((0.0..=1.0).contains(&r.mean_coverage));
        }
    }

    #[test]
    fn region_covers_at_least_as_well_as_dense_rag_overall() {
        // Mechanistic check: on cross-file causal tasks, the causal region should
        // not cover *less* of the required set than lexical top-k retrieval.
        let report = run_experiment(&ExperimentConfig {
            tasks: 300,
            ..ExperimentConfig::default()
        });
        let cov = |name: &str| {
            report
                .overall
                .iter()
                .find(|r| r.strategy == name)
                .map(|r| r.mean_coverage)
                .unwrap()
        };
        assert!(
            cov("ccos-region") >= cov("rag-dense"),
            "region coverage {:.3} must be >= dense-RAG {:.3}",
            cov("ccos-region"),
            cov("rag-dense")
        );
    }
}
