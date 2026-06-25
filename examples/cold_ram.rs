//! How much RAM does a COLD entry actually keep resident after its content is
//! spilled to disk? This is the O(N) footprint slice 5 would have to bound — so,
//! per the project's measure-first habit, quantify it before choosing a fix.
//!
//! Builds N realistic nodes (files + symbols + edges), demotes them all to COLD,
//! spills every content blob to disk, then reports the **stuck resident bytes**
//! (`cold_resident_bytes`) vs the **offloaded disk bytes** (`cold_spilled_bytes`),
//! cross-checked against the process's actual **VmRSS** delta.
//!
//! Run: `cargo run --release --example cold_ram`

use ccos::memory::{EdgeType, MemoryGraph, NodeType};
use std::path::Path;

/// Resident set size in KiB from /proc (Linux); 0 if unavailable.
fn vmrss_kb() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmRSS:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|n| n.parse().ok())
        })
        .unwrap_or(0)
}

fn build_cold(files: usize, dir: &Path) -> MemoryGraph {
    // Keep everything resident while wiring nodes + edges (add_edge needs both
    // endpoints resident).
    let mut g = MemoryGraph::new(0.2, usize::MAX);
    for f in 0..files {
        let file_id = format!("file:src/module_{f}.rs");
        g.upsert_node(
            file_id.clone().into(),
            file_id.clone(),
            format!("// module {f} header: {}\n", "context ".repeat(20)),
            NodeType::Module,
        );
        for s in 0..3 {
            let sym = format!("sym:src/module_{f}.rs:function_{s}");
            g.upsert_node(
                sym.clone().into(),
                sym.clone(),
                format!(
                    "pub fn function_{s}() -> u32 {{ {} 0 }}\n",
                    "let _x = 1; ".repeat(12)
                ),
                NodeType::Symbol,
            );
            g.add_edge(file_id.clone().into(), sym.into(), 0.6, EdgeType::Contains);
        }
        if f > 0 {
            let prev = format!("file:src/module_{}.rs", f - 1);
            g.add_edge(
                file_id.clone().into(),
                prev.into(),
                0.5,
                EdgeType::DependsOn,
            );
        }
    }
    // Demote the entire graph to COLD, then spill every content blob to disk.
    g.max_in_memory_nodes = 0;
    g.enforce_paging();
    g.attach_cold_spill(dir, 0).unwrap();
    g
}

fn main() {
    println!("# COLD-tier resident RAM per entry (content spilled to disk)\n");
    println!(
        "{:>8} {:>10} {:>14} {:>13} {:>12} {:>11}",
        "files", "cold", "resident/node", "disk/node", "VmRSS Δ/node", "res:disk"
    );

    for &files in &[2000usize, 8000, 30000] {
        let dir =
            std::env::temp_dir().join(format!("ccos_coldram_{}_{}", files, std::process::id()));
        let rss_before = vmrss_kb();
        let g = build_cold(files, &dir);
        let rss_after = vmrss_kb();

        let n = g.cold_count();
        let resident = g.cold_resident_bytes();
        let disk = g.cold_spilled_bytes();
        let rss_delta = (rss_after.saturating_sub(rss_before)) as usize * 1024;
        let ratio = if disk == 0 {
            0.0
        } else {
            resident as f64 / disk as f64
        };
        println!(
            "{:>8} {:>10} {:>11} B {:>11} B {:>10} B {:>10.2}x",
            files,
            n,
            resident / n.max(1),
            disk / n.max(1),
            rss_delta / n.max(1),
            ratio
        );
        drop(g);
        std::fs::remove_dir_all(&dir).ok();
    }

    // Extrapolate from the largest run: at what node count does the stuck resident
    // metadata reach 1 GiB?
    let dir = std::env::temp_dir().join(format!("ccos_coldram_x_{}", std::process::id()));
    let g = build_cold(30000, &dir);
    let per = g.cold_resident_bytes() / g.cold_count().max(1);
    let gib = 1024usize * 1024 * 1024;
    println!(
        "\nAt ~{} resident bytes/cold-node, the COLD metadata alone hits 1 GiB at ~{} nodes.",
        per,
        gib / per.max(1)
    );
    println!(
        "Disk holds the content; RAM still holds ids+labels+edges+hash-stub per entry — the O(N) slice 5 bounds."
    );
    drop(g);
    std::fs::remove_dir_all(&dir).ok();

    // ── slice 5 payoff: deep-spill bounds that resident metadata ──────────────
    // Archive each cold entry's label + full edges to the same on-disk store,
    // keeping only the neighbour ids resident (lossless — it all faults back on
    // page-in). The measured-dominant edge cost shrinks to ids, not dropped.
    let dir = std::env::temp_dir().join(format!("ccos_coldram_deep_{}", std::process::id()));
    let mut g = build_cold(30000, &dir);
    let before = g.cold_resident_bytes();
    g.set_cold_resident_budget(Some(before / 2)); // ask to halve the resident metadata
    let after = g.cold_resident_bytes();
    println!(
        "\nslice 5 — deep-spill: resident metadata {} B → {} B (−{}%), {} of {} entries archived \
         (label+edges → disk, neighbour ids kept, lossless).",
        before,
        after,
        (before - after) * 100 / before.max(1),
        g.cold_deep_spilled_count(),
        g.cold_count(),
    );
    println!(
        "Honest: the guard skips entries it can't shrink (here the symbols, whose one edge is \
         archived under their file), and the remainder is the irreducible per-entry floor — \
         struct + id + content-hash — that deep-spill leaves resident by design."
    );
    drop(g);
    std::fs::remove_dir_all(&dir).ok();
}
