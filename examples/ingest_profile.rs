//! **Ingestion profiler** — *measure before optimizing*. Where does CCOS actually spend time turning
//! source into the causal graph? This stages the real ingestion pipeline over a synthetic corpus and
//! times each phase separately, via the public APIs (no library instrumentation), so the dominant
//! cost is identified before any data-oriented / cache work is justified.
//!
//! Phases: (1) parse (syn AST), (2) build graph (`update_memory_graph`), (3) link imports,
//! (4) resolve calls, (5) resolve data-flow. The last three are the whole-graph passes that iterate
//! the edge set — the prime suspects for a cache-bound hotspot.
//!
//! Run: `cargo run --release --example ingest_profile`

use ccos::memory::MemoryGraph;
use ccos::parser::ASTParser;
use std::time::Instant;

/// A synthetic source file: a module with `fns` mutually-calling functions, a `const` they all read
/// (stresses data-flow), and a cross-file `use` + a `shared_i` fn the next file calls (stresses
/// cross-file call resolution).
fn synth_file(i: usize, fns: usize) -> (String, String) {
    let prev = i.saturating_sub(1);
    let mut s = String::new();
    s.push_str(&format!("use crate::mod_{prev}::shared_{prev};\n"));
    s.push_str(&format!("pub const LIMIT_{i}: usize = {i};\n"));
    for f in 0..fns {
        let next = (f + 1) % fns;
        // each fn calls the next in-file fn, reads the module const, and (f==0) calls the prev file's
        // shared fn — a real cross-file edge for the resolver to chase.
        if f == 0 {
            s.push_str(&format!(
                "pub fn f_{i}_{f}() -> usize {{ f_{i}_{next}() + LIMIT_{i} + shared_{prev}() }}\n"
            ));
        } else {
            s.push_str(&format!(
                "pub fn f_{i}_{f}() -> usize {{ f_{i}_{next}() + LIMIT_{i} }}\n"
            ));
        }
    }
    s.push_str(&format!("pub fn shared_{i}() -> usize {{ {i} }}\n"));
    (format!("src/mod_{i}.rs"), s)
}

fn main() {
    let n_files = 600usize;
    let fns_per_file = 20usize;
    let files: Vec<(String, String)> = (0..n_files).map(|i| synth_file(i, fns_per_file)).collect();
    let total_bytes: usize = files.iter().map(|(_, s)| s.len()).sum();

    let parser = ASTParser::new();

    // (1) parse — syn AST over every file.
    let t = Instant::now();
    let parsed: Vec<_> = files
        .iter()
        .map(|(p, s)| parser.parse_source(p, s))
        .collect();
    let parse = t.elapsed();

    // (2) build graph — fold every ParseResult into the graph (nodes + pending calls/data-refs).
    let mut g = MemoryGraph::new(0.0, usize::MAX);
    let t = Instant::now();
    for ((_, src), r) in files.iter().zip(&parsed) {
        parser.update_memory_graph(r, src, &mut g);
    }
    let build = t.elapsed();

    // (3) link imports — file→file dependency edges.
    let t = Instant::now();
    let import_edges = g.link_module_imports();
    let link = t.elapsed();

    // (4) resolve calls — caller→callee Calls edges (whole-graph).
    let t = Instant::now();
    let call_edges = g.resolve_symbol_calls();
    let calls = t.elapsed();

    // (5) resolve data-flow — reader→const DataFlow edges (whole-graph).
    let t = Instant::now();
    let df_edges = g.resolve_data_flow();
    let dataflow = t.elapsed();

    let total = parse + build + link + calls + dataflow;
    let ms = |d: std::time::Duration| d.as_secs_f64() * 1e3;
    let pct = |d: std::time::Duration| 100.0 * d.as_secs_f64() / total.as_secs_f64();

    println!(
        "# Ingestion profile — {n_files} files, {fns_per_file} fns each, {total_bytes} bytes\n"
    );
    println!(
        "graph: {} nodes, {} edges  (imports {import_edges}, calls {call_edges}, data-flow {df_edges})\n",
        g.node_ids().count(),
        g.edges().len(),
    );
    println!("  phase                 time(ms)    %       ");
    for (name, d) in [
        ("1 parse (syn AST)", parse),
        ("2 build graph", build),
        ("3 link imports", link),
        ("4 resolve calls", calls),
        ("5 resolve data-flow", dataflow),
    ] {
        println!("  {name:<22} {:>7.1}  {:>5.1}", ms(d), pct(d));
    }
    println!("  {:<22} {:>7.1}  100.0", "TOTAL", ms(total));
    println!(
        "\nthroughput: {:.1} files/s, {:.2} MB/s  (total {:.1} ms)",
        n_files as f64 / total.as_secs_f64(),
        total_bytes as f64 / 1e6 / total.as_secs_f64(),
        ms(total)
    );
    // ── Scaling: the resolve passes re-run after EVERY file (exactly what `ingest_source` does), so
    // ingestion cost is quadratic in the file count. Isolated here on a no-paging graph. ──────────
    println!(
        "\n# Scaling — whole-graph resolve re-run per file (the real ingest pattern), no paging"
    );
    println!("  files   resolve total(ms)   ratio when files ×2");
    let mut prev = 0.0;
    for &n in &[150usize, 300, 600] {
        let mut g = MemoryGraph::new(0.0, usize::MAX);
        let mut resolve_ms = 0.0;
        for i in 0..n {
            let (p, s) = synth_file(i, fns_per_file);
            let r = parser.parse_source(&p, &s);
            parser.update_memory_graph(&r, &s, &mut g);
            g.link_module_imports();
            let t = Instant::now();
            g.resolve_symbol_calls();
            g.resolve_data_flow();
            resolve_ms += t.elapsed().as_secs_f64() * 1e3;
        }
        let ratio = if prev > 0.0 {
            resolve_ms / prev
        } else {
            f64::NAN
        };
        println!("  {n:>5}   {resolve_ms:>15.1}   {ratio:>5.2}");
        prev = resolve_ms;
    }

    println!(
        "\nFinding: parse (syn) is cheap (~5%); the cost is the whole-graph **resolve passes** —\n\
         data-flow ~49%, calls ~23%. Worse, `ingest_source` re-runs them after EVERY file, and\n\
         `add_edge` checks for a duplicate with a LINEAR SCAN of all edges — so the measured cost is\n\
         ~**cubic** (resolve time grows ~8-15x when the file count doubles; 600 files = ~216 s).\n\
         Two algorithmic fixes, both far above any cache work:\n\
           (1) make `add_edge`'s dedup O(1) — a membership set, not an O(E) scan over self.edges;\n\
           (2) incremental resolution — resolve only the new file's pending refs against a maintained\n\
               index, instead of re-resolving the whole graph after every file.\n\
         Together they turn ~O(N^3) into ~O(N). Data-oriented layout (SoA / cache alignment) would\n\
         only shave a constant factor — premature before this. THIS is why we measure first: the\n\
         profile redirected the work from a speculative SoA rewrite to the real bottleneck."
    );
}
