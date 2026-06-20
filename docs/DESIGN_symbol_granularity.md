# Design — symbol-span granularity (root cause #1 of the real-code failure)

## The measured problem (see `FIELD_CAMPAIGN_H.md`)

On CCOS's own `src/` (32 files, 130 k tokens), `recall around <symptom>` returns a
**single whole-file node** that blows any realistic budget, so coverage is 0/2 deps and
the symptom file itself is absent. Root cause: **every node renders as its entire file.**

The mechanism is one function, `external_memory.rs::content_for` (line ~425):

```rust
fn content_for(&self, node_id: &str, node: &GraphNode) -> String {
    let file_key = format!("file:{}", file_of(node_id));
    self.sources.get(&file_key).cloned()        // <- the WHOLE file, for ANY node
        .unwrap_or_else(|| node.content.clone()) //    (sym/use/mod all inherit it)
}
```

A `sym:` node's own stored `content` is only a label (`"Function render at line 5"`),
because the parser (`parser.rs`) captures a symbol's **start `line` only** — `Symbol` has
no extent. So the file source is the only "real" content available, and every node borrows
all of it. One `sym:memory.rs:MemoryGraph` node = 9 747 tokens.

## The non-obvious finding: granularity must reach the *file* node too

I simulated span granularity over the **real** 973-node region CCOS produced (its actual
scores, depth=1, anchor `file:src/mcp.rs`), brace-matching every symbol's true span. The
model **reproduces the measured current behavior exactly** (2048 budget → 1 node, 7235 tok,
symptom absent, 1/2 deps), which makes its projection credible:

| model | budget | #nodes | symptom in window? | deps covered | files represented |
| ----- | ------ | ------ | ------------------ | ------------ | ----------------- |
| **CURRENT** — every node = whole file | 2048 | 1 | ❌ | 1/2 | 1 |
| **P1** — `sym`=span, **`file`=whole** | 2048 | 1 | ❌ | 1/2 | 1 |
| **P2** — `sym`=span, **`file`=header** | 2048 | **20** | ✅ | **2/2** | **5** |
| **P2** | 4096 | 33 | ✅ | 2/2 | 4 |

**P1 (symbol spans only) changes nothing** on this scenario: the top-ranked node is the
direct-dependency *file* node `file:src/external_memory.rs` (score 0.875, 7235 tok); keeping
file nodes whole means that one node still eats the budget before any cheap symbol fits.
**Only P2 works** — when file nodes are thin (a signature/header unit) so the budget-bearing
units are *all* small, the same region + same scores fits the symptom + both deps + 5 files
in 2048 tokens.

> Takeaway: the fix is not "make symbols granular", it is "**no single node may carry a
> whole file**". Symbols carry their span; files carry a header/index; the agent
> reconstructs from spans.

## Implementation plan

### A. Capture spans (`parser.rs`)
- Add `end_line: usize` to `Symbol` (and `start_line` for clarity; keep `line` as start).
- **Line-based parser** (`extract_symbols`): after a symbol's start line, brace-match
  forward (`{` / `}` counter) to the closing brace for `{}`-bodied items; single-line items
  (`const`/`static`/`type`/`use`) end on `;`. Inherits the parser's existing
  string/comment-brace fragility — acceptable, documented, same risk class as today.
- **syn path** (`syn_ast`, `--features syn-parser`): set `end_line = span().end().line` —
  **free**, exact, because `proc-macro2` is already built with `span-locations`.

### B. Store the span as node content (`parser.rs::update_memory_graph`)
- Pass the `source` into `update_memory_graph` (the caller has it right after `parse_source`).
- For each symbol, slice `lines[start-1 ..= end]` and store **that** as the node's `content`
  (replacing today's `"{:?} {} at line {}"` label).
- `use:` node content = its single import line. `mod:` (inline) = its module body span;
  `pub mod x;` (out-of-line) = the one line.

### C. Thin the file node + fix `content_for` (`external_memory.rs`)
- `content_for`: `sym:`/`use:`/`mod:` → **return `node.content`** (the stored span); do
  **not** look up the whole file.
- `file:` → a **header**: the file path + a generated list of its symbol signatures (first
  line of each symbol span). This is the P2 unit. (Whole-file source stays available in
  `self.sources` for callers that explicitly want it — e.g. a future `open_file` tool — but
  windows never spend budget on it.)
- Add a **dedup guard** in `assemble_window`: if a file's header and several of its symbol
  spans are both selected, don't double-count overlapping lines.

### D. Tests
- `recall around` on a ≥3-file fixture with a large anchor file returns the symptom **and**
  ≥1 real dependency within a 2048 budget (the regression this whole campaign is about).
- Span correctness: a known multi-line `fn` round-trips its body; a single-line `const`
  is one line; nested braces inside a fn don't truncate early.
- The full region's token count is bounded by ~Σ(unique spans), not ~15× (no whole-file
  duplication).

## Scope / effort

- **Files**: `parser.rs` (span capture both paths + content slicing), `external_memory.rs`
  (`content_for` + file-header + dedup), small touch where `update_memory_graph` is called
  to thread `source`. ~150–250 lines incl. tests.
- **Risk**: brace-matching in the line-based parser (mitigated: syn path is exact; the
  fragility equals the parser's current behaviour). Persisted `workspace.ccos` from before
  the change still loads (node content is just shorter on re-ingest).
- **Independent of** the other two root causes (propagation-flood → degree-aware depth;
  hub dominance → inverse-degree/IDF weighting). Those are separate levers with their own
  evidence in `FIELD_CAMPAIGN_H.md`; granularity is the dominant one for the *budget*
  problem and the right first move.

## Cheap interim mitigation (optional, 1 line)

A per-node content cap in `content_for` (truncate any single node to, say, `budget/4`
tokens) would immediately stop one node from blowing the whole window — a band-aid that
buys correctness before the real granularity work lands. Not a substitute for P2.
