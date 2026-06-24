# Measuring the new recall strategies (hybrid fusion, LSA) — honest findings

> Reproduce: `cargo run --release --example recall_eval` (default INT4 TF-IDF) and
> `cargo run --release --example recall_eval --features learned-embed` (LSA).

This is an **LLM-free** measurement: a synthetic corpus (~60 small files) with
**ground-truth** relevant files, three task types that each isolate where a signal
*should* help, and a hit-rate metric (did the target file land in the recalled
window at a tight 160-token budget?). The point is to check whether the strategies
added this session help **in measurement**, not just in micro unit-tests — and to
report it whether or not it flatters the features.

The three task types:
- **plain** — the query carries the target's own distinctive term (lexical should suffice).
- **decoy+fail** — the query is only *common* words a decoy file also has, so lexical
  points at a decoy; the target is the active **failing** file (isolates the causal vote).
- **synonym** — the query uses a synonym the target file never literally contains
  (isolates the latent/LSA benefit).

## Results

Default — **INT4 TF-IDF**:

| strategy    | plain | decoy+fail | synonym | overall |
|-------------|------:|-----------:|--------:|--------:|
| working_set |  50%  |    100%    |   50%   |   67%   |
| lexical     |  50%  |     0%     |    0%   |   17%   |
| semantic    |  50%  |     0%     |   12%   |   21%   |
| **hybrid**  | 100%  |    62%     |   12%   | **58%** |

Opt-in — **LSA (`learned-embed`)**:

| strategy    | plain | decoy+fail | synonym | overall |
|-------------|------:|-----------:|--------:|--------:|
| working_set |  50%  |    100%    |   50%   |   67%   |
| lexical     |  50%  |     0%     |    0%   |   17%   |
| semantic    |  62%  |     0%     |    0%   |   21%   |
| **hybrid**  | 100%  |    12%     |    0%   | **38%** |

## What this says (the robust signal, not the noise)

1. **Hybrid fusion (slice A) is measurably the best query strategy.** On the default
   embedder it leads every other query strategy overall (58% vs lexical 17%, semantic
   21%) and is the only one that recovers the target in the **decoy+fail** case (62%
   where lexical scores 0%) — the sparse causal vote pulling the active failing file
   into the window. This validates the slice-A design *in measurement*, not just in a
   unit test.

2. **LSA (slice B) does *not* help here — and appears to hurt.** Turning on the
   learned embedder drops hybrid overall from 58% to **38%** (decoy+fail 62%→12%,
   synonym 12%→0%). LSA works in the controlled micro-test (`lsa.rs`), but its
   classic strength is *dense ranking*; CCOS uses the semantic signal only to pick a
   single **entry** node before region expansion, and at corpus scale the rank-48
   truncation over ~60 docs adds enough noise to that one pick that it pollutes the
   RRF fusion. **Honest disposition:** LSA correctly stays **opt-in and off by
   default** — the data says "don't turn it on for this recall architecture yet."
   A genuine win would likely need LSA wired into a *ranking* stage (slice A's fusion
   over more candidates), not entry selection — a future experiment, not a current
   claim.

3. **Lexical is a strong, narrow baseline** — exactly the paper's honest stance. Its
   17% overall here is *by construction*: two of the three task types (decoy, synonym)
   are designed to defeat literal matching, so lexical can only win the plain third.
   This is not "lexical is bad," it's "this benchmark stresses the non-lexical signals."

## Honest limitations of this benchmark

- **Synthetic and small** (~60 files, 24 tasks); absolute numbers are
  benchmark-sensitive — read the *ordering* and *direction*, not the exact percentages.
- **`working_set` is a no-query control** — its `decoy+fail` 100% is just the failure
  cue making the target the single hottest node; its 50% elsewhere is id-ordering luck,
  not a meaningful signal.
- **`plain` lexical sits at 50%, not ~100%** — a region-expansion/budget artifact
  (the tight 160-token window plus `use`-chain region sometimes crowds the target out),
  which is why hybrid's anchor-proximity ranking does better there. It means the plain
  column understates lexical; it does not change the headline (hybrid best, LSA hurts).
- The **synonym** column never cleanly isolates LSA, because the glossary bridge doc
  literally contains the synonym and so intercepts the query as a better lexical/semantic
  match than the target — a real difficulty of testing latent links in an
  entry-selection (not ranking) recall. Treat the synonym column as inconclusive for LSA.

## Bottom line

Hybrid fusion earns its place on the default path; the learned LSA embedder has **not**
earned a recommendation and stays opt-in — measured, not assumed.
