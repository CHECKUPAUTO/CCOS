# CCOS research paper (arXiv source)

[`ccos_regions.tex`](ccos_regions.tex) — *Causal Context Regions: A Spatial
Memory Model for Long-Horizon LLM Coding Agents*. Self-contained LaTeX
(`article` class, standard packages, inline `thebibliography`), ready for arXiv
submission.

## Build

```bash
pdflatex ccos_regions.tex
pdflatex ccos_regions.tex   # second pass resolves cross-references
```

Any standard TeX distribution (TeX Live / MikTeX) or the arXiv build system
works; no external `.bib` or non-standard packages are required.

## What it claims (and what it doesn't)

The paper is deliberately explicit about the boundary between proven and
hypothesised results:

- **Proven / measured** (reproducible, LLM-free): the formal region definition
  and causal-distance metric (§4), the determinism + replay theorem (§5), the
  locality evaluation (§7) — region recall 0.97 vs flat 0.35 at ≈48% fewer
  tokens, regions 95.5% internally connected — and the **hypothesis simulation**
  (§8) under a stated retrieval oracle: lexical RAG 0% vs structure-aware regional
  selection 100% on cross-file causal tasks (a strong graph-BFS baseline ties the
  regional method). Regenerate with
  [`../../scripts/region_benchmark.sh`](../../scripts/region_benchmark.sh) and
  `ccos experiment`.
- **Still hypothesised** (a falsifiable protocol, *not yet run*): the
  **real-LLM** comparison against RAG, GraphRAG, MemGPT and LangGraph on
  long-horizon tasks (§9), with explicit hypotheses H1–H3 and threats to validity.
  No numbers are reported for the LLM experiments; the simulation tests only the
  *necessary* (retrieval) condition, not the *sufficient* (generation) one.

See [`../context_regions.md`](../context_regions.md) for the engineering-level
description of the same system.
