# Pure retrieval challenges RAG — distilled SciRust retrieval, measured

> Reproduce: `cargo run --release --example pure_retrieval_vs_rag`

The goal: take SciRust's *pure* semantic-retrieval platform (`scirust-retrieval`) and use it to
challenge CCOS's existing lexical RAG on relevance — deterministically, pure-Rust, zero FFI — by
implementing its `Encoder` over the embeddings CCOS already owns and **measuring**.

## Why distilled, not linked (the moat decision)

The mission's first instinct was to add `scirust-retrieval` as a `path` dependency. Inspecting the
crate stops that cold: **`scirust-retrieval` depends on `scirust-core`**, whose `Cargo.toml` has
`default = ["rayon"]` and unconditionally pulls `nalgebra` + `ndarray` + `matrixmultiply`. Linking it
would drag `rayon`'s **non-deterministic parallel `f32` reduction order** into CCOS's build — breaking
the sacred `replay == live` bit-exactness *and* the mission's own *"accumulation f32 à ordre fixe, bit
pour bit"* — and would forfeit CCOS's zero-extra-dependency, air-gappable identity. It is the exact
trap the #14 fusion avoided by **distilling** SciRust rather than linking it (and Cargo unifies
features additively, so a `default-features = false` on CCOS's side cannot remove rayon without
editing the scirust repo — which is forbidden).

The retrieval *algorithms*, however, are pure: `index.rs`, `hybrid.rs`, `metrics.rs`, `vector.rs`,
`rerank.rs`, `feedback.rs` reference **no** `scirust_core` and **no** `rayon`. So CCOS reimplements
them in `src/retrieval.rs` — every reduction left-to-right in a single `f32`, every ranking sorted by
score then by an ascending-id tie-break — over CCOS's `TfidfEmbedder`. The oracle tests carry the
**hand-derived** values from SciRust's own `scirust-retrieval` test vectors (e.g. BM25 `cat`/`cat cat`
→ `0.250692` / `0.182322`; nDCG `0.9433884`), so the port is verified, not merely mimicked.

## The components (`ccos::retrieval`, zero new deps)

- **`vector`** — `dot` / `norm` / `cosine` / `normalized`, fixed-order `f32`.
- **`DenseIndex`** — exact brute-force top-k cosine over L2-normalised vectors; deterministic.
- **`Bm25Index`** — classic BM25 (`k1=1.2`, `b=0.75`), IDF-weighted, length-normalised.
- **`reciprocal_rank_fusion`** — fuse incomparable scorers by rank, no calibration.
- **`SemanticRetriever`** (dense) and **`HybridRetriever`** (dense ⊕ BM25 via RRF).
- **`metrics`** — Recall@k, Precision@k, MRR, MAP, nDCG@k.
- **`CcosEncoder`** — the bridge: implements `Encoder` over CCOS's corpus-fitted TF-IDF embedder.

## The measurement (real output of the run)

Eval set = CCOS's own `src/*.rs` (the same corpus + ground truth as `rag_crux`): for each file `A`
with cross-file dependencies, the **query** is `A`'s text and the **relevant** docs are `A`'s true
dependency files (the file→file edges the AST resolved). Three retrievers, scored side by side:

```
files: 61   queries (files with deps): 39   relevant pairs: 141

  metric (%)              Recall@1/5/10 |  Prec@1/5/10  |  nDCG@1/5/10  |   MRR   MAP
  ----------------------------------------------------------------------------------
  ccos RAG (lexical)         24  52  66 |   49  25  18  |   49  50  56  | 0.626 0.491
  pure dense (distilled)     24  52  66 |   49  25  18  |   49  50  56  | 0.626 0.491
  pure hybrid (dense+BM25)   22  52  63 |   46  26  17  |   46  49  54  | 0.615 0.467
```

(Absolute figures are measured on *this revision's* `src/` tree, so they drift as the codebase grows —
re-run the example for the current numbers. What does **not** drift: pure dense equals ccos's RAG to
the digit, and every run is bit-for-bit identical.)

## The honest verdict

- **Pure dense reproduces ccos's RAG bit-for-bit.** It is an exact-cosine index over the *same* TF-IDF
  embedding ccos's lexical RAG uses, so every metric matches to the digit (24/52/66, MRR 0.626). That
  is the point of a faithful distillation: the distilled retriever *is* ccos's retriever — but now as a
  clean, serialisable, auditable `DenseIndex` rather than an ad-hoc cosine loop.
- **Hybrid trades slightly on this task.** Fusing BM25's exact-term/IDF signal *lowers* the scores
  here (Recall@1 24→22, MRR 0.626→0.615). Honest negative result: on *file-dependency* retrieval the
  TF-IDF cosine already captures the lexical overlap, and BM25's rare-term emphasis dilutes rather than
  sharpens it. Hybrid is the right tool for keyword-precise corpora; this structural task is not one.
- **The decisive, un-rowable win is determinism.** Every number above is reproducible **bit for bit**
  (fixed-order `f32`, id-tie-broken ranking, zero RNG, zero generative step). Re-run → identical; there
  is no hallucinating generator between query and result, and the index serialises and audits. A neural
  / generative RAG stage offers none of that.

## Scope & next axes

The dense column equals lexical *because the encoder is TF-IDF* — pure dense over a lexical embedding
cannot out-recall lexical. The genuine semantic lift would come from encoding through CCOS's
**`learned-embed` LSA projection** (synonymy/transitivity TF-IDF cannot see); `CcosEncoder` is the seam
where that swaps in. Two further axes from `scirust-retrieval` remain optional follow-ups, and both
touch `scirust-core` so would be **distilled** the same way if pursued: the contrastive
`ImprovementLoop` (learn a projection from confirmed (query, relevant-doc) pairs and watch Recall@k
climb) and `RetrievalAccess` premium gating (which can ride CCOS's *own* ed25519 license from #29
rather than linking `scirust-license`). The headline this PR establishes: **pure retrieval ties ccos's
RAG on relevance and wins decisively on reproducibility** — and it does so with zero new dependencies.
