//! Pure, deterministic semantic **retrieval** — a dependency-free distillation of SciRust's
//! `scirust-retrieval` pure modules, applied over the embeddings CCOS already owns.
//!
//! **Why distilled, not linked.** `scirust-retrieval` depends on `scirust-core`, whose default
//! features pull `rayon` (non-deterministic parallel `f32` reduction order) plus `nalgebra`/`ndarray`.
//! Linking it would break CCOS's sacred `replay == live` bit-exactness and its zero-extra-dependency,
//! air-gappable identity — the exact reason the #14 fusion *distilled* SciRust rather than linking it.
//! The retrieval algorithms themselves are pure (their own modules use no `scirust-core`, no `rayon`),
//! so we reimplement them here over CCOS's [`TfidfEmbedder`]: every
//! reduction accumulates left-to-right in a single `f32`, every ranking sorts by score then by an
//! ascending-id tie-break, so a run is reproducible **bit for bit** — an auditable alternative to a
//! stochastic / generative RAG stage. The oracle tests carry **hand-derived** values (not captured
//! outputs), matching SciRust's own `scirust-retrieval` test vectors.
//!
//! Layers: [`vector`] primitives → [`DenseIndex`] (exact top-k cosine) and [`Bm25Index`] (lexical) →
//! [`SemanticRetriever`] / [`HybridRetriever`] (dense, or dense⊕BM25 fused by [`reciprocal_rank_fusion`])
//! → [`metrics`] (Recall@k, Precision@k, MRR, MAP, nDCG@k). [`CcosEncoder`] is the bridge: it turns text
//! into a dense vector with CCOS's TF-IDF embedder, so "challenges RAG" becomes a measured number
//! (`examples/pure_retrieval_vs_rag.rs`), not a claim.

use crate::embeddings::{tokenize, TfidfEmbedder};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Deterministic dense-vector primitives. All reductions accumulate left-to-right in a single `f32`,
/// so a run is bit-reproducible.
pub mod vector {
    /// Dot product `Σ aᵢ·bᵢ`, summed in index order.
    pub fn dot(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len(), "dot: length mismatch");
        let mut acc = 0.0f32;
        for (x, y) in a.iter().zip(b) {
            acc += x * y;
        }
        acc
    }

    /// Euclidean (L2) norm `√(Σ aᵢ²)`.
    pub fn norm(a: &[f32]) -> f32 {
        dot(a, a).sqrt()
    }

    /// Cosine similarity in `[-1, 1]`. Returns `0.0` when either operand is the zero vector (not `NaN`).
    pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let (na, nb) = (norm(a), norm(b));
        if na <= 0.0 || nb <= 0.0 {
            return 0.0;
        }
        dot(a, b) / (na * nb)
    }

    /// An L2-normalised copy. The zero vector maps to itself.
    pub fn normalized(a: &[f32]) -> Vec<f32> {
        let n = norm(a);
        if n <= 0.0 {
            return a.to_vec();
        }
        let inv = 1.0 / n;
        a.iter().map(|&x| x * inv).collect()
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn dot_and_norm_match_hand_values() {
            // a·b = 1·4 + 2·5 + 3·6 = 32; |a| = √14.
            let (a, b) = ([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]);
            assert!((dot(&a, &b) - 32.0).abs() < 1e-6);
            assert!((norm(&a) - 14.0_f32.sqrt()).abs() < 1e-6);
        }

        #[test]
        fn cosine_of_known_geometry() {
            assert!((cosine(&[1.0, 0.0], &[3.0, 0.0]) - 1.0).abs() < 1e-6); // identical → 1
            assert!(cosine(&[1.0, 0.0], &[0.0, 5.0]).abs() < 1e-6); // orthogonal → 0
            assert!((cosine(&[1.0, 0.0], &[-2.0, 0.0]) + 1.0).abs() < 1e-6); // opposite → -1
            let c = cosine(&[1.0, 0.0], &[1.0, 1.0]); // 45° → 1/√2
            assert!(
                (c - core::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6,
                "cos {c}"
            );
        }

        #[test]
        fn the_zero_vector_never_produces_nan() {
            assert_eq!(cosine(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
            assert_eq!(normalized(&[0.0, 0.0]), vec![0.0, 0.0]);
        }

        #[test]
        fn normalized_has_unit_norm_and_preserves_direction() {
            let v = normalized(&[3.0, 4.0]); // |[3,4]| = 5 → [0.6, 0.8]
            assert!(
                (v[0] - 0.6).abs() < 1e-6 && (v[1] - 0.8).abs() < 1e-6,
                "{v:?}"
            );
            assert!((norm(&v) - 1.0).abs() < 1e-6);
        }
    }
}

/// A document id paired with a similarity score (higher is more relevant).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Scored {
    /// The document id.
    pub id: u64,
    /// The similarity score.
    pub score: f32,
}

/// Errors from index operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RetrievalError {
    /// A vector's length did not match the index's dimension.
    DimMismatch {
        /// The dimension the index expects.
        expected: usize,
        /// The dimension that was supplied.
        got: usize,
    },
}

impl std::fmt::Display for RetrievalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RetrievalError::DimMismatch { expected, got } => write!(
                f,
                "vector dimension {got} does not match index dimension {expected}"
            ),
        }
    }
}

impl std::error::Error for RetrievalError {}

/// Anything that turns text into a dense embedding vector. Implement this over your own embedding
/// source to drive a [`SemanticRetriever`]; CCOS's bridge is [`CcosEncoder`]. `encode` takes `&mut
/// self` to permit an internal cache; an immutable source simply ignores the mutability.
pub trait Encoder {
    /// The dimension of the vectors this encoder produces.
    fn embedding_dim(&self) -> usize;

    /// Encode one text into a dense vector.
    fn encode(&mut self, text: &str) -> Vec<f32>;

    /// Encode a batch of texts. Defaults to encoding each in turn.
    fn encode_batch(&mut self, texts: &[String]) -> Vec<Vec<f32>> {
        texts.iter().map(|t| self.encode(t)).collect()
    }
}

/// Sort `scored` by descending score with a deterministic ascending-id tie-break, then keep the top
/// `k`. The total order makes the ranking a pure function of the scores (independent of insertion or
/// hash order) — the property that distinguishes this from a stochastic RAG stage.
fn rank_truncate(mut scored: Vec<Scored>, k: usize) -> Vec<Scored> {
    scored.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(core::cmp::Ordering::Equal)
            .then(a.id.cmp(&b.id))
    });
    scored.truncate(k);
    scored
}

/// A flat **exact** dense-retrieval index over `dim`-dimensional embeddings: brute-force top-k by
/// cosine similarity. Vectors are stored L2-normalised, so a query scores against each document by a
/// single dot product, and `search` returns the exact top-k — no approximation, no randomised
/// structure — so every ranking is deterministic and auditable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DenseIndex {
    dim: usize,
    ids: Vec<u64>,
    normed: Vec<Vec<f32>>,
}

impl DenseIndex {
    /// A new empty index for `dim`-dimensional vectors.
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            ids: Vec::new(),
            normed: Vec::new(),
        }
    }

    /// The embedding dimension this index expects.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of indexed documents.
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Whether the index holds no documents.
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Add a document `vector` under `id` (L2-normalised on the way in). Returns
    /// [`RetrievalError::DimMismatch`] if its length is wrong.
    pub fn add(&mut self, id: u64, vector: &[f32]) -> Result<(), RetrievalError> {
        if vector.len() != self.dim {
            return Err(RetrievalError::DimMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }
        self.ids.push(id);
        self.normed.push(vector::normalized(vector));
        Ok(())
    }

    /// Exact top-`k` documents by cosine similarity to `query`, score-descending with an
    /// ascending-id tie-break. Empty when the index is empty, `k == 0`, or the query dimension is wrong.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<Scored> {
        if k == 0 || self.is_empty() || query.len() != self.dim {
            return Vec::new();
        }
        let q = vector::normalized(query);
        let scored = self
            .ids
            .iter()
            .zip(&self.normed)
            .map(|(&id, v)| Scored {
                id,
                score: vector::dot(&q, v),
            })
            .collect();
        rank_truncate(scored, k)
    }
}

/// Tokenise for the lexical (BM25) path: split on non-alphanumeric, lower-case, drop empties. Kept
/// local (rather than [`crate::embeddings::tokenize`]) so the hand-derived BM25 oracle values hold.
fn bm25_tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_lowercase())
        .collect()
}

/// A classic **BM25** lexical index (`k1`, `b`; defaults `1.2` / `0.75`). The deterministic sparse
/// counterpart to [`DenseIndex`]: exact term matching with IDF weighting and document-length
/// normalisation, the half of a hybrid retriever that a dense embedder's smoothing can miss.
#[derive(Debug, Clone)]
pub struct Bm25Index {
    k1: f32,
    b: f32,
    vocab: BTreeMap<String, usize>,
    postings: Vec<Vec<(usize, u32)>>, // term id → [(doc index, term frequency)]
    doc_len: Vec<u32>,
    ids: Vec<u64>,
    total_len: u64,
}

impl Default for Bm25Index {
    fn default() -> Self {
        Self::new(1.2, 0.75)
    }
}

impl Bm25Index {
    /// A new empty index with the given BM25 parameters.
    pub fn new(k1: f32, b: f32) -> Self {
        Self {
            k1,
            b,
            vocab: BTreeMap::new(),
            postings: Vec::new(),
            doc_len: Vec::new(),
            ids: Vec::new(),
            total_len: 0,
        }
    }

    /// Number of indexed documents.
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Whether the index holds no documents.
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Add document `text` under `id`.
    pub fn add(&mut self, id: u64, text: &str) {
        let tokens = bm25_tokenize(text);
        let doc_idx = self.ids.len();
        let mut tf: BTreeMap<String, u32> = BTreeMap::new();
        for tok in &tokens {
            *tf.entry(tok.clone()).or_insert(0) += 1;
        }
        for (term, count) in tf {
            let term_id = match self.vocab.get(&term) {
                Some(&i) => i,
                None => {
                    let i = self.postings.len();
                    self.vocab.insert(term, i);
                    self.postings.push(Vec::new());
                    i
                }
            };
            self.postings[term_id].push((doc_idx, count));
        }
        self.doc_len.push(tokens.len() as u32);
        self.total_len += tokens.len() as u64;
        self.ids.push(id);
    }

    /// Top-`k` documents for `query` by BM25 score, score-descending with an ascending-id tie-break.
    pub fn search(&self, query: &str, k: usize) -> Vec<Scored> {
        if k == 0 || self.is_empty() {
            return Vec::new();
        }
        let n = self.ids.len() as f32;
        let avgdl = self.total_len as f32 / n;

        // Distinct query terms, in first-seen order (deterministic).
        let mut query_terms: Vec<usize> = Vec::new();
        let mut seen: BTreeMap<usize, ()> = BTreeMap::new();
        for tok in bm25_tokenize(query) {
            if let Some(&tid) = self.vocab.get(&tok) {
                if seen.insert(tid, ()).is_none() {
                    query_terms.push(tid);
                }
            }
        }

        let mut acc: BTreeMap<usize, f32> = BTreeMap::new();
        for &tid in &query_terms {
            let df = self.postings[tid].len() as f32;
            let idf = (1.0 + (n - df + 0.5) / (df + 0.5)).ln();
            for &(doc_idx, tf) in &self.postings[tid] {
                let tf = tf as f32;
                let dl = self.doc_len[doc_idx] as f32;
                let denom = tf + self.k1 * (1.0 - self.b + self.b * dl / avgdl);
                *acc.entry(doc_idx).or_insert(0.0) += idf * tf * (self.k1 + 1.0) / denom;
            }
        }

        let scored = acc
            .into_iter()
            .map(|(doc_idx, score)| Scored {
                id: self.ids[doc_idx],
                score,
            })
            .collect();
        rank_truncate(scored, k)
    }
}

/// **Reciprocal-rank fusion** of several rankings: each id scores `Σ 1/(rrf_k + rank + 1)` over the
/// rankings it appears in (rank 0-based), then the fused list is the score-descending top-`k`. Fuses
/// incomparable scorers (dense cosine + BM25) by *rank*, so no score calibration is needed.
pub fn reciprocal_rank_fusion(rankings: &[Vec<u64>], rrf_k: f32, k: usize) -> Vec<Scored> {
    let mut scores: BTreeMap<u64, f32> = BTreeMap::new();
    for ranking in rankings {
        for (rank, &id) in ranking.iter().enumerate() {
            *scores.entry(id).or_insert(0.0) += 1.0 / (rrf_k + rank as f32 + 1.0);
        }
    }
    let scored = scores
        .into_iter()
        .map(|(id, score)| Scored { id, score })
        .collect();
    rank_truncate(scored, k)
}

/// End-to-end **dense** retriever: an [`Encoder`] feeding a [`DenseIndex`].
pub struct SemanticRetriever<E: Encoder> {
    encoder: E,
    index: DenseIndex,
}

impl<E: Encoder> SemanticRetriever<E> {
    /// Build a retriever over `encoder`; the index dimension is taken from it.
    pub fn new(encoder: E) -> Self {
        let dim = encoder.embedding_dim();
        Self {
            encoder,
            index: DenseIndex::new(dim),
        }
    }

    /// Encode `text` and add it to the index under `id`.
    pub fn index_text(&mut self, id: u64, text: &str) -> Result<(), RetrievalError> {
        let v = self.encoder.encode(text);
        self.index.add(id, &v)
    }

    /// Number of indexed documents.
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Whether nothing has been indexed yet.
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Encode `query` and return the exact top-`k` documents by similarity.
    pub fn retrieve(&mut self, query: &str, k: usize) -> Vec<Scored> {
        let q = self.encoder.encode(query);
        self.index.search(&q, k)
    }

    /// Borrow the underlying index (for inspection).
    pub fn index(&self) -> &DenseIndex {
        &self.index
    }
}

/// **Hybrid** retriever: dense ([`DenseIndex`]) and lexical ([`Bm25Index`]) candidate lists fused by
/// [`reciprocal_rank_fusion`]. Each side retrieves a pool of `max(k·5, 20)` before fusion.
pub struct HybridRetriever<E: Encoder> {
    encoder: E,
    dense: DenseIndex,
    sparse: Bm25Index,
    rrf_k: f32,
}

impl<E: Encoder> HybridRetriever<E> {
    /// Build a hybrid retriever over `encoder` with RRF constant `rrf_k`.
    pub fn new(encoder: E, rrf_k: f32) -> Self {
        let dim = encoder.embedding_dim();
        Self {
            encoder,
            dense: DenseIndex::new(dim),
            sparse: Bm25Index::default(),
            rrf_k,
        }
    }

    /// Encode + index `text` under `id` into both the dense and the lexical index.
    pub fn index_text(&mut self, id: u64, text: &str) -> Result<(), RetrievalError> {
        let emb = self.encoder.encode(text);
        self.dense.add(id, &emb)?;
        self.sparse.add(id, text);
        Ok(())
    }

    /// Number of indexed documents.
    pub fn len(&self) -> usize {
        self.dense.len()
    }

    /// Whether nothing has been indexed yet.
    pub fn is_empty(&self) -> bool {
        self.dense.is_empty()
    }

    /// Retrieve the fused top-`k` for `query`.
    pub fn retrieve(&mut self, query: &str, k: usize) -> Vec<Scored> {
        let pool = (k * 5).max(20);
        let emb = self.encoder.encode(query);
        let dense_ids: Vec<u64> = self
            .dense
            .search(&emb, pool)
            .into_iter()
            .map(|s| s.id)
            .collect();
        let sparse_ids: Vec<u64> = self
            .sparse
            .search(query, pool)
            .into_iter()
            .map(|s| s.id)
            .collect();
        reciprocal_rank_fusion(&[dense_ids, sparse_ids], self.rrf_k, k)
    }
}

/// CCOS's **bridge** to the [`Encoder`] trait: a TF-IDF embedder (the deterministic dense-vector
/// source CCOS already owns) fitted on a corpus. `encode(text)` is the corpus-fitted TF-IDF vector,
/// so the same text always encodes to the same vector and `embedding_dim` is the embedder dimension.
pub struct CcosEncoder {
    embedder: TfidfEmbedder,
    dim: usize,
}

impl CcosEncoder {
    /// Fit a TF-IDF encoder of dimension `dim` on `corpus` (each entry one document's text).
    pub fn fit(corpus: &[String], dim: usize) -> Self {
        let toks: Vec<Vec<String>> = corpus.iter().map(|t| tokenize(t)).collect();
        let mut embedder = TfidfEmbedder::new(dim);
        embedder.fit(&toks);
        Self { embedder, dim }
    }
}

impl Encoder for CcosEncoder {
    fn embedding_dim(&self) -> usize {
        self.dim
    }

    fn encode(&mut self, text: &str) -> Vec<f32> {
        self.embedder.embed_str(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_search_returns_exact_topk_in_similarity_order() {
        let mut idx = DenseIndex::new(2);
        idx.add(10, &[1.0, 0.0]).unwrap(); // cos with [1,0] = 1.000
        idx.add(20, &[0.0, 1.0]).unwrap(); // cos = 0.000
        idx.add(30, &[0.9, 0.1]).unwrap(); // cos = 0.9/√0.82 = 0.9938837
        let hits = idx.search(&[1.0, 0.0], 2);
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].id, 10);
        assert!((hits[0].score - 1.0).abs() < 1e-6);
        assert_eq!(hits[1].id, 30);
        assert!(
            (hits[1].score - 0.993_883_7).abs() < 1e-5,
            "score {}",
            hits[1].score
        );
    }

    #[test]
    fn dense_ties_break_by_ascending_id() {
        let mut idx = DenseIndex::new(2);
        idx.add(42, &[1.0, 0.0]).unwrap();
        idx.add(7, &[2.0, 0.0]).unwrap(); // identical direction → identical score
        let hits = idx.search(&[1.0, 0.0], 2);
        assert_eq!(hits[0].id, 7, "tie goes to the smaller id");
        assert_eq!(hits[1].id, 42);
    }

    #[test]
    fn dense_degenerate_and_mismatch() {
        let mut idx = DenseIndex::new(3);
        idx.add(1, &[1.0, 0.0, 0.0]).unwrap();
        assert!(idx.search(&[1.0, 0.0, 0.0], 0).is_empty()); // k = 0
        assert!(idx.search(&[1.0, 0.0], 5).is_empty()); // wrong dim
        assert!(DenseIndex::new(3).search(&[1.0, 0.0, 0.0], 5).is_empty()); // empty
        assert_eq!(
            idx.add(2, &[1.0, 2.0]),
            Err(RetrievalError::DimMismatch {
                expected: 3,
                got: 2
            })
        );
    }

    #[test]
    fn bm25_matches_hand_computed_scores() {
        // n=2 docs, b=0; idf("cat") = ln(1 + (2-2+0.5)/(2+0.5)) = ln(1.2) = 0.182322.
        // d1 "cat cat": tf=2, k1=1.2 → 0.182322·2·2.2/(2+1.2) = 0.182322·4.4/3.2 = 0.250692.
        // d0 "cat":     tf=1        → 0.182322·1·2.2/(1+1.2) = 0.182322.
        let mut bm = Bm25Index::new(1.2, 0.0);
        bm.add(0, "cat");
        bm.add(1, "cat cat");
        let hits = bm.search("cat", 2);
        assert_eq!(hits[0].id, 1, "more occurrences rank first");
        assert!(
            (hits[0].score - 0.250_692).abs() < 1e-4,
            "d1 {}",
            hits[0].score
        );
        assert_eq!(hits[1].id, 0);
        assert!(
            (hits[1].score - 0.182_322).abs() < 1e-4,
            "d0 {}",
            hits[1].score
        );
    }

    #[test]
    fn bm25_penalises_longer_documents_and_pinpoints_rare_terms() {
        let mut bm = Bm25Index::new(1.2, 0.75);
        bm.add(0, "cat");
        bm.add(1, "cat foo bar baz");
        let hits = bm.search("cat", 2);
        assert_eq!(hits[0].id, 0, "shorter doc with same tf ranks higher");
        assert!(hits[0].score > hits[1].score);

        let mut bm = Bm25Index::default();
        bm.add(0, "the quick brown fox");
        bm.add(1, "the lazy dog");
        bm.add(2, "the sphinx of quartz");
        let hits = bm.search("sphinx", 3);
        assert_eq!(hits.len(), 1, "only one doc has the rare term");
        assert_eq!(hits[0].id, 2);
    }

    #[test]
    fn rrf_fuses_by_rank_with_hand_value() {
        // doc 5: dense rank0 + sparse rank1 → 1/(60+1) + 1/(60+2) = 0.016393 + 0.016129 = 0.032522.
        // doc 9: dense rank1 + sparse rank0 → same 0.032522 (tie → smaller id 5 first).
        // doc 1: sparse rank2 only          → 1/(60+3) = 0.015873.
        let fused = reciprocal_rank_fusion(&[vec![5, 9], vec![9, 5, 1]], 60.0, 3);
        assert_eq!(fused[0].id, 5);
        assert_eq!(fused[1].id, 9);
        assert_eq!(fused[2].id, 1);
        assert!(
            (fused[0].score - (1.0 / 61.0 + 1.0 / 62.0)).abs() < 1e-6,
            "{}",
            fused[0].score
        );
        assert!((fused[2].score - 1.0 / 63.0).abs() < 1e-6);
    }

    #[test]
    fn semantic_retriever_finds_an_identical_document_at_rank_one() {
        // TASK 2 invariant: a query identical to an indexed document retrieves it at rank 1, cosine ~1.
        let corpus = [
            "database connection pool timeout".to_string(),
            "render the user interface widget".to_string(),
            "parse the abstract syntax tree".to_string(),
        ];
        let enc = CcosEncoder::fit(&corpus, 128);
        let mut r = SemanticRetriever::new(enc);
        for (i, doc) in corpus.iter().enumerate() {
            r.index_text(i as u64, doc).unwrap();
        }
        let hits = r.retrieve(&corpus[1], 3);
        assert_eq!(
            hits[0].id, 1,
            "identical text retrieves itself first: {hits:?}"
        );
        assert!(
            hits[0].score > 0.999,
            "self-cosine ~1.0, got {}",
            hits[0].score
        );
    }

    #[test]
    fn ccos_encoder_is_deterministic_and_reports_its_dimension() {
        let corpus = [
            "alpha beta gamma".to_string(),
            "beta gamma delta".to_string(),
        ];
        let mut enc = CcosEncoder::fit(&corpus, 64);
        assert_eq!(enc.embedding_dim(), 64);
        let a = enc.encode("alpha beta gamma");
        let b = enc.encode("alpha beta gamma");
        assert_eq!(
            a, b,
            "the same text encodes to the same vector, bit for bit"
        );
        assert_eq!(a.len(), 64, "vector has the reported dimension");
    }

    #[test]
    fn hybrid_retriever_indexes_both_sides_and_returns_a_ranking() {
        let corpus = [
            "connection pool timeout retry".to_string(),
            "user interface widget render".to_string(),
        ];
        let enc = CcosEncoder::fit(&corpus, 128);
        let mut h = HybridRetriever::new(enc, 60.0);
        for (i, d) in corpus.iter().enumerate() {
            h.index_text(i as u64, d).unwrap();
        }
        let hits = h.retrieve("connection pool timeout", 2);
        assert_eq!(
            hits[0].id, 0,
            "the lexically + semantically matching doc fuses to the top"
        );
    }
}

/// Ranking-quality metrics for retrieval evaluation — each in `[0, 1]` over a ranked id list and a
/// relevance set, so a pure-retrieval system is measured the way RAG benchmarks measure theirs.
pub mod metrics {
    use std::collections::{HashMap, HashSet};

    /// Recall@k: fraction of all relevant docs that appear in the top-`k`.
    pub fn recall_at_k(retrieved: &[u64], relevant: &HashSet<u64>, k: usize) -> f64 {
        if relevant.is_empty() {
            return 0.0;
        }
        let hits = retrieved
            .iter()
            .take(k)
            .filter(|id| relevant.contains(id))
            .count();
        hits as f64 / relevant.len() as f64
    }

    /// Precision@k: fraction of the top-`k` returned that are relevant. The denominator is
    /// `min(k, |retrieved|)` so a short result list is not penalised for non-existent positions.
    pub fn precision_at_k(retrieved: &[u64], relevant: &HashSet<u64>, k: usize) -> f64 {
        let depth = retrieved.len().min(k);
        if depth == 0 {
            return 0.0;
        }
        let hits = retrieved
            .iter()
            .take(k)
            .filter(|id| relevant.contains(id))
            .count();
        hits as f64 / depth as f64
    }

    /// Reciprocal rank: `1 / rank` of the first relevant doc (1-based), or `0.0` if none are relevant.
    pub fn reciprocal_rank(retrieved: &[u64], relevant: &HashSet<u64>) -> f64 {
        for (i, id) in retrieved.iter().enumerate() {
            if relevant.contains(id) {
                return 1.0 / (i as f64 + 1.0);
            }
        }
        0.0
    }

    /// Mean reciprocal rank over several `(ranking, relevant-set)` queries.
    pub fn mean_reciprocal_rank(queries: &[(Vec<u64>, HashSet<u64>)]) -> f64 {
        if queries.is_empty() {
            return 0.0;
        }
        let total: f64 = queries.iter().map(|(r, rel)| reciprocal_rank(r, rel)).sum();
        total / queries.len() as f64
    }

    /// Average precision for one query: the mean of the precision values at each rank where a relevant
    /// doc occurs, divided by the number of relevant docs.
    pub fn average_precision(retrieved: &[u64], relevant: &HashSet<u64>) -> f64 {
        if relevant.is_empty() {
            return 0.0;
        }
        let (mut hits, mut sum) = (0usize, 0.0f64);
        for (i, id) in retrieved.iter().enumerate() {
            if relevant.contains(id) {
                hits += 1;
                sum += hits as f64 / (i as f64 + 1.0);
            }
        }
        sum / relevant.len() as f64
    }

    /// DCG of the first `k` ranks: `Σ gainᵢ / log₂(rank+1)` (rank 1-based).
    fn dcg_at_k(gains: impl Iterator<Item = f64>, k: usize) -> f64 {
        gains
            .take(k)
            .enumerate()
            .map(|(i, g)| g / (i as f64 + 2.0).log2())
            .sum()
    }

    /// nDCG@k with graded relevance `gains` (use `1.0`/`0.0` for binary): the returned ranking's DCG
    /// over the ideal DCG (gains sorted descending). `0.0` when there is no positive gain to recover.
    pub fn ndcg_at_k(retrieved: &[u64], gains: &HashMap<u64, f64>, k: usize) -> f64 {
        let actual = dcg_at_k(
            retrieved
                .iter()
                .map(|id| gains.get(id).copied().unwrap_or(0.0)),
            k,
        );
        let mut ideal: Vec<f64> = gains.values().copied().filter(|&g| g > 0.0).collect();
        ideal.sort_by(|a, b| b.partial_cmp(a).unwrap_or(core::cmp::Ordering::Equal));
        let best = dcg_at_k(ideal.into_iter(), k);
        if best <= 0.0 {
            return 0.0;
        }
        actual / best
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn set(ids: &[u64]) -> HashSet<u64> {
            ids.iter().copied().collect()
        }

        #[test]
        fn recall_and_precision_hand_values() {
            // retrieved [1,2,3,4]; relevant {2,4,9}; hits in top-4 = {2,4}.
            let (r, rel) = ([1, 2, 3, 4], set(&[2, 4, 9]));
            assert!((recall_at_k(&r, &rel, 4) - 2.0 / 3.0).abs() < 1e-12);
            assert!((precision_at_k(&r, &rel, 4) - 0.5).abs() < 1e-12);
            assert!((precision_at_k(&r, &rel, 2) - 0.5).abs() < 1e-12);
        }

        #[test]
        fn reciprocal_rank_and_mrr_hand_values() {
            assert!((reciprocal_rank(&[5, 2, 7], &set(&[2])) - 0.5).abs() < 1e-12);
            assert_eq!(reciprocal_rank(&[5, 7], &set(&[2])), 0.0);
            let q = vec![(vec![2u64, 9], set(&[2])), (vec![9u64, 2], set(&[2]))];
            assert!((mean_reciprocal_rank(&q) - 0.75).abs() < 1e-12); // (1 + 0.5)/2
        }

        #[test]
        fn average_precision_hand_value() {
            // retrieved [2,1,4,3]; relevant {2,4}: rank1 prec 1, rank3 prec 2/3 → (1 + 2/3)/2.
            let ap = average_precision(&[2, 1, 4, 3], &set(&[2, 4]));
            assert!((ap - (1.0 + 2.0 / 3.0) / 2.0).abs() < 1e-12, "ap {ap}");
        }

        #[test]
        fn ndcg_matches_hand_computed_value() {
            // gains id1=3,id2=2,id3=0,id4=1. ranking [1,4,3,2]:
            // DCG  = 3/log2(2) + 1/log2(3) + 0 + 2/log2(5) = 3 + 0.6309298 + 0.8613531 = 4.4922829
            // IDCG (3,2,1,0)   = 3 + 2/log2(3) + 1/log2(4) = 3 + 1.2618595 + 0.5      = 4.7618595
            // nDCG = 4.4922829 / 4.7618595 = 0.9433884
            let gains: HashMap<u64, f64> = [(1, 3.0), (2, 2.0), (3, 0.0), (4, 1.0)]
                .into_iter()
                .collect();
            let n = ndcg_at_k(&[1, 4, 3, 2], &gains, 4);
            assert!((n - 0.943_388_4).abs() < 1e-6, "ndcg {n}");
            let ideal = ndcg_at_k(&[1, 2, 4, 3], &gains, 4);
            assert!((ideal - 1.0).abs() < 1e-12, "ideal ndcg {ideal}");
        }

        #[test]
        fn empty_relevance_is_zero_not_nan() {
            assert_eq!(recall_at_k(&[1, 2], &set(&[]), 2), 0.0);
            assert_eq!(average_precision(&[1, 2], &set(&[])), 0.0);
            assert_eq!(ndcg_at_k(&[1, 2], &HashMap::new(), 2), 0.0);
        }
    }
}
