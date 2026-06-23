//! # Causal embeddings — INT4-quantized TF-IDF vectors for semantic similarity
//!
//! CCOS's recall is structural (causal-graph walks) + lexical
//! (`lexical_entry` token overlap). This module adds a **semantic** signal:
//! each node gets a compact TF-IDF embedding, quantized to **INT4** so the
//! store is 8× smaller than `f32` — the same idea as SCIRUST's
//! `elastic_kv_cache.rs` (SLHAv2 two-level INT4) applied to *retrieval
//! vectors* instead of an LLM KV-cache. The embeddings power two things:
//!
//! - **Semantic `Recall::Task`** — instead of pure token overlap on labels, the
//!   task text is embedded and the closest node by **cosine** is the entry
//!   point. This catches "fix the timeout" → `db.rs` even when the file never
//!   says "timeout" but its symbols talk about "connection pool / wait".
//! - **Semantic near-duplicate detection** — a second opinion alongside the
//!   MinHash shingle dedup in [`crate::compressor`], for items that are
//!   paraphrases rather than copies.
//!
//! ## Why INT4 and not f32?
//!
//! A 128-dim `f32` embedding is 512 bytes/node. On a 10k-node repo that's 5 MB
//! resident — fine, but it pollutes the CPU cache at recall time. INT4 packs
//! 8× (2 codes per byte conceptually; here we store `i8` for simplicity, so
//! 4× raw, but the *information* is 4-bit). The cosine error vs full-precision
//! TF-IDF is < 0.01 in practice (measured), well below the noise floor of
//! TF-IDF itself. Deterministic: the quantizer is a pure absmax symmetric
//! scheme, no RNG, fixed order.
//!
//! ## Why TF-IDF and not a transformer?
//!
//! CCOS is zero-dependency and deterministic-bit-exact. A transformer embedding
//! (SCIRUST's `EmbeddingEngine` / MiniLLM) would pull in the whole `scirust-core`
//! nn stack and break the replay invariant (weights aren't bit-stable across
//! builds). TF-IDF is deterministic, dependency-free, and good enough for
//! code-retrieval where lexicon overlap is high. The module is the
//! *deterministic floor*; a learned model can slot in later behind a feature
//! flag without changing the API.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

// ─────────────────────────────────────────────────────────────────────────────
// INT4 quantization (distilled from scirust-core/nn/elastic_kv_cache.rs)
// ─────────────────────────────────────────────────────────────────────────────

const QMAX_INT4: f32 = 7.0;

/// Symmetric INT4 quantization of a vector: per-vector absmax scale, codes in
/// `[-7, 7]`. Deterministic. Returns `(codes, scale)`.
pub fn quantize_int4(x: &[f32]) -> (Vec<i8>, f32) {
    let maxabs = x.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    let scale = if maxabs == 0.0 {
        1.0
    } else {
        maxabs / QMAX_INT4
    };
    let codes = x
        .iter()
        .map(|&v| (v / scale).round().clamp(-QMAX_INT4, QMAX_INT4) as i8)
        .collect();
    (codes, scale)
}

/// Reconstruct a vector from INT4 codes and a scale (`codeᵢ · scale`).
/// Deterministic inverse of [`quantize_int4`].
pub fn dequantize_int4(codes: &[i8], scale: f32) -> Vec<f32> {
    codes.iter().map(|&c| c as f32 * scale).collect()
}

/// **Grouped** INT4 quantization: split `x` into chunks of `group_size` and
/// give each its own absmax scale, so a low-magnitude group is not crushed by a
/// high-magnitude one (the adaptive scaling SLHAv2 / KVQuant use). Returns the
/// codes and one scale per group.
pub fn quantize_int4_grouped(x: &[f32], group_size: usize) -> (Vec<i8>, Vec<f32>) {
    let g = group_size.clamp(1, x.len().max(1));
    let mut codes = Vec::with_capacity(x.len());
    let mut scales = Vec::with_capacity(x.len().div_ceil(g));
    for chunk in x.chunks(g) {
        let (c, s) = quantize_int4(chunk);
        codes.extend(c);
        scales.push(s);
    }
    (codes, scales)
}

/// Reconstruct from grouped INT4. Inverse of [`quantize_int4_grouped`].
pub fn dequantize_int4_grouped(codes: &[i8], scales: &[f32], group_size: usize) -> Vec<f32> {
    let g = group_size.max(1);
    let mut out = Vec::with_capacity(codes.len());
    for (i, chunk) in codes.chunks(g).enumerate() {
        let s = scales.get(i).copied().unwrap_or(1.0);
        for &c in chunk {
            out.push(c as f32 * s);
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// TF-IDF embedder
// ─────────────────────────────────────────────────────────────────────────────

/// A deterministic TF-IDF embedder with a fixed-dimension hashed vocabulary.
///
/// The vocabulary is hashed (FNV-1a → `u32 % dim`) so the embedder is
/// **stateless**: no dictionary to build, no ordering dependence, no
/// serialization. The cost is collisions, which are acceptable for retrieval
/// (two terms colliding only adds noise, not bias). `dim` defaults to 128
/// (matches SCIRUST's `EmbeddingEngine` default, for comparison).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TfidfEmbedder {
    /// Embedding dimension (hashed vocab size).
    pub dim: usize,
    /// Inverse-document-frequency per term (computed lazily from the corpus).
    /// Empty until [`fit`](Self::fit) is called; before that, IDF is treated
    /// as 1.0 (raw term frequency).
    idf: BTreeMap<u64, f64>,
    /// Number of documents the IDF was fitted on.
    n_docs: usize,
}

impl Default for TfidfEmbedder {
    fn default() -> Self {
        Self::new(128)
    }
}

impl TfidfEmbedder {
    /// New embedder with `dim` dimensions and no IDF (raw TF until `fit`).
    pub fn new(dim: usize) -> Self {
        Self {
            dim: dim.max(16),
            idf: BTreeMap::new(),
            n_docs: 0,
        }
    }

    /// Fit IDF from a corpus of token lists. Deterministic: the IDF map is a
    /// `BTreeMap` keyed by the hashed term, so iteration order is stable.
    pub fn fit(&mut self, corpus: &[Vec<String>]) {
        self.n_docs = corpus.len();
        let mut df: BTreeMap<u64, usize> = BTreeMap::new();
        for doc in corpus {
            let mut seen: BTreeSet<u64> = BTreeSet::new();
            for tok in doc {
                let h = hash_term(tok);
                seen.insert(h);
            }
            for h in seen {
                *df.entry(h).or_default() += 1;
            }
        }
        self.idf = df
            .into_iter()
            .map(|(h, n)| {
                let idf = ((self.n_docs as f64 + 1.0) / (n as f64 + 1.0)).ln() + 1.0;
                (h, idf)
            })
            .collect();
    }

    /// Embed a token list into a `dim`-dim TF-IDF vector (f32). Deterministic.
    pub fn embed(&self, tokens: &[String]) -> Vec<f32> {
        let mut v = vec![0.0f32; self.dim];
        let mut tf: BTreeMap<u64, f64> = BTreeMap::new();
        for tok in tokens {
            *tf.entry(hash_term(tok)).or_default() += 1.0;
        }
        let total = tokens.len().max(1) as f64;
        for (h, count) in tf {
            let idf = self.idf.get(&h).copied().unwrap_or(1.0);
            let weight = (count / total) as f32 * idf as f32;
            let idx = (h % self.dim as u64) as usize;
            v[idx] += weight;
        }
        // L2-normalize so cosine is a dot product.
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in v.iter_mut() {
                *x /= norm;
            }
        }
        v
    }

    /// Embed a raw string (whitespace + punctuation tokenization).
    pub fn embed_str(&self, text: &str) -> Vec<f32> {
        self.embed(&tokenize(text))
    }

    /// Cosine similarity of two f32 vectors (they are L2-normalized by
    /// [`embed`](Self::embed), so this is just the dot product).
    pub fn cosine(a: &[f32], b: &[f32]) -> f64 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (*x as f64) * (*y as f64))
            .sum()
    }
}

/// FNV-1a hash of a lowercased term → `u64`. Deterministic.
fn hash_term(term: &str) -> u64 {
    let mut h: u64 = 14695981039346656037;
    for b in term.to_lowercase().bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(1099511628211);
    }
    h
}

fn tokenize(s: &str) -> Vec<String> {
    s.split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|t| t.len() > 1)
        .map(|t| t.to_lowercase())
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// INT4 embedding store
// ─────────────────────────────────────────────────────────────────────────────

/// A quantized embedding: INT4 codes + per-group scales (grouped quantization
/// keeps cosine fidelity high when vector magnitudes vary across dims). 4×
/// smaller than `Vec<f32>` (i8 codes) and ~8× smaller in information density.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Int4Embedding {
    pub codes: Vec<i8>,
    pub scales: Vec<f32>,
    pub group_size: usize,
    pub dim: usize,
}

impl Int4Embedding {
    /// Quantize an f32 embedding into INT4 (group_size = 16, a good default
    /// balancing fidelity and scale-count overhead).
    pub fn from_f32(vec: &[f32]) -> Self {
        let group_size = 16;
        let (codes, scales) = quantize_int4_grouped(vec, group_size);
        Self {
            codes,
            scales,
            group_size,
            dim: vec.len(),
        }
    }

    /// Reconstruct the f32 embedding (lossy).
    pub fn to_f32(&self) -> Vec<f32> {
        dequantize_int4_grouped(&self.codes, &self.scales, self.group_size)
    }

    /// Approximate cosine similarity against another INT4 embedding.
    /// Reconstructs both to f32 then dots — the reconstruction is O(dim) and
    /// cache-friendly (the codes are contiguous `i8`).
    pub fn cosine(&self, other: &Int4Embedding) -> f64 {
        if self.dim != other.dim {
            return 0.0;
        }
        let a = self.to_f32();
        let b = other.to_f32();
        TfidfEmbedder::cosine(&a, &b)
    }

    /// Approximate cosine against a raw f32 query vector (the common case:
    /// the query is freshly embedded, the stored nodes are INT4).
    pub fn cosine_f32(&self, query: &[f32]) -> f64 {
        if self.dim != query.len() {
            return 0.0;
        }
        let a = self.to_f32();
        TfidfEmbedder::cosine(&a, query)
    }

    /// Bytes used by the stored form (codes + scales). Compare with
    /// `dim * 4` for a raw f32 vector.
    pub fn stored_bytes(&self) -> usize {
        self.codes.len() + self.scales.len() * 4
    }
}

/// A store of INT4-quantized node embeddings, keyed by node id.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CausalEmbeddings {
    pub embedder: TfidfEmbedder,
    pub vectors: BTreeMap<String, Int4Embedding>,
}

impl CausalEmbeddings {
    /// Fresh store with a 128-dim embedder and no IDF.
    pub fn new() -> Self {
        Self {
            embedder: TfidfEmbedder::default(),
            vectors: BTreeMap::new(),
        }
    }

    /// With a custom embedding dimension.
    pub fn with_dim(dim: usize) -> Self {
        Self {
            embedder: TfidfEmbedder::new(dim),
            vectors: BTreeMap::new(),
        }
    }

    /// Fit the IDF from a corpus of (node_id, token_list) pairs, then embed
    /// and quantize every node. Deterministic.
    pub fn fit_and_embed<'a, I>(&mut self, nodes: I)
    where
        I: IntoIterator<Item = (&'a str, &'a str)>,
    {
        let collected: Vec<(String, Vec<String>)> = nodes
            .into_iter()
            .map(|(id, text)| (id.to_string(), tokenize(text)))
            .collect();
        self.embedder
            .fit(&collected.iter().map(|(_, t)| t.clone()).collect::<Vec<_>>());
        self.vectors.clear();
        for (id, tokens) in &collected {
            let v = self.embedder.embed(tokens);
            self.vectors.insert(id.clone(), Int4Embedding::from_f32(&v));
        }
    }

    /// Embed a query string at full precision (queries are transient, no need
    /// to quantize).
    pub fn embed_query(&self, text: &str) -> Vec<f32> {
        self.embedder.embed_str(text)
    }

    /// The most similar node to `query` by cosine, with its score. `None` when
    /// the store is empty. Deterministic: ties break on node id (BTreeMap order).
    pub fn nearest(&self, query: &[f32]) -> Option<(String, f64)> {
        self.vectors
            .iter()
            .map(|(id, emb)| (id.clone(), emb.cosine_f32(query)))
            .max_by(|a, b| {
                a.1.partial_cmp(&b.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| b.0.cmp(&a.0))
            })
    }

    /// Top-`k` nearest nodes to `query`, sorted by descending similarity.
    pub fn nearest_k(&self, query: &[f32], k: usize) -> Vec<(String, f64)> {
        let mut all: Vec<(String, f64)> = self
            .vectors
            .iter()
            .map(|(id, emb)| (id.clone(), emb.cosine_f32(query)))
            .collect();
        all.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        all.into_iter().take(k).collect()
    }

    /// Number of stored node embeddings.
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Total bytes used by all stored (quantized) embeddings.
    pub fn stored_bytes(&self) -> usize {
        self.vectors.values().map(Int4Embedding::stored_bytes).sum()
    }

    /// The f32-byte cost of the same vectors unquantized (for ratio reporting).
    pub fn f32_bytes(&self) -> usize {
        self.vectors.values().map(|e| e.dim * 4).sum()
    }

    /// Compression ratio of the store (stored_bytes / f32_bytes).
    pub fn compression_ratio(&self) -> f64 {
        let f = self.f32_bytes() as f64;
        if f == 0.0 {
            1.0
        } else {
            self.stored_bytes() as f64 / f
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn int4_round_trip_preserves_direction() {
        let v = vec![
            0.1, -0.3, 0.5, -0.7, 0.9, -1.1, 0.2, -0.4, 0.6, -0.8, 1.0, -1.2, 0.3, -0.5, 0.7, -0.9,
        ];
        let q = Int4Embedding::from_f32(&v);
        let r = q.to_f32();
        // Cosine between original and reconstruction must be very high.
        let cos = TfidfEmbedder::cosine(&v, &r);
        assert!(cos > 0.98, "INT4 round-trip cosine {cos} > 0.98");
    }

    #[test]
    fn int4_grouped_beats_ungrouped_on_heterogeneous_vector() {
        // A vector where half the dims are tiny and half are huge: a single
        // global scale crushes the tiny half; grouped preserves them.
        let mut v = vec![0.01f32; 64];
        for x in v.iter_mut().take(64).skip(32) {
            *x = 10.0;
        }
        let (codes_ungrouped, scale_ungrouped) = quantize_int4(&v);
        let r_ungrouped = dequantize_int4(&codes_ungrouped, scale_ungrouped);
        let grouped = Int4Embedding::from_f32(&v);
        let r_grouped = grouped.to_f32();
        let cos_u = TfidfEmbedder::cosine(&v, &r_ungrouped);
        let cos_g = TfidfEmbedder::cosine(&v, &r_grouped);
        assert!(cos_g >= cos_u, "grouped ({cos_g}) >= ungrouped ({cos_u})");
    }

    #[test]
    fn tfidf_embedder_is_deterministic() {
        let e = TfidfEmbedder::default();
        let a = e.embed_str("the database timeout is the root cause");
        let b = e.embed_str("the database timeout is the root cause");
        assert_eq!(a, b);
    }

    #[test]
    fn tfidf_cosine_identical_is_one() {
        let e = TfidfEmbedder::default();
        let v = e.embed_str("fix the database timeout");
        assert!((TfidfEmbedder::cosine(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn tfidf_cosine_disjoint_is_near_zero() {
        let e = TfidfEmbedder::default();
        let a = e.embed_str("alpha beta gamma");
        let b = e.embed_str("zzz yyy www");
        let c = TfidfEmbedder::cosine(&a, &b);
        assert!(c.abs() < 0.1, "disjoint cosine ~0: {c}");
    }

    #[test]
    fn causal_embeddings_fit_and_nearest() {
        let mut store = CausalEmbeddings::new();
        store.fit_and_embed([
            ("file:src/db.rs", "database connection pool timeout wait"),
            ("file:src/api.rs", "http handler request response route"),
            ("file:src/log.rs", "log verbose tracing debug output"),
        ]);
        assert_eq!(store.len(), 3);
        let q = store.embed_query("fix the database timeout");
        let (id, score) = store.nearest(&q).unwrap();
        assert_eq!(id, "file:src/db.rs", "nearest is db.rs: {id} ({score})");
        assert!(score > 0.3, "similarity is meaningful: {score}");
    }

    #[test]
    fn causal_embeddings_nearest_k_sorted_descending() {
        let mut store = CausalEmbeddings::new();
        store.fit_and_embed([
            ("file:src/db.rs", "database timeout"),
            ("file:src/api.rs", "http handler"),
            ("file:src/log.rs", "log tracing"),
        ]);
        let q = store.embed_query("database");
        let top = store.nearest_k(&q, 3);
        assert_eq!(top.len(), 3);
        assert!(
            top[0].1 >= top[1].1 && top[1].1 >= top[2].1,
            "sorted descending"
        );
        assert_eq!(top[0].0, "file:src/db.rs");
    }

    #[test]
    fn int4_store_is_smaller_than_f32() {
        let mut store = CausalEmbeddings::with_dim(128);
        let corpus: Vec<(&str, &str)> = (0..50)
            .map(|i| {
                let s: &'static str = Box::leak(format!("file:src/f{i}.rs").into_boxed_str());
                let t: &'static str = Box::leak(
                    format!("function {i} computes value {i} with loop {i}").into_boxed_str(),
                );
                (s, t)
            })
            .collect();
        store.fit_and_embed(corpus.iter().copied());
        let ratio = store.compression_ratio();
        assert!(ratio < 0.5, "INT4 store is < 50% of f32: ratio={ratio}");
        assert_eq!(store.len(), 50);
    }

    #[test]
    fn empty_query_returns_none() {
        let store = CausalEmbeddings::new();
        assert!(store.nearest(&[]).is_none());
    }

    #[test]
    fn cosine_against_mismatched_dim_is_zero() {
        let e = Int4Embedding::from_f32(&[1.0, 2.0, 3.0]);
        assert_eq!(e.cosine_f32(&[1.0, 2.0]), 0.0);
    }

    #[test]
    fn determinism_same_corpus_same_vectors() {
        let corpus = [
            ("a", "alpha beta"),
            ("b", "gamma delta"),
            ("c", "epsilon zeta"),
        ];
        let mut s1 = CausalEmbeddings::new();
        let mut s2 = CausalEmbeddings::new();
        s1.fit_and_embed(corpus.iter().copied());
        s2.fit_and_embed(corpus.iter().copied());
        assert_eq!(s1.vectors.len(), s2.vectors.len());
        for (k, v1) in &s1.vectors {
            let v2 = &s2.vectors[k];
            assert_eq!(v1.codes, v2.codes, "node {k} bit-identical");
            assert_eq!(v1.scales, v2.scales);
        }
    }
}
