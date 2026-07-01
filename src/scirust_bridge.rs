//! Bridge to the external `scirust-retrieval` crate (optional, behind the
//! `scirust-retrieval` feature).
//!
//! CCOS's native dense retrieval ([`CausalEmbeddings::nearest_k`]) stores
//! INT4-quantized vectors and scans them itself. This bridge lets the *same*
//! deterministic TF-IDF embeddings instead feed scirust-retrieval's exact,
//! full-precision [`DenseIndex`], so the two can be compared head-to-head in the
//! eval harness (the `scirust-dense` strategy) — isolating the cost of CCOS's
//! INT4 quantization from the embedding model itself.
//!
//! Nothing here touches the default build: `scirust-retrieval` is an *optional*
//! dependency gated behind the feature of the same name, and we depend only on
//! its pure core (`default-features = false`) — no `scirust-core` autodiff/nn
//! stack, only `serde`/`sha2`, already in our tree. The default `cargo build`
//! stays byte-identical and pulls neither the crate nor any new dependency.
//!
//! Determinism (the replay invariant) is preserved end to end: the TF-IDF
//! embedder is stateless and order-free, and [`DenseIndex::search`] breaks score
//! ties by ascending id, so a fixed corpus and query always yield the same
//! ranking.
//!
//! [`CausalEmbeddings::nearest_k`]: crate::embeddings::CausalEmbeddings::nearest_k

use crate::embeddings::CausalEmbeddings;
use scirust_retrieval::{DenseIndex, Encoder};

/// Adapts CCOS's [`CausalEmbeddings`] (deterministic TF-IDF, read here at full
/// f32 precision) to scirust-retrieval's [`Encoder`] trait, so CCOS embeddings
/// can drive any scirust-retrieval index.
pub struct CcosEncoder {
    embeddings: CausalEmbeddings,
}

impl CcosEncoder {
    /// Build an encoder whose TF-IDF statistics are fit on `docs` (`(id, text)`
    /// pairs). The ids only label the corpus during fitting; [`Encoder::encode`]
    /// embeds by text.
    pub fn fit(docs: &[(&str, &str)]) -> Self {
        let mut embeddings = CausalEmbeddings::new();
        embeddings.fit_and_embed(docs.iter().map(|&(id, text)| (id, text)));
        Self { embeddings }
    }
}

impl Encoder for CcosEncoder {
    fn embedding_dim(&self) -> usize {
        self.embeddings.embedder.dim
    }

    fn encode(&mut self, text: &str) -> Vec<f32> {
        self.embeddings.embed_query(text)
    }
}

/// Rank `docs` (`(id, text)` pairs) best-first by dense cosine similarity to
/// `query`, using scirust-retrieval's exact [`DenseIndex`] over CCOS TF-IDF
/// embeddings.
///
/// Returns the document ids in ranked order. Deterministic for a fixed
/// `docs`/`query`.
pub fn dense_rank(docs: &[(&str, &str)], query: &str) -> Vec<String> {
    if docs.is_empty() {
        return Vec::new();
    }
    let mut encoder = CcosEncoder::fit(docs);
    let mut index = DenseIndex::new(encoder.embedding_dim());
    for (i, &(_, text)) in docs.iter().enumerate() {
        let v = encoder.encode(text);
        // `add` only errors on a dimension mismatch, impossible here (every
        // vector comes from the same encoder); ignore defensively.
        let _ = index.add(i as u64, &v);
    }
    let q = encoder.encode(query);
    index
        .search(&q, docs.len())
        .into_iter()
        .map(|scored| docs[scored.id as usize].0.to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_rank_orders_by_relevance_and_is_deterministic() {
        let docs = [
            ("a", "the quick brown fox jumps over"),
            ("b", "lorem ipsum dolor sit amet consectetur"),
            ("c", "a brown fox is quick and brown again"),
        ];
        let query = "brown fox quick";

        let first = dense_rank(&docs, query);
        let second = dense_rank(&docs, query);
        assert_eq!(first, second, "ranking must be deterministic (replay invariant)");
        assert_eq!(first.len(), docs.len(), "every doc is ranked");

        // The unrelated lorem-ipsum doc shares no query term, so it ranks last;
        // a fox/brown/quick doc ranks first.
        assert_eq!(first.last().map(String::as_str), Some("b"));
        assert!(matches!(first.first().map(String::as_str), Some("a") | Some("c")));
    }

    #[test]
    fn dense_rank_empty_corpus_is_empty() {
        assert!(dense_rank(&[], "anything").is_empty());
    }
}
