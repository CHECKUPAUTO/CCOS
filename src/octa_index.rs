//! **OctaSoma semantic memory** — region-sharded, embedding-based semantic anchors for the
//! causal graph, behind the off-by-default `octasoma` cargo feature and the Pro
//! [`Feature::OctaSomaMemory`](crate::license::Feature::OctaSomaMemory) runtime gate.
//!
//! The `OctaIndex`/`ShardedOctaIndex` types below are **vendored from the `octasoma` repo**
//! (`integration/ccos/octa_index.rs` at the rev pinned in `Cargo.toml`), with one adaptation:
//! `region_of` is rewritten without a let-chain, since CCOS is edition 2021. Everything else is
//! the adapter as octasoma ships it, so refreshing the vendored copy stays a diff, not a port.
//!
//! Why sharded, and why anchors: the real-scale benchmark in octasoma's
//! `docs/integration-ecosystem.md` showed a single **global** 3-D index is only a coarse router
//! (~0 % exact hits at ~800 nodes) — the validated 99 %-hit cascade is **CCOS narrowing causally
//! first, then an exact rerank inside that small region**. So the index is one small OctaSoma
//! store per causal region, queried with [`ShardedOctaIndex::semantic_anchors_in`] when the
//! region is known, and the anchors it returns are node URIs the caller expands through the
//! causal graph as usual (the window assembly, budgets, and event log are untouched).
//!
//! The quarantine boundary, stated plainly (same contract as [`crate::neural_embed`]):
//!
//! - **Off by default.** `cargo build` compiles none of this; the feature pulls the `octasoma`
//!   crate (`#![forbid(unsafe_code)]`, one dependency), pinned to a reviewed rev.
//! - **Replay-exactness follows the embedder.** With octasoma's deterministic `HashEmbedder`
//!   the index is bit-replayable; with a neural embedder (e.g. octasoma's `OllamaEmbedder`,
//!   local-only) vectors depend on model weights and server build — semantic quality up,
//!   replay-exactness gone. The choice is the caller's and is visible in the type.
//! - **Pro-gated at runtime.** Construction goes through [`SemanticMemoryAccess::unlock`],
//!   which consults CCOS's offline license exactly like
//!   [`RetrievalAccess`](crate::retrieval::RetrievalAccess); on the community tier it returns
//!   the standard no-silent-downgrade refusal and the free core recall strategies
//!   (working-set / around / task / INT4 TF-IDF semantic / hybrid) remain fully functional.

use crate::external_memory::{ExternalMemory, Recall, RecallWindow};
use crate::memory::MemoryGraph;
use octasoma::{Embedder, FractalMemory3D, ShardedMemory};

/// A semantic index over CCOS nodes: content → embedding → 3-D octree, keyed by
/// the node's URI (`sym:…`, `mod:…`, `file:…`).
pub struct OctaIndex<E: Embedder> {
    core: FractalMemory3D,
    embedder: E,
}

impl<E: Embedder> OctaIndex<E> {
    /// Creates an empty index for the given embedder
    /// (`OllamaEmbedder` in production, `HashEmbedder` for offline tests).
    pub fn new(embedder: E) -> Self {
        let core = FractalMemory3D::new(embedder.dim(), 42);
        Self { core, embedder }
    }

    /// Loads a previously saved index (`.frac`) for `embedder`.
    pub fn open(embedder: E, path: &str) -> std::io::Result<Self> {
        let core = FractalMemory3D::load_from_disk(path, embedder.dim())?;
        Ok(Self { core, embedder })
    }

    /// Indexes a CCOS node: embed its `content`, store it under its `uri`.
    /// Call this for every node created/updated in `ingest_source`.
    pub fn index_node(&mut self, uri: &str, content: &str) {
        if let Ok(v) = self.embedder.embed(content) {
            self.core.insert(&v, Some(uri.as_bytes()));
        }
    }

    /// Returns the `k` semantically-nearest node URIs to `text`, each with a score
    /// in `(0, 1]` (`1 / (1 + distance²)`). These are the **anchors** CCOS feeds to
    /// `assemble_window` for causal expansion.
    pub fn semantic_anchors(&self, text: &str, k: usize) -> Vec<(String, f64)> {
        let Ok(v) = self.embedder.embed(text) else {
            return Vec::new();
        };
        self.core
            .nearest_embedding(&v, k)
            .into_iter()
            .filter_map(|(id, d2)| {
                self.core.get_payload(id).map(|b| {
                    (
                        String::from_utf8_lossy(b).into_owned(),
                        1.0 / (1.0 + d2 as f64),
                    )
                })
            })
            .collect()
    }

    /// Persists the index to a `.frac` file (mirror CCOS's `checkpoint`).
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        self.core.save_to_disk(path)
    }

    /// Number of indexed nodes.
    pub fn len(&self) -> usize {
        self.core.item_count()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.core.item_count() == 0
    }
}

/// Derives a CCOS **causal region** key from a node URI of the form
/// `kind:path[:symbol]` (e.g. `sym:src/db.rs:query` → `src/db.rs`,
/// `mod:src/cache.rs` → `src/cache.rs`, `file:src/main.rs` → `src/main.rs`).
///
/// Falls back to the whole URI when it doesn't match that shape. CCOS usually
/// already knows each node's file/region, so prefer the explicit
/// [`ShardedOctaIndex::index_node_in`] when you do.
pub fn region_of(uri: &str) -> String {
    // Drop the `kind:` prefix.
    let rest = uri.split_once(':').map(|(_, r)| r).unwrap_or(uri);
    // A `sym:` URI carries a trailing `:symbol`; the region is the file path.
    // (Vendored upstream uses an edition-2024 let-chain here; nested `if` for 2021.)
    if uri.starts_with("sym:") {
        if let Some(i) = rest.rfind(':') {
            return rest[..i].to_string();
        }
    }
    rest.to_string()
}

/// A **region-sharded** semantic index for CCOS: one small OctaSoma index per
/// causal region (file). This is the deployment the real-scale benchmark
/// validated — OctaSoma's 3-D projection is a coarse router that fails as a
/// single global index but works *within* a region, so CCOS narrows causally
/// first and OctaSoma reranks inside the region it gives you.
///
/// Use [`ShardedOctaIndex::semantic_anchors_in`] when CCOS knows the region
/// (the validated 99 %-hit path); fall back to [`ShardedOctaIndex::semantic_anchors`]
/// (a coarse cross-region merge) only when no causal scope is known.
pub struct ShardedOctaIndex<E: Embedder> {
    mem: ShardedMemory<E>,
}

impl<E: Embedder> ShardedOctaIndex<E> {
    /// Creates an empty sharded index for `embedder`.
    pub fn new(embedder: E) -> Self {
        Self {
            mem: ShardedMemory::new(embedder),
        }
    }

    /// Reopens a sharded index previously written by [`ShardedOctaIndex::save`].
    pub fn open(embedder: E, dir: &str) -> std::io::Result<Self> {
        Ok(Self {
            mem: ShardedMemory::open_dir(embedder, dir)?,
        })
    }

    /// Indexes a node into an **explicit** causal region (recommended: CCOS
    /// already knows each node's file/region).
    pub fn index_node_in(&mut self, region: &str, uri: &str, content: &str) {
        let _ = self.mem.insert(region, uri, content);
    }

    /// Indexes a node, deriving its region from the URI via [`region_of`].
    pub fn index_node(&mut self, uri: &str, content: &str) {
        let region = region_of(uri);
        let _ = self.mem.insert(&region, uri, content);
    }

    /// Semantic anchors **within** a known causal region — the validated path.
    /// Scores are `1 / (1 + distance²)` in `(0, 1]`, descending.
    pub fn semantic_anchors_in(&self, region: &str, text: &str, k: usize) -> Vec<(String, f64)> {
        self.mem
            .recall_scored(region, text, k)
            .unwrap_or_default()
            .into_iter()
            .map(|(uri, d2)| (uri, 1.0 / (1.0 + d2 as f64)))
            .collect()
    }

    /// Coarse anchors across **all** regions (use only when no causal scope is
    /// known; cross-region distances are merely a heuristic).
    pub fn semantic_anchors(&self, text: &str, k: usize) -> Vec<(String, f64)> {
        self.mem
            .recall_global_scored(text, k)
            .unwrap_or_default()
            .into_iter()
            .map(|(uri, d2)| (uri, 1.0 / (1.0 + d2 as f64)))
            .collect()
    }

    /// Persists every region's shard under `dir` (mirror CCOS's `checkpoint`).
    pub fn save(&self, dir: &str) -> std::io::Result<()> {
        self.mem.save_dir(dir)
    }

    /// Number of causal regions (shards).
    pub fn regions(&self) -> usize {
        self.mem.regions()
    }

    /// Total indexed nodes across all regions.
    pub fn len(&self) -> usize {
        self.mem.len()
    }

    /// Whether nothing has been indexed yet.
    pub fn is_empty(&self) -> bool {
        self.mem.is_empty()
    }
}

// ──────────────── Composition over the causal graph (CCOS-side, not vendored) ────────────────

impl<E: Embedder> ShardedOctaIndex<E> {
    /// Feeds every content-carrying node of a causal graph into the index,
    /// **deterministically**: nodes are visited in sorted-id order (never `HashMap`
    /// iteration order), so with a deterministic embedder the resulting index is
    /// bit-identical across runs — the same discipline as the rest of CCOS. Each
    /// node lands in the causal region derived from its URI ([`region_of`]).
    /// Returns the number of nodes fed.
    ///
    /// Intended for a **freshly built** index (octasoma stores are insertion-only:
    /// re-feeding the same graph into a used index duplicates entries). Rebuild
    /// after ingest deltas, or mirror your `ingest_source` calls with
    /// [`ShardedOctaIndex::index_node_in`] instead.
    pub fn index_graph(&mut self, graph: &MemoryGraph) -> usize {
        let mut entries: Vec<_> = graph
            .node_entries()
            .filter(|(_, n)| !n.content.is_empty())
            .collect();
        entries.sort_by(|a, b| a.0.cmp(b.0));
        let fed = entries.len();
        for (id, node) in entries {
            let region = region_of(&id.0);
            self.index_node_in(&region, &id.0, &node.content);
        }
        fed
    }
}

/// **Anchor-first semantic recall** over any [`ExternalMemory`]: OctaSoma resolves the
/// entry node semantically, then CCOS expands it causally — the window comes from
/// [`Recall::Around`] on the anchor, with the same region membership, proximity decay,
/// token budget, and determinism as every other window (the event log and replay
/// invariant are untouched, since this composes the public recall surface).
///
/// Degradation is **visible, never silent**: when no anchor is available (empty index,
/// or the embedder failed on the query) the window comes from the free lexical
/// [`Recall::Task`] entry and `strategy` says so.
pub fn recall_semantic<E: Embedder, M: ExternalMemory + ?Sized>(
    mem: &M,
    idx: &ShardedOctaIndex<E>,
    text: &str,
    budget_tokens: usize,
) -> RecallWindow {
    finish_with_anchor(mem, idx.semantic_anchors(text, 1), text, budget_tokens)
}

/// Region-scoped [`recall_semantic`] — the validated cascade shape when the causal
/// scope is already known: the anchor is resolved **within** `region`'s shard only
/// (exact 3-D rerank inside a small region, the 99 %-hit deployment), then expanded
/// causally exactly like [`recall_semantic`].
pub fn recall_semantic_in<E: Embedder, M: ExternalMemory + ?Sized>(
    mem: &M,
    idx: &ShardedOctaIndex<E>,
    region: &str,
    text: &str,
    budget_tokens: usize,
) -> RecallWindow {
    finish_with_anchor(
        mem,
        idx.semantic_anchors_in(region, text, 1),
        text,
        budget_tokens,
    )
}

fn finish_with_anchor<M: ExternalMemory + ?Sized>(
    mem: &M,
    anchors: Vec<(String, f64)>,
    text: &str,
    budget_tokens: usize,
) -> RecallWindow {
    match anchors.into_iter().next() {
        Some((anchor, _score)) => {
            let mut w = mem.recall(&Recall::Around(anchor), budget_tokens);
            w.strategy = "octa-semantic".into();
            w
        }
        None => {
            let mut w = mem.recall(&Recall::Task(text.to_string()), budget_tokens);
            w.strategy = "octa-semantic-fallback-task".into();
            w
        }
    }
}

// ───────────────────────── CCOS-side premium gate (not vendored) ─────────────────────────

/// **Premium gate** for the OctaSoma semantic-memory tier. Compiling the backend is the
/// `octasoma` cargo feature; *using* it goes through this gate, which consults CCOS's own
/// offline license ([`Feature::OctaSomaMemory`](crate::license::Feature)) exactly like
/// [`RetrievalAccess`](crate::retrieval::RetrievalAccess) gates adaptive retrieval. On the
/// community tier `unlock` returns a [`LicenseError`](crate::license::LicenseError) (with
/// CCOS's standard no-silent-downgrade log) and the caller keeps the free core recall
/// strategies — nothing degrades.
pub struct SemanticMemoryAccess {
    #[allow(dead_code)]
    gated: (),
}

impl SemanticMemoryAccess {
    /// Unlock the OctaSoma semantic-memory tier from CCOS's `licensing` state at `now`.
    /// `Ok` only on the Pro tier; otherwise the standard `Feature::OctaSomaMemory` refusal
    /// (the core stays usable).
    pub fn unlock(
        licensing: &crate::license::Licensing,
        now: u64,
    ) -> Result<Self, crate::license::LicenseError> {
        licensing.require(crate::license::Feature::OctaSomaMemory, now)?;
        Ok(Self { gated: () })
    }

    /// Construct the **region-sharded** index (the validated deployment) — reachable only
    /// behind [`Self::unlock`].
    pub fn sharded_index<E: Embedder>(&self, embedder: E) -> ShardedOctaIndex<E> {
        ShardedOctaIndex::new(embedder)
    }

    /// Build the sharded index from a causal graph in one deterministic pass
    /// (see [`ShardedOctaIndex::index_graph`]) — the one-call path from an
    /// ingested [`CcosMemory`](crate::external_memory::CcosMemory) to semantic
    /// recall via [`recall_semantic`].
    pub fn sharded_index_from_graph<E: Embedder>(
        &self,
        embedder: E,
        graph: &MemoryGraph,
    ) -> ShardedOctaIndex<E> {
        let mut idx = ShardedOctaIndex::new(embedder);
        idx.index_graph(graph);
        idx
    }

    /// Reopen a previously saved sharded index (mirror of CCOS's `checkpoint`).
    pub fn open_sharded_index<E: Embedder>(
        &self,
        embedder: E,
        dir: &str,
    ) -> std::io::Result<ShardedOctaIndex<E>> {
        ShardedOctaIndex::open(embedder, dir)
    }

    /// Construct the single **global** index (coarse router — prefer
    /// [`Self::sharded_index`], see the module docs).
    pub fn index<E: Embedder>(&self, embedder: E) -> OctaIndex<E> {
        OctaIndex::new(embedder)
    }

    /// Reopen a previously saved global index.
    pub fn open_index<E: Embedder>(
        &self,
        embedder: E,
        path: &str,
    ) -> std::io::Result<OctaIndex<E>> {
        OctaIndex::open(embedder, path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::license::{License, Licensing};
    use octasoma::HashEmbedder;

    const NOW: u64 = 1_000;

    fn pro() -> Licensing {
        Licensing::licensed(License {
            licensee: "acme".into(),
            expires_at: None,
        })
    }

    #[test]
    fn octasoma_memory_is_gated_by_the_license() {
        // Community tier → locked (the free core recall strategies still work; only the
        // OctaSoma backend is gated).
        assert!(SemanticMemoryAccess::unlock(&Licensing::community(), NOW).is_err());
        // A valid Pro license → unlocked, and the sharded index is constructible.
        let access = SemanticMemoryAccess::unlock(&pro(), NOW).expect("pro unlocks octasoma");
        let idx = access.sharded_index(HashEmbedder::new(64));
        assert!(idx.is_empty());
    }

    #[test]
    fn region_of_derives_the_causal_region_from_node_uris() {
        assert_eq!(region_of("sym:src/db.rs:query"), "src/db.rs");
        assert_eq!(region_of("mod:src/cache.rs"), "src/cache.rs");
        assert_eq!(region_of("file:src/main.rs"), "src/main.rs");
        // No `kind:` shape → the whole URI is its own region.
        assert_eq!(region_of("plain"), "plain");
    }

    #[test]
    fn sharded_anchors_hit_within_the_causal_region_deterministically() {
        let access = SemanticMemoryAccess::unlock(&pro(), NOW).expect("pro unlocks octasoma");
        let mut idx = access.sharded_index(HashEmbedder::new(128));

        idx.index_node_in("src/db.rs", "sym:src/db.rs:query", "fn query(conn: &Conn)");
        idx.index_node_in("src/db.rs", "sym:src/db.rs:pool", "fn pool() -> Pool");
        idx.index_node("sym:src/cache.rs:get", "fn get(k)"); // derives region via region_of
        idx.index_node_in("src/cache.rs", "sym:src/cache.rs:put", "fn put(k, v)");

        assert_eq!(idx.regions(), 2);

        // HashEmbedder is exact-text: the same text must come back first with score 1.0
        // (distance² = 0 → 1/(1+0)), and the run is bit-deterministic.
        let anchors = idx.semantic_anchors_in("src/db.rs", "fn query(conn: &Conn)", 2);
        assert_eq!(
            anchors.first().map(|(u, _)| u.as_str()),
            Some("sym:src/db.rs:query")
        );
        assert_eq!(anchors.first().map(|(_, s)| *s), Some(1.0));

        // The other region's nodes never leak into an in-region query.
        assert!(anchors.iter().all(|(u, _)| u.starts_with("sym:src/db.rs")));
    }

    #[test]
    fn recall_semantic_expands_the_anchor_through_the_causal_graph() {
        use crate::external_memory::CcosMemory;

        let mut mem = CcosMemory::new();
        mem.ingest_source(
            "src/db.rs",
            "pub fn query() -> i64 { 1 }\npub fn pool() -> i64 { 2 }\n",
        );
        mem.ingest_source("src/cache.rs", "pub fn get() -> i64 { 3 }\n");

        let access = SemanticMemoryAccess::unlock(&pro(), NOW).expect("pro unlocks octasoma");
        let idx = access.sharded_index_from_graph(HashEmbedder::new(128), mem.graph());
        assert!(!idx.is_empty());

        // Query with a real node's exact content: HashEmbedder anchors on that node
        // (distance 0), and the window is its causal region — assembled by CCOS's own
        // Recall::Around machinery, so budgets/determinism are the usual ones.
        let (_, node) = mem
            .graph()
            .node_entries()
            .find(|(id, n)| id.0.contains("db.rs") && !n.content.is_empty())
            .expect("db.rs produced content-carrying nodes");
        let query = node.content.clone();

        let w = recall_semantic(&mem, &idx, &query, 512);
        assert_eq!(w.strategy, "octa-semantic");
        assert!(
            w.items.iter().any(|i| i.uri.contains("db.rs")),
            "the window covers the anchor's causal region: {:?}",
            w.items.iter().map(|i| &i.uri).collect::<Vec<_>>()
        );

        // Determinism: the same query yields the same window.
        let w2 = recall_semantic(&mem, &idx, &query, 512);
        assert_eq!(w.strategy, w2.strategy);
        assert_eq!(
            w.items.iter().map(|i| &i.uri).collect::<Vec<_>>(),
            w2.items.iter().map(|i| &i.uri).collect::<Vec<_>>()
        );

        // No anchor available (empty index) → the visible lexical fallback, never a
        // silent one.
        let empty = access.sharded_index(HashEmbedder::new(128));
        let fallback = recall_semantic(&mem, &empty, "query", 512);
        assert_eq!(fallback.strategy, "octa-semantic-fallback-task");
    }
}
