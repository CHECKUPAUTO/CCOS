pub mod core;
pub mod storage;
pub mod brain;
pub mod ffi;

pub use storage::LsmTree;
pub use core::types::{Vector, Metric};
pub use brain::BrainWorkers;
pub use core::search::SearchEngine;

use anyhow::Result;
use std::path::Path;

/// Top-level orchestrator for the Neural Store.
/// It coordinates storage (LsmTree), search (SearchEngine), and background workers (BrainWorkers).
pub struct NeuralStore {
    lsm: LsmTree<usize>,
    workers: BrainWorkers,
}

impl NeuralStore {
    /// Initializes a new NeuralStore or opens an existing one from the given path.
    ///
    /// Initialization sequence:
    /// 1. Load WAL & Restore MemTable (handled by LsmTree::open)
    /// 2. Start Background Workers
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        // Step 1: Initialize Storage (WAL recovery is internal to LsmTree::open)
        let lsm = LsmTree::<usize>::open(path)?;

        // Step 2: Start Background Workers
        let workers = BrainWorkers::start()?;

        Ok(Self { lsm, workers })
    }

    /// Adds a new vector to the store.
    pub fn put(&mut self, id: usize, vector: Vector) -> Result<()> {
        self.lsm.put(id, vector)
    }

    /// Retrieves a vector by its ID.
    pub fn get(&self, id: &usize) -> Option<std::sync::Arc<Vector>> {
        self.lsm.get(id)
    }

    /// Performs a similarity search over all stored vectors.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        let entries = self.lsm.get_all();
        if entries.is_empty() {
            return Vec::new();
        }

        // Convert LsmTree vectors to the format expected by SearchEngine
        let data: Vec<Vec<f32>> = entries
            .iter()
            .map(|(_, vec)| vec.0.clone())
            .collect();

        // Extract IDs for mapping search results back to store IDs
        let ids: Vec<usize> = entries.iter().map(|(id, _)| *id).collect();

        // Perform cosine search
        let raw_results = SearchEngine::cosine_search(query, &data, k);

        // Map the index-based results from SearchEngine back to our store IDs
        raw_results
            .into_iter()
            .filter_map(|(idx, score)| {
                ids.get(idx).map(|&id| (id, score))
            })
            .collect()
    }

    /// Returns the current number of elements in the store.
    pub fn len(&self) -> usize {
        self.lsm.len()
    }
}
