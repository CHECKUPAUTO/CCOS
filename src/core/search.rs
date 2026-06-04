use rayon::prelude::*;
use crate::core::metrics::{CosineSimilarity, MahalanobisDistance};

pub struct SearchEngine;

impl SearchEngine {
    /// Performs a parallel similarity search using Cosine Similarity.
    /// Returns the indices of the top K most similar vectors.
    pub fn cosine_search(query: &[f32], data: &[Vec<f32>], k: usize) -> Vec<(usize, f32)> {
        // Ensure query is normalized for high-performance path
        let mut norm_query = query.to_vec();
        CosineSimilarity::normalize(&mut norm_query);

        // Parallel map using Rayon with chunking to optimize cache locality
        let scores: Vec<(usize, f32)> = data.par_iter()
            .enumerate()
            .map(|(idx, vec)| {
                // For maximum performance, we assume the database vectors are already normalized.
                let score = CosineSimilarity::compute_normalized(&norm_query, vec);
                (idx, score)
            })
            .collect();

        // Sort and take top K
        let mut sorted = scores;
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        sorted.into_iter().take(k).collect()
    }

    /// Performs a parallel search using Mahalanobis Distance.
    pub fn mahalanobis_search(query: &[f32], data: &[Vec<f32>], w_inv: &[f32], k: usize) -> Vec<(usize, f32)> {
        let scores: Vec<(usize, f32)> = data.par_iter()
            .enumerate()
            .map(|(idx, vec)| {
                let dist = MahalanobisDistance::compute_dense(query, vec, w_inv);
                (idx, dist)
            })
            .collect();

        // For distance, we want the SMALLEST values (closest)
        let mut sorted = scores;
        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        sorted.into_iter().take(k).collect()
    }
}
