use crate::core::simd::{SimdOps, SimdDispatcher};

pub struct CosineSimilarity;

impl CosineSimilarity {
    /// Computes the cosine similarity between two vectors.
    /// If vectors are already normalized to unit length, this is equivalent to a dot product.
    pub fn compute(a: &[f32], b: &[f32]) -> f32 {
        let backend = SimdDispatcher::get_backend();
        let dot = backend.dot_product(a, b);
        let norm_a = Self::norm(a);
        let norm_b = Self::norm(b);
        dot / (norm_a * norm_b)
    }

    /// Computes the similarity for pre-normalized vectors (High Performance path).
    pub fn compute_normalized(a: &[f32], b: &[f32]) -> f32 {
        let backend = SimdDispatcher::get_backend();
        backend.dot_product(a, b)
    }

    /// Calculates the L2 norm of a vector.
    pub fn norm(v: &[f32]) -> f32 {
        let backend = SimdDispatcher::get_backend();
        let dot = backend.dot_product(v, v);
        dot.sqrt()
    }

    /// Normalizes a vector in-place.
    pub fn normalize(v: &mut [f32]) {
        let n = Self::norm(v);
        if n > 0.0 {
            let inv_n = 1.0 / n;
            for x in v.iter_mut() {
                *x *= inv_n;
            }
        }
    }

    /// Batch computes similarities between a query vector and a matrix of vectors.
    pub fn compute_batch(query: &[f32], matrix: &[&[f32]]) -> Vec<f32> {
        let backend = SimdDispatcher::get_backend();
        matrix.iter().map(|v| backend.dot_product(query, v)).collect()
    }
}
