use crate::core::simd::{SimdOps, SimdDispatcher};

pub struct MahalanobisDistance;

impl MahalanobisDistance {
    /// Computes the Mahalanobis distance using a dense inverse covariance matrix W_inv.
    /// Formula: sqrt((x-y)T * W_inv * (x-y))
    pub fn compute_dense(x: &[f32], y: &[f32], w_inv: &[f32]) -> f32 {
        assert_eq!(x.len(), y.len());
        let dim = x.len();
        assert_eq!(w_inv.len(), dim * dim);

        let backend = SimdDispatcher::get_backend();
        let mut diff = vec![0.0f32; dim];
        backend.subtract(x, y, &mut diff);

        // Correct quadratic form calculation xT * W * x
        let mut quad_form = 0.0;
        for i in 0..dim {
            let row = &w_inv[i*dim .. (i+1)*dim];
            let dot = backend.dot_product(row, &diff);
            quad_form += diff[i] * dot;
        }

        quad_form.sqrt()
    }

    /// Computes the Mahalanobis distance for a diagonal inverse covariance matrix (Weighted Euclidean).
    /// W_inv is provided as a vector of weights.
    pub fn compute_diagonal(x: &[f32], y: &[f32], w_diag: &[f32]) -> f32 {
        assert_eq!(x.len(), y.len());
        assert_eq!(x.len(), w_diag.len());

        let backend = SimdDispatcher::get_backend();
        let mut diff = vec![0.0f32; x.len()];
        backend.subtract(x, y, &mut diff);

        // Weighted dot product: sum (diff[i]^2 * w_diag[i])
        let mut quad_form = 0.0;
        for i in 0..x.len() {
            quad_form += diff[i] * diff[i] * w_diag[i];
        }

        quad_form.sqrt()
    }
}
