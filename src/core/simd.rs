use std::arch::*;

/// Trait defining primitive SIMD operations for vector math.
pub trait SimdOps: Send + Sync {
    fn dot_product(&self, a: &[f32], b: &[f32]) -> f32;
    fn subtract(&self, a: &[f32], b: &[f32], out: &mut [f32]);
}

// --- Scalar Fallback ---
pub struct ScalarOps;

impl SimdOps for ScalarOps {
    #[inline]
    fn dot_product(&self, a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "Slices must have the same length");
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    #[inline]
    fn subtract(&self, a: &[f32], b: &[f32], out: &mut [f32]) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), out.len());
        for i in 0..a.len() {
            out[i] = a[i] - b[i];
        }
    }
}

// --- x86 AVX2 Implementation ---
#[cfg(target_arch = "x86_64")]
pub struct Avx2Ops;

#[cfg(target_arch = "x86_64")]
impl SimdOps for Avx2Ops {
    fn dot_product(&self, a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        unsafe {
            use std::arch::x86_64::*;
            let mut sum = _mm256_setzero_ps();
            let chunks = a.len() / 8;
            for i in 0..chunks {
                let va = _mm256_loadu_ps(a.as_ptr().add(i * 8));
                let vb = _mm256_loadu_ps(b.as_ptr().add(i * 8));
                sum = _mm256_fmadd_ps(va, vb, sum);
            }

            // Horizontal sum of the register
            let mut res = [0.0f32; 8];
            _mm256_storeu_ps(res.as_mut_ptr(), sum);
            let remaining = a.len() % 8;
            let mut total = res.iter().sum::<f32>();
            for i in (a.len() - remaining)..a.len() {
                total += a[i] * b[i];
            }
            total
        }
    }

    fn subtract(&self, a: &[f32], b: &[f32], out: &mut [f32]) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), out.len());
        unsafe {
            use std::arch::x86_64::*;
            let chunks = a.len() / 8;
            for i in 0..chunks {
                let va = _mm256_loadu_ps(a.as_ptr().add(i * 8));
                let vb = _mm256_loadu_ps(b.as_ptr().add(i * 8));
                let vr = _mm256_sub_ps(va, vb);
                _mm256_storeu_ps(out.as_mut_ptr().add(i * 8), vr);
            }
            for i in (a.len() - (a.len() % 8))..a.len() {
                out[i] = a[i] - b[i];
            }
        }
    }
}

// --- x86 AVX-512 Implementation ---
#[cfg(target_arch = "x86_64")]
pub struct Avx512Ops;

#[cfg(target_arch = "x86_64")]
impl SimdOps for Avx512Ops {
    fn dot_product(&self, a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        unsafe {
            use std::arch::x86_64::*;
            let mut sum = _mm512_setzero_ps();
            let chunks = a.len() / 16;
            for i in 0..chunks {
                let va = _mm512_loadu_ps(a.as_ptr().add(i * 16));
                let vb = _mm512_loadu_ps(b.as_ptr().add(i * 16));
                sum = _mm512_fmadd_ps(va, vb, sum);
            }

            let mut res = [0.0f32; 16];
            _mm512_storeu_ps(res.as_mut_ptr(), sum);
            let remaining = a.len() % 16;
            let mut total = res.iter().sum::<f32>();
            for i in (a.len() - remaining)..a.len() {
                total += a[i] * b[i];
            }
            total
        }
    }

    fn subtract(&self, a: &[f32], b: &[f32], out: &mut [f32]) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), out.len());
        unsafe {
            use std::arch::x86_64::*;
            let chunks = a.len() / 16;
            for i in 0..chunks {
                let va = _mm512_loadu_ps(a.as_ptr().add(i * 16));
                let vb = _mm512_loadu_ps(b.as_ptr().add(i * 16));
                let vr = _mm512_sub_ps(va, vb);
                _mm512_storeu_ps(out.as_mut_ptr().add(i * 16), vr);
            }
            for i in (a.len() - (a.len() % 16))..a.len() {
                out[i] = a[i] - b[i];
            }
        }
    }
}

// --- ARM Neon Implementation ---
#[cfg(target_arch = "aarch64")]
pub struct NeonOps;

#[cfg(target_arch = "aarch64")]
impl SimdOps for NeonOps {
    fn dot_product(&self, a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        unsafe {
            use std::arch::aarch64::*;
            let mut sum = vdupq_n_f32(0.0);
            let chunks = a.len() / 4;
            for i in 0..chunks {
                let va = vld1q_f32(a.as_ptr().add(i * 4));
                let vb = vld1q_f32(b.as_ptr().add(i * 4));
                sum = vfmaq_f32(sum, va, vb);
            }

            let mut res = [0.0f32; 4];
            vst1q_f32(res.as_mut_ptr(), sum);
            let remaining = a.len() % 4;
            let mut total = res.iter().sum::<f32>();
            for i in (a.len() - remaining)..a.len() {
                total += a[i] * b[i];
            }
            total
        }
    }

    fn subtract(&self, a: &[f32], b: &[f32], out: &mut [f32]) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), out.len());
        unsafe {
            use std::arch::aarch64::*;
            let chunks = a.len() / 4;
            for i in 0..chunks {
                let va = vld1q_f32(a.as_ptr().add(i * 4));
                let vb = vld1q_f32(b.as_ptr().add(i * 4));
                let vr = vsubq_f32(va, vb);
                vst1q_f32(out.as_mut_ptr().add(i * 4), vr);
            }
            for i in (a.len() - (a.len() % 4))..a.len() {
                out[i] = a[i] - b[i];
            }
        }
    }
}

/// Dispatcher to select the most efficient SIMD backend at runtime.
pub struct SimdDispatcher;

impl SimdDispatcher {
    pub fn get_backend() -> Box<dyn SimdOps + Send + Sync> {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            if is_x86_feature_detected!("avx512f") {
                return Box::new(Avx512Ops);
            }
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                return Box::new(Avx2Ops);
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            // Neon is standard on aarch64
            return Box::new(NeonOps);
        }

        Box::new(ScalarOps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_near(a: f32, b: f32, epsilon: f32) {
        let diff = (a - b).abs();
        let rel_diff = diff / a.abs().max(b.abs()).max(1.0);
        assert!(diff < epsilon || rel_diff < epsilon, "Values {} and {} are not close enough (diff: {}, rel_diff: {})", a, b, diff, rel_diff);
    }

    fn generate_random_vec(len: usize) -> Vec<f32> {
        // Use a simple deterministic pseudo-random sequence for repeatability
        (0..len).map(|i| (i as f32 * 0.12345).sin()).collect()
    }

    #[test]
    fn test_simd_correctness() {
        let backend = SimdDispatcher::get_backend();
        let scalar = ScalarOps;
        let sizes = [0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 1024];

        for &size in &sizes {
            // Test dot_product
            let a = generate_random_vec(size);
            let b = generate_random_vec(size);

            let scalar_res = scalar.dot_product(&a, &b);
            let simd_res = backend.dot_product(&a, &b);

            assert_near(scalar_res, simd_res, 1e-4);

            // Test subtract
            let mut out_scalar = vec![0.0; size];
            let mut out_simd = vec![0.0; size];

            scalar.subtract(&a, &b, &mut out_scalar);
            backend.subtract(&a, &b, &mut out_simd);

            for i in 0..size {
                assert_near(out_scalar[i], out_simd[i], 1e-6);
            }
        }
    }

    #[test]
    fn test_scalar_correctness() {
        let scalar = ScalarOps;
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert_eq!(scalar.dot_product(&a, &b), 4.0 + 10.0 + 18.0);

        let mut out = vec![0.0; 3];
        scalar.subtract(&a, &b, &mut out);
        assert_eq!(out, vec![-3.0, -3.0, -3.0]);
    }
}

