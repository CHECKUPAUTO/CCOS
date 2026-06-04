use neural_store::core::{simd::{SimdOps, ScalarOps, SimdDispatcher}, metrics::CosineSimilarity};
use std::time::Instant;

#[test]
fn test_simd_correctness() {
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b = vec![0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5];

    // Scalar result
    let scalar_res = ScalarOps::dot_product(&a, &b);

    // Dispatcher result
    let backend = SimdDispatcher::get_backend();
    let simd_res = backend.dot_product(&a, &b);

    assert!((scalar_res - simd_res).abs() < 1e-5, "SIMD result {} differs from scalar {}", simd_res, scalar_res);
}

#[test]
fn test_cosine_correctness() {
    let a = vec![1.0, 0.0];
    let b = vec![0.0, 1.0];
    let res = CosineSimilarity::compute(&a, &b);
    assert!((res - 0.0).abs() < 1e-5);

    let c = vec![1.0, 1.0];
    let d = vec![1.0, 1.0];
    let res2 = CosineSimilarity::compute(&c, &d);
    assert!((res2 - 1.0).abs() < 1e-5);
}

#[test]
fn benchmark_throughput() {
    let dim = 128;
    let num_vectors = 10_000;
    let query = vec![0.1f32; dim];
    let data: Vec<Vec<f32>> = (0..num_vectors).map(|_| vec![0.2f32; dim]).collect();

    let start = Instant::now();
    // We simulate a batch search using the dispatcher's dot product
    let backend = SimdDispatcher::get_backend();
    for v in &data {
        backend.dot_product(&query, v);
    }
    let duration = start.elapsed();

    let mvps = (num_vectors as f64) / (duration.as_secs_f64() * 1_000_000.0);
    println!("Throughput: {:.4} MVPS", mvps);
}
