//! Latent-semantic projection (LSA / truncated SVD), distilled deterministically
//! from a corpus's TF-IDF matrix.
//!
//! This is the *learned* half of CCOS's "distill, don't couple" rule applied to
//! retrieval: rather than couple a black-box neural embedder onto the live path
//! (which would break the replay invariant), we **learn a linear projection**
//! from the session's own term co-occurrence and apply it deterministically.
//!
//! Concretely, given the document–term matrix `M` (`docs × dim`, each row a
//! node's hashed TF-IDF vector), the top-`rank` right singular vectors of `M`
//! span the latent-semantic space. They are the top eigenvectors of the small,
//! fixed-size Gram matrix `Mᵀ M` (`dim × dim`), which we diagonalise with a
//! **cyclic Jacobi** rotation sweep — zero-dependency, and reproducible bit-for-bit
//! because the sweep count and pair order are fixed and the arithmetic is `f64`.
//! Projecting a `dim`-vector `x` into the latent space is then `latent[j] =
//! dot(x, V[j])`, and cosine similarity there captures synonymy/transitivity that
//! raw TF-IDF (which needs a literal shared term) misses.

// Matrix algebra here is naturally index-based: transpose/column access and
// simultaneous two-row updates have no clean iterator form, so the
// range-loop lint is silenced for the module rather than obscuring the math.
#![allow(clippy::needless_range_loop)]

/// Top-`rank` right singular vectors of the document–term matrix `rows`
/// (`docs × dim`): the top eigenvectors of `Mᵀ M`. Returns a `rank × dim` matrix
/// `V` — project a `dim`-vector `x` into latent space with
/// `latent[j] = dot(x, V[j])`. Deterministic. Empty when there is nothing to
/// project (`dim == 0` or `rank == 0`).
pub fn lsa_projection(rows: &[Vec<f32>], rank: usize) -> Vec<Vec<f32>> {
    let dim = rows.first().map(Vec::len).unwrap_or(0);
    let rank = rank.min(dim);
    if dim == 0 || rank == 0 {
        return Vec::new();
    }
    // Gram matrix C = Mᵀ M (dim × dim), symmetric PSD. Build the upper triangle
    // from the (typically sparse) rows, then mirror.
    let mut c = vec![vec![0f64; dim]; dim];
    for row in rows {
        for i in 0..dim {
            let ri = row[i] as f64;
            if ri == 0.0 {
                continue;
            }
            let ci = &mut c[i];
            for (j, cij) in ci.iter_mut().enumerate().skip(i) {
                *cij += ri * row[j] as f64;
            }
        }
    }
    for i in 0..dim {
        for j in 0..i {
            c[i][j] = c[j][i];
        }
    }
    let (eigvals, eigvecs) = jacobi_eigen(c);
    // Order eigenpairs by eigenvalue descending; ties by index (deterministic).
    let mut order: Vec<usize> = (0..dim).collect();
    order.sort_by(|&a, &b| {
        eigvals[b]
            .partial_cmp(&eigvals[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    order
        .into_iter()
        .take(rank)
        .map(|k| {
            let mut v: Vec<f32> = (0..dim).map(|i| eigvecs[i][k] as f32).collect();
            // Deterministic sign: first non-zero component positive. (Cosine is
            // sign-invariant, but this pins the stored bytes regardless.)
            if let Some(&first) = v.iter().find(|&&x| x != 0.0) {
                if first < 0.0 {
                    for x in &mut v {
                        *x = -*x;
                    }
                }
            }
            v
        })
        .collect()
}

/// Project a `dim`-vector into the `rank`-dim latent space of a `projection`
/// (a `rank × dim` matrix from [`lsa_projection`]): `out[j] = dot(x, V[j])`.
pub fn project(x: &[f32], projection: &[Vec<f32>]) -> Vec<f32> {
    projection
        .iter()
        .map(|v| {
            v.iter()
                .zip(x)
                .map(|(&a, &b)| a as f64 * b as f64)
                .sum::<f64>() as f32
        })
        .collect()
}

/// Cyclic Jacobi eigendecomposition of a symmetric matrix `a`. Returns
/// `(eigenvalues, eigenvectors)` where eigenvector `k` is **column `k`** of the
/// returned matrix (`eigenvectors[i][k]`). A fixed sweep count makes this
/// reproducible; for a small fixed `dim` it converges comfortably.
fn jacobi_eigen(mut a: Vec<Vec<f64>>) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = a.len();
    let mut v = vec![vec![0f64; n]; n];
    for (i, vi) in v.iter_mut().enumerate() {
        vi[i] = 1.0;
    }
    const SWEEPS: usize = 16;
    for _ in 0..SWEEPS {
        // Stop early once the matrix is (numerically) diagonal.
        let mut off = 0.0;
        for p in 0..n {
            for q in (p + 1)..n {
                off += a[p][q] * a[p][q];
            }
        }
        if off <= 0.0 {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p][q];
                if apq == 0.0 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * apq);
                let t = if theta == 0.0 {
                    1.0
                } else {
                    theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt())
                };
                let cth = 1.0 / (t * t + 1.0).sqrt();
                let sth = t * cth;
                // A ← Jᵀ A J : rotate columns p,q then rows p,q.
                for row in a.iter_mut() {
                    let akp = row[p];
                    let akq = row[q];
                    row[p] = cth * akp - sth * akq;
                    row[q] = sth * akp + cth * akq;
                }
                for k in 0..n {
                    let apk = a[p][k];
                    let aqk = a[q][k];
                    a[p][k] = cth * apk - sth * aqk;
                    a[q][k] = sth * apk + cth * aqk;
                }
                // V ← V J.
                for row in v.iter_mut() {
                    let vkp = row[p];
                    let vkq = row[q];
                    row[p] = cth * vkp - sth * vkq;
                    row[q] = sth * vkp + cth * vkq;
                }
            }
        }
    }
    let eig = (0..n).map(|i| a[i][i]).collect();
    (eig, v)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matvec(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
        a.iter()
            .map(|row| row.iter().zip(x).map(|(r, v)| r * v).sum())
            .collect()
    }

    #[test]
    fn jacobi_diagonalises_a_known_symmetric_matrix() {
        // Eigenvalues of [[2,1],[1,2]] are 3 and 1.
        let a = vec![vec![2.0, 1.0], vec![1.0, 2.0]];
        let (mut eig, vecs) = jacobi_eigen(a.clone());
        eig.sort_by(|x, y| y.partial_cmp(x).unwrap());
        assert!((eig[0] - 3.0).abs() < 1e-9, "top eigenvalue ~3: {}", eig[0]);
        assert!(
            (eig[1] - 1.0).abs() < 1e-9,
            "second eigenvalue ~1: {}",
            eig[1]
        );
        // Each column is a unit eigenvector: A v ≈ λ v.
        for k in 0..2 {
            let vk: Vec<f64> = (0..2).map(|i| vecs[i][k]).collect();
            let norm = (vk[0] * vk[0] + vk[1] * vk[1]).sqrt();
            assert!((norm - 1.0).abs() < 1e-9, "eigenvector {k} is unit-norm");
            let av = matvec(&a, &vk);
            let lambda = av[0] / vk[0];
            for i in 0..2 {
                assert!(
                    (av[i] - lambda * vk[i]).abs() < 1e-7,
                    "A v = λ v for col {k}"
                );
            }
        }
    }

    #[test]
    fn jacobi_eigenvectors_are_orthonormal() {
        let a = vec![
            vec![4.0, 1.0, 0.5],
            vec![1.0, 3.0, 0.2],
            vec![0.5, 0.2, 2.0],
        ];
        let (_, v) = jacobi_eigen(a);
        // Columns are pairwise orthonormal.
        for p in 0..3 {
            for q in 0..3 {
                let dot: f64 = (0..3).map(|i| v[i][p] * v[i][q]).sum();
                let want = if p == q { 1.0 } else { 0.0 };
                assert!(
                    (dot - want).abs() < 1e-7,
                    "⟨col{p},col{q}⟩ = {dot}, want {want}"
                );
            }
        }
    }

    #[test]
    fn lsa_projection_is_deterministic() {
        let rows = vec![
            vec![1.0f32, 0.0, 2.0, 0.0],
            vec![0.0, 1.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0, 3.0],
        ];
        let a = lsa_projection(&rows, 2);
        let b = lsa_projection(&rows, 2);
        assert_eq!(a, b, "same matrix ⇒ bit-identical projection");
        assert_eq!(a.len(), 2, "rank-2 projection has 2 basis vectors");
        assert_eq!(a[0].len(), 4, "each basis vector spans the term dimension");
    }

    #[test]
    fn lsa_recovers_a_positive_synonym_link_raw_tfidf_misses() {
        // Terms [car, automobile, wheel, engine]. `car` and `automobile` co-occur
        // in doc0 (they are synonyms), so the latent space pulls them together.
        // A query for `automobile` should then match doc1 (which has `car wheel`
        // but never `automobile`) — a positive link raw TF-IDF cannot see.
        let rows = vec![
            vec![1.0f32, 1.0, 0.0, 0.0], // doc0: car + automobile
            vec![1.0, 0.0, 1.0, 0.0],    // doc1: car + wheel  (the target)
            vec![0.0, 1.0, 0.0, 1.0],    // doc2: automobile + engine
        ];
        let q = [0.0f32, 1.0, 0.0, 0.0]; // an "automobile" query
        let proj = lsa_projection(&rows, 2); // truncate to expose the latent factor
        let cos = cosine(&project(&q, &proj), &project(&rows[1], &proj));
        // Raw TF-IDF cosine of `automobile` vs doc1 is exactly 0 (disjoint terms).
        assert_eq!(
            cosine(&q, &rows[1]),
            0.0,
            "raw overlap is zero by construction"
        );
        assert!(
            cos > 0.05,
            "LSA gives the synonym-bridged doc a positive similarity (raw is 0): {cos}"
        );
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na == 0.0 || nb == 0.0 {
            0.0
        } else {
            dot / (na * nb)
        }
    }
}
