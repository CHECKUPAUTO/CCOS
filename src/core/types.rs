/// Core types for the vector store system.

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Vector(pub Vec<f32>);

impl Vector {
    /// Creates a new Vector from a Vec<f32>.
    pub fn new(data: Vec<f32>) -> Self {
        Self(data)
    }

    /// Returns the dimensionality of the vector.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns the vector data as a slice.
    pub fn as_slice(&self) -> &[f32] {
        &self.0
    }

    /// Converts the Vector into the underlying Vec<f32>.
    pub fn into_vec(self) -> Vec<f32> {
        self.0
    }
}

/// Supported distance metrics for vector similarity search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    /// Euclidean distance (L2 norm).
    L2,
    /// Cosine similarity.
    Cosine,
    /// Dot product / Inner product.
    InnerProduct,
}

/// Shared storage constants.
pub mod constants {
    /// Default dimensionality for vectors if not specified.
    pub const DEFAULT_DIMENSION: usize = 128;

    /// Maximum number of entries allowed in a single segment by default.
    pub const MAX_SEGMENT_SIZE: usize = 65536;

    /// The size of the memory-mapped file header in bytes.
    pub const HEADER_SIZE: usize = 1024;
}
