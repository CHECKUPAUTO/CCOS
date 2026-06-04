use crossbeam_skiplist::SkipMap;
use crate::core::types::Vector;
use std::sync::Arc;

/// A lock-free concurrent memory table using a SkipMap.
/// This serves as the active writable layer in the storage engine,
/// providing fast insertions and lookups before data is flushed to disk.
pub struct MemTable<K>
where
    K: Ord + Sync + Send + 'static
{
    map: SkipMap<K, Arc<Vector>>,
}

impl<K> MemTable<K>
where
    K: Ord + Sync + Send + 'static
{
    /// Creates a new empty MemTable.
    pub fn new() -> Self {
        Self {
            map: SkipMap::new(),
        }
    }

    /// Inserts a key-vector pair into the memory table.
    /// If the key already exists, the value is updated.
    pub fn put(&self, key: K, vector: Vector) {
        self.map.insert(key, Arc::new(vector));
    }

    /// Retrieves a vector from the memory table by its key.
    /// Returns an Arc to avoid cloning the underlying Vector data.
    pub fn get(&self, key: &K) -> Option<Arc<Vector>> {
        self.map.get(key).map(|entry| Arc::clone(entry.value()))
    }

    /// Removes a key-vector pair from the memory table.
    pub fn remove(&self, key: &K) {
        self.map.remove(key);
    }

    /// Returns an iterator over entries in the specified range [start, end).
    /// The keys in the output are cloned.
    pub fn range<'a>(&'a self, start: &'a K, end: &'a K) -> impl Iterator<Item = (K, Arc<Vector>)> + 'a
    where
        K: Clone
    {
        self.map.range(start..end).map(move |entry| (entry.key().clone(), Arc::clone(entry.value())))
    }

    /// Returns all entries in the memory table.
    pub fn get_all(&self) -> Vec<(K, Arc<Vector>)>
    where
        K: Clone
    {
        self.map.iter().map(|entry| (entry.key().clone(), Arc::clone(entry.value()))).collect()
    }

    /// Returns the number of entries currently in the memory table.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Checks if the memory table is empty.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

impl<K> Default for MemTable<K>
where
    K: Ord + Sync + Send + 'static
{
    fn default() -> Self {
        Self::new()
    }
}
