use crate::core::types::Vector;
use crate::storage::memtable::MemTable;
use crate::storage::wal::Wal;
use serde::{Serialize, Deserialize};
use std::path::Path;
use anyhow::Result;
use std::sync::Arc;

/// A Log-Structured Merge Tree implementation.
/// It manages a write-ahead log (WAL) for durability and a MemTable for fast access.
pub struct LsmTree<K>
where
    K: Ord + Sync + Send + Serialize + for<'de> Deserialize<'de> + Clone + 'static,
{
    memtable: MemTable<K>,
    wal: Wal,
}

impl<K> LsmTree<K>
where
    K: Ord + Sync + Send + Serialize + for<'de> Deserialize<'de> + Clone + 'static,
{
    /// Opens an existing LSM tree or creates a new one at the given path.
    /// This process includes recovering the MemTable from the WAL.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let base_path = path.as_ref();
        if !base_path.exists() {
            std::fs::create_dir_all(base_path)?;
        }

        let wal_path = base_path.join("wal.log");
        let wal = Wal::open(wal_path)?;
        let memtable = MemTable::new();

        // Recovery: Read WAL and populate MemTable
        wal.recover(|bytes| {
            let (key, vector): (K, Vector) = bincode::deserialize(bytes)?;
            memtable.put(key, vector);
            Ok(())
        })?;

        Ok(Self { memtable, wal })
    }

    /// Writes a key-vector pair to the store.
    /// Flow: Write to WAL -> Sync -> Insert into MemTable.
    pub fn put(&mut self, key: K, vector: Vector) -> Result<()> {
        // 1. Serialize for WAL
        let bytes = bincode::serialize(&(key.clone(), vector.clone()))?;

        // 2. Write to WAL
        self.wal.append(&bytes)?;

        // 3. Sync WAL to disk
        self.wal.flush()?;

        // 4. Insert into MemTable
        self.memtable.put(key, vector);

        Ok(())
    }

    /// Retrieves a vector from the store.
    pub fn get(&self, key: &K) -> Option<std::sync::Arc<Vector>> {
        self.memtable.get(key)
    }

    /// Removes a key-vector pair from the store.
    /// Note: In a full LSM implementation, this would append a tombstone to the WAL.
    pub fn remove(&mut self, key: &K) -> Result<()> {
        // For now, we'll just remove it from MemTable as per current requirements,
        // but ideally this should be logged in WAL too.
        self.memtable.remove(key);
        Ok(())
    }

    /// Returns the number of elements in the active MemTable.
    pub fn len(&self) -> usize {
        self.memtable.len()
    }

    /// Returns all key-vector pairs from the store.
    pub fn get_all(&self) -> Vec<(K, std::sync::Arc<Vector>)> {
        self.memtable.get_all()
    }

    /// Checks if the active MemTable is empty.
    pub fn is_empty(&self) -> bool {
        self.memtable.is_empty()
    }
}

impl<K> Default for LsmTree<K>
where
    K: Ord + Sync + Send + Serialize + for<'de> Deserialize<'de> + Clone + 'static,
{
    // We can't implement Default in a way that opens a file,
    // so this is provided for completeness if needed.
    fn default() -> Self {
        panic!("LsmTree must be opened using LsmTree::open()");
    }
}
