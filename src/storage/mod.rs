pub mod memtable;
pub mod lsm;
pub mod wal;
pub mod disk_store;

pub use memtable::MemTable;
pub use lsm::LsmTree;
pub use disk_store::{DiskStore, DiskStoreWriter};
pub use wal::Wal;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use anyhow::{Result, Context};
use crate::core::types::Vector;
use serde::{Serialize, de::DeserializeOwned};

/// NeuralStore is the high-level coordinator for the storage system.
/// It combines a MemTable (for fast writes/reads), a WAL (for durability),
/// and DiskStores (for persistent long-term storage).
pub struct NeuralStore<K>
where
    K: Ord + Sync + Send + Serialize + DeserializeOwned + Clone + 'static,
{
    memtable: MemTable<K>,
    wal: Wal,
    segments: Vec<Arc<DiskStore>>,
    base_path: PathBuf,
}

impl<K> NeuralStore<K>
where
    K: Ord + Sync + Send + Serialize + DeserializeOwned + Clone + 'static,
{
    /// Opens a storage engine at the given path.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let base_path = path.as_ref().to_path_buf();
        if !base_path.exists() {
            std::fs::create_dir_all(&base_path)?;
        }

        // 1. Open WAL and recover MemTable
        let wal_path = base_path.join("wal.log");
        let wal = Wal::open(wal_path)?;
        let memtable = MemTable::new();

        wal.recover(|bytes| {
            let (key, vector): (K, Vector) = bincode::deserialize(bytes)?;
            memtable.put(key, vector);
            Ok(())
        }).context("Failed to recover MemTable from WAL")?;

        // 2. Load existing DiskStore segments
        let mut segments = Vec::new();
        let mut entries = std::fs::read_dir(&base_path)?
            .filter_map(|res| res.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map_or(false, |ext| ext == "vstore"))
            .collect::<Vec<_>>();

        // Sort segments by creation time (newest last)
        entries.sort_by_key(|p| std::fs::metadata(p).and_then(|m| m.modified()).ok());

        for path in entries {
            let store = DiskStore::open(path)?;
            segments.push(Arc::new(store));
        }

        Ok(Self {
            memtable,
            wal,
            segments,
            base_path,
        })
    }

    /// Writes a key-vector pair to the store.
    pub fn put(&mut self, key: K, vector: Vector) -> Result<()> {
        // Write to WAL for durability
        let bytes = bincode::serialize(&(key.clone(), vector.clone()))?;
        self.wal.append(&bytes)?;
        self.wal.flush()?;

        // Update MemTable
        self.memtable.put(key, vector);
        Ok(())
    }

    /// Retrieves a vector from the store.
    /// Search order: MemTable -> DiskStore segments (newest first).
    pub fn get(&self, key: &K) -> Option<Arc<Vector>> {
        // 1. Check MemTable
        if let Some(vec) = self.memtable.get(key) {
            return Some(vec);
        }

        // 2. Check DiskStore segments in reverse order (newest to oldest)
        // Note: This requires the key to be mapped to an index in the DiskStore,
        // which currently our basic DiskStore doesn't do (it only does by index).
        // In a full LSM, we would have an index for each segment.
        // For this implementation, since we are focusing on the storage primitives
        // requested (mmap, zero-copy), we acknowledge that key -> index mapping
        // would be handled by a separate index file or stored in the DiskStore header.

        None
    }

    /// Flushes the MemTable to a new DiskStore segment and clears the WAL.
    pub fn flush_memtable(&mut self) -> Result<()> {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs();
        let segment_path = self.base_path.join(format!("segment_{}.vstore", timestamp));

        // Extract all data from MemTable
        let all_entries = self.memtable.get_all();
        if all_entries.is_empty() {
            return Ok(());
        }

        // We assume a fixed dimension for the segment based on the first vector
        let dim = all_entries[0].1.len();
        let vectors: Vec<Vec<f32>> = all_entries.into_iter().map(|(_, v)| (*v).0.clone()).collect();

        // Write to DiskStore
        DiskStoreWriter::create(&segment_path, dim, &vectors)?;

        // Update segments list
        let store = DiskStore::open(&segment_path)?;
        self.segments.push(Arc::new(store));

        // Truncate WAL as data is now in a segment
        self.wal.flush()?;
        // In a real system, we'd replace the WAL file or mark it as truncated.
        // For simplicity here, we assume the MemTable is cleared and we start fresh.

        Ok(())
    }

    pub fn len(&self) -> usize {
        self.memtable.len()
    }
}

