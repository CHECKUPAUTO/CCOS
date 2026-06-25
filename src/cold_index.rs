//! Lever 2, brick 1 — an on-disk **sorted key→bytes segment** with a sparse
//! resident index. The foundation for bounding the COLD *entry count*
//! (`docs/DESIGN_cold_entry_count.md`): husks will live in segments like these,
//! keyed by `NodeId`, so the resident footprint becomes `O(N / stride)` (the sparse
//! index) plus a bounded cache instead of one struct per node.
//!
//! Dependency-free, `std` only. A segment is immutable once written (LSM-style); a
//! lookup binary-searches the in-RAM sparse index to a start offset, then scans at
//! most `STRIDE` records forward — the file is sorted, so a key greater than the
//! target ends the scan. Written atomically (`util::write_durable`), so a crash
//! never leaves a half-segment.
//!
//! This brick is **standalone and unwired**: it is exercised only by its own tests
//! here; a later brick threads serialized husks through it. Like the LZSS codec, it
//! is verified by a property round-trip before anything depends on it.
//!
//! ## Format
//! ```text
//! records:  [ u32 key_len | key | u32 val_len | val ] *   (ascending key order)
//! sparse:   u32 count | [ u32 key_len | key | u64 offset ] *   (every STRIDE-th record)
//! footer:   u64 sparse_offset | u64 MAGIC                  (fixed 16 bytes at EOF)
//! ```

use crate::util::write_durable;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

const MAGIC: u64 = 0xC01D_1DEC_0000_0001;
const STRIDE: usize = 64; // one sparse entry per 64 records

fn put_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}
fn put_bytes(buf: &mut Vec<u8>, b: &[u8]) {
    put_u32(buf, b.len() as u32);
    buf.extend_from_slice(b);
}

/// Write `records` (which **must be sorted by key, ascending, no duplicates**) to a
/// new segment at `path`, atomically. Returns an error if the records are unsorted.
pub fn write_segment(path: &Path, records: &[(&str, &[u8])]) -> io::Result<()> {
    for w in records.windows(2) {
        if w[0].0 >= w[1].0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "segment records must be strictly ascending by key",
            ));
        }
    }
    let mut buf = Vec::new();
    let mut sparse: Vec<(&str, u64)> = Vec::new();
    for (i, (k, v)) in records.iter().enumerate() {
        if i % STRIDE == 0 {
            sparse.push((k, buf.len() as u64));
        }
        put_bytes(&mut buf, k.as_bytes());
        put_bytes(&mut buf, v);
    }
    let sparse_offset = buf.len() as u64;
    put_u32(&mut buf, sparse.len() as u32);
    for (k, off) in &sparse {
        put_bytes(&mut buf, k.as_bytes());
        buf.extend_from_slice(&off.to_le_bytes());
    }
    buf.extend_from_slice(&sparse_offset.to_le_bytes());
    buf.extend_from_slice(&MAGIC.to_le_bytes());
    write_durable(path, &buf)
}

/// A read handle over a written segment: the sparse index lives in RAM, the records
/// stay on disk and are read on demand.
pub struct Segment {
    path: PathBuf,
    sparse: Vec<(String, u64)>,
    records_end: u64,
}

impl Segment {
    /// Open a segment written by [`write_segment`], loading only its sparse index.
    pub fn open(path: impl Into<PathBuf>) -> io::Result<Self> {
        let path = path.into();
        let mut f = std::fs::File::open(&path)?;
        let len = f.seek(SeekFrom::End(0))?;
        if len < 16 {
            return Err(corrupt("segment shorter than its footer"));
        }
        f.seek(SeekFrom::Start(len - 16))?;
        let sparse_offset = read_u64(&mut f)?;
        if read_u64(&mut f)? != MAGIC {
            return Err(corrupt("bad segment magic"));
        }
        if sparse_offset > len - 16 {
            return Err(corrupt("sparse offset past end"));
        }
        f.seek(SeekFrom::Start(sparse_offset))?;
        let count = read_u32(&mut f)? as usize;
        let mut sparse = Vec::with_capacity(count);
        for _ in 0..count {
            let key = read_string(&mut f)?;
            let off = read_u64(&mut f)?;
            sparse.push((key, off));
        }
        Ok(Segment {
            path,
            sparse,
            records_end: sparse_offset,
        })
    }

    /// Number of sparse-index entries (≈ `record_count / STRIDE`) — the resident cost.
    pub fn sparse_len(&self) -> usize {
        self.sparse.len()
    }

    /// Fetch the value for `key`, or `None` if absent. One sparse binary search plus
    /// a scan of at most `STRIDE` records.
    pub fn get(&self, key: &str) -> io::Result<Option<Vec<u8>>> {
        if self.sparse.is_empty() {
            return Ok(None);
        }
        // Largest sparse key ≤ `key` (records before the first sparse key can't exist,
        // since the first record is always a sparse entry).
        let start = match self.sparse.binary_search_by(|(k, _)| k.as_str().cmp(key)) {
            Ok(i) => return self.read_at(self.sparse[i].1, key), // exact sparse key
            Err(0) => return Ok(None),                           // before the first key
            Err(i) => self.sparse[i - 1].1,
        };
        self.read_at(start, key)
    }

    fn read_at(&self, start: u64, key: &str) -> io::Result<Option<Vec<u8>>> {
        let mut f = std::fs::File::open(&self.path)?;
        f.seek(SeekFrom::Start(start))?;
        let mut pos = start;
        while pos < self.records_end {
            let k = read_string(&mut f)?;
            let v = read_bytes(&mut f)?;
            pos += 8 + k.len() as u64 + v.len() as u64;
            match k.as_str().cmp(key) {
                std::cmp::Ordering::Equal => return Ok(Some(v)),
                std::cmp::Ordering::Greater => return Ok(None), // sorted: passed it
                std::cmp::Ordering::Less => {}
            }
        }
        Ok(None)
    }
}

fn corrupt(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}
fn read_u32(f: &mut impl Read) -> io::Result<u32> {
    let mut b = [0u8; 4];
    f.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}
fn read_u64(f: &mut impl Read) -> io::Result<u64> {
    let mut b = [0u8; 8];
    f.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}
fn read_bytes(f: &mut impl Read) -> io::Result<Vec<u8>> {
    let len = read_u32(f)? as usize;
    let mut b = vec![0u8; len];
    f.read_exact(&mut b)?;
    Ok(b)
}
fn read_string(f: &mut impl Read) -> io::Result<String> {
    String::from_utf8(read_bytes(f)?).map_err(|_| corrupt("non-utf8 key"))
}

/// Lever 2, brick 2 — a staged on-disk key→bytes store: a resident write buffer
/// (the "memtable", a `BTreeMap`) flushed to an immutable sorted [`Segment`] once it
/// grows past `buffer_limit`, with reads layered **newest-first** (buffer, then
/// segments newest→oldest). Updates are last-write-wins; a re-`open` reloads the
/// segments, so the store survives a restart.
///
/// Resident cost is the buffer (bounded by `buffer_limit`) plus each segment's
/// sparse index — `O(total / STRIDE)` overall, not one entry per key, which is what
/// will let the COLD tier hold far more husks than fit in RAM. Still **unwired**;
/// deletion (tombstones) and segment compaction are later bricks.
pub struct HuskStore {
    dir: PathBuf,
    buffer: std::collections::BTreeMap<String, Vec<u8>>,
    buffer_limit: usize,
    segments: Vec<Segment>, // oldest → newest
    next_seq: u64,
}

impl HuskStore {
    /// Open (creating if needed) a store under `dir`, reloading any segments left by
    /// a previous run. `buffer_limit` is the memtable size that triggers a flush.
    pub fn open(dir: impl Into<PathBuf>, buffer_limit: usize) -> io::Result<Self> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)?;
        let mut found: Vec<(u64, PathBuf)> = Vec::new();
        for entry in std::fs::read_dir(&dir)? {
            let path = entry?.path();
            if let Some(seq) = path
                .file_name()
                .and_then(|n| n.to_str())
                .and_then(|n| n.strip_prefix("seg-"))
                .and_then(|n| n.parse::<u64>().ok())
            {
                found.push((seq, path));
            }
        }
        found.sort_by_key(|(seq, _)| *seq);
        let next_seq = found.last().map_or(0, |(seq, _)| seq + 1);
        let segments = found
            .iter()
            .map(|(_, p)| Segment::open(p))
            .collect::<io::Result<Vec<_>>>()?;
        Ok(Self {
            dir,
            buffer: std::collections::BTreeMap::new(),
            buffer_limit: buffer_limit.max(1),
            segments,
            next_seq,
        })
    }

    /// Insert/update `key`, flushing the memtable to a segment if it is now full.
    pub fn put(&mut self, key: &str, value: Vec<u8>) -> io::Result<()> {
        self.buffer.insert(key.to_string(), value);
        if self.buffer.len() >= self.buffer_limit {
            self.flush()?;
        }
        Ok(())
    }

    /// Fetch the current value for `key` — the memtable wins, else the newest segment
    /// that has it (last-write-wins).
    pub fn get(&self, key: &str) -> io::Result<Option<Vec<u8>>> {
        if let Some(v) = self.buffer.get(key) {
            return Ok(Some(v.clone()));
        }
        for seg in self.segments.iter().rev() {
            if let Some(v) = seg.get(key)? {
                return Ok(Some(v));
            }
        }
        Ok(None)
    }

    /// Flush the memtable to a new immutable segment (a no-op if empty).
    pub fn flush(&mut self) -> io::Result<()> {
        if self.buffer.is_empty() {
            return Ok(());
        }
        let path = self.dir.join(format!("seg-{:020}", self.next_seq));
        let recs: Vec<(&str, &[u8])> = self
            .buffer
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_slice()))
            .collect();
        write_segment(&path, &recs)?;
        self.segments.push(Segment::open(&path)?);
        self.next_seq += 1;
        self.buffer.clear();
        Ok(())
    }

    /// Total resident index entries: the memtable plus every segment's sparse index —
    /// the bounded `O(total / STRIDE)` footprint, not one per key.
    pub fn resident_index_len(&self) -> usize {
        self.buffer.len() + self.segments.iter().map(Segment::sparse_len).sum::<usize>()
    }

    /// Number of flushed segments (for tests/observability).
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn tmp(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!("ccos_coldidx_{}_{}", tag, std::process::id()))
    }

    #[test]
    fn round_trips_and_misses() {
        let path = tmp("rt");
        let recs: Vec<(&str, &[u8])> = vec![
            ("file:a", b"husk-a"),
            ("file:b", b""),
            ("sym:a:f", b"\x00\x01\x02longer-value"),
        ];
        write_segment(&path, &recs).unwrap();
        let seg = Segment::open(&path).unwrap();
        for (k, v) in &recs {
            assert_eq!(seg.get(k).unwrap().as_deref(), Some(*v), "get {k}");
        }
        assert_eq!(seg.get("file:zzz").unwrap(), None, "absent key past end");
        assert_eq!(seg.get("aaa").unwrap(), None, "absent key before start");
        assert_eq!(
            seg.get("file:aa").unwrap(),
            None,
            "absent key in the middle"
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn rejects_unsorted_and_bad_magic() {
        let path = tmp("bad");
        assert!(write_segment(&path, &[("b", b"1"), ("a", b"2")]).is_err());
        // A truncated / non-segment file is a clean error, not a panic.
        write_durable(&path, b"not a segment at all").unwrap();
        assert!(Segment::open(&path).is_err());
        std::fs::remove_file(&path).ok();
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(120))]

        /// For any sorted, deduped key set (well past STRIDE so the sparse index has
        /// many entries), every key reads back its exact value and absent keys miss.
        #[test]
        fn segment_round_trips_any_sorted_map(
            map in prop::collection::btree_map(
                "[a-z][a-z0-9:]{0,12}",
                prop::collection::vec(any::<u8>(), 0..40),
                0..400,
            )
        ) {
            let path = tmp("prop");
            let recs: Vec<(&str, &[u8])> =
                map.iter().map(|(k, v)| (k.as_str(), v.as_slice())).collect();
            write_segment(&path, &recs).unwrap();
            let seg = Segment::open(&path).unwrap();
            for (k, v) in &map {
                let got = seg.get(k).unwrap();
                prop_assert_eq!(got.as_deref(), Some(v.as_slice()), "key {}", k);
            }
            prop_assert_eq!(seg.get("\u{7f}absent").unwrap(), None);
            std::fs::remove_file(&path).ok();
        }
    }

    use std::sync::atomic::{AtomicU64, Ordering};
    static CASE: AtomicU64 = AtomicU64::new(0);
    fn store_dir() -> PathBuf {
        let n = CASE.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("ccos_huskstore_{}_{}", std::process::id(), n))
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(80))]

        /// HuskStore must behave as a last-write-wins map under any put/flush/get
        /// stream — including across a re-open, which reloads the segments from disk.
        #[test]
        fn husk_store_matches_a_btreemap_model(
            ops in prop::collection::vec(
                (0u8..3u8, "[a-z]{1,3}", prop::collection::vec(any::<u8>(), 0..16)),
                1..200,
            )
        ) {
            let dir = store_dir();
            let mut store = HuskStore::open(&dir, 8).unwrap(); // tiny buffer ⇒ frequent flushes
            let mut model: std::collections::BTreeMap<String, Vec<u8>> = Default::default();
            for (op, k, v) in &ops {
                match op % 3 {
                    0 => {
                        store.put(k, v.clone()).unwrap();
                        model.insert(k.clone(), v.clone());
                    }
                    1 => store.flush().unwrap(),
                    _ => {
                        let got = store.get(k).unwrap();
                        prop_assert_eq!(got.as_ref(), model.get(k));
                    }
                }
            }
            for (k, v) in &model {
                let got = store.get(k).unwrap();
                prop_assert_eq!(got.as_ref(), Some(v), "live get {}", k);
            }
            // Survives a flush + re-open (segments reloaded from disk).
            store.flush().unwrap();
            let reopened = HuskStore::open(&dir, 8).unwrap();
            for (k, v) in &model {
                let got = reopened.get(k).unwrap();
                prop_assert_eq!(got.as_ref(), Some(v), "post-reopen get {}", k);
            }
            std::fs::remove_dir_all(&dir).ok();
        }
    }
}
