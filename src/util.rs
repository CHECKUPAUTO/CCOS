//! Small shared utilities used across the kernel.

use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{self, Write};
use std::path::Path;

/// Hex-encoded SHA-256 of a string — the canonical content hash used
/// throughout CCOS (file hashes, prompt/response hashes, chain links).
pub fn sha256_hex(input: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Write `bytes` to `path` **durably and atomically**: write to a temporary
/// sibling, `fsync` it, rename it over `path`, then best-effort `fsync` the
/// parent directory. After this returns the data has reached stable storage and
/// `path` is never left half-written — the basis of CCOS's "replayable after a
/// crash" guarantee. A plain [`std::fs::write`] only reaches the kernel page
/// cache, so a power loss or daemon crash can corrupt or truncate the file. The
/// extra cost is one `fsync`, negligible at an agent's inference cadence.
pub fn write_durable(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let mut tmp = path.as_os_str().to_os_string();
    tmp.push(".tmp");
    let tmp = std::path::PathBuf::from(tmp);
    {
        let mut f = File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?; // flush contents + metadata to disk before we rename
    }
    std::fs::rename(&tmp, path)?; // atomic replace on a POSIX filesystem

    // Make the rename itself durable by fsync-ing the directory entry. Opening a
    // directory for fsync is not portable everywhere, so this is best-effort.
    let dir = match path.parent() {
        Some(p) if !p.as_os_str().is_empty() => p,
        _ => Path::new("."),
    };
    if let Ok(d) = File::open(dir) {
        let _ = d.sync_all();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_is_stable_and_distinct() {
        assert_eq!(sha256_hex("hello"), sha256_hex("hello"));
        assert_ne!(sha256_hex("hello"), sha256_hex("world"));
        // Known vector for "abc".
        assert_eq!(
            sha256_hex("abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn write_durable_writes_and_replaces_atomically() {
        let path = std::env::temp_dir().join(format!("ccos-durable-{}.bin", std::process::id()));
        let _ = std::fs::remove_file(&path);
        write_durable(&path, b"first").unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"first");
        // Overwriting replaces the whole file (no leftover temp sibling).
        write_durable(&path, b"second").unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"second");
        let mut tmp = path.clone().into_os_string();
        tmp.push(".tmp");
        assert!(
            !std::path::Path::new(&tmp).exists(),
            "temp sibling is renamed away, not left behind"
        );
        let _ = std::fs::remove_file(&path);
    }
}
