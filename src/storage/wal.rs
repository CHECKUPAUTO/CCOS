use std::fs::{File, OpenOptions};
use std::io::{Seek, SeekFrom};
use std::path::Path;
use memmap2::MmapMut;
use crc32fast::Hasher;
use anyhow::{Result, anyhow};

pub struct Wal {
    file: File,
    mmap: MmapMut,
    write_pos: usize,
}

impl Wal {
    const INITIAL_SIZE: usize = 1024 * 1024 * 64; // 64MB initial size

    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(path)?;

        let len = file.metadata()?.len();
        if len == 0 {
            file.set_len(Self::INITIAL_SIZE as u64)?;
        }

        let mmap = unsafe { MmapMut::map_mut(&file)? };

        // Find the first empty or corrupted entry to determine current write position
        let write_pos = Self::find_write_pos(&mmap)?;

        Ok(Self {
            file,
            mmap,
            write_pos,
        })
    }

    fn find_write_pos(mmap: &[u8]) -> Result<usize> {
        let mut pos = 0;
        while pos + 8 <= mmap.len() {
            let len = u32::from_le_bytes(mmap[pos..pos+4].try_into().unwrap()) as usize;
            if len == 0 {
                break; // End of valid entries (empty space)
            }
            if pos + 8 + len > mmap.len() {
                break; // Partial entry at the end
            }

            let checksum = u32::from_le_bytes(mmap[pos+4..pos+8].try_into().unwrap());
            let payload = &mmap[pos+8..pos+8+len];

            if !Self::verify_checksum(payload, checksum) {
                // Found corruption. In a real NVMe WAL we might want to truncate here,
                // but for now we just stop at the first corrupted entry.
                break;
            }

            pos += 8 + len;
        }
        Ok(pos)
    }

    fn verify_checksum(data: &[u8], expected: u32) -> bool {
        let mut hasher = Hasher::new();
        hasher.update(data);
        hasher.finalize() == expected
    }

    pub fn append(&mut self, payload: &[u8]) -> Result<()> {
        let len = payload.len();
        if len > u32::MAX as usize {
            return Err(anyhow!("Payload too large for WAL entry"));
        }

        let checksum = {
            let mut hasher = Hasher::new();
            hasher.update(payload);
            hasher.finalize()
        };

        let total_size = 4 + 4 + len;
        if self.write_pos + total_size > self.mmap.len() {
            self.grow(total_size)?;
        }

        // Write [Length (u32)][Checksum (u32)][Payload]
        let len_bytes = (len as u32).to_le_bytes();
        let checksum_bytes = checksum.to_le_bytes();

        self.mmap[self.write_pos..self.write_pos+4].copy_from_slice(&len_bytes);
        self.mmap[self.write_pos+4..self.write_pos+8].copy_from_slice(&checksum_bytes);
        self.mmap[self.write_pos+8..self.write_pos+total_size].copy_from_slice(payload);

        self.write_pos += total_size;
        Ok(())
    }

    pub fn recover<F>(&self, mut handler: F) -> Result<()>
    where
        F: FnMut(&[u8]) -> Result<()>
    {
        let mut pos = 0;
        while pos + 8 <= self.mmap.len() {
            let len = u32::from_le_bytes(self.mmap[pos..pos+4].try_into().unwrap()) as usize;
            if len == 0 {
                break; // End of log
            }
            if pos + 8 + len > self.mmap.len() {
                break; // Partial entry
            }

            let checksum = u32::from_le_bytes(self.mmap[pos+4..pos+8].try_into().unwrap());
            let payload = &self.mmap[pos+8..pos+8+len];

            if !Self::verify_checksum(payload, checksum) {
                return Err(anyhow!("WAL corruption detected at position {}", pos));
            }

            handler(payload)?;
            pos += 8 + len;
        }
        Ok(())
    }

    pub fn flush(&self) -> Result<()> {
        self.mmap.flush()?;
        Ok(())
    }

    fn grow(&mut self, needed: usize) -> Result<()> {
        let current_size = self.mmap.len();
        let new_size = (current_size + needed).next_power_of_two();

        // We must ensure the file is actually extended on disk before re-mapping
        self.file.set_len(new_size as u64)?;
        self.mmap = unsafe { MmapMut::map_mut(&self.file)? };
        Ok(())
    }
}
