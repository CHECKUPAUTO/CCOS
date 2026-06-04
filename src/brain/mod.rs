pub mod clustering;
pub mod gc;

use std::thread;
use std::sync::Arc;
use anyhow::Result;

pub struct BrainWorkers {
    // Handle to background threads or tasks
    // For now, we'll just keep track of them as JoinHandles if needed,
    // but since they are long-running workers, we might just let them run.
}

impl BrainWorkers {
    /// Starts the background worker threads.
    pub fn start() -> Result<Self> {
        println!("Starting Brain Workers...");

        // Start GC worker
        thread::spawn(|| {
            gc::run_gc_loop();
        });

        // Start Clustering worker
        thread::spawn(|| {
            clustering::run_clustering_loop();
        });

        Ok(Self {})
    }
}
