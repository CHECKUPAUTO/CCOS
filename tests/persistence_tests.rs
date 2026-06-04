use neural_store::{LsmTree, Vector};
use std::sync::{Arc, Mutex};
use std::thread;
use std::fs;
use std::path::PathBuf;
use anyhow::Result;

fn get_temp_dir(test_name: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push("neural_store_tests");
    path.push(test_name);
    if path.exists() {
        fs::remove_dir_all(&path).unwrap();
    }
    fs::create_dir_all(&path).unwrap();
    path
}

#[test]
fn test_atomicity_recovery() -> Result<()> {
    let dir = get_temp_dir("atomicity");

    // Scope for first instance
    {
        let mut tree = LsmTree::<String>::open(&dir)?;
        let vec1 = Vector::new(vec![1.0, 2.0, 3.0]);
        let vec2 = Vector::new(vec![4.0, 5.0, 6.0]);

        tree.put("key1".to_string(), vec1.clone())?;
        tree.put("key2".to_string(), vec2.clone())?;

        assert_eq!(tree.len(), 2);
    } // tree is dropped here, simulating a crash

    // Re-open and verify recovery from WAL
    {
        let tree = LsmTree::<String>::open(&dir)?;
        assert_eq!(tree.len(), 2);

        let v1 = tree.get(&"key1".to_string()).expect("Key1 should exist");
        let v2 = tree.get(&"key2".to_string()).expect("Key2 should exist");

        assert_eq!(v1.0, vec![1.0, 2.0, 3.0]);
        assert_eq!(v2.0, vec![4.0, 5.0, 6.0]);
    }

    fs::remove_dir_all(dir)?;
    Ok(())
}

#[test]
fn test_concurrency_load() -> Result<()> {
    let dir = get_temp_dir("concurrency");
    let tree = Arc::new(Mutex::new(LsmTree::<u64>::open(&dir)?));

    let num_writers = 8;
    let num_readers = 8;
    let ops_per_thread = 100;

    let mut handles = vec![];

    // Writers: put unique keys
    for w in 0..num_writers {
        let tree_clone = Arc::clone(&tree);
        handles.push(thread::spawn(move || {
            for i in 0..ops_per_thread {
                let key = (w as u64 * ops_per_thread) + i as u64;
                let val = Vector::new(vec![key as f32]);
                let mut lock = tree_clone.lock().unwrap();
                lock.put(key, val).expect("Put should succeed");
            }
        }));
    }

    // Readers: get keys
    for r in 0..num_readers {
        let tree_clone = Arc::clone(&tree);
        handles.push(thread::spawn(move || {
            for i in 0..ops_per_thread {
                let key = (r as u64 * ops_per_thread) + i as u64;
                let lock = tree_clone.lock().unwrap();
                // We don't assert existence here because readers might run before writers
                let _ = lock.get(&key);
            }
        }));
    }

    for handle in handles {
        handle.join().expect("Thread should finish without panicking");
    }

    {
        let lock = tree.lock().unwrap();
        assert_eq!(lock.len(), (num_writers * ops_per_thread) as usize);
    }

    fs::remove_dir_all(dir)?;
    Ok(())
}

#[test]
fn test_zero_copy_integrity() -> Result<()> {
    let dir = get_temp_dir("integrity");

    // Create a large vector to check integrity and potential "zero-copy" behavior (via Arc)
    let data = (0..1024).map(|i| i as f32).collect::<Vec<f32>>();
    let vec_orig = Vector::new(data.clone());
    let key = "large_vec".to_string();

    {
        let mut tree = LsmTree::<String>::open(&dir)?;
        tree.put(key.clone(), vec_orig.clone())?;
    }

    {
        let tree = LsmTree::<String>::open(&dir)?;
        let v1 = tree.get(&key).expect("Should retrieve vector");

        // Integrity check: content is identical
        assert_eq!(v1.0, data);

        // Zero-copy / Arc behavior: repeated gets return clones of the same Arc
        let v2 = tree.get(&key).expect("Should retrieve vector again");
        assert!(Arc::ptr_eq(&v1, &v2), "Retrieve should be zero-copy (returning existing Arc)");
    }

    fs::remove_dir_all(dir)?;
    Ok(())
}
