use neural_store::{NeuralStore, Vector};
use std::fs;
use std::path::Path;
use std::time::Duration;
use std::thread;

/// Helper to create a normalized vector of given dimension
fn create_normalized_vector(dim: usize, value: f32) -> Vector {
    let mut vec = vec![0.0; dim];
    vec[0] = value; // Put all magnitude in the first element for simplicity

    // Normalize it
    let norm = (vec.iter().map(|x| x * x).sum::<f32>()).sqrt();
    if norm > 0.0 {
        for x in vec.iter_mut() {
            *x /= norm;
        }
    }
    Vector(vec)
}

#[test]
fn test_e2e_store_integration() {
    let test_path = Path::new("tests/e2e_temp_store");
    if test_path.exists() {
        fs::remove_dir_all(test_path).unwrap();
    }

    // 1. Initialize the store via high-level API
    {
        let mut store = NeuralStore::open(test_path).expect("Failed to open store");
        println!("Store initialized.");

        // 2. Insert a large dataset of vectors
        let dim = 128;
        let num_vectors = 100;
        for i in 0..num_vectors {
            // Create distinct normalized vectors
            // Vector for i=0 will be [1, 0, ...], i=1 will be [0.9, 0.1, ...], etc.
            let mut data = vec![0.0; dim];
            data[i % dim] = 1.0;
            // Normalize (though it's already unit length if only one element is 1)
            let v = Vector(data);
            store.put(i, v).expect("Failed to insert vector");
        }
        println!("Inserted {} vectors.", num_vectors);

        // 3. Perform searches and verify correctness
        // Query for the first vector [1, 0, ...]
        let mut query = vec![0.0; dim];
        query[0] = 1.0;

        let results = store.search(&query, 5);
        assert!(!results.is_empty(), "Search results should not be empty");

        // The first result should be the vector with ID 0 (since it's identical to query)
        let (best_id, best_score) = results[0];
        assert_eq!(best_id, 0, "Best match should be ID 0, got {}", best_id);
        assert!(best_score > 0.9, "Similarity score should be high, got {}", best_score);
        println!("Search verification successful.");

        // 4. Trigger a manual GC cycle or wait for background clustering to occur
        // Since there is currently no manual trigger and the workers are minimal’
        // we simulate a period of activity and verify stability.
        thread::sleep(Duration::from_millis(500));
        assert_eq!(store.len(), num_vectors, "Store size should remain constant after background activity");
        println!("System stability verified.");
    }

    // 5. Verify persistence by restarting the store and ensuring data is recovered from WAL
    {
        let store = NeuralStore::open(test_path).expect("Failed to re-open store");
        println!("Store re-opened for persistence check.");

        assert_eq!(store.len(), 100, "Recovered store should have 100 elements, got {}", store.len());

        // Check a specific recovered vector
        let v0 = store.get(&0).expect("Vector 0 should be recovered");
        assert_eq!(v0.0[0], 1.0, "Recovered Vector 0 should have value 1.0 at index 0");
        println!("Persistence verification successful.");
    }

    // Cleanup
    if test_path.exists() {
        fs::remove_dir_all(test_path).unwrap();
    }
}
