use std::sync::{OnceLock, Mutex};
use crate::NeuralStore;
use crate::core::types::Vector;
use std::slice;

static STORE: OnceLock<Mutex<NeuralStore>> = OnceLock::new();

#[repr(C)]
pub struct SearchResult {
    pub id: usize,
    pub score: f32,
}

/// Initialize the store with a default capacity.
#[no_mangle]
pub extern "C" fn ns_init() -> i32 {
    match NeuralStore::open("data") {
        Ok(store) => {
            if STORE.set(Mutex::new(store)).is_ok() {
                0
            } else {
                -1
            }
        }
        Err(_) => -1,
    }
}

/// Store a vector in the store.
/// id: The ID to associate with the vector.
/// vector: Pointer to the f32 array.
/// len: Number of elements in the array.
#[no_mangle]
pub extern "C" fn ns_put(id: usize, vector: *const f32, len: usize) -> i32 {
    if vector.is_null() {
        return -1;
    }

    let mutex = match STORE.get() {
        Some(m) => m,
        None => return -2,
    };
    let mut store = mutex.lock().unwrap();

    let data = unsafe { slice::from_raw_parts(vector, len) };
    let vector_obj = Vector::new(data.to_vec());

    match store.put(id, vector_obj) {
        Ok(_) => 0,
        Err(_) => -3,
    }
}

/// Search for the top K results.
/// query: Pointer to the query vector.
/// len: Length of the query vector.
/// k: Number of results to return.
/// out_count: Pointer to store the number of results actually returned.
#[no_mangle]
pub extern "C" fn ns_search(query: *const f32, len: usize, k: usize, out_count: *mut usize) -> *mut SearchResult {
    if query.is_null() || out_count.is_null() {
        return std::ptr::null_mut();
    }

    let mutex = match STORE.get() {
        Some(m) => m,
        None => return std::ptr::null_mut(),
    };
    let store = mutex.lock().unwrap();

    let query_slice = unsafe { slice::from_raw_parts(query, len) };
    let results = store.search(query_slice, k);

    unsafe { *out_count = results.len() };

    if results.is_empty() {
        return std::ptr::null_mut();
    }

    let mut c_results = Vec::with_capacity(results.len());
    for (id, score) in results {
        c_results.push(SearchResult { id, score });
    }

    let ptr = c_results.into_boxed_slice();
    Box::into_raw(ptr) as *mut SearchResult
}

/// Free memory allocated by the store (e.g., search results).
#[no_mangle]
pub extern "C" fn ns_free(ptr: *mut SearchResult, len: usize) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        // Reconstruct the boxed slice and let it drop.
        let _ = Box::from_raw(slice::from_raw_parts_mut(ptr, len));
    }
}
