//! Background teardown for large engine state.
//!
//! A phase can leave hundreds of thousands of arena allocations and analysis
//! records behind. Freeing them is pure bookkeeping that the caller never
//! observes, so the last owner hands the value to a detached thread instead of
//! paying for the drop on the result path.

/// Drop `value` on a background thread.
///
/// Falls back to dropping inline if the thread cannot be spawned: the closure
/// owning `value` is dropped synchronously by the failed spawn.
pub(crate) fn defer_drop<T: Send + 'static>(value: T) {
    let _ = std::thread::Builder::new()
        .name("numsim-deferred-drop".to_string())
        .spawn(move || drop(value));
}
