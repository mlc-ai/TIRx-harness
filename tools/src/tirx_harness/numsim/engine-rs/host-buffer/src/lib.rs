#![deny(unsafe_op_in_unsafe_fn)]

//! Audited Python-buffer boundary for NumSim host-backed global allocations.

use pyo3::buffer::PyBuffer;
use pyo3::{Py, PyAny};
use std::ops::{Deref, DerefMut};
use std::slice;
use std::sync::Arc;

struct HostByteBufferInner {
    buffer: PyBuffer<u8>,
    _owner: Py<PyAny>,
}

/// One writable, C-contiguous Python byte buffer retained for a synchronous
/// native NumSim call.
pub struct HostByteBuffer {
    inner: Arc<HostByteBufferInner>,
    byte_len: usize,
}

impl HostByteBuffer {
    pub fn new(buffer: PyBuffer<u8>, owner: Py<PyAny>) -> Result<Self, &'static str> {
        if buffer.readonly() {
            return Err("NumSim host buffer is read-only");
        }
        if !buffer.is_c_contiguous() {
            return Err("NumSim host buffer is not C-contiguous");
        }
        if buffer.item_size() != 1 {
            return Err("NumSim host buffer elements are not bytes");
        }
        let byte_len = buffer.len_bytes();
        Ok(Self {
            inner: Arc::new(HostByteBufferInner {
                buffer,
                _owner: owner,
            }),
            byte_len,
        })
    }

    pub fn byte_len(&self) -> usize {
        self.byte_len
    }

    /// Consume this buffer into unique, non-overlapping regions.
    pub fn into_regions(self, maximum_region_bytes: usize) -> Vec<HostByteRegion> {
        assert!(
            maximum_region_bytes > 0,
            "host-buffer region size must be positive"
        );
        (0..self.byte_len)
            .step_by(maximum_region_bytes)
            .map(|start| HostByteRegion {
                inner: self.inner.clone(),
                start,
                end: (start + maximum_region_bytes).min(self.byte_len),
            })
            .collect()
    }
}

/// One C-contiguous Python byte buffer (read-only or writable) borrowed as
/// the immutable initial bytes of a NumSim allocation: it is only ever read,
/// the allocation's stripes copy on write, so a non-mutating analysis can
/// work on the caller's arrays without snapshotting them.
///
/// Python callers must not mutate the array during the synchronous native
/// call; the buffer protocol pins the exporter and `_owner` the NumPy owner.
pub struct HostByteSource {
    inner: Arc<HostByteBufferInner>,
    byte_len: usize,
}

impl HostByteSource {
    pub fn new(buffer: PyBuffer<u8>, owner: Py<PyAny>) -> Result<Self, &'static str> {
        if !buffer.is_c_contiguous() {
            return Err("NumSim host buffer is not C-contiguous");
        }
        if buffer.item_size() != 1 {
            return Err("NumSim host buffer elements are not bytes");
        }
        let byte_len = buffer.len_bytes();
        Ok(Self {
            inner: Arc::new(HostByteBufferInner {
                buffer,
                _owner: owner,
            }),
            byte_len,
        })
    }

    pub fn byte_len(&self) -> usize {
        self.byte_len
    }
}

impl AsRef<[u8]> for HostByteSource {
    fn as_ref(&self) -> &[u8] {
        if self.byte_len == 0 {
            return &[];
        }
        // SAFETY: PyBuffer pins the exporter and `_owner` the NumPy owner for
        // the lifetime of `inner`; the buffer is C-contiguous bytes of
        // `byte_len`, and no mutable slice is ever produced from a source.
        unsafe { slice::from_raw_parts(self.inner.buffer.buf_ptr().cast::<u8>(), self.byte_len) }
    }
}

/// One uniquely owned byte interval from a [`HostByteBuffer`].
///
/// NumSim stores each region behind exactly one stripe lock. Python callers
/// must not access an ndarray concurrently with its synchronous native run;
/// the buffer protocol pins the allocation but cannot serialize other Python
/// references to it.
pub struct HostByteRegion {
    inner: Arc<HostByteBufferInner>,
    start: usize,
    end: usize,
}

impl Deref for HostByteRegion {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        let byte_len = self.end - self.start;
        if byte_len == 0 {
            return &[];
        }
        // SAFETY: PyBuffer pins the exporter; `_owner` pins the actual NumPy
        // owner used to create the raw byte view. `into_regions` constructs
        // disjoint intervals, and no mutable slice can be produced through a
        // shared `HostByteRegion` reference.
        unsafe {
            slice::from_raw_parts(
                self.inner.buffer.buf_ptr().cast::<u8>().add(self.start),
                byte_len,
            )
        }
    }
}

impl DerefMut for HostByteRegion {
    fn deref_mut(&mut self) -> &mut [u8] {
        let byte_len = self.end - self.start;
        if byte_len == 0 {
            return &mut [];
        }
        // SAFETY: regions never overlap and mutable access requires unique
        // access to this region. The containing NumSim stripe write lock
        // provides that uniqueness across worker threads.
        unsafe {
            slice::from_raw_parts_mut(
                self.inner.buffer.buf_ptr().cast::<u8>().add(self.start),
                byte_len,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::{PyByteArray, PyByteArrayMethods};
    use pyo3::Python;

    #[test]
    fn source_borrows_a_read_only_buffer_without_copying() {
        Python::attach(|py| {
            use pyo3::types::PyBytesMethods;
            let bytes = pyo3::types::PyBytes::new(py, b"numsim");
            let buffer = PyBuffer::<u8>::get(bytes.as_any()).unwrap();
            assert!(buffer.readonly());
            let source = HostByteSource::new(buffer, bytes.clone().unbind().into_any()).unwrap();
            assert_eq!(source.byte_len(), 6);
            assert_eq!(source.as_ref(), b"numsim");
            assert_eq!(source.as_ref().as_ptr(), bytes.as_bytes().as_ptr());
            let array = PyByteArray::new(py, b"rw");
            let buffer = PyBuffer::<u8>::get(array.as_any()).unwrap();
            let source = HostByteSource::new(buffer, array.clone().unbind().into_any()).unwrap();
            assert_eq!(source.as_ref(), b"rw");
        });
    }

    #[test]
    fn partitions_and_writes_one_python_buffer() {
        Python::attach(|py| {
            let value = PyByteArray::new(py, &[1, 2, 3, 4, 5]);
            let buffer = PyBuffer::<u8>::get(value.as_any()).unwrap();
            let owner = value.clone().into_any().unbind();
            let host = HostByteBuffer::new(buffer, owner).unwrap();
            let mut regions = host.into_regions(3);
            assert_eq!(&*regions[0], &[1, 2, 3]);
            assert_eq!(&*regions[1], &[4, 5]);
            regions[1][0] = 9;
            assert_eq!(value.to_vec(), vec![1, 2, 3, 9, 5]);
        });
    }
}
