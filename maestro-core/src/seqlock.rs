//! # seqlock — lock-free single-writer snapshot cell (spec §3.5)
//!
//! The audio thread writes the engine snapshot once per block; the control
//! plane reads it on demand. A seqlock gives the reader a consistent copy
//! without the writer ever blocking, allocating, or taking a lock — the
//! writer's cost is two atomic stores and a `memcpy` of a POD struct.
//!
//! Single writer only (the audio thread). Readers retry while a write is in
//! progress (odd sequence) or if the sequence moved underneath them.

use std::cell::UnsafeCell;
use std::sync::atomic::{fence, AtomicU32, Ordering};

pub struct SeqLock<T: Copy> {
    seq: AtomicU32,
    data: UnsafeCell<T>,
}

// Safety: access to `data` is mediated by the sequence protocol below; T is
// Copy/POD so a torn read is detected (sequence mismatch) and discarded.
unsafe impl<T: Copy + Send> Sync for SeqLock<T> {}
unsafe impl<T: Copy + Send> Send for SeqLock<T> {}

impl<T: Copy> SeqLock<T> {
    pub fn new(initial: T) -> Self {
        Self { seq: AtomicU32::new(0), data: UnsafeCell::new(initial) }
    }

    /// Publish a new value. **Single writer only** (the audio thread).
    /// RT-safe: no locks, no allocation, bounded time.
    pub fn write(&self, value: T) {
        let s = self.seq.load(Ordering::Relaxed);
        self.seq.store(s.wrapping_add(1), Ordering::Relaxed); // odd: write in progress
        fence(Ordering::Release);
        // Safety: protocol guarantees readers discard torn reads; single writer.
        unsafe { std::ptr::write_volatile(self.data.get(), value) };
        self.seq.store(s.wrapping_add(2), Ordering::Release); // even: stable
    }

    /// Read a consistent copy. Retries while the writer is mid-publish; the
    /// writer publishes at most once per audio block, so this converges in
    /// a handful of iterations at worst.
    pub fn read(&self) -> T {
        loop {
            let s1 = self.seq.load(Ordering::Acquire);
            if s1 & 1 == 1 {
                std::hint::spin_loop();
                continue;
            }
            // Safety: torn reads are detected by the sequence check below.
            let value = unsafe { std::ptr::read_volatile(self.data.get()) };
            fence(Ordering::Acquire);
            let s2 = self.seq.load(Ordering::Relaxed);
            if s1 == s2 {
                return value;
            }
            std::hint::spin_loop();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn write_then_read_round_trips() {
        let cell = SeqLock::new((0u64, 0u32));
        cell.write((42, 7));
        assert_eq!(cell.read(), (42, 7));
    }

    #[test]
    fn concurrent_reader_never_sees_torn_pair() {
        // Writer publishes (n, n); a torn read would surface (n, m) with n != m.
        let cell = Arc::new(SeqLock::new((0u64, 0u64)));
        let reader = {
            let cell = Arc::clone(&cell);
            std::thread::spawn(move || {
                for _ in 0..200_000 {
                    let (a, b) = cell.read();
                    assert_eq!(a, b, "torn read: {a} vs {b}");
                }
            })
        };
        for n in 0..200_000u64 {
            cell.write((n, n));
        }
        reader.join().unwrap();
    }
}
