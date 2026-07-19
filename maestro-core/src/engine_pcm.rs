//! # engine_pcm — engine-resident PCM with a monotone load frontier (ADR-112 §5.3)
//!
//! One representation for bulk-loaded AND streaming track PCM: both channels
//! are allocated at their exact final size up front, samples are published
//! monotonically behind an atomic **frontier**, and the render path reads
//! only below a frontier value it Acquire-loaded once per block.
//!
//! ## The safety argument (seqlock-style, inverted)
//!
//! This module owns the codebase's one `MaybeUninit` region. Its soundness is
//! the same *style* of argument as [`crate::seqlock`], with the publication
//! order inverted (data first, then the counter):
//!
//! - **Single writer.** Exactly one thread appends (the control-plane / FFI
//!   caller thread; on `wasm32-unknown-unknown` it is the same worklet thread
//!   as the reader). Writer exclusivity is structural: appends go through
//!   `ControlPlane`, which is not `Sync` and is serialized by the FFI layer.
//! - **Monotone frontier, publish-after-write.** The writer fills
//!   `[frontier, frontier + n)` and only then Release-stores the new
//!   frontier. It never writes below a published frontier.
//! - **Reader reads only below an Acquire-loaded frontier.** The render
//!   loop loads the frontier once per block; the §5.4 starvation predicate
//!   structurally prevents any read at or above it. The Acquire/Release pair
//!   makes the copied samples visible before the frontier moves (load-bearing
//!   on native, where writer and reader are different threads; trivially
//!   correct on wasm32 where they share the one worklet thread).
//! - **Uninitialized memory is never read.** Release builds skip zero-fill
//!   (zeroing ~580 MiB on the worklet thread would stall the outgoing song's
//!   render during crossfade prep — ADR-112 §5.3); debug builds zero-fill so
//!   any predicate bug renders silence, not noise.
//!
//! `read_pcm` and every other consumer accesses samples exclusively through
//! [`EnginePcm::copy_interleaved`] / [`EnginePcm::fill_interleaved`], keeping
//! the `unsafe` confined to this file.

use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use maestro_api::SampleFormat;

/// One channel of frontier-guarded samples. `UnsafeCell<MaybeUninit<T>>` has
/// the same in-memory representation as `T`; interior mutability is required
/// because the writer appends through the same `Arc<EnginePcm>` the audio
/// thread reads.
struct ChannelBuf<T>(Box<[UnsafeCell<MaybeUninit<T>>]>);

impl<T: Copy> ChannelBuf<T> {
    fn new_uninit(len: usize, zero: T) -> Self {
        let mut v = Vec::with_capacity(len);
        v.resize_with(len, || UnsafeCell::new(MaybeUninit::uninit()));
        let buf = Self(v.into_boxed_slice());
        // Debug builds zero-fill so a predicate bug renders silence, not
        // noise (ADR-112 §5.3). Release relies on the frontier guard.
        if cfg!(debug_assertions) {
            for cell in buf.0.iter() {
                // SAFETY: sole owner during construction; nothing reads yet.
                unsafe { (*cell.get()).write(zero) };
            }
        }
        buf
    }

    fn from_vec(v: Vec<T>) -> Self {
        let boxed: Box<[T]> = v.into_boxed_slice();
        let len = boxed.len();
        let ptr = Box::into_raw(boxed) as *mut UnsafeCell<MaybeUninit<T>>;
        // SAFETY: `UnsafeCell<MaybeUninit<T>>` is layout-identical to `T`
        // (both are repr(transparent) wrappers); length is unchanged; the
        // allocation was produced by Box for the same total size/align.
        Self(unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, len)) })
    }

    /// Write `src` at `at`. Caller (EnginePcm) guarantees `at` is the current
    /// frontier and `at + src.len() <= len` — the region is unpublished.
    fn write_at(&self, at: usize, src: &[T]) {
        debug_assert!(at + src.len() <= self.0.len());
        // SAFETY: single writer (module contract); the target region is at or
        // above the published frontier, so no reader touches it until the
        // subsequent Release store publishes it.
        unsafe {
            std::ptr::copy_nonoverlapping(
                src.as_ptr(),
                self.0.as_ptr().add(at) as *mut T,
                src.len(),
            );
        }
    }

    /// Read `count` samples starting at `idx`. Caller (EnginePcm) guarantees
    /// the region is at or below an Acquire-loaded frontier.
    #[inline]
    fn read(&self, idx: usize, count: usize) -> &[T] {
        debug_assert!(idx + count <= self.0.len());
        // SAFETY: the region [idx, idx+count) is below a published frontier
        // (caller contract) — initialized by the writer before the Release
        // store that published it, made visible by the reader's Acquire load.
        unsafe { std::slice::from_raw_parts(self.0.as_ptr().add(idx) as *const T, count) }
    }
}

/// Per-format channel pair. `I16` is the Tier 2a opt-in (ADR-112 §6.2):
/// samples quantize once at append and convert back in the read copy loop.
enum Channels {
    F32 { left: ChannelBuf<f32>, right: ChannelBuf<f32> },
    I16 { left: ChannelBuf<i16>, right: ChannelBuf<i16> },
}

/// i16 quantization scale: symmetric ±32767 so the round trip is centered
/// (quantization error ≤ 0.5/32767 ≈ −96 dBFS).
const I16_SCALE: f32 = 32767.0;

#[inline]
fn f32_to_i16(s: f32) -> i16 {
    // Clamp symmetric at ±32767 (matching I16_SCALE) so out-of-range input
    // can't produce a stored value that decodes below −1.0.
    (s * I16_SCALE).round().clamp(-I16_SCALE, I16_SCALE) as i16
}

#[inline]
fn i16_to_f32(s: i16) -> f32 {
    s as f32 / I16_SCALE
}

/// Engine-resident PCM with a monotone load frontier (ADR-112 §5.3). Single
/// writer (the control plane / FFI caller thread), single RT reader. Replaces
/// the immutable `Vec<Vec<f32>>` interior of `PcmData` for track playback.
pub struct EnginePcm {
    channels: Channels,
    /// Frames per channel — the exact allocation, fixed at construction.
    len: u64,
    /// Valid frames `[0, frontier)`. Writer: fill `[frontier, frontier+n)`,
    /// then store(Release). Reader: load(Acquire) once per block.
    frontier: AtomicU64,
    /// Set by finalize after verification; `frontier == len` implied.
    complete: AtomicBool,
}

// SAFETY: shared access is mediated by the frontier protocol documented in
// the module header — single writer, monotone Release-published frontier,
// readers only below an Acquire-loaded frontier.
unsafe impl Send for EnginePcm {}
unsafe impl Sync for EnginePcm {}

impl EnginePcm {
    /// Exact-size allocation for a streaming load (frontier 0). No zero-fill
    /// in release (ADR-112 §5.3); debug builds zero-fill.
    pub fn new_streaming(len: u64, format: SampleFormat) -> Self {
        let n = len as usize;
        let channels = match format {
            SampleFormat::F32 => Channels::F32 {
                left: ChannelBuf::new_uninit(n, 0.0),
                right: ChannelBuf::new_uninit(n, 0.0),
            },
            SampleFormat::I16 => Channels::I16 {
                left: ChannelBuf::new_uninit(n, 0),
                right: ChannelBuf::new_uninit(n, 0),
            },
        };
        Self { channels, len, frontier: AtomicU64::new(0), complete: AtomicBool::new(false) }
    }

    /// A complete buffer from fully-decoded channels (the bulk-load shape).
    /// Takes ownership of the Vecs — no copy for `F32`.
    pub fn new_complete(left: Vec<f32>, right: Vec<f32>, format: SampleFormat) -> Self {
        debug_assert_eq!(left.len(), right.len());
        let len = left.len() as u64;
        let channels = match format {
            SampleFormat::F32 => {
                Channels::F32 { left: ChannelBuf::from_vec(left), right: ChannelBuf::from_vec(right) }
            }
            SampleFormat::I16 => Channels::I16 {
                left: ChannelBuf::from_vec(left.iter().copied().map(f32_to_i16).collect()),
                right: ChannelBuf::from_vec(right.iter().copied().map(f32_to_i16).collect()),
            },
        };
        Self { channels, len, frontier: AtomicU64::new(len), complete: AtomicBool::new(true) }
    }

    /// Frames per channel (the exact allocation — NOT the frontier).
    #[inline]
    pub fn len(&self) -> u64 {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Resident bytes for both channels — allocated capacity, not frontier
    /// (§4.11 budget accounting must never under-count a streaming track,
    /// ADR-112 §5.3).
    pub fn bytes(&self) -> u64 {
        let sample = match self.channels {
            Channels::F32 { .. } => 4,
            Channels::I16 { .. } => 2,
        };
        self.len * 2 * sample
    }

    pub fn format(&self) -> SampleFormat {
        match self.channels {
            Channels::F32 { .. } => SampleFormat::F32,
            Channels::I16 { .. } => SampleFormat::I16,
        }
    }

    /// The load frontier: valid frames `[0, frontier)`. Acquire — pairs with
    /// the writer's Release store so the samples below it are visible.
    #[inline]
    pub fn frontier(&self) -> u64 {
        self.frontier.load(Ordering::Acquire)
    }

    /// True once finalize verified the totals (or the buffer was constructed
    /// complete). Acquire: observing `true` also makes every sample visible,
    /// so callers may cache the answer and skip the frontier thereafter.
    #[inline]
    pub fn is_complete(&self) -> bool {
        self.complete.load(Ordering::Acquire)
    }

    /// Append `left`/`right` at frame `at` (must equal the current frontier;
    /// enforced by the control-plane lifecycle) and publish the new frontier.
    /// Writer side only.
    pub fn append(&self, at: u64, left: &[f32], right: &[f32]) {
        // Real asserts, not debug-only: these two bound the unsafe writes
        // below — a mismatched right channel would run past its buffer in
        // release. Writer side, off the render path.
        assert_eq!(left.len(), right.len(), "append channels must be the same length");
        debug_assert_eq!(at, self.frontier.load(Ordering::Relaxed), "append must be at the frontier");
        assert!(at + left.len() as u64 <= self.len, "append past the allocation");
        let at_usize = at as usize;
        match &self.channels {
            Channels::F32 { left: l, right: r } => {
                l.write_at(at_usize, left);
                r.write_at(at_usize, right);
            }
            Channels::I16 { left: l, right: r } => {
                // Quantize once at load (ADR-112 §6.2). Bounded scratch: the
                // transport caps appends at MAX_STREAM_APPEND_FRAMES; this
                // runs on the writer (control-plane) side, never the RT path.
                let ql: Vec<i16> = left.iter().copied().map(f32_to_i16).collect();
                let qr: Vec<i16> = right.iter().copied().map(f32_to_i16).collect();
                l.write_at(at_usize, &ql);
                r.write_at(at_usize, &qr);
            }
        }
        self.frontier.store(at + left.len() as u64, Ordering::Release);
    }

    /// Mark the buffer complete. Writer side; requires `frontier == len`.
    /// A REAL assert (not debug-only): `complete` licenses readers to touch
    /// the whole `[0, len)` range, so publishing it with an unwritten tail
    /// would be UB, not a logic bug. Off the render path (control-plane
    /// thread, once per stream) — the check costs nothing that matters.
    pub fn mark_complete(&self) {
        assert_eq!(
            self.frontier.load(Ordering::Relaxed),
            self.len,
            "mark_complete before the frontier reached the allocation"
        );
        self.complete.store(true, Ordering::Release);
    }

    /// Copy `count` frames starting at PCM index `idx` into `dest`
    /// (interleaved stereo, `dest.len() >= count * 2`). Reader side. The
    /// caller must have established `idx + count <= frontier` for a frontier
    /// it Acquire-loaded (the §5.4 predicate does exactly this; complete
    /// buffers satisfy it for every in-bounds read).
    #[inline]
    pub fn copy_interleaved(&self, idx: usize, count: usize, dest: &mut [f32]) {
        debug_assert!(
            idx as u64 + count as u64 <= self.frontier.load(Ordering::Relaxed)
                || self.complete.load(Ordering::Relaxed),
            "read at/above the load frontier — the §5.4 predicate must prevent this"
        );
        match &self.channels {
            Channels::F32 { left, right } => {
                let l = left.read(idx, count);
                let r = right.read(idx, count);
                for k in 0..count {
                    dest[k * 2] = l[k];
                    dest[k * 2 + 1] = r[k];
                }
            }
            Channels::I16 { left, right } => {
                let l = left.read(idx, count);
                let r = right.read(idx, count);
                for k in 0..count {
                    dest[k * 2] = i16_to_f32(l[k]);
                    dest[k * 2 + 1] = i16_to_f32(r[k]);
                }
            }
        }
    }

    /// Fill `count` interleaved frames of `dest` with the sample at `idx`
    /// (the Bresenham long-wrap clamp arm — duplicates the loop tail).
    /// Same read contract as [`Self::copy_interleaved`].
    #[inline]
    pub fn fill_interleaved(&self, idx: usize, count: usize, dest: &mut [f32]) {
        let (l, r) = match &self.channels {
            Channels::F32 { left, right } => (left.read(idx, 1)[0], right.read(idx, 1)[0]),
            Channels::I16 { left, right } => {
                (i16_to_f32(left.read(idx, 1)[0]), i16_to_f32(right.read(idx, 1)[0]))
            }
        };
        for k in 0..count {
            dest[k * 2] = l;
            dest[k * 2 + 1] = r;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn bulk_construction_is_complete_and_byte_exact() {
        let l: Vec<f32> = (0..1000).map(|i| i as f32 / 1000.0).collect();
        let r: Vec<f32> = (0..1000).map(|i| -(i as f32) / 1000.0).collect();
        let pcm = EnginePcm::new_complete(l.clone(), r.clone(), SampleFormat::F32);
        assert!(pcm.is_complete());
        assert_eq!(pcm.frontier(), 1000);
        assert_eq!(pcm.len(), 1000);
        assert_eq!(pcm.bytes(), 1000 * 2 * 4);
        let mut dest = vec![0.0f32; 2000];
        pcm.copy_interleaved(0, 1000, &mut dest);
        for i in 0..1000 {
            assert_eq!(dest[i * 2], l[i]);
            assert_eq!(dest[i * 2 + 1], r[i]);
        }
    }

    #[test]
    fn streaming_appends_advance_the_frontier_monotonically() {
        let pcm = EnginePcm::new_streaming(10, SampleFormat::F32);
        assert!(!pcm.is_complete());
        assert_eq!(pcm.frontier(), 0);
        pcm.append(0, &[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
        assert_eq!(pcm.frontier(), 3);
        pcm.append(3, &[7.0; 7], &[8.0; 7]);
        assert_eq!(pcm.frontier(), 10);
        pcm.mark_complete();
        assert!(pcm.is_complete());
        let mut dest = vec![0.0f32; 6];
        pcm.copy_interleaved(0, 3, &mut dest);
        assert_eq!(dest, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    #[should_panic(expected = "append past the allocation")]
    fn append_past_the_allocation_panics() {
        let pcm = EnginePcm::new_streaming(4, SampleFormat::F32);
        pcm.append(0, &[0.0; 5], &[0.0; 5]);
    }

    #[test]
    fn fill_interleaved_duplicates_the_tail_sample() {
        let pcm = EnginePcm::new_complete(vec![1.0, 2.0], vec![3.0, 4.0], SampleFormat::F32);
        let mut dest = vec![0.0f32; 6];
        pcm.fill_interleaved(1, 3, &mut dest);
        assert_eq!(dest, vec![2.0, 4.0, 2.0, 4.0, 2.0, 4.0]);
    }

    #[test]
    fn i16_storage_halves_bytes_and_round_trips_within_quantization() {
        let l: Vec<f32> = (0..256).map(|i| ((i as f32) / 128.0 - 1.0) * 0.9).collect();
        let pcm = EnginePcm::new_complete(l.clone(), l.clone(), SampleFormat::I16);
        assert_eq!(pcm.bytes(), 256 * 2 * 2);
        assert_eq!(pcm.format(), SampleFormat::I16);
        let mut dest = vec![0.0f32; 512];
        pcm.copy_interleaved(0, 256, &mut dest);
        for i in 0..256 {
            assert!((dest[i * 2] - l[i]).abs() <= 0.5 / I16_SCALE + f32::EPSILON);
        }
    }

    #[test]
    fn i16_streaming_append_quantizes_at_the_frontier() {
        let pcm = EnginePcm::new_streaming(4, SampleFormat::I16);
        pcm.append(0, &[0.5, -0.5], &[1.0, -1.0]);
        pcm.append(2, &[0.0, 0.25], &[0.0, -0.25]);
        assert_eq!(pcm.frontier(), 4);
        let mut dest = vec![0.0f32; 8];
        pcm.copy_interleaved(0, 4, &mut dest);
        assert!((dest[0] - 0.5).abs() <= 1.0 / I16_SCALE);
        assert!((dest[1] - 1.0).abs() <= 1.0 / I16_SCALE);
        assert!((dest[7] + 0.25).abs() <= 1.0 / I16_SCALE);
    }

    /// The cross-thread publication contract (native shape): a reader that
    /// Acquire-loads the frontier sees every sample below it, for the whole
    /// run. Miri-checked (this test and the module are pure Rust).
    #[test]
    fn concurrent_reader_only_sees_published_samples() {
        const LEN: u64 = 4_096;
        const CHUNK: usize = 64;
        let pcm = Arc::new(EnginePcm::new_streaming(LEN, SampleFormat::F32));
        let reader = {
            let pcm = Arc::clone(&pcm);
            std::thread::spawn(move || {
                let mut dest = vec![0.0f32; 2 * CHUNK];
                let mut seen = 0u64;
                while seen < LEN {
                    let f = pcm.frontier();
                    assert!(f >= seen, "frontier moved backwards");
                    if f > seen {
                        // Read the newly published span: every sample must be
                        // the value the writer stored (index-derived).
                        let mut at = seen;
                        while at < f {
                            let n = ((f - at) as usize).min(CHUNK);
                            pcm.copy_interleaved(at as usize, n, &mut dest);
                            for k in 0..n {
                                let expect = (at + k as u64) as f32;
                                assert_eq!(dest[k * 2], expect);
                                assert_eq!(dest[k * 2 + 1], -expect);
                            }
                            at += n as u64;
                        }
                        seen = f;
                    } else {
                        std::hint::spin_loop();
                    }
                }
            })
        };
        let mut at = 0u64;
        while at < LEN {
            let n = ((LEN - at) as usize).min(CHUNK);
            let l: Vec<f32> = (0..n).map(|k| (at + k as u64) as f32).collect();
            let r: Vec<f32> = l.iter().map(|v| -v).collect();
            pcm.append(at, &l, &r);
            at += n as u64;
        }
        pcm.mark_complete();
        reader.join().unwrap();
    }
}
