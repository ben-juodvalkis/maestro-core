//! FIFO Buffer Manager — queue-based frame management.
//!
//! Ported from: `audio-engine/worklets/simple-playback-worklet-final.ts` (FIFOBufferManager class)
//! Architecture: ADR071
//!
//! Solves the double-offset problem by:
//! 1. Storing frames in a simple queue (FIFO)
//! 2. No frame index calculations — just sequential consumption
//! 3. Trusts that the frame reader has already handled offset
//!
//! ## Thread safety
//!
//! On native targets the audio render thread calls [`FifoBuffer::get_next_frame`]
//! while the control thread calls [`FifoBuffer::add_frames`]. The current
//! implementation uses a `VecDeque` behind a lock-free approach (single producer,
//! single consumer). For the initial port we use a simple `VecDeque` since the
//! engine is single-threaded on WASM and will use a proper SPSC queue on native
//! in a future iteration.

use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// AudioFrame
// ---------------------------------------------------------------------------

/// A single audio frame containing interleaved channel data.
///
/// Each element in `channels` is a `Vec<f32>` of length `frame_size` (typically 4096).
/// Index 0 = left, index 1 = right, etc.
#[derive(Debug, Clone)]
pub struct AudioFrame {
    pub channels: Vec<Vec<f32>>,
}

impl AudioFrame {
    /// Create a new audio frame with the given channel data.
    pub fn new(channels: Vec<Vec<f32>>) -> Self {
        Self { channels }
    }

    /// Create a stereo frame of silence with the given frame size.
    pub fn silence(frame_size: usize, num_channels: usize) -> Self {
        Self {
            channels: (0..num_channels).map(|_| vec![0.0; frame_size]).collect(),
        }
    }

    /// Number of channels in this frame.
    pub fn num_channels(&self) -> usize {
        self.channels.len()
    }

    /// Number of samples per channel.
    pub fn frame_size(&self) -> usize {
        self.channels.first().map_or(0, |ch| ch.len())
    }

    /// Get a reference to channel data. Returns `None` if index is out of range.
    pub fn channel(&self, index: usize) -> Option<&[f32]> {
        self.channels.get(index).map(|v| v.as_slice())
    }
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

/// Metrics tracked by the FIFO buffer.
#[derive(Debug, Clone, Default)]
pub struct FifoMetrics {
    pub frames_added: u64,
    pub frames_consumed: u64,
    pub overflows: u64,
    pub underflows: u64,
    pub max_queue_depth: usize,
}

impl FifoMetrics {
    /// Current fill rate as a fraction of max queue size.
    pub fn fill_rate(&self, current_depth: usize, max_size: usize) -> f64 {
        if max_size == 0 {
            return 0.0;
        }
        current_depth as f64 / max_size as f64
    }
}

// ---------------------------------------------------------------------------
// FifoBuffer
// ---------------------------------------------------------------------------

/// FIFO queue for audio frames.
///
/// Frames are added at the back and consumed from the front.
/// An overflow is signalled (and the add is rejected) if the queue would
/// exceed `max_queue_size`.
pub struct FifoBuffer {
    queue: VecDeque<AudioFrame>,
    /// Maximum number of frames allowed in the queue.
    pub max_queue_size: usize,
    metrics: FifoMetrics,
}

impl FifoBuffer {
    /// Create a new FIFO buffer with the given capacity.
    pub fn new(max_queue_size: usize) -> Self {
        Self {
            queue: VecDeque::with_capacity(max_queue_size),
            max_queue_size,
            metrics: FifoMetrics::default(),
        }
    }

    /// Create with default capacity (100 frames).
    pub fn with_defaults() -> Self {
        Self::new(100)
    }

    /// Add one or more frames to the back of the queue.
    ///
    /// Returns `true` if the frames were added, `false` if the queue would
    /// overflow (no frames are added in the overflow case).
    pub fn add_frames(&mut self, frames: impl IntoIterator<Item = AudioFrame>) -> bool {
        let frames: Vec<AudioFrame> = frames.into_iter().collect();
        let count = frames.len();

        if self.queue.len() + count > self.max_queue_size {
            self.metrics.overflows += 1;
            return false;
        }

        for frame in frames {
            self.queue.push_back(frame);
        }
        self.metrics.frames_added += count as u64;
        if self.queue.len() > self.metrics.max_queue_depth {
            self.metrics.max_queue_depth = self.queue.len();
        }
        true
    }

    /// Add a single frame to the back of the queue.
    pub fn add_frame(&mut self, frame: AudioFrame) -> bool {
        self.add_frames(std::iter::once(frame))
    }

    /// Consume the next frame from the front of the queue.
    ///
    /// Returns `None` (and records an underflow) if the queue is empty.
    pub fn get_next_frame(&mut self) -> Option<AudioFrame> {
        match self.queue.pop_front() {
            Some(frame) => {
                self.metrics.frames_consumed += 1;
                Some(frame)
            }
            None => {
                self.metrics.underflows += 1;
                None
            }
        }
    }

    /// Whether any frames are available.
    #[inline]
    pub fn has_frames(&self) -> bool {
        !self.queue.is_empty()
    }

    /// Current number of frames in the queue.
    #[inline]
    pub fn queue_depth(&self) -> usize {
        self.queue.len()
    }

    /// Clear the queue, discarding all frames.
    pub fn clear(&mut self) {
        self.queue.clear();
    }

    /// Get a copy of the current metrics.
    pub fn metrics(&self) -> FifoMetrics {
        self.metrics.clone()
    }

    /// Fill rate as a fraction 0.0–1.0.
    pub fn fill_rate(&self) -> f64 {
        self.metrics
            .fill_rate(self.queue.len(), self.max_queue_size)
    }
}

impl std::fmt::Debug for FifoBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FifoBuffer")
            .field("depth", &self.queue.len())
            .field("max", &self.max_queue_size)
            .field("metrics", &self.metrics)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timing::FRAME_SIZE;

    fn make_stereo_frame(value: f32) -> AudioFrame {
        AudioFrame::new(vec![
            vec![value; FRAME_SIZE as usize],
            vec![value; FRAME_SIZE as usize],
        ])
    }

    // -- construction ------------------------------------------------------

    #[test]
    fn new_buffer_is_empty() {
        let buf = FifoBuffer::with_defaults();
        assert!(!buf.has_frames());
        assert_eq!(buf.queue_depth(), 0);
        assert_eq!(buf.max_queue_size, 100);
    }

    // -- add / get ---------------------------------------------------------

    #[test]
    fn add_and_consume_single_frame() {
        let mut buf = FifoBuffer::with_defaults();
        let frame = make_stereo_frame(0.5);
        assert!(buf.add_frame(frame));
        assert_eq!(buf.queue_depth(), 1);
        assert!(buf.has_frames());

        let out = buf.get_next_frame().unwrap();
        assert_eq!(out.num_channels(), 2);
        assert_eq!(out.frame_size(), FRAME_SIZE as usize);
        assert_eq!(out.channels[0][0], 0.5);
        assert!(!buf.has_frames());
    }

    #[test]
    fn fifo_order_preserved() {
        let mut buf = FifoBuffer::with_defaults();
        buf.add_frame(make_stereo_frame(1.0));
        buf.add_frame(make_stereo_frame(2.0));
        buf.add_frame(make_stereo_frame(3.0));

        assert_eq!(buf.get_next_frame().unwrap().channels[0][0], 1.0);
        assert_eq!(buf.get_next_frame().unwrap().channels[0][0], 2.0);
        assert_eq!(buf.get_next_frame().unwrap().channels[0][0], 3.0);
        assert!(buf.get_next_frame().is_none());
    }

    #[test]
    fn add_multiple_frames() {
        let mut buf = FifoBuffer::with_defaults();
        let frames = vec![make_stereo_frame(0.1), make_stereo_frame(0.2)];
        assert!(buf.add_frames(frames));
        assert_eq!(buf.queue_depth(), 2);
    }

    // -- overflow / underflow -----------------------------------------------

    #[test]
    fn overflow_rejected() {
        let mut buf = FifoBuffer::new(3);
        assert!(buf.add_frame(make_stereo_frame(1.0)));
        assert!(buf.add_frame(make_stereo_frame(2.0)));
        assert!(buf.add_frame(make_stereo_frame(3.0)));
        // Queue is full — next add should fail
        assert!(!buf.add_frame(make_stereo_frame(4.0)));
        assert_eq!(buf.metrics().overflows, 1);
        assert_eq!(buf.queue_depth(), 3);
    }

    #[test]
    fn underflow_tracked() {
        let mut buf = FifoBuffer::with_defaults();
        assert!(buf.get_next_frame().is_none());
        assert_eq!(buf.metrics().underflows, 1);
        assert!(buf.get_next_frame().is_none());
        assert_eq!(buf.metrics().underflows, 2);
    }

    // -- clear --------------------------------------------------------------

    #[test]
    fn clear_empties_queue() {
        let mut buf = FifoBuffer::with_defaults();
        buf.add_frame(make_stereo_frame(1.0));
        buf.add_frame(make_stereo_frame(2.0));
        buf.clear();
        assert!(!buf.has_frames());
        assert_eq!(buf.queue_depth(), 0);
    }

    // -- metrics ------------------------------------------------------------

    #[test]
    fn metrics_tracking() {
        let mut buf = FifoBuffer::new(10);
        buf.add_frame(make_stereo_frame(1.0));
        buf.add_frame(make_stereo_frame(2.0));
        buf.get_next_frame();

        let m = buf.metrics();
        assert_eq!(m.frames_added, 2);
        assert_eq!(m.frames_consumed, 1);
        assert_eq!(m.max_queue_depth, 2);
        assert_eq!(m.overflows, 0);
        assert_eq!(m.underflows, 0);
    }

    #[test]
    fn fill_rate_calculation() {
        let mut buf = FifoBuffer::new(10);
        assert_eq!(buf.fill_rate(), 0.0);
        buf.add_frame(make_stereo_frame(1.0));
        assert!((buf.fill_rate() - 0.1).abs() < 1e-10);
    }

    // -- AudioFrame ---------------------------------------------------------

    #[test]
    fn audio_frame_silence() {
        let frame = AudioFrame::silence(128, 2);
        assert_eq!(frame.num_channels(), 2);
        assert_eq!(frame.frame_size(), 128);
        assert!(frame.channel(0).unwrap().iter().all(|&s| s == 0.0));
    }

    #[test]
    fn audio_frame_channel_access() {
        let frame = AudioFrame::new(vec![vec![1.0; 4], vec![2.0; 4]]);
        assert_eq!(frame.channel(0).unwrap()[0], 1.0);
        assert_eq!(frame.channel(1).unwrap()[0], 2.0);
        assert!(frame.channel(2).is_none());
    }
}
