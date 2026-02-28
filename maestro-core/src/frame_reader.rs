//! Frame reader — reads audio buffers and delivers frames to the FIFO queue.
//!
//! Ported from:
//! - `src/domains/core/audio/frame-reader/frame-reader-core.ts` (FrameReaderCore)
//! - `src/domains/core/audio/frame-reader/buffer-manager.ts` (BufferManager)
//! - `src/domains/core/audio/frame-reader/state-manager.ts` (StateManager)
//! - `src/domains/core/audio/frame-reader/types.ts`
//! - `src/domains/core/audio/frame-reader/utils.ts`
//!
//! Architecture: ADR073 (Dynamic Frame Delivery), ADR074 (Adaptive Frame Buffer)
//!
//! ## Design simplification
//!
//! The TypeScript version has five separate classes because the worklet runs in a
//! different thread and communicates via `MessagePort`. In Rust, the frame reader
//! and FIFO buffer live in the same process, so we merge BufferManager + StateManager
//! into a single `FrameReader` struct and call it directly.

use crate::fifo_buffer::{AudioFrame, FifoBuffer};
use crate::timing::{DEFAULT_SAMPLE_RATE, FRAME_SIZE};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the adaptive frame delivery system (ADR074).
#[derive(Debug, Clone)]
pub struct AdaptiveFrameConfig {
    /// Queue depth below which boost logic activates (default: 10).
    pub low_queue_threshold: usize,
    /// Queue depth above which backoff logic activates (default: 20).
    pub high_queue_threshold: usize,
    /// Maximum frames per delivery interval during catch-up (default: 3).
    pub max_catch_up_frames: usize,
    /// Enable probabilistic reduction during high-queue states (default: true).
    pub enable_probabilistic_reduction: bool,
}

impl Default for AdaptiveFrameConfig {
    fn default() -> Self {
        Self {
            low_queue_threshold: 10,
            high_queue_threshold: 20,
            max_catch_up_frames: 3,
            enable_probabilistic_reduction: true,
        }
    }
}

/// Configuration for the frame reader.
#[derive(Debug, Clone)]
pub struct FrameReaderConfig {
    /// Samples per frame (default: 4096).
    pub frame_size: u32,
    /// Sample rate in Hz (default: 48000).
    pub sample_rate: u32,
    /// Number of channels (default: 2 = stereo).
    pub channel_count: u32,
    /// Adaptive frame delivery settings.
    pub adaptive: AdaptiveFrameConfig,
}

impl Default for FrameReaderConfig {
    fn default() -> Self {
        Self {
            frame_size: FRAME_SIZE,
            sample_rate: DEFAULT_SAMPLE_RATE,
            channel_count: 2,
            adaptive: AdaptiveFrameConfig::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

/// Current state of the frame reader.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameReaderState {
    Idle,
    Reading,
    Paused,
    Switching,
    Error,
}

/// Statistics tracked by the frame reader.
#[derive(Debug, Clone, Default)]
pub struct FrameReaderStats {
    pub total_frames_read: u64,
    pub current_position: u64,
    pub buffer_switches: u64,
    pub last_switch_position: u64,
}

// ---------------------------------------------------------------------------
// AudioBuffer (platform-agnostic)
// ---------------------------------------------------------------------------

/// Platform-agnostic audio buffer.
///
/// In the browser this wraps a Web Audio `AudioBuffer`. In native Rust it
/// owns `Vec<Vec<f32>>` channel data decoded by `symphonia` (Step 5).
#[derive(Debug, Clone)]
pub struct AudioBuffer {
    /// Per-channel sample data. Index 0 = left, 1 = right.
    pub channels: Vec<Vec<f32>>,
    /// Sample rate in Hz.
    pub sample_rate: u32,
}

impl AudioBuffer {
    /// Create a new audio buffer.
    pub fn new(channels: Vec<Vec<f32>>, sample_rate: u32) -> Self {
        Self {
            channels,
            sample_rate,
        }
    }

    /// Number of channels.
    pub fn num_channels(&self) -> usize {
        self.channels.len()
    }

    /// Length in samples (per channel).
    pub fn length(&self) -> usize {
        self.channels.first().map_or(0, |ch| ch.len())
    }

    /// Duration in seconds.
    pub fn duration(&self) -> f64 {
        if self.sample_rate == 0 {
            return 0.0;
        }
        self.length() as f64 / self.sample_rate as f64
    }

    /// Get channel data by index.
    pub fn channel_data(&self, index: usize) -> Option<&[f32]> {
        self.channels.get(index).map(|v| v.as_slice())
    }

    /// Validate the buffer for use by the frame reader.
    pub fn validate(&self, expected_channels: u32, expected_sample_rate: u32) -> Result<(), String> {
        if self.num_channels() != expected_channels as usize {
            return Err(format!(
                "Expected {} channels, got {}",
                expected_channels,
                self.num_channels()
            ));
        }
        if self.sample_rate != expected_sample_rate {
            return Err(format!(
                "Expected {}Hz sample rate, got {}Hz",
                expected_sample_rate, self.sample_rate
            ));
        }
        if self.length() == 0 {
            return Err("AudioBuffer is empty".into());
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// FrameReader
// ---------------------------------------------------------------------------

/// Reads audio data from a buffer and delivers frames to a FIFO queue.
///
/// Combines the responsibilities of TypeScript's `FrameReaderCore`,
/// `BufferManager`, and `StateManager` into a single struct.
pub struct FrameReader {
    config: FrameReaderConfig,
    state: FrameReaderState,
    stats: FrameReaderStats,

    // Buffer management
    source_buffer: Option<AudioBuffer>,
    read_position: u64,

    // Switch coordination
    switch_pending: bool,

    // Tempo-aware frame delivery (ADR073)
    playback_rate: f64,

    // Adaptive queue monitoring (ADR074)
    last_known_queue_depth: usize,
}

impl FrameReader {
    /// Create a new frame reader with the given configuration.
    pub fn new(config: FrameReaderConfig) -> Self {
        Self {
            config,
            state: FrameReaderState::Idle,
            stats: FrameReaderStats::default(),
            source_buffer: None,
            read_position: 0,
            switch_pending: false,
            playback_rate: 1.0,
            last_known_queue_depth: 0,
        }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(FrameReaderConfig::default())
    }

    // -----------------------------------------------------------------------
    // Buffer management
    // -----------------------------------------------------------------------

    /// Set the source audio buffer. Validates the buffer before accepting it.
    pub fn set_source_buffer(&mut self, buffer: AudioBuffer) -> Result<(), String> {
        buffer.validate(self.config.channel_count, self.config.sample_rate)?;
        self.source_buffer = Some(buffer);
        self.read_position = 0;
        Ok(())
    }

    /// Switch to a new audio buffer, optionally maintaining the read position.
    ///
    /// If `maintain_position` is true and the new buffer has a different length,
    /// the position is scaled proportionally.
    pub fn switch_source(
        &mut self,
        new_buffer: AudioBuffer,
        maintain_position: bool,
    ) -> Result<(), String> {
        new_buffer.validate(self.config.channel_count, self.config.sample_rate)?;

        if let Some(old) = &self.source_buffer {
            if maintain_position && new_buffer.length() != old.length() {
                let ratio = new_buffer.length() as f64 / old.length() as f64;
                self.read_position = (self.read_position as f64 * ratio) as u64;
            } else if !maintain_position {
                self.read_position = 0;
            }
        } else {
            self.read_position = 0;
        }

        self.source_buffer = Some(new_buffer);
        self.stats.buffer_switches += 1;
        self.stats.last_switch_position = self.read_position;
        self.switch_pending = false;
        Ok(())
    }

    /// Set the read position (only valid when idle or paused).
    pub fn set_initial_position(&mut self, position: u64) -> Result<(), String> {
        if self.state == FrameReaderState::Reading {
            return Err("Cannot set position while reading".into());
        }
        if let Some(buf) = &self.source_buffer {
            if position >= buf.length() as u64 {
                return Err(format!(
                    "Position {} out of bounds [0, {})",
                    position,
                    buf.length()
                ));
            }
        }
        self.read_position = position;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // State / lifecycle
    // -----------------------------------------------------------------------

    /// Current state.
    pub fn state(&self) -> FrameReaderState {
        self.state
    }

    /// Start reading.
    pub fn start(&mut self) {
        self.state = FrameReaderState::Reading;
    }

    /// Stop reading and reset send state.
    pub fn stop(&mut self) {
        self.state = FrameReaderState::Idle;
    }

    /// Pause reading.
    pub fn pause(&mut self) {
        if self.state == FrameReaderState::Reading {
            self.state = FrameReaderState::Paused;
        }
    }

    /// Resume reading.
    pub fn resume(&mut self) {
        if self.state == FrameReaderState::Paused {
            self.state = FrameReaderState::Reading;
        }
    }

    /// Mark that a variation switch is pending (pauses frame delivery).
    pub fn set_switch_pending(&mut self, pending: bool) {
        self.switch_pending = pending;
    }

    /// Whether a switch is pending.
    pub fn is_switch_pending(&self) -> bool {
        self.switch_pending
    }

    /// Statistics snapshot.
    pub fn stats(&self) -> &FrameReaderStats {
        &self.stats
    }

    /// Current read position in samples.
    pub fn read_position(&self) -> u64 {
        self.read_position
    }

    // -----------------------------------------------------------------------
    // Tempo-aware frame delivery (ADR073 / ADR074)
    // -----------------------------------------------------------------------

    /// Update the current playback rate (tempo ratio).
    pub fn set_playback_rate(&mut self, rate: f64) {
        self.playback_rate = rate;
    }

    /// Update the last-known queue depth from the FIFO buffer.
    pub fn update_queue_depth(&mut self, depth: usize) {
        self.last_known_queue_depth = depth;
    }

    /// Calculate how many frames to deliver this interval.
    ///
    /// Implements ADR073 (tempo-aware) + ADR074 (adaptive queue feedback):
    /// - Base frames: `ceil(playback_rate)` capped at `max_catch_up_frames`
    /// - Boost if queue is low, back off if queue is high
    pub fn calculate_frames_per_interval(&self) -> usize {
        // ADR073: Base frame count from playback rate
        let base_frames = if self.playback_rate <= 1.0 {
            1
        } else {
            (self.playback_rate.ceil() as usize).min(self.config.adaptive.max_catch_up_frames)
        };

        // ADR074: Adaptive adjustment based on queue depth
        let depth = self.last_known_queue_depth;
        let cfg = &self.config.adaptive;

        if depth < cfg.low_queue_threshold {
            // Low queue → boost delivery (up to max_catch_up_frames)
            base_frames.max(2).min(cfg.max_catch_up_frames)
        } else if depth > cfg.high_queue_threshold {
            // High queue → back off
            if cfg.enable_probabilistic_reduction {
                // Gradual reduction: halve the frame count (minimum 1)
                (base_frames / 2).max(1)
            } else {
                1
            }
        } else {
            // Normal range → use base calculation
            base_frames
        }
    }

    // -----------------------------------------------------------------------
    // Frame reading
    // -----------------------------------------------------------------------

    /// Read the next frame from the source buffer, advancing the read position.
    ///
    /// Returns `None` if no source buffer is set or if reading is not active.
    /// Handles looping: when the read position reaches the end of the buffer,
    /// it wraps around to the beginning.
    pub fn read_next_frame(&mut self) -> Option<AudioFrame> {
        if self.state != FrameReaderState::Reading || self.switch_pending {
            return None;
        }

        let buffer = self.source_buffer.as_ref()?;
        let frame_size = self.config.frame_size as usize;
        let buf_len = buffer.length();

        if buf_len == 0 {
            return None;
        }

        let mut channels = Vec::with_capacity(self.config.channel_count as usize);

        for ch_idx in 0..self.config.channel_count as usize {
            let ch_data = match buffer.channel_data(ch_idx) {
                Some(d) => d,
                None => {
                    // Missing channel → silence
                    channels.push(vec![0.0; frame_size]);
                    continue;
                }
            };

            let mut frame_data = vec![0.0f32; frame_size];
            let mut pos = self.read_position as usize;

            for sample in frame_data.iter_mut() {
                *sample = ch_data[pos % buf_len];
                pos += 1;
            }

            channels.push(frame_data);
        }

        // Advance position with looping
        self.read_position = ((self.read_position as usize + frame_size) % buf_len) as u64;
        self.stats.total_frames_read += 1;
        self.stats.current_position = self.read_position;

        Some(AudioFrame::new(channels))
    }

    /// Read and deliver frames into the given FIFO buffer.
    ///
    /// Reads `calculate_frames_per_interval()` frames and pushes them
    /// into the FIFO. Returns the number of frames actually delivered.
    pub fn deliver_frames(&mut self, fifo: &mut FifoBuffer) -> usize {
        let count = self.calculate_frames_per_interval();
        let mut delivered = 0;

        for _ in 0..count {
            if let Some(frame) = self.read_next_frame() {
                if fifo.add_frame(frame) > 0 {
                    delivered += 1;
                } else {
                    break; // FIFO full
                }
            } else {
                break; // No more frames to read
            }
        }

        delivered
    }

    /// Clean up all resources.
    pub fn cleanup(&mut self) {
        self.source_buffer = None;
        self.read_position = 0;
        self.state = FrameReaderState::Idle;
        self.switch_pending = false;
    }
}

// ---------------------------------------------------------------------------
// Utility functions (from frame-reader/utils.ts)
// ---------------------------------------------------------------------------

/// Calculate the interval in milliseconds between frame deliveries.
pub fn calculate_interval_ms(frame_size: u32, sample_rate: u32) -> f64 {
    (frame_size as f64 / sample_rate as f64) * 1000.0
}

/// Check if audio data contains any non-zero samples.
pub fn has_audio_data(data: &[f32]) -> bool {
    data.iter().any(|&s| s != 0.0)
}

/// Calculate RMS (root mean square) of audio data.
pub fn calculate_rms(data: &[f32]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let sum: f64 = data.iter().map(|&s| (s as f64) * (s as f64)).sum();
    (sum / data.len() as f64).sqrt()
}

/// Calculate peak amplitude of audio data.
pub fn calculate_peak(data: &[f32]) -> f32 {
    data.iter().map(|s| s.abs()).fold(0.0f32, f32::max)
}

/// Perform linear interpolation between two samples.
#[inline]
pub fn linear_interpolate(sample1: f32, sample2: f32, fraction: f32) -> f32 {
    sample1 + fraction * (sample2 - sample1)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_buffer(length: usize) -> AudioBuffer {
        // Create a stereo buffer with ascending values per channel
        let left: Vec<f32> = (0..length).map(|i| i as f32 / length as f32).collect();
        let right: Vec<f32> = (0..length).map(|i| -(i as f32 / length as f32)).collect();
        AudioBuffer::new(vec![left, right], 48_000)
    }

    // -- AudioBuffer -------------------------------------------------------

    #[test]
    fn audio_buffer_properties() {
        let buf = make_test_buffer(48_000);
        assert_eq!(buf.num_channels(), 2);
        assert_eq!(buf.length(), 48_000);
        assert!((buf.duration() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn audio_buffer_validation() {
        let buf = make_test_buffer(48_000);
        assert!(buf.validate(2, 48_000).is_ok());
        assert!(buf.validate(1, 48_000).is_err()); // wrong channels
        assert!(buf.validate(2, 44_100).is_err()); // wrong sample rate
    }

    #[test]
    fn audio_buffer_empty_rejected() {
        let buf = AudioBuffer::new(vec![vec![], vec![]], 48_000);
        assert!(buf.validate(2, 48_000).is_err());
    }

    // -- FrameReader construction ------------------------------------------

    #[test]
    fn frame_reader_defaults() {
        let fr = FrameReader::with_defaults();
        assert_eq!(fr.state(), FrameReaderState::Idle);
        assert_eq!(fr.read_position(), 0);
    }

    // -- set_source_buffer --------------------------------------------------

    #[test]
    fn set_source_buffer_valid() {
        let mut fr = FrameReader::with_defaults();
        let buf = make_test_buffer(8192);
        assert!(fr.set_source_buffer(buf).is_ok());
        assert_eq!(fr.read_position(), 0);
    }

    #[test]
    fn set_source_buffer_invalid_channels() {
        let mut fr = FrameReader::with_defaults();
        let mono = AudioBuffer::new(vec![vec![0.0; 1000]], 48_000);
        assert!(fr.set_source_buffer(mono).is_err());
    }

    // -- read_next_frame ---------------------------------------------------

    #[test]
    fn read_frame_requires_reading_state() {
        let mut fr = FrameReader::with_defaults();
        fr.set_source_buffer(make_test_buffer(8192)).unwrap();
        // Idle → no frame
        assert!(fr.read_next_frame().is_none());
    }

    #[test]
    fn read_frame_returns_data() {
        let mut fr = FrameReader::with_defaults();
        fr.set_source_buffer(make_test_buffer(8192)).unwrap();
        fr.start();

        let frame = fr.read_next_frame().unwrap();
        assert_eq!(frame.num_channels(), 2);
        assert_eq!(frame.frame_size(), FRAME_SIZE as usize);
        assert_eq!(fr.stats().total_frames_read, 1);
    }

    #[test]
    fn read_frame_advances_position() {
        let mut fr = FrameReader::with_defaults();
        fr.set_source_buffer(make_test_buffer(8192)).unwrap();
        fr.start();

        fr.read_next_frame();
        assert_eq!(fr.read_position(), FRAME_SIZE as u64);

        fr.read_next_frame();
        // 8192 = 2 * 4096, second read wraps to 0
        assert_eq!(fr.read_position(), 0);
    }

    #[test]
    fn read_frame_loops_on_buffer_boundary() {
        let mut fr = FrameReader::with_defaults();
        // Buffer that's exactly 2 frames long
        let buf_len = (FRAME_SIZE * 2) as usize;
        fr.set_source_buffer(make_test_buffer(buf_len)).unwrap();
        fr.start();

        fr.read_next_frame(); // pos = 4096
        fr.read_next_frame(); // pos = 8192 → loops to 0
        assert_eq!(fr.read_position(), 0);

        // Third read starts again from 0
        let frame = fr.read_next_frame().unwrap();
        assert!(frame.frame_size() > 0);
    }

    #[test]
    fn read_frame_blocked_during_switch() {
        let mut fr = FrameReader::with_defaults();
        fr.set_source_buffer(make_test_buffer(8192)).unwrap();
        fr.start();
        fr.set_switch_pending(true);
        assert!(fr.read_next_frame().is_none());
    }

    // -- switch_source ------------------------------------------------------

    #[test]
    fn switch_source_maintain_position() {
        let mut fr = FrameReader::with_defaults();
        fr.set_source_buffer(make_test_buffer(8192)).unwrap();
        fr.start();
        fr.read_next_frame(); // pos = 4096

        // Switch to same-length buffer, maintain position
        fr.switch_source(make_test_buffer(8192), true).unwrap();
        assert_eq!(fr.read_position(), 4096);
        assert_eq!(fr.stats().buffer_switches, 1);
    }

    #[test]
    fn switch_source_reset_position() {
        let mut fr = FrameReader::with_defaults();
        fr.set_source_buffer(make_test_buffer(8192)).unwrap();
        fr.start();
        fr.read_next_frame(); // pos = 4096

        fr.switch_source(make_test_buffer(8192), false).unwrap();
        assert_eq!(fr.read_position(), 0);
    }

    #[test]
    fn switch_source_scales_position() {
        let mut fr = FrameReader::with_defaults();
        // Start with 8192, read one frame → pos = 4096
        fr.set_source_buffer(make_test_buffer(8192)).unwrap();
        fr.start();
        fr.read_next_frame();
        assert_eq!(fr.read_position(), 4096);

        // Switch to 16384 (double length), maintain position → 4096 * 2 = 8192
        fr.switch_source(make_test_buffer(16384), true).unwrap();
        assert_eq!(fr.read_position(), 8192);
    }

    // -- lifecycle ----------------------------------------------------------

    #[test]
    fn pause_resume() {
        let mut fr = FrameReader::with_defaults();
        fr.set_source_buffer(make_test_buffer(8192)).unwrap();

        fr.start();
        assert_eq!(fr.state(), FrameReaderState::Reading);

        fr.pause();
        assert_eq!(fr.state(), FrameReaderState::Paused);
        assert!(fr.read_next_frame().is_none()); // paused → no reads

        fr.resume();
        assert_eq!(fr.state(), FrameReaderState::Reading);
        assert!(fr.read_next_frame().is_some());
    }

    #[test]
    fn stop_resets_state() {
        let mut fr = FrameReader::with_defaults();
        fr.set_source_buffer(make_test_buffer(8192)).unwrap();
        fr.start();
        fr.read_next_frame();
        fr.stop();
        assert_eq!(fr.state(), FrameReaderState::Idle);
    }

    // -- calculate_frames_per_interval (ADR073 + ADR074) -------------------

    #[test]
    fn frames_per_interval_normal_tempo() {
        let fr = FrameReader::with_defaults();
        // playback_rate = 1.0, queue in normal range (default last_known = 0 → low)
        // Low queue → boost to min(max(1,2), 3) = 2
        assert_eq!(fr.calculate_frames_per_interval(), 2);
    }

    #[test]
    fn frames_per_interval_high_tempo() {
        let mut fr = FrameReader::with_defaults();
        fr.set_playback_rate(1.5);
        fr.update_queue_depth(15); // normal range
        // ceil(1.5) = 2, within normal range → 2
        assert_eq!(fr.calculate_frames_per_interval(), 2);
    }

    #[test]
    fn frames_per_interval_very_high_tempo() {
        let mut fr = FrameReader::with_defaults();
        fr.set_playback_rate(3.0);
        fr.update_queue_depth(15); // normal range
        // ceil(3.0) = 3, capped at max_catch_up (3) → 3
        assert_eq!(fr.calculate_frames_per_interval(), 3);
    }

    #[test]
    fn frames_per_interval_low_queue_boosts() {
        let mut fr = FrameReader::with_defaults();
        fr.set_playback_rate(1.0);
        fr.update_queue_depth(5); // below low threshold (10)
        // base = 1, but low queue → max(1,2).min(3) = 2
        assert_eq!(fr.calculate_frames_per_interval(), 2);
    }

    #[test]
    fn frames_per_interval_high_queue_backs_off() {
        let mut fr = FrameReader::with_defaults();
        fr.set_playback_rate(2.0);
        fr.update_queue_depth(25); // above high threshold (20)
        // base = 2, high queue → (2/2).max(1) = 1
        assert_eq!(fr.calculate_frames_per_interval(), 1);
    }

    // -- deliver_frames -----------------------------------------------------

    #[test]
    fn deliver_frames_to_fifo() {
        let mut fr = FrameReader::with_defaults();
        let buf_len = (FRAME_SIZE * 10) as usize;
        fr.set_source_buffer(make_test_buffer(buf_len)).unwrap();
        fr.start();
        fr.update_queue_depth(15); // normal range

        let mut fifo = FifoBuffer::with_defaults();
        let delivered = fr.deliver_frames(&mut fifo);
        assert!(delivered > 0);
        assert!(fifo.has_frames());
        assert_eq!(fifo.queue_depth(), delivered);
    }

    #[test]
    fn deliver_frames_respects_fifo_capacity() {
        let mut fr = FrameReader::with_defaults();
        let buf_len = (FRAME_SIZE * 10) as usize;
        fr.set_source_buffer(make_test_buffer(buf_len)).unwrap();
        fr.start();
        fr.update_queue_depth(5); // low queue → wants to deliver 2

        let mut fifo = FifoBuffer::new(1); // capacity 1
        let delivered = fr.deliver_frames(&mut fifo);
        assert_eq!(delivered, 1);
    }

    // -- utility functions --------------------------------------------------

    #[test]
    fn interval_ms_calculation() {
        let ms = calculate_interval_ms(4096, 48_000);
        assert!((ms - 85.333).abs() < 0.01);
    }

    #[test]
    fn has_audio_data_test() {
        assert!(!has_audio_data(&[0.0, 0.0, 0.0]));
        assert!(has_audio_data(&[0.0, 0.1, 0.0]));
    }

    #[test]
    fn rms_calculation() {
        let data = vec![1.0f32; 100];
        assert!((calculate_rms(&data) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn peak_calculation() {
        let data = vec![0.1f32, -0.5, 0.3, 0.8, -0.2];
        assert_eq!(calculate_peak(&data), 0.8);
    }

    #[test]
    fn linear_interpolation() {
        assert_eq!(linear_interpolate(0.0, 1.0, 0.5), 0.5);
        assert_eq!(linear_interpolate(0.0, 1.0, 0.0), 0.0);
        assert_eq!(linear_interpolate(0.0, 1.0, 1.0), 1.0);
        assert_eq!(linear_interpolate(2.0, 4.0, 0.25), 2.5);
    }

    // -- cleanup -----------------------------------------------------------

    #[test]
    fn cleanup_resets_everything() {
        let mut fr = FrameReader::with_defaults();
        fr.set_source_buffer(make_test_buffer(8192)).unwrap();
        fr.start();
        fr.read_next_frame();
        fr.cleanup();

        assert_eq!(fr.state(), FrameReaderState::Idle);
        assert_eq!(fr.read_position(), 0);
        assert!(!fr.is_switch_pending());
    }
}
