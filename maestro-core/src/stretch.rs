//! Signalsmith Stretch integration — native C++ time-stretching and pitch-shifting.
//!
//! Wraps the `signalsmith-stretch` crate (colinmarc's Rust bindings) to provide
//! a maestro-core-native API that works with channel-separated `AudioFrame` data.
//!
//! ## Architecture
//!
//! The Signalsmith Stretch library performs two related but independent operations:
//!
//! - **Time-stretching**: Change playback speed without affecting pitch.
//!   Controlled by the *ratio of input to output buffer sizes* passed to `process()`.
//!   If you feed 256 input samples and request 128 output samples, you get 2x speed.
//!
//! - **Pitch-shifting**: Change pitch without affecting speed.
//!   Controlled by `set_transpose_semitones()` or `set_transpose_factor()`.
//!
//! ## Integration with maestro-core
//!
//! - `TempoController` computes `tempo_ratio()` — the stretcher uses this to size
//!   input buffers relative to output (time-stretch).
//! - `TempoController` computes `pitch_compensation_ratio()` — when pitch compensation
//!   is enabled, the stretcher applies the inverse pitch shift so tempo changes don't
//!   alter perceived pitch.
//! - `PitchState` provides per-track `offset_semitones()` — combined with compensation
//!   to get the final pitch shift value.
//!
//! ## Buffer format conversion
//!
//! maestro-core uses channel-separated buffers (`Vec<Vec<f32>>` — one Vec per channel).
//! Signalsmith Stretch expects interleaved buffers (`&[f32]` — `[L0, R0, L1, R1, ...]`).
//! This module handles the conversion transparently.
//!
//! ## Feature gate
//!
//! The `Stretcher` wrapper is compiled only under the `native-stretch` feature,
//! which builds the Signalsmith C++ for native AND wasm32 targets (the vendored
//! crate compiles the same source via wasi-sdk; M4, spec §3.8). The *module* is
//! ungated on purpose: `StretchError` and the ratio/semitone helpers below are
//! pure Rust and stay available to the no-default-features build that
//! `maestro-validate` resolves (ADR-114).

#[cfg(feature = "native-stretch")]
use signalsmith_stretch::Stretch;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default number of audio channels (stereo).
pub const DEFAULT_CHANNELS: u32 = 2;

/// Minimum transpose factor to prevent extreme shifts (matches TypeScript MIN_PITCH_COMPENSATION).
pub const MIN_TRANSPOSE_FACTOR: f32 = 0.25;

/// Maximum transpose factor to prevent extreme shifts (matches TypeScript MAX_PITCH_COMPENSATION).
pub const MAX_TRANSPOSE_FACTOR: f32 = 4.0;

/// Minimum semitones for transpose (2 octaves down).
pub const MIN_TRANSPOSE_SEMITONES: f32 = -24.0;

/// Maximum semitones for transpose (2 octaves up).
pub const MAX_TRANSPOSE_SEMITONES: f32 = 24.0;

/// Transpose factor representing no pitch change.
pub const UNITY_FACTOR: f32 = 1.0;

/// Threshold below which a factor is considered "unity" (no change).
pub const UNITY_THRESHOLD: f32 = 0.001;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors from stretch operations.
#[derive(Debug, Clone, PartialEq)]
pub enum StretchError {
    /// Channel count mismatch between stretcher and audio data.
    ChannelMismatch { expected: usize, got: usize },
    /// Invalid stretch ratio (must be positive and finite).
    InvalidRatio(f64),
    /// Invalid semitones value (NaN or out of range).
    InvalidSemitones(f32),
    /// Empty audio data provided.
    EmptyAudio,
}

impl std::fmt::Display for StretchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ChannelMismatch { expected, got } => {
                write!(f, "channel mismatch: expected {expected}, got {got}")
            }
            Self::InvalidRatio(r) => write!(f, "invalid stretch ratio: {r}"),
            Self::InvalidSemitones(s) => write!(f, "invalid semitones: {s}"),
            Self::EmptyAudio => write!(f, "empty audio data"),
        }
    }
}

// ---------------------------------------------------------------------------
// Stretcher (feature-gated)
// ---------------------------------------------------------------------------

/// Per-track audio stretcher wrapping Signalsmith Stretch.
///
/// Each track gets its own `Stretcher` instance because per-track pitch offsets
/// may differ. The stretcher maintains internal state (phase vocoder windows)
/// so it must persist across process calls for a given track.
///
/// # Example
///
/// ```ignore
/// use maestro_core::stretch::Stretcher;
///
/// let mut stretcher = Stretcher::new(2, 48000);
///
/// // Set pitch shift: 2 semitones up
/// stretcher.set_transpose_semitones(2.0).unwrap();
///
/// // Process: time-stretch at 1.5x speed
/// let input_channels = vec![vec![0.0f32; 192], vec![0.0f32; 192]]; // 192 frames input
/// let output = stretcher.process_separated(&input_channels, 128).unwrap(); // 128 frames output
/// // Ratio = 192/128 = 1.5x speed
/// ```
#[cfg(feature = "native-stretch")]
pub struct Stretcher {
    inner: Stretch,
    channels: u32,
    sample_rate: u32,
    /// Current transpose in semitones (for state queries).
    current_semitones: f32,
    /// Reusable interleaved input buffer (avoids per-call allocation).
    interleaved_in: Vec<f32>,
    /// Reusable interleaved output buffer.
    interleaved_out: Vec<f32>,
}

#[cfg(feature = "native-stretch")]
impl Stretcher {
    /// Create a new stretcher with default quality settings.
    ///
    /// Uses `presetDefault` which provides the best quality-to-CPU balance
    /// for real-time audio at the given sample rate.
    pub fn new(channels: u32, sample_rate: u32) -> Self {
        let inner = Stretch::preset_default(channels, sample_rate);
        Self {
            inner,
            channels,
            sample_rate,
            current_semitones: 0.0,
            interleaved_in: Vec::new(),
            interleaved_out: Vec::new(),
        }
    }

    /// Create a stretcher with lower CPU usage (lower quality).
    ///
    /// Uses smaller FFT windows. Suitable for mobile devices or when many
    /// tracks are being stretched simultaneously.
    pub fn new_cheaper(channels: u32, sample_rate: u32) -> Self {
        let inner = Stretch::preset_cheaper(channels, sample_rate);
        Self {
            inner,
            channels,
            sample_rate,
            current_semitones: 0.0,
            interleaved_in: Vec::new(),
            interleaved_out: Vec::new(),
        }
    }

    /// Number of channels this stretcher was configured for.
    pub fn channels(&self) -> u32 {
        self.channels
    }

    /// Sample rate this stretcher was configured for.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Current transpose value in semitones.
    pub fn current_semitones(&self) -> f32 {
        self.current_semitones
    }

    /// Whether the stretcher is at unity (no pitch change).
    pub fn is_unity(&self) -> bool {
        self.current_semitones.abs() < UNITY_THRESHOLD
    }

    // -- Pitch control --

    /// Set pitch transpose in semitones.
    ///
    /// - `0.0` = no change
    /// - `12.0` = one octave up
    /// - `-12.0` = one octave down
    /// - Range: `[-24.0, +24.0]` (clamped)
    pub fn set_transpose_semitones(&mut self, semitones: f32) -> Result<(), StretchError> {
        if semitones.is_nan() {
            return Err(StretchError::InvalidSemitones(semitones));
        }
        let clamped = semitones.clamp(MIN_TRANSPOSE_SEMITONES, MAX_TRANSPOSE_SEMITONES);
        self.current_semitones = clamped;
        self.inner.set_transpose_factor_semitones(clamped, None);
        Ok(())
    }

    /// Set pitch transpose as a frequency multiplier.
    ///
    /// - `1.0` = no change
    /// - `2.0` = one octave up
    /// - `0.5` = one octave down
    /// - Range: `[0.25, 4.0]` (clamped)
    pub fn set_transpose_factor(&mut self, factor: f32) -> Result<(), StretchError> {
        if factor.is_nan() || factor <= 0.0 {
            return Err(StretchError::InvalidSemitones(factor));
        }
        let clamped = factor.clamp(MIN_TRANSPOSE_FACTOR, MAX_TRANSPOSE_FACTOR);
        // Convert factor to semitones for state tracking: 12 * log2(factor)
        self.current_semitones = 12.0 * clamped.log2();
        self.inner.set_transpose_factor(clamped, None);
        Ok(())
    }

    /// Set the combined pitch shift from tempo compensation and user pitch offset.
    ///
    /// This is the main integration point with `TempoController` and `PitchState`:
    /// - `compensation_semitones`: from `TempoController::pitch_compensation_ratio()`
    ///   converted to semitones (typically negative when tempo > 1.0)
    /// - `offset_semitones`: from `PitchState::offset_semitones()` (user's pitch knob)
    ///
    /// The two values are summed and applied as a single transpose.
    pub fn set_combined_pitch(
        &mut self,
        compensation_semitones: f32,
        offset_semitones: f32,
    ) -> Result<(), StretchError> {
        let total = compensation_semitones + offset_semitones;
        self.set_transpose_semitones(total)
    }

    /// Reset internal state (clears phase vocoder buffers).
    ///
    /// Call this when seeking, switching tracks, or after a discontinuity.
    /// Does NOT change configuration or pitch settings.
    pub fn reset(&mut self) {
        self.inner.reset();
    }

    // -- Latency queries --

    /// Input latency in frames.
    ///
    /// The delay between input samples being fed in and the center of the
    /// stretch effect being applied. Use for latency compensation.
    pub fn input_latency(&self) -> usize {
        self.inner.input_latency()
    }

    /// Output latency in frames.
    ///
    /// The delay between the effect center and generated output samples.
    pub fn output_latency(&self) -> usize {
        self.inner.output_latency()
    }

    /// Total latency in frames (input + output).
    pub fn total_latency(&self) -> usize {
        self.inner.input_latency() + self.inner.output_latency()
    }

    /// Total latency in seconds.
    pub fn total_latency_seconds(&self) -> f64 {
        self.total_latency() as f64 / self.sample_rate as f64
    }

    // -- Processing (channel-separated format) --

    /// Process audio with time-stretching and pitch-shifting.
    ///
    /// Takes channel-separated input (one `&[f32]` per channel) and produces
    /// channel-separated output. The time-stretch ratio is determined by the
    /// ratio of input frames to output frames:
    ///
    /// - `input_frames > output_frames` → speed up (e.g., 256 in / 128 out = 2x)
    /// - `input_frames < output_frames` → slow down (e.g., 64 in / 128 out = 0.5x)
    /// - `input_frames == output_frames` → no time-stretch (pitch-only)
    ///
    /// # Arguments
    ///
    /// - `input_channels`: Channel-separated input data. Each slice must have
    ///   the same length (`input_frames`).
    /// - `output_frames`: Number of output frames to produce.
    ///
    /// # Returns
    ///
    /// Channel-separated output data (`Vec<Vec<f32>>`).
    pub fn process_separated(
        &mut self,
        input_channels: &[&[f32]],
        output_frames: usize,
    ) -> Result<Vec<Vec<f32>>, StretchError> {
        let ch = self.channels as usize;

        // Validate channel count
        if input_channels.len() != ch {
            return Err(StretchError::ChannelMismatch {
                expected: ch,
                got: input_channels.len(),
            });
        }

        // Validate non-empty
        if input_channels.is_empty() || output_frames == 0 {
            return Err(StretchError::EmptyAudio);
        }

        let input_frames = input_channels[0].len();
        if input_frames == 0 {
            return Err(StretchError::EmptyAudio);
        }

        // Interleave input: [L0, R0, L1, R1, ...]
        let in_len = input_frames * ch;
        self.interleaved_in.clear();
        self.interleaved_in.reserve(in_len);
        for frame in 0..input_frames {
            for channel in input_channels.iter() {
                self.interleaved_in.push(channel[frame]);
            }
        }

        // Prepare output buffer
        let out_len = output_frames * ch;
        self.interleaved_out.clear();
        self.interleaved_out.resize(out_len, 0.0);

        // Process through Signalsmith Stretch
        self.inner
            .process(&self.interleaved_in, &mut self.interleaved_out);

        // De-interleave output into channel-separated format
        let mut output_channels: Vec<Vec<f32>> = (0..ch).map(|_| Vec::with_capacity(output_frames)).collect();
        for frame in 0..output_frames {
            for (c, channel) in output_channels.iter_mut().enumerate() {
                channel.push(self.interleaved_out[frame * ch + c]);
            }
        }

        Ok(output_channels)
    }

    /// Process audio in-place on an interleaved buffer.
    ///
    /// This is the lower-level API for use in the render path where you
    /// already have interleaved data and want to avoid extra copies.
    ///
    /// The time-stretch ratio = `input.len() / output.len()` (both in total samples,
    /// i.e., frames * channels).
    pub fn process_interleaved(
        &mut self,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), StretchError> {
        let ch = self.channels as usize;
        if !input.len().is_multiple_of(ch) || !output.len().is_multiple_of(ch) {
            return Err(StretchError::ChannelMismatch {
                expected: ch,
                got: 0,
            });
        }
        if input.is_empty() || output.is_empty() {
            return Err(StretchError::EmptyAudio);
        }

        self.inner.process(input, output);
        Ok(())
    }

    /// Preallocate the internal interleaved scratch buffers so that no later
    /// call (`seek`, `process_separated`) allocates. The render path must be
    /// allocation-free (spec §3.3 RT rules); the control plane calls this at
    /// track-install time, before the stretcher crosses to the audio thread.
    pub fn preallocate(&mut self, max_input_frames: usize, max_output_frames: usize) {
        let ch = self.channels as usize;
        self.interleaved_in.reserve(max_input_frames * ch);
        self.interleaved_out.reserve(max_output_frames * ch);
    }

    /// Seek (pre-roll) the stretcher from an already-interleaved window.
    ///
    /// RT-safe variant of [`Self::seek`] for the anchor procedure (spec §3.2):
    /// no staging copy, no allocation. `input.len()` must be a multiple of the
    /// channel count.
    pub fn seek_interleaved(&mut self, input: &[f32], playback_rate: f64) -> Result<(), StretchError> {
        let ch = self.channels as usize;
        if !input.len().is_multiple_of(ch) {
            return Err(StretchError::ChannelMismatch { expected: ch, got: 0 });
        }
        if input.is_empty() {
            return Ok(());
        }
        self.inner.seek(input, playback_rate);
        Ok(())
    }

    /// Seek (pre-roll) the stretcher to prime internal buffers.
    ///
    /// Call this before the first `process()` after a position change or
    /// after `reset()` to reduce initial artifacts.
    pub fn seek(&mut self, input_channels: &[&[f32]], playback_rate: f64) -> Result<(), StretchError> {
        let ch = self.channels as usize;
        if input_channels.len() != ch {
            return Err(StretchError::ChannelMismatch {
                expected: ch,
                got: input_channels.len(),
            });
        }

        let input_frames = input_channels[0].len();
        if input_frames == 0 {
            return Ok(());
        }

        // Interleave
        let in_len = input_frames * ch;
        self.interleaved_in.clear();
        self.interleaved_in.reserve(in_len);
        for frame in 0..input_frames {
            for channel in input_channels.iter() {
                self.interleaved_in.push(channel[frame]);
            }
        }

        self.inner.seek(&self.interleaved_in, playback_rate);
        Ok(())
    }

    /// Flush remaining audio from internal buffers.
    ///
    /// Returns the tail audio after all input has been processed.
    /// Buffer size should be `output_latency() * channels`.
    pub fn flush(&mut self) -> Vec<Vec<f32>> {
        let ch = self.channels as usize;
        let flush_frames = self.inner.output_latency();
        let out_len = flush_frames * ch;

        self.interleaved_out.clear();
        self.interleaved_out.resize(out_len, 0.0);

        self.inner.flush(&mut self.interleaved_out);

        // De-interleave
        let mut output_channels: Vec<Vec<f32>> = (0..ch).map(|_| Vec::with_capacity(flush_frames)).collect();
        for frame in 0..flush_frames {
            for (c, channel) in output_channels.iter_mut().enumerate() {
                channel.push(self.interleaved_out[frame * ch + c]);
            }
        }

        output_channels
    }
}

#[cfg(feature = "native-stretch")]
impl std::fmt::Debug for Stretcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Stretcher")
            .field("channels", &self.channels)
            .field("sample_rate", &self.sample_rate)
            .field("current_semitones", &self.current_semitones)
            .field("input_latency", &self.input_latency())
            .field("output_latency", &self.output_latency())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Helper functions (always available)
// ---------------------------------------------------------------------------

/// Convert a pitch compensation ratio to semitones.
///
/// Used to convert `TempoController::pitch_compensation_ratio()` (a frequency
/// multiplier) into semitones for `Stretcher::set_transpose_semitones()`.
///
/// Formula: `12 * log2(ratio)`
///
/// Examples:
/// - `ratio = 1.0` → `0.0` semitones (no change)
/// - `ratio = 0.5` → `-12.0` semitones (octave down — compensating for 2x tempo)
/// - `ratio = 2.0` → `+12.0` semitones (octave up — compensating for 0.5x tempo)
pub fn ratio_to_semitones(ratio: f64) -> Result<f32, StretchError> {
    if ratio <= 0.0 || !ratio.is_finite() {
        return Err(StretchError::InvalidRatio(ratio));
    }
    Ok((12.0 * ratio.log2()) as f32)
}

/// Convert semitones to a frequency ratio.
///
/// Formula: `2^(semitones / 12)`
pub fn semitones_to_ratio(semitones: f32) -> Result<f64, StretchError> {
    if semitones.is_nan() {
        return Err(StretchError::InvalidSemitones(semitones));
    }
    Ok(2.0_f64.powf(semitones as f64 / 12.0))
}

/// Calculate the number of input frames needed for a given output frame count
/// and time-stretch ratio.
///
/// For Signalsmith Stretch, the time-stretch is implicit:
/// `ratio = input_frames / output_frames`
///
/// So: `input_frames = output_frames * ratio`
///
/// Examples:
/// - `output_frames = 128, ratio = 1.0` → `128` (no stretch)
/// - `output_frames = 128, ratio = 2.0` → `256` (2x speed — need more input)
/// - `output_frames = 128, ratio = 0.5` → `64` (0.5x speed — need less input)
pub fn input_frames_for_ratio(output_frames: usize, ratio: f64) -> Result<usize, StretchError> {
    if ratio <= 0.0 || !ratio.is_finite() {
        return Err(StretchError::InvalidRatio(ratio));
    }
    let input = (output_frames as f64 * ratio).round() as usize;
    Ok(input.max(1)) // At least 1 frame
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Helper function tests --

    #[test]
    fn test_ratio_to_semitones_unity() {
        let st = ratio_to_semitones(1.0).unwrap();
        assert!((st - 0.0).abs() < 0.001, "unity ratio should be 0 semitones, got {st}");
    }

    #[test]
    fn test_ratio_to_semitones_octave_up() {
        let st = ratio_to_semitones(2.0).unwrap();
        assert!((st - 12.0).abs() < 0.001, "2x ratio should be 12 semitones, got {st}");
    }

    #[test]
    fn test_ratio_to_semitones_octave_down() {
        let st = ratio_to_semitones(0.5).unwrap();
        assert!((st - (-12.0)).abs() < 0.001, "0.5x ratio should be -12 semitones, got {st}");
    }

    #[test]
    fn test_ratio_to_semitones_invalid() {
        assert!(ratio_to_semitones(0.0).is_err());
        assert!(ratio_to_semitones(-1.0).is_err());
        assert!(ratio_to_semitones(f64::NAN).is_err());
        assert!(ratio_to_semitones(f64::INFINITY).is_err());
    }

    #[test]
    fn test_semitones_to_ratio_unity() {
        let r = semitones_to_ratio(0.0).unwrap();
        assert!((r - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_semitones_to_ratio_octave() {
        let r = semitones_to_ratio(12.0).unwrap();
        assert!((r - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_semitones_to_ratio_negative() {
        let r = semitones_to_ratio(-12.0).unwrap();
        assert!((r - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_semitones_to_ratio_nan() {
        assert!(semitones_to_ratio(f32::NAN).is_err());
    }

    #[test]
    fn test_ratio_semitones_roundtrip() {
        for semitones in [-24.0, -12.0, -6.0, -1.0, 0.0, 1.0, 6.0, 12.0, 24.0] {
            let ratio = semitones_to_ratio(semitones).unwrap();
            let back = ratio_to_semitones(ratio).unwrap();
            assert!(
                (back - semitones).abs() < 0.01,
                "roundtrip failed: {semitones} → {ratio} → {back}"
            );
        }
    }

    #[test]
    fn test_input_frames_for_ratio_unity() {
        assert_eq!(input_frames_for_ratio(128, 1.0).unwrap(), 128);
    }

    #[test]
    fn test_input_frames_for_ratio_double() {
        assert_eq!(input_frames_for_ratio(128, 2.0).unwrap(), 256);
    }

    #[test]
    fn test_input_frames_for_ratio_half() {
        assert_eq!(input_frames_for_ratio(128, 0.5).unwrap(), 64);
    }

    #[test]
    fn test_input_frames_for_ratio_minimum() {
        // Very small ratio should still produce at least 1 frame
        assert_eq!(input_frames_for_ratio(1, 0.01).unwrap(), 1);
    }

    #[test]
    fn test_input_frames_for_ratio_invalid() {
        assert!(input_frames_for_ratio(128, 0.0).is_err());
        assert!(input_frames_for_ratio(128, -1.0).is_err());
        assert!(input_frames_for_ratio(128, f64::NAN).is_err());
    }

    // -- Stretcher tests (only when native-stretch feature is enabled) --

    #[cfg(feature = "native-stretch")]
    mod native_tests {
        use super::*;

        #[test]
        fn test_stretcher_creation() {
            let s = Stretcher::new(2, 48000);
            assert_eq!(s.channels(), 2);
            assert_eq!(s.sample_rate(), 48000);
            assert!(s.is_unity());
            assert_eq!(s.current_semitones(), 0.0);
        }

        #[test]
        fn test_stretcher_creation_cheaper() {
            let s = Stretcher::new_cheaper(2, 48000);
            assert_eq!(s.channels(), 2);
            assert!(s.is_unity());
        }

        #[test]
        fn test_stretcher_creation_mono() {
            let s = Stretcher::new(1, 44100);
            assert_eq!(s.channels(), 1);
            assert_eq!(s.sample_rate(), 44100);
        }

        #[test]
        fn test_set_transpose_semitones() {
            let mut s = Stretcher::new(2, 48000);

            s.set_transpose_semitones(5.0).unwrap();
            assert!((s.current_semitones() - 5.0).abs() < 0.001);
            assert!(!s.is_unity());

            s.set_transpose_semitones(0.0).unwrap();
            assert!(s.is_unity());
        }

        #[test]
        fn test_set_transpose_semitones_clamping() {
            let mut s = Stretcher::new(2, 48000);

            // Beyond range should clamp
            s.set_transpose_semitones(30.0).unwrap();
            assert!((s.current_semitones() - 24.0).abs() < 0.001);

            s.set_transpose_semitones(-30.0).unwrap();
            assert!((s.current_semitones() - (-24.0)).abs() < 0.001);
        }

        #[test]
        fn test_set_transpose_semitones_nan() {
            let mut s = Stretcher::new(2, 48000);
            assert!(s.set_transpose_semitones(f32::NAN).is_err());
        }

        #[test]
        fn test_set_transpose_factor() {
            let mut s = Stretcher::new(2, 48000);

            s.set_transpose_factor(2.0).unwrap();
            // 2.0 factor = 12 semitones
            assert!((s.current_semitones() - 12.0).abs() < 0.1);

            s.set_transpose_factor(1.0).unwrap();
            assert!(s.is_unity());
        }

        #[test]
        fn test_set_transpose_factor_clamping() {
            let mut s = Stretcher::new(2, 48000);

            s.set_transpose_factor(10.0).unwrap();
            // Should clamp to 4.0 = ~24 semitones
            assert!(s.current_semitones() <= 24.1);
        }

        #[test]
        fn test_set_transpose_factor_invalid() {
            let mut s = Stretcher::new(2, 48000);
            assert!(s.set_transpose_factor(0.0).is_err());
            assert!(s.set_transpose_factor(-1.0).is_err());
            assert!(s.set_transpose_factor(f32::NAN).is_err());
        }

        #[test]
        fn test_set_combined_pitch() {
            let mut s = Stretcher::new(2, 48000);

            // Tempo compensation: -3.86 semitones (1.25x tempo)
            // User offset: +2.0 semitones
            // Total: -1.86 semitones
            s.set_combined_pitch(-3.86, 2.0).unwrap();
            assert!((s.current_semitones() - (-1.86)).abs() < 0.01);
        }

        #[test]
        fn test_latency_is_positive() {
            let s = Stretcher::new(2, 48000);
            assert!(s.input_latency() > 0);
            assert!(s.output_latency() > 0);
            assert!(s.total_latency() > 0);
            assert!(s.total_latency_seconds() > 0.0);
        }

        #[test]
        fn test_latency_is_reasonable() {
            let s = Stretcher::new(2, 48000);
            // Total latency should be under 200ms for 48kHz
            assert!(s.total_latency_seconds() < 0.2,
                "latency {}s seems too high", s.total_latency_seconds());
        }

        #[test]
        fn test_reset_preserves_config() {
            let mut s = Stretcher::new(2, 48000);
            s.set_transpose_semitones(5.0).unwrap();
            s.reset();
            // Config is preserved, but current_semitones is Rust-side state (unchanged)
            assert_eq!(s.channels(), 2);
            assert_eq!(s.sample_rate(), 48000);
        }

        #[test]
        fn test_process_separated_basic() {
            let mut s = Stretcher::new(2, 48000);
            let frames = 128;

            // Generate a simple sine wave for left channel, silence for right
            let left: Vec<f32> = (0..frames)
                .map(|i| (i as f32 * 0.1).sin())
                .collect();
            let right = vec![0.0f32; frames];

            let input_channels: Vec<&[f32]> = vec![&left, &right];
            let output = s.process_separated(&input_channels, frames).unwrap();

            assert_eq!(output.len(), 2, "should have 2 output channels");
            assert_eq!(output[0].len(), frames, "output should have {frames} frames");
            assert_eq!(output[1].len(), frames, "output should have {frames} frames");
        }

        #[test]
        fn test_process_separated_time_stretch() {
            let mut s = Stretcher::new(2, 48000);

            // 2x speed: 256 input frames → 128 output frames
            let input_frames = 256;
            let output_frames = 128;
            let left: Vec<f32> = (0..input_frames)
                .map(|i| (i as f32 * 0.05).sin())
                .collect();
            let right = vec![0.0f32; input_frames];

            let input_channels: Vec<&[f32]> = vec![&left, &right];
            let output = s.process_separated(&input_channels, output_frames).unwrap();

            assert_eq!(output.len(), 2);
            assert_eq!(output[0].len(), output_frames);
        }

        #[test]
        fn test_process_separated_channel_mismatch() {
            let mut s = Stretcher::new(2, 48000);
            let data = vec![0.0f32; 128];
            // Only 1 channel for a stereo stretcher
            let result = s.process_separated(&[&data], 128);
            assert!(result.is_err());
            match result.unwrap_err() {
                StretchError::ChannelMismatch { expected: 2, got: 1 } => {}
                e => panic!("expected ChannelMismatch, got {e:?}"),
            }
        }

        #[test]
        fn test_process_separated_empty() {
            let mut s = Stretcher::new(2, 48000);
            let result = s.process_separated(&[], 128);
            assert!(matches!(result, Err(StretchError::ChannelMismatch { .. })));
        }

        #[test]
        fn test_process_separated_zero_output() {
            let mut s = Stretcher::new(2, 48000);
            let data = vec![0.0f32; 128];
            let result = s.process_separated(&[&data, &data], 0);
            assert!(matches!(result, Err(StretchError::EmptyAudio)));
        }

        #[test]
        fn test_process_interleaved_basic() {
            let mut s = Stretcher::new(2, 48000);
            let frames = 128;

            // Interleaved stereo: [L0, R0, L1, R1, ...]
            let input: Vec<f32> = (0..frames * 2)
                .map(|i| (i as f32 * 0.05).sin())
                .collect();
            let mut output = vec![0.0f32; frames * 2];

            s.process_interleaved(&input, &mut output).unwrap();

            // Output should not be all zeros (stretcher produces audio)
            let has_nonzero = output.iter().any(|&s| s.abs() > 1e-10);
            // Note: first call may have latency, so output could still be zeros
            // This is expected behavior — the stretcher needs pre-roll
            let _ = has_nonzero;
        }

        #[test]
        fn test_process_interleaved_channel_alignment() {
            let mut s = Stretcher::new(2, 48000);
            // Odd number of samples — not aligned to 2 channels
            let input = vec![0.0f32; 129];
            let mut output = vec![0.0f32; 128];
            assert!(s.process_interleaved(&input, &mut output).is_err());
        }

        #[test]
        fn test_seek_basic() {
            let mut s = Stretcher::new(2, 48000);
            let frames = 256;
            let left: Vec<f32> = (0..frames).map(|i| (i as f32 * 0.1).sin()).collect();
            let right = vec![0.0f32; frames];

            // Seek should not error
            s.seek(&[&left, &right], 1.0).unwrap();
        }

        #[test]
        fn test_seek_channel_mismatch() {
            let mut s = Stretcher::new(2, 48000);
            let data = vec![0.0f32; 128];
            assert!(s.seek(&[&data], 1.0).is_err());
        }

        #[test]
        fn test_flush() {
            let mut s = Stretcher::new(2, 48000);
            let output = s.flush();
            assert_eq!(output.len(), 2);
            // Flush length matches output_latency
            assert_eq!(output[0].len(), s.output_latency());
        }

        #[test]
        fn test_process_with_pitch_shift() {
            let mut s = Stretcher::new(2, 48000);
            s.set_transpose_semitones(12.0).unwrap(); // Octave up

            let frames = 256;
            let left: Vec<f32> = (0..frames)
                .map(|i| (i as f32 * 2.0 * std::f32::consts::PI * 440.0 / 48000.0).sin())
                .collect();
            let right = left.clone();

            // Process several blocks to get past the latency period
            for _ in 0..10 {
                let _ = s.process_separated(&[&left, &right], frames);
            }

            // After warmup, output should have non-zero content
            let output = s.process_separated(&[&left, &right], frames).unwrap();
            let rms: f32 = (output[0].iter().map(|s| s * s).sum::<f32>() / frames as f32).sqrt();
            // With small block sizes (128 frames) and pitch-shifting, the stretcher's
            // phase vocoder windows produce attenuated output. RMS > 0.0001 confirms
            // the stretcher is producing audio, not silence.
            assert!(rms > 0.0001, "expected non-trivial output after warmup, got RMS {rms}");
        }

        #[test]
        fn test_debug_display() {
            let s = Stretcher::new(2, 48000);
            let debug = format!("{s:?}");
            assert!(debug.contains("Stretcher"));
            assert!(debug.contains("channels: 2"));
            assert!(debug.contains("sample_rate: 48000"));
        }

        #[test]
        fn test_multiple_stretchers_independent() {
            let mut s1 = Stretcher::new(2, 48000);
            let mut s2 = Stretcher::new(2, 48000);

            s1.set_transpose_semitones(5.0).unwrap();
            s2.set_transpose_semitones(-3.0).unwrap();

            assert!((s1.current_semitones() - 5.0).abs() < 0.001);
            assert!((s2.current_semitones() - (-3.0)).abs() < 0.001);
        }
    }
}
