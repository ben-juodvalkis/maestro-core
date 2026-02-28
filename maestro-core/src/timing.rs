//! Timing conversion utilities for audio engine synchronization.
//!
//! The audio engine uses sample-based timing internally for perfect synchronization.
//! These utilities provide clean conversions between samples, time, and beats.
//!
//! Ported from: `src/domains/core/utils/timing-utils.ts`
//! Constants from: `src/config/audio-constants.ts`

// ---------------------------------------------------------------------------
// Constants (from audio-constants.ts)
// ---------------------------------------------------------------------------

/// Default sample rate (48 kHz).
pub const DEFAULT_SAMPLE_RATE: u32 = 48_000;

/// Default worklet frame size (Web Audio API quantum).
pub const WORKLET_FRAME_SIZE: u32 = 128;

/// Default frame size for bulk processing.
pub const FRAME_SIZE: u32 = 4096;

/// Default time signature (beats per bar).
pub const DEFAULT_TIME_SIGNATURE: u32 = 4;

/// Seconds per minute — used in BPM conversions.
pub const SECONDS_PER_MINUTE: f64 = 60.0;

/// Minimum allowed tempo (BPM).
pub const MIN_TEMPO: f64 = 50.0;

/// Maximum allowed tempo (BPM).
pub const MAX_TEMPO: f64 = 150.0;

/// Default tempo (BPM).
pub const DEFAULT_TEMPO: f64 = 100.0;

/// Default gain (volume multiplier).
pub const DEFAULT_GAIN: f64 = 1.0;

/// Minimum gain.
pub const MIN_GAIN: f64 = 0.0;

/// Maximum gain.
pub const MAX_GAIN: f64 = 2.0;

/// Maximum number of audio channels.
pub const MAX_CHANNELS: u32 = 4;

/// Default number of audio channels.
pub const DEFAULT_CHANNELS: u32 = 2;

/// Scaling factor for fractional beat counter.
/// Uses 1,000,000 for 6 decimal precision with integer atomics.
pub const BEAT_POSITION_SCALE_FACTOR: u64 = 1_000_000;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that can occur during timing conversions.
#[derive(Debug, Clone, PartialEq)]
pub enum TimingError {
    /// Tempo must be greater than 0.
    InvalidTempo(f64),
    /// Sample rate must be greater than 0.
    InvalidSampleRate(f64),
    /// Value must be non-negative.
    NegativeValue { name: &'static str, value: f64 },
    /// Frame size must be greater than 0.
    InvalidFrameSize(u32),
    /// Track length must be greater than 0.
    InvalidTrackLength(f64),
}

impl std::fmt::Display for TimingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TimingError::InvalidTempo(v) => {
                write!(f, "Invalid tempo: {v}. Tempo must be greater than 0")
            }
            TimingError::InvalidSampleRate(v) => {
                write!(f, "Invalid sample rate: {v}. Sample rate must be greater than 0")
            }
            TimingError::NegativeValue { name, value } => {
                write!(f, "Invalid {name}: {value}. {name} must be non-negative")
            }
            TimingError::InvalidFrameSize(v) => {
                write!(f, "Invalid frame size: {v}. Frame size must be greater than 0")
            }
            TimingError::InvalidTrackLength(v) => {
                write!(
                    f,
                    "Invalid track length: {v}. Track length must be greater than 0"
                )
            }
        }
    }
}

impl std::error::Error for TimingError {}

// ---------------------------------------------------------------------------
// Conversion functions
// ---------------------------------------------------------------------------

/// Convert time in seconds to sample position (rounded to nearest integer).
#[inline]
pub fn time_to_samples(time: f64, sample_rate: u32) -> i64 {
    (time * sample_rate as f64).round() as i64
}

/// Convert sample position to time in seconds.
#[inline]
pub fn samples_to_time(samples: i64, sample_rate: u32) -> f64 {
    samples as f64 / sample_rate as f64
}

/// Calculate elapsed samples between two sample positions.
#[inline]
pub fn get_elapsed_samples(current_samples: i64, start_samples: i64) -> i64 {
    current_samples - start_samples
}

/// Convert beats to sample position.
///
/// Returns `Err` if tempo ≤ 0, sample_rate ≤ 0, or beat < 0.
pub fn beat_to_samples(
    beat: f64,
    tempo: f64,
    sample_rate: u32,
) -> Result<i64, TimingError> {
    if tempo <= 0.0 {
        return Err(TimingError::InvalidTempo(tempo));
    }
    if sample_rate == 0 {
        return Err(TimingError::InvalidSampleRate(sample_rate as f64));
    }
    if beat < 0.0 {
        return Err(TimingError::NegativeValue {
            name: "beat",
            value: beat,
        });
    }
    let spb = (sample_rate as f64 * SECONDS_PER_MINUTE) / tempo;
    Ok((beat * spb).round() as i64)
}

/// Convert sample position to beats.
///
/// Returns `Err` if tempo ≤ 0, sample_rate ≤ 0, or samples < 0.
pub fn samples_to_beat(
    samples: i64,
    tempo: f64,
    sample_rate: u32,
) -> Result<f64, TimingError> {
    if tempo <= 0.0 {
        return Err(TimingError::InvalidTempo(tempo));
    }
    if sample_rate == 0 {
        return Err(TimingError::InvalidSampleRate(sample_rate as f64));
    }
    if samples < 0 {
        return Err(TimingError::NegativeValue {
            name: "samples",
            value: samples as f64,
        });
    }
    let spb = (sample_rate as f64 * SECONDS_PER_MINUTE) / tempo;
    Ok(samples as f64 / spb)
}

/// Calculate samples per beat for a given tempo and sample rate.
///
/// Returns `Err` if tempo ≤ 0 or sample_rate ≤ 0.
pub fn samples_per_beat(tempo: f64, sample_rate: u32) -> Result<f64, TimingError> {
    if tempo <= 0.0 {
        return Err(TimingError::InvalidTempo(tempo));
    }
    if sample_rate == 0 {
        return Err(TimingError::InvalidSampleRate(sample_rate as f64));
    }
    Ok((sample_rate as f64 * SECONDS_PER_MINUTE) / tempo)
}

/// Calculate the duration of one beat in seconds.
///
/// Returns `Err` if tempo ≤ 0.
pub fn beat_duration(tempo: f64) -> Result<f64, TimingError> {
    if tempo <= 0.0 {
        return Err(TimingError::InvalidTempo(tempo));
    }
    Ok(SECONDS_PER_MINUTE / tempo)
}

/// Calculate the duration of one beat in milliseconds.
///
/// Returns `Err` if tempo ≤ 0.
pub fn beat_duration_ms(tempo: f64) -> Result<f64, TimingError> {
    Ok(beat_duration(tempo)? * 1000.0)
}

/// Convert milliseconds to sample position (rounded).
#[inline]
pub fn ms_to_samples(ms: f64, sample_rate: u32) -> i64 {
    ((ms / 1000.0) * sample_rate as f64).round() as i64
}

/// Convert seconds to sample position (rounded).
#[inline]
pub fn seconds_to_samples(seconds: f64, sample_rate: u32) -> i64 {
    (seconds * sample_rate as f64).round() as i64
}

/// Round a floating-point sample value to the nearest integer.
#[inline]
pub fn round_samples(samples: f64) -> i64 {
    samples.round() as i64
}

/// Floor a floating-point sample value (round down).
#[inline]
pub fn floor_samples(samples: f64) -> i64 {
    samples.floor() as i64
}

// ---------------------------------------------------------------------------
// Loop position
// ---------------------------------------------------------------------------

/// Information about the current position within a loop.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoopInfo {
    /// Zero-based loop index (0 = first loop).
    pub current_loop: u64,
    /// Beat position within the current loop iteration.
    pub position_in_loop: f64,
    /// Total number of loops completed (current_loop + 1).
    pub total_loops: u64,
}

/// Calculate loop position information from a beat position and track length.
///
/// Returns `Err` if `current_beat < 0` or `track_length_in_beats ≤ 0`.
pub fn calculate_loop_position(
    current_beat: f64,
    track_length_in_beats: f64,
) -> Result<LoopInfo, TimingError> {
    if current_beat < 0.0 {
        return Err(TimingError::NegativeValue {
            name: "current beat",
            value: current_beat,
        });
    }
    if track_length_in_beats <= 0.0 {
        return Err(TimingError::InvalidTrackLength(track_length_in_beats));
    }
    let current_loop = (current_beat / track_length_in_beats).floor() as u64;
    let position_in_loop = current_beat % track_length_in_beats;
    Ok(LoopInfo {
        current_loop,
        position_in_loop,
        total_loops: current_loop + 1,
    })
}

// ---------------------------------------------------------------------------
// Latency compensation
// ---------------------------------------------------------------------------

/// Calculate latency-compensated sample position.
///
/// Subtracts the latency (in ms) from the target sample position,
/// clamping to 0 if the result would be negative.
pub fn calculate_latency_compensation(
    target_sample: i64,
    latency_ms: f64,
    sample_rate: u32,
) -> Result<i64, TimingError> {
    if target_sample < 0 {
        return Err(TimingError::NegativeValue {
            name: "target sample",
            value: target_sample as f64,
        });
    }
    if latency_ms < 0.0 {
        return Err(TimingError::NegativeValue {
            name: "latency",
            value: latency_ms,
        });
    }
    if sample_rate == 0 {
        return Err(TimingError::InvalidSampleRate(sample_rate as f64));
    }
    let latency_samples = (latency_ms / 1000.0) * sample_rate as f64;
    Ok((target_sample as f64 - latency_samples).max(0.0) as i64)
}

// ---------------------------------------------------------------------------
// Beat alignment
// ---------------------------------------------------------------------------

/// Calculate the nearest beat-aligned position in samples.
pub fn calculate_beat_aligned_position(
    position_in_samples: i64,
    tempo: f64,
    sample_rate: u32,
) -> Result<i64, TimingError> {
    if position_in_samples < 0 {
        return Err(TimingError::NegativeValue {
            name: "position",
            value: position_in_samples as f64,
        });
    }
    let spb = samples_per_beat(tempo, sample_rate)?;
    let nearest_beat = (position_in_samples as f64 / spb).round();
    Ok((nearest_beat * spb) as i64)
}

// ---------------------------------------------------------------------------
// Position drift
// ---------------------------------------------------------------------------

/// Result of a position-drift calculation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PositionDrift {
    pub drift_in_samples: i64,
    pub drift_in_ms: f64,
    pub drift_in_beats: f64,
}

/// Calculate drift between expected and actual position.
pub fn calculate_position_drift(
    expected_position: i64,
    actual_position: i64,
    tempo: f64,
    sample_rate: u32,
) -> Result<PositionDrift, TimingError> {
    if expected_position < 0 {
        return Err(TimingError::NegativeValue {
            name: "expected position",
            value: expected_position as f64,
        });
    }
    if actual_position < 0 {
        return Err(TimingError::NegativeValue {
            name: "actual position",
            value: actual_position as f64,
        });
    }
    let spb = samples_per_beat(tempo, sample_rate)?;
    let drift_in_samples = actual_position - expected_position;
    let drift_in_ms = (drift_in_samples as f64 / sample_rate as f64) * 1000.0;
    let drift_in_beats = drift_in_samples as f64 / spb;
    Ok(PositionDrift {
        drift_in_samples,
        drift_in_ms,
        drift_in_beats,
    })
}

// ---------------------------------------------------------------------------
// Frame alignment
// ---------------------------------------------------------------------------

/// Align a sample position to the nearest frame boundary.
pub fn calculate_frame_aligned_position(
    samples: i64,
    frame_size: u32,
) -> Result<i64, TimingError> {
    if samples < 0 {
        return Err(TimingError::NegativeValue {
            name: "samples",
            value: samples as f64,
        });
    }
    if frame_size == 0 {
        return Err(TimingError::InvalidFrameSize(frame_size));
    }
    let fs = frame_size as f64;
    Ok(((samples as f64 / fs).round() * fs) as i64)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- time_to_samples / samples_to_time ----------------------------------

    #[test]
    fn time_to_samples_basic() {
        assert_eq!(time_to_samples(1.0, 48_000), 48_000);
    }

    #[test]
    fn time_to_samples_rounds() {
        assert_eq!(time_to_samples(0.5, 48_000), 24_000);
        assert_eq!(time_to_samples(0.00001, 48_000), 0); // 0.48 rounds to 0
        assert_eq!(time_to_samples(0.000011, 48_000), 1); // 0.528 rounds to 1
    }

    #[test]
    fn time_to_samples_zero() {
        assert_eq!(time_to_samples(0.0, 48_000), 0);
    }

    #[test]
    fn time_to_samples_different_rates() {
        assert_eq!(time_to_samples(1.0, 44_100), 44_100);
        assert_eq!(time_to_samples(1.0, 96_000), 96_000);
    }

    #[test]
    fn samples_to_time_basic() {
        assert_eq!(samples_to_time(48_000, 48_000), 1.0);
    }

    #[test]
    fn samples_to_time_decimals() {
        assert_eq!(samples_to_time(24_000, 48_000), 0.5);
        assert_eq!(samples_to_time(12_000, 48_000), 0.25);
    }

    #[test]
    fn samples_to_time_zero() {
        assert_eq!(samples_to_time(0, 48_000), 0.0);
    }

    // -- get_elapsed_samples ------------------------------------------------

    #[test]
    fn elapsed_samples_basic() {
        assert_eq!(get_elapsed_samples(100_000, 50_000), 50_000);
    }

    #[test]
    fn elapsed_samples_zero_start() {
        assert_eq!(get_elapsed_samples(48_000, 0), 48_000);
    }

    #[test]
    fn elapsed_samples_negative() {
        assert_eq!(get_elapsed_samples(10_000, 20_000), -10_000);
    }

    #[test]
    fn elapsed_samples_equal() {
        assert_eq!(get_elapsed_samples(48_000, 48_000), 0);
    }

    // -- beat_to_samples ----------------------------------------------------

    #[test]
    fn beat_to_samples_120bpm() {
        assert_eq!(beat_to_samples(1.0, 120.0, 48_000).unwrap(), 24_000);
        assert_eq!(beat_to_samples(2.0, 120.0, 48_000).unwrap(), 48_000);
        assert_eq!(beat_to_samples(4.0, 120.0, 48_000).unwrap(), 96_000);
    }

    #[test]
    fn beat_to_samples_60bpm() {
        assert_eq!(beat_to_samples(1.0, 60.0, 48_000).unwrap(), 48_000);
        assert_eq!(beat_to_samples(0.5, 60.0, 48_000).unwrap(), 24_000);
    }

    #[test]
    fn beat_to_samples_fractional() {
        assert_eq!(beat_to_samples(1.5, 120.0, 48_000).unwrap(), 36_000);
        assert_eq!(beat_to_samples(0.25, 120.0, 48_000).unwrap(), 6_000);
    }

    #[test]
    fn beat_to_samples_zero_beat() {
        assert_eq!(beat_to_samples(0.0, 120.0, 48_000).unwrap(), 0);
    }

    #[test]
    fn beat_to_samples_invalid_tempo() {
        assert!(beat_to_samples(1.0, 0.0, 48_000).is_err());
        assert!(beat_to_samples(1.0, -120.0, 48_000).is_err());
    }

    #[test]
    fn beat_to_samples_negative_beat() {
        assert!(beat_to_samples(-1.0, 120.0, 48_000).is_err());
    }

    // -- samples_to_beat ----------------------------------------------------

    #[test]
    fn samples_to_beat_120bpm() {
        assert_eq!(samples_to_beat(24_000, 120.0, 48_000).unwrap(), 1.0);
        assert_eq!(samples_to_beat(48_000, 120.0, 48_000).unwrap(), 2.0);
        assert_eq!(samples_to_beat(96_000, 120.0, 48_000).unwrap(), 4.0);
    }

    #[test]
    fn samples_to_beat_60bpm() {
        assert_eq!(samples_to_beat(48_000, 60.0, 48_000).unwrap(), 1.0);
        assert_eq!(samples_to_beat(24_000, 60.0, 48_000).unwrap(), 0.5);
    }

    #[test]
    fn samples_to_beat_fractional() {
        assert_eq!(samples_to_beat(36_000, 120.0, 48_000).unwrap(), 1.5);
        assert_eq!(samples_to_beat(6_000, 120.0, 48_000).unwrap(), 0.25);
    }

    #[test]
    fn samples_to_beat_zero() {
        assert_eq!(samples_to_beat(0, 120.0, 48_000).unwrap(), 0.0);
    }

    #[test]
    fn samples_to_beat_invalid_tempo() {
        assert!(samples_to_beat(48_000, 0.0, 48_000).is_err());
    }

    #[test]
    fn samples_to_beat_negative() {
        assert!(samples_to_beat(-1, 120.0, 48_000).is_err());
    }

    // -- samples_per_beat ---------------------------------------------------

    #[test]
    fn samples_per_beat_basic() {
        assert_eq!(samples_per_beat(120.0, 48_000).unwrap(), 24_000.0);
        assert_eq!(samples_per_beat(60.0, 48_000).unwrap(), 48_000.0);
        assert_eq!(samples_per_beat(240.0, 48_000).unwrap(), 12_000.0);
    }

    #[test]
    fn samples_per_beat_different_rates() {
        assert_eq!(samples_per_beat(120.0, 44_100).unwrap(), 22_050.0);
        assert_eq!(samples_per_beat(120.0, 96_000).unwrap(), 48_000.0);
    }

    #[test]
    fn samples_per_beat_invalid() {
        assert!(samples_per_beat(0.0, 48_000).is_err());
    }

    // -- beat_duration / beat_duration_ms -----------------------------------

    #[test]
    fn beat_duration_basic() {
        assert_eq!(beat_duration(60.0).unwrap(), 1.0);
        assert_eq!(beat_duration(120.0).unwrap(), 0.5);
        assert_eq!(beat_duration(240.0).unwrap(), 0.25);
    }

    #[test]
    fn beat_duration_fractional_tempo() {
        let d90 = beat_duration(90.0).unwrap();
        assert!((d90 - 0.6667).abs() < 0.001);
        let d140 = beat_duration(140.0).unwrap();
        assert!((d140 - 0.4286).abs() < 0.001);
    }

    #[test]
    fn beat_duration_invalid() {
        assert!(beat_duration(0.0).is_err());
    }

    #[test]
    fn beat_duration_ms_basic() {
        assert_eq!(beat_duration_ms(60.0).unwrap(), 1000.0);
        assert_eq!(beat_duration_ms(120.0).unwrap(), 500.0);
        assert_eq!(beat_duration_ms(240.0).unwrap(), 250.0);
    }

    #[test]
    fn beat_duration_ms_fractional() {
        let d90 = beat_duration_ms(90.0).unwrap();
        assert!((d90 - 666.67).abs() < 0.1);
        let d140 = beat_duration_ms(140.0).unwrap();
        assert!((d140 - 428.57).abs() < 0.1);
    }

    // -- ms_to_samples / seconds_to_samples ---------------------------------

    #[test]
    fn ms_to_samples_basic() {
        assert_eq!(ms_to_samples(1000.0, 48_000), 48_000);
        assert_eq!(ms_to_samples(500.0, 48_000), 24_000);
    }

    #[test]
    fn seconds_to_samples_basic() {
        assert_eq!(seconds_to_samples(1.0, 48_000), 48_000);
        assert_eq!(seconds_to_samples(0.5, 48_000), 24_000);
    }

    // -- round_samples / floor_samples --------------------------------------

    #[test]
    fn round_samples_basic() {
        assert_eq!(round_samples(1.4), 1);
        assert_eq!(round_samples(1.5), 2);
        assert_eq!(round_samples(1.6), 2);
    }

    #[test]
    fn floor_samples_basic() {
        assert_eq!(floor_samples(1.9), 1);
        assert_eq!(floor_samples(2.0), 2);
    }

    // -- calculate_loop_position --------------------------------------------

    #[test]
    fn loop_position_first_loop() {
        let info = calculate_loop_position(2.0, 16.0).unwrap();
        assert_eq!(info.current_loop, 0);
        assert_eq!(info.position_in_loop, 2.0);
        assert_eq!(info.total_loops, 1);
    }

    #[test]
    fn loop_position_second_loop() {
        let info = calculate_loop_position(18.0, 16.0).unwrap();
        assert_eq!(info.current_loop, 1);
        assert_eq!(info.position_in_loop, 2.0);
        assert_eq!(info.total_loops, 2);
    }

    #[test]
    fn loop_position_exact_boundary() {
        let info = calculate_loop_position(16.0, 16.0).unwrap();
        assert_eq!(info.current_loop, 1);
        assert_eq!(info.position_in_loop, 0.0);
        assert_eq!(info.total_loops, 2);
    }

    #[test]
    fn loop_position_invalid() {
        assert!(calculate_loop_position(-1.0, 16.0).is_err());
        assert!(calculate_loop_position(5.0, 0.0).is_err());
        assert!(calculate_loop_position(5.0, -1.0).is_err());
    }

    // -- calculate_latency_compensation -------------------------------------

    #[test]
    fn latency_compensation_basic() {
        let result =
            calculate_latency_compensation(48_000, 10.0, 48_000).unwrap();
        // 10ms at 48kHz = 480 samples
        assert_eq!(result, 48_000 - 480);
    }

    #[test]
    fn latency_compensation_clamps_to_zero() {
        let result =
            calculate_latency_compensation(100, 1000.0, 48_000).unwrap();
        assert_eq!(result, 0);
    }

    // -- calculate_beat_aligned_position ------------------------------------

    #[test]
    fn beat_aligned_position_basic() {
        // At 120 BPM, samples_per_beat = 24000
        // Position 25000 → nearest beat = round(25000/24000) = 1 → 24000
        let result =
            calculate_beat_aligned_position(25_000, 120.0, 48_000).unwrap();
        assert_eq!(result, 24_000);
    }

    #[test]
    fn beat_aligned_position_exact() {
        let result =
            calculate_beat_aligned_position(48_000, 120.0, 48_000).unwrap();
        assert_eq!(result, 48_000);
    }

    // -- calculate_position_drift -------------------------------------------

    #[test]
    fn position_drift_basic() {
        let drift =
            calculate_position_drift(48_000, 48_480, 120.0, 48_000).unwrap();
        assert_eq!(drift.drift_in_samples, 480);
        assert!((drift.drift_in_ms - 10.0).abs() < 0.01);
        assert!((drift.drift_in_beats - 0.02).abs() < 0.001);
    }

    // -- calculate_frame_aligned_position -----------------------------------

    #[test]
    fn frame_aligned_basic() {
        assert_eq!(
            calculate_frame_aligned_position(130, 128).unwrap(),
            128
        );
        assert_eq!(
            calculate_frame_aligned_position(256, 128).unwrap(),
            256
        );
    }

    #[test]
    fn frame_aligned_invalid() {
        assert!(calculate_frame_aligned_position(-1, 128).is_err());
        assert!(calculate_frame_aligned_position(100, 0).is_err());
    }
}
