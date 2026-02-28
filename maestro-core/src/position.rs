//! Position calculation and loop boundary management.
//!
//! Ported from:
//! - `src/domains/core/utils/AudioMath.ts` (loop/position math)
//! - `src/domains/state/services/position-sync/position-sync-core.ts` (position sync logic)
//! - `src/domains/state/types/position-sync-types.ts` (types)
//!
//! This module provides sample-accurate position calculations, loop boundary
//! handling, frame alignment, and latency compensation for audio playback.
//! Browser-specific concerns (AudioContext, WebView bridge, caching, performance
//! monitoring) are stripped — those belong in the platform layer, not the core.

use crate::timing::{
    self, TimingError, DEFAULT_SAMPLE_RATE, WORKLET_FRAME_SIZE,
};

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that can occur during position calculations.
#[derive(Debug, Clone, PartialEq)]
pub enum PositionError {
    /// A timing conversion failed.
    Timing(TimingError),
    /// Loop boundaries are invalid (start >= end or negative).
    InvalidLoopBoundaries { start: i64, end: i64 },
    /// Frame size must be > 0.
    InvalidFrameSize(u32),
    /// A required parameter was negative.
    NegativeValue { name: &'static str, value: f64 },
}

impl std::fmt::Display for PositionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PositionError::Timing(e) => write!(f, "Position timing error: {e}"),
            PositionError::InvalidLoopBoundaries { start, end } => {
                write!(f, "Invalid loop boundaries: start={start}, end={end}")
            }
            PositionError::InvalidFrameSize(v) => {
                write!(f, "Invalid frame size: {v}. Must be > 0")
            }
            PositionError::NegativeValue { name, value } => {
                write!(f, "Invalid {name}: {value}. Must be non-negative")
            }
        }
    }
}

impl std::error::Error for PositionError {}

impl From<TimingError> for PositionError {
    fn from(e: TimingError) -> Self {
        PositionError::Timing(e)
    }
}

// ---------------------------------------------------------------------------
// Position data types
// ---------------------------------------------------------------------------

/// Complete synchronization position data for a track.
///
/// Ported from TypeScript `SyncPositionData`.
#[derive(Debug, Clone, PartialEq)]
pub struct SyncPositionData {
    /// Global beat count (1-based integer).
    pub global_beat_count: u32,
    /// Fractional beat position (e.g. 3.7 = beat 4, 70% through).
    pub beat_position: f64,
    /// Fractional part of the beat (0.0–0.999…).
    pub fractional_beat: f64,
    /// Absolute sample position.
    pub sample_position: i64,
    /// Virtual position (may differ from sample_position after compensation).
    pub virtual_position: i64,
    /// Whether this position has been frame-aligned.
    pub frame_aligned: bool,
    /// Number of complete loops.
    pub loop_count: u32,
    /// Position within the current loop (in samples).
    pub position_in_loop: i64,
}

impl SyncPositionData {
    /// Create a zero-position (stopped state).
    pub fn zero() -> Self {
        Self {
            global_beat_count: 0,
            beat_position: 0.0,
            fractional_beat: 0.0,
            sample_position: 0,
            virtual_position: 0,
            frame_aligned: false,
            loop_count: 0,
            position_in_loop: 0,
        }
    }
}

/// Offset data for mid-loop playback starts.
///
/// Ported from TypeScript `OffsetData`.
#[derive(Debug, Clone, PartialEq)]
pub struct OffsetData {
    /// Absolute sample position to start at.
    pub sample_position: i64,
    /// Frame index for the target sample.
    pub frame_index: u32,
    /// Sample offset within the frame.
    pub sample_offset: u32,
    /// Target beat the user clicked.
    pub target_beat: f64,
    /// Current beat at time of calculation.
    pub current_beat: f64,
    /// Difference between target and current beat.
    pub beat_difference: f64,
}

/// Result of a loop boundary crossing check.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoopBoundaryCrossing {
    /// New position after processing.
    pub new_position: i64,
    /// Whether the loop boundary was crossed.
    pub crossed_boundary: bool,
    /// Samples processed before the loop boundary.
    pub samples_before_loop: i64,
    /// Samples processed after wrapping to loop start.
    pub samples_after_loop: i64,
}

/// Latency compensation result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LatencyCompensation {
    pub system_latency_ms: f64,
    pub processing_latency_ms: f64,
    pub total_latency_ms: f64,
    pub total_latency_samples: i64,
    pub system_latency_samples: i64,
    pub processing_latency_samples: i64,
}

/// Frame information for a sample position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameInfo {
    /// Zero-based frame index.
    pub frame_index: u32,
    /// Sample offset within the frame.
    pub sample_offset: u32,
    /// Frame-aligned sample position (start of the frame).
    pub aligned_position: i64,
}

// ---------------------------------------------------------------------------
// Loop boundary functions (from AudioMath.ts)
// ---------------------------------------------------------------------------

/// Validate that loop boundaries are well-formed.
///
/// Returns `true` if `loop_start >= 0` and `loop_end > loop_start`.
#[inline]
pub fn validate_loop_boundaries(loop_start: i64, loop_end: i64) -> bool {
    loop_start >= 0 && loop_end > loop_start
}

/// Normalize a sample position to within loop boundaries.
///
/// Handles positions before loop start (wraps from end) and positions
/// beyond loop end (wraps using modulo).
///
/// Ported from `AudioMath.normalizeToLoop`.
pub fn normalize_to_loop(
    position: i64,
    loop_start: i64,
    loop_end: i64,
) -> Result<i64, PositionError> {
    let loop_length = loop_end - loop_start;
    if loop_length <= 0 {
        return Err(PositionError::InvalidLoopBoundaries {
            start: loop_start,
            end: loop_end,
        });
    }

    if position < loop_start {
        // Wrap from the end of the loop
        let negative_offset = loop_start - position;
        let wrapped_offset = negative_offset % loop_length;
        Ok(loop_end - wrapped_offset)
    } else if position >= loop_end {
        // Wrap using modulo
        let relative = position - loop_start;
        Ok(loop_start + (relative % loop_length))
    } else {
        Ok(position)
    }
}

/// Calculate how many complete loops have elapsed.
///
/// Ported from `AudioMath.calculateLoopCount`.
#[inline]
pub fn calculate_loop_count(position: i64, loop_length: i64) -> u32 {
    if loop_length <= 0 || position < 0 {
        return 0;
    }
    (position / loop_length) as u32
}

/// Calculate position within the current loop iteration.
///
/// Ported from `AudioMath.calculatePositionInLoop`.
#[inline]
pub fn calculate_position_in_loop(position: i64, loop_length: i64) -> i64 {
    if loop_length <= 0 || position < 0 {
        return 0;
    }
    position % loop_length
}

/// Check if frame processing crosses a loop boundary and compute new position.
///
/// Ported from `AudioMath.handleLoopBoundaryCrossing`.
pub fn handle_loop_boundary_crossing(
    current_position: i64,
    frames_to_process: i64,
    loop_start: i64,
    loop_end: i64,
) -> Result<LoopBoundaryCrossing, PositionError> {
    if !validate_loop_boundaries(loop_start, loop_end) {
        return Err(PositionError::InvalidLoopBoundaries {
            start: loop_start,
            end: loop_end,
        });
    }

    let remaining_samples = loop_end - current_position;

    if frames_to_process > remaining_samples {
        let samples_before_loop = remaining_samples;
        let samples_after_loop = frames_to_process - remaining_samples;
        let new_position = loop_start + samples_after_loop;

        Ok(LoopBoundaryCrossing {
            new_position,
            crossed_boundary: true,
            samples_before_loop,
            samples_after_loop,
        })
    } else {
        Ok(LoopBoundaryCrossing {
            new_position: current_position + frames_to_process,
            crossed_boundary: false,
            samples_before_loop: frames_to_process,
            samples_after_loop: 0,
        })
    }
}

/// Apply tolerance-based position correction for edge cases.
///
/// Positions very close to loop start or end snap to loop start.
/// Ported from `AudioMath.correctPositionEdgeCases`.
pub fn correct_position_edge_cases(
    position: i64,
    loop_start: i64,
    loop_end: i64,
    tolerance: i64,
) -> i64 {
    if (position - loop_end).abs() < tolerance {
        return loop_start;
    }
    if (position - loop_start).abs() < tolerance {
        return loop_start;
    }
    position
}

// ---------------------------------------------------------------------------
// Frame alignment functions (from AudioMath.ts)
// ---------------------------------------------------------------------------

/// Align a sample position to a frame boundary (rounds down).
///
/// Ported from `AudioMath.alignToFrameBoundary`.
#[inline]
pub fn align_to_frame_boundary(position: i64, frame_size: u32) -> Result<i64, PositionError> {
    if frame_size == 0 {
        return Err(PositionError::InvalidFrameSize(frame_size));
    }
    let fs = frame_size as i64;
    Ok((position / fs) * fs)
}

/// Check if a position is aligned to a frame boundary.
///
/// Ported from `AudioMath.isFrameAligned`.
#[inline]
pub fn is_frame_aligned(position: i64, frame_size: u32) -> bool {
    if frame_size == 0 {
        return false;
    }
    position % frame_size as i64 == 0
}

/// Calculate frame information for a sample position.
///
/// Ported from `AudioMath.calculateFrameInfo`.
pub fn calculate_frame_info(
    sample_position: i64,
    frame_size: u32,
) -> Result<FrameInfo, PositionError> {
    if frame_size == 0 {
        return Err(PositionError::InvalidFrameSize(frame_size));
    }
    let fs = frame_size as i64;
    let frame_index = (sample_position / fs) as u32;
    let sample_offset = (sample_position % fs) as u32;
    let aligned_position = (sample_position / fs) * fs;

    Ok(FrameInfo {
        frame_index,
        sample_offset,
        aligned_position,
    })
}

/// Convert a sample position to a frame index.
///
/// Ported from `AudioMath.sampleToFrameIndex`.
#[inline]
pub fn sample_to_frame_index(sample_position: i64, frame_size: u32) -> u32 {
    if frame_size == 0 {
        return 0;
    }
    (sample_position / frame_size as i64) as u32
}

/// Calculate the sample offset within a frame.
///
/// Ported from `AudioMath.sampleOffsetInFrame`.
#[inline]
pub fn sample_offset_in_frame(sample_position: i64, frame_size: u32) -> u32 {
    if frame_size == 0 {
        return 0;
    }
    (sample_position % frame_size as i64) as u32
}

// ---------------------------------------------------------------------------
// Beat ↔ sample position conversions (from AudioMath.ts)
// ---------------------------------------------------------------------------

/// Convert a beat position to a sample position using the given BPM.
/// Floors the result for exact sample boundaries.
///
/// Ported from `AudioMath.beatToSamplePosition`.
#[inline]
pub fn beat_to_sample_position(beat_position: f64, bpm: f64) -> Result<i64, PositionError> {
    let spb = timing::samples_per_beat(bpm, DEFAULT_SAMPLE_RATE)?;
    Ok((beat_position * spb).floor() as i64)
}

/// Convert a beat position to a precise (non-floored) sample position.
/// Useful for intermediate calculations where precision matters.
///
/// Ported from `AudioMath.beatToSamplePositionPrecise`.
#[inline]
pub fn beat_to_sample_position_precise(beat_position: f64, bpm: f64) -> Result<f64, PositionError> {
    let spb = timing::samples_per_beat(bpm, DEFAULT_SAMPLE_RATE)?;
    Ok(beat_position * spb)
}

/// Quantize a precise sample position to the nearest integer sample.
///
/// Ported from `AudioMath.quantizeToSamples`.
#[inline]
pub fn quantize_to_samples(precise_position: f64) -> i64 {
    precise_position.round() as i64
}

/// Convert a sample position to a beat position.
///
/// Ported from `AudioMath.sampleToBeatPosition`.
#[inline]
pub fn sample_to_beat_position(sample_position: i64, bpm: f64) -> Result<f64, PositionError> {
    let spb = timing::samples_per_beat(bpm, DEFAULT_SAMPLE_RATE)?;
    Ok(sample_position as f64 / spb)
}

/// Convert a sample position to a beat position with a configurable sample rate.
#[inline]
pub fn sample_to_beat_position_with_rate(
    sample_position: i64,
    bpm: f64,
    sample_rate: u32,
) -> Result<f64, PositionError> {
    let spb = timing::samples_per_beat(bpm, sample_rate)?;
    Ok(sample_position as f64 / spb)
}

// ---------------------------------------------------------------------------
// Latency compensation (from AudioMath.ts)
// ---------------------------------------------------------------------------

/// Calculate total latency compensation breakdown.
///
/// Ported from `AudioMath.calculateTotalLatencyCompensation`.
pub fn calculate_total_latency_compensation(
    system_latency_ms: f64,
    processing_latency_ms: f64,
) -> LatencyCompensation {
    let sample_rate = DEFAULT_SAMPLE_RATE;
    let system_samples = timing::ms_to_samples(system_latency_ms, sample_rate);
    let processing_samples = timing::ms_to_samples(processing_latency_ms, sample_rate);
    let total_ms = system_latency_ms + processing_latency_ms;
    let total_samples = system_samples + processing_samples;

    LatencyCompensation {
        system_latency_ms,
        processing_latency_ms,
        total_latency_ms: total_ms,
        total_latency_samples: total_samples,
        system_latency_samples: system_samples,
        processing_latency_samples: processing_samples,
    }
}

// ---------------------------------------------------------------------------
// Frame availability (from AudioMath.ts)
// ---------------------------------------------------------------------------

/// Check if a target frame is available for processing.
///
/// Ported from `AudioMath.checkFrameAvailability`.
#[inline]
pub fn check_frame_availability(
    target_frame_index: u32,
    current_global_frame: u32,
    frame_distance_threshold: u32,
) -> bool {
    let frame_difference = target_frame_index as i64 - current_global_frame as i64;
    frame_difference <= frame_distance_threshold as i64
}

// ---------------------------------------------------------------------------
// Position consistency validation (from AudioMath.ts)
// ---------------------------------------------------------------------------

/// Validate that beat and sample positions are consistent.
///
/// Returns `true` if converting `sample_position` to beats yields a value
/// within `tolerance` of `beat_position`.
///
/// Ported from `AudioMath.validatePositionConsistency`.
pub fn validate_position_consistency(
    beat_position: f64,
    sample_position: i64,
    bpm: f64,
    tolerance: f64,
) -> Result<bool, PositionError> {
    let expected_beat = sample_to_beat_position(sample_position, bpm)?;
    Ok((beat_position - expected_beat).abs() <= tolerance)
}

// ---------------------------------------------------------------------------
// Offset calculation
// ---------------------------------------------------------------------------

/// Calculate offset data for a mid-loop playback start.
///
/// Given a target beat and current beat position, calculates the exact
/// sample position, frame index, and offsets needed to start playback
/// at the target beat.
pub fn calculate_beat_offset(
    target_beat: f64,
    current_beat: f64,
    bpm: f64,
    frame_size: u32,
) -> Result<OffsetData, PositionError> {
    let sample_position = beat_to_sample_position(target_beat, bpm)?;
    let frame_info = calculate_frame_info(sample_position, frame_size)?;

    Ok(OffsetData {
        sample_position,
        frame_index: frame_info.frame_index,
        sample_offset: frame_info.sample_offset,
        target_beat,
        current_beat,
        beat_difference: target_beat - current_beat,
    })
}

/// Calculate offset data using the default worklet frame size (128 samples).
pub fn calculate_beat_offset_default(
    target_beat: f64,
    current_beat: f64,
    bpm: f64,
) -> Result<OffsetData, PositionError> {
    calculate_beat_offset(target_beat, current_beat, bpm, WORKLET_FRAME_SIZE)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- validate_loop_boundaries -------------------------------------------

    #[test]
    fn valid_loop_boundaries() {
        assert!(validate_loop_boundaries(0, 100));
        assert!(validate_loop_boundaries(1000, 2000));
    }

    #[test]
    fn invalid_loop_boundaries() {
        assert!(!validate_loop_boundaries(100, 100)); // equal
        assert!(!validate_loop_boundaries(200, 100)); // start > end
        assert!(!validate_loop_boundaries(-1, 100));  // negative start
    }

    // -- normalize_to_loop --------------------------------------------------

    #[test]
    fn normalize_within_loop() {
        assert_eq!(normalize_to_loop(50, 0, 100).unwrap(), 50);
        assert_eq!(normalize_to_loop(0, 0, 100).unwrap(), 0);
        assert_eq!(normalize_to_loop(99, 0, 100).unwrap(), 99);
    }

    #[test]
    fn normalize_beyond_loop_end() {
        assert_eq!(normalize_to_loop(100, 0, 100).unwrap(), 0);
        assert_eq!(normalize_to_loop(150, 0, 100).unwrap(), 50);
        assert_eq!(normalize_to_loop(250, 0, 100).unwrap(), 50);
    }

    #[test]
    fn normalize_before_loop_start() {
        // position -10, loop [0, 100) → wrap from end: 100 - 10 = 90
        assert_eq!(normalize_to_loop(-10, 0, 100).unwrap(), 90);
    }

    #[test]
    fn normalize_with_offset_loop() {
        // Loop [1000, 2000)
        assert_eq!(normalize_to_loop(1500, 1000, 2000).unwrap(), 1500);
        assert_eq!(normalize_to_loop(2500, 1000, 2000).unwrap(), 1500);
        assert_eq!(normalize_to_loop(500, 1000, 2000).unwrap(), 1500);
    }

    #[test]
    fn normalize_invalid_boundaries() {
        assert!(normalize_to_loop(50, 100, 100).is_err());
        assert!(normalize_to_loop(50, 200, 100).is_err());
    }

    // -- calculate_loop_count -----------------------------------------------

    #[test]
    fn loop_count_basic() {
        assert_eq!(calculate_loop_count(0, 100), 0);
        assert_eq!(calculate_loop_count(99, 100), 0);
        assert_eq!(calculate_loop_count(100, 100), 1);
        assert_eq!(calculate_loop_count(250, 100), 2);
    }

    #[test]
    fn loop_count_edge_cases() {
        assert_eq!(calculate_loop_count(-1, 100), 0);
        assert_eq!(calculate_loop_count(50, 0), 0);
        assert_eq!(calculate_loop_count(50, -10), 0);
    }

    // -- calculate_position_in_loop -----------------------------------------

    #[test]
    fn position_in_loop_basic() {
        assert_eq!(calculate_position_in_loop(0, 100), 0);
        assert_eq!(calculate_position_in_loop(50, 100), 50);
        assert_eq!(calculate_position_in_loop(150, 100), 50);
        assert_eq!(calculate_position_in_loop(200, 100), 0);
    }

    #[test]
    fn position_in_loop_edge_cases() {
        assert_eq!(calculate_position_in_loop(-1, 100), 0);
        assert_eq!(calculate_position_in_loop(50, 0), 0);
    }

    // -- handle_loop_boundary_crossing --------------------------------------

    #[test]
    fn no_boundary_crossing() {
        let result = handle_loop_boundary_crossing(50, 30, 0, 100).unwrap();
        assert!(!result.crossed_boundary);
        assert_eq!(result.new_position, 80);
        assert_eq!(result.samples_before_loop, 30);
        assert_eq!(result.samples_after_loop, 0);
    }

    #[test]
    fn boundary_crossing() {
        let result = handle_loop_boundary_crossing(80, 40, 0, 100).unwrap();
        assert!(result.crossed_boundary);
        assert_eq!(result.new_position, 20); // 0 + (40 - 20)
        assert_eq!(result.samples_before_loop, 20);
        assert_eq!(result.samples_after_loop, 20);
    }

    #[test]
    fn exact_loop_end() {
        let result = handle_loop_boundary_crossing(80, 20, 0, 100).unwrap();
        assert!(!result.crossed_boundary);
        assert_eq!(result.new_position, 100);
    }

    #[test]
    fn crossing_invalid_boundaries() {
        assert!(handle_loop_boundary_crossing(50, 30, 100, 50).is_err());
    }

    // -- correct_position_edge_cases ----------------------------------------

    #[test]
    fn edge_case_near_loop_end() {
        assert_eq!(correct_position_edge_cases(99, 0, 100, 2), 0);
    }

    #[test]
    fn edge_case_near_loop_start() {
        assert_eq!(correct_position_edge_cases(1, 0, 100, 2), 0);
    }

    #[test]
    fn edge_case_not_near_boundary() {
        assert_eq!(correct_position_edge_cases(50, 0, 100, 2), 50);
    }

    // -- frame alignment ----------------------------------------------------

    #[test]
    fn align_to_frame_basic() {
        assert_eq!(align_to_frame_boundary(130, 128).unwrap(), 128);
        assert_eq!(align_to_frame_boundary(256, 128).unwrap(), 256);
        assert_eq!(align_to_frame_boundary(0, 128).unwrap(), 0);
        assert_eq!(align_to_frame_boundary(127, 128).unwrap(), 0);
    }

    #[test]
    fn align_to_frame_invalid() {
        assert!(align_to_frame_boundary(100, 0).is_err());
    }

    #[test]
    fn is_frame_aligned_basic() {
        assert!(is_frame_aligned(0, 128));
        assert!(is_frame_aligned(128, 128));
        assert!(is_frame_aligned(256, 128));
        assert!(!is_frame_aligned(1, 128));
        assert!(!is_frame_aligned(129, 128));
    }

    #[test]
    fn is_frame_aligned_zero_size() {
        assert!(!is_frame_aligned(0, 0));
    }

    // -- calculate_frame_info -----------------------------------------------

    #[test]
    fn frame_info_basic() {
        let info = calculate_frame_info(130, 128).unwrap();
        assert_eq!(info.frame_index, 1);
        assert_eq!(info.sample_offset, 2);
        assert_eq!(info.aligned_position, 128);
    }

    #[test]
    fn frame_info_aligned() {
        let info = calculate_frame_info(256, 128).unwrap();
        assert_eq!(info.frame_index, 2);
        assert_eq!(info.sample_offset, 0);
        assert_eq!(info.aligned_position, 256);
    }

    #[test]
    fn frame_info_zero() {
        let info = calculate_frame_info(0, 128).unwrap();
        assert_eq!(info.frame_index, 0);
        assert_eq!(info.sample_offset, 0);
        assert_eq!(info.aligned_position, 0);
    }

    #[test]
    fn frame_info_invalid() {
        assert!(calculate_frame_info(100, 0).is_err());
    }

    // -- sample_to_frame_index / sample_offset_in_frame ---------------------

    #[test]
    fn sample_to_frame_index_basic() {
        assert_eq!(sample_to_frame_index(0, 128), 0);
        assert_eq!(sample_to_frame_index(127, 128), 0);
        assert_eq!(sample_to_frame_index(128, 128), 1);
        assert_eq!(sample_to_frame_index(256, 128), 2);
    }

    #[test]
    fn sample_offset_basic() {
        assert_eq!(sample_offset_in_frame(0, 128), 0);
        assert_eq!(sample_offset_in_frame(1, 128), 1);
        assert_eq!(sample_offset_in_frame(130, 128), 2);
    }

    // -- beat_to_sample_position / sample_to_beat_position ------------------

    #[test]
    fn beat_to_sample_120bpm() {
        // At 120 BPM, 48kHz: 1 beat = 24000 samples
        assert_eq!(beat_to_sample_position(1.0, 120.0).unwrap(), 24_000);
        assert_eq!(beat_to_sample_position(4.0, 120.0).unwrap(), 96_000);
        assert_eq!(beat_to_sample_position(0.0, 120.0).unwrap(), 0);
    }

    #[test]
    fn beat_to_sample_fractional() {
        assert_eq!(beat_to_sample_position(0.5, 120.0).unwrap(), 12_000);
        assert_eq!(beat_to_sample_position(1.5, 120.0).unwrap(), 36_000);
    }

    #[test]
    fn beat_to_sample_precise() {
        let precise = beat_to_sample_position_precise(1.0, 120.0).unwrap();
        assert!((precise - 24_000.0).abs() < 0.001);
    }

    #[test]
    fn sample_to_beat_120bpm() {
        let beat = sample_to_beat_position(24_000, 120.0).unwrap();
        assert!((beat - 1.0).abs() < 1e-10);

        let beat4 = sample_to_beat_position(96_000, 120.0).unwrap();
        assert!((beat4 - 4.0).abs() < 1e-10);
    }

    #[test]
    fn quantize_to_samples_basic() {
        assert_eq!(quantize_to_samples(24000.4), 24000);
        assert_eq!(quantize_to_samples(24000.5), 24001);
        assert_eq!(quantize_to_samples(24000.6), 24001);
    }

    // -- latency compensation -----------------------------------------------

    #[test]
    fn latency_compensation_basic() {
        let comp = calculate_total_latency_compensation(10.0, 5.0);
        assert!((comp.total_latency_ms - 15.0).abs() < 0.001);
        // 10ms at 48kHz = 480 samples, 5ms = 240 samples
        assert_eq!(comp.system_latency_samples, 480);
        assert_eq!(comp.processing_latency_samples, 240);
        assert_eq!(comp.total_latency_samples, 720);
    }

    #[test]
    fn latency_compensation_zero() {
        let comp = calculate_total_latency_compensation(0.0, 0.0);
        assert_eq!(comp.total_latency_samples, 0);
        assert!((comp.total_latency_ms).abs() < 0.001);
    }

    // -- frame availability -------------------------------------------------

    #[test]
    fn frame_available_within_threshold() {
        assert!(check_frame_availability(5, 0, 10));
        assert!(check_frame_availability(10, 0, 10));
    }

    #[test]
    fn frame_not_available_beyond_threshold() {
        assert!(!check_frame_availability(15, 0, 10));
    }

    #[test]
    fn frame_available_past_frames() {
        // Target is behind current — always available
        assert!(check_frame_availability(0, 5, 10));
    }

    // -- position consistency -----------------------------------------------

    #[test]
    fn position_consistent() {
        // At 120 BPM: beat 1.0 = 24000 samples
        assert!(validate_position_consistency(1.0, 24_000, 120.0, 0.001).unwrap());
    }

    #[test]
    fn position_inconsistent() {
        // 30000 samples at 120 BPM = 1.25 beats, but we claim 2.0
        assert!(!validate_position_consistency(2.0, 30_000, 120.0, 0.001).unwrap());
    }

    // -- calculate_beat_offset ----------------------------------------------

    #[test]
    fn beat_offset_basic() {
        let offset = calculate_beat_offset(4.0, 1.0, 120.0, 128).unwrap();
        assert_eq!(offset.sample_position, 96_000);
        assert!((offset.beat_difference - 3.0).abs() < 1e-10);
        assert_eq!(offset.frame_index, 750); // 96000 / 128
        assert_eq!(offset.sample_offset, 0);
    }

    #[test]
    fn beat_offset_fractional() {
        let offset = calculate_beat_offset(2.5, 1.0, 120.0, 128).unwrap();
        assert_eq!(offset.sample_position, 60_000);
        assert!((offset.beat_difference - 1.5).abs() < 1e-10);
    }

    #[test]
    fn beat_offset_default_frame_size() {
        let offset = calculate_beat_offset_default(1.0, 0.0, 120.0).unwrap();
        assert_eq!(offset.sample_position, 24_000);
        assert_eq!(offset.frame_index, 187); // 24000 / 128 = 187.5 → floor = 187
        assert_eq!(offset.sample_offset, 64); // 24000 % 128 = 64
    }

    // -- SyncPositionData::zero ---------------------------------------------

    #[test]
    fn sync_position_zero() {
        let pos = SyncPositionData::zero();
        assert_eq!(pos.global_beat_count, 0);
        assert_eq!(pos.beat_position, 0.0);
        assert_eq!(pos.sample_position, 0);
        assert_eq!(pos.loop_count, 0);
    }
}
