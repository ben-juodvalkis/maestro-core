//! Beat tracking service — beat, bar, and loop detection with event emission.
//!
//! Ported from:
//! - `src/domains/core/services/BeatTrackingService.ts`
//! - `src/domains/core/algorithms/BeatDetector.ts`
//! - `src/domains/core/services/BeatStateManager.ts`
//! - `src/domains/infrastructure/config/BeatTrackingConfig.ts`
//!
//! The TypeScript codebase splits this across four classes:
//! - `BeatTrackingService` — orchestration and public API
//! - `BeatDetector` — stateless beat info calculation from position data
//! - `BeatStateManager` — change detection and event emission
//! - `BeatTrackingConfig` — configuration types
//!
//! In Rust we consolidate into two structs:
//! - `BeatDetector` — pure math: converts a fractional beat position into
//!   beat/bar/loop information (stateless, no allocations)
//! - `BeatTracker` — owns a `BeatDetector`, tracks state changes, and
//!   produces events when beat/bar/loop boundaries are crossed
//!
//! Browser-specific concerns (AudioWorkletNode, AudioContext, MessagePort,
//! metronome worklet, DOM event emitter) are stripped. The Rust core uses a
//! simple callback/event model; the platform layer (Capacitor plugin or WASM
//! bridge) wires events to the appropriate transport.

use crate::position;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Default beats per bar (4/4 time).
pub const DEFAULT_BEATS_PER_BAR: u32 = 4;

/// Default bars per loop.
pub const DEFAULT_BARS_PER_LOOP: u32 = 4;

/// Threshold for "near beat boundary" detection.
pub const NEAR_BOUNDARY_THRESHOLD: f64 = 0.05;

/// Musical subdivision values for beat alignment validation.
pub const MUSICAL_SUBDIVISIONS: &[f64] = &[1.0, 0.5, 0.25];

/// Alignment tolerance for beat position validation.
pub const ALIGNMENT_TOLERANCE: f64 = 0.01;

/// Configuration for beat tracking.
#[derive(Debug, Clone)]
pub struct BeatTrackingConfig {
    pub beats_per_bar: u32,
    pub bars_per_loop: u32,
}

impl Default for BeatTrackingConfig {
    fn default() -> Self {
        Self {
            beats_per_bar: DEFAULT_BEATS_PER_BAR,
            bars_per_loop: DEFAULT_BARS_PER_LOOP,
        }
    }
}

impl BeatTrackingConfig {
    /// Total beats in one complete loop.
    #[inline]
    pub fn beats_per_loop(&self) -> u32 {
        self.beats_per_bar * self.bars_per_loop
    }
}

// ---------------------------------------------------------------------------
// Beat info types
// ---------------------------------------------------------------------------

/// Complete beat information at a point in time.
///
/// Ported from TypeScript `BeatInfo`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BeatInfo {
    /// Current beat within the bar (1-based, e.g. 1–4 in 4/4).
    pub beat: u32,
    /// Current bar within the loop (1-based).
    pub bar: u32,
    /// Total beats elapsed since start (1-based).
    pub absolute_beat: u32,
    /// Fractional part of the beat (0.0–0.999…).
    pub beat_fraction: f64,
    /// Full fractional beat position (e.g. beat 3 with 0.7 fraction = 2.7).
    pub precise_beat: f64,
}

/// Bar change information.
///
/// Ported from TypeScript `BarInfo`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BarInfo {
    /// Current bar (1-based).
    pub bar: u32,
    /// Current beat within the bar (1-based).
    pub beat: u32,
}

/// Loop completion information.
///
/// Ported from TypeScript `LoopInfo`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoopInfo {
    /// Number of complete loops (0 = still in first loop).
    pub loop_count: u32,
}

/// Complete beat status snapshot.
///
/// Ported from TypeScript `BeatStatus`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BeatStatus {
    /// Current beat within bar (1-based).
    pub beat: u32,
    /// Current bar within loop (1-based).
    pub bar: u32,
    /// Total beats elapsed (1-based).
    pub absolute_beat: u32,
    /// Fractional part of beat.
    pub beat_fraction: f64,
    /// Number of complete loops.
    pub loop_count: u32,
    /// Whether tracking is active.
    pub is_playing: bool,
}

/// Beat alignment information for validation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BeatAlignment {
    /// The beat info.
    pub info: BeatInfo,
    /// Whether the position is near a beat boundary.
    pub is_near_beat_boundary: bool,
    /// How close to a perfect beat boundary (0.0 = perfect, 0.5 = worst).
    pub alignment_accuracy: f64,
}

/// Result of beat position validation.
#[derive(Debug, Clone, PartialEq)]
pub struct BeatValidation {
    /// Whether the beat position is valid.
    pub is_valid: bool,
    /// Nearest musically-aligned beat.
    pub aligned_beat: f64,
    /// Whether the position falls on a musical subdivision.
    pub musically_correct: bool,
    /// Suggestion for correction (if any).
    pub suggestion: Option<String>,
}

/// Change detection flags from a state update.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BeatChangeDetection {
    pub beat_changed: bool,
    pub bar_changed: bool,
    pub loop_changed: bool,
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

/// Events emitted by the beat tracker.
#[derive(Debug, Clone, PartialEq)]
pub enum BeatEvent {
    /// A new beat was reached.
    Beat(BeatInfo),
    /// A new bar was reached.
    Bar(BarInfo),
    /// A complete loop was finished.
    Loop(LoopInfo),
    /// Tracking started.
    Play,
    /// Tracking stopped.
    Stop,
    /// Position was seeked.
    Seek { time: f64 },
}

// ---------------------------------------------------------------------------
// BeatDetector — pure stateless calculations
// ---------------------------------------------------------------------------

/// Stateless beat position calculator.
///
/// Converts a fractional beat position (from `BeatTimeline`) into musical
/// abstractions (beat in bar, bar in loop, loop count). No internal state,
/// no allocations, no side effects.
///
/// Ported from TypeScript `BeatDetector`.
#[derive(Debug, Clone)]
pub struct BeatDetector {
    config: BeatTrackingConfig,
}

impl BeatDetector {
    /// Create a new detector with the given configuration.
    pub fn new(config: BeatTrackingConfig) -> Self {
        Self { config }
    }

    /// Create with default 4/4 time, 4 bars per loop.
    pub fn with_defaults() -> Self {
        Self::new(BeatTrackingConfig::default())
    }

    /// Calculate beat information from a fractional beat position.
    ///
    /// The `exact_beat` parameter is the global fractional beat count
    /// (e.g. 5.7 means we're 70% through the 6th beat).
    ///
    /// This is the core algorithm ported from `BeatDetector.getCurrentBeatInfo()`.
    pub fn calculate_beat_info(&self, exact_beat: f64) -> BeatInfo {
        let absolute_beat = exact_beat.floor().max(0.0) as u32;
        let beat_fraction = if exact_beat > 0.0 {
            exact_beat.fract()
        } else {
            0.0
        };

        // Convert to 0-based for calculations
        let beats_completed = absolute_beat.saturating_sub(1);
        let beat_in_bar = beats_completed % self.config.beats_per_bar;
        let current_beat = beat_in_bar + 1; // back to 1-based

        // Calculate current bar (1-based)
        let beats_per_loop = self.config.beats_per_loop();
        let beats_in_current_loop = if beats_per_loop > 0 {
            beats_completed % beats_per_loop
        } else {
            0
        };
        let current_bar = if self.config.beats_per_bar > 0 {
            beats_in_current_loop / self.config.beats_per_bar + 1
        } else {
            1
        };

        // Precise beat: (absoluteBeat - 1) + beatFraction
        let precise_beat = if absolute_beat > 0 {
            (absolute_beat - 1) as f64 + beat_fraction
        } else {
            0.0
        };

        BeatInfo {
            beat: current_beat,
            bar: current_bar,
            absolute_beat: absolute_beat.max(1), // Never return 0 for absolute beat
            beat_fraction,
            precise_beat,
        }
    }

    /// Calculate the number of complete loops from a fractional beat position.
    ///
    /// Ported from `BeatDetector.getLoopCount()`.
    pub fn calculate_loop_count(&self, exact_beat: f64) -> u32 {
        let absolute_beat = exact_beat.floor().max(0.0) as u32;
        let beats_per_loop = self.config.beats_per_loop();
        if beats_per_loop == 0 || absolute_beat == 0 {
            return 0;
        }
        (absolute_beat - 1) / beats_per_loop
    }

    /// Get beat info with alignment accuracy data.
    pub fn calculate_beat_alignment(&self, exact_beat: f64) -> BeatAlignment {
        let info = self.calculate_beat_info(exact_beat);
        let distance_to_boundary = info.beat_fraction.min(1.0 - info.beat_fraction);
        let is_near_boundary = distance_to_boundary < NEAR_BOUNDARY_THRESHOLD;

        BeatAlignment {
            info,
            is_near_beat_boundary: is_near_boundary,
            alignment_accuracy: distance_to_boundary,
        }
    }

    /// Validate a target beat position for musical correctness.
    ///
    /// Ported from `TimingCalculator.validateBeatPosition`.
    pub fn validate_beat_position(&self, target_beat: f64) -> BeatValidation {
        if target_beat < 0.0 {
            return BeatValidation {
                is_valid: false,
                aligned_beat: 0.0,
                musically_correct: false,
                suggestion: Some("Beat position must be non-negative".into()),
            };
        }

        let aligned_beat = target_beat.round();
        let is_on_subdivision = MUSICAL_SUBDIVISIONS.iter().any(|&sub| {
            let remainder = target_beat % sub;
            remainder.abs() < ALIGNMENT_TOLERANCE || (sub - remainder).abs() < ALIGNMENT_TOLERANCE
        });

        BeatValidation {
            is_valid: target_beat >= 0.0,
            aligned_beat,
            musically_correct: is_on_subdivision,
            suggestion: if !is_on_subdivision {
                Some(format!(
                    "Consider aligning to beat {:.1} for musical accuracy",
                    aligned_beat
                ))
            } else {
                None
            },
        }
    }

    /// Optimize a rough beat position to the nearest musical subdivision.
    ///
    /// Ported from `TimingCalculator.optimizeBeatPosition`.
    pub fn optimize_beat_position(&self, rough_beat: f64) -> (f64, f64, &'static str) {
        let mut best_beat = rough_beat.round();
        let mut best_distance = (rough_beat - best_beat).abs();
        let mut reasoning = "Rounded to nearest integer beat";

        for &sub in MUSICAL_SUBDIVISIONS {
            let quantized = (rough_beat / sub).round() * sub;
            let distance = (rough_beat - quantized).abs();
            if distance < best_distance {
                best_beat = quantized;
                best_distance = distance;
                reasoning = if (sub - 1.0).abs() < f64::EPSILON {
                    "Aligned to whole beat"
                } else if (sub - 0.5).abs() < f64::EPSILON {
                    "Aligned to half beat"
                } else {
                    "Aligned to quarter beat"
                };
            }
        }

        let confidence = 1.0 - (best_distance / 0.5).min(1.0);
        (best_beat, confidence, reasoning)
    }

    /// Get current configuration.
    pub fn config(&self) -> &BeatTrackingConfig {
        &self.config
    }

    /// Update configuration.
    pub fn set_config(&mut self, config: BeatTrackingConfig) {
        self.config = config;
    }

    /// Calculate beat offset data for a mid-loop playback start.
    ///
    /// Delegates to `position::calculate_beat_offset_default`.
    pub fn calculate_beat_offset(
        &self,
        target_beat: f64,
        current_beat: f64,
        bpm: f64,
    ) -> Result<position::OffsetData, position::PositionError> {
        position::calculate_beat_offset_default(target_beat, current_beat, bpm)
    }
}

// ---------------------------------------------------------------------------
// BeatTracker — stateful tracker with event emission
// ---------------------------------------------------------------------------

/// Stateful beat tracker with change detection and event collection.
///
/// Wraps a `BeatDetector` and tracks previous state to detect beat, bar,
/// and loop boundary crossings. Events are collected into a `Vec` rather
/// than using callbacks, following the Rust "collect then dispatch" pattern.
/// The caller (engine or platform layer) is responsible for dispatching events.
///
/// Ported from TypeScript `BeatTrackingService` + `BeatStateManager`.
#[derive(Debug)]
pub struct BeatTracker {
    detector: BeatDetector,
    tracking: bool,

    // Previous state for change detection
    previous_beat: u32,
    previous_bar: u32,
    previous_loop: u32,
}

impl BeatTracker {
    /// Create a new tracker with the given configuration.
    pub fn new(config: BeatTrackingConfig) -> Self {
        Self {
            detector: BeatDetector::new(config),
            tracking: false,
            previous_beat: 1,
            previous_bar: 1,
            previous_loop: 0,
        }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(BeatTrackingConfig::default())
    }

    // -----------------------------------------------------------------------
    // Lifecycle
    // -----------------------------------------------------------------------

    /// Start tracking. Returns a `Play` event.
    ///
    /// If already tracking, returns `None`.
    pub fn start(&mut self) -> Option<BeatEvent> {
        if self.tracking {
            return None;
        }
        self.tracking = true;
        Some(BeatEvent::Play)
    }

    /// Stop tracking and reset state. Returns a `Stop` event.
    ///
    /// If not tracking, still resets and returns the event.
    pub fn stop(&mut self) -> BeatEvent {
        self.tracking = false;
        self.previous_beat = 1;
        self.previous_bar = 1;
        self.previous_loop = 0;
        BeatEvent::Stop
    }

    /// Handle a seek operation. Returns a `Seek` event and any
    /// beat/bar/loop change events from the new position.
    pub fn seek(&mut self, time: f64, exact_beat: f64) -> Vec<BeatEvent> {
        let mut events = Vec::new();

        let info = self.detector.calculate_beat_info(exact_beat);
        let loop_count = self.detector.calculate_loop_count(exact_beat);

        // Update previous state
        self.previous_beat = info.beat;
        self.previous_bar = info.bar;
        self.previous_loop = loop_count;

        events.push(BeatEvent::Seek { time });

        // Force-emit current position events after seek
        events.push(BeatEvent::Beat(info));
        if info.beat == 1 {
            events.push(BeatEvent::Bar(BarInfo {
                bar: info.bar,
                beat: info.beat,
            }));
        }

        events
    }

    /// Whether tracking is active.
    #[inline]
    pub fn is_tracking(&self) -> bool {
        self.tracking
    }

    // -----------------------------------------------------------------------
    // Queries (delegate to detector)
    // -----------------------------------------------------------------------

    /// Get current beat info from a fractional beat position.
    pub fn get_beat_info(&self, exact_beat: f64) -> BeatInfo {
        self.detector.calculate_beat_info(exact_beat)
    }

    /// Get current loop count.
    pub fn get_loop_count(&self, exact_beat: f64) -> u32 {
        self.detector.calculate_loop_count(exact_beat)
    }

    /// Get a full status snapshot.
    pub fn get_status(&self, exact_beat: f64) -> BeatStatus {
        let info = self.detector.calculate_beat_info(exact_beat);
        let loop_count = self.detector.calculate_loop_count(exact_beat);

        BeatStatus {
            beat: info.beat,
            bar: info.bar,
            absolute_beat: info.absolute_beat,
            beat_fraction: info.beat_fraction,
            loop_count,
            is_playing: self.tracking,
        }
    }

    /// Get beat info with alignment data.
    pub fn get_beat_alignment(&self, exact_beat: f64) -> BeatAlignment {
        self.detector.calculate_beat_alignment(exact_beat)
    }

    /// Validate a beat position.
    pub fn validate_beat_position(&self, target_beat: f64) -> BeatValidation {
        self.detector.validate_beat_position(target_beat)
    }

    /// Optimize a rough beat position.
    pub fn optimize_beat_position(&self, rough_beat: f64) -> (f64, f64, &'static str) {
        self.detector.optimize_beat_position(rough_beat)
    }

    /// Access the underlying detector.
    pub fn detector(&self) -> &BeatDetector {
        &self.detector
    }

    /// Access configuration.
    pub fn config(&self) -> &BeatTrackingConfig {
        self.detector.config()
    }

    /// Update configuration.
    pub fn set_config(&mut self, config: BeatTrackingConfig) {
        self.detector.set_config(config);
    }

    // -----------------------------------------------------------------------
    // Change detection + event collection
    // -----------------------------------------------------------------------

    /// Update the tracker with a new fractional beat position.
    ///
    /// Returns any beat/bar/loop boundary events that occurred since the
    /// last update. Call this once per frame (or per metronome event).
    ///
    /// This is the core method ported from `BeatStateManager.checkAndTriggerCallbacks`.
    pub fn update(&mut self, exact_beat: f64) -> (BeatChangeDetection, Vec<BeatEvent>) {
        if !self.tracking {
            return (
                BeatChangeDetection {
                    beat_changed: false,
                    bar_changed: false,
                    loop_changed: false,
                },
                Vec::new(),
            );
        }

        let info = self.detector.calculate_beat_info(exact_beat);
        let loop_count = self.detector.calculate_loop_count(exact_beat);
        let mut events = Vec::new();
        let mut detection = BeatChangeDetection {
            beat_changed: false,
            bar_changed: false,
            loop_changed: false,
        };

        // Check for beat change
        if info.beat != self.previous_beat {
            self.previous_beat = info.beat;
            detection.beat_changed = true;
            events.push(BeatEvent::Beat(info));
        }

        // Check for bar change
        if info.bar != self.previous_bar {
            self.previous_bar = info.bar;
            detection.bar_changed = true;
            events.push(BeatEvent::Bar(BarInfo {
                bar: info.bar,
                beat: info.beat,
            }));
        }

        // Check for loop change
        if loop_count != self.previous_loop {
            self.previous_loop = loop_count;
            detection.loop_changed = true;
            events.push(BeatEvent::Loop(LoopInfo { loop_count }));
        }

        (detection, events)
    }

    /// Get previous state (for debugging).
    pub fn previous_state(&self) -> (u32, u32, u32) {
        (self.previous_beat, self.previous_bar, self.previous_loop)
    }
}

// ---------------------------------------------------------------------------
// Staggered track start calculations
// ---------------------------------------------------------------------------

/// Entry in a staggered track start sequence.
#[derive(Debug, Clone, PartialEq)]
pub struct StaggeredTrackStart {
    pub track_id: String,
    pub start_beat: f64,
    pub delay: f64,
}

/// Calculate staggered track starts for multi-track arrangements.
///
/// Ported from `TimingCalculator.calculateStaggeredTrackStarts`.
pub fn calculate_staggered_track_starts(
    base_track_id: &str,
    additional_tracks: &[&str],
    stagger_pattern: &[f64],
    current_beat: f64,
) -> Vec<StaggeredTrackStart> {
    let mut starts = Vec::with_capacity(1 + additional_tracks.len());

    // Base track starts at current beat
    starts.push(StaggeredTrackStart {
        track_id: base_track_id.to_string(),
        start_beat: current_beat,
        delay: 0.0,
    });

    // Additional tracks staggered by pattern
    let mut cumulative_delay = 0.0;
    for (i, &track_id) in additional_tracks.iter().enumerate() {
        let stagger = if i < stagger_pattern.len() {
            stagger_pattern[i]
        } else {
            // Repeat last pattern value if pattern is shorter than tracks
            stagger_pattern.last().copied().unwrap_or(1.0)
        };
        cumulative_delay += stagger;

        starts.push(StaggeredTrackStart {
            track_id: track_id.to_string(),
            start_beat: current_beat + cumulative_delay,
            delay: cumulative_delay,
        });
    }

    starts
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- BeatDetector -------------------------------------------------------

    mod detector {
        use super::*;

        fn make_detector() -> BeatDetector {
            BeatDetector::with_defaults() // 4/4, 4 bars
        }

        #[test]
        fn beat_info_at_start() {
            let d = make_detector();
            // exact_beat = 1.0 → absolute_beat = 1, beat 1, bar 1
            let info = d.calculate_beat_info(1.0);
            assert_eq!(info.beat, 1);
            assert_eq!(info.bar, 1);
            assert_eq!(info.absolute_beat, 1);
            assert!((info.beat_fraction).abs() < 1e-10);
        }

        #[test]
        fn beat_info_mid_bar() {
            let d = make_detector();
            // exact_beat = 3.5 → absolute_beat = 3, beat 3, bar 1
            let info = d.calculate_beat_info(3.5);
            assert_eq!(info.beat, 3);
            assert_eq!(info.bar, 1);
            assert_eq!(info.absolute_beat, 3);
            assert!((info.beat_fraction - 0.5).abs() < 1e-10);
        }

        #[test]
        fn beat_info_bar_boundary() {
            let d = make_detector();
            // exact_beat = 5.0 → absolute_beat = 5, beat 1 of bar 2
            let info = d.calculate_beat_info(5.0);
            assert_eq!(info.beat, 1);
            assert_eq!(info.bar, 2);
            assert_eq!(info.absolute_beat, 5);
        }

        #[test]
        fn beat_info_second_bar() {
            let d = make_detector();
            // exact_beat = 7.0 → absolute_beat = 7, beat 3 of bar 2
            let info = d.calculate_beat_info(7.0);
            assert_eq!(info.beat, 3);
            assert_eq!(info.bar, 2);
        }

        #[test]
        fn beat_info_loop_boundary() {
            let d = make_detector();
            // 4 bars × 4 beats = 16 beats per loop
            // exact_beat = 17.0 → second loop, beat 1, bar 1
            let info = d.calculate_beat_info(17.0);
            assert_eq!(info.beat, 1);
            assert_eq!(info.bar, 1);
            assert_eq!(info.absolute_beat, 17);
        }

        #[test]
        fn beat_info_end_of_bar() {
            let d = make_detector();
            // exact_beat = 4.0 → absolute_beat = 4, beat 4, bar 1
            let info = d.calculate_beat_info(4.0);
            assert_eq!(info.beat, 4);
            assert_eq!(info.bar, 1);
        }

        #[test]
        fn beat_info_end_of_loop() {
            let d = make_detector();
            // exact_beat = 16.0 → absolute_beat = 16, beat 4, bar 4
            let info = d.calculate_beat_info(16.0);
            assert_eq!(info.beat, 4);
            assert_eq!(info.bar, 4);
        }

        #[test]
        fn beat_info_zero() {
            let d = make_detector();
            let info = d.calculate_beat_info(0.0);
            assert_eq!(info.beat, 1);
            assert_eq!(info.bar, 1);
            assert_eq!(info.absolute_beat, 1);
            assert!((info.beat_fraction).abs() < 1e-10);
        }

        #[test]
        fn beat_info_precise_beat() {
            let d = make_detector();
            // exact_beat = 3.7 → preciseBeat = (3-1) + 0.7 = 2.7
            let info = d.calculate_beat_info(3.7);
            assert!((info.precise_beat - 2.7).abs() < 1e-10);
        }

        #[test]
        fn loop_count_first_loop() {
            let d = make_detector();
            assert_eq!(d.calculate_loop_count(1.0), 0);
            assert_eq!(d.calculate_loop_count(16.0), 0);
        }

        #[test]
        fn loop_count_second_loop() {
            let d = make_detector();
            assert_eq!(d.calculate_loop_count(17.0), 1);
            assert_eq!(d.calculate_loop_count(32.0), 1);
        }

        #[test]
        fn loop_count_third_loop() {
            let d = make_detector();
            assert_eq!(d.calculate_loop_count(33.0), 2);
        }

        #[test]
        fn loop_count_zero() {
            let d = make_detector();
            assert_eq!(d.calculate_loop_count(0.0), 0);
        }

        #[test]
        fn custom_time_signature() {
            let d = BeatDetector::new(BeatTrackingConfig {
                beats_per_bar: 3,
                bars_per_loop: 4,
            });
            // 3/4 time: beat 4 should be beat 1 of bar 2
            let info = d.calculate_beat_info(4.0);
            assert_eq!(info.beat, 1);
            assert_eq!(info.bar, 2);

            // Loop is 3*4=12 beats
            assert_eq!(d.calculate_loop_count(13.0), 1);
        }

        #[test]
        fn beat_alignment_on_beat() {
            let d = make_detector();
            let alignment = d.calculate_beat_alignment(3.0);
            assert!(alignment.is_near_beat_boundary);
            assert!(alignment.alignment_accuracy < 0.001);
        }

        #[test]
        fn beat_alignment_off_beat() {
            let d = make_detector();
            let alignment = d.calculate_beat_alignment(3.3);
            assert!(!alignment.is_near_beat_boundary);
            assert!((alignment.alignment_accuracy - 0.3).abs() < 0.001);
        }

        #[test]
        fn validate_beat_position_valid() {
            let d = make_detector();
            let v = d.validate_beat_position(4.0);
            assert!(v.is_valid);
            assert!(v.musically_correct);
            assert!(v.suggestion.is_none());
        }

        #[test]
        fn validate_beat_position_half_beat() {
            let d = make_detector();
            let v = d.validate_beat_position(2.5);
            assert!(v.is_valid);
            assert!(v.musically_correct);
        }

        #[test]
        fn validate_beat_position_odd_fraction() {
            let d = make_detector();
            let v = d.validate_beat_position(2.37);
            assert!(v.is_valid);
            assert!(!v.musically_correct);
            assert!(v.suggestion.is_some());
        }

        #[test]
        fn validate_beat_position_negative() {
            let d = make_detector();
            let v = d.validate_beat_position(-1.0);
            assert!(!v.is_valid);
        }

        #[test]
        fn optimize_beat_whole() {
            let d = make_detector();
            let (beat, confidence, _reason) = d.optimize_beat_position(2.02);
            assert!((beat - 2.0).abs() < 0.001);
            assert!(confidence > 0.9);
        }

        #[test]
        fn optimize_beat_half() {
            let d = make_detector();
            let (beat, confidence, _reason) = d.optimize_beat_position(2.48);
            assert!((beat - 2.5).abs() < 0.001);
            assert!(confidence > 0.9);
        }

        #[test]
        fn optimize_beat_quarter() {
            let d = make_detector();
            let (beat, _confidence, _reason) = d.optimize_beat_position(2.24);
            assert!((beat - 2.25).abs() < 0.001);
        }

        #[test]
        fn calculate_beat_offset_delegates() {
            let d = make_detector();
            let offset = d.calculate_beat_offset(4.0, 1.0, 120.0).unwrap();
            assert_eq!(offset.sample_position, 96_000);
            assert!((offset.beat_difference - 3.0).abs() < 1e-10);
        }
    }

    // -- BeatTracker --------------------------------------------------------

    mod tracker {
        use super::*;

        fn make_tracker() -> BeatTracker {
            BeatTracker::with_defaults()
        }

        #[test]
        fn start_stop_lifecycle() {
            let mut t = make_tracker();
            assert!(!t.is_tracking());

            let event = t.start();
            assert!(t.is_tracking());
            assert_eq!(event, Some(BeatEvent::Play));

            // Double start is no-op
            assert_eq!(t.start(), None);

            let event = t.stop();
            assert!(!t.is_tracking());
            assert_eq!(event, BeatEvent::Stop);
        }

        #[test]
        fn update_not_tracking_returns_empty() {
            let mut t = make_tracker();
            let (detection, events) = t.update(5.0);
            assert!(!detection.beat_changed);
            assert!(events.is_empty());
        }

        #[test]
        fn update_detects_beat_change() {
            let mut t = make_tracker();
            t.start();

            // First update: beat 1 → 2 (previous was 1)
            let (detection, events) = t.update(2.0);
            assert!(detection.beat_changed);
            assert_eq!(events.len(), 1);
            match &events[0] {
                BeatEvent::Beat(info) => {
                    assert_eq!(info.beat, 2);
                    assert_eq!(info.bar, 1);
                }
                _ => panic!("Expected Beat event"),
            }
        }

        #[test]
        fn update_detects_bar_change() {
            let mut t = make_tracker();
            t.start();

            // First advance to beat 4 (still bar 1) to set previous_beat to 4
            t.update(4.0);

            // Move to beat 5 → beat 1 of bar 2 (previous bar was 1)
            let (detection, events) = t.update(5.0);
            assert!(detection.beat_changed); // beat 4 → 1
            assert!(detection.bar_changed);  // bar 1 → 2
            assert_eq!(events.len(), 2); // Beat + Bar
        }

        #[test]
        fn update_detects_loop_change() {
            let mut t = make_tracker();
            t.start();

            // Advance to beat 16 (last beat of loop 0) to set state
            t.update(16.0);
            assert_eq!(t.previous_state(), (4, 4, 0));

            // Move to beat 17 → beat 1, bar 1, loop 1
            let (detection, events) = t.update(17.0);
            assert!(detection.beat_changed);  // beat 4 → 1
            assert!(detection.bar_changed);   // bar 4 → 1
            assert!(detection.loop_changed);  // loop 0 → 1
            assert_eq!(events.len(), 3); // Beat + Bar + Loop
        }

        #[test]
        fn update_no_change_same_beat() {
            let mut t = make_tracker();
            t.start();

            // First update to beat 2
            t.update(2.0);

            // Same beat, different fraction
            let (detection, events) = t.update(2.5);
            assert!(!detection.beat_changed);
            assert!(!detection.bar_changed);
            assert!(!detection.loop_changed);
            assert!(events.is_empty());
        }

        #[test]
        fn sequential_beat_updates() {
            let mut t = make_tracker();
            t.start();

            // Beat 1 → 2
            let (d, _) = t.update(2.0);
            assert!(d.beat_changed);

            // Beat 2 → 3
            let (d, _) = t.update(3.0);
            assert!(d.beat_changed);

            // Beat 3 → 4
            let (d, _) = t.update(4.0);
            assert!(d.beat_changed);

            // Beat 4 → 5 (bar change)
            let (d, events) = t.update(5.0);
            assert!(d.beat_changed);
            assert!(d.bar_changed);
            assert!(events.iter().any(|e| matches!(e, BeatEvent::Bar(_))));
        }

        #[test]
        fn seek_updates_state() {
            let mut t = make_tracker();
            t.start();

            let events = t.seek(1.5, 8.0);
            assert!(events.iter().any(|e| matches!(e, BeatEvent::Seek { .. })));
            assert!(events.iter().any(|e| matches!(e, BeatEvent::Beat(_))));

            // Verify internal state was updated
            assert_eq!(t.previous_state(), (4, 2, 0));
        }

        #[test]
        fn stop_resets_state() {
            let mut t = make_tracker();
            t.start();
            t.update(10.0); // Move to some position

            t.stop();
            assert_eq!(t.previous_state(), (1, 1, 0));
        }

        #[test]
        fn get_status() {
            let mut t = make_tracker();
            t.start();

            let status = t.get_status(5.5);
            assert_eq!(status.beat, 1);
            assert_eq!(status.bar, 2);
            assert_eq!(status.absolute_beat, 5);
            assert!(status.is_playing);
            assert_eq!(status.loop_count, 0);
        }

        #[test]
        fn get_status_not_playing() {
            let t = make_tracker();
            let status = t.get_status(5.0);
            assert!(!status.is_playing);
        }
    }

    // -- Staggered track starts ---------------------------------------------

    #[test]
    fn staggered_starts_basic() {
        let starts = calculate_staggered_track_starts(
            "bass",
            &["drums", "keys"],
            &[2.0, 4.0],
            1.0,
        );

        assert_eq!(starts.len(), 3);
        assert_eq!(starts[0].track_id, "bass");
        assert!((starts[0].start_beat - 1.0).abs() < 1e-10);
        assert!((starts[0].delay).abs() < 1e-10);

        assert_eq!(starts[1].track_id, "drums");
        assert!((starts[1].start_beat - 3.0).abs() < 1e-10);
        assert!((starts[1].delay - 2.0).abs() < 1e-10);

        assert_eq!(starts[2].track_id, "keys");
        assert!((starts[2].start_beat - 7.0).abs() < 1e-10);
        assert!((starts[2].delay - 6.0).abs() < 1e-10);
    }

    #[test]
    fn staggered_starts_pattern_shorter_than_tracks() {
        let starts = calculate_staggered_track_starts(
            "lead",
            &["rhythm", "bass", "pad"],
            &[1.0], // Only one pattern value
            0.0,
        );

        assert_eq!(starts.len(), 4);
        // All additional tracks use 1.0 stagger
        assert!((starts[1].delay - 1.0).abs() < 1e-10);
        assert!((starts[2].delay - 2.0).abs() < 1e-10);
        assert!((starts[3].delay - 3.0).abs() < 1e-10);
    }

    #[test]
    fn staggered_starts_empty_additional() {
        let starts = calculate_staggered_track_starts("solo", &[], &[], 5.0);
        assert_eq!(starts.len(), 1);
        assert_eq!(starts[0].track_id, "solo");
        assert!((starts[0].start_beat - 5.0).abs() < 1e-10);
    }

    // -- BeatTrackingConfig -------------------------------------------------

    #[test]
    fn config_defaults() {
        let config = BeatTrackingConfig::default();
        assert_eq!(config.beats_per_bar, 4);
        assert_eq!(config.bars_per_loop, 4);
        assert_eq!(config.beats_per_loop(), 16);
    }

    #[test]
    fn config_custom() {
        let config = BeatTrackingConfig {
            beats_per_bar: 3,
            bars_per_loop: 8,
        };
        assert_eq!(config.beats_per_loop(), 24);
    }
}
