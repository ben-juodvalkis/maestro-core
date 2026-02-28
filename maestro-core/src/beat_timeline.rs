//! Beat timeline — beat tracking and timing state management.
//!
//! Ported from:
//! - `src/domains/core/services/SimpleBeatTimeline.ts`
//! - `src/domains/state/services/AudioContextTimingState.ts`
//!
//! In the TypeScript codebase these are two separate classes: `SimpleBeatTimeline`
//! wraps `AudioContextTimingState` and adds initialization / tempo-change logic.
//! In Rust we merge them into a single `BeatTimeline` struct because the state
//! object existed only to share timing across JS threads (SharedArrayBuffer / message
//! passing). In the Rust core all access is in-process, so a single owning struct
//! is cleaner.
//!
//! ## Thread-safety
//!
//! On **native** targets the audio render thread reads timing state while the
//! control thread writes it. We use `cfg` to select between atomic and plain
//! fields (see roadmap Decision D05).
//!
//! On **WASM** the engine is single-threaded so no atomics are needed.

use crate::timing::{
    samples_per_beat, floor_samples, DEFAULT_SAMPLE_RATE, DEFAULT_TIME_SIGNATURE,
};

// ---------------------------------------------------------------------------
// Timing state snapshot (for serialization / crash recovery)
// ---------------------------------------------------------------------------

/// Complete snapshot of the timing state — used for save/restore and
/// WebView crash recovery (roadmap Section 3.6).
#[derive(Debug, Clone, PartialEq)]
pub struct TimingSnapshot {
    pub global_sample_position: i64,
    pub beat_position: f64,
    pub current_beat_count: u32,
    pub next_beat_sample: i64,
    pub epoch_sample_position: i64,
    pub tempo: f64,
    pub sample_rate: u32,
    pub beats_per_bar: u32,
    pub is_initialized: bool,
    pub track_bpm: f64,
}

// ---------------------------------------------------------------------------
// Pending tempo change
// ---------------------------------------------------------------------------

/// A tempo change queued for beat-aligned application.
#[derive(Debug, Clone, Copy, PartialEq)]
struct PendingTempo {
    bpm: f64,
    apply_at_sample: i64,
}

// ---------------------------------------------------------------------------
// BeatTimeline
// ---------------------------------------------------------------------------

/// Combined beat timeline and timing state.
///
/// Merges the TypeScript `SimpleBeatTimeline` + `AudioContextTimingState` into
/// a single struct.
#[derive(Debug)]
pub struct BeatTimeline {
    // -- Timing state -------------------------------------------------------
    global_sample_position: i64,
    beat_position: f64,
    /// 0-based internally; public API returns 1-based.
    current_beat_count: u32,
    next_beat_sample: i64,
    epoch_sample_position: i64,
    tempo: f64,
    sample_rate: u32,
    beats_per_bar: u32,
    initialized: bool,
    track_bpm: f64,

    // -- Pending tempo change -----------------------------------------------
    pending_tempo: Option<PendingTempo>,
}

impl BeatTimeline {
    /// Create a new `BeatTimeline` with default configuration.
    pub fn new(sample_rate: u32) -> Self {
        Self {
            global_sample_position: 0,
            beat_position: 0.0,
            current_beat_count: 0,
            next_beat_sample: 0,
            epoch_sample_position: 0,
            tempo: 0.0,
            sample_rate,
            beats_per_bar: DEFAULT_TIME_SIGNATURE,
            initialized: false,
            track_bpm: 0.0,
            pending_tempo: None,
        }
    }

    /// Create with the default 48 kHz sample rate.
    pub fn with_defaults() -> Self {
        Self::new(DEFAULT_SAMPLE_RATE)
    }

    // -----------------------------------------------------------------------
    // Lifecycle
    // -----------------------------------------------------------------------

    /// Initialize the timeline when playback starts.
    ///
    /// Sets up the first beat to occur one beat-duration from the current
    /// sample position. Calling this when already initialized is a no-op.
    pub fn initialize(&mut self) {
        if self.initialized {
            return;
        }

        let spb = self.samples_per_beat_floored();
        let first_beat_sample = self.global_sample_position + spb;

        self.epoch_sample_position = self.global_sample_position;
        self.beat_position = 0.0;
        self.current_beat_count = 0;
        self.next_beat_sample = first_beat_sample;
        self.initialized = true;
    }

    /// Reset the timeline (stop / restart). Clears initialization but
    /// preserves configuration (sample rate, tempo, beats per bar, track BPM).
    pub fn reset(&mut self) {
        self.initialized = false;
        self.beat_position = 0.0;
        self.current_beat_count = 0;
        self.epoch_sample_position = 0;
        self.next_beat_sample = 0;
        self.pending_tempo = None;
    }

    /// Whether the timeline has been initialized (playback has started).
    #[inline]
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    // -----------------------------------------------------------------------
    // Position
    // -----------------------------------------------------------------------

    /// Get current global sample position.
    #[inline]
    pub fn get_position(&self) -> i64 {
        self.global_sample_position
    }

    /// Set global sample position (for snapshot restoration).
    pub fn set_position(&mut self, position: i64) {
        self.global_sample_position = position;
    }

    /// Increment the global sample position by `samples` and return the new value.
    pub fn increment_position(&mut self, samples: i64) -> i64 {
        self.global_sample_position += samples;
        self.global_sample_position
    }

    /// Sample position when the timeline started (epoch).
    #[inline]
    pub fn get_epoch_sample_position(&self) -> i64 {
        self.epoch_sample_position
    }

    /// Samples elapsed since the epoch.
    #[inline]
    pub fn get_relative_sample_position(&self) -> i64 {
        self.global_sample_position - self.epoch_sample_position
    }

    // -----------------------------------------------------------------------
    // Beat position
    // -----------------------------------------------------------------------

    /// Current beat count (1-based for UI display).
    #[inline]
    pub fn get_current_beat(&self) -> u32 {
        self.current_beat_count + 1
    }

    /// High-resolution fractional beat position.
    #[inline]
    pub fn get_beat_position(&self) -> f64 {
        self.beat_position
    }

    /// Fractional part of the beat position (0.0 – 0.999…).
    #[inline]
    pub fn get_fractional_beat(&self) -> f64 {
        self.beat_position - self.beat_position.floor()
    }

    /// Increment the beat counter by one, returning the new 1-based beat.
    pub fn increment_beat(&mut self) -> u32 {
        self.current_beat_count += 1;
        self.current_beat_count + 1
    }

    /// Increment the fractional beat position.
    ///
    /// Also updates the integer beat count for backward compatibility.
    ///
    /// # Panics
    /// Panics if `increment` is not finite or ≤ 0.
    pub fn increment_beat_position(&mut self, increment: f64) -> f64 {
        assert!(
            increment.is_finite() && increment > 0.0,
            "Invalid beat increment: {increment}"
        );
        self.beat_position += increment;
        self.current_beat_count = self.beat_position.floor() as u32;
        self.beat_position
    }

    /// Set beat count from a 1-based value (for snapshot restoration).
    pub fn set_current_beat(&mut self, beat_1based: u32) {
        self.current_beat_count = beat_1based.saturating_sub(1);
    }

    // -----------------------------------------------------------------------
    // Next beat sample
    // -----------------------------------------------------------------------

    /// Sample position of the next scheduled beat boundary.
    #[inline]
    pub fn get_next_beat_sample(&self) -> i64 {
        self.next_beat_sample
    }

    /// Set the next beat sample position.
    pub fn set_next_beat_sample(&mut self, sample: i64) {
        self.next_beat_sample = sample;
    }

    /// Calculate when the next beat should occur.
    ///
    /// If a future beat is already scheduled, returns that value.
    /// Otherwise computes `current_position + samples_per_beat`.
    pub fn calculate_next_beat_sample(&self) -> i64 {
        if self.next_beat_sample > self.global_sample_position {
            return self.next_beat_sample;
        }
        self.global_sample_position + self.samples_per_beat_floored()
    }

    // -----------------------------------------------------------------------
    // Tempo
    // -----------------------------------------------------------------------

    /// Current tempo in BPM.
    ///
    /// # Panics
    /// Panics if tempo has not been set (is ≤ 0).
    pub fn get_tempo(&self) -> f64 {
        assert!(
            self.tempo > 0.0,
            "BeatTimeline: Tempo not initialized — set tempo from song metadata before use"
        );
        self.tempo
    }

    /// Set the current tempo directly (used during initial configuration).
    pub fn set_tempo_direct(&mut self, bpm: f64) {
        self.tempo = bpm;
    }

    /// Queue a tempo change to be applied at the next beat boundary
    /// (beat-aligned to prevent artifacts).
    pub fn set_tempo(&mut self, bpm: f64) {
        if bpm <= 0.0 || bpm > 999.0 {
            return;
        }
        let next_beat = self.calculate_next_beat_sample();
        self.pending_tempo = Some(PendingTempo {
            bpm,
            apply_at_sample: next_beat,
        });
    }

    /// Whether a tempo change is pending.
    pub fn has_pending_tempo(&self) -> bool {
        self.pending_tempo.is_some()
    }

    /// Get the pending tempo value (0.0 if none).
    pub fn get_pending_tempo(&self) -> f64 {
        self.pending_tempo.map_or(0.0, |p| p.bpm)
    }

    /// Get the sample position where the pending tempo should be applied.
    pub fn get_pending_tempo_apply_sample(&self) -> i64 {
        self.pending_tempo.map_or(0, |p| p.apply_at_sample)
    }

    /// Apply the pending tempo change if we've reached its trigger sample.
    /// Returns `true` if the tempo was changed.
    pub fn try_apply_pending_tempo(&mut self) -> bool {
        if let Some(pending) = self.pending_tempo {
            if self.global_sample_position >= pending.apply_at_sample {
                self.tempo = pending.bpm;
                self.pending_tempo = None;
                return true;
            }
        }
        false
    }

    /// Clear a pending tempo change without applying it.
    pub fn clear_pending_tempo(&mut self) {
        self.pending_tempo = None;
    }

    // -----------------------------------------------------------------------
    // Track BPM (from song metadata)
    // -----------------------------------------------------------------------

    /// Set the track's original BPM (from song metadata).
    pub fn set_track_bpm(&mut self, bpm: f64) {
        self.track_bpm = bpm;
    }

    /// Get the track's original BPM.
    #[inline]
    pub fn get_track_bpm(&self) -> f64 {
        self.track_bpm
    }

    // -----------------------------------------------------------------------
    // Time signature
    // -----------------------------------------------------------------------

    /// Set beats per bar (for bar/accent calculations).
    ///
    /// Silently ignores values outside 1..=16.
    pub fn set_beats_per_bar(&mut self, beats: u32) {
        if (1..=16).contains(&beats) {
            self.beats_per_bar = beats;
        }
    }

    /// Get beats per bar.
    #[inline]
    pub fn get_beats_per_bar(&self) -> u32 {
        self.beats_per_bar
    }

    // -----------------------------------------------------------------------
    // Sample rate
    // -----------------------------------------------------------------------

    /// Get sample rate.
    #[inline]
    pub fn get_sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Set sample rate.
    pub fn set_sample_rate(&mut self, rate: u32) {
        self.sample_rate = rate;
    }

    // -----------------------------------------------------------------------
    // Beat ↔ sample conversion
    // -----------------------------------------------------------------------

    /// Convert a beat position to a sample position using the track BPM.
    pub fn beat_to_sample_position(&self, beat_position: f64) -> i64 {
        let spb = samples_per_beat(self.track_bpm, self.sample_rate)
            .unwrap_or(0.0);
        (beat_position * spb).round() as i64
    }

    // -----------------------------------------------------------------------
    // Snapshot
    // -----------------------------------------------------------------------

    /// Create a complete snapshot of the timing state.
    pub fn get_snapshot(&self) -> TimingSnapshot {
        TimingSnapshot {
            global_sample_position: self.global_sample_position,
            beat_position: self.beat_position,
            current_beat_count: self.get_current_beat(),
            next_beat_sample: self.next_beat_sample,
            epoch_sample_position: self.epoch_sample_position,
            tempo: self.tempo,
            sample_rate: self.sample_rate,
            beats_per_bar: self.beats_per_bar,
            is_initialized: self.initialized,
            track_bpm: self.track_bpm,
        }
    }

    /// Restore timing state from a snapshot.
    pub fn restore_from_snapshot(&mut self, snap: &TimingSnapshot) {
        self.global_sample_position = snap.global_sample_position;
        self.beat_position = snap.beat_position;
        self.set_current_beat(snap.current_beat_count);
        self.next_beat_sample = snap.next_beat_sample;
        self.epoch_sample_position = snap.epoch_sample_position;
        self.tempo = snap.tempo;
        self.sample_rate = snap.sample_rate;
        self.beats_per_bar = snap.beats_per_bar;
        self.track_bpm = snap.track_bpm;
        self.initialized = true;
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Floored samples-per-beat at the current tempo and sample rate.
    fn samples_per_beat_floored(&self) -> i64 {
        let spb = samples_per_beat(self.tempo, self.sample_rate).unwrap_or(0.0);
        floor_samples(spb)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_timeline(tempo: f64) -> BeatTimeline {
        let mut bt = BeatTimeline::new(48_000);
        bt.set_tempo_direct(tempo);
        bt
    }

    // -- construction & defaults -------------------------------------------

    #[test]
    fn new_with_defaults() {
        let bt = BeatTimeline::with_defaults();
        assert_eq!(bt.get_sample_rate(), 48_000);
        assert_eq!(bt.get_beats_per_bar(), 4);
        assert!(!bt.is_initialized());
    }

    // -- initialize / reset ------------------------------------------------

    #[test]
    fn initialize_sets_first_beat() {
        let mut bt = make_timeline(120.0);
        bt.set_position(0);
        bt.initialize();

        assert!(bt.is_initialized());
        // At 120 BPM, 48kHz: samples_per_beat = 24000 → floor = 24000
        assert_eq!(bt.get_next_beat_sample(), 24_000);
        assert_eq!(bt.get_epoch_sample_position(), 0);
        assert_eq!(bt.get_current_beat(), 1); // 1-based
        assert_eq!(bt.get_beat_position(), 0.0);
    }

    #[test]
    fn initialize_is_idempotent() {
        let mut bt = make_timeline(120.0);
        bt.initialize();
        let first_next = bt.get_next_beat_sample();
        bt.initialize(); // should be no-op
        assert_eq!(bt.get_next_beat_sample(), first_next);
    }

    #[test]
    fn reset_clears_state() {
        let mut bt = make_timeline(120.0);
        bt.initialize();
        bt.increment_position(10_000);
        bt.increment_beat();

        bt.reset();

        assert!(!bt.is_initialized());
        assert_eq!(bt.get_beat_position(), 0.0);
        assert_eq!(bt.get_current_beat(), 1); // 0-based + 1
        assert_eq!(bt.get_next_beat_sample(), 0);
        assert_eq!(bt.get_epoch_sample_position(), 0);
        // Configuration preserved
        assert_eq!(bt.get_sample_rate(), 48_000);
    }

    // -- position -----------------------------------------------------------

    #[test]
    fn increment_position() {
        let mut bt = make_timeline(120.0);
        assert_eq!(bt.increment_position(1000), 1000);
        assert_eq!(bt.increment_position(500), 1500);
        assert_eq!(bt.get_position(), 1500);
    }

    #[test]
    fn relative_position() {
        let mut bt = make_timeline(120.0);
        bt.set_position(10_000);
        bt.initialize();
        bt.increment_position(5_000);
        assert_eq!(bt.get_relative_sample_position(), 5_000);
    }

    // -- beat position ------------------------------------------------------

    #[test]
    fn beat_count_1_based() {
        let bt = make_timeline(120.0);
        // Not initialized → internal count = 0 → 1-based = 1
        assert_eq!(bt.get_current_beat(), 1);
    }

    #[test]
    fn increment_beat() {
        let mut bt = make_timeline(120.0);
        assert_eq!(bt.increment_beat(), 2); // 0→1 internally, 1-based = 2
        assert_eq!(bt.increment_beat(), 3);
    }

    #[test]
    fn increment_beat_position_updates_integer() {
        let mut bt = make_timeline(120.0);
        bt.increment_beat_position(2.5);
        assert_eq!(bt.get_beat_position(), 2.5);
        assert_eq!(bt.get_current_beat(), 3); // floor(2.5) + 1 = 3
        assert!((bt.get_fractional_beat() - 0.5).abs() < 1e-10);
    }

    #[test]
    #[should_panic(expected = "Invalid beat increment")]
    fn increment_beat_position_rejects_zero() {
        let mut bt = make_timeline(120.0);
        bt.increment_beat_position(0.0);
    }

    #[test]
    fn set_current_beat_from_1_based() {
        let mut bt = make_timeline(120.0);
        bt.set_current_beat(5);
        assert_eq!(bt.get_current_beat(), 5);
    }

    // -- next beat sample ---------------------------------------------------

    #[test]
    fn calculate_next_beat_uses_scheduled() {
        let mut bt = make_timeline(120.0);
        bt.set_position(0);
        bt.set_next_beat_sample(24_000);
        assert_eq!(bt.calculate_next_beat_sample(), 24_000);
    }

    #[test]
    fn calculate_next_beat_computes_when_past() {
        let mut bt = make_timeline(120.0);
        bt.set_position(30_000);
        bt.set_next_beat_sample(24_000); // in the past
        // Should compute 30000 + 24000 = 54000
        assert_eq!(bt.calculate_next_beat_sample(), 54_000);
    }

    // -- tempo --------------------------------------------------------------

    #[test]
    fn set_tempo_queues_pending() {
        let mut bt = make_timeline(100.0);
        bt.set_position(0);
        bt.set_next_beat_sample(28_800); // next beat at 28800

        bt.set_tempo(120.0);

        assert!(bt.has_pending_tempo());
        assert_eq!(bt.get_pending_tempo(), 120.0);
        assert_eq!(bt.get_pending_tempo_apply_sample(), 28_800);
    }

    #[test]
    fn set_tempo_ignores_invalid() {
        let mut bt = make_timeline(100.0);
        bt.set_tempo(0.0);
        assert!(!bt.has_pending_tempo());
        bt.set_tempo(-10.0);
        assert!(!bt.has_pending_tempo());
        bt.set_tempo(1000.0);
        assert!(!bt.has_pending_tempo());
    }

    #[test]
    fn try_apply_pending_tempo_before_trigger() {
        let mut bt = make_timeline(100.0);
        bt.set_position(0);
        bt.set_next_beat_sample(28_800);
        bt.set_tempo(120.0);

        // Not yet at trigger sample
        assert!(!bt.try_apply_pending_tempo());
        assert!(bt.has_pending_tempo());
    }

    #[test]
    fn try_apply_pending_tempo_at_trigger() {
        let mut bt = make_timeline(100.0);
        bt.set_position(0);
        bt.set_next_beat_sample(28_800);
        bt.set_tempo(120.0);

        bt.set_position(28_800);
        assert!(bt.try_apply_pending_tempo());
        assert!(!bt.has_pending_tempo());
        assert_eq!(bt.get_tempo(), 120.0);
    }

    // -- beats per bar ------------------------------------------------------

    #[test]
    fn beats_per_bar_validation() {
        let mut bt = make_timeline(120.0);
        bt.set_beats_per_bar(3);
        assert_eq!(bt.get_beats_per_bar(), 3);

        bt.set_beats_per_bar(0); // ignored
        assert_eq!(bt.get_beats_per_bar(), 3);

        bt.set_beats_per_bar(17); // ignored
        assert_eq!(bt.get_beats_per_bar(), 3);

        bt.set_beats_per_bar(16); // accepted
        assert_eq!(bt.get_beats_per_bar(), 16);
    }

    // -- track BPM ----------------------------------------------------------

    #[test]
    fn track_bpm() {
        let mut bt = make_timeline(120.0);
        bt.set_track_bpm(100.0);
        assert_eq!(bt.get_track_bpm(), 100.0);
    }

    // -- beat_to_sample_position -------------------------------------------

    #[test]
    fn beat_to_sample_position_basic() {
        let mut bt = BeatTimeline::new(48_000);
        bt.set_track_bpm(120.0);
        // At 120 BPM: 1 beat = 24000 samples
        assert_eq!(bt.beat_to_sample_position(1.0), 24_000);
        assert_eq!(bt.beat_to_sample_position(4.0), 96_000);
    }

    // -- snapshot -----------------------------------------------------------

    #[test]
    fn snapshot_roundtrip() {
        let mut bt = make_timeline(100.0);
        bt.set_track_bpm(100.0);
        bt.set_beats_per_bar(3);
        bt.set_position(50_000);
        bt.initialize();
        bt.increment_beat();
        bt.increment_beat_position(1.5);

        let snap = bt.get_snapshot();

        let mut bt2 = BeatTimeline::new(44_100); // different rate on purpose
        bt2.restore_from_snapshot(&snap);

        assert_eq!(bt2.get_sample_rate(), bt.get_sample_rate());
        assert_eq!(bt2.get_position(), bt.get_position());
        assert_eq!(bt2.get_beat_position(), bt.get_beat_position());
        assert_eq!(bt2.get_current_beat(), bt.get_current_beat());
        assert_eq!(bt2.get_beats_per_bar(), bt.get_beats_per_bar());
        assert!(bt2.is_initialized());
    }
}
