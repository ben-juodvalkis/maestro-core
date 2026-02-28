//! Tempo control and ratio calculation for the audio engine.
//!
//! Manages tempo state, ratio computation, per-track ratio tracking,
//! and pitch compensation for tempo-shifted playback.
//!
//! Ported from:
//! - `audio-engine/managers/tempo-controller.ts` (TempoController class)
//! - `src/config/audio-constants.ts` (tempo bounds)
//!
//! Browser-specific concerns stripped: ConstantSourceNode, AudioParam,
//! AudioContext, Web Audio graph wiring.  The Rust core owns the state;
//! the platform layer (Web Audio / AVAudioSourceNode / Oboe) reads the
//! current ratio and applies it.

use crate::timing::{MAX_TEMPO, MIN_TEMPO};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Tolerance for floating-point BPM comparisons.
const TEMPO_TOLERANCE: f64 = 0.001;

/// Tolerance for floating-point ratio comparisons.
const RATIO_TOLERANCE: f64 = 0.000_001;

/// Minimum pitch-compensation factor (matches worklet parameter min).
pub const MIN_PITCH_COMPENSATION: f64 = 0.25;

/// Maximum pitch-compensation factor (matches worklet parameter max).
pub const MAX_PITCH_COMPENSATION: f64 = 4.0;

/// Default tempo increment for UI buttons (BPM).
pub const TEMPO_INCREMENT: f64 = 3.0;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that can occur during tempo operations.
#[derive(Debug, Clone, PartialEq)]
pub enum TempoError {
    /// Tempo must be positive.
    InvalidTempo(f64),
    /// Tempo ratio must be positive and finite.
    InvalidRatio(f64),
    /// Track not registered for per-track ratio calculation.
    TrackNotRegistered(String),
    /// Tempo not yet initialised (both current and calculated must be set).
    NotInitialized,
}

impl std::fmt::Display for TempoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TempoError::InvalidTempo(v) => {
                write!(f, "Invalid tempo: {v}. Tempo must be positive")
            }
            TempoError::InvalidRatio(v) => {
                write!(f, "Invalid tempo ratio: {v}. Must be positive and finite")
            }
            TempoError::TrackNotRegistered(id) => {
                write!(f, "Track '{id}' not registered for per-track tempo ratio")
            }
            TempoError::NotInitialized => {
                write!(
                    f,
                    "Tempo not initialized — both current and calculated tempo must be set"
                )
            }
        }
    }
}

impl std::error::Error for TempoError {}

// ---------------------------------------------------------------------------
// Per-track tempo ratio entry
// ---------------------------------------------------------------------------

/// Tracks the calculated tempo ratio for an individual track.
#[derive(Debug, Clone, PartialEq)]
pub struct TrackTempoEntry {
    /// Original (metadata) tempo in BPM.
    pub original_tempo: f64,
    /// Calculated ratio = global_tempo / original_tempo.
    pub ratio: f64,
}

// ---------------------------------------------------------------------------
// TempoController
// ---------------------------------------------------------------------------

/// Manages global and per-track tempo state.
///
/// The controller stores:
/// - A *current* (desired) tempo in BPM.
/// - A *calculated* (original / metadata) tempo in BPM.
/// - Per-track original tempos and their derived ratios.
/// - Pitch-compensation state (inverse of tempo ratio).
///
/// The platform layer is responsible for reading [`tempo_ratio`] and
/// [`pitch_compensation_ratio`] and applying them to the audio graph.
///
/// [`tempo_ratio`]: TempoController::tempo_ratio
/// [`pitch_compensation_ratio`]: TempoController::pitch_compensation_ratio
pub struct TempoController {
    /// Desired playback tempo (BPM).  0 means "not yet set".
    current_tempo: f64,
    /// Original (track metadata) tempo (BPM).  0 means "not yet set".
    calculated_tempo: f64,
    /// Per-track original tempos and derived ratios.
    track_entries: Vec<(String, TrackTempoEntry)>,
    /// Whether automatic pitch compensation is enabled.
    pitch_compensation_enabled: bool,
}

impl TempoController {
    /// Create a new, uninitialised controller.
    pub fn new() -> Self {
        Self {
            current_tempo: 0.0,
            calculated_tempo: 0.0,
            track_entries: Vec::new(),
            pitch_compensation_enabled: true,
        }
    }

    /// Create a controller pre-loaded with the given tempos.
    pub fn with_tempos(current_tempo: f64, calculated_tempo: f64) -> Result<Self, TempoError> {
        if current_tempo <= 0.0 || !current_tempo.is_finite() {
            return Err(TempoError::InvalidTempo(current_tempo));
        }
        if calculated_tempo <= 0.0 || !calculated_tempo.is_finite() {
            return Err(TempoError::InvalidTempo(calculated_tempo));
        }
        Ok(Self {
            current_tempo,
            calculated_tempo,
            track_entries: Vec::new(),
            pitch_compensation_enabled: true,
        })
    }

    // -----------------------------------------------------------------------
    // Global tempo getters / setters
    // -----------------------------------------------------------------------

    /// Current (desired) playback tempo in BPM.
    ///
    /// Returns `Err` if the tempo has not been set yet.
    pub fn current_tempo(&self) -> Result<f64, TempoError> {
        if self.current_tempo == 0.0 {
            return Err(TempoError::NotInitialized);
        }
        Ok(self.current_tempo)
    }

    /// Original (track metadata) tempo in BPM.
    ///
    /// Returns `Err` if the calculated tempo has not been set yet.
    pub fn calculated_tempo(&self) -> Result<f64, TempoError> {
        if self.calculated_tempo == 0.0 {
            return Err(TempoError::NotInitialized);
        }
        Ok(self.calculated_tempo)
    }

    /// Set the desired playback tempo in BPM.
    ///
    /// Returns `true` if the tempo actually changed (outside tolerance).
    pub fn set_tempo(&mut self, tempo: f64) -> Result<bool, TempoError> {
        if tempo <= 0.0 || !tempo.is_finite() {
            return Err(TempoError::InvalidTempo(tempo));
        }
        if (self.current_tempo - tempo).abs() < TEMPO_TOLERANCE {
            return Ok(false); // no change
        }
        self.current_tempo = tempo;
        self.update_all_track_ratios();
        Ok(true)
    }

    /// Set the calculated (original / metadata) tempo in BPM.
    pub fn set_calculated_tempo(&mut self, tempo: f64) -> Result<(), TempoError> {
        if tempo <= 0.0 || !tempo.is_finite() {
            return Err(TempoError::InvalidTempo(tempo));
        }
        self.calculated_tempo = tempo;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Ratio
    // -----------------------------------------------------------------------

    /// Global tempo ratio = current_tempo / calculated_tempo.
    ///
    /// A ratio of 1.0 means playback at original speed.
    pub fn tempo_ratio(&self) -> Result<f64, TempoError> {
        if self.current_tempo == 0.0 || self.calculated_tempo == 0.0 {
            return Err(TempoError::NotInitialized);
        }
        Ok(self.current_tempo / self.calculated_tempo)
    }

    /// Set the global tempo via a ratio (desired / original).
    ///
    /// Derives the new BPM as `calculated_tempo * ratio`.
    /// Returns `true` if the tempo actually changed.
    pub fn set_tempo_ratio(&mut self, ratio: f64) -> Result<bool, TempoError> {
        if ratio <= 0.0 || !ratio.is_finite() {
            return Err(TempoError::InvalidRatio(ratio));
        }
        if self.calculated_tempo == 0.0 {
            return Err(TempoError::NotInitialized);
        }
        let new_tempo = self.calculated_tempo * ratio;
        self.set_tempo(new_tempo)
    }

    // -----------------------------------------------------------------------
    // Pitch compensation
    // -----------------------------------------------------------------------

    /// Whether automatic pitch compensation is enabled.
    pub fn pitch_compensation_enabled(&self) -> bool {
        self.pitch_compensation_enabled
    }

    /// Enable or disable automatic pitch compensation.
    pub fn set_pitch_compensation_enabled(&mut self, enabled: bool) {
        self.pitch_compensation_enabled = enabled;
    }

    /// Pitch-compensation ratio (inverse of tempo ratio, clamped).
    ///
    /// When tempo ratio = 1.2x, the compensation ratio = 1/1.2 ≈ 0.833x
    /// so that the perceived pitch stays constant despite the speed change.
    ///
    /// Returns `None` if pitch compensation is disabled or tempo is not set.
    pub fn pitch_compensation_ratio(&self) -> Option<f64> {
        if !self.pitch_compensation_enabled {
            return None;
        }
        let ratio = self.tempo_ratio().ok()?;
        Some(clamp_pitch_compensation(1.0 / ratio))
    }

    // -----------------------------------------------------------------------
    // Per-track tempo ratios
    // -----------------------------------------------------------------------

    /// Register a track with its original tempo.
    pub fn register_track(
        &mut self,
        track_id: impl Into<String>,
        original_tempo: f64,
    ) -> Result<(), TempoError> {
        if original_tempo <= 0.0 || !original_tempo.is_finite() {
            return Err(TempoError::InvalidTempo(original_tempo));
        }
        let id = track_id.into();
        let ratio = if self.current_tempo > 0.0 {
            self.current_tempo / original_tempo
        } else {
            1.0
        };
        // Upsert: replace existing entry or push new one
        if let Some(entry) = self.track_entries.iter_mut().find(|(eid, _)| eid == &id) {
            entry.1 = TrackTempoEntry {
                original_tempo,
                ratio,
            };
        } else {
            self.track_entries.push((
                id,
                TrackTempoEntry {
                    original_tempo,
                    ratio,
                },
            ));
        }
        Ok(())
    }

    /// Unregister a track.
    pub fn unregister_track(&mut self, track_id: &str) {
        self.track_entries.retain(|(id, _)| id != track_id);
    }

    /// Check whether a track is registered.
    pub fn is_track_registered(&self, track_id: &str) -> bool {
        self.track_entries.iter().any(|(id, _)| id == track_id)
    }

    /// Number of registered tracks.
    pub fn registered_track_count(&self) -> usize {
        self.track_entries.len()
    }

    /// Get the tempo ratio for a specific track.
    pub fn track_tempo_ratio(&self, track_id: &str) -> Result<f64, TempoError> {
        self.track_entries
            .iter()
            .find(|(id, _)| id == track_id)
            .map(|(_, entry)| entry.ratio)
            .ok_or_else(|| TempoError::TrackNotRegistered(track_id.to_owned()))
    }

    /// Get all per-track tempo ratios as (track_id, ratio) pairs.
    pub fn all_track_tempo_ratios(&self) -> Vec<(&str, f64)> {
        self.track_entries
            .iter()
            .map(|(id, entry)| (id.as_str(), entry.ratio))
            .collect()
    }

    /// Recalculate per-track ratios after a global tempo change.
    ///
    /// Returns the IDs of tracks whose ratios actually changed.
    fn update_all_track_ratios(&mut self) -> Vec<String> {
        let mut changed = Vec::new();
        if self.current_tempo <= 0.0 {
            return changed;
        }
        for (id, entry) in &mut self.track_entries {
            let new_ratio = self.current_tempo / entry.original_tempo;
            if (entry.ratio - new_ratio).abs() >= RATIO_TOLERANCE {
                entry.ratio = new_ratio;
                changed.push(id.clone());
            }
        }
        changed
    }

    // -----------------------------------------------------------------------
    // Convenience helpers
    // -----------------------------------------------------------------------

    /// Reset to an uninitialised state, clearing all track registrations.
    pub fn reset(&mut self) {
        self.current_tempo = 0.0;
        self.calculated_tempo = 0.0;
        self.track_entries.clear();
    }

    /// Whether both current and calculated tempo have been set.
    pub fn is_initialized(&self) -> bool {
        self.current_tempo > 0.0 && self.calculated_tempo > 0.0
    }
}

impl Default for TempoController {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Pure functions
// ---------------------------------------------------------------------------

/// Calculate tempo from audio parameters.
///
/// Formula: `(60 * beats * sample_rate) / samples`
pub fn calculate_tempo(
    length_in_beats: f64,
    length_in_samples: u64,
    sample_rate: u32,
) -> Result<f64, TempoError> {
    if length_in_beats <= 0.0 || !length_in_beats.is_finite() {
        return Err(TempoError::InvalidTempo(length_in_beats));
    }
    if length_in_samples == 0 {
        return Err(TempoError::InvalidTempo(0.0));
    }
    if sample_rate == 0 {
        return Err(TempoError::InvalidTempo(0.0));
    }
    Ok((60.0 * length_in_beats * sample_rate as f64) / length_in_samples as f64)
}

/// Clamp a BPM value to the allowed range [`MIN_TEMPO`, `MAX_TEMPO`].
pub fn clamp_tempo(bpm: f64) -> f64 {
    bpm.clamp(MIN_TEMPO, MAX_TEMPO)
}

/// Calculate the tempo ratio between desired and original BPM.
///
/// Returns `Err` if `original_bpm` is not positive.
pub fn tempo_ratio(desired_bpm: f64, original_bpm: f64) -> Result<f64, TempoError> {
    if original_bpm <= 0.0 || !original_bpm.is_finite() {
        return Err(TempoError::InvalidTempo(original_bpm));
    }
    Ok(desired_bpm / original_bpm)
}

/// Derive the global (desired) BPM from original BPM and a ratio.
pub fn tempo_from_ratio(original_bpm: f64, ratio: f64) -> f64 {
    original_bpm * ratio
}

/// Clamp a pitch-compensation factor to the allowed range.
pub fn clamp_pitch_compensation(factor: f64) -> f64 {
    factor.clamp(MIN_PITCH_COMPENSATION, MAX_PITCH_COMPENSATION)
}

/// Calculate the pitch-compensation factor for a given tempo ratio.
///
/// Returns the clamped inverse of the ratio so that perceived pitch
/// stays constant when playback speed changes.
pub fn pitch_compensation_for_ratio(tempo_ratio: f64) -> f64 {
    if tempo_ratio <= 0.0 || !tempo_ratio.is_finite() {
        return 1.0; // safe fallback
    }
    clamp_pitch_compensation(1.0 / tempo_ratio)
}

/// Validate that a BPM value is within the allowed range.
pub fn is_valid_tempo(bpm: f64) -> bool {
    bpm.is_finite() && (MIN_TEMPO..=MAX_TEMPO).contains(&bpm)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Pure function tests
    // -----------------------------------------------------------------------

    mod pure_functions {
        use super::*;

        #[test]
        fn calculate_tempo_basic() {
            // 16 beats in 460800 samples at 48kHz = 100 BPM
            let result = calculate_tempo(16.0, 460800, 48000).unwrap();
            assert!((result - 100.0).abs() < 0.001);
        }

        #[test]
        fn calculate_tempo_120bpm() {
            // 120 BPM → 24000 samples/beat at 48kHz
            // 4 beats → 96000 samples
            let result = calculate_tempo(4.0, 96000, 48000).unwrap();
            assert!((result - 120.0).abs() < 0.001);
        }

        #[test]
        fn calculate_tempo_invalid_beats() {
            assert!(calculate_tempo(0.0, 100000, 48000).is_err());
            assert!(calculate_tempo(-1.0, 100000, 48000).is_err());
        }

        #[test]
        fn calculate_tempo_invalid_samples() {
            assert!(calculate_tempo(16.0, 0, 48000).is_err());
        }

        #[test]
        fn calculate_tempo_invalid_sample_rate() {
            assert!(calculate_tempo(16.0, 100000, 0).is_err());
        }

        #[test]
        fn clamp_tempo_within_range() {
            assert_eq!(clamp_tempo(100.0), 100.0);
            assert_eq!(clamp_tempo(50.0), 50.0);
            assert_eq!(clamp_tempo(150.0), 150.0);
        }

        #[test]
        fn clamp_tempo_below_min() {
            assert_eq!(clamp_tempo(30.0), MIN_TEMPO);
        }

        #[test]
        fn clamp_tempo_above_max() {
            assert_eq!(clamp_tempo(200.0), MAX_TEMPO);
        }

        #[test]
        fn tempo_ratio_basic() {
            let r = tempo_ratio(120.0, 100.0).unwrap();
            assert!((r - 1.2).abs() < 0.0001);
        }

        #[test]
        fn tempo_ratio_unity() {
            let r = tempo_ratio(100.0, 100.0).unwrap();
            assert!((r - 1.0).abs() < 0.0001);
        }

        #[test]
        fn tempo_ratio_slower() {
            let r = tempo_ratio(80.0, 100.0).unwrap();
            assert!((r - 0.8).abs() < 0.0001);
        }

        #[test]
        fn tempo_ratio_invalid_original() {
            assert!(tempo_ratio(100.0, 0.0).is_err());
            assert!(tempo_ratio(100.0, -50.0).is_err());
        }

        #[test]
        fn tempo_from_ratio_basic() {
            assert!((tempo_from_ratio(100.0, 1.2) - 120.0).abs() < 0.001);
            assert!((tempo_from_ratio(100.0, 0.8) - 80.0).abs() < 0.001);
            assert!((tempo_from_ratio(128.0, 1.0) - 128.0).abs() < 0.001);
        }

        #[test]
        fn pitch_compensation_unity() {
            let c = pitch_compensation_for_ratio(1.0);
            assert!((c - 1.0).abs() < 0.0001);
        }

        #[test]
        fn pitch_compensation_faster() {
            // 1.2x tempo → 1/1.2 ≈ 0.833 compensation
            let c = pitch_compensation_for_ratio(1.2);
            assert!((c - (1.0 / 1.2)).abs() < 0.0001);
        }

        #[test]
        fn pitch_compensation_slower() {
            // 0.8x tempo → 1/0.8 = 1.25 compensation
            let c = pitch_compensation_for_ratio(0.8);
            assert!((c - 1.25).abs() < 0.0001);
        }

        #[test]
        fn pitch_compensation_clamped_low() {
            // Very fast tempo → compensation clamped to MIN
            let c = pitch_compensation_for_ratio(10.0);
            assert!((c - MIN_PITCH_COMPENSATION).abs() < 0.0001);
        }

        #[test]
        fn pitch_compensation_clamped_high() {
            // Very slow tempo → compensation clamped to MAX
            let c = pitch_compensation_for_ratio(0.1);
            assert!((c - MAX_PITCH_COMPENSATION).abs() < 0.0001);
        }

        #[test]
        fn pitch_compensation_invalid_ratio() {
            assert!((pitch_compensation_for_ratio(0.0) - 1.0).abs() < 0.0001);
            assert!((pitch_compensation_for_ratio(-1.0) - 1.0).abs() < 0.0001);
            assert!((pitch_compensation_for_ratio(f64::NAN) - 1.0).abs() < 0.0001);
        }

        #[test]
        fn clamp_pitch_compensation_range() {
            assert_eq!(clamp_pitch_compensation(0.1), MIN_PITCH_COMPENSATION);
            assert_eq!(clamp_pitch_compensation(5.0), MAX_PITCH_COMPENSATION);
            assert!((clamp_pitch_compensation(1.0) - 1.0).abs() < 0.0001);
        }

        #[test]
        fn is_valid_tempo_in_range() {
            assert!(is_valid_tempo(50.0));
            assert!(is_valid_tempo(100.0));
            assert!(is_valid_tempo(150.0));
        }

        #[test]
        fn is_valid_tempo_out_of_range() {
            assert!(!is_valid_tempo(49.9));
            assert!(!is_valid_tempo(150.1));
            assert!(!is_valid_tempo(f64::NAN));
            assert!(!is_valid_tempo(f64::INFINITY));
        }
    }

    // -----------------------------------------------------------------------
    // TempoController tests
    // -----------------------------------------------------------------------

    mod controller {
        use super::*;

        #[test]
        fn new_is_uninitialised() {
            let tc = TempoController::new();
            assert!(!tc.is_initialized());
            assert!(tc.current_tempo().is_err());
            assert!(tc.calculated_tempo().is_err());
            assert!(tc.tempo_ratio().is_err());
        }

        #[test]
        fn with_tempos_initialises() {
            let tc = TempoController::with_tempos(120.0, 100.0).unwrap();
            assert!(tc.is_initialized());
            assert!((tc.current_tempo().unwrap() - 120.0).abs() < 0.001);
            assert!((tc.calculated_tempo().unwrap() - 100.0).abs() < 0.001);
            assert!((tc.tempo_ratio().unwrap() - 1.2).abs() < 0.0001);
        }

        #[test]
        fn with_tempos_invalid() {
            assert!(TempoController::with_tempos(0.0, 100.0).is_err());
            assert!(TempoController::with_tempos(100.0, -1.0).is_err());
            assert!(TempoController::with_tempos(f64::NAN, 100.0).is_err());
        }

        #[test]
        fn set_tempo_updates_ratio() {
            let mut tc = TempoController::with_tempos(100.0, 100.0).unwrap();
            assert!((tc.tempo_ratio().unwrap() - 1.0).abs() < 0.0001);

            let changed = tc.set_tempo(120.0).unwrap();
            assert!(changed);
            assert!((tc.tempo_ratio().unwrap() - 1.2).abs() < 0.0001);
        }

        #[test]
        fn set_tempo_no_change_within_tolerance() {
            let mut tc = TempoController::with_tempos(100.0, 100.0).unwrap();
            let changed = tc.set_tempo(100.0005).unwrap();
            assert!(!changed);
        }

        #[test]
        fn set_tempo_invalid() {
            let mut tc = TempoController::with_tempos(100.0, 100.0).unwrap();
            assert!(tc.set_tempo(0.0).is_err());
            assert!(tc.set_tempo(-10.0).is_err());
            assert!(tc.set_tempo(f64::NAN).is_err());
        }

        #[test]
        fn set_calculated_tempo() {
            let mut tc = TempoController::new();
            tc.set_calculated_tempo(128.0).unwrap();
            assert!((tc.calculated_tempo().unwrap() - 128.0).abs() < 0.001);
        }

        #[test]
        fn set_calculated_tempo_invalid() {
            let mut tc = TempoController::new();
            assert!(tc.set_calculated_tempo(0.0).is_err());
            assert!(tc.set_calculated_tempo(-5.0).is_err());
        }

        #[test]
        fn set_tempo_ratio() {
            let mut tc = TempoController::with_tempos(100.0, 100.0).unwrap();
            let changed = tc.set_tempo_ratio(1.5).unwrap();
            assert!(changed);
            assert!((tc.current_tempo().unwrap() - 150.0).abs() < 0.001);
            assert!((tc.tempo_ratio().unwrap() - 1.5).abs() < 0.0001);
        }

        #[test]
        fn set_tempo_ratio_no_change() {
            let mut tc = TempoController::with_tempos(100.0, 100.0).unwrap();
            let changed = tc.set_tempo_ratio(1.0).unwrap();
            assert!(!changed);
        }

        #[test]
        fn set_tempo_ratio_invalid() {
            let mut tc = TempoController::with_tempos(100.0, 100.0).unwrap();
            assert!(tc.set_tempo_ratio(0.0).is_err());
            assert!(tc.set_tempo_ratio(-1.0).is_err());
            assert!(tc.set_tempo_ratio(f64::NAN).is_err());
        }

        #[test]
        fn set_tempo_ratio_not_initialized() {
            let mut tc = TempoController::new();
            assert!(tc.set_tempo_ratio(1.0).is_err());
        }

        #[test]
        fn pitch_compensation_ratio_unity() {
            let tc = TempoController::with_tempos(100.0, 100.0).unwrap();
            let comp = tc.pitch_compensation_ratio().unwrap();
            assert!((comp - 1.0).abs() < 0.0001);
        }

        #[test]
        fn pitch_compensation_ratio_faster() {
            let tc = TempoController::with_tempos(120.0, 100.0).unwrap();
            let comp = tc.pitch_compensation_ratio().unwrap();
            // 1/1.2 ≈ 0.8333
            assert!((comp - (1.0 / 1.2)).abs() < 0.0001);
        }

        #[test]
        fn pitch_compensation_disabled() {
            let mut tc = TempoController::with_tempos(120.0, 100.0).unwrap();
            tc.set_pitch_compensation_enabled(false);
            assert!(tc.pitch_compensation_ratio().is_none());
        }

        #[test]
        fn pitch_compensation_toggle() {
            let mut tc = TempoController::with_tempos(120.0, 100.0).unwrap();
            assert!(tc.pitch_compensation_enabled());
            tc.set_pitch_compensation_enabled(false);
            assert!(!tc.pitch_compensation_enabled());
            tc.set_pitch_compensation_enabled(true);
            assert!(tc.pitch_compensation_enabled());
            assert!(tc.pitch_compensation_ratio().is_some());
        }

        #[test]
        fn reset_clears_state() {
            let mut tc = TempoController::with_tempos(120.0, 100.0).unwrap();
            tc.register_track("t1", 100.0).unwrap();
            tc.reset();
            assert!(!tc.is_initialized());
            assert_eq!(tc.registered_track_count(), 0);
        }
    }

    // -----------------------------------------------------------------------
    // Per-track ratio tests
    // -----------------------------------------------------------------------

    mod per_track {
        use super::*;

        #[test]
        fn register_and_query_track() {
            let mut tc = TempoController::with_tempos(120.0, 100.0).unwrap();
            tc.register_track("drums", 100.0).unwrap();
            let ratio = tc.track_tempo_ratio("drums").unwrap();
            assert!((ratio - 1.2).abs() < 0.0001);
        }

        #[test]
        fn register_track_different_original() {
            let mut tc = TempoController::with_tempos(120.0, 100.0).unwrap();
            // A track originally at 128 BPM, played at 120 BPM
            tc.register_track("bass", 128.0).unwrap();
            let ratio = tc.track_tempo_ratio("bass").unwrap();
            assert!((ratio - (120.0 / 128.0)).abs() < 0.0001);
        }

        #[test]
        fn register_track_invalid_tempo() {
            let mut tc = TempoController::with_tempos(120.0, 100.0).unwrap();
            assert!(tc.register_track("bad", 0.0).is_err());
            assert!(tc.register_track("bad", -50.0).is_err());
        }

        #[test]
        fn unregister_track() {
            let mut tc = TempoController::with_tempos(120.0, 100.0).unwrap();
            tc.register_track("t1", 100.0).unwrap();
            assert!(tc.is_track_registered("t1"));
            tc.unregister_track("t1");
            assert!(!tc.is_track_registered("t1"));
            assert!(tc.track_tempo_ratio("t1").is_err());
        }

        #[test]
        fn query_unregistered_track() {
            let tc = TempoController::with_tempos(120.0, 100.0).unwrap();
            assert!(tc.track_tempo_ratio("nope").is_err());
        }

        #[test]
        fn global_tempo_change_updates_tracks() {
            let mut tc = TempoController::with_tempos(100.0, 100.0).unwrap();
            tc.register_track("t1", 100.0).unwrap();
            tc.register_track("t2", 128.0).unwrap();

            // Change global tempo to 120 BPM
            tc.set_tempo(120.0).unwrap();

            let r1 = tc.track_tempo_ratio("t1").unwrap();
            let r2 = tc.track_tempo_ratio("t2").unwrap();
            assert!((r1 - 1.2).abs() < 0.0001);
            assert!((r2 - (120.0 / 128.0)).abs() < 0.0001);
        }

        #[test]
        fn all_track_ratios() {
            let mut tc = TempoController::with_tempos(120.0, 100.0).unwrap();
            tc.register_track("a", 100.0).unwrap();
            tc.register_track("b", 120.0).unwrap();
            let all = tc.all_track_tempo_ratios();
            assert_eq!(all.len(), 2);
        }

        #[test]
        fn registered_track_count() {
            let mut tc = TempoController::new();
            assert_eq!(tc.registered_track_count(), 0);
            tc.register_track("a", 100.0).unwrap();
            assert_eq!(tc.registered_track_count(), 1);
            tc.register_track("b", 120.0).unwrap();
            assert_eq!(tc.registered_track_count(), 2);
            tc.unregister_track("a");
            assert_eq!(tc.registered_track_count(), 1);
        }

        #[test]
        fn register_track_before_tempo_set() {
            let mut tc = TempoController::new();
            tc.register_track("early", 100.0).unwrap();
            // Ratio defaults to 1.0 when global tempo not yet set
            let r = tc.track_tempo_ratio("early").unwrap();
            assert!((r - 1.0).abs() < 0.0001);
        }

        #[test]
        fn re_register_track_updates_entry() {
            let mut tc = TempoController::with_tempos(120.0, 100.0).unwrap();
            tc.register_track("t1", 100.0).unwrap();
            assert!((tc.track_tempo_ratio("t1").unwrap() - 1.2).abs() < 0.0001);

            // Re-register with different original tempo
            tc.register_track("t1", 120.0).unwrap();
            assert!((tc.track_tempo_ratio("t1").unwrap() - 1.0).abs() < 0.0001);
            // Should still only have one entry
            assert_eq!(tc.registered_track_count(), 1);
        }

        #[test]
        fn set_tempo_ratio_updates_tracks() {
            let mut tc = TempoController::with_tempos(100.0, 100.0).unwrap();
            tc.register_track("t1", 100.0).unwrap();

            tc.set_tempo_ratio(1.5).unwrap();
            let r = tc.track_tempo_ratio("t1").unwrap();
            assert!((r - 1.5).abs() < 0.0001);
        }
    }

    // -----------------------------------------------------------------------
    // Default trait
    // -----------------------------------------------------------------------

    #[test]
    fn default_is_new() {
        let tc = TempoController::default();
        assert!(!tc.is_initialized());
    }
}
