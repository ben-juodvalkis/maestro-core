//! Volume mixing, gain control, and level metering for the audio engine.
//!
//! The `Mixer` manages per-track and master volume state, mute/solo, pan,
//! and provides RMS/peak level metering from PCM sample buffers.
//!
//! Ported from:
//! - `src/domains/infrastructure/services/implementations/VolumeControlServiceImpl.ts`
//! - `audio-engine/managers/audio-node-manager.ts` (gain, RMS, peak)
//! - `src/hooks/audio/useSimpleLevelMeters.ts` (peak detection)
//!
//! Browser-specific concerns stripped: GainNode, AnalyserNode, AudioContext,
//! Web Audio graph wiring, Playwright test-mode muting, event bus emissions,
//! service lifecycle (doInit/doStart/doStop/doDestroy), setTimeout-based fading.
//! The Rust core owns the numeric state; the platform layer applies it to the
//! audio graph (Web Audio GainNode / AVAudioMixerNode / Oboe gain).

use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default master volume.
pub const DEFAULT_MASTER_VOLUME: f64 = 1.0;

/// Default per-track volume (matches TypeScript DEFAULT_TRACK_CONTROLS).
pub const DEFAULT_TRACK_VOLUME: f64 = 0.75;

/// Minimum volume (silence).
pub const MIN_VOLUME: f64 = 0.0;

/// Maximum volume.
pub const MAX_VOLUME: f64 = 1.0;

/// Minimum pan value (full left).
pub const MIN_PAN: f64 = -1.0;

/// Maximum pan value (full right).
pub const MAX_PAN: f64 = 1.0;

/// Default pan (center).
pub const DEFAULT_PAN: f64 = 0.0;

/// Maximum number of volume change history entries.
const MAX_HISTORY_SIZE: usize = 100;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that can occur during mixer operations.
#[derive(Debug, Clone, PartialEq)]
pub enum MixerError {
    /// Track not registered in the mixer.
    TrackNotRegistered(String),
    /// Invalid volume value (NaN or infinite).
    InvalidVolume(f64),
    /// Invalid pan value (NaN or infinite).
    InvalidPan(f64),
    /// Empty sample buffer passed to level metering.
    EmptyBuffer,
}

impl std::fmt::Display for MixerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MixerError::TrackNotRegistered(id) => {
                write!(f, "Track '{id}' not registered in mixer")
            }
            MixerError::InvalidVolume(v) => {
                write!(f, "Invalid volume: {v}. Must be finite")
            }
            MixerError::InvalidPan(v) => {
                write!(f, "Invalid pan: {v}. Must be finite")
            }
            MixerError::EmptyBuffer => {
                write!(f, "Empty sample buffer — cannot compute level")
            }
        }
    }
}

impl std::error::Error for MixerError {}

// ---------------------------------------------------------------------------
// Volume change history
// ---------------------------------------------------------------------------

/// Type of volume change event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VolumeChangeType {
    /// Per-track volume change.
    Track,
    /// Master volume change.
    Master,
    /// Track was muted.
    Mute,
    /// Track was unmuted.
    Unmute,
}

impl std::fmt::Display for VolumeChangeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VolumeChangeType::Track => write!(f, "track"),
            VolumeChangeType::Master => write!(f, "master"),
            VolumeChangeType::Mute => write!(f, "mute"),
            VolumeChangeType::Unmute => write!(f, "unmute"),
        }
    }
}

/// A recorded volume change event.
#[derive(Debug, Clone, PartialEq)]
pub struct VolumeChange {
    /// Track ID (None for master changes).
    pub track_id: Option<String>,
    /// Type of change.
    pub change_type: VolumeChangeType,
    /// Volume before the change.
    pub previous_value: f64,
    /// Volume after the change.
    pub new_value: f64,
}

// ---------------------------------------------------------------------------
// Per-track mixer state
// ---------------------------------------------------------------------------

/// Mixer state for a single track.
#[derive(Debug, Clone, PartialEq)]
pub struct TrackMixerState {
    /// Volume level (0.0–1.0).
    pub volume: f64,
    /// Pan position (−1.0 left, 0.0 center, +1.0 right).
    pub pan: f64,
    /// Whether the track is muted.
    pub muted: bool,
    /// Whether the track is soloed.
    pub solo: bool,
}

impl Default for TrackMixerState {
    fn default() -> Self {
        Self {
            volume: DEFAULT_TRACK_VOLUME,
            pan: DEFAULT_PAN,
            muted: false,
            solo: false,
        }
    }
}

impl TrackMixerState {
    /// Effective gain for this track accounting for mute state.
    ///
    /// Returns 0.0 if muted, otherwise the stored volume.
    pub fn effective_gain(&self) -> f64 {
        if self.muted {
            0.0
        } else {
            self.volume
        }
    }
}

// ---------------------------------------------------------------------------
// Mixer
// ---------------------------------------------------------------------------

/// Central volume/gain mixer for the audio engine.
///
/// Manages per-track volume, pan, mute/solo state, and master volume.
/// Provides level-metering pure functions for computing RMS and peak
/// from raw PCM sample buffers.
///
/// Solo logic: when any track is soloed, only soloed tracks produce output.
/// This matches standard DAW behavior.
pub struct Mixer {
    /// Master volume (0.0–1.0).
    master_volume: f64,
    /// Per-track mixer state keyed by track ID.
    tracks: HashMap<String, TrackMixerState>,
    /// IDs of currently soloed tracks (subset of registered tracks).
    soloed_tracks: HashSet<String>,
    /// Rolling history of volume changes (capped at `MAX_HISTORY_SIZE`).
    history: Vec<VolumeChange>,
}

impl Mixer {
    /// Create a new mixer with default master volume and no tracks.
    pub fn new() -> Self {
        Self {
            master_volume: DEFAULT_MASTER_VOLUME,
            tracks: HashMap::new(),
            soloed_tracks: HashSet::new(),
            history: Vec::new(),
        }
    }

    // -----------------------------------------------------------------------
    // Track registration
    // -----------------------------------------------------------------------

    /// Register a track in the mixer with default state.
    ///
    /// If already registered this is a no-op (existing state preserved).
    pub fn register_track(&mut self, track_id: impl Into<String>) {
        let id = track_id.into();
        self.tracks.entry(id).or_default();
    }

    /// Register a track with a specific initial volume.
    pub fn register_track_with_volume(
        &mut self,
        track_id: impl Into<String>,
        volume: f64,
    ) {
        let id = track_id.into();
        self.tracks.entry(id).or_insert_with(|| TrackMixerState {
            volume: clamp_volume(volume),
            ..Default::default()
        });
    }

    /// Unregister a track, removing all its mixer state.
    pub fn unregister_track(&mut self, track_id: &str) {
        self.tracks.remove(track_id);
        self.soloed_tracks.remove(track_id);
    }

    /// Whether a track is registered.
    pub fn is_track_registered(&self, track_id: &str) -> bool {
        self.tracks.contains_key(track_id)
    }

    /// Number of registered tracks.
    pub fn registered_track_count(&self) -> usize {
        self.tracks.len()
    }

    // -----------------------------------------------------------------------
    // Master volume
    // -----------------------------------------------------------------------

    /// Get master volume (0.0–1.0).
    pub fn master_volume(&self) -> f64 {
        self.master_volume
    }

    /// Set master volume (clamped to 0.0–1.0).
    ///
    /// Returns `true` if the value actually changed.
    pub fn set_master_volume(&mut self, volume: f64) -> Result<bool, MixerError> {
        if !volume.is_finite() {
            return Err(MixerError::InvalidVolume(volume));
        }
        let clamped = clamp_volume(volume);
        if (self.master_volume - clamped).abs() < f64::EPSILON {
            return Ok(false);
        }
        let previous = self.master_volume;
        self.master_volume = clamped;
        self.record_change(VolumeChange {
            track_id: None,
            change_type: VolumeChangeType::Master,
            previous_value: previous,
            new_value: clamped,
        });
        Ok(true)
    }

    // -----------------------------------------------------------------------
    // Per-track volume
    // -----------------------------------------------------------------------

    /// Get a track's stored volume (0.0–1.0).
    pub fn track_volume(&self, track_id: &str) -> Result<f64, MixerError> {
        self.tracks
            .get(track_id)
            .map(|s| s.volume)
            .ok_or_else(|| MixerError::TrackNotRegistered(track_id.to_owned()))
    }

    /// Set a track's volume (clamped to 0.0–1.0).
    ///
    /// Returns `true` if the value actually changed.
    pub fn set_track_volume(
        &mut self,
        track_id: &str,
        volume: f64,
    ) -> Result<bool, MixerError> {
        if !volume.is_finite() {
            return Err(MixerError::InvalidVolume(volume));
        }
        let state = self
            .tracks
            .get_mut(track_id)
            .ok_or_else(|| MixerError::TrackNotRegistered(track_id.to_owned()))?;
        let clamped = clamp_volume(volume);
        if (state.volume - clamped).abs() < f64::EPSILON {
            return Ok(false);
        }
        let previous = state.volume;
        state.volume = clamped;
        self.record_change(VolumeChange {
            track_id: Some(track_id.to_owned()),
            change_type: VolumeChangeType::Track,
            previous_value: previous,
            new_value: clamped,
        });
        Ok(true)
    }

    /// Get all track volumes as (track_id, volume) pairs.
    pub fn all_track_volumes(&self) -> Vec<(&str, f64)> {
        self.tracks
            .iter()
            .map(|(id, s)| (id.as_str(), s.volume))
            .collect()
    }

    // -----------------------------------------------------------------------
    // Mute / unmute
    // -----------------------------------------------------------------------

    /// Mute a track.  Idempotent — no-op if already muted.
    pub fn mute_track(&mut self, track_id: &str) -> Result<(), MixerError> {
        let state = self
            .tracks
            .get_mut(track_id)
            .ok_or_else(|| MixerError::TrackNotRegistered(track_id.to_owned()))?;
        if state.muted {
            return Ok(());
        }
        let prev = state.volume;
        state.muted = true;
        self.record_change(VolumeChange {
            track_id: Some(track_id.to_owned()),
            change_type: VolumeChangeType::Mute,
            previous_value: prev,
            new_value: 0.0,
        });
        Ok(())
    }

    /// Unmute a track.  Idempotent — no-op if not muted.
    pub fn unmute_track(&mut self, track_id: &str) -> Result<(), MixerError> {
        let state = self
            .tracks
            .get_mut(track_id)
            .ok_or_else(|| MixerError::TrackNotRegistered(track_id.to_owned()))?;
        if !state.muted {
            return Ok(());
        }
        let vol = state.volume;
        state.muted = false;
        self.record_change(VolumeChange {
            track_id: Some(track_id.to_owned()),
            change_type: VolumeChangeType::Unmute,
            previous_value: 0.0,
            new_value: vol,
        });
        Ok(())
    }

    /// Whether a track is muted.
    pub fn is_track_muted(&self, track_id: &str) -> Result<bool, MixerError> {
        self.tracks
            .get(track_id)
            .map(|s| s.muted)
            .ok_or_else(|| MixerError::TrackNotRegistered(track_id.to_owned()))
    }

    /// Mute all registered tracks.
    pub fn mute_all(&mut self) {
        let ids: Vec<String> = self.tracks.keys().cloned().collect();
        for id in ids {
            let _ = self.mute_track(&id);
        }
    }

    /// Unmute all registered tracks.
    pub fn unmute_all(&mut self) {
        let ids: Vec<String> = self.tracks.keys().cloned().collect();
        for id in ids {
            let _ = self.unmute_track(&id);
        }
    }

    /// Get IDs of all currently muted tracks.
    pub fn muted_track_ids(&self) -> Vec<&str> {
        self.tracks
            .iter()
            .filter(|(_, s)| s.muted)
            .map(|(id, _)| id.as_str())
            .collect()
    }

    // -----------------------------------------------------------------------
    // Solo
    // -----------------------------------------------------------------------

    /// Solo a track.  When any track is soloed, only soloed tracks produce
    /// output.
    pub fn solo_track(&mut self, track_id: &str) -> Result<(), MixerError> {
        if !self.tracks.contains_key(track_id) {
            return Err(MixerError::TrackNotRegistered(track_id.to_owned()));
        }
        let state = self.tracks.get_mut(track_id).unwrap();
        state.solo = true;
        self.soloed_tracks.insert(track_id.to_owned());
        Ok(())
    }

    /// Unsolo a track.
    pub fn unsolo_track(&mut self, track_id: &str) -> Result<(), MixerError> {
        if !self.tracks.contains_key(track_id) {
            return Err(MixerError::TrackNotRegistered(track_id.to_owned()));
        }
        let state = self.tracks.get_mut(track_id).unwrap();
        state.solo = false;
        self.soloed_tracks.remove(track_id);
        Ok(())
    }

    /// Whether a track is soloed.
    pub fn is_track_soloed(&self, track_id: &str) -> Result<bool, MixerError> {
        self.tracks
            .get(track_id)
            .map(|s| s.solo)
            .ok_or_else(|| MixerError::TrackNotRegistered(track_id.to_owned()))
    }

    /// Whether any track is currently soloed.
    pub fn has_soloed_tracks(&self) -> bool {
        !self.soloed_tracks.is_empty()
    }

    /// Unsolo all tracks.
    pub fn unsolo_all(&mut self) {
        for state in self.tracks.values_mut() {
            state.solo = false;
        }
        self.soloed_tracks.clear();
    }

    // -----------------------------------------------------------------------
    // Pan
    // -----------------------------------------------------------------------

    /// Get a track's pan position (−1.0 left, 0.0 center, +1.0 right).
    pub fn track_pan(&self, track_id: &str) -> Result<f64, MixerError> {
        self.tracks
            .get(track_id)
            .map(|s| s.pan)
            .ok_or_else(|| MixerError::TrackNotRegistered(track_id.to_owned()))
    }

    /// Set a track's pan position (clamped to −1.0 to +1.0).
    ///
    /// Returns `true` if the value actually changed.
    pub fn set_track_pan(
        &mut self,
        track_id: &str,
        pan: f64,
    ) -> Result<bool, MixerError> {
        if !pan.is_finite() {
            return Err(MixerError::InvalidPan(pan));
        }
        let state = self
            .tracks
            .get_mut(track_id)
            .ok_or_else(|| MixerError::TrackNotRegistered(track_id.to_owned()))?;
        let clamped = pan.clamp(MIN_PAN, MAX_PAN);
        if (state.pan - clamped).abs() < f64::EPSILON {
            return Ok(false);
        }
        state.pan = clamped;
        Ok(true)
    }

    // -----------------------------------------------------------------------
    // Effective gain (combining volume, mute, solo, master)
    // -----------------------------------------------------------------------

    /// Compute the effective output gain for a track, accounting for
    /// mute, solo, and master volume.
    ///
    /// Solo logic: if any track is soloed and this track is NOT soloed,
    /// its effective gain is 0.
    pub fn effective_track_gain(&self, track_id: &str) -> Result<f64, MixerError> {
        let state = self
            .tracks
            .get(track_id)
            .ok_or_else(|| MixerError::TrackNotRegistered(track_id.to_owned()))?;

        // Muted → silence
        if state.muted {
            return Ok(0.0);
        }

        // Solo logic: if any track is soloed and this one isn't, silence it
        if !self.soloed_tracks.is_empty() && !state.solo {
            return Ok(0.0);
        }

        Ok(state.volume * self.master_volume)
    }

    /// Compute pan gains for stereo output (left, right).
    ///
    /// Uses constant-power (equal-power) panning:
    /// - `left  = cos(θ)` where `θ = pan * π/4 + π/4`
    /// - `right = sin(θ)`
    ///
    /// At center (pan=0): left ≈ 0.707, right ≈ 0.707 (−3 dB each).
    /// At full left (pan=−1): left = 1.0, right = 0.0.
    /// At full right (pan=+1): left = 0.0, right = 1.0.
    pub fn pan_gains(pan: f64) -> (f64, f64) {
        let clamped = pan.clamp(MIN_PAN, MAX_PAN);
        // Map [-1, 1] to [0, π/2]
        let angle = (clamped + 1.0) * std::f64::consts::FRAC_PI_4;
        (angle.cos(), angle.sin())
    }

    // -----------------------------------------------------------------------
    // Track mixer state access
    // -----------------------------------------------------------------------

    /// Get the full mixer state for a track.
    pub fn track_state(&self, track_id: &str) -> Result<&TrackMixerState, MixerError> {
        self.tracks
            .get(track_id)
            .ok_or_else(|| MixerError::TrackNotRegistered(track_id.to_owned()))
    }

    // -----------------------------------------------------------------------
    // Reset
    // -----------------------------------------------------------------------

    /// Reset all volumes to defaults, unmute and unsolo all tracks.
    pub fn reset_all(&mut self) {
        self.master_volume = DEFAULT_MASTER_VOLUME;
        for state in self.tracks.values_mut() {
            state.volume = DEFAULT_TRACK_VOLUME;
            state.pan = DEFAULT_PAN;
            state.muted = false;
            state.solo = false;
        }
        self.soloed_tracks.clear();
    }

    /// Remove all tracks and reset master volume.
    pub fn clear(&mut self) {
        self.master_volume = DEFAULT_MASTER_VOLUME;
        self.tracks.clear();
        self.soloed_tracks.clear();
        self.history.clear();
    }

    // -----------------------------------------------------------------------
    // History
    // -----------------------------------------------------------------------

    /// Get the volume change history.
    pub fn volume_change_history(&self) -> &[VolumeChange] {
        &self.history
    }

    /// Clear the volume change history.
    pub fn clear_history(&mut self) {
        self.history.clear();
    }

    // -----------------------------------------------------------------------
    // Internals
    // -----------------------------------------------------------------------

    fn record_change(&mut self, change: VolumeChange) {
        if self.history.len() >= MAX_HISTORY_SIZE {
            self.history.remove(0);
        }
        self.history.push(change);
    }
}

impl Default for Mixer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Pure functions — level metering
// ---------------------------------------------------------------------------

/// Compute RMS (root mean square) level from a buffer of PCM samples.
///
/// Returns a value in the range [0.0, 1.0] (clamped).
///
/// Ported from `AudioNodeManager.getRMSLevel()`.
pub fn rms_level(samples: &[f32]) -> Result<f64, MixerError> {
    if samples.is_empty() {
        return Err(MixerError::EmptyBuffer);
    }
    let sum: f64 = samples.iter().map(|&s| (s as f64) * (s as f64)).sum();
    let rms = (sum / samples.len() as f64).sqrt();
    Ok(rms.min(1.0))
}

/// Compute peak level from a buffer of PCM samples.
///
/// Returns the maximum absolute sample value, clamped to [0.0, 1.0].
///
/// Ported from `AudioNodeManager.getPeakLevel()`.
pub fn peak_level(samples: &[f32]) -> Result<f64, MixerError> {
    if samples.is_empty() {
        return Err(MixerError::EmptyBuffer);
    }
    let peak: f64 = samples
        .iter()
        .map(|&s| (s as f64).abs())
        .fold(0.0_f64, f64::max);
    Ok(peak.min(1.0))
}

/// Compute peak level by sampling every Nth sample (for performance).
///
/// Matches the TypeScript `useSimpleLevelMeters` pattern that samples
/// every 4th bin for efficiency.
pub fn peak_level_sampled(samples: &[f32], step: usize) -> Result<f64, MixerError> {
    if samples.is_empty() {
        return Err(MixerError::EmptyBuffer);
    }
    let step = step.max(1); // Minimum step of 1
    let peak: f64 = samples
        .iter()
        .step_by(step)
        .map(|&s| (s as f64).abs())
        .fold(0.0_f64, f64::max);
    Ok(peak.min(1.0))
}

/// Convert a linear level (0.0–1.0) to decibels.
///
/// Returns `f64::NEG_INFINITY` for a level of 0.0.
pub fn level_to_db(level: f64) -> f64 {
    if level <= 0.0 {
        f64::NEG_INFINITY
    } else {
        20.0 * level.log10()
    }
}

/// Convert decibels to linear level (0.0–1.0).
///
/// Returns 0.0 for `f64::NEG_INFINITY`.
pub fn db_to_level(db: f64) -> f64 {
    if db == f64::NEG_INFINITY {
        0.0
    } else {
        10.0_f64.powf(db / 20.0)
    }
}

/// Mix multiple mono sample buffers into a single output buffer,
/// applying per-track gain.
///
/// Each entry in `sources` is `(samples, gain)`.
/// The output buffer is summed (not averaged).
pub fn mix_mono(sources: &[(&[f32], f64)], output: &mut [f32]) {
    // Zero the output
    for s in output.iter_mut() {
        *s = 0.0;
    }
    for &(samples, gain) in sources {
        let len = samples.len().min(output.len());
        let g = gain as f32;
        for i in 0..len {
            output[i] += samples[i] * g;
        }
    }
}

/// Compute volume for a fade at a given progress (0.0–1.0) using
/// quadratic ease-out: `1 − (1 − t)²`.
///
/// Matches the TypeScript fade implementation.
pub fn ease_out_quad(progress: f64) -> f64 {
    let t = progress.clamp(0.0, 1.0);
    1.0 - (1.0 - t) * (1.0 - t)
}

/// Compute the interpolated volume for a fade.
///
/// - `start_volume`: volume at the beginning of the fade.
/// - `target_volume`: volume at the end of the fade.
/// - `progress`: 0.0 (start) to 1.0 (end).
///
/// Uses quadratic ease-out for a smooth curve.
pub fn fade_volume(start_volume: f64, target_volume: f64, progress: f64) -> f64 {
    let eased = ease_out_quad(progress);
    start_volume + (target_volume - start_volume) * eased
}

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

/// Clamp a volume value to the valid range [0.0, 1.0].
fn clamp_volume(v: f64) -> f64 {
    v.clamp(MIN_VOLUME, MAX_VOLUME)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Pure functions — level metering
    // -----------------------------------------------------------------------

    mod level_metering {
        use super::*;

        #[test]
        fn rms_silence() {
            let samples = vec![0.0_f32; 256];
            let level = rms_level(&samples).unwrap();
            assert!((level - 0.0).abs() < 0.0001);
        }

        #[test]
        fn rms_full_scale() {
            let samples = vec![1.0_f32; 256];
            let level = rms_level(&samples).unwrap();
            assert!((level - 1.0).abs() < 0.0001);
        }

        #[test]
        fn rms_half_scale() {
            let samples = vec![0.5_f32; 256];
            let level = rms_level(&samples).unwrap();
            assert!((level - 0.5).abs() < 0.0001);
        }

        #[test]
        fn rms_mixed_signal() {
            // Alternating +1 and -1 → RMS = 1.0
            let samples: Vec<f32> = (0..256).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
            let level = rms_level(&samples).unwrap();
            assert!((level - 1.0).abs() < 0.0001);
        }

        #[test]
        fn rms_clamped_above_one() {
            let samples = vec![2.0_f32; 256];
            let level = rms_level(&samples).unwrap();
            assert!((level - 1.0).abs() < 0.0001); // Clamped to 1.0
        }

        #[test]
        fn rms_empty_buffer_error() {
            assert_eq!(rms_level(&[]), Err(MixerError::EmptyBuffer));
        }

        #[test]
        fn peak_silence() {
            let samples = vec![0.0_f32; 256];
            let level = peak_level(&samples).unwrap();
            assert!((level - 0.0).abs() < 0.0001);
        }

        #[test]
        fn peak_full_scale() {
            let mut samples = vec![0.0_f32; 256];
            samples[100] = 1.0;
            let level = peak_level(&samples).unwrap();
            assert!((level - 1.0).abs() < 0.0001);
        }

        #[test]
        fn peak_negative() {
            let mut samples = vec![0.0_f32; 256];
            samples[50] = -0.8;
            let level = peak_level(&samples).unwrap();
            assert!((level - 0.8).abs() < 0.0001);
        }

        #[test]
        fn peak_clamped() {
            let samples = vec![3.0_f32; 256];
            let level = peak_level(&samples).unwrap();
            assert!((level - 1.0).abs() < 0.0001);
        }

        #[test]
        fn peak_empty_buffer_error() {
            assert_eq!(peak_level(&[]), Err(MixerError::EmptyBuffer));
        }

        #[test]
        fn peak_sampled_basic() {
            let mut samples = vec![0.0_f32; 256];
            samples[0] = 0.5; // Will be sampled (step=4, index 0)
            samples[1] = 0.9; // Will be skipped at step=4
            let level = peak_level_sampled(&samples, 4).unwrap();
            assert!((level - 0.5).abs() < 0.0001);
        }

        #[test]
        fn peak_sampled_step_one_equals_full() {
            let samples: Vec<f32> = (0..100).map(|i| (i as f32) / 100.0).collect();
            let full = peak_level(&samples).unwrap();
            let sampled = peak_level_sampled(&samples, 1).unwrap();
            assert!((full - sampled).abs() < 0.0001);
        }

        #[test]
        fn peak_sampled_empty_error() {
            assert_eq!(peak_level_sampled(&[], 4), Err(MixerError::EmptyBuffer));
        }

        #[test]
        fn peak_sampled_zero_step_treated_as_one() {
            let samples = vec![0.5_f32; 10];
            let level = peak_level_sampled(&samples, 0).unwrap();
            assert!((level - 0.5).abs() < 0.0001);
        }
    }

    // -----------------------------------------------------------------------
    // Pure functions — dB conversion
    // -----------------------------------------------------------------------

    mod db_conversion {
        use super::*;

        #[test]
        fn unity_is_zero_db() {
            assert!((level_to_db(1.0) - 0.0).abs() < 0.0001);
        }

        #[test]
        fn half_is_about_minus_6db() {
            let db = level_to_db(0.5);
            assert!((db - (-6.0206)).abs() < 0.01);
        }

        #[test]
        fn silence_is_neg_infinity() {
            assert!(level_to_db(0.0).is_infinite());
            assert!(level_to_db(0.0) < 0.0);
        }

        #[test]
        fn roundtrip() {
            for level in &[0.1, 0.25, 0.5, 0.75, 1.0] {
                let db = level_to_db(*level);
                let back = db_to_level(db);
                assert!((*level - back).abs() < 0.0001);
            }
        }

        #[test]
        fn neg_infinity_to_level() {
            assert!((db_to_level(f64::NEG_INFINITY) - 0.0).abs() < 0.0001);
        }
    }

    // -----------------------------------------------------------------------
    // Pure functions — mixing
    // -----------------------------------------------------------------------

    mod mixing {
        use super::*;

        #[test]
        fn mix_mono_single_source() {
            let source = [0.5_f32, 0.3, -0.2];
            let mut output = [0.0_f32; 3];
            mix_mono(&[(&source, 1.0)], &mut output);
            assert!((output[0] - 0.5).abs() < 0.0001);
            assert!((output[1] - 0.3).abs() < 0.0001);
            assert!((output[2] - (-0.2)).abs() < 0.0001);
        }

        #[test]
        fn mix_mono_with_gain() {
            let source = [1.0_f32; 4];
            let mut output = [0.0_f32; 4];
            mix_mono(&[(&source, 0.5)], &mut output);
            for &s in &output {
                assert!((s - 0.5).abs() < 0.0001);
            }
        }

        #[test]
        fn mix_mono_two_sources() {
            let a = [0.3_f32; 4];
            let b = [0.4_f32; 4];
            let mut output = [0.0_f32; 4];
            mix_mono(&[(&a, 1.0), (&b, 1.0)], &mut output);
            for &s in &output {
                assert!((s - 0.7).abs() < 0.001);
            }
        }

        #[test]
        fn mix_mono_empty_sources() {
            let mut output = [99.0_f32; 4];
            mix_mono(&[], &mut output);
            for &s in &output {
                assert!((s - 0.0).abs() < 0.0001);
            }
        }

        #[test]
        fn mix_mono_mismatched_lengths() {
            let short = [1.0_f32; 2];
            let mut output = [0.0_f32; 4];
            mix_mono(&[(&short, 1.0)], &mut output);
            assert!((output[0] - 1.0).abs() < 0.0001);
            assert!((output[1] - 1.0).abs() < 0.0001);
            assert!((output[2] - 0.0).abs() < 0.0001);
            assert!((output[3] - 0.0).abs() < 0.0001);
        }
    }

    // -----------------------------------------------------------------------
    // Pure functions — fade
    // -----------------------------------------------------------------------

    mod fade {
        use super::*;

        #[test]
        fn ease_out_quad_at_zero() {
            assert!((ease_out_quad(0.0) - 0.0).abs() < 0.0001);
        }

        #[test]
        fn ease_out_quad_at_one() {
            assert!((ease_out_quad(1.0) - 1.0).abs() < 0.0001);
        }

        #[test]
        fn ease_out_quad_at_half() {
            // 1 - (1-0.5)^2 = 1 - 0.25 = 0.75
            assert!((ease_out_quad(0.5) - 0.75).abs() < 0.0001);
        }

        #[test]
        fn ease_out_quad_clamped() {
            assert!((ease_out_quad(-0.5) - 0.0).abs() < 0.0001);
            assert!((ease_out_quad(1.5) - 1.0).abs() < 0.0001);
        }

        #[test]
        fn fade_volume_full_range() {
            assert!((fade_volume(0.0, 1.0, 0.0) - 0.0).abs() < 0.0001);
            assert!((fade_volume(0.0, 1.0, 1.0) - 1.0).abs() < 0.0001);
        }

        #[test]
        fn fade_volume_reverse() {
            assert!((fade_volume(1.0, 0.0, 0.0) - 1.0).abs() < 0.0001);
            assert!((fade_volume(1.0, 0.0, 1.0) - 0.0).abs() < 0.0001);
        }

        #[test]
        fn fade_volume_midpoint() {
            // start=0, target=1, progress=0.5 → eased=0.75 → volume=0.75
            let vol = fade_volume(0.0, 1.0, 0.5);
            assert!((vol - 0.75).abs() < 0.0001);
        }
    }

    // -----------------------------------------------------------------------
    // Pure functions — pan gains
    // -----------------------------------------------------------------------

    mod pan_gains {
        use super::*;

        #[test]
        fn center_pan() {
            let (l, r) = Mixer::pan_gains(0.0);
            // At center: cos(π/4) ≈ sin(π/4) ≈ 0.7071
            assert!((l - r).abs() < 0.0001);
            assert!((l - std::f64::consts::FRAC_1_SQRT_2).abs() < 0.0001);
        }

        #[test]
        fn full_left() {
            let (l, r) = Mixer::pan_gains(-1.0);
            assert!((l - 1.0).abs() < 0.0001);
            assert!((r - 0.0).abs() < 0.0001);
        }

        #[test]
        fn full_right() {
            let (l, r) = Mixer::pan_gains(1.0);
            assert!((l - 0.0).abs() < 0.0001);
            assert!((r - 1.0).abs() < 0.0001);
        }

        #[test]
        fn clamped_beyond_range() {
            let (l1, r1) = Mixer::pan_gains(-2.0);
            let (l2, r2) = Mixer::pan_gains(-1.0);
            assert!((l1 - l2).abs() < 0.0001);
            assert!((r1 - r2).abs() < 0.0001);

            let (l3, r3) = Mixer::pan_gains(2.0);
            let (l4, r4) = Mixer::pan_gains(1.0);
            assert!((l3 - l4).abs() < 0.0001);
            assert!((r3 - r4).abs() < 0.0001);
        }

        #[test]
        fn constant_power_sum() {
            // For constant-power panning: L² + R² ≈ 1.0 at all positions
            for i in -10..=10 {
                let pan = i as f64 / 10.0;
                let (l, r) = Mixer::pan_gains(pan);
                let power = l * l + r * r;
                assert!(
                    (power - 1.0).abs() < 0.0001,
                    "Power at pan={pan}: {power}"
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Mixer — registration
    // -----------------------------------------------------------------------

    mod registration {
        use super::*;

        #[test]
        fn register_track_default() {
            let mut m = Mixer::new();
            m.register_track("drums");
            assert!(m.is_track_registered("drums"));
            assert_eq!(m.registered_track_count(), 1);
            let state = m.track_state("drums").unwrap();
            assert!((state.volume - DEFAULT_TRACK_VOLUME).abs() < 0.0001);
            assert!((state.pan - DEFAULT_PAN).abs() < 0.0001);
            assert!(!state.muted);
            assert!(!state.solo);
        }

        #[test]
        fn register_track_with_volume() {
            let mut m = Mixer::new();
            m.register_track_with_volume("drums", 0.5);
            assert!((m.track_volume("drums").unwrap() - 0.5).abs() < 0.0001);
        }

        #[test]
        fn register_same_track_is_noop() {
            let mut m = Mixer::new();
            m.register_track("drums");
            m.set_track_volume("drums", 0.3).unwrap();
            m.register_track("drums"); // Should not reset volume
            assert!((m.track_volume("drums").unwrap() - 0.3).abs() < 0.0001);
        }

        #[test]
        fn unregister_track() {
            let mut m = Mixer::new();
            m.register_track("drums");
            m.unregister_track("drums");
            assert!(!m.is_track_registered("drums"));
            assert!(m.track_volume("drums").is_err());
        }

        #[test]
        fn unregister_soloed_track_removes_from_solo_set() {
            let mut m = Mixer::new();
            m.register_track("drums");
            m.solo_track("drums").unwrap();
            assert!(m.has_soloed_tracks());
            m.unregister_track("drums");
            assert!(!m.has_soloed_tracks());
        }
    }

    // -----------------------------------------------------------------------
    // Mixer — master volume
    // -----------------------------------------------------------------------

    mod master_volume {
        use super::*;

        #[test]
        fn default_is_one() {
            let m = Mixer::new();
            assert!((m.master_volume() - 1.0).abs() < 0.0001);
        }

        #[test]
        fn set_master_volume() {
            let mut m = Mixer::new();
            let changed = m.set_master_volume(0.5).unwrap();
            assert!(changed);
            assert!((m.master_volume() - 0.5).abs() < 0.0001);
        }

        #[test]
        fn set_master_volume_clamped() {
            let mut m = Mixer::new();
            m.set_master_volume(1.5).unwrap();
            assert!((m.master_volume() - 1.0).abs() < 0.0001);
            m.set_master_volume(-0.5).unwrap();
            assert!((m.master_volume() - 0.0).abs() < 0.0001);
        }

        #[test]
        fn set_master_volume_no_change() {
            let mut m = Mixer::new();
            let changed = m.set_master_volume(1.0).unwrap();
            assert!(!changed);
        }

        #[test]
        fn set_master_volume_nan_error() {
            let mut m = Mixer::new();
            assert!(m.set_master_volume(f64::NAN).is_err());
        }

        #[test]
        fn set_master_volume_infinity_error() {
            let mut m = Mixer::new();
            assert!(m.set_master_volume(f64::INFINITY).is_err());
        }
    }

    // -----------------------------------------------------------------------
    // Mixer — track volume
    // -----------------------------------------------------------------------

    mod track_volume {
        use super::*;

        #[test]
        fn default_track_volume() {
            let mut m = Mixer::new();
            m.register_track("drums");
            assert!((m.track_volume("drums").unwrap() - DEFAULT_TRACK_VOLUME).abs() < 0.0001);
        }

        #[test]
        fn set_track_volume() {
            let mut m = Mixer::new();
            m.register_track("drums");
            let changed = m.set_track_volume("drums", 0.5).unwrap();
            assert!(changed);
            assert!((m.track_volume("drums").unwrap() - 0.5).abs() < 0.0001);
        }

        #[test]
        fn set_track_volume_clamped() {
            let mut m = Mixer::new();
            m.register_track("drums");
            m.set_track_volume("drums", 2.0).unwrap();
            assert!((m.track_volume("drums").unwrap() - 1.0).abs() < 0.0001);
        }

        #[test]
        fn set_track_volume_no_change() {
            let mut m = Mixer::new();
            m.register_track_with_volume("drums", 0.5);
            let changed = m.set_track_volume("drums", 0.5).unwrap();
            assert!(!changed);
        }

        #[test]
        fn set_track_volume_unregistered_error() {
            let mut m = Mixer::new();
            assert!(m.set_track_volume("nope", 0.5).is_err());
        }

        #[test]
        fn set_track_volume_nan_error() {
            let mut m = Mixer::new();
            m.register_track("drums");
            assert!(m.set_track_volume("drums", f64::NAN).is_err());
        }

        #[test]
        fn all_track_volumes() {
            let mut m = Mixer::new();
            m.register_track_with_volume("drums", 0.5);
            m.register_track_with_volume("bass", 0.8);
            let vols = m.all_track_volumes();
            assert_eq!(vols.len(), 2);
        }
    }

    // -----------------------------------------------------------------------
    // Mixer — mute / unmute
    // -----------------------------------------------------------------------

    mod mute {
        use super::*;

        #[test]
        fn mute_track() {
            let mut m = Mixer::new();
            m.register_track("drums");
            m.mute_track("drums").unwrap();
            assert!(m.is_track_muted("drums").unwrap());
        }

        #[test]
        fn mute_idempotent() {
            let mut m = Mixer::new();
            m.register_track("drums");
            m.mute_track("drums").unwrap();
            m.mute_track("drums").unwrap(); // No panic, no error
            assert!(m.is_track_muted("drums").unwrap());
        }

        #[test]
        fn unmute_restores_volume() {
            let mut m = Mixer::new();
            m.register_track_with_volume("drums", 0.6);
            m.mute_track("drums").unwrap();
            assert!((m.track_state("drums").unwrap().effective_gain() - 0.0).abs() < 0.0001);
            m.unmute_track("drums").unwrap();
            assert!((m.track_state("drums").unwrap().effective_gain() - 0.6).abs() < 0.0001);
        }

        #[test]
        fn unmute_idempotent() {
            let mut m = Mixer::new();
            m.register_track("drums");
            m.unmute_track("drums").unwrap(); // Already unmuted — no-op
            assert!(!m.is_track_muted("drums").unwrap());
        }

        #[test]
        fn mute_unregistered_error() {
            let mut m = Mixer::new();
            assert!(m.mute_track("nope").is_err());
        }

        #[test]
        fn mute_all_and_unmute_all() {
            let mut m = Mixer::new();
            m.register_track("drums");
            m.register_track("bass");
            m.mute_all();
            assert!(m.is_track_muted("drums").unwrap());
            assert!(m.is_track_muted("bass").unwrap());
            m.unmute_all();
            assert!(!m.is_track_muted("drums").unwrap());
            assert!(!m.is_track_muted("bass").unwrap());
        }

        #[test]
        fn muted_track_ids() {
            let mut m = Mixer::new();
            m.register_track("drums");
            m.register_track("bass");
            m.mute_track("drums").unwrap();
            let muted = m.muted_track_ids();
            assert_eq!(muted.len(), 1);
            assert!(muted.contains(&"drums"));
        }
    }

    // -----------------------------------------------------------------------
    // Mixer — solo
    // -----------------------------------------------------------------------

    mod solo {
        use super::*;

        #[test]
        fn solo_track() {
            let mut m = Mixer::new();
            m.register_track("drums");
            m.register_track("bass");
            m.solo_track("drums").unwrap();
            assert!(m.is_track_soloed("drums").unwrap());
            assert!(!m.is_track_soloed("bass").unwrap());
            assert!(m.has_soloed_tracks());
        }

        #[test]
        fn solo_affects_effective_gain() {
            let mut m = Mixer::new();
            m.register_track_with_volume("drums", 0.8);
            m.register_track_with_volume("bass", 0.6);
            m.solo_track("drums").unwrap();

            // Soloed track has normal gain
            assert!((m.effective_track_gain("drums").unwrap() - 0.8).abs() < 0.0001);
            // Non-soloed track is silenced
            assert!((m.effective_track_gain("bass").unwrap() - 0.0).abs() < 0.0001);
        }

        #[test]
        fn unsolo_restores_gain() {
            let mut m = Mixer::new();
            m.register_track_with_volume("drums", 0.8);
            m.register_track_with_volume("bass", 0.6);
            m.solo_track("drums").unwrap();
            m.unsolo_track("drums").unwrap();

            assert!((m.effective_track_gain("bass").unwrap() - 0.6).abs() < 0.0001);
        }

        #[test]
        fn unsolo_all() {
            let mut m = Mixer::new();
            m.register_track("drums");
            m.register_track("bass");
            m.solo_track("drums").unwrap();
            m.solo_track("bass").unwrap();
            m.unsolo_all();
            assert!(!m.has_soloed_tracks());
        }

        #[test]
        fn solo_unregistered_error() {
            let mut m = Mixer::new();
            assert!(m.solo_track("nope").is_err());
        }
    }

    // -----------------------------------------------------------------------
    // Mixer — pan
    // -----------------------------------------------------------------------

    mod pan {
        use super::*;

        #[test]
        fn default_pan() {
            let mut m = Mixer::new();
            m.register_track("drums");
            assert!((m.track_pan("drums").unwrap() - 0.0).abs() < 0.0001);
        }

        #[test]
        fn set_pan() {
            let mut m = Mixer::new();
            m.register_track("drums");
            let changed = m.set_track_pan("drums", -0.5).unwrap();
            assert!(changed);
            assert!((m.track_pan("drums").unwrap() - (-0.5)).abs() < 0.0001);
        }

        #[test]
        fn set_pan_clamped() {
            let mut m = Mixer::new();
            m.register_track("drums");
            m.set_track_pan("drums", -2.0).unwrap();
            assert!((m.track_pan("drums").unwrap() - (-1.0)).abs() < 0.0001);
            m.set_track_pan("drums", 2.0).unwrap();
            assert!((m.track_pan("drums").unwrap() - 1.0).abs() < 0.0001);
        }

        #[test]
        fn set_pan_no_change() {
            let mut m = Mixer::new();
            m.register_track("drums");
            let changed = m.set_track_pan("drums", 0.0).unwrap();
            assert!(!changed);
        }

        #[test]
        fn set_pan_nan_error() {
            let mut m = Mixer::new();
            m.register_track("drums");
            assert!(m.set_track_pan("drums", f64::NAN).is_err());
        }

        #[test]
        fn set_pan_unregistered_error() {
            let mut m = Mixer::new();
            assert!(m.set_track_pan("nope", 0.0).is_err());
        }
    }

    // -----------------------------------------------------------------------
    // Mixer — effective gain
    // -----------------------------------------------------------------------

    mod effective_gain {
        use super::*;

        #[test]
        fn normal_gain() {
            let mut m = Mixer::new();
            m.register_track_with_volume("drums", 0.8);
            // master=1.0, volume=0.8, not muted, no solos
            assert!((m.effective_track_gain("drums").unwrap() - 0.8).abs() < 0.0001);
        }

        #[test]
        fn gain_with_master() {
            let mut m = Mixer::new();
            m.register_track_with_volume("drums", 0.8);
            m.set_master_volume(0.5).unwrap();
            assert!((m.effective_track_gain("drums").unwrap() - 0.4).abs() < 0.0001);
        }

        #[test]
        fn muted_is_zero() {
            let mut m = Mixer::new();
            m.register_track_with_volume("drums", 0.8);
            m.mute_track("drums").unwrap();
            assert!((m.effective_track_gain("drums").unwrap() - 0.0).abs() < 0.0001);
        }

        #[test]
        fn solo_non_soloed_is_zero() {
            let mut m = Mixer::new();
            m.register_track("drums");
            m.register_track("bass");
            m.solo_track("drums").unwrap();
            assert!((m.effective_track_gain("bass").unwrap() - 0.0).abs() < 0.0001);
        }

        #[test]
        fn muted_takes_precedence_over_solo() {
            let mut m = Mixer::new();
            m.register_track("drums");
            m.solo_track("drums").unwrap();
            m.mute_track("drums").unwrap();
            assert!((m.effective_track_gain("drums").unwrap() - 0.0).abs() < 0.0001);
        }

        #[test]
        fn unregistered_error() {
            let m = Mixer::new();
            assert!(m.effective_track_gain("nope").is_err());
        }
    }

    // -----------------------------------------------------------------------
    // Mixer — history
    // -----------------------------------------------------------------------

    mod history {
        use super::*;

        #[test]
        fn volume_change_recorded() {
            let mut m = Mixer::new();
            m.register_track("drums");
            m.set_track_volume("drums", 0.5).unwrap();
            let h = m.volume_change_history();
            assert_eq!(h.len(), 1);
            assert_eq!(h[0].change_type, VolumeChangeType::Track);
            assert_eq!(h[0].track_id.as_deref(), Some("drums"));
        }

        #[test]
        fn master_change_recorded() {
            let mut m = Mixer::new();
            m.set_master_volume(0.5).unwrap();
            let h = m.volume_change_history();
            assert_eq!(h.len(), 1);
            assert_eq!(h[0].change_type, VolumeChangeType::Master);
            assert!(h[0].track_id.is_none());
        }

        #[test]
        fn mute_unmute_recorded() {
            let mut m = Mixer::new();
            m.register_track("drums");
            m.mute_track("drums").unwrap();
            m.unmute_track("drums").unwrap();
            let h = m.volume_change_history();
            assert_eq!(h.len(), 2);
            assert_eq!(h[0].change_type, VolumeChangeType::Mute);
            assert_eq!(h[1].change_type, VolumeChangeType::Unmute);
        }

        #[test]
        fn history_capped() {
            let mut m = Mixer::new();
            m.register_track("drums");
            for i in 0..(MAX_HISTORY_SIZE + 10) {
                m.set_track_volume("drums", (i as f64 % 100.0) / 100.0).unwrap();
            }
            assert_eq!(m.volume_change_history().len(), MAX_HISTORY_SIZE);
        }

        #[test]
        fn no_change_no_history() {
            let mut m = Mixer::new();
            m.register_track_with_volume("drums", 0.5);
            m.set_track_volume("drums", 0.5).unwrap(); // No change
            assert!(m.volume_change_history().is_empty());
        }

        #[test]
        fn clear_history() {
            let mut m = Mixer::new();
            m.set_master_volume(0.5).unwrap();
            m.clear_history();
            assert!(m.volume_change_history().is_empty());
        }
    }

    // -----------------------------------------------------------------------
    // Mixer — reset and clear
    // -----------------------------------------------------------------------

    mod reset {
        use super::*;

        #[test]
        fn reset_all_restores_defaults() {
            let mut m = Mixer::new();
            m.register_track("drums");
            m.register_track("bass");
            m.set_master_volume(0.3).unwrap();
            m.set_track_volume("drums", 0.1).unwrap();
            m.set_track_pan("drums", -0.5).unwrap();
            m.mute_track("bass").unwrap();
            m.solo_track("drums").unwrap();

            m.reset_all();

            assert!((m.master_volume() - DEFAULT_MASTER_VOLUME).abs() < 0.0001);
            assert!((m.track_volume("drums").unwrap() - DEFAULT_TRACK_VOLUME).abs() < 0.0001);
            assert!((m.track_pan("drums").unwrap() - DEFAULT_PAN).abs() < 0.0001);
            assert!(!m.is_track_muted("drums").unwrap());
            assert!(!m.is_track_muted("bass").unwrap());
            assert!(!m.has_soloed_tracks());
        }

        #[test]
        fn clear_removes_everything() {
            let mut m = Mixer::new();
            m.register_track("drums");
            m.set_master_volume(0.5).unwrap();
            m.clear();
            assert_eq!(m.registered_track_count(), 0);
            assert!((m.master_volume() - DEFAULT_MASTER_VOLUME).abs() < 0.0001);
            assert!(m.volume_change_history().is_empty());
        }
    }

    // -----------------------------------------------------------------------
    // Mixer — default trait
    // -----------------------------------------------------------------------

    #[test]
    fn default_is_new() {
        let m = Mixer::default();
        assert!((m.master_volume() - DEFAULT_MASTER_VOLUME).abs() < 0.0001);
        assert_eq!(m.registered_track_count(), 0);
    }

    // -----------------------------------------------------------------------
    // Error display
    // -----------------------------------------------------------------------

    mod errors {
        use super::*;

        #[test]
        fn track_not_registered_display() {
            let err = MixerError::TrackNotRegistered("drums".into());
            assert_eq!(err.to_string(), "Track 'drums' not registered in mixer");
        }

        #[test]
        fn invalid_volume_display() {
            let err = MixerError::InvalidVolume(f64::NAN);
            assert!(err.to_string().contains("Invalid volume"));
        }

        #[test]
        fn invalid_pan_display() {
            let err = MixerError::InvalidPan(f64::INFINITY);
            assert!(err.to_string().contains("Invalid pan"));
        }

        #[test]
        fn empty_buffer_display() {
            let err = MixerError::EmptyBuffer;
            assert_eq!(err.to_string(), "Empty sample buffer — cannot compute level");
        }

        #[test]
        fn error_is_std_error() {
            let err: Box<dyn std::error::Error> = Box::new(MixerError::EmptyBuffer);
            assert!(!err.to_string().is_empty());
        }
    }

    // -----------------------------------------------------------------------
    // VolumeChangeType display
    // -----------------------------------------------------------------------

    #[test]
    fn volume_change_type_display() {
        assert_eq!(VolumeChangeType::Track.to_string(), "track");
        assert_eq!(VolumeChangeType::Master.to_string(), "master");
        assert_eq!(VolumeChangeType::Mute.to_string(), "mute");
        assert_eq!(VolumeChangeType::Unmute.to_string(), "unmute");
    }
}
