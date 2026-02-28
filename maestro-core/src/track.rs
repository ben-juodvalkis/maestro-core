//! Track data model — state machine, metadata, and per-track controls.
//!
//! The `Track` struct is the core representation of an audio track in the
//! engine.  It owns the track's lifecycle state (idle → loading → playing …),
//! metadata (name, tempo, duration), and user-controllable parameters
//! (volume, pitch, pan, armed state).
//!
//! Ported from:
//! - `audio-engine/core/track.ts` (Track class, TrackState, TrackMetadata)
//! - `src/domains/core/types/track-redux.ts` (TrackControls, DEFAULT_TRACK_CONTROLS)
//!
//! Browser-specific concerns stripped: AnalyserNode, ConstantSourceNode,
//! Zustand store access (getCurrentBeat, getBeatState), AudioBuffer.
//! The engine (Step 7) will own beat state queries and the platform layer
//! handles audio nodes.

use crate::pitch;
use crate::timing::{self, DEFAULT_SAMPLE_RATE};

// ---------------------------------------------------------------------------
// Constants (matching TypeScript defaults)
// ---------------------------------------------------------------------------

/// Default track volume (0.0–1.0).
pub const DEFAULT_VOLUME: f64 = 0.75;

/// Default pitch offset in cents.
pub const DEFAULT_PITCH_CENTS: f64 = 0.0;

/// Default pan position (-1.0 left, 0.0 center, 1.0 right).
pub const DEFAULT_PAN: f64 = 0.0;

/// Minimum pitch offset in cents (−1 octave).
pub const MIN_PITCH_CENTS: f64 = -1200.0;

/// Maximum pitch offset in cents (+1 octave).
pub const MAX_PITCH_CENTS: f64 = 1200.0;

// ---------------------------------------------------------------------------
// Track status (state machine states)
// ---------------------------------------------------------------------------

/// Lifecycle status of a track.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackStatus {
    /// Initial state — no audio loaded.
    Idle,
    /// Audio is being fetched / decoded.
    Loading,
    /// Audio is decoded and ready for playback.
    Ready,
    /// Audio is decoded and pre-created (worklet staged).
    Preloaded,
    /// Currently playing audio.
    Playing,
    /// Playback is paused.
    Paused,
    /// An error occurred.
    Error,
}

impl TrackStatus {
    /// Valid transitions from this status.
    fn valid_transitions(self) -> &'static [TrackStatus] {
        match self {
            TrackStatus::Idle => &[TrackStatus::Loading, TrackStatus::Preloaded, TrackStatus::Error],
            TrackStatus::Loading => {
                &[TrackStatus::Ready, TrackStatus::Preloaded, TrackStatus::Error]
            }
            TrackStatus::Ready => &[
                TrackStatus::Playing,
                TrackStatus::Loading,
                TrackStatus::Preloaded,
                TrackStatus::Error,
            ],
            TrackStatus::Preloaded => {
                &[TrackStatus::Playing, TrackStatus::Ready, TrackStatus::Error]
            }
            TrackStatus::Playing => {
                &[TrackStatus::Paused, TrackStatus::Ready, TrackStatus::Error]
            }
            TrackStatus::Paused => {
                &[TrackStatus::Playing, TrackStatus::Ready, TrackStatus::Error]
            }
            TrackStatus::Error => &[TrackStatus::Idle, TrackStatus::Loading],
        }
    }

    /// Whether the transition from `self` to `to` is valid.
    pub fn can_transition_to(self, to: TrackStatus) -> bool {
        self == to || self.valid_transitions().contains(&to)
    }
}

impl std::fmt::Display for TrackStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TrackStatus::Idle => write!(f, "idle"),
            TrackStatus::Loading => write!(f, "loading"),
            TrackStatus::Ready => write!(f, "ready"),
            TrackStatus::Preloaded => write!(f, "preloaded"),
            TrackStatus::Playing => write!(f, "playing"),
            TrackStatus::Paused => write!(f, "paused"),
            TrackStatus::Error => write!(f, "error"),
        }
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that can occur during track operations.
#[derive(Debug, Clone, PartialEq)]
pub enum TrackError {
    /// Invalid state machine transition.
    InvalidTransition {
        from: TrackStatus,
        to: TrackStatus,
    },
    /// Position must be non-negative.
    NegativePosition(f64),
    /// Position exceeds duration.
    PositionExceedsDuration {
        position: f64,
        duration: f64,
    },
    /// Duration must be non-negative.
    NegativeDuration(f64),
    /// Tempo must be positive.
    InvalidTempo(f64),
}

impl std::fmt::Display for TrackError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TrackError::InvalidTransition { from, to } => {
                write!(f, "Invalid state transition from '{from}' to '{to}'")
            }
            TrackError::NegativePosition(pos) => {
                write!(f, "Position cannot be negative: {pos}")
            }
            TrackError::PositionExceedsDuration { position, duration } => {
                write!(f, "Position {position} exceeds duration {duration}")
            }
            TrackError::NegativeDuration(dur) => {
                write!(f, "Duration cannot be negative: {dur}")
            }
            TrackError::InvalidTempo(tempo) => {
                write!(f, "Tempo must be positive: {tempo}")
            }
        }
    }
}

impl std::error::Error for TrackError {}

// ---------------------------------------------------------------------------
// Variation URLs
// ---------------------------------------------------------------------------

/// URLs for the three variation levels of a stem.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct VariationUrls {
    pub simple: Option<String>,
    pub default: Option<String>,
    pub complex: Option<String>,
}

// ---------------------------------------------------------------------------
// Track metadata
// ---------------------------------------------------------------------------

/// Descriptive metadata for a track.
#[derive(Debug, Clone, Default)]
pub struct TrackMetadata {
    /// Human-readable track name (e.g., "Drums").
    pub name: Option<String>,
    /// Artist name.
    pub artist: Option<String>,
    /// Duration in seconds (may differ from computed length).
    pub duration: Option<f64>,
    /// Rhythmic/tonal scale: 1 = rhythmic, 5 = tonal.
    pub rhythmic_tonal: Option<f64>,
    /// URLs for variation audio files.
    pub variation_urls: VariationUrls,
    /// Gapless playback metadata from audio database.
    pub gapless_metadata: Option<GaplessInfo>,
}

/// Gapless playback info (mirrors `pcm_cache::GaplessMetadata`
/// but uses f64 for consistency with the track model).
#[derive(Debug, Clone, PartialEq)]
pub struct GaplessInfo {
    pub original_samples: u64,
    pub encoder_delay: u32,
    pub end_padding: u32,
}

// ---------------------------------------------------------------------------
// Track controls (user-facing parameters)
// ---------------------------------------------------------------------------

/// User-controllable track parameters — the knobs and buttons.
#[derive(Debug, Clone, PartialEq)]
pub struct TrackControls {
    /// Track is armed for transport control.
    pub armed: bool,
    /// Track is currently playing (derived from status, but stored for
    /// fast access from the UI layer).
    pub playing: bool,
    /// Volume level (0.0–1.0).
    pub volume: f64,
    /// Pitch offset in cents (−1200 to +1200).
    pub pitch_cents: f64,
    /// Pan position (−1.0 left, 0.0 center, +1.0 right).
    pub pan: f64,
    /// Track is muted.
    pub muted: bool,
    /// Track is soloed.
    pub solo: bool,
    /// Currently selected variation index (0-based).
    pub current_variation: u32,
}

impl Default for TrackControls {
    fn default() -> Self {
        Self {
            armed: false,
            playing: false,
            volume: DEFAULT_VOLUME,
            pitch_cents: DEFAULT_PITCH_CENTS,
            pan: DEFAULT_PAN,
            muted: false,
            solo: false,
            current_variation: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Track
// ---------------------------------------------------------------------------

/// Core track model — owns state machine, metadata, and controls.
pub struct Track {
    /// Unique track identifier.
    id: String,
    /// URL of the audio resource loaded in the engine.
    engine_url: String,
    /// Current lifecycle status.
    status: TrackStatus,
    /// Playback position in seconds.
    position: f64,
    /// Total duration in seconds.
    duration: f64,
    /// Error message (set when status == Error).
    error_message: Option<String>,
    /// Descriptive metadata.
    metadata: TrackMetadata,
    /// User-controllable parameters.
    controls: TrackControls,
    /// Original (metadata) tempo in BPM.
    original_tempo: f64,
    /// Track length in beats (computed from sync_length or duration + tempo).
    track_length_beats: f64,
    /// Exact sync length in samples (from audio metadata).
    sync_length_samples: u64,
    /// Calculated start position for staggered entry (beats).
    start_position: f64,
}

impl Track {
    /// Create a new track in the `Idle` state.
    pub fn new(id: &str, engine_url: &str, metadata: Option<TrackMetadata>) -> Self {
        let meta = metadata.unwrap_or_default();
        let duration = meta.duration.unwrap_or(0.0);

        Self {
            id: id.to_string(),
            engine_url: engine_url.to_string(),
            status: TrackStatus::Idle,
            position: 0.0,
            duration,
            error_message: None,
            metadata: meta,
            controls: TrackControls::default(),
            original_tempo: 0.0,
            track_length_beats: 0.0,
            sync_length_samples: 0,
            start_position: 0.0,
        }
    }

    // -----------------------------------------------------------------------
    // Getters
    // -----------------------------------------------------------------------

    /// Track identifier.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Audio resource URL.
    pub fn engine_url(&self) -> &str {
        &self.engine_url
    }

    /// Current lifecycle status.
    pub fn status(&self) -> TrackStatus {
        self.status
    }

    /// Playback position in seconds.
    pub fn position(&self) -> f64 {
        self.position
    }

    /// Total duration in seconds.
    pub fn duration(&self) -> f64 {
        self.duration
    }

    /// Error message (if status == Error).
    pub fn error_message(&self) -> Option<&str> {
        self.error_message.as_deref()
    }

    /// Descriptive metadata.
    pub fn metadata(&self) -> &TrackMetadata {
        &self.metadata
    }

    /// User-controllable parameters.
    pub fn controls(&self) -> &TrackControls {
        &self.controls
    }

    /// Original (metadata) tempo in BPM.
    pub fn original_tempo(&self) -> f64 {
        self.original_tempo
    }

    /// Whether the track is currently playing.
    pub fn is_playing(&self) -> bool {
        self.status == TrackStatus::Playing
    }

    /// Track length in beats (computed from sync_length or duration).
    pub fn track_length_beats(&self) -> f64 {
        self.track_length_beats
    }

    /// Exact sync length in samples.
    pub fn sync_length_samples(&self) -> u64 {
        self.sync_length_samples
    }

    /// Calculated start position (beats).
    pub fn start_position(&self) -> f64 {
        self.start_position
    }

    // -----------------------------------------------------------------------
    // State mutation — status transitions
    // -----------------------------------------------------------------------

    /// Set the lifecycle status with state-machine validation.
    ///
    /// Same-status transitions are idempotent no-ops (return `Ok(false)`).
    /// Returns `Ok(true)` on successful transition, `Err` on invalid transition.
    pub fn set_status(&mut self, new_status: TrackStatus) -> Result<bool, TrackError> {
        if self.status == new_status {
            return Ok(false); // idempotent no-op
        }
        if !self.status.can_transition_to(new_status) {
            return Err(TrackError::InvalidTransition {
                from: self.status,
                to: new_status,
            });
        }
        self.status = new_status;
        // Sync playing control
        self.controls.playing = new_status == TrackStatus::Playing;
        Ok(true)
    }

    /// Set the status to `Error` with an error message.
    pub fn set_error(&mut self, message: &str) -> Result<bool, TrackError> {
        let changed = self.set_status(TrackStatus::Error)?;
        self.error_message = Some(message.to_string());
        Ok(changed)
    }

    /// Restore state directly (bypasses validation — for migration / deserialization).
    pub fn restore_state(
        &mut self,
        status: TrackStatus,
        position: f64,
        duration: f64,
    ) {
        self.status = status;
        self.position = position;
        self.duration = duration;
        self.controls.playing = status == TrackStatus::Playing;
        self.recalculate_track_length();
    }

    // -----------------------------------------------------------------------
    // State mutation — position / duration
    // -----------------------------------------------------------------------

    /// Set the playback position in seconds.
    pub fn set_position(&mut self, position: f64) -> Result<(), TrackError> {
        if position < 0.0 {
            return Err(TrackError::NegativePosition(position));
        }
        if self.duration > 0.0 && position > self.duration {
            return Err(TrackError::PositionExceedsDuration {
                position,
                duration: self.duration,
            });
        }
        self.position = position;
        Ok(())
    }

    /// Set the total duration in seconds.
    pub fn set_duration(&mut self, duration: f64) -> Result<(), TrackError> {
        if duration < 0.0 {
            return Err(TrackError::NegativeDuration(duration));
        }
        self.duration = duration;
        self.recalculate_track_length();
        Ok(())
    }

    // -----------------------------------------------------------------------
    // State mutation — metadata / tempo
    // -----------------------------------------------------------------------

    /// Update metadata.  Also updates duration if provided.
    pub fn set_metadata(&mut self, metadata: TrackMetadata) {
        if let Some(dur) = metadata.duration {
            if dur >= 0.0 {
                self.duration = dur;
            }
        }
        self.metadata = metadata;
        self.recalculate_track_length();
    }

    /// Set the original (metadata) tempo in BPM.
    pub fn set_original_tempo(&mut self, tempo: f64) -> Result<(), TrackError> {
        if tempo <= 0.0 || !tempo.is_finite() {
            return Err(TrackError::InvalidTempo(tempo));
        }
        self.original_tempo = tempo;
        self.recalculate_track_length();
        Ok(())
    }

    /// Set exact sync length in samples (from audio metadata).
    ///
    /// This is more precise than duration-based calculation and is the
    /// preferred method when gapless metadata is available.
    pub fn set_sync_length_samples(&mut self, samples: u64) {
        self.sync_length_samples = samples;
        self.recalculate_track_length();
    }

    /// Set audio track length from sync_length and sample_rate.
    ///
    /// Combines `set_sync_length_samples` with a sample-rate-aware
    /// length calculation.
    pub fn set_audio_track_length(&mut self, sync_length: u64, sample_rate: u32) {
        self.sync_length_samples = sync_length;
        // Recalculate using actual sample rate
        if self.original_tempo > 0.0 {
            if let Ok(spb) = timing::samples_per_beat(self.original_tempo, sample_rate) {
                self.track_length_beats = sync_length as f64 / spb;
                return;
            }
        }
        // Fallback to default sample rate
        self.recalculate_track_length();
    }

    // -----------------------------------------------------------------------
    // Control mutation
    // -----------------------------------------------------------------------

    /// Set the armed state.
    pub fn set_armed(&mut self, armed: bool) {
        self.controls.armed = armed;
    }

    /// Set volume, clamped to [0.0, 1.0].
    pub fn set_volume(&mut self, volume: f64) {
        self.controls.volume = volume.clamp(0.0, 1.0);
    }

    /// Set pitch offset in cents, clamped to [−1200, +1200].
    pub fn set_pitch_cents(&mut self, cents: f64) {
        self.controls.pitch_cents = cents.clamp(MIN_PITCH_CENTS, MAX_PITCH_CENTS);
    }

    /// Set pan position, clamped to [−1.0, +1.0].
    pub fn set_pan(&mut self, pan: f64) {
        self.controls.pan = pan.clamp(-1.0, 1.0);
    }

    /// Set muted state.
    pub fn set_muted(&mut self, muted: bool) {
        self.controls.muted = muted;
    }

    /// Set solo state.
    pub fn set_solo(&mut self, solo: bool) {
        self.controls.solo = solo;
    }

    /// Set the current variation index.
    pub fn set_current_variation(&mut self, index: u32) {
        self.controls.current_variation = index;
    }

    // -----------------------------------------------------------------------
    // Calculations
    // -----------------------------------------------------------------------

    /// Calculate the playback rate ratio for this track given a global tempo.
    ///
    /// `ratio = global_tempo / original_tempo`
    pub fn calculate_tempo_ratio(&self, global_tempo: f64) -> Result<f64, TrackError> {
        if self.original_tempo <= 0.0 {
            return Err(TrackError::InvalidTempo(self.original_tempo));
        }
        Ok(global_tempo / self.original_tempo)
    }

    /// Calculate start position for staggered entry using modulo arithmetic.
    ///
    /// Returns the beat offset within the track where playback should begin
    /// given the global beat count.
    pub fn calculate_start_position(&self, global_beat_count: f64) -> f64 {
        if self.track_length_beats <= 0.0 {
            return 0.0;
        }
        global_beat_count % self.track_length_beats
    }

    /// Set the start position from a global beat count.
    pub fn set_start_position_from_beat(&mut self, global_beat_count: f64) {
        self.start_position = self.calculate_start_position(global_beat_count);
    }

    /// Calculate track length in beats.
    ///
    /// Prefers sync_length_samples (exact) over duration-based calculation.
    pub fn length_in_beats(&self) -> f64 {
        // Prefer exact sync length
        if self.sync_length_samples > 0 && self.original_tempo > 0.0 {
            if let Ok(spb) =
                timing::samples_per_beat(self.original_tempo, DEFAULT_SAMPLE_RATE)
            {
                return self.sync_length_samples as f64 / spb;
            }
        }
        // Fallback to duration-based calculation
        if self.duration <= 0.0 || self.original_tempo <= 0.0 {
            return 0.0;
        }
        (self.duration * self.original_tempo) / 60.0
    }

    /// Get the pitch offset in semitones (derived from cents).
    pub fn pitch_semitones(&self) -> f64 {
        pitch::cents_to_semitones(self.controls.pitch_cents).unwrap_or(0.0)
    }

    /// Get the pitch ratio (derived from cents).
    pub fn pitch_ratio(&self) -> f64 {
        pitch::cents_to_ratio(self.controls.pitch_cents).unwrap_or(1.0)
    }

    // -----------------------------------------------------------------------
    // Serialization (debug / state snapshot)
    // -----------------------------------------------------------------------

    /// Create a serializable snapshot of the track state.
    pub fn to_snapshot(&self) -> TrackSnapshot {
        TrackSnapshot {
            id: self.id.clone(),
            engine_url: self.engine_url.clone(),
            status: self.status,
            position: self.position,
            duration: self.duration,
            original_tempo: self.original_tempo,
            track_length_beats: self.track_length_beats,
            is_playing: self.is_playing(),
            controls: self.controls.clone(),
        }
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Recalculate `track_length_beats` from sync_length or duration.
    fn recalculate_track_length(&mut self) {
        self.track_length_beats = self.length_in_beats();
    }
}

impl std::fmt::Debug for Track {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Track")
            .field("id", &self.id)
            .field("status", &self.status)
            .field("position", &self.position)
            .field("duration", &self.duration)
            .field("original_tempo", &self.original_tempo)
            .field("track_length_beats", &self.track_length_beats)
            .finish()
    }
}

impl std::fmt::Display for Track {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Track({}, status={}, position={}/{})",
            self.id, self.status, self.position, self.duration
        )
    }
}

// ---------------------------------------------------------------------------
// Track snapshot (for serialization / debug)
// ---------------------------------------------------------------------------

/// Serializable snapshot of a track's state.
#[derive(Debug, Clone)]
pub struct TrackSnapshot {
    pub id: String,
    pub engine_url: String,
    pub status: TrackStatus,
    pub position: f64,
    pub duration: f64,
    pub original_tempo: f64,
    pub track_length_beats: f64,
    pub is_playing: bool,
    pub controls: TrackControls,
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    #[test]
    fn new_track_defaults() {
        let track = Track::new("t1", "http://cdn.example.com/drums.m4a", None);
        assert_eq!(track.id(), "t1");
        assert_eq!(track.engine_url(), "http://cdn.example.com/drums.m4a");
        assert_eq!(track.status(), TrackStatus::Idle);
        assert_eq!(track.position(), 0.0);
        assert_eq!(track.duration(), 0.0);
        assert!(!track.is_playing());
        assert_eq!(track.original_tempo(), 0.0);
        assert_eq!(track.track_length_beats(), 0.0);
    }

    #[test]
    fn new_track_with_metadata() {
        let meta = TrackMetadata {
            name: Some("Drums".to_string()),
            duration: Some(10.0),
            ..Default::default()
        };
        let track = Track::new("t1", "url", Some(meta));
        assert_eq!(track.duration(), 10.0);
        assert_eq!(track.metadata().name.as_deref(), Some("Drums"));
    }

    #[test]
    fn default_controls() {
        let track = Track::new("t1", "url", None);
        let c = track.controls();
        assert!(!c.armed);
        assert!(!c.playing);
        assert_eq!(c.volume, DEFAULT_VOLUME);
        assert_eq!(c.pitch_cents, DEFAULT_PITCH_CENTS);
        assert_eq!(c.pan, DEFAULT_PAN);
        assert!(!c.muted);
        assert!(!c.solo);
        assert_eq!(c.current_variation, 0);
    }

    // -----------------------------------------------------------------------
    // State machine transitions
    // -----------------------------------------------------------------------

    mod state_machine {
        use super::*;

        #[test]
        fn idle_to_loading() {
            let mut t = Track::new("t1", "url", None);
            assert!(t.set_status(TrackStatus::Loading).unwrap());
            assert_eq!(t.status(), TrackStatus::Loading);
        }

        #[test]
        fn idle_to_preloaded() {
            let mut t = Track::new("t1", "url", None);
            assert!(t.set_status(TrackStatus::Preloaded).unwrap());
        }

        #[test]
        fn loading_to_ready() {
            let mut t = Track::new("t1", "url", None);
            t.set_status(TrackStatus::Loading).unwrap();
            assert!(t.set_status(TrackStatus::Ready).unwrap());
        }

        #[test]
        fn ready_to_playing() {
            let mut t = Track::new("t1", "url", None);
            t.set_status(TrackStatus::Loading).unwrap();
            t.set_status(TrackStatus::Ready).unwrap();
            assert!(t.set_status(TrackStatus::Playing).unwrap());
            assert!(t.is_playing());
            assert!(t.controls().playing);
        }

        #[test]
        fn playing_to_paused() {
            let mut t = Track::new("t1", "url", None);
            t.set_status(TrackStatus::Loading).unwrap();
            t.set_status(TrackStatus::Ready).unwrap();
            t.set_status(TrackStatus::Playing).unwrap();
            assert!(t.set_status(TrackStatus::Paused).unwrap());
            assert!(!t.is_playing());
            assert!(!t.controls().playing);
        }

        #[test]
        fn paused_to_playing() {
            let mut t = Track::new("t1", "url", None);
            t.set_status(TrackStatus::Loading).unwrap();
            t.set_status(TrackStatus::Ready).unwrap();
            t.set_status(TrackStatus::Playing).unwrap();
            t.set_status(TrackStatus::Paused).unwrap();
            assert!(t.set_status(TrackStatus::Playing).unwrap());
        }

        #[test]
        fn idempotent_same_status() {
            let mut t = Track::new("t1", "url", None);
            assert!(!t.set_status(TrackStatus::Idle).unwrap()); // no change
        }

        #[test]
        fn invalid_idle_to_playing() {
            let mut t = Track::new("t1", "url", None);
            let result = t.set_status(TrackStatus::Playing);
            assert!(matches!(
                result,
                Err(TrackError::InvalidTransition {
                    from: TrackStatus::Idle,
                    to: TrackStatus::Playing
                })
            ));
        }

        #[test]
        fn invalid_idle_to_paused() {
            let mut t = Track::new("t1", "url", None);
            let result = t.set_status(TrackStatus::Paused);
            assert!(matches!(result, Err(TrackError::InvalidTransition { .. })));
        }

        #[test]
        fn any_to_error() {
            for initial in [
                TrackStatus::Idle,
                TrackStatus::Loading,
                TrackStatus::Ready,
                TrackStatus::Preloaded,
                TrackStatus::Playing,
                TrackStatus::Paused,
            ] {
                let mut t = Track::new("t1", "url", None);
                t.restore_state(initial, 0.0, 10.0);
                assert!(t.set_status(TrackStatus::Error).is_ok());
            }
        }

        #[test]
        fn error_to_idle() {
            let mut t = Track::new("t1", "url", None);
            t.set_status(TrackStatus::Error).unwrap();
            assert!(t.set_status(TrackStatus::Idle).unwrap());
        }

        #[test]
        fn error_to_loading() {
            let mut t = Track::new("t1", "url", None);
            t.set_status(TrackStatus::Error).unwrap();
            assert!(t.set_status(TrackStatus::Loading).unwrap());
        }

        #[test]
        fn set_error_with_message() {
            let mut t = Track::new("t1", "url", None);
            t.set_error("decode failed").unwrap();
            assert_eq!(t.status(), TrackStatus::Error);
            assert_eq!(t.error_message(), Some("decode failed"));
        }
    }

    // -----------------------------------------------------------------------
    // Position / duration validation
    // -----------------------------------------------------------------------

    mod position {
        use super::*;

        #[test]
        fn set_valid_position() {
            let mut t = Track::new("t1", "url", None);
            t.set_duration(10.0).unwrap();
            t.set_position(5.0).unwrap();
            assert_eq!(t.position(), 5.0);
        }

        #[test]
        fn set_position_zero() {
            let mut t = Track::new("t1", "url", None);
            t.set_position(0.0).unwrap();
            assert_eq!(t.position(), 0.0);
        }

        #[test]
        fn set_position_at_duration() {
            let mut t = Track::new("t1", "url", None);
            t.set_duration(10.0).unwrap();
            t.set_position(10.0).unwrap();
            assert_eq!(t.position(), 10.0);
        }

        #[test]
        fn negative_position_rejected() {
            let mut t = Track::new("t1", "url", None);
            let result = t.set_position(-1.0);
            assert!(matches!(result, Err(TrackError::NegativePosition(_))));
        }

        #[test]
        fn position_exceeds_duration() {
            let mut t = Track::new("t1", "url", None);
            t.set_duration(10.0).unwrap();
            let result = t.set_position(11.0);
            assert!(matches!(
                result,
                Err(TrackError::PositionExceedsDuration { .. })
            ));
        }

        #[test]
        fn position_allowed_when_no_duration() {
            let mut t = Track::new("t1", "url", None);
            // duration is 0.0, any position should be allowed
            t.set_position(999.0).unwrap();
        }

        #[test]
        fn negative_duration_rejected() {
            let mut t = Track::new("t1", "url", None);
            let result = t.set_duration(-5.0);
            assert!(matches!(result, Err(TrackError::NegativeDuration(_))));
        }

        #[test]
        fn set_duration_updates_length() {
            let mut t = Track::new("t1", "url", None);
            t.set_original_tempo(120.0).unwrap();
            t.set_duration(10.0).unwrap();
            // 10s at 120 BPM = 20 beats
            assert!((t.track_length_beats() - 20.0).abs() < 0.01);
        }
    }

    // -----------------------------------------------------------------------
    // Tempo and beat calculations
    // -----------------------------------------------------------------------

    mod tempo {
        use super::*;

        #[test]
        fn set_original_tempo() {
            let mut t = Track::new("t1", "url", None);
            t.set_original_tempo(120.0).unwrap();
            assert_eq!(t.original_tempo(), 120.0);
        }

        #[test]
        fn set_original_tempo_invalid() {
            let mut t = Track::new("t1", "url", None);
            assert!(matches!(
                t.set_original_tempo(0.0),
                Err(TrackError::InvalidTempo(_))
            ));
            assert!(matches!(
                t.set_original_tempo(-10.0),
                Err(TrackError::InvalidTempo(_))
            ));
            assert!(matches!(
                t.set_original_tempo(f64::NAN),
                Err(TrackError::InvalidTempo(_))
            ));
        }

        #[test]
        fn calculate_tempo_ratio() {
            let mut t = Track::new("t1", "url", None);
            t.set_original_tempo(100.0).unwrap();
            assert!((t.calculate_tempo_ratio(150.0).unwrap() - 1.5).abs() < 0.001);
            assert!((t.calculate_tempo_ratio(100.0).unwrap() - 1.0).abs() < 0.001);
            assert!((t.calculate_tempo_ratio(50.0).unwrap() - 0.5).abs() < 0.001);
        }

        #[test]
        fn calculate_tempo_ratio_no_tempo() {
            let t = Track::new("t1", "url", None);
            assert!(matches!(
                t.calculate_tempo_ratio(120.0),
                Err(TrackError::InvalidTempo(_))
            ));
        }

        #[test]
        fn length_in_beats_from_duration() {
            let mut t = Track::new("t1", "url", None);
            t.set_original_tempo(120.0).unwrap();
            t.set_duration(30.0).unwrap();
            // 30s × 120 BPM / 60s = 60 beats
            assert!((t.length_in_beats() - 60.0).abs() < 0.01);
        }

        #[test]
        fn length_in_beats_from_sync_samples() {
            let mut t = Track::new("t1", "url", None);
            t.set_original_tempo(120.0).unwrap();
            // At 120 BPM, 48kHz: samples_per_beat = 24000
            // 96000 samples = 4 beats
            t.set_sync_length_samples(96000);
            assert!((t.length_in_beats() - 4.0).abs() < 0.01);
            assert!((t.track_length_beats() - 4.0).abs() < 0.01);
        }

        #[test]
        fn sync_samples_preferred_over_duration() {
            let mut t = Track::new("t1", "url", None);
            t.set_original_tempo(120.0).unwrap();
            t.set_duration(100.0).unwrap(); // would give 200 beats
            t.set_sync_length_samples(96000); // gives 4 beats (exact)
            assert!((t.length_in_beats() - 4.0).abs() < 0.01);
        }

        #[test]
        fn set_audio_track_length() {
            let mut t = Track::new("t1", "url", None);
            t.set_original_tempo(120.0).unwrap();
            t.set_audio_track_length(96000, 48000);
            assert!((t.track_length_beats() - 4.0).abs() < 0.01);
        }

        #[test]
        fn set_audio_track_length_different_sample_rate() {
            let mut t = Track::new("t1", "url", None);
            t.set_original_tempo(120.0).unwrap();
            // At 120 BPM, 44100 Hz: samples_per_beat = 22050
            // 88200 samples = 4 beats
            t.set_audio_track_length(88200, 44100);
            assert!((t.track_length_beats() - 4.0).abs() < 0.01);
        }

        #[test]
        fn length_in_beats_zero_when_no_tempo() {
            let mut t = Track::new("t1", "url", None);
            t.set_duration(10.0).unwrap();
            assert_eq!(t.length_in_beats(), 0.0);
        }

        #[test]
        fn length_in_beats_zero_when_no_duration() {
            let mut t = Track::new("t1", "url", None);
            t.set_original_tempo(120.0).unwrap();
            assert_eq!(t.length_in_beats(), 0.0);
        }
    }

    // -----------------------------------------------------------------------
    // Start position (modulo arithmetic)
    // -----------------------------------------------------------------------

    mod start_position {
        use super::*;

        #[test]
        fn start_position_modulo() {
            let mut t = Track::new("t1", "url", None);
            t.set_original_tempo(120.0).unwrap();
            t.set_sync_length_samples(96000); // 4 beats

            // Global beat 0 → start at beat 0
            assert_eq!(t.calculate_start_position(0.0), 0.0);
            // Global beat 2 → start at beat 2
            assert!((t.calculate_start_position(2.0) - 2.0).abs() < 0.001);
            // Global beat 4 → start at beat 0 (wrapped)
            assert!(t.calculate_start_position(4.0).abs() < 0.001);
            // Global beat 5 → start at beat 1
            assert!((t.calculate_start_position(5.0) - 1.0).abs() < 0.001);
        }

        #[test]
        fn start_position_zero_length() {
            let t = Track::new("t1", "url", None);
            assert_eq!(t.calculate_start_position(100.0), 0.0);
        }

        #[test]
        fn set_start_position_from_beat() {
            let mut t = Track::new("t1", "url", None);
            t.set_original_tempo(120.0).unwrap();
            t.set_sync_length_samples(96000); // 4 beats
            t.set_start_position_from_beat(5.0);
            assert!((t.start_position() - 1.0).abs() < 0.001);
        }
    }

    // -----------------------------------------------------------------------
    // Controls — clamping
    // -----------------------------------------------------------------------

    mod controls {
        use super::*;

        #[test]
        fn volume_clamped() {
            let mut t = Track::new("t1", "url", None);
            t.set_volume(1.5);
            assert_eq!(t.controls().volume, 1.0);
            t.set_volume(-0.5);
            assert_eq!(t.controls().volume, 0.0);
            t.set_volume(0.5);
            assert_eq!(t.controls().volume, 0.5);
        }

        #[test]
        fn pitch_cents_clamped() {
            let mut t = Track::new("t1", "url", None);
            t.set_pitch_cents(2000.0);
            assert_eq!(t.controls().pitch_cents, MAX_PITCH_CENTS);
            t.set_pitch_cents(-2000.0);
            assert_eq!(t.controls().pitch_cents, MIN_PITCH_CENTS);
            t.set_pitch_cents(500.0);
            assert_eq!(t.controls().pitch_cents, 500.0);
        }

        #[test]
        fn pan_clamped() {
            let mut t = Track::new("t1", "url", None);
            t.set_pan(5.0);
            assert_eq!(t.controls().pan, 1.0);
            t.set_pan(-5.0);
            assert_eq!(t.controls().pan, -1.0);
            t.set_pan(0.3);
            assert!((t.controls().pan - 0.3).abs() < 0.001);
        }

        #[test]
        fn armed_toggle() {
            let mut t = Track::new("t1", "url", None);
            assert!(!t.controls().armed);
            t.set_armed(true);
            assert!(t.controls().armed);
            t.set_armed(false);
            assert!(!t.controls().armed);
        }

        #[test]
        fn muted_and_solo() {
            let mut t = Track::new("t1", "url", None);
            t.set_muted(true);
            assert!(t.controls().muted);
            t.set_solo(true);
            assert!(t.controls().solo);
        }

        #[test]
        fn current_variation() {
            let mut t = Track::new("t1", "url", None);
            t.set_current_variation(2);
            assert_eq!(t.controls().current_variation, 2);
        }
    }

    // -----------------------------------------------------------------------
    // Pitch conversions
    // -----------------------------------------------------------------------

    mod pitch_conversion {
        use super::*;

        #[test]
        fn pitch_semitones_from_cents() {
            let mut t = Track::new("t1", "url", None);
            t.set_pitch_cents(700.0); // 7 semitones
            assert!((t.pitch_semitones() - 7.0).abs() < 0.001);
        }

        #[test]
        fn pitch_ratio_from_cents() {
            let mut t = Track::new("t1", "url", None);
            t.set_pitch_cents(1200.0); // 1 octave up
            assert!((t.pitch_ratio() - 2.0).abs() < 0.001);
        }

        #[test]
        fn pitch_ratio_default() {
            let t = Track::new("t1", "url", None);
            assert!((t.pitch_ratio() - 1.0).abs() < 0.001);
        }
    }

    // -----------------------------------------------------------------------
    // Display / Debug / Snapshot
    // -----------------------------------------------------------------------

    mod display {
        use super::*;

        #[test]
        fn track_display() {
            let t = Track::new("drums", "url", None);
            let s = t.to_string();
            assert!(s.contains("drums"));
            assert!(s.contains("idle"));
        }

        #[test]
        fn track_debug() {
            let t = Track::new("drums", "url", None);
            let s = format!("{:?}", t);
            assert!(s.contains("Track"));
            assert!(s.contains("drums"));
        }

        #[test]
        fn track_status_display() {
            assert_eq!(TrackStatus::Idle.to_string(), "idle");
            assert_eq!(TrackStatus::Playing.to_string(), "playing");
            assert_eq!(TrackStatus::Error.to_string(), "error");
        }

        #[test]
        fn snapshot() {
            let mut t = Track::new("t1", "url", None);
            t.set_original_tempo(120.0).unwrap();
            t.set_volume(0.5);
            let snap = t.to_snapshot();
            assert_eq!(snap.id, "t1");
            assert_eq!(snap.status, TrackStatus::Idle);
            assert_eq!(snap.controls.volume, 0.5);
        }
    }

    // -----------------------------------------------------------------------
    // Restore state
    // -----------------------------------------------------------------------

    #[test]
    fn restore_state_bypasses_validation() {
        let mut t = Track::new("t1", "url", None);
        // Directly jump to Playing without going through Loading → Ready
        t.restore_state(TrackStatus::Playing, 5.0, 10.0);
        assert_eq!(t.status(), TrackStatus::Playing);
        assert_eq!(t.position(), 5.0);
        assert_eq!(t.duration(), 10.0);
        assert!(t.controls().playing);
    }

    // -----------------------------------------------------------------------
    // Error display
    // -----------------------------------------------------------------------

    #[test]
    fn error_display() {
        let e = TrackError::InvalidTransition {
            from: TrackStatus::Idle,
            to: TrackStatus::Playing,
        };
        assert!(e.to_string().contains("idle"));
        assert!(e.to_string().contains("playing"));

        let e = TrackError::NegativePosition(-1.0);
        assert!(e.to_string().contains("-1"));

        let e = TrackError::PositionExceedsDuration {
            position: 15.0,
            duration: 10.0,
        };
        assert!(e.to_string().contains("15"));
        assert!(e.to_string().contains("10"));

        let e = TrackError::NegativeDuration(-5.0);
        assert!(e.to_string().contains("-5"));

        let e = TrackError::InvalidTempo(0.0);
        assert!(e.to_string().contains('0'));
    }

    // -----------------------------------------------------------------------
    // Metadata update
    // -----------------------------------------------------------------------

    #[test]
    fn set_metadata_updates_duration() {
        let mut t = Track::new("t1", "url", None);
        t.set_original_tempo(120.0).unwrap();
        t.set_metadata(TrackMetadata {
            duration: Some(30.0),
            ..Default::default()
        });
        assert_eq!(t.duration(), 30.0);
        assert!((t.track_length_beats() - 60.0).abs() < 0.01);
    }

    #[test]
    fn set_metadata_ignores_negative_duration() {
        let mut t = Track::new("t1", "url", None);
        t.set_duration(10.0).unwrap();
        t.set_metadata(TrackMetadata {
            duration: Some(-5.0),
            ..Default::default()
        });
        // Duration should remain 10.0 since -5.0 is invalid
        assert_eq!(t.duration(), 10.0);
    }
}
