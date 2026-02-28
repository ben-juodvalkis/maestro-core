//! # MaestroEngine — orchestration layer
//!
//! The top-level coordinator that owns all sub-systems and exposes the public
//! API consumed by the Capacitor plugin (native) or wasm-bindgen (web).
//!
//! ## Design
//!
//! Mirrors the TypeScript `AudioEngineCoordinator` as a thin facade that
//! delegates to owned sub-components:
//!
//! - [`TransportController`] — transport state machine + safety presets
//! - [`Mixer`] — volume, pan, mute, solo, level metering
//! - [`TempoController`] — global/per-track tempo ratios, pitch compensation
//! - [`BeatTracker`] — beat/bar/loop event emission
//! - [`BeatTimeline`] — sample-accurate position tracking
//! - [`PcmCache`] — decoded audio data cache
//! - Track registry — `HashMap<String, Track>`
//! - [`EventCollector`] — accumulates events during operations
//!
//! ## Event model
//!
//! Operations on the engine push events into the internal [`EventCollector`].
//! After each operation the caller should drain events via [`drain_events`]
//! and dispatch them to the platform layer.
//!
//! ```
//! use maestro_core::engine::MaestroEngine;
//!
//! let mut engine = MaestroEngine::new(48000);
//! engine.initialize();
//! for event in engine.drain_events() {
//!     // dispatch to Capacitor plugin / wasm-bindgen / test harness
//! }
//! ```
//!
//! ## Thread safety
//!
//! On native targets the engine is `!Send + !Sync` by default.  The platform
//! layer wraps it in an `Arc<Mutex<>>` or uses message passing as needed.
//! On WASM the engine runs single-threaded in the AudioWorklet.
//!
//! [`TransportController`]: crate::transport::TransportController
//! [`Mixer`]: crate::mixer::Mixer
//! [`TempoController`]: crate::tempo::TempoController
//! [`BeatTracker`]: crate::beat_tracker::BeatTracker
//! [`BeatTimeline`]: crate::beat_timeline::BeatTimeline
//! [`PcmCache`]: crate::pcm_cache::PcmCache
//! [`EventCollector`]: crate::events::EventCollector
//! [`drain_events`]: MaestroEngine::drain_events

use std::collections::HashMap;

use crate::beat_tracker::{BeatTracker, BeatTrackingConfig};
use crate::beat_timeline::BeatTimeline;
use crate::events::{EngineEvent, EngineState, EventCollector};
use crate::mixer::Mixer;
use crate::pcm_cache::{PcmCache, PcmData, VariationType};
use crate::pitch;
use crate::tempo::TempoController;
use crate::track::{Track, TrackMetadata, TrackStatus};
use crate::transport::{
    OperationContext, OperationSource, SafetyPreset, TransportAction, TransportController,
    TransportState,
};

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors from engine operations.
#[derive(Debug, Clone, PartialEq)]
pub enum EngineError {
    /// Invalid engine state for the requested operation.
    InvalidState {
        current: EngineState,
        operation: &'static str,
    },
    /// State transition not allowed.
    InvalidStateTransition {
        from: EngineState,
        to: EngineState,
    },
    /// Track not found.
    TrackNotFound(String),
    /// Track already exists.
    TrackAlreadyExists(String),
    /// Tempo error (delegates to TempoController).
    Tempo(String),
    /// Pitch error.
    Pitch(String),
    /// Mixer error.
    Mixer(String),
    /// Track operation error.
    Track(String),
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EngineError::InvalidState { current, operation } => {
                write!(
                    f,
                    "Cannot '{operation}' in engine state '{current}'"
                )
            }
            EngineError::InvalidStateTransition { from, to } => {
                write!(f, "Invalid state transition: {from} → {to}")
            }
            EngineError::TrackNotFound(id) => write!(f, "Track '{id}' not found"),
            EngineError::TrackAlreadyExists(id) => write!(f, "Track '{id}' already exists"),
            EngineError::Tempo(msg) => write!(f, "Tempo error: {msg}"),
            EngineError::Pitch(msg) => write!(f, "Pitch error: {msg}"),
            EngineError::Mixer(msg) => write!(f, "Mixer error: {msg}"),
            EngineError::Track(msg) => write!(f, "Track error: {msg}"),
        }
    }
}

impl std::error::Error for EngineError {}

// ---------------------------------------------------------------------------
// Beat info snapshot (for API queries)
// ---------------------------------------------------------------------------

/// Beat information snapshot for the public API.
///
/// Matches the TypeScript `getBeatInfo()` return shape.
#[derive(Debug, Clone, PartialEq)]
pub struct BeatInfoSnapshot {
    pub current_beat: u32,
    pub beat_fraction: f64,
    pub bpm: f64,
    pub time_signature_numerator: u32,
    pub time_signature_denominator: u32,
}

// ---------------------------------------------------------------------------
// Track info snapshot (for API queries)
// ---------------------------------------------------------------------------

/// Lightweight track info for the public API.
///
/// Avoids exposing the full `Track` struct across FFI boundaries.
#[derive(Debug, Clone, PartialEq)]
pub struct TrackInfo {
    pub id: String,
    pub status: TrackStatus,
    pub armed: bool,
    pub volume: f64,
    pub pitch_cents: f64,
    pub pan: f64,
    pub muted: bool,
    pub solo: bool,
    pub duration: f64,
    pub position: f64,
    pub current_variation: u32,
}

// ---------------------------------------------------------------------------
// MaestroEngine
// ---------------------------------------------------------------------------

/// The top-level audio engine coordinator.
///
/// Owns all sub-systems and provides the public API matching the roadmap's
/// `MaestroEngine` interface (Section 1.3).
pub struct MaestroEngine {
    // -- Configuration --
    sample_rate: u32,

    // -- Lifecycle state --
    state: EngineState,

    // -- Sub-systems (owned) --
    transport: TransportController,
    mixer: Mixer,
    tempo: TempoController,
    beat_tracker: BeatTracker,
    timeline: BeatTimeline,
    pcm_cache: PcmCache,

    // -- Track registry --
    tracks: HashMap<String, Track>,

    // -- Event collection --
    events: EventCollector,
}

impl std::fmt::Debug for MaestroEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MaestroEngine")
            .field("sample_rate", &self.sample_rate)
            .field("state", &self.state)
            .field("track_count", &self.tracks.len())
            .field("transport_state", &self.transport.state())
            .field("pending_events", &self.events.len())
            .finish()
    }
}

impl MaestroEngine {
    // ===================================================================
    // Construction & lifecycle
    // ===================================================================

    /// Create a new engine with the given sample rate.
    ///
    /// The engine starts in `Uninitialized` state.  Call [`initialize`] to
    /// set up all sub-systems.
    ///
    /// [`initialize`]: MaestroEngine::initialize
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            state: EngineState::Uninitialized,
            transport: TransportController::new(),
            mixer: Mixer::new(),
            tempo: TempoController::new(),
            beat_tracker: BeatTracker::with_defaults(),
            timeline: BeatTimeline::new(sample_rate),
            pcm_cache: PcmCache::new(),
            tracks: HashMap::new(),
            events: EventCollector::with_capacity(16),
        }
    }

    /// Initialize the engine — transitions from Uninitialized → Ready.
    ///
    /// This creates all sub-systems and marks the engine as operational.
    /// On native, the platform layer should call this after setting up the
    /// audio session.  On WASM, call after AudioContext creation.
    pub fn initialize(&mut self) -> Result<(), EngineError> {
        self.transition_state(EngineState::Initializing)?;

        // Initialize timeline
        self.timeline.initialize();

        self.transition_state(EngineState::ServicesReady)?;

        self.events.push(EngineEvent::Initialized {
            sample_rate: self.sample_rate,
        });

        Ok(())
    }

    /// Get the current engine state.
    pub fn state(&self) -> EngineState {
        self.state
    }

    /// Get the sample rate.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Suspend the engine (e.g., app backgrounded).
    pub fn suspend(&mut self) -> Result<(), EngineError> {
        self.transition_state(EngineState::Suspended)?;
        Ok(())
    }

    /// Resume from suspended state.
    pub fn resume(&mut self) -> Result<(), EngineError> {
        self.transition_state(EngineState::Ready)?;
        Ok(())
    }

    /// Destroy the engine — releases all resources.
    pub fn destroy(&mut self) {
        // Destroying is always valid, no need to check transition
        self.state = EngineState::Destroying;

        // Stop transport
        self.transport.reset();

        // Stop beat tracking
        if self.beat_tracker.is_tracking() {
            self.beat_tracker.stop();
        }

        // Reset timeline
        self.timeline.reset();

        // Clear tracks
        self.tracks.clear();

        // Clear mixer
        self.mixer.clear();

        // Clear tempo registrations
        self.tempo.reset();

        // Clear cache
        self.pcm_cache.clear_all();

        // Clear pending events
        self.events.clear();
    }

    // ===================================================================
    // Track management
    // ===================================================================

    /// Create and register a new track.
    ///
    /// The track starts in `Idle` status.  The caller should then load audio
    /// data via [`load_track_audio`] and transition the track to `Ready`.
    ///
    /// [`load_track_audio`]: MaestroEngine::load_track_audio
    pub fn create_track(
        &mut self,
        id: &str,
        engine_url: &str,
        metadata: Option<TrackMetadata>,
        original_tempo: Option<f64>,
    ) -> Result<(), EngineError> {
        self.require_min_state(EngineState::ServicesReady, "create_track")?;

        if self.tracks.contains_key(id) {
            return Err(EngineError::TrackAlreadyExists(id.to_string()));
        }

        // Create track
        let track = Track::new(id, engine_url, metadata);
        self.tracks.insert(id.to_string(), track);

        // Register in mixer
        self.mixer.register_track(id);

        // Register in tempo controller if original tempo provided
        if let Some(tempo) = original_tempo {
            let _ = self.tempo.register_track(id, tempo);
        }

        // Advance state if this is the first track
        if self.tracks.len() == 1 && self.state == EngineState::ServicesReady {
            self.transition_state(EngineState::EnginesReady)?;
        }

        self.events.push(EngineEvent::TrackCreated {
            track_id: id.to_string(),
        });

        Ok(())
    }

    /// Remove a track from the engine.
    pub fn remove_track(&mut self, track_id: &str) -> Result<(), EngineError> {
        if !self.tracks.contains_key(track_id) {
            return Err(EngineError::TrackNotFound(track_id.to_string()));
        }

        // If track was playing, stop it first
        if self.transport.is_track_active(track_id) {
            let action = self.transport.on_track_stopped(track_id);
            self.handle_transport_action(action);
        }

        // Unregister from subsystems
        self.mixer.unregister_track(track_id);
        self.tempo.unregister_track(track_id);
        self.pcm_cache.clear_track(track_id);

        // Remove track
        self.tracks.remove(track_id);

        self.events.push(EngineEvent::TrackRemoved {
            track_id: track_id.to_string(),
        });

        Ok(())
    }

    /// Load decoded PCM audio data for a track.
    ///
    /// The platform layer is responsible for decoding (symphonia on native,
    /// AudioContext.decodeAudioData on web).  This method stores the decoded
    /// data and transitions the track to Ready.
    pub fn load_track_audio(
        &mut self,
        track_id: &str,
        variation: VariationType,
        pcm_data: PcmData,
        url: &str,
    ) -> Result<(), EngineError> {
        let track = self
            .tracks
            .get_mut(track_id)
            .ok_or_else(|| EngineError::TrackNotFound(track_id.to_string()))?;

        let previous_state = track.status();

        // Transition to Loading if idle
        if track.status() == TrackStatus::Idle {
            track
                .set_status(TrackStatus::Loading)
                .map_err(|e| EngineError::Track(e.to_string()))?;
        }

        // Store in cache
        let duration = pcm_data.duration_seconds();
        self.pcm_cache
            .store(track_id, variation, pcm_data, url, None, 0)
            .map_err(|e| EngineError::Track(e.to_string()))?;

        // Set track duration from loaded data
        track
            .set_duration(duration)
            .map_err(|e| EngineError::Track(e.to_string()))?;

        // Transition to Ready
        track
            .set_status(TrackStatus::Ready)
            .map_err(|e| EngineError::Track(e.to_string()))?;

        self.events.push(EngineEvent::TrackStateChanged {
            track_id: track_id.to_string(),
            state: TrackStatus::Ready,
            previous_state,
        });
        self.events.push(EngineEvent::TrackLoaded {
            track_id: track_id.to_string(),
        });

        // Check if all tracks are now loaded
        if self.all_tracks_loaded() {
            self.events.push(EngineEvent::AllTracksLoaded);
            // Advance to Ready state if needed
            if self.state == EngineState::EnginesReady {
                self.transition_state(EngineState::Ready)?;
            }
        }

        Ok(())
    }

    /// Arm a track for playback.
    pub fn arm_track(&mut self, track_id: &str) -> Result<(), EngineError> {
        let track = self
            .tracks
            .get_mut(track_id)
            .ok_or_else(|| EngineError::TrackNotFound(track_id.to_string()))?;

        track.set_armed(true);

        self.events.push(EngineEvent::TrackArmed {
            track_id: track_id.to_string(),
        });

        Ok(())
    }

    /// Disarm a track.
    pub fn disarm_track(&mut self, track_id: &str) -> Result<(), EngineError> {
        let track = self
            .tracks
            .get_mut(track_id)
            .ok_or_else(|| EngineError::TrackNotFound(track_id.to_string()))?;

        track.set_armed(false);

        self.events.push(EngineEvent::TrackDisarmed {
            track_id: track_id.to_string(),
        });

        Ok(())
    }

    /// Toggle a track's armed state.
    pub fn toggle_track(&mut self, track_id: &str) -> Result<bool, EngineError> {
        let new_armed = {
            let track = self
                .tracks
                .get(track_id)
                .ok_or_else(|| EngineError::TrackNotFound(track_id.to_string()))?;
            !track.controls().armed
        };

        if new_armed {
            self.arm_track(track_id)?;
        } else {
            self.disarm_track(track_id)?;
        }

        Ok(new_armed)
    }

    // ===================================================================
    // Transport (play / pause / stop)
    // ===================================================================

    /// Start playback with a safety preset.
    ///
    /// Arms all armed tracks for playback and starts the transport.
    pub fn play(&mut self, preset: SafetyPreset) -> Result<(), EngineError> {
        self.require_min_state(EngineState::EnginesReady, "play")?;

        self.transport.set_operation_context(OperationContext {
            source: OperationSource::TransportButton,
            safety_preset: preset,
            reason: Some("play".into()),
            track_id: None,
        });

        // Start all armed tracks that are in Ready or Paused state
        let track_ids: Vec<String> = self
            .tracks
            .iter()
            .filter(|(_, t)| {
                t.controls().armed
                    && matches!(t.status(), TrackStatus::Ready | TrackStatus::Paused)
            })
            .map(|(id, _)| id.clone())
            .collect();

        for track_id in &track_ids {
            self.start_track(track_id)?;
        }

        self.transport.clear_operation_context();

        Ok(())
    }

    /// Pause playback.
    pub fn pause(&mut self) -> Result<(), EngineError> {
        self.require_min_state(EngineState::EnginesReady, "pause")?;

        // Use SliderOperation preset to prevent auto-stop when tracks report stopped
        self.transport.set_operation_context(OperationContext {
            source: OperationSource::TransportButton,
            safety_preset: SafetyPreset::SliderOperation,
            reason: Some("pause".into()),
            track_id: None,
        });

        // Pause all playing tracks
        let playing_ids: Vec<String> = self
            .tracks
            .iter()
            .filter(|(_, t)| t.status() == TrackStatus::Playing)
            .map(|(id, _)| id.clone())
            .collect();

        for track_id in &playing_ids {
            self.pause_track(track_id)?;
        }

        self.transport.clear_operation_context();

        // Explicitly pause transport (not auto-stop)
        let action = self.transport.pause();
        self.handle_transport_action(action);

        self.events.push(EngineEvent::PlaybackPaused);
        self.events.push(EngineEvent::PlaybackStateChange {
            is_playing: false,
            is_paused: true,
        });

        Ok(())
    }

    /// Stop playback — resets all tracks to Ready.
    pub fn stop(&mut self) -> Result<(), EngineError> {
        self.require_min_state(EngineState::EnginesReady, "stop")?;

        // Use SliderOperation preset to prevent auto-stop while we handle it explicitly
        self.transport.set_operation_context(OperationContext {
            source: OperationSource::TransportButton,
            safety_preset: SafetyPreset::SliderOperation,
            reason: Some("stop".into()),
            track_id: None,
        });

        // Stop all non-idle tracks
        let active_ids: Vec<String> = self
            .tracks
            .iter()
            .filter(|(_, t)| {
                matches!(
                    t.status(),
                    TrackStatus::Playing | TrackStatus::Paused
                )
            })
            .map(|(id, _)| id.clone())
            .collect();

        for track_id in &active_ids {
            self.stop_track(track_id)?;
        }

        self.transport.clear_operation_context();

        // Explicitly stop transport
        let action = self.transport.stop();
        self.handle_transport_action(action);

        self.events.push(EngineEvent::PlaybackStopped);
        self.events.push(EngineEvent::PlaybackStateChange {
            is_playing: false,
            is_paused: false,
        });

        Ok(())
    }

    /// Get the current transport state.
    pub fn transport_state(&self) -> TransportState {
        self.transport.state()
    }

    // ===================================================================
    // Individual track playback
    // ===================================================================

    /// Start a single track playing.
    fn start_track(&mut self, track_id: &str) -> Result<(), EngineError> {
        let track = self
            .tracks
            .get_mut(track_id)
            .ok_or_else(|| EngineError::TrackNotFound(track_id.to_string()))?;

        let previous_state = track.status();

        // Transition track to Playing
        track
            .set_status(TrackStatus::Playing)
            .map_err(|e| EngineError::Track(e.to_string()))?;

        // Notify transport
        let action = self.transport.on_track_started(track_id);
        let active_count = self.transport.active_track_count();

        self.handle_transport_action(action);

        // Emit events
        self.events.push(EngineEvent::TrackStateChanged {
            track_id: track_id.to_string(),
            state: TrackStatus::Playing,
            previous_state,
        });
        self.events.push(EngineEvent::TrackStarted {
            track_id: track_id.to_string(),
            active_track_count: active_count,
        });
        self.events.push(EngineEvent::TrackPlayingStateChanged {
            track_id: track_id.to_string(),
            is_playing: true,
        });

        Ok(())
    }

    /// Pause a single track.
    fn pause_track(&mut self, track_id: &str) -> Result<(), EngineError> {
        let track = self
            .tracks
            .get_mut(track_id)
            .ok_or_else(|| EngineError::TrackNotFound(track_id.to_string()))?;

        let previous_state = track.status();

        track
            .set_status(TrackStatus::Paused)
            .map_err(|e| EngineError::Track(e.to_string()))?;

        // Notify transport
        let action = self.transport.on_track_stopped(track_id);
        let active_count = self.transport.active_track_count();
        self.handle_transport_action(action);

        self.events.push(EngineEvent::TrackStateChanged {
            track_id: track_id.to_string(),
            state: TrackStatus::Paused,
            previous_state,
        });
        self.events.push(EngineEvent::TrackStopped {
            track_id: track_id.to_string(),
            active_track_count: active_count,
        });
        self.events.push(EngineEvent::TrackPlayingStateChanged {
            track_id: track_id.to_string(),
            is_playing: false,
        });

        Ok(())
    }

    /// Stop a single track (reset to Ready).
    fn stop_track(&mut self, track_id: &str) -> Result<(), EngineError> {
        let track = self
            .tracks
            .get_mut(track_id)
            .ok_or_else(|| EngineError::TrackNotFound(track_id.to_string()))?;

        let previous_state = track.status();

        track
            .set_status(TrackStatus::Ready)
            .map_err(|e| EngineError::Track(e.to_string()))?;

        // Reset position
        let _ = track.set_position(0.0);

        // Notify transport
        let action = self.transport.on_track_stopped(track_id);
        let active_count = self.transport.active_track_count();
        self.handle_transport_action(action);

        self.events.push(EngineEvent::TrackStateChanged {
            track_id: track_id.to_string(),
            state: TrackStatus::Ready,
            previous_state,
        });
        self.events.push(EngineEvent::TrackStopped {
            track_id: track_id.to_string(),
            active_track_count: active_count,
        });
        self.events.push(EngineEvent::TrackPlayingStateChanged {
            track_id: track_id.to_string(),
            is_playing: false,
        });

        Ok(())
    }

    // ===================================================================
    // Tempo control
    // ===================================================================

    /// Set the tempo ratio (1.0 = original tempo).
    pub fn set_tempo_ratio(&mut self, ratio: f64) -> Result<(), EngineError> {
        let previous_tempo = self.tempo.current_tempo().unwrap_or(120.0);

        self.tempo
            .set_tempo_ratio(ratio)
            .map_err(|e| EngineError::Tempo(e.to_string()))?;

        let new_tempo = self.tempo.current_tempo().unwrap_or(120.0);

        if (new_tempo - previous_tempo).abs() > 0.01 {
            self.events.push(EngineEvent::TempoChanged {
                tempo: new_tempo,
                previous_tempo,
            });
        }

        Ok(())
    }

    /// Get the current tempo ratio.
    pub fn tempo_ratio(&self) -> f64 {
        self.tempo.tempo_ratio().unwrap_or(1.0)
    }

    /// Get the current effective BPM.
    pub fn current_bpm(&self) -> f64 {
        self.tempo.current_tempo().unwrap_or(120.0)
    }

    /// Set the global tempo directly in BPM.
    pub fn set_tempo(&mut self, bpm: f64) -> Result<(), EngineError> {
        let previous_tempo = self.tempo.current_tempo().unwrap_or(120.0);

        self.tempo
            .set_tempo(bpm)
            .map_err(|e| EngineError::Tempo(e.to_string()))?;

        // Update timeline
        self.timeline.set_tempo(bpm);

        if (bpm - previous_tempo).abs() > 0.01 {
            self.events.push(EngineEvent::TempoChanged {
                tempo: bpm,
                previous_tempo,
            });
        }

        Ok(())
    }

    // ===================================================================
    // Volume control
    // ===================================================================

    /// Set a track's volume (0.0–1.0).
    pub fn set_track_volume(
        &mut self,
        track_id: &str,
        volume: f64,
    ) -> Result<(), EngineError> {
        if !self.tracks.contains_key(track_id) {
            return Err(EngineError::TrackNotFound(track_id.to_string()));
        }

        // Update track controls
        if let Some(track) = self.tracks.get_mut(track_id) {
            track.set_volume(volume);
        }

        // Update mixer
        self.mixer
            .set_track_volume(track_id, volume)
            .map_err(|e| EngineError::Mixer(e.to_string()))?;

        self.events.push(EngineEvent::TrackVolumeChanged {
            track_id: track_id.to_string(),
            volume,
        });

        Ok(())
    }

    /// Mute a track.
    pub fn mute_track(&mut self, track_id: &str) -> Result<(), EngineError> {
        if !self.tracks.contains_key(track_id) {
            return Err(EngineError::TrackNotFound(track_id.to_string()));
        }

        self.mixer
            .mute_track(track_id)
            .map_err(|e| EngineError::Mixer(e.to_string()))?;

        Ok(())
    }

    /// Unmute a track.
    pub fn unmute_track(&mut self, track_id: &str) -> Result<(), EngineError> {
        if !self.tracks.contains_key(track_id) {
            return Err(EngineError::TrackNotFound(track_id.to_string()));
        }

        self.mixer
            .unmute_track(track_id)
            .map_err(|e| EngineError::Mixer(e.to_string()))?;

        Ok(())
    }

    // ===================================================================
    // Pitch control
    // ===================================================================

    /// Set a track's pitch offset in cents.
    pub fn set_track_pitch_cents(
        &mut self,
        track_id: &str,
        cents: f64,
    ) -> Result<(), EngineError> {
        let track = self
            .tracks
            .get_mut(track_id)
            .ok_or_else(|| EngineError::TrackNotFound(track_id.to_string()))?;

        track.set_pitch_cents(cents);

        let semitones =
            pitch::cents_to_semitones(cents).map_err(|e| EngineError::Pitch(e.to_string()))?;

        self.events.push(EngineEvent::TrackPitchChanged {
            track_id: track_id.to_string(),
            offset_cents: cents,
            offset_semitones: semitones,
        });

        Ok(())
    }

    // ===================================================================
    // Variation control
    // ===================================================================

    /// Set a track's current variation.
    pub fn set_track_variation(
        &mut self,
        track_id: &str,
        variation_index: u32,
        variation_name: &str,
    ) -> Result<(), EngineError> {
        let track = self
            .tracks
            .get_mut(track_id)
            .ok_or_else(|| EngineError::TrackNotFound(track_id.to_string()))?;

        let previous_index = track.controls().current_variation;
        track.set_current_variation(variation_index);

        self.events.push(EngineEvent::VariationChanged {
            track_id: track_id.to_string(),
            previous_variation_index: previous_index,
            new_variation_index: variation_index,
            variation_name: variation_name.to_string(),
        });

        Ok(())
    }

    // ===================================================================
    // Beat tracking
    // ===================================================================

    /// Update beat tracking with the current fractional beat position.
    ///
    /// Called by the platform layer at the audio callback rate.  Returns
    /// any beat/bar/loop events that occurred.
    pub fn update_beat_position(&mut self, exact_beat: f64) {
        let (changes, beat_events) = self.beat_tracker.update(exact_beat);

        // Convert beat tracker events to engine events
        for beat_event in beat_events {
            match beat_event {
                crate::beat_tracker::BeatEvent::Beat(info) => {
                    self.events.push(EngineEvent::BeatPrecise {
                        beat_info: info,
                    });
                }
                crate::beat_tracker::BeatEvent::Bar(bar_info) => {
                    self.events.push(EngineEvent::BarChanged {
                        bar: bar_info.bar,
                        beat: bar_info.beat,
                    });
                }
                crate::beat_tracker::BeatEvent::Loop(loop_info) => {
                    self.events.push(EngineEvent::LoopCompleted {
                        loop_count: loop_info.loop_count,
                    });
                }
                _ => {} // Play/Stop/Seek handled by transport
            }
        }

        // Emit position update if beat changed
        if changes.beat_changed {
            let config = self.beat_tracker.config();
            let total_beats =
                (config.beats_per_bar * config.bars_per_loop) as f64;
            let loop_progress = if total_beats > 0.0 {
                (exact_beat % total_beats) / total_beats
            } else {
                0.0
            };

            self.events.push(EngineEvent::PositionUpdated {
                beat: exact_beat,
                loop_progress,
            });
        }
    }

    /// Configure beat tracking parameters.
    pub fn set_beat_tracking_config(&mut self, config: BeatTrackingConfig) {
        self.beat_tracker.set_config(config);
    }

    // ===================================================================
    // State queries
    // ===================================================================

    /// Get all track IDs.
    pub fn track_ids(&self) -> Vec<String> {
        self.tracks.keys().cloned().collect()
    }

    /// Get track info by ID.
    pub fn get_track_info(&self, track_id: &str) -> Option<TrackInfo> {
        let track = self.tracks.get(track_id)?;
        let (muted, solo) = self
            .mixer
            .track_state(track_id)
            .map(|s| (s.muted, s.solo))
            .unwrap_or((false, false));

        Some(TrackInfo {
            id: track.id().to_string(),
            status: track.status(),
            armed: track.controls().armed,
            volume: track.controls().volume,
            pitch_cents: track.controls().pitch_cents,
            pan: track.controls().pan,
            muted,
            solo,
            duration: track.duration(),
            position: track.position(),
            current_variation: track.controls().current_variation,
        })
    }

    /// Get all tracks' info.
    pub fn get_all_tracks(&self) -> Vec<TrackInfo> {
        self.tracks
            .keys()
            .filter_map(|id| self.get_track_info(id))
            .collect()
    }

    /// Get the number of registered tracks.
    pub fn track_count(&self) -> usize {
        self.tracks.len()
    }

    /// Get beat info snapshot.
    ///
    /// Returns bpm=0.0 if the timeline has not been configured with a tempo
    /// yet (e.g., before any song is loaded).
    pub fn get_beat_info(&self) -> BeatInfoSnapshot {
        let config = self.beat_tracker.config();
        // timeline.get_tempo() panics if tempo == 0; guard against that
        let tempo = self.tempo.current_tempo().unwrap_or(0.0);
        BeatInfoSnapshot {
            current_beat: self.timeline.get_current_beat(),
            beat_fraction: self.timeline.get_fractional_beat(),
            bpm: tempo,
            time_signature_numerator: config.beats_per_bar,
            time_signature_denominator: 4, // Always 4/4 in current implementation
        }
    }

    /// Check if any track is currently playing.
    pub fn is_playing(&self) -> bool {
        self.transport.is_playing()
    }

    /// Check if playback is paused.
    pub fn is_paused(&self) -> bool {
        self.transport.is_paused()
    }

    /// Check if transport is stopped.
    pub fn is_stopped(&self) -> bool {
        self.transport.is_stopped()
    }

    /// Get the number of actively playing tracks.
    pub fn active_track_count(&self) -> usize {
        self.transport.active_track_count()
    }

    // ===================================================================
    // Event system
    // ===================================================================

    /// Drain all pending events.
    ///
    /// The platform layer should call this after each engine operation and
    /// dispatch the events to the appropriate targets.
    pub fn drain_events(&mut self) -> Vec<EngineEvent> {
        self.events.drain()
    }

    /// Peek at pending events without draining.
    pub fn peek_events(&self) -> &[EngineEvent] {
        self.events.peek()
    }

    /// Number of pending events.
    pub fn pending_event_count(&self) -> usize {
        self.events.len()
    }

    // ===================================================================
    // Direct subsystem access (for advanced use / testing)
    // ===================================================================

    /// Get a reference to the transport controller.
    pub fn transport(&self) -> &TransportController {
        &self.transport
    }

    /// Get a mutable reference to the transport controller.
    pub fn transport_mut(&mut self) -> &mut TransportController {
        &mut self.transport
    }

    /// Get a reference to the mixer.
    pub fn mixer(&self) -> &Mixer {
        &self.mixer
    }

    /// Get a mutable reference to the mixer.
    pub fn mixer_mut(&mut self) -> &mut Mixer {
        &mut self.mixer
    }

    /// Get a reference to the tempo controller.
    pub fn tempo_controller(&self) -> &TempoController {
        &self.tempo
    }

    /// Get a mutable reference to the tempo controller.
    pub fn tempo_controller_mut(&mut self) -> &mut TempoController {
        &mut self.tempo
    }

    /// Get a reference to the beat tracker.
    pub fn beat_tracker(&self) -> &BeatTracker {
        &self.beat_tracker
    }

    /// Get a mutable reference to the beat tracker.
    pub fn beat_tracker_mut(&mut self) -> &mut BeatTracker {
        &mut self.beat_tracker
    }

    /// Get a reference to the timeline.
    pub fn timeline(&self) -> &BeatTimeline {
        &self.timeline
    }

    /// Get a mutable reference to the timeline.
    pub fn timeline_mut(&mut self) -> &mut BeatTimeline {
        &mut self.timeline
    }

    /// Get a reference to the PCM cache.
    pub fn pcm_cache(&self) -> &PcmCache {
        &self.pcm_cache
    }

    /// Get a mutable reference to the PCM cache.
    pub fn pcm_cache_mut(&mut self) -> &mut PcmCache {
        &mut self.pcm_cache
    }

    /// Get a reference to a track by ID.
    pub fn track(&self, track_id: &str) -> Option<&Track> {
        self.tracks.get(track_id)
    }

    /// Get a mutable reference to a track by ID.
    pub fn track_mut(&mut self, track_id: &str) -> Option<&mut Track> {
        self.tracks.get_mut(track_id)
    }

    // ===================================================================
    // Internal helpers
    // ===================================================================

    /// Transition the engine state with validation.
    fn transition_state(&mut self, to: EngineState) -> Result<(), EngineError> {
        if !self.state.can_transition_to(to) {
            return Err(EngineError::InvalidStateTransition {
                from: self.state,
                to,
            });
        }
        self.state = to;
        Ok(())
    }

    /// Require the engine to be in at least the given state.
    fn require_min_state(
        &self,
        min_state: EngineState,
        operation: &'static str,
    ) -> Result<(), EngineError> {
        let state_order = |s: EngineState| -> u8 {
            match s {
                EngineState::Uninitialized => 0,
                EngineState::Initializing => 1,
                EngineState::ServicesReady => 2,
                EngineState::EnginesReady => 3,
                EngineState::Ready => 4,
                EngineState::Suspended => 4, // Post-Ready state — operations valid in Ready should work in Suspended
                EngineState::Error => 0,
                EngineState::Destroying => 0,
            }
        };

        if state_order(self.state) < state_order(min_state) {
            return Err(EngineError::InvalidState {
                current: self.state,
                operation,
            });
        }

        Ok(())
    }

    /// Execute a transport action.
    fn handle_transport_action(&mut self, action: TransportAction) {
        match action {
            TransportAction::Initialize => {
                // Start beat tracking and timeline
                self.beat_tracker.start();
                self.timeline.initialize();
                self.events.push(EngineEvent::TransportChanged {
                    state: TransportState::Playing,
                });
                self.events.push(EngineEvent::PlaybackStarted);
                self.events.push(EngineEvent::PlaybackStateChange {
                    is_playing: true,
                    is_paused: false,
                });
            }
            TransportAction::Resume => {
                self.beat_tracker.start();
                self.events.push(EngineEvent::TransportChanged {
                    state: TransportState::Playing,
                });
                self.events.push(EngineEvent::PlaybackStarted);
                self.events.push(EngineEvent::PlaybackStateChange {
                    is_playing: true,
                    is_paused: false,
                });
            }
            TransportAction::Pause => {
                if self.beat_tracker.is_tracking() {
                    self.beat_tracker.stop();
                }
                self.events.push(EngineEvent::TransportChanged {
                    state: TransportState::Paused,
                });
            }
            TransportAction::Shutdown => {
                if self.beat_tracker.is_tracking() {
                    self.beat_tracker.stop();
                }
                self.timeline.reset();
                self.events.push(EngineEvent::TransportChanged {
                    state: TransportState::Stopped,
                });
            }
            TransportAction::None => {}
        }
    }

    /// Check if all registered tracks have been loaded.
    fn all_tracks_loaded(&self) -> bool {
        !self.tracks.is_empty()
            && self.tracks.values().all(|t| {
                matches!(
                    t.status(),
                    TrackStatus::Ready
                        | TrackStatus::Playing
                        | TrackStatus::Paused
                        | TrackStatus::Preloaded
                )
            })
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::EngineEvent;

    // -- Helper --

    fn make_pcm_data(channels: u32, length: u64, sample_rate: u32) -> PcmData {
        let channel_data: Vec<Vec<f32>> = (0..channels)
            .map(|_| vec![0.0f32; length as usize])
            .collect();
        PcmData {
            channel_data,
            sample_rate,
            num_channels: channels,
            length_samples: length,
        }
    }

    fn initialized_engine() -> MaestroEngine {
        let mut engine = MaestroEngine::new(48000);
        engine.initialize().unwrap();
        engine.drain_events(); // Clear initialization events
        engine
    }

    fn engine_with_track() -> MaestroEngine {
        let mut engine = initialized_engine();
        engine
            .create_track("drums", "http://example.com/drums.m4a", None, Some(120.0))
            .unwrap();
        engine
            .load_track_audio(
                "drums",
                VariationType::Default,
                make_pcm_data(2, 48000 * 10, 48000),
                "http://example.com/drums.m4a",
            )
            .unwrap();
        engine.arm_track("drums").unwrap();
        engine.drain_events(); // Clear setup events
        engine
    }

    fn engine_with_two_tracks() -> MaestroEngine {
        let mut engine = initialized_engine();
        engine
            .create_track("drums", "http://example.com/drums.m4a", None, Some(120.0))
            .unwrap();
        engine
            .load_track_audio(
                "drums",
                VariationType::Default,
                make_pcm_data(2, 48000 * 10, 48000),
                "http://example.com/drums.m4a",
            )
            .unwrap();
        engine.arm_track("drums").unwrap();

        engine
            .create_track("bass", "http://example.com/bass.m4a", None, Some(120.0))
            .unwrap();
        engine
            .load_track_audio(
                "bass",
                VariationType::Default,
                make_pcm_data(2, 48000 * 10, 48000),
                "http://example.com/bass.m4a",
            )
            .unwrap();
        engine.arm_track("bass").unwrap();

        engine.drain_events();
        engine
    }

    // -- Construction & lifecycle -----------------------------------------

    #[test]
    fn new_engine_starts_uninitialized() {
        let engine = MaestroEngine::new(48000);
        assert_eq!(engine.state(), EngineState::Uninitialized);
        assert_eq!(engine.sample_rate(), 48000);
        assert_eq!(engine.track_count(), 0);
    }

    #[test]
    fn initialize_transitions_to_services_ready() {
        let mut engine = MaestroEngine::new(48000);
        engine.initialize().unwrap();
        assert_eq!(engine.state(), EngineState::ServicesReady);

        let events = engine.drain_events();
        assert!(events
            .iter()
            .any(|e| matches!(e, EngineEvent::Initialized { sample_rate: 48000 })));
    }

    #[test]
    fn double_initialize_fails() {
        let mut engine = MaestroEngine::new(48000);
        engine.initialize().unwrap();
        assert!(engine.initialize().is_err());
    }

    #[test]
    fn destroy_resets_everything() {
        let mut engine = engine_with_track();
        engine.play(SafetyPreset::Permissive).unwrap();
        engine.drain_events();

        engine.destroy();
        assert_eq!(engine.state(), EngineState::Destroying);
        assert_eq!(engine.track_count(), 0);
        assert!(engine.is_stopped());
    }

    #[test]
    fn suspend_and_resume() {
        let mut engine = engine_with_track();

        // Need to be in Ready state to suspend
        assert_eq!(engine.state(), EngineState::Ready);
        engine.suspend().unwrap();
        assert_eq!(engine.state(), EngineState::Suspended);

        engine.resume().unwrap();
        assert_eq!(engine.state(), EngineState::Ready);
    }

    #[test]
    fn operations_work_while_suspended() {
        let mut engine = engine_with_track();
        engine.suspend().unwrap();
        assert_eq!(engine.state(), EngineState::Suspended);

        // Operations requiring EnginesReady should succeed in Suspended state
        // (Suspended is post-Ready, so it ranks >= EnginesReady)
        engine.play(SafetyPreset::Permissive).unwrap();
        assert!(engine.is_playing());

        engine.pause().unwrap();
        assert!(engine.is_paused());

        engine.stop().unwrap();
        assert!(engine.is_stopped());

        // Track creation requires ServicesReady — should also work
        engine
            .create_track("extra", "url", None, Some(120.0))
            .unwrap();
    }

    // -- Track management ------------------------------------------------

    #[test]
    fn create_track_in_services_ready() {
        let mut engine = initialized_engine();
        engine
            .create_track("drums", "http://example.com/drums.m4a", None, Some(120.0))
            .unwrap();

        assert_eq!(engine.track_count(), 1);
        // First track advances state to EnginesReady
        assert_eq!(engine.state(), EngineState::EnginesReady);

        let events = engine.drain_events();
        assert!(events.iter().any(|e| matches!(
            e,
            EngineEvent::TrackCreated { track_id } if track_id == "drums"
        )));
    }

    #[test]
    fn create_duplicate_track_fails() {
        let mut engine = initialized_engine();
        engine
            .create_track("drums", "http://example.com/drums.m4a", None, None)
            .unwrap();
        let result = engine.create_track("drums", "http://example.com/drums.m4a", None, None);
        assert!(matches!(result, Err(EngineError::TrackAlreadyExists(_))));
    }

    #[test]
    fn create_track_before_init_fails() {
        let mut engine = MaestroEngine::new(48000);
        let result = engine.create_track("drums", "url", None, None);
        assert!(matches!(result, Err(EngineError::InvalidState { .. })));
    }

    #[test]
    fn remove_track() {
        let mut engine = engine_with_track();
        engine.remove_track("drums").unwrap();
        assert_eq!(engine.track_count(), 0);

        let events = engine.drain_events();
        assert!(events.iter().any(|e| matches!(
            e,
            EngineEvent::TrackRemoved { track_id } if track_id == "drums"
        )));
    }

    #[test]
    fn remove_nonexistent_track_fails() {
        let mut engine = initialized_engine();
        assert!(matches!(
            engine.remove_track("nope"),
            Err(EngineError::TrackNotFound(_))
        ));
    }

    #[test]
    fn load_track_audio_transitions_to_ready() {
        let mut engine = initialized_engine();
        engine
            .create_track("drums", "http://example.com/drums.m4a", None, None)
            .unwrap();
        engine.drain_events();

        engine
            .load_track_audio(
                "drums",
                VariationType::Default,
                make_pcm_data(2, 48000 * 5, 48000),
                "http://example.com/drums.m4a",
            )
            .unwrap();

        let track = engine.track("drums").unwrap();
        assert_eq!(track.status(), TrackStatus::Ready);

        let events = engine.drain_events();
        assert!(events.iter().any(|e| matches!(
            e,
            EngineEvent::TrackLoaded { track_id } if track_id == "drums"
        )));
    }

    #[test]
    fn all_tracks_loaded_emits_event() {
        let mut engine = initialized_engine();
        engine
            .create_track("drums", "url1", None, None)
            .unwrap();
        engine
            .create_track("bass", "url2", None, None)
            .unwrap();
        engine.drain_events();

        engine
            .load_track_audio(
                "drums",
                VariationType::Default,
                make_pcm_data(2, 48000, 48000),
                "url1",
            )
            .unwrap();
        let events = engine.drain_events();
        // Not all loaded yet
        assert!(!events
            .iter()
            .any(|e| matches!(e, EngineEvent::AllTracksLoaded)));

        engine
            .load_track_audio(
                "bass",
                VariationType::Default,
                make_pcm_data(2, 48000, 48000),
                "url2",
            )
            .unwrap();
        let events = engine.drain_events();
        assert!(events
            .iter()
            .any(|e| matches!(e, EngineEvent::AllTracksLoaded)));
    }

    // -- Arm/disarm/toggle -----------------------------------------------

    #[test]
    fn arm_and_disarm_track() {
        let mut engine = engine_with_track();

        // Already armed from setup — disarm
        engine.disarm_track("drums").unwrap();
        let track = engine.track("drums").unwrap();
        assert!(!track.controls().armed);

        let events = engine.drain_events();
        assert!(events.iter().any(|e| matches!(
            e,
            EngineEvent::TrackDisarmed { track_id } if track_id == "drums"
        )));

        // Re-arm
        engine.arm_track("drums").unwrap();
        let track = engine.track("drums").unwrap();
        assert!(track.controls().armed);

        let events = engine.drain_events();
        assert!(events.iter().any(|e| matches!(
            e,
            EngineEvent::TrackArmed { track_id } if track_id == "drums"
        )));
    }

    #[test]
    fn toggle_track() {
        let mut engine = engine_with_track();

        // Armed → disarmed
        let result = engine.toggle_track("drums").unwrap();
        assert!(!result);
        assert!(!engine.track("drums").unwrap().controls().armed);

        // Disarmed → armed
        let result = engine.toggle_track("drums").unwrap();
        assert!(result);
        assert!(engine.track("drums").unwrap().controls().armed);
    }

    // -- Transport (play/pause/stop) -------------------------------------

    #[test]
    fn play_starts_armed_tracks() {
        let mut engine = engine_with_track();
        engine.play(SafetyPreset::Permissive).unwrap();

        let track = engine.track("drums").unwrap();
        assert_eq!(track.status(), TrackStatus::Playing);
        assert!(engine.is_playing());
        assert_eq!(engine.active_track_count(), 1);

        let events = engine.drain_events();
        assert!(events.iter().any(|e| matches!(
            e,
            EngineEvent::TrackStarted { track_id, active_track_count: 1 } if track_id == "drums"
        )));
        assert!(events
            .iter()
            .any(|e| matches!(e, EngineEvent::PlaybackStarted)));
    }

    #[test]
    fn play_ignores_unarmed_tracks() {
        let mut engine = engine_with_track();
        engine.disarm_track("drums").unwrap();
        engine.drain_events();

        engine.play(SafetyPreset::Permissive).unwrap();
        let track = engine.track("drums").unwrap();
        assert_eq!(track.status(), TrackStatus::Ready); // Not playing
        assert!(engine.is_stopped());
    }

    #[test]
    fn pause_pauses_all_playing_tracks() {
        let mut engine = engine_with_two_tracks();
        engine.play(SafetyPreset::Permissive).unwrap();
        engine.drain_events();

        engine.pause().unwrap();

        assert_eq!(
            engine.track("drums").unwrap().status(),
            TrackStatus::Paused
        );
        assert_eq!(
            engine.track("bass").unwrap().status(),
            TrackStatus::Paused
        );

        let events = engine.drain_events();
        assert!(events
            .iter()
            .any(|e| matches!(e, EngineEvent::PlaybackPaused)));
    }

    #[test]
    fn stop_resets_tracks_to_ready() {
        let mut engine = engine_with_track();
        engine.play(SafetyPreset::Permissive).unwrap();
        engine.drain_events();

        engine.stop().unwrap();

        let track = engine.track("drums").unwrap();
        assert_eq!(track.status(), TrackStatus::Ready);
        assert!(engine.is_stopped());

        let events = engine.drain_events();
        assert!(events
            .iter()
            .any(|e| matches!(e, EngineEvent::PlaybackStopped)));
    }

    #[test]
    fn play_before_engines_ready_fails() {
        let mut engine = initialized_engine();
        // No tracks created yet — still ServicesReady
        let result = engine.play(SafetyPreset::Permissive);
        assert!(matches!(result, Err(EngineError::InvalidState { .. })));
    }

    #[test]
    fn transport_state_transitions() {
        let mut engine = engine_with_track();

        assert_eq!(engine.transport_state(), TransportState::Stopped);

        engine.play(SafetyPreset::Permissive).unwrap();
        assert_eq!(engine.transport_state(), TransportState::Playing);

        engine.pause().unwrap();
        assert_eq!(engine.transport_state(), TransportState::Paused);

        engine.stop().unwrap();
        assert_eq!(engine.transport_state(), TransportState::Stopped);
    }

    // -- Tempo control ---------------------------------------------------

    #[test]
    fn set_tempo_emits_event() {
        let mut engine = initialized_engine();
        engine.set_tempo(140.0).unwrap();

        let events = engine.drain_events();
        assert!(events.iter().any(|e| matches!(
            e,
            EngineEvent::TempoChanged { tempo, .. } if (*tempo - 140.0).abs() < 0.01
        )));
    }

    #[test]
    fn current_bpm_reflects_tempo() {
        let mut engine = initialized_engine();
        engine.set_tempo(95.0).unwrap();
        assert!((engine.current_bpm() - 95.0).abs() < 0.01);
    }

    // -- Volume control --------------------------------------------------

    #[test]
    fn set_track_volume_updates_track_and_mixer() {
        let mut engine = engine_with_track();
        engine.set_track_volume("drums", 0.5).unwrap();

        let track = engine.track("drums").unwrap();
        assert!((track.controls().volume - 0.5).abs() < 0.01);

        let events = engine.drain_events();
        assert!(events.iter().any(|e| matches!(
            e,
            EngineEvent::TrackVolumeChanged { track_id, volume }
            if track_id == "drums" && (*volume - 0.5).abs() < 0.01
        )));
    }

    #[test]
    fn set_volume_nonexistent_track_fails() {
        let mut engine = initialized_engine();
        assert!(matches!(
            engine.set_track_volume("nope", 0.5),
            Err(EngineError::TrackNotFound(_))
        ));
    }

    #[test]
    fn mute_unmute_track() {
        let mut engine = engine_with_track();
        engine.mute_track("drums").unwrap();
        engine.unmute_track("drums").unwrap();
        // No error = success
    }

    // -- Pitch control ---------------------------------------------------

    #[test]
    fn set_track_pitch_emits_event() {
        let mut engine = engine_with_track();
        engine.set_track_pitch_cents("drums", 200.0).unwrap();

        let track = engine.track("drums").unwrap();
        assert!((track.controls().pitch_cents - 200.0).abs() < 0.01);

        let events = engine.drain_events();
        assert!(events.iter().any(|e| matches!(
            e,
            EngineEvent::TrackPitchChanged { track_id, offset_cents, offset_semitones }
            if track_id == "drums"
                && (*offset_cents - 200.0).abs() < 0.01
                && (*offset_semitones - 2.0).abs() < 0.01
        )));
    }

    // -- Variation control -----------------------------------------------

    #[test]
    fn set_track_variation_emits_event() {
        let mut engine = engine_with_track();
        engine
            .set_track_variation("drums", 2, "Complex")
            .unwrap();

        let track = engine.track("drums").unwrap();
        assert_eq!(track.controls().current_variation, 2);

        let events = engine.drain_events();
        assert!(events.iter().any(|e| matches!(
            e,
            EngineEvent::VariationChanged {
                track_id,
                previous_variation_index: 0,
                new_variation_index: 2,
                variation_name,
            }
            if track_id == "drums" && variation_name == "Complex"
        )));
    }

    // -- Beat tracking ---------------------------------------------------

    #[test]
    fn update_beat_position_emits_events() {
        let mut engine = engine_with_track();
        engine.play(SafetyPreset::Permissive).unwrap();
        engine.drain_events();

        // BeatTracker initializes with previous_beat=1. BeatInfo at exact_beat=0.5
        // computes beat=1 (1-based), so we need to progress past beat boundaries.
        // At exact_beat=2.0 → absolute_beat=2, beats_completed=1, beat_in_bar=1,
        // current_beat=2 which differs from previous_beat=1 → beat change detected.
        engine.update_beat_position(2.0);
        engine.update_beat_position(3.5);
        engine.update_beat_position(5.0);

        let events = engine.drain_events();
        assert!(events
            .iter()
            .any(|e| matches!(e, EngineEvent::BeatPrecise { .. })));
    }

    // -- State queries ---------------------------------------------------

    #[test]
    fn get_track_info() {
        let engine = engine_with_track();
        let info = engine.get_track_info("drums").unwrap();
        assert_eq!(info.id, "drums");
        assert_eq!(info.status, TrackStatus::Ready);
        assert!(info.armed);
        assert!((info.volume - 0.75).abs() < 0.01); // Default volume
    }

    #[test]
    fn get_all_tracks() {
        let engine = engine_with_two_tracks();
        let all = engine.get_all_tracks();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn get_beat_info() {
        let engine = initialized_engine();
        // TempoController not initialized yet — bpm should be 0.0 (safe default)
        let info = engine.get_beat_info();
        assert_eq!(info.time_signature_numerator, 4);
        assert_eq!(info.time_signature_denominator, 4);
        assert!((info.bpm - 0.0).abs() < 0.01);
    }

    #[test]
    fn get_beat_info_with_tempo() {
        let mut engine = initialized_engine();
        engine.set_tempo(120.0).unwrap();
        let info = engine.get_beat_info();
        assert_eq!(info.time_signature_numerator, 4);
        assert!((info.bpm - 120.0).abs() < 0.01);
    }

    // -- Direct subsystem access -----------------------------------------

    #[test]
    fn direct_subsystem_access() {
        let mut engine = initialized_engine();

        // Transport
        assert_eq!(engine.transport().state(), TransportState::Stopped);
        engine.transport_mut().force_state(TransportState::Playing);
        assert!(engine.transport().is_playing());
        engine.transport_mut().reset(); // Clean up

        // Mixer
        assert!((engine.mixer().master_volume() - 1.0).abs() < 0.01);

        // Tempo — starts uninitialized (current_tempo == 0), so set it first
        engine.set_tempo(120.0).unwrap();
        assert!(engine.tempo_controller().current_tempo().is_ok());

        // Beat tracker
        assert!(!engine.beat_tracker().is_tracking());

        // Timeline
        assert!(engine.timeline().is_initialized());

        // PCM cache
        assert!(engine.pcm_cache().is_empty());
    }

    // -- Event drain mechanics -------------------------------------------

    #[test]
    fn drain_events_clears_pending() {
        let mut engine = MaestroEngine::new(48000);
        engine.initialize().unwrap();
        // Don't drain yet — should have Initialized event
        assert!(!engine.drain_events().is_empty());

        // After drain, generate new events
        engine
            .create_track("drums", "url", None, None)
            .unwrap();
        assert!(engine.pending_event_count() > 0);

        let events = engine.drain_events();
        assert!(!events.is_empty());
        assert_eq!(engine.pending_event_count(), 0);
    }

    #[test]
    fn peek_events_does_not_drain() {
        let mut engine = MaestroEngine::new(48000);
        engine.initialize().unwrap();

        let peeked = engine.peek_events();
        assert!(!peeked.is_empty());

        // Still there
        assert_eq!(engine.pending_event_count(), peeked.len());
    }

    // -- Error cases -----------------------------------------------------

    #[test]
    fn engine_error_display() {
        let err = EngineError::InvalidState {
            current: EngineState::Uninitialized,
            operation: "play",
        };
        assert_eq!(
            err.to_string(),
            "Cannot 'play' in engine state 'uninitialized'"
        );

        let err = EngineError::TrackNotFound("xyz".into());
        assert_eq!(err.to_string(), "Track 'xyz' not found");

        let err = EngineError::TrackAlreadyExists("abc".into());
        assert_eq!(err.to_string(), "Track 'abc' already exists");
    }

    // -- Remove playing track cleans up transport ------------------------

    #[test]
    fn remove_playing_track_updates_transport() {
        let mut engine = engine_with_track();
        engine.play(SafetyPreset::Permissive).unwrap();
        assert!(engine.is_playing());
        engine.drain_events();

        engine.remove_track("drums").unwrap();
        assert!(engine.is_stopped());
        assert_eq!(engine.active_track_count(), 0);
    }

    // -- Debug formatting ------------------------------------------------

    #[test]
    fn engine_debug_format() {
        let engine = MaestroEngine::new(48000);
        let debug = format!("{:?}", engine);
        assert!(debug.contains("MaestroEngine"));
        assert!(debug.contains("48000"));
        // Debug derive uses variant name "Uninitialized", not Display "uninitialized"
        assert!(debug.contains("Uninitialized"));
    }
}
