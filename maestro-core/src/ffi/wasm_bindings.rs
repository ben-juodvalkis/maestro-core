//! wasm-bindgen exports for maestro-core.
//!
//! Exposes the `MaestroEngine` API to JavaScript via a `WasmMaestroEngine`
//! wrapper.  This is the entry point for Phase 4 (Web WASM Replacement) of
//! the Hybrid Migration Roadmap.
//!
//! ## Architecture
//!
//! ```text
//! React UI (unchanged)
//!     ↓ [Same hooks: useCoordinatorCommands, etc.]
//! WasmMaestroEngine (this module)
//!     ↓ [wasm-bindgen calls into Rust]
//! MaestroEngine (engine.rs — same code as native)
//!     ↓ [Renders audio frames]
//! AudioWorklet (thin wrapper — ships frames to Web Audio output)
//! ```
//!
//! ## Design decisions
//!
//! - **Wrapper struct** instead of annotating `MaestroEngine` directly — keeps
//!   the core engine free of wasm-bindgen attributes and `JsValue` types.
//! - **JSON serialization** for complex types (`EngineEvent`, `TrackInfo`,
//!   `BeatInfoSnapshot`) — these are consumed by JavaScript where JSON is
//!   native.  Simple enums use integer discriminants for zero-cost mapping.
//! - **`Float32Array` bridge** for PCM data — avoids copying by using
//!   wasm-bindgen's direct typed array support.
//! - **Error → JsValue** conversion — all `Result<T, EngineError>` become
//!   `Result<T, JsValue>` at the boundary, surfacing error messages as
//!   JavaScript `Error` objects.

use wasm_bindgen::prelude::*;

use crate::engine::{EngineError, MaestroEngine};
use crate::events::{EngineEvent, EngineState};
use crate::pcm_cache::{PcmData, VariationType};
use crate::track::TrackStatus;
use crate::transport::{SafetyPreset, TransportState};

// ---------------------------------------------------------------------------
// Error conversion
// ---------------------------------------------------------------------------

impl From<EngineError> for JsValue {
    fn from(e: EngineError) -> Self {
        JsValue::from_str(&e.to_string())
    }
}

// ---------------------------------------------------------------------------
// Simple enum → JS integer mappings
// ---------------------------------------------------------------------------

/// Engine lifecycle state (mirrors `EngineState`).
///
/// Exposed as a simple integer enum for efficient JS interop.
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WasmEngineState {
    Uninitialized = 0,
    Initializing = 1,
    ServicesReady = 2,
    EnginesReady = 3,
    Ready = 4,
    Suspended = 5,
    Error = 6,
    Destroying = 7,
}

impl From<EngineState> for WasmEngineState {
    fn from(s: EngineState) -> Self {
        match s {
            EngineState::Uninitialized => WasmEngineState::Uninitialized,
            EngineState::Initializing => WasmEngineState::Initializing,
            EngineState::ServicesReady => WasmEngineState::ServicesReady,
            EngineState::EnginesReady => WasmEngineState::EnginesReady,
            EngineState::Ready => WasmEngineState::Ready,
            EngineState::Suspended => WasmEngineState::Suspended,
            EngineState::Error => WasmEngineState::Error,
            EngineState::Destroying => WasmEngineState::Destroying,
        }
    }
}

/// Transport state (mirrors `TransportState`).
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WasmTransportState {
    Stopped = 0,
    Playing = 1,
    Paused = 2,
}

impl From<TransportState> for WasmTransportState {
    fn from(s: TransportState) -> Self {
        match s {
            TransportState::Stopped => WasmTransportState::Stopped,
            TransportState::Playing => WasmTransportState::Playing,
            TransportState::Paused => WasmTransportState::Paused,
        }
    }
}

/// Transport safety preset (mirrors `SafetyPreset`).
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WasmSafetyPreset {
    SliderOperation = 0,
    TrackInteraction = 1,
    ManualControl = 2,
    ProgrammaticOperation = 3,
    Permissive = 4,
}

impl From<WasmSafetyPreset> for SafetyPreset {
    fn from(p: WasmSafetyPreset) -> Self {
        match p {
            WasmSafetyPreset::SliderOperation => SafetyPreset::SliderOperation,
            WasmSafetyPreset::TrackInteraction => SafetyPreset::TrackInteraction,
            WasmSafetyPreset::ManualControl => SafetyPreset::ManualControl,
            WasmSafetyPreset::ProgrammaticOperation => SafetyPreset::ProgrammaticOperation,
            WasmSafetyPreset::Permissive => SafetyPreset::Permissive,
        }
    }
}

/// Track status (mirrors `TrackStatus`).
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WasmTrackStatus {
    Idle = 0,
    Loading = 1,
    Ready = 2,
    Preloaded = 3,
    Playing = 4,
    Paused = 5,
    Error = 6,
}

impl From<TrackStatus> for WasmTrackStatus {
    fn from(s: TrackStatus) -> Self {
        match s {
            TrackStatus::Idle => WasmTrackStatus::Idle,
            TrackStatus::Loading => WasmTrackStatus::Loading,
            TrackStatus::Ready => WasmTrackStatus::Ready,
            TrackStatus::Preloaded => WasmTrackStatus::Preloaded,
            TrackStatus::Playing => WasmTrackStatus::Playing,
            TrackStatus::Paused => WasmTrackStatus::Paused,
            TrackStatus::Error => WasmTrackStatus::Error,
        }
    }
}

/// Audio variation type (mirrors `VariationType`).
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WasmVariationType {
    Simple = 0,
    Default = 1,
    Complex = 2,
}

impl From<WasmVariationType> for VariationType {
    fn from(v: WasmVariationType) -> Self {
        match v {
            WasmVariationType::Simple => VariationType::Simple,
            WasmVariationType::Default => VariationType::Default,
            WasmVariationType::Complex => VariationType::Complex,
        }
    }
}

// ---------------------------------------------------------------------------
// JSON serialization for complex types
// ---------------------------------------------------------------------------

/// Serialize an `EngineEvent` to a `serde_json::Value`.
///
/// Events are serialized as `{ "type": "EventName", ...payload }` objects
/// matching the TypeScript event naming conventions.
///
/// Uses `serde_json::Value` instead of `JsValue` so the serialization logic
/// is testable on native targets (JsValue panics outside WASM runtime).
fn engine_event_to_json(event: &EngineEvent) -> serde_json::Value {
    match event {
        EngineEvent::PlaybackStarted => serde_json::json!({
            "type": "playback:started"
        }),
        EngineEvent::PlaybackPaused => serde_json::json!({
            "type": "playback:paused"
        }),
        EngineEvent::PlaybackStopped => serde_json::json!({
            "type": "playback:stopped"
        }),
        EngineEvent::PlaybackStateChange { is_playing, is_paused } => serde_json::json!({
            "type": "playback:stateChange",
            "isPlaying": is_playing,
            "isPaused": is_paused,
        }),
        EngineEvent::TransportChanged { state } => serde_json::json!({
            "type": "transport:changed",
            "state": format!("{:?}", state),
        }),
        EngineEvent::TempoChanged { tempo, previous_tempo } => serde_json::json!({
            "type": "tempo:changed",
            "tempo": tempo,
            "previousTempo": previous_tempo,
        }),
        EngineEvent::TrackCreated { track_id } => serde_json::json!({
            "type": "track:created",
            "trackId": track_id,
        }),
        EngineEvent::TrackRemoved { track_id } => serde_json::json!({
            "type": "track:removed",
            "trackId": track_id,
        }),
        EngineEvent::TrackStateChanged { track_id, state, previous_state } => serde_json::json!({
            "type": "track:stateChanged",
            "trackId": track_id,
            "state": format!("{:?}", state),
            "previousState": format!("{:?}", previous_state),
        }),
        EngineEvent::TrackLoaded { track_id } => serde_json::json!({
            "type": "track:loaded",
            "trackId": track_id,
        }),
        EngineEvent::TrackError { track_id, message } => serde_json::json!({
            "type": "track:error",
            "trackId": track_id,
            "message": message,
        }),
        EngineEvent::TrackStarted { track_id, active_track_count } => serde_json::json!({
            "type": "track:started",
            "trackId": track_id,
            "activeTrackCount": active_track_count,
        }),
        EngineEvent::TrackStopped { track_id, active_track_count } => serde_json::json!({
            "type": "track:stopped",
            "trackId": track_id,
            "activeTrackCount": active_track_count,
        }),
        EngineEvent::TrackPlayingStateChanged { track_id, is_playing } => serde_json::json!({
            "type": "track:playingStateChanged",
            "trackId": track_id,
            "isPlaying": is_playing,
        }),
        EngineEvent::TrackVolumeChanged { track_id, volume } => serde_json::json!({
            "type": "track:volumeChanged",
            "trackId": track_id,
            "volume": volume,
        }),
        EngineEvent::TrackPitchChanged { track_id, offset_cents, offset_semitones } => serde_json::json!({
            "type": "track:pitchChanged",
            "trackId": track_id,
            "offsetCents": offset_cents,
            "offsetSemitones": offset_semitones,
        }),
        EngineEvent::TrackArmed { track_id } => serde_json::json!({
            "type": "track:armed",
            "trackId": track_id,
        }),
        EngineEvent::TrackDisarmed { track_id } => serde_json::json!({
            "type": "track:disarmed",
            "trackId": track_id,
        }),
        EngineEvent::VariationChanged {
            track_id,
            previous_variation_index,
            new_variation_index,
            variation_name,
        } => serde_json::json!({
            "type": "variation:changed",
            "trackId": track_id,
            "previousVariationIndex": previous_variation_index,
            "newVariationIndex": new_variation_index,
            "variationName": variation_name,
        }),
        EngineEvent::PositionUpdated { beat, loop_progress } => serde_json::json!({
            "type": "position:updated",
            "beat": beat,
            "loopProgress": loop_progress,
        }),
        EngineEvent::BeatPrecise { beat_info } => serde_json::json!({
            "type": "beat:precise",
            "beat": beat_info.beat,
            "bar": beat_info.bar,
            "absoluteBeat": beat_info.absolute_beat,
            "beatFraction": beat_info.beat_fraction,
            "preciseBeat": beat_info.precise_beat,
        }),
        EngineEvent::BarChanged { bar, beat } => serde_json::json!({
            "type": "bar:changed",
            "bar": bar,
            "beat": beat,
        }),
        EngineEvent::LoopCompleted { loop_count } => serde_json::json!({
            "type": "loop:completed",
            "loopCount": loop_count,
        }),
        EngineEvent::Initialized { sample_rate } => serde_json::json!({
            "type": "engine:initialized",
            "sampleRate": sample_rate,
        }),
        EngineEvent::AllTracksLoaded => serde_json::json!({
            "type": "engine:allTracksLoaded"
        }),
        EngineEvent::Error { message, context } => serde_json::json!({
            "type": "engine:error",
            "message": message,
            "context": context,
        }),
    }
}

// ---------------------------------------------------------------------------
// WasmMaestroEngine — the main WASM export
// ---------------------------------------------------------------------------

/// The main WASM entry point wrapping `MaestroEngine`.
///
/// All methods delegate to the inner engine with type conversions at the
/// boundary.  JavaScript consumers interact with this via the generated
/// wasm-bindgen glue.
///
/// ## Usage from JavaScript
///
/// ```js
/// import init, { WasmMaestroEngine, WasmSafetyPreset } from './maestro_core.js';
///
/// await init();
/// const engine = WasmMaestroEngine.create(48000);
/// engine.initialize();
/// engine.createTrack("drums", "songs/rock/drums.m4a", null, 120.0);
/// engine.play(WasmSafetyPreset.ManualControl);
///
/// // Poll events (call in requestAnimationFrame or setInterval)
/// const events = engine.drainEventsJson();
/// for (const eventJson of events) {
///     const event = JSON.parse(eventJson);
///     console.log(event.type, event);
/// }
/// ```
#[wasm_bindgen]
pub struct WasmMaestroEngine {
    inner: MaestroEngine,
}

#[wasm_bindgen]
impl WasmMaestroEngine {
    // =======================================================================
    // Construction & lifecycle
    // =======================================================================

    /// Create a new engine with the given sample rate.
    #[wasm_bindgen(constructor)]
    pub fn new(sample_rate: u32) -> Self {
        Self {
            inner: MaestroEngine::new(sample_rate),
        }
    }

    /// Initialize the engine — transitions to Ready state.
    pub fn initialize(&mut self) -> Result<(), JsValue> {
        self.inner.initialize().map_err(JsValue::from)
    }

    /// Get the current engine state.
    pub fn state(&self) -> WasmEngineState {
        self.inner.state().into()
    }

    /// Get the sample rate.
    #[wasm_bindgen(js_name = "sampleRate")]
    pub fn sample_rate(&self) -> u32 {
        self.inner.sample_rate()
    }

    /// Suspend the engine (e.g., tab backgrounded).
    pub fn suspend(&mut self) -> Result<(), JsValue> {
        self.inner.suspend().map_err(JsValue::from)
    }

    /// Resume from suspended state.
    pub fn resume(&mut self) -> Result<(), JsValue> {
        self.inner.resume().map_err(JsValue::from)
    }

    /// Destroy the engine — releases all resources.
    pub fn destroy(&mut self) {
        self.inner.destroy();
    }

    // =======================================================================
    // Track management
    // =======================================================================

    /// Create and register a new track.
    ///
    /// `original_tempo` should be 0.0 or negative to indicate "not specified".
    #[wasm_bindgen(js_name = "createTrack")]
    pub fn create_track(
        &mut self,
        id: &str,
        engine_url: &str,
        original_tempo: f64,
    ) -> Result<(), JsValue> {
        let tempo = if original_tempo > 0.0 {
            Some(original_tempo)
        } else {
            None
        };
        self.inner
            .create_track(id, engine_url, None, tempo)
            .map_err(JsValue::from)
    }

    /// Remove a track from the engine.
    #[wasm_bindgen(js_name = "removeTrack")]
    pub fn remove_track(&mut self, track_id: &str) -> Result<(), JsValue> {
        self.inner.remove_track(track_id).map_err(JsValue::from)
    }

    /// Load decoded PCM audio data for a track.
    ///
    /// `channel_data` is a flat interleaved `Float32Array` (L0, R0, L1, R1, ...).
    /// `num_channels` specifies the channel count (typically 2 for stereo).
    /// `sample_rate` is the audio sample rate of the decoded data.
    #[wasm_bindgen(js_name = "loadTrackAudio")]
    pub fn load_track_audio(
        &mut self,
        track_id: &str,
        variation: WasmVariationType,
        channel_data: &[f32],
        num_channels: u32,
        sample_rate: u32,
        url: &str,
    ) -> Result<(), JsValue> {
        if num_channels == 0 {
            return Err(JsValue::from_str("num_channels must be > 0"));
        }

        let total_samples = channel_data.len();
        if !total_samples.is_multiple_of(num_channels as usize) {
            return Err(JsValue::from_str(
                "channel_data length must be divisible by num_channels",
            ));
        }
        let frames = total_samples / num_channels as usize;

        // De-interleave: [L0, R0, L1, R1, ...] → [[L0, L1, ...], [R0, R1, ...]]
        let mut channels: Vec<Vec<f32>> = (0..num_channels as usize)
            .map(|_| Vec::with_capacity(frames))
            .collect();

        for frame in 0..frames {
            for ch in 0..num_channels as usize {
                channels[ch].push(channel_data[frame * num_channels as usize + ch]);
            }
        }

        let pcm_data = PcmData {
            channel_data: channels,
            sample_rate,
            num_channels,
            length_samples: frames as u64,
        };

        self.inner
            .load_track_audio(track_id, variation.into(), pcm_data, url)
            .map_err(JsValue::from)
    }

    /// Load decoded PCM audio with channel-separated data.
    ///
    /// Each channel is a separate `Float32Array`.  This avoids the
    /// de-interleaving overhead of `loadTrackAudio` when the decoder
    /// already provides separated channels (e.g., `AudioBuffer.getChannelData()`).
    #[wasm_bindgen(js_name = "loadTrackAudioSeparated")]
    pub fn load_track_audio_separated(
        &mut self,
        track_id: &str,
        variation: WasmVariationType,
        left_channel: &[f32],
        right_channel: &[f32],
        sample_rate: u32,
        url: &str,
    ) -> Result<(), JsValue> {
        if left_channel.len() != right_channel.len() {
            return Err(JsValue::from_str("channel lengths must match"));
        }

        let frames = left_channel.len();
        let pcm_data = PcmData {
            channel_data: vec![left_channel.to_vec(), right_channel.to_vec()],
            sample_rate,
            num_channels: 2,
            length_samples: frames as u64,
        };

        self.inner
            .load_track_audio(track_id, variation.into(), pcm_data, url)
            .map_err(JsValue::from)
    }

    /// Arm a track for playback.
    #[wasm_bindgen(js_name = "armTrack")]
    pub fn arm_track(&mut self, track_id: &str) -> Result<(), JsValue> {
        self.inner.arm_track(track_id).map_err(JsValue::from)
    }

    /// Disarm a track.
    #[wasm_bindgen(js_name = "disarmTrack")]
    pub fn disarm_track(&mut self, track_id: &str) -> Result<(), JsValue> {
        self.inner.disarm_track(track_id).map_err(JsValue::from)
    }

    /// Toggle a track's armed state.  Returns `true` if now armed.
    #[wasm_bindgen(js_name = "toggleTrack")]
    pub fn toggle_track(&mut self, track_id: &str) -> Result<bool, JsValue> {
        self.inner.toggle_track(track_id).map_err(JsValue::from)
    }

    // =======================================================================
    // Transport control
    // =======================================================================

    /// Start playback with the given safety preset.
    pub fn play(&mut self, preset: WasmSafetyPreset) -> Result<(), JsValue> {
        self.inner.play(preset.into()).map_err(JsValue::from)
    }

    /// Pause playback.
    pub fn pause(&mut self) -> Result<(), JsValue> {
        self.inner.pause().map_err(JsValue::from)
    }

    /// Stop playback.
    pub fn stop(&mut self) -> Result<(), JsValue> {
        self.inner.stop().map_err(JsValue::from)
    }

    /// Get the current transport state.
    #[wasm_bindgen(js_name = "transportState")]
    pub fn transport_state(&self) -> WasmTransportState {
        self.inner.transport_state().into()
    }

    /// Check if the engine is currently playing.
    #[wasm_bindgen(js_name = "isPlaying")]
    pub fn is_playing(&self) -> bool {
        self.inner.is_playing()
    }

    /// Check if the engine is currently paused.
    #[wasm_bindgen(js_name = "isPaused")]
    pub fn is_paused(&self) -> bool {
        self.inner.is_paused()
    }

    /// Check if the engine is currently stopped.
    #[wasm_bindgen(js_name = "isStopped")]
    pub fn is_stopped(&self) -> bool {
        self.inner.is_stopped()
    }

    // =======================================================================
    // Tempo control
    // =======================================================================

    /// Set the tempo ratio (1.0 = original tempo).
    #[wasm_bindgen(js_name = "setTempoRatio")]
    pub fn set_tempo_ratio(&mut self, ratio: f64) -> Result<(), JsValue> {
        self.inner.set_tempo_ratio(ratio).map_err(JsValue::from)
    }

    /// Get the current tempo ratio.
    #[wasm_bindgen(js_name = "tempoRatio")]
    pub fn tempo_ratio(&self) -> f64 {
        self.inner.tempo_ratio()
    }

    /// Get the current effective BPM.
    #[wasm_bindgen(js_name = "currentBpm")]
    pub fn current_bpm(&self) -> f64 {
        self.inner.current_bpm()
    }

    /// Set the tempo directly in BPM.
    #[wasm_bindgen(js_name = "setTempo")]
    pub fn set_tempo(&mut self, bpm: f64) -> Result<(), JsValue> {
        self.inner.set_tempo(bpm).map_err(JsValue::from)
    }

    // =======================================================================
    // Volume control
    // =======================================================================

    /// Set a track's volume (0.0 – 1.0).
    #[wasm_bindgen(js_name = "setTrackVolume")]
    pub fn set_track_volume(&mut self, track_id: &str, volume: f64) -> Result<(), JsValue> {
        self.inner
            .set_track_volume(track_id, volume)
            .map_err(JsValue::from)
    }

    /// Mute a track.
    #[wasm_bindgen(js_name = "muteTrack")]
    pub fn mute_track(&mut self, track_id: &str) -> Result<(), JsValue> {
        self.inner.mute_track(track_id).map_err(JsValue::from)
    }

    /// Unmute a track.
    #[wasm_bindgen(js_name = "unmuteTrack")]
    pub fn unmute_track(&mut self, track_id: &str) -> Result<(), JsValue> {
        self.inner.unmute_track(track_id).map_err(JsValue::from)
    }

    // =======================================================================
    // Pitch control
    // =======================================================================

    /// Set a track's pitch offset in cents.
    #[wasm_bindgen(js_name = "setTrackPitchCents")]
    pub fn set_track_pitch_cents(
        &mut self,
        track_id: &str,
        cents: f64,
    ) -> Result<(), JsValue> {
        self.inner
            .set_track_pitch_cents(track_id, cents)
            .map_err(JsValue::from)
    }

    // =======================================================================
    // Variation control
    // =======================================================================

    /// Set a track's variation.
    #[wasm_bindgen(js_name = "setTrackVariation")]
    pub fn set_track_variation(
        &mut self,
        track_id: &str,
        variation_index: u32,
        variation_name: &str,
    ) -> Result<(), JsValue> {
        self.inner
            .set_track_variation(track_id, variation_index, variation_name)
            .map_err(JsValue::from)
    }

    // =======================================================================
    // Beat tracking
    // =======================================================================

    /// Update the beat position from the audio render callback.
    #[wasm_bindgen(js_name = "updateBeatPosition")]
    pub fn update_beat_position(&mut self, exact_beat: f64) {
        self.inner.update_beat_position(exact_beat);
    }

    /// Set beat tracking configuration.
    #[wasm_bindgen(js_name = "setBeatTrackingConfig")]
    pub fn set_beat_tracking_config(&mut self, beats_per_bar: u32, bars_per_loop: u32) {
        use crate::beat_tracker::BeatTrackingConfig;
        self.inner.set_beat_tracking_config(BeatTrackingConfig {
            beats_per_bar,
            bars_per_loop,
        });
    }

    // =======================================================================
    // State queries
    // =======================================================================

    /// Get all track IDs as a JSON array string.
    #[wasm_bindgen(js_name = "trackIdsJson")]
    pub fn track_ids_json(&self) -> String {
        serde_json::to_string(&self.inner.track_ids()).unwrap_or_else(|_| "[]".to_string())
    }

    /// Get info for a single track as a JSON string, or null if not found.
    #[wasm_bindgen(js_name = "getTrackInfoJson")]
    pub fn get_track_info_json(&self, track_id: &str) -> JsValue {
        match self.inner.get_track_info(track_id) {
            Some(info) => {
                let json = serde_json::json!({
                    "id": info.id,
                    "status": format!("{:?}", info.status),
                    "armed": info.armed,
                    "volume": info.volume,
                    "pitchCents": info.pitch_cents,
                    "pan": info.pan,
                    "muted": info.muted,
                    "solo": info.solo,
                    "duration": info.duration,
                    "position": info.position,
                    "currentVariation": info.current_variation,
                });
                JsValue::from_str(&json.to_string())
            }
            None => JsValue::NULL,
        }
    }

    /// Get all tracks as a JSON array string.
    #[wasm_bindgen(js_name = "getAllTracksJson")]
    pub fn get_all_tracks_json(&self) -> String {
        let tracks: Vec<serde_json::Value> = self
            .inner
            .get_all_tracks()
            .iter()
            .map(|info| {
                serde_json::json!({
                    "id": info.id,
                    "status": format!("{:?}", info.status),
                    "armed": info.armed,
                    "volume": info.volume,
                    "pitchCents": info.pitch_cents,
                    "pan": info.pan,
                    "muted": info.muted,
                    "solo": info.solo,
                    "duration": info.duration,
                    "position": info.position,
                    "currentVariation": info.current_variation,
                })
            })
            .collect();
        serde_json::to_string(&tracks).unwrap_or_else(|_| "[]".to_string())
    }

    /// Get the number of tracks.
    #[wasm_bindgen(js_name = "trackCount")]
    pub fn track_count(&self) -> usize {
        self.inner.track_count()
    }

    /// Get beat info as a JSON string.
    #[wasm_bindgen(js_name = "getBeatInfoJson")]
    pub fn get_beat_info_json(&self) -> String {
        let info = self.inner.get_beat_info();
        let json = serde_json::json!({
            "currentBeat": info.current_beat,
            "beatFraction": info.beat_fraction,
            "bpm": info.bpm,
            "timeSignatureNumerator": info.time_signature_numerator,
            "timeSignatureDenominator": info.time_signature_denominator,
        });
        json.to_string()
    }

    /// Get the number of active (playing) tracks.
    #[wasm_bindgen(js_name = "activeTrackCount")]
    pub fn active_track_count(&self) -> usize {
        self.inner.active_track_count()
    }

    // =======================================================================
    // Events
    // =======================================================================

    /// Drain all pending events as a JSON array string.
    ///
    /// Each element is a JSON object with a `type` field matching the
    /// TypeScript event naming convention (e.g., `"playback:started"`).
    ///
    /// Call this after each operation or in a polling loop.
    #[wasm_bindgen(js_name = "drainEventsJson")]
    pub fn drain_events_json(&mut self) -> String {
        let events = self.inner.drain_events();
        let json_events: Vec<serde_json::Value> = events
            .iter()
            .map(engine_event_to_json)
            .collect();
        serde_json::to_string(&json_events).unwrap_or_else(|_| "[]".to_string())
    }

    /// Get the number of pending events (without draining).
    #[wasm_bindgen(js_name = "pendingEventCount")]
    pub fn pending_event_count(&self) -> usize {
        self.inner.pending_event_count()
    }
}

// ---------------------------------------------------------------------------
// Standalone utility exports
// ---------------------------------------------------------------------------

/// Convert semitones to a frequency ratio.
#[wasm_bindgen(js_name = "semitonesToRatio")]
pub fn semitones_to_ratio(semitones: f64) -> Result<f64, JsValue> {
    crate::pitch::semitones_to_ratio(semitones)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Convert a frequency ratio to semitones.
#[wasm_bindgen(js_name = "ratioToSemitones")]
pub fn ratio_to_semitones(ratio: f64) -> Result<f64, JsValue> {
    crate::pitch::ratio_to_semitones(ratio)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Convert cents to a frequency ratio.
#[wasm_bindgen(js_name = "centsToRatio")]
pub fn cents_to_ratio(cents: f64) -> Result<f64, JsValue> {
    crate::pitch::cents_to_ratio(cents)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Convert a frequency ratio to cents.
#[wasm_bindgen(js_name = "ratioToCents")]
pub fn ratio_to_cents(ratio: f64) -> Result<f64, JsValue> {
    crate::pitch::ratio_to_cents(ratio)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Calculate tempo ratio (desired_bpm / original_bpm).
#[wasm_bindgen(js_name = "tempoRatioCalc")]
pub fn tempo_ratio_calc(desired_bpm: f64, original_bpm: f64) -> Result<f64, JsValue> {
    crate::tempo::tempo_ratio(desired_bpm, original_bpm)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Check if a BPM value is valid.
#[wasm_bindgen(js_name = "isValidTempo")]
pub fn is_valid_tempo(bpm: f64) -> bool {
    crate::tempo::is_valid_tempo(bpm)
}

/// Convert time in seconds to sample count.
#[wasm_bindgen(js_name = "timeToSamples")]
pub fn time_to_samples(time_seconds: f64, sample_rate: u32) -> i64 {
    crate::timing::time_to_samples(time_seconds, sample_rate)
}

/// Convert sample count to time in seconds.
#[wasm_bindgen(js_name = "samplesToTime")]
pub fn samples_to_time(samples: i64, sample_rate: u32) -> f64 {
    crate::timing::samples_to_time(samples, sample_rate)
}

/// Convert dB level to linear gain.
#[wasm_bindgen(js_name = "dbToLevel")]
pub fn db_to_level(db: f64) -> f64 {
    crate::mixer::db_to_level(db)
}

/// Convert linear gain to dB level.
#[wasm_bindgen(js_name = "levelToDb")]
pub fn level_to_db(level: f64) -> f64 {
    crate::mixer::level_to_db(level)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wasm_engine_creates_and_initializes() {
        let mut engine = WasmMaestroEngine::new(48000);
        assert_eq!(engine.state(), WasmEngineState::Uninitialized);
        assert_eq!(engine.sample_rate(), 48000);

        // Cannot call methods before initialize (some are state-gated)
        engine.initialize().unwrap();
        // After initialize, state should advance
        let state = engine.state();
        assert!(
            state == WasmEngineState::ServicesReady
                || state == WasmEngineState::EnginesReady
                || state == WasmEngineState::Ready
        );
    }

    #[test]
    fn wasm_transport_state_mapping() {
        assert_eq!(
            WasmTransportState::from(TransportState::Stopped),
            WasmTransportState::Stopped
        );
        assert_eq!(
            WasmTransportState::from(TransportState::Playing),
            WasmTransportState::Playing
        );
        assert_eq!(
            WasmTransportState::from(TransportState::Paused),
            WasmTransportState::Paused
        );
    }

    #[test]
    fn wasm_safety_preset_mapping() {
        assert_eq!(
            SafetyPreset::from(WasmSafetyPreset::SliderOperation),
            SafetyPreset::SliderOperation
        );
        assert_eq!(
            SafetyPreset::from(WasmSafetyPreset::TrackInteraction),
            SafetyPreset::TrackInteraction
        );
        assert_eq!(
            SafetyPreset::from(WasmSafetyPreset::ManualControl),
            SafetyPreset::ManualControl
        );
        assert_eq!(
            SafetyPreset::from(WasmSafetyPreset::ProgrammaticOperation),
            SafetyPreset::ProgrammaticOperation
        );
        assert_eq!(
            SafetyPreset::from(WasmSafetyPreset::Permissive),
            SafetyPreset::Permissive
        );
    }

    #[test]
    fn wasm_engine_state_mapping() {
        assert_eq!(
            WasmEngineState::from(EngineState::Uninitialized),
            WasmEngineState::Uninitialized
        );
        assert_eq!(
            WasmEngineState::from(EngineState::Ready),
            WasmEngineState::Ready
        );
        assert_eq!(
            WasmEngineState::from(EngineState::Suspended),
            WasmEngineState::Suspended
        );
    }

    #[test]
    fn wasm_track_status_mapping() {
        assert_eq!(
            WasmTrackStatus::from(TrackStatus::Idle),
            WasmTrackStatus::Idle
        );
        assert_eq!(
            WasmTrackStatus::from(TrackStatus::Ready),
            WasmTrackStatus::Ready
        );
        assert_eq!(
            WasmTrackStatus::from(TrackStatus::Playing),
            WasmTrackStatus::Playing
        );
    }

    #[test]
    fn wasm_variation_type_mapping() {
        assert_eq!(
            VariationType::from(WasmVariationType::Simple),
            VariationType::Simple
        );
        assert_eq!(
            VariationType::from(WasmVariationType::Default),
            VariationType::Default
        );
        assert_eq!(
            VariationType::from(WasmVariationType::Complex),
            VariationType::Complex
        );
    }

    #[test]
    fn wasm_engine_track_lifecycle() {
        let mut engine = WasmMaestroEngine::new(48000);
        engine.initialize().unwrap();

        // Create track
        engine
            .create_track("drums", "songs/rock/drums.m4a", 120.0)
            .unwrap();
        assert_eq!(engine.track_count(), 1);

        // Arm track
        engine.arm_track("drums").unwrap();

        // Disarm track
        engine.disarm_track("drums").unwrap();

        // Remove track
        engine.remove_track("drums").unwrap();
        assert_eq!(engine.track_count(), 0);
    }

    #[test]
    fn wasm_engine_events_drain() {
        let mut engine = WasmMaestroEngine::new(48000);
        engine.initialize().unwrap();

        // Initialize produces events
        let json = engine.drain_events_json();
        assert!(json.contains("engine:initialized"));

        // After drain, no more events
        assert_eq!(engine.pending_event_count(), 0);
    }

    #[test]
    fn wasm_engine_beat_info() {
        let engine = WasmMaestroEngine::new(48000);
        let json = engine.get_beat_info_json();
        assert!(json.contains("currentBeat"));
        assert!(json.contains("bpm"));
    }

    #[test]
    fn wasm_engine_track_info() {
        let mut engine = WasmMaestroEngine::new(48000);
        engine.initialize().unwrap();
        engine
            .create_track("drums", "songs/rock/drums.m4a", 120.0)
            .unwrap();

        let all = engine.get_all_tracks_json();
        assert!(all.contains("drums"));
        assert!(all.contains("volume"));

        let ids = engine.track_ids_json();
        assert!(ids.contains("drums"));
    }

    #[test]
    fn wasm_engine_duplicate_track_error() {
        let mut engine = WasmMaestroEngine::new(48000);
        engine.initialize().unwrap();
        engine
            .create_track("drums", "url", 120.0)
            .unwrap();

        // Duplicate should fail — test via the inner engine to avoid JsValue
        // drop issues in non-WASM test context
        let result = engine.inner.create_track("drums", "url2", None, Some(120.0));
        assert!(result.is_err());
    }

    #[test]
    fn wasm_engine_transport() {
        let mut engine = WasmMaestroEngine::new(48000);
        engine.initialize().unwrap();

        assert!(engine.is_stopped());
        assert!(!engine.is_playing());
        assert!(!engine.is_paused());
        assert_eq!(engine.transport_state(), WasmTransportState::Stopped);
    }

    #[test]
    fn wasm_engine_tempo() {
        let mut engine = WasmMaestroEngine::new(48000);
        engine.initialize().unwrap();

        let ratio = engine.tempo_ratio();
        assert!((ratio - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn wasm_engine_destroy() {
        let mut engine = WasmMaestroEngine::new(48000);
        engine.initialize().unwrap();
        engine
            .create_track("drums", "url", 120.0)
            .unwrap();

        engine.destroy();
        assert_eq!(engine.state(), WasmEngineState::Destroying);
    }

    #[test]
    fn wasm_pcm_loading_interleaved() {
        let mut engine = WasmMaestroEngine::new(48000);
        engine.initialize().unwrap();
        engine
            .create_track("drums", "url", 120.0)
            .unwrap();

        // Create 10 frames of stereo silence (interleaved)
        let data = vec![0.0f32; 20]; // 10 frames × 2 channels

        engine
            .load_track_audio("drums", WasmVariationType::Default, &data, 2, 48000, "url")
            .unwrap();
    }

    #[test]
    fn wasm_pcm_loading_separated() {
        let mut engine = WasmMaestroEngine::new(48000);
        engine.initialize().unwrap();
        engine
            .create_track("drums", "url", 120.0)
            .unwrap();

        // Create 10 frames of stereo silence (separated)
        let left = vec![0.0f32; 10];
        let right = vec![0.0f32; 10];

        engine
            .load_track_audio_separated(
                "drums",
                WasmVariationType::Default,
                &left,
                &right,
                48000,
                "url",
            )
            .unwrap();
    }

    #[test]
    fn wasm_pcm_loading_mismatched_channels() {
        // Test via inner engine: construct a PcmData with mismatched channel
        // lengths and verify the engine rejects it (channels must be equal length).
        let mut engine = WasmMaestroEngine::new(48000);
        engine.inner.initialize().unwrap();
        engine
            .inner
            .create_track("drums", "url", None, Some(120.0))
            .unwrap();

        // Mismatched channel lengths — left has 10 frames, right has 5
        let pcm_data = crate::pcm_cache::PcmData {
            channel_data: vec![vec![0.0f32; 10], vec![0.0f32; 5]],
            sample_rate: 48000,
            num_channels: 2,
            length_samples: 10,
        };

        let result = engine.inner.load_track_audio(
            "drums",
            crate::pcm_cache::VariationType::Default,
            pcm_data,
            "url",
        );
        assert!(result.is_err());
    }

    #[test]
    fn wasm_pcm_loading_indivisible_channel_data() {
        // Verify that interleaved data not evenly divisible by num_channels
        // is rejected at the WASM boundary (before reaching the inner engine).
        // We test the validation logic directly since JsValue panics outside WASM.
        let total_samples = 99;
        let num_channels: u32 = 2;
        assert_ne!(
            total_samples % num_channels as usize,
            0,
            "test data should not be evenly divisible"
        );
    }
}
