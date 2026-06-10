//! # maestro-api
//!
//! The single API schema for the Maestro audio engine (spec §3.6).
//!
//! Every transport — UniFFI (native) and wasm-bindgen (web) — carries exactly
//! these types as JSON. The TypeScript side consumes the ts-rs–generated
//! bindings committed under `bindings/`; CI fails if regeneration differs,
//! so binding drift is a build error, never a runtime surprise.
//!
//! The transport surface is total and tiny (spec §3.6):
//!
//! ```text
//! init(config) ; dispatch(command_json) -> result_json    // control plane
//! snapshot_json() -> String ; drain_events_json() -> String
//! load_pcm(track, variation, bytes, meta) -> result       // the one bulk-data call
//! maestro_render(core, buffers, frames)                   // C ABI, audio thread only
//! ```
//!
//! **M0 scope:** the walking-skeleton subset — transport play/stop, init-time
//! tempo, track install/activate/deactivate, snapshot, edge events. Breadth
//! (volume/pan/pitch/variations/safety presets) lands in M2 as additional
//! variants; adding an operation = one enum variant + one match arm.

use serde::{Deserialize, Serialize};

/// Engine configuration passed to `init`. The engine sample rate is fixed at
/// 48 kHz by design (spec §4.7); `sample_rate` exists so the shim's assertion
/// is explicit, not implicit.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[cfg_attr(feature = "ts", derive(ts_rs::TS), ts(export))]
pub struct EngineConfig {
    /// Must be 48_000. `init` fails on anything else — the engine never resamples.
    pub sample_rate: u32,
    /// Initial engine tempo in milli-BPM (e.g. 120 BPM = 120_000).
    pub initial_tempo_mbpm: u32,
    /// Beats per bar from the song's time-signature metadata. The engine never
    /// hardcodes 4/4 (spec §4.12); bar display derives from this.
    pub beats_per_bar: u32,
}

/// Commands accepted by `dispatch`. Applied by the audio thread at the next
/// block boundary; nothing timing-critical crosses the bridge (spec §3.5).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase", rename_all_fields = "camelCase")]
#[cfg_attr(feature = "ts", derive(ts_rs::TS), ts(export))]
pub enum Command {
    /// Start the transport. The master frame clock `F` advances only while playing.
    Play,
    /// Stop the transport and reset the timeline epoch (bar 1 beat 1).
    /// Individually deactivating the last audible track does NOT do this (spec §3.1).
    Stop,
    /// Set engine tempo in milli-BPM. M0: accepted only while stopped
    /// (live tempo changes are M1). Clamped range 40_000..=300_000 (spec §4.8);
    /// out-of-range is rejected with an error result, never clamped silently.
    SetTempo { mbpm: u32 },
    /// Make an installed track audible. M0 quantize policy: immediate
    /// (sample-exact within the current block, under an activation ramp).
    Activate { slot: u8 },
    /// Silence a track. Its musical position remains derived from the timeline;
    /// reactivation cannot desync by construction (spec §3.2).
    Deactivate { slot: u8 },
    /// Remove a track and free its slot. PCM is reclaimed off the audio thread.
    RemoveTrack { slot: u8 },
}

/// Result of `dispatch`, returned as JSON.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase", rename_all_fields = "camelCase")]
#[cfg_attr(feature = "ts", derive(ts_rs::TS), ts(export))]
pub enum CommandResult {
    Ok,
    Rejected { reason: String },
}

/// Per-file PCM metadata handed to `load_pcm` (spec §3.4). Lengths are
/// reconciled in Rust, decoder-agnostically; mismatch is a hard load failure.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[cfg_attr(feature = "ts", derive(ts_rs::TS), ts(export))]
pub struct LoadMeta {
    /// Stable slot index (0..MAX_TRACKS) chosen by the backend.
    pub slot: u8,
    /// Track's original tempo in milli-BPM (integer; never 0 — load fails).
    pub original_tempo_mbpm: u32,
    /// Exact musical loop length in beats (integer; authoritative for wrapping).
    pub loop_beats: u32,
    /// Encoded file's measured priming/padding (per file, not per stem).
    pub encoder_delay: u32,
    pub end_padding: u32,
    /// Real audio length in samples as authored (pre-encode).
    pub original_samples: u64,
    /// Initial gain/pan so M0 can hard-pan the metronome stems without a mixer surface.
    pub gain: f32,
    pub pan: f32,
}

/// Edge-triggered events drained via `drain_events_json`. JS state is cosmetic;
/// a missed event can never desync anything (spec §3.5).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase", rename_all_fields = "camelCase")]
#[cfg_attr(feature = "ts", derive(ts_rs::TS), ts(export))]
pub enum Event {
    TrackLoaded { slot: u8, reconciliation: String },
    TrackLoadFailed { slot: u8, reason: String },
    Started,
    Stopped,
    LoopWrapped { slot: u8, wrap_count: u32 },
    Underrun,
    BudgetExceeded { render_ns: u64, period_ns: u64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[cfg_attr(feature = "ts", derive(ts_rs::TS), ts(export))]
pub enum TransportState {
    Stopped,
    Playing,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[cfg_attr(feature = "ts", derive(ts_rs::TS), ts(export))]
pub enum TrackStatus {
    Empty,
    Loaded,
    Active,
}

/// Per-track slice of the snapshot. Track identity across the bridge is the
/// slot index; the backend owns the trackId ↔ slot map (spec §3.5).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[cfg_attr(feature = "ts", derive(ts_rs::TS), ts(export))]
pub struct TrackSnap {
    pub slot: u8,
    pub status: TrackStatus,
    pub loop_count: u32,
    /// Derived from bu(F) mod Λ. Display only — floats never feed timing.
    pub loop_progress: f32,
    pub rms_l: f32,
    pub rms_r: f32,
    pub peak_l: f32,
    pub peak_r: f32,
}

/// The only state egress (spec §3.5): written by the audio thread once per
/// block behind a seqlock, serialized on demand by the control plane.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[cfg_attr(feature = "ts", derive(ts_rs::TS), ts(export))]
pub struct Snapshot {
    pub engine_frame: u64,
    pub tempo_mbpm: u32,
    pub transport: TransportState,
    /// Derived from bu(F); display only.
    pub bar: u32,
    pub beat_in_bar: f32,
    pub timeline_epoch: u32,
    /// Render-budget telemetry (spec §3.3): the audio thread times itself.
    /// The C0 gate "p99 < 50% of period" is exactly
    /// `blocks_over_half_period / blocks_total ≤ 0.01`; "max < 90%" uses
    /// `render_ns_max` against `callback_period_ns`. `underruns` counts blocks
    /// whose render exceeded the full callback period.
    pub underruns: u32,
    pub render_ns_max: u64,
    pub callback_period_ns: u64,
    pub blocks_total: u64,
    pub blocks_over_half_period: u64,
    pub tracks: Vec<TrackSnap>,
}
