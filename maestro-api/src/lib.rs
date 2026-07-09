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
//! init(config) ; dispatch(command_json) -> result_json    // control plane, ≤ 60 Hz
//! snapshot_json() -> String ; drain_events_json() -> String
//! load_pcm(track, variation, bytes, meta) -> result       // the one bulk-data call
//! maestro_render(core, buffers, frames)                   // C ABI, audio thread only
//! ```
//!
//! **M2 scope:** the full control plane (spec §5 M2) — volume/pan/mute/solo,
//! full track lifecycle, transport safety policy, async variation switching
//! under the §4.11 memory policy, per-track level metering, and the §3.4
//! decode/trim surface (per-file measured metadata; `sample_rate` asserted).

use serde::{Deserialize, Serialize};

/// FR2 schema fingerprint machinery (ADR-100) — shared by build.rs (which
/// embeds [`SCHEMA_FINGERPRINT`]) and gen-ts (which writes the app bundle's
/// `bindings/fingerprint.ts` half of the handshake).
pub mod fingerprint;

/// The maestro-api schema fingerprint this build was compiled against
/// (ADR-100): a hash over the committed `bindings/` content, computed by
/// build.rs. Both transports expose it (wasm: `schema_fingerprint()` on the
/// worklet `ready` message; native: the plugin's `getSchemaFingerprint`) and
/// the backends hard-fail init on mismatch or absence — a stale committed
/// engine artifact must never silently reinterpret the schema (PRD §2, the
/// ADR-098 park-as-swap bug).
pub const SCHEMA_FINGERPRINT: &str = env!("MAESTRO_SCHEMA_FINGERPRINT");

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
    /// §4.11 per-song PCM budget override in bytes (ADR-098). Omitted → the
    /// engine default of 96 MiB (native jetsam headroom). Composition roots
    /// pass a device-RAM-tiered value so a whole song's variations can stay
    /// resident for instant switching. Zero is rejected — never clamped.
    #[serde(default)]
    pub pcm_budget_bytes: Option<u64>,
}

/// Transport side-effect policy for track activation/deactivation — the
/// engine-side encoding of the app's transport safety presets (ADR-092):
/// `Manual` never touches the transport (slider/programmatic operations);
/// `Auto` lets `Activate` start a stopped transport and lets `Deactivate`
/// stop the transport (epoch reset) when the last armed track disarms.
/// The §3.1 default stands: without `Auto`, individually stopping the last
/// stem does NOT stop the transport or reset the epoch.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[cfg_attr(feature = "ts", derive(ts_rs::TS), ts(export))]
pub enum TransportPolicy {
    #[default]
    Manual,
    Auto,
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
    /// Individually deactivating the last audible track does NOT do this
    /// (spec §3.1) unless that `Deactivate` carries `TransportPolicy::Auto`.
    Stop,
    /// Freeze the master clock `F` — for all tracks at once, with **no epoch
    /// reset** (distinct from `Stop`, spec §3.1). Positions are derived from
    /// `F`, so pause cannot desync anything; `Resume` simply lets `F` continue.
    /// Interruptions map to this command (spec §4.9).
    Pause,
    /// Continue from `Pause`. Nothing is re-anchored — there is nothing to
    /// re-anchor (positions are computed, never advanced).
    Resume,
    /// Set engine tempo in milli-BPM, **live or stopped** (M1). Applied at the
    /// next block boundary; `bu` is continuous across the change by
    /// construction (spec §3.1). The engine echoes the applied value via
    /// `Event::TempoApplied` with the exact application frame.
    /// Clamped range 40_000..=300_000 (spec §4.8); out-of-range is rejected
    /// with an error result, never clamped silently.
    SetTempo { mbpm: u32 },
    /// Make an installed track audible. `quantize` chooses the activation
    /// frame policy; the engine computes the frame — JS never does (spec §3.5).
    /// `transport: Auto` additionally starts a stopped transport (safety
    /// presets, ADR-092).
    Activate {
        slot: u8,
        #[serde(default)]
        quantize: QuantizePolicy,
        #[serde(default)]
        transport: TransportPolicy,
    },
    /// Silence a track. Its musical position remains derived from the timeline;
    /// reactivation cannot desync by construction (spec §3.2). With
    /// `transport: Auto`, deactivating the LAST active track also stops the
    /// transport (epoch reset) — the engine decides, echoing `Stopped`.
    Deactivate {
        slot: u8,
        #[serde(default)]
        transport: TransportPolicy,
    },
    /// Remove a track and free its slot (and its cached variations). PCM is
    /// reclaimed off the audio thread.
    RemoveTrack { slot: u8 },
    /// Per-track gain in [0.0, 2.0] (rejected outside — never clamped). A
    /// track at effective gain 0 skips its read/stretch DSP entirely and
    /// re-anchors under the activation ramp on becoming audible again
    /// (spec §3.2 — silence costs nothing and cannot desync).
    SetGain { slot: u8, gain: f32 },
    /// Per-track equal-power pan in [-1.0, 1.0] (rejected outside).
    SetPan { slot: u8, pan: f32 },
    /// Mute/unmute. Muted tracks skip DSP and re-anchor on unmute (spec §3.2).
    SetMute { slot: u8, muted: bool },
    /// Solo on/off. While any track is soloed, non-soloed tracks are silent
    /// (and skip DSP); clearing the last solo restores them under the
    /// activation ramp.
    SetSolo { slot: u8, solo: bool },
    /// Per-track pitch shift in semitones (spec §3.2 / §4.8: ±12, rejected —
    /// never clamped — outside that range). A pitch shift on a unity track
    /// promotes it to the stretch path under the one-block crossfade; position
    /// math is unaffected.
    SetPitch { slot: u8, semitones: f32 },
    /// Switch a track to another variation (spec §3.2 / §4.11). If the
    /// variation's PCM is cached, the swap applies at the next block boundary
    /// (Arc swap + re-anchor + one-block crossfade). Otherwise the command is
    /// acknowledged, `Event::VariationLoadRequired` asks the platform to
    /// decode and call `load_pcm`, and the swap applies on load completion —
    /// `Event::VariationApplied` signals it either way.
    SwitchVariation { slot: u8, variation: u8 },
}

/// Activation-frame policy for [`Command::Activate`] (spec §3.1):
/// `Immediate` activates at the current block (sample-exact under the
/// activation ramp); `NextBar` computes the activation frame `F_q` by exact
/// integer inversion of `bu` and splits the render block at `F_q` so
/// activation is sample-exact within a block.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[cfg_attr(feature = "ts", derive(ts_rs::TS), ts(export))]
pub enum QuantizePolicy {
    #[default]
    Immediate,
    NextBar,
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
/// Every field is MEASURED by the encoding pipeline from the actual encoded
/// file (per file, not per stem) — nothing here is asserted by hand.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[cfg_attr(feature = "ts", derive(ts_rs::TS), ts(export))]
pub struct LoadMeta {
    /// Stable slot index (0..MAX_TRACKS) chosen by the backend.
    pub slot: u8,
    /// Decoded sample rate as the shim observed it. Must be 48_000 — the
    /// engine never resamples (spec §3.4#3/§4.7); anything else is a hard
    /// `TrackLoadFailed`, re-validated here in Rust (defense in depth, §4.12).
    pub sample_rate: u32,
    /// Track's original tempo in milli-BPM (integer; never 0 — load fails).
    pub original_tempo_mbpm: u32,
    /// Exact musical loop length in beats (integer; authoritative for wrapping).
    pub loop_beats: u32,
    /// Encoded file's measured priming/padding (per file, not per stem).
    pub encoder_delay: u32,
    pub end_padding: u32,
    /// Real audio length in samples as authored (pre-encode).
    pub original_samples: u64,
    /// Variation index of this file. Loading a different variation into an
    /// occupied slot performs an engine-side variation switch (Arc PCM swap +
    /// re-anchor + one-block crossfade at a block boundary, spec §3.2); all
    /// variations of a stem must declare identical musical identity
    /// (`loop_beats` / `original_samples` / `original_tempo_mbpm`) — and
    /// reconcile the same way — or the load fails loudly (spec §3.4).
    #[serde(default)]
    pub variation: u8,
    /// §4.11 prewarm (ADR-098): park this file's PCM in the slot's variation
    /// cache WITHOUT switching to it. Requires an occupied slot (rejected on
    /// an empty one); identity/reconciliation validation is unchanged. If the
    /// engine has a pending async switch to exactly this variation, the load
    /// completes that switch instead. A park never touches the RT thread.
    #[serde(default)]
    pub cache_only: bool,
    /// Initial gain/pan (same validation as SetGain/SetPan).
    pub gain: f32,
    pub pan: f32,
}

/// Edge-triggered events drained via `drain_events_json`. JS state is cosmetic;
/// a missed event can never desync anything (spec §3.5).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase", rename_all_fields = "camelCase")]
#[cfg_attr(feature = "ts", derive(ts_rs::TS), ts(export))]
pub enum Event {
    /// `reconciliation` names which §3.4 case fired ("alreadyTrimmed" /
    /// "trimmedHere"); asserted stable across variations of a stem.
    TrackLoaded { slot: u8, variation: u8, reconciliation: String },
    TrackLoadFailed { slot: u8, reason: String },
    TrackRemoved { slot: u8 },
    Started,
    Stopped,
    /// `F` froze at `at_frame` (no epoch reset — distinct from `Stopped`).
    Paused { at_frame: u64 },
    Resumed { at_frame: u64 },
    /// Tempo echo (spec §3.1/§3.5): the engine applied `mbpm` at exactly
    /// `at_frame` (the block boundary where the segment was appended). The
    /// device analyzer derives the expected onset grid from this, never from
    /// wall-clock assumptions (spec §6.4).
    TempoApplied { mbpm: u32, at_frame: u64 },
    /// A track became audible at exactly `at_frame` (immediate activation, or
    /// the quantized activation frame `F_q` for `QuantizePolicy::NextBar`).
    TrackActivated { slot: u8, at_frame: u64 },
    TrackDeactivated { slot: u8, at_frame: u64 },
    /// Mute state applied at exactly `at_frame` — the analyzer's predicted
    /// silent-window boundary for mute legs (spec §6.4 methodology).
    MuteApplied { slot: u8, muted: bool, at_frame: u64 },
    SoloApplied { slot: u8, solo: bool, at_frame: u64 },
    /// Variation switch applied at a block boundary (spec §3.2). For an async
    /// switch (§4.11) this fires on load completion.
    VariationApplied { slot: u8, variation: u8, at_frame: u64 },
    /// Async variation switch (§4.11): the requested variation is not cached;
    /// the platform must decode it and call `load_pcm` with this slot and
    /// variation. The engine keeps playing the current variation meanwhile.
    VariationLoadRequired { slot: u8, variation: u8 },
    /// A cached (non-audible) variation was evicted under the §4.11 per-song
    /// PCM budget.
    VariationEvicted { slot: u8, variation: u8 },
    LoopWrapped { slot: u8, wrap_count: u32 },
    Underrun,
    BudgetExceeded { render_ns: u64, period_ns: u64 },
    /// The RT event queue overflowed and `dropped` edge events were lost
    /// since the last drain. Never silent (spec §6.7): the audio thread
    /// counts drops in the snapshot and the control plane surfaces them here.
    EventQueueOverflow { dropped: u32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[cfg_attr(feature = "ts", derive(ts_rs::TS), ts(export))]
pub enum TransportState {
    Stopped,
    Playing,
    /// `F` frozen, timeline epoch intact (spec §3.1).
    Paused,
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
    /// Currently rendering variation (spec §3.5).
    pub variation: u8,
    pub gain: f32,
    pub pan: f32,
    pub muted: bool,
    pub solo: bool,
    pub loop_count: u32,
    /// Derived from bu(F) mod Λ. Display only — floats never feed timing.
    pub loop_progress: f32,
    /// Per-track level metering over the last rendered block (post gain/pan,
    /// the track's contribution to the mix). Zero while inaudible.
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
    /// Edge events dropped on RT event-queue overflow (spec §6.7 — surfaced,
    /// never silent). Monotonic since engine creation.
    pub events_dropped: u32,
    /// PCM bytes resident for this engine instance (installed + cached
    /// variations) — the §4.11 per-song budget gate reads this.
    pub pcm_bytes_in_use: u64,
    pub tracks: Vec<TrackSnap>,
}

#[cfg(test)]
mod fingerprint_tests {
    use std::path::Path;

    use super::fingerprint::{
        fingerprint_bindings, fingerprint_files, SCHEMA_FINGERPRINT_PREFIX,
    };
    use super::SCHEMA_FINGERPRINT;

    /// Pins the build.rs `#[path]` share: the embedded constant must equal a
    /// fresh recomputation over the same committed bindings.
    #[test]
    fn embedded_fingerprint_matches_recomputation() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("bindings");
        let recomputed = fingerprint_bindings(&dir).expect("bindings dir fingerprints");
        assert_eq!(SCHEMA_FINGERPRINT, recomputed);
    }

    /// The FR3 gate extracts the fingerprint from artifacts by grepping for
    /// this exact shape — pin it.
    #[test]
    fn fingerprint_shape_is_greppable() {
        let hex = SCHEMA_FINGERPRINT
            .strip_prefix(SCHEMA_FINGERPRINT_PREFIX)
            .expect("magic prefix present");
        assert_eq!(hex.len(), 16);
        assert!(hex.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
    }

    /// Negative control: the fingerprint must MOVE under any content or
    /// filename change — a hash blind to either would green-light skew.
    #[test]
    fn fingerprint_moves_under_content_and_name_changes() {
        let base = vec![
            ("A.ts".to_string(), b"export type A = 1;".to_vec()),
            ("B.ts".to_string(), b"export type B = 2;".to_vec()),
        ];
        let original = fingerprint_files(&base);

        let mut content_changed = base.clone();
        content_changed[1].1 = b"export type B = 3;".to_vec();
        assert_ne!(original, fingerprint_files(&content_changed));

        let mut renamed = base;
        renamed[1].0 = "C.ts".to_string();
        assert_ne!(original, fingerprint_files(&renamed));
    }
}
