//! # control_plane — the non-RT half of the engine (spec §3.3 / §3.4 / §3.5)
//!
//! Runs on the caller's thread behind the FFI. Validates commands, performs
//! decode-length reconciliation and loop-identity validation, allocates and
//! preconfigures every resource the audio thread will touch (PCM buffers,
//! stretchers, scratch), then transfers them across the SPSC command queue.
//! It also drains the event and reclaim queues — the audio thread never frees
//! memory — and owns the §4.11 variation PCM cache (per-song budget, eviction
//! of the least-recently-audible variation, async switch bookkeeping).
//!
//! Since ADR-112 it additionally owns the **streaming load lifecycle**: the
//! per-slot `Streaming` state, the §5.5 begin-time validation order (every
//! identity-shaped check runs BEFORE any byte moves), trim-on-the-fly across
//! appends, and the finalize honesty check against `declared_total_samples`.
//! It is the single writer of the `EnginePcm` load frontier; the audio thread
//! is the single reader. Bulk `load_pcm` is a composition over that same core
//! — see [`ControlPlane::load_pcm`].
//!
//! State egress is exactly one call: [`ControlPlane::snapshot`] reads the
//! seqlock'd POD the audio thread publishes each block and converts it to the
//! `maestro-api` schema. JS holds render-only mirrors (spec §3.5).

use std::sync::Arc;

use maestro_api::{
    AppendResult, Command, CommandResult, EngineConfig, Event, LoadMeta, Snapshot,
    StreamLoadMeta, TrackSnap, TrackStatus, TransportPolicy, TransportState,
};

use crate::audio_core::{AudioCore, RawSnapshot, Reclaimable, RtCommand, RtEvent, MAX_TRACKS};
use crate::engine_pcm::EnginePcm;
use crate::seqlock::SeqLock;
use crate::stretch::Stretcher;
use crate::timeline::{Timeline, TrackTimeline, ENGINE_SAMPLE_RATE, MAX_TEMPO_MBPM, MIN_TEMPO_MBPM};

/// Queue capacities (spec §3.3): fixed, preallocated. The control plane blocks
/// (its own thread, never audio) when the command queue is full.
const COMMAND_QUEUE_CAPACITY: usize = 256;
const EVENT_QUEUE_CAPACITY: usize = 256;
const RECLAIM_QUEUE_CAPACITY: usize = 256;

/// §4.11 per-song PCM budget: installed + cached variation PCM for one engine
/// instance must stay under this; above it, the least-recently-audible cached
/// (never an installed/audible) variation is evicted.
pub const PCM_BUDGET_BYTES: u64 = 96 * 1024 * 1024;

/// ADR-112 §5.2: per-append size cap (10 s at 48 kHz). Bounds the
/// worklet-thread copy per `append_pcm_chunk` message; enforced engine-side
/// on the streaming transport (the internal bulk composition is exempt —
/// bulk `load_pcm` is one big copy by design and unchanged on the wire).
pub const MAX_STREAM_APPEND_FRAMES: usize = 480_000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineError {
    UnsupportedSampleRate { got: u32 },
    InvalidConfig { reason: String },
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSampleRate { got } => {
                write!(f, "engine sample rate is fixed at {ENGINE_SAMPLE_RATE} Hz, got {got}")
            }
            Self::InvalidConfig { reason } => write!(f, "invalid engine config: {reason}"),
        }
    }
}

impl std::error::Error for EngineError {}

/// Which §3.4 reconciliation case fired for a loaded file. Reported in the
/// load event and asserted stable across variations of a stem (spec §3.4#4):
/// same decoder + same encoder must reconcile the same way — a flip means a
/// mis-tagged or mis-decoded file, which is a hard failure, not a curiosity.
///
/// All three cases are EXACT integer equalities against the per-file
/// MEASURED metadata — there are no heuristics, and anything else remains a
/// hard `TrackLoadFailed` (spec §3.4#4; third case added at M4, ADR-095).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reconciliation {
    /// Decoded length == original_samples: decoder already applied gapless trim.
    AlreadyTrimmed,
    /// Decoded length == original + delay + padding: raw decode; we trim exactly.
    TrimmedHere,
    /// Decoded length == original + end_padding: the decoder applied the
    /// priming (encoder delay) but NOT the trailing padding — Chrome's
    /// `decodeAudioData` behavior, found by this exact gate firing on the
    /// first M4 in-app WASM run (ADR-095). We truncate exactly the padding.
    TrimmedPadding,
}

impl Reconciliation {
    fn as_str(self) -> &'static str {
        match self {
            Self::AlreadyTrimmed => "alreadyTrimmed",
            Self::TrimmedHere => "trimmedHere",
            Self::TrimmedPadding => "trimmedPadding",
        }
    }
}

/// §4.8 rate clamp: every loaded track's implicit rate `E/O` must stay inside
/// [0.25, 4.0]. Checked at load (against the current tempo) and at SetTempo
/// (against every loaded track) — out-of-range is rejected, never clamped.
const MAX_RATE_NUM: u64 = 4; // rate ≤ 4.0  ⇔  E ≤ 4·O
const MAX_RATE_DEN: u64 = 4; // rate ≥ 0.25 ⇔  4·E ≥ O

/// §4.8 pitch clamp: ±12 semitones; out-of-range commands are rejected.
const MAX_PITCH_SEMITONES: f32 = 12.0;

/// Gain accepted range (ADR-092): [0, 2] (unity + headroom); rejected outside.
const MAX_GAIN: f32 = 2.0;

/// A §4.11-cached (parked) variation. Carries its own timing map because
/// variations of a stem may differ in loop length (ADR-109) — the map must
/// travel with the PCM into any future swap, atomically.
struct CachedVariation {
    variation: u8,
    pcm: Arc<EnginePcm>,
    map: TrackTimeline,
    /// Last-audible stamp — the §4.11 eviction order. Stamped when the
    /// variation stops being current (or at load if never current).
    last_audible: u64,
}

/// Control-side mirror of an installed track: the SHARED musical identity
/// (`o_mbpm` + reconciliation case — hard-gated equal across variations),
/// the installed variation's own timing map (per-variation since ADR-109),
/// the installed PCM, and the §4.11 variation cache for this slot.
struct LoadedSlot {
    /// Original tempo, shared across all variations of the stem (hard gate):
    /// `u(bu) = bu/O` must be variation-independent so a switch resumes at
    /// the same global musical position.
    o_mbpm: u32,
    variation: u8,
    /// §3.4#4: which reconciliation case fired for the first loaded file of
    /// this stem; later variations must match.
    reconciliation: Reconciliation,
    /// The installed variation's timing map (loop_beats/Λ/pcm_len — may
    /// differ per variation, ADR-109). Rotates with `current_pcm`.
    map: TrackTimeline,
    /// The PCM currently installed in the audio core (Arc-shared with it).
    current_pcm: Arc<EnginePcm>,
    /// Cached non-current variations, each with its own map.
    cached: Vec<CachedVariation>,
    /// §4.11 async switch: a requested variation awaiting platform load.
    pending_variation: Option<u8>,
}

/// An in-progress streaming load (ADR-112 §5.5). The track is installed in
/// the audio core (visible in the snapshot, starvation-protected); this is
/// the control-side bookkeeping that turns appends into frontier advances.
struct StreamingSlot {
    /// The begin-time `LoadMeta` (identity + trim metadata).
    meta: LoadMeta,
    /// §3.4#4 case fixed at begin from `declared_total_samples`.
    reconciliation: Reconciliation,
    /// Raw stream samples the host declared it will deliver.
    declared_total: u64,
    /// Raw stream samples received so far (kept + discarded).
    received_total: u64,
    /// Next expected append sequence number.
    next_seq: u32,
    /// The buffer being filled (Arc-shared with the audio core).
    pcm: Arc<EnginePcm>,
    map: TrackTimeline,
}

enum SlotState {
    Empty,
    Loaded(Box<LoadedSlot>),
    Streaming(Box<StreamingSlot>),
}

/// The control-plane handle. One per engine instance; not `Sync` — the FFI
/// layer (UniFFI/WASM) serializes access on its own terms.
pub struct ControlPlane {
    commands: rtrb::Producer<RtCommand>,
    events: rtrb::Consumer<RtEvent>,
    reclaim: rtrb::Consumer<Reclaimable>,
    snapshot: Arc<SeqLock<RawSnapshot>>,
    config: EngineConfig,
    /// Control-side mirror of the last accepted tempo, for §4.8 load-time
    /// rate validation. The snapshot lags by a block; this mirror is always
    /// consistent with the command stream this plane has issued.
    tempo_mbpm: u32,
    /// Control-side transport mirror (ADR-113): true from an accepted Play
    /// until an accepted Stop. The snapshot's `playing` lags by a block, so
    /// validating `SetTimelineOrigin` against it alone would accept an
    /// origin dispatched right after a Play — which the RT arm then drops
    /// SILENTLY (the queue applies Play first). This flag tracks dispatch
    /// order exactly, turning that race into a loud rejection.
    play_commanded: bool,
    /// Control-side mirrors; the authoritative runtime state is the audio
    /// core's. Owns the §4.11 variation caches.
    slots: [SlotState; MAX_TRACKS],
    /// §4.11 budget actually enforced: the config override, or
    /// [`PCM_BUDGET_BYTES`] when the config omits one (ADR-098).
    pcm_budget_bytes: u64,
    /// Monotonic stamp source for the §4.11 least-recently-audible order.
    lru_clock: u64,
    /// RT event-queue drops already surfaced (spec §6.7).
    events_dropped_seen: u32,
    /// Events originated control-side (loads, eviction, overflow), merged
    /// into `drain_events`.
    pending_events: Vec<Event>,
}

/// §4.8 rate-clamp check, in integers: `0.25 ≤ E/O ≤ 4` ⇔ `4E ≥ O ∧ E ≤ 4O`.
fn rate_in_range(tempo_mbpm: u32, o_mbpm: u32, slot: usize) -> Result<(), String> {
    let (e, o) = (tempo_mbpm as u64, o_mbpm as u64);
    if e * MAX_RATE_DEN < o || e > o * MAX_RATE_NUM {
        return Err(format!(
            "tempo {tempo_mbpm} mBPM puts slot {slot} (original {o_mbpm} mBPM) outside the \
             rate clamp [0.25, 4.0] (spec §4.8)"
        ));
    }
    Ok(())
}

/// §3.4#4 decoder-agnostic length reconciliation — the double-trim killer.
/// Three EXACT cases (ADR-095 added the third); anything else fails. Where
/// the metadata makes cases coincide (delay or padding of 0) the trims
/// coincide too — order cannot change the outcome. Shared by bulk `load_pcm`
/// (classifying the actual decoded length) and `begin_pcm_stream`
/// (classifying the declared total, ADR-112 §5.5).
fn classify_reconciliation(meta: &LoadMeta, total: u64) -> Result<Reconciliation, String> {
    let raw_len = meta.original_samples + meta.encoder_delay as u64 + meta.end_padding as u64;
    let padded_len = meta.original_samples + meta.end_padding as u64;
    if total == meta.original_samples {
        Ok(Reconciliation::AlreadyTrimmed)
    } else if total == raw_len {
        Ok(Reconciliation::TrimmedHere)
    } else if total == padded_len {
        Ok(Reconciliation::TrimmedPadding)
    } else {
        Err(format!(
            "length mismatch: decoded {total}, expected {} (already-trimmed), {raw_len} \
             (raw decode), or {padded_len} (delay-only trim) — no heuristics, load fails \
             (spec §3.4)",
            meta.original_samples
        ))
    }
}

/// The reconciliation case's keep-window start in the raw stream: samples
/// `[D, D + original_samples)` are stored, everything else is discarded
/// (ADR-112 §5.5 trim-on-the-fly; identical math for bulk and streaming).
fn keep_window_start(reconciliation: Reconciliation, meta: &LoadMeta) -> u64 {
    match reconciliation {
        Reconciliation::TrimmedHere => meta.encoder_delay as u64,
        Reconciliation::AlreadyTrimmed | Reconciliation::TrimmedPadding => 0,
    }
}

/// A complete `EnginePcm` from fully-decoded raw channels, trimmed per the
/// reconciliation case. `AlreadyTrimmed` f32 takes the Vecs over without a
/// copy (today's bulk behavior); the trim cases copy exactly the keep-window.
fn build_complete_pcm(
    meta: &LoadMeta,
    reconciliation: Reconciliation,
    left: Vec<f32>,
    right: Vec<f32>,
) -> EnginePcm {
    // ADR-119 FR2: the HOST declares the storage format; the engine never
    // decides it (FR3 — the default stays F32, so every existing load and
    // every existing digest is byte-unchanged).
    let format = meta.sample_format;
    if reconciliation == Reconciliation::AlreadyTrimmed {
        // The f32 case keeps its ZERO-COPY Vec takeover — the host's buffers
        // become the engine's storage untouched. Only i16 pays a conversion,
        // and `new_complete` quantizes in the same one pass it already uses
        // for the streaming path (engine_pcm.rs), on the control-plane
        // thread. Never on the RT path.
        return EnginePcm::new_complete(left, right, format);
    }
    let d = keep_window_start(reconciliation, meta) as usize;
    let keep = meta.original_samples as usize;
    let pcm = EnginePcm::new_streaming(meta.original_samples, format);
    pcm.append(0, &left[d..d + keep], &right[d..d + keep]);
    pcm.mark_complete();
    pcm
}

/// Gain/pan validation (rejected, never clamped — project principle).
fn validate_gain(gain: f32) -> Result<(), String> {
    if !gain.is_finite() || !(0.0..=MAX_GAIN).contains(&gain) {
        return Err(format!("gain {gain} outside [0, {MAX_GAIN}] (ADR-092)"));
    }
    Ok(())
}

fn validate_pan(pan: f32) -> Result<(), String> {
    if !pan.is_finite() || !(-1.0..=1.0).contains(&pan) {
        return Err(format!("pan {pan} outside [-1, 1]"));
    }
    Ok(())
}

/// Create an engine pair: the control plane stays with the caller; the
/// `Box<AudioCore>` goes to the platform's audio callback (its pointer is the
/// `core` argument of `maestro_render`).
pub fn create_engine(config: EngineConfig) -> Result<(ControlPlane, Box<AudioCore>), EngineError> {
    if config.sample_rate != ENGINE_SAMPLE_RATE {
        return Err(EngineError::UnsupportedSampleRate { got: config.sample_rate });
    }
    if config.beats_per_bar == 0 {
        return Err(EngineError::InvalidConfig { reason: "beats_per_bar must be ≥ 1".into() });
    }
    // §4.11 budget override (ADR-098): rejected, never clamped.
    if config.pcm_budget_bytes == Some(0) {
        return Err(EngineError::InvalidConfig {
            reason: "pcm_budget_bytes must be ≥ 1 byte (omit it for the 96 MiB default)".into(),
        });
    }
    let timeline = Timeline::new(config.initial_tempo_mbpm)
        .map_err(|e| EngineError::InvalidConfig { reason: e.to_string() })?;

    let (cmd_tx, cmd_rx) = rtrb::RingBuffer::new(COMMAND_QUEUE_CAPACITY);
    let (evt_tx, evt_rx) = rtrb::RingBuffer::new(EVENT_QUEUE_CAPACITY);
    let (rec_tx, rec_rx) = rtrb::RingBuffer::new(RECLAIM_QUEUE_CAPACITY);
    let snapshot = Arc::new(SeqLock::new(RawSnapshot::empty(config.initial_tempo_mbpm)));

    let core = Box::new(AudioCore::new(
        timeline,
        config.beats_per_bar,
        cmd_rx,
        evt_tx,
        rec_tx,
        Arc::clone(&snapshot),
    ));
    let plane = ControlPlane {
        commands: cmd_tx,
        events: evt_rx,
        reclaim: rec_rx,
        snapshot,
        tempo_mbpm: config.initial_tempo_mbpm,
        play_commanded: false,
        pcm_budget_bytes: config.pcm_budget_bytes.unwrap_or(PCM_BUDGET_BYTES),
        config,
        slots: std::array::from_fn(|_| SlotState::Empty),
        lru_clock: 0,
        events_dropped_seen: 0,
        pending_events: Vec::new(),
    };
    Ok((plane, core))
}

impl ControlPlane {
    /// Validate and enqueue a command. Rejection is explicit and reasoned;
    /// nothing is ever silently clamped or dropped (project principle:
    /// no fallbacks).
    pub fn dispatch(&mut self, cmd: Command) -> CommandResult {
        match &cmd {
            Command::SetTempo { mbpm } => {
                if !(MIN_TEMPO_MBPM..=MAX_TEMPO_MBPM).contains(mbpm) {
                    return CommandResult::Rejected {
                        reason: format!(
                            "tempo {mbpm} mBPM outside [{MIN_TEMPO_MBPM}, {MAX_TEMPO_MBPM}]"
                        ),
                    };
                }
                // §4.8: every loaded track's rate E/O must stay in [0.25, 4].
                // Live tempo changes are M1 surface — applied at the next
                // block boundary; the engine echoes Event::TempoApplied.
                for (slot, s) in self.slots.iter().enumerate() {
                    let o_mbpm = match s {
                        SlotState::Loaded(l) => l.o_mbpm,
                        // Streaming installs are loaded slots for §4.8: the
                        // starvation predicate consumes source at the live
                        // rate, so the clamp applies identically (ADR-112 §5.5).
                        SlotState::Streaming(st) => st.meta.original_tempo_mbpm,
                        SlotState::Empty => continue,
                    };
                    if let Err(reason) = rate_in_range(*mbpm, o_mbpm, slot) {
                        return CommandResult::Rejected { reason };
                    }
                }
                if let Err(reason) = self.send(RtCommand::SetTempo { mbpm: *mbpm }) {
                    return CommandResult::Rejected { reason };
                }
                // Mirror reflects only commands actually issued.
                self.tempo_mbpm = *mbpm;
            }
            Command::Play => {
                let result = self.send_cmd(RtCommand::Play);
                if matches!(result, CommandResult::Ok) {
                    self.play_commanded = true;
                }
                return result;
            }
            Command::Stop => {
                let result = self.send_cmd(RtCommand::Stop);
                if matches!(result, CommandResult::Ok) {
                    self.play_commanded = false;
                }
                return result;
            }
            Command::Pause => return self.send_cmd(RtCommand::Pause),
            Command::Resume => return self.send_cmd(RtCommand::Resume),
            Command::Activate { slot, quantize, transport } => {
                if let Err(r) = self.expect_loaded(*slot) {
                    return r;
                }
                return self.send_cmd(RtCommand::Activate {
                    slot: *slot,
                    quantize: *quantize,
                    auto_start: *transport == TransportPolicy::Auto,
                });
            }
            Command::Deactivate { slot, transport } => {
                if let Err(r) = self.expect_loaded(*slot) {
                    return r;
                }
                return self.send_cmd(RtCommand::Deactivate {
                    slot: *slot,
                    auto_stop: *transport == TransportPolicy::Auto,
                });
            }
            Command::RemoveTrack { slot } => {
                if let Err(r) = self.expect_loaded(*slot) {
                    return r;
                }
                if let Err(reason) = self.send(RtCommand::RemoveTrack { slot: *slot }) {
                    return CommandResult::Rejected { reason };
                }
                // Frees the slot mirror AND its §4.11 variation cache; the
                // audio core hands its TrackState back via the reclaim queue.
                self.slots[*slot as usize] = SlotState::Empty;
            }
            Command::SetGain { slot, gain } => {
                if let Err(r) = self.expect_loaded(*slot) {
                    return r;
                }
                if let Err(reason) = validate_gain(*gain) {
                    return CommandResult::Rejected { reason };
                }
                return self.send_cmd(RtCommand::SetGain { slot: *slot, gain: *gain });
            }
            Command::SetPan { slot, pan } => {
                if let Err(r) = self.expect_loaded(*slot) {
                    return r;
                }
                if let Err(reason) = validate_pan(*pan) {
                    return CommandResult::Rejected { reason };
                }
                return self.send_cmd(RtCommand::SetPan { slot: *slot, pan: *pan });
            }
            Command::SetMute { slot, muted } => {
                if let Err(r) = self.expect_loaded(*slot) {
                    return r;
                }
                return self.send_cmd(RtCommand::SetMute { slot: *slot, muted: *muted });
            }
            Command::SetSolo { slot, solo } => {
                if let Err(r) = self.expect_loaded(*slot) {
                    return r;
                }
                return self.send_cmd(RtCommand::SetSolo { slot: *slot, solo: *solo });
            }
            Command::SetPitch { slot, semitones } => {
                if let Err(r) = self.expect_loaded(*slot) {
                    return r;
                }
                // §4.8: ±12 semitones, rejected (never clamped) outside.
                if !semitones.is_finite() || semitones.abs() > MAX_PITCH_SEMITONES {
                    return CommandResult::Rejected {
                        reason: format!(
                            "pitch {semitones} semitones outside ±{MAX_PITCH_SEMITONES} (spec §4.8)"
                        ),
                    };
                }
                return self.send_cmd(RtCommand::SetPitch { slot: *slot, semitones: *semitones });
            }
            Command::SwitchVariation { slot, variation } => {
                return self.switch_variation(*slot, *variation);
            }
            Command::SetTimelineOrigin { bu } => {
                // ADR-113: stopped-transport only. `play_commanded` is the
                // dispatch-order truth — the snapshot alone is one block
                // stale, so an origin dispatched right after a Play would
                // have been accepted here and then dropped silently by the
                // RT arm (queue order applies Play first). The snapshot
                // check stays as belt-and-braces.
                if self.play_commanded || self.raw_snapshot().playing {
                    return CommandResult::Rejected {
                        reason: "setTimelineOrigin requires a stopped transport (ADR-113)".into(),
                    };
                }
                return self.send_cmd(RtCommand::SetTimelineOrigin { bu: *bu });
            }
            Command::TrimPcmCache { target_bytes } => {
                // FR17 (ADR-104): control-plane only — nothing here touches
                // the RT thread, installed PCM, or the configured budget.
                self.evict_lru_until(*target_bytes);
            }
        }
        CommandResult::Ok
    }

    /// §4.11 variation switch. Cached → swap at the next block boundary;
    /// not cached → acknowledged, `VariationLoadRequired` asks the platform
    /// to decode + `load_pcm`, and the swap applies on load completion.
    fn switch_variation(&mut self, slot: u8, variation: u8) -> CommandResult {
        if let Err(r) = self.expect_loaded(slot) {
            return r;
        }
        // One stream per slot, no interleaving (ADR-112 §5.5 v1).
        if matches!(self.slots[slot as usize], SlotState::Streaming(_)) {
            return CommandResult::Rejected {
                reason: format!("stream in progress on slot {slot} (ADR-112 §5.5)"),
            };
        }
        let SlotState::Loaded(l) = &mut self.slots[slot as usize] else { unreachable!() };
        if l.variation == variation {
            l.pending_variation = None;
            return CommandResult::Ok; // already current — no-op
        }
        if let Some(idx) = l.cached.iter().position(|c| c.variation == variation) {
            // Send first, mutate the control-side mirror only on success —
            // a rejected send (wasm full-queue case) must leave the mirror
            // consistent with what the audio core actually received. The
            // cached entry's own map rides the command: lengths may differ
            // per variation (ADR-109), and map + PCM must install atomically.
            let pcm = Arc::clone(&l.cached[idx].pcm);
            let map = l.cached[idx].map;
            if let Err(reason) = self.send(RtCommand::SwapVariation {
                slot,
                variation,
                pcm: Arc::clone(&pcm),
                map,
            }) {
                return CommandResult::Rejected { reason };
            }
            self.lru_clock += 1;
            let stamp = self.lru_clock;
            let SlotState::Loaded(l) = &mut self.slots[slot as usize] else { unreachable!() };
            let idx = l
                .cached
                .iter()
                .position(|c| c.variation == variation)
                .expect("cached entry still present");
            l.cached.swap_remove(idx);
            let outgoing = CachedVariation {
                variation: l.variation,
                pcm: Arc::clone(&l.current_pcm),
                map: l.map,
                last_audible: stamp,
            };
            l.current_pcm = pcm;
            l.map = map;
            l.variation = variation;
            l.pending_variation = None;
            l.cached.push(outgoing);
            self.enforce_pcm_budget();
        } else {
            l.pending_variation = Some(variation);
            self.pending_events.push(Event::VariationLoadRequired { slot, variation });
        }
        CommandResult::Ok
    }

    /// §4.11 eviction: above the per-song budget, drop the least-recently-
    /// audible CACHED variation (installed/audible ones are never evicted).
    fn enforce_pcm_budget(&mut self) {
        self.evict_lru_until(self.pcm_budget_bytes);
    }

    /// The shared eviction loop: drop least-recently-audible CACHED entries
    /// until resident ≤ `limit`. Callers: `enforce_pcm_budget` (the standing
    /// §4.11 budget) and `Command::TrimPcmCache` (FR17 one-shot pressure
    /// target). Installed PCM is structurally unevictable — only `cached`
    /// entries are candidates — so a `limit` of 0 empties every slot cache
    /// and stops.
    fn evict_lru_until(&mut self, limit: u64) {
        loop {
            if self.pcm_bytes_in_use() <= limit {
                return;
            }
            // Globally oldest cached entry across all slots.
            let mut oldest: Option<(usize, usize, u64)> = None;
            for (s, state) in self.slots.iter().enumerate() {
                if let SlotState::Loaded(l) = state {
                    for (i, c) in l.cached.iter().enumerate() {
                        if oldest.is_none_or(|(_, _, best)| c.last_audible < best) {
                            oldest = Some((s, i, c.last_audible));
                        }
                    }
                }
            }
            let Some((s, i, _)) = oldest else { return }; // nothing evictable
            let SlotState::Loaded(l) = &mut self.slots[s] else { unreachable!() };
            let evicted = l.cached.swap_remove(i);
            self.pending_events
                .push(Event::VariationEvicted { slot: s as u8, variation: evicted.variation });
        }
    }

    /// PCM bytes resident in this engine instance (installed + cached) —
    /// reported in the snapshot; the §4.11 budget gate reads it.
    pub fn pcm_bytes_in_use(&self) -> u64 {
        self.slots
            .iter()
            .map(|s| match s {
                SlotState::Loaded(l) => {
                    l.current_pcm.bytes() + l.cached.iter().map(|c| c.pcm.bytes()).sum::<u64>()
                }
                // Allocated capacity, not frontier: a streaming track's
                // budget cost is its full size from begin (ADR-112 §5.3).
                SlotState::Streaming(st) => st.pcm.bytes(),
                SlotState::Empty => 0,
            })
            .sum()
    }

    /// The bulk-data lane (spec §3.6). Performs §3.4 length reconciliation
    /// and loop-identity validation; on success installs the track at its slot.
    ///
    /// Since ADR-112 this is a COMPOSITION, not a separate path: a fresh
    /// install on an empty slot is literally begin + one (uncapped) append +
    /// finalize over the streaming internals, so there is one validation/trim
    /// implementation and no second code path to drift. The wire contract is
    /// unchanged. Occupied-slot paths (variation swap, §4.11 park) cannot pass
    /// through a v1 begin — it requires an Empty slot — so they construct a
    /// complete `EnginePcm` directly through the same shared trim helpers
    /// (ADR-112 §14#1); T-STREAM pins the byte-exact equivalence.
    ///
    /// `left`/`right` are the decoded PCM channels at 48 kHz. The platform shim
    /// guarantees no resampling and no channel conversion happened (spec §3.4#3)
    /// and reports the decoded rate in `meta.sample_rate` (re-validated here —
    /// defense in depth, §4.12); this function guarantees the length is exactly
    /// consistent with the stem's musical identity, or the load fails loudly.
    pub fn load_pcm(&mut self, meta: &LoadMeta, left: Vec<f32>, right: Vec<f32>) -> Result<Reconciliation, String> {
        let slot = meta.slot;
        let variation = meta.variation;
        let result = self.load_pcm_inner(meta, left, right);
        match &result {
            Ok(rec) => self.pending_events.push(Event::TrackLoaded {
                slot,
                variation,
                reconciliation: rec.as_str().to_string(),
            }),
            Err(reason) => self
                .pending_events
                .push(Event::TrackLoadFailed { slot, reason: reason.clone() }),
        }
        result
    }

    fn load_pcm_inner(&mut self, meta: &LoadMeta, left: Vec<f32>, right: Vec<f32>) -> Result<Reconciliation, String> {
        if meta.slot as usize >= MAX_TRACKS {
            return Err(format!("slot {} out of range (MAX_TRACKS = {MAX_TRACKS})", meta.slot));
        }
        // One stream per slot, no interleaving (ADR-112 §5.5 v1).
        if matches!(self.slots[meta.slot as usize], SlotState::Streaming(_)) {
            return Err(format!("stream in progress on slot {} (ADR-112 §5.5)", meta.slot));
        }
        // §3.4#3 / §4.7: the engine never resamples; the shim must hand us
        // 48 kHz exactly. Hard failure, no fallback.
        if meta.sample_rate != ENGINE_SAMPLE_RATE {
            return Err(format!(
                "sample rate {} != {ENGINE_SAMPLE_RATE} — the engine never resamples; \
                 load fails (spec §3.4)",
                meta.sample_rate
            ));
        }
        if left.len() != right.len() {
            return Err(format!("channel length mismatch: L={} R={}", left.len(), right.len()));
        }
        validate_gain(meta.gain)?;
        validate_pan(meta.pan)?;
        let l_dec = left.len() as u64;

        // §3.4#4 decoder-agnostic length reconciliation — the double-trim
        // killer. Shared verbatim with the streaming begin (ADR-112 §5.5):
        // bulk classifies by the actual decoded length, begin by the
        // declared total; the keep-window trim below is the same math.
        let reconciliation = classify_reconciliation(meta, l_dec)?;

        // Loop identity (§3.4#2): loop_beats is authoritative; PCM validates
        // against it, never the reverse.
        let map = TrackTimeline::new(meta.original_tempo_mbpm, meta.loop_beats, meta.original_samples)
            .map_err(|e| e.to_string())?;

        // §4.8: the track's implicit rate at the current engine tempo must be
        // inside [0.25, 4.0] — rejected here, never discovered mid-render.
        rate_in_range(self.tempo_mbpm, meta.original_tempo_mbpm, meta.slot as usize)?;

        // Fresh install on an empty slot: bulk load IS the streaming
        // composition — begin + one (uncapped) append + finalize over the
        // same internals (ADR-112 §5.3: one implementation, zero wire change).
        if matches!(self.slots[meta.slot as usize], SlotState::Empty) && !meta.cache_only {
            let smeta = StreamLoadMeta {
                meta: meta.clone(),
                declared_total_samples: l_dec,
                // ADR-119 FR2: carry the bulk meta's format through, do NOT
                // re-hardcode F32 here. A fresh install is the streaming
                // composition (ADR-112 §5.3), so this branch is where the
                // INSTALLED variation's storage is actually decided —
                // `build_complete_pcm` below only ever sees occupied slots
                // (swaps and §4.11 parks).
                //
                // Missing this leaves installs f32 while parks are i16, which
                // is precisely PRD §3.1's rejected Option A arriving by
                // accident: the accounting only half-halves, and one user
                // switch makes the formats trade places (the parked buffer
                // BECOMES the installed one, with no re-expansion anywhere).
                sample_format: meta.sample_format,
            };
            self.begin_stream_inner(&smeta)?;
            // A failed append on the just-begun stream tears it down so the
            // bulk contract (failure leaves the slot empty) holds.
            self.append_stream_inner(meta.slot, &left, &right)
                .inspect_err(|_| self.teardown_stream(meta.slot))?;
            return self.finalize_stream_inner(meta.slot);
        }

        // Occupied-slot paths (variation swap / §4.11 park): build the
        // complete buffer directly — AlreadyTrimmed f32 takes over the Vecs
        // (zero copy, today's bulk behavior); the trim cases copy exactly
        // the keep-window through the same trim math as streaming appends.
        let pcm = Arc::new(build_complete_pcm(meta, reconciliation, left, right));

        // Variation switch (spec §3.2): loading a different variation into an
        // occupied slot swaps the PCM + timing map engine-side (atomic Arc +
        // map swap + re-anchor + one-block crossfade at a block boundary) —
        // also the completion path of an async SwitchVariation (§4.11).
        // Variations of a stem may differ in loop length (ADR-109) — each
        // file's own loop identity was validated by `TrackTimeline::new`
        // above — but they must share the original tempo `O` (so `u = bu/O`
        // is variation-independent and a switch resumes at the same global
        // musical position) AND reconcile the same way (spec §3.4#4).
        if let SlotState::Loaded(l) = &self.slots[meta.slot as usize] {
            // Shared-identity + reconciliation-stability validation applies
            // to EVERY load onto an occupied slot — §4.11 cache-only prewarms
            // and same-variation reloads included: a mismatch means a
            // mis-tagged or mis-decoded file, never a curiosity.
            {
                if l.o_mbpm != meta.original_tempo_mbpm {
                    return Err(format!(
                        "variation {} identity mismatch on slot {}: installed O={} vs \
                         incoming O={} — all variations of a stem must share the original \
                         tempo (spec §3.4, ADR-109)",
                        meta.variation, meta.slot, l.o_mbpm, meta.original_tempo_mbpm,
                    ));
                }
                if l.reconciliation != reconciliation {
                    return Err(format!(
                        "variation {} of slot {} reconciled as {:?} but the installed stem \
                         reconciled as {:?} — reconciliation must be stable across variations \
                         (spec §3.4#4); the file is mis-tagged or mis-decoded",
                        meta.variation, meta.slot, reconciliation, l.reconciliation,
                    ));
                }
                // A re-load of the INSTALLED variation index must agree with
                // what is installed — a length change under the same index is
                // a mis-tagged file, never a curiosity (ADR-109).
                if l.variation == meta.variation && l.map != map {
                    return Err(format!(
                        "variation {} re-load on slot {} declares loop_beats={} / \
                         pcm_len={} but the installed copy has loop_beats={} / pcm_len={} — \
                         the file is mis-tagged or mis-decoded (ADR-109)",
                        meta.variation,
                        meta.slot,
                        map.loop_beats(),
                        map.pcm_len(),
                        l.map.loop_beats(),
                        l.map.pcm_len(),
                    ));
                }
            }
            if l.variation != meta.variation {
                // §4.11 prewarm (ADR-098): park in the slot's variation cache
                // without swapping — the RT thread is never touched, so
                // positions/digests are unaffected by construction. A pending
                // async switch to exactly this variation completes instead
                // (the user asked for it; parking would strand the switch).
                if meta.cache_only && l.pending_variation != Some(meta.variation) {
                    self.lru_clock += 1;
                    let stamp = self.lru_clock;
                    let SlotState::Loaded(l) = &mut self.slots[meta.slot as usize] else {
                        unreachable!()
                    };
                    // A re-prewarm of an already-cached variation replaces
                    // the entry (fresh stamp, fresh PCM, fresh map).
                    l.cached.retain(|c| c.variation != meta.variation);
                    l.cached.push(CachedVariation {
                        variation: meta.variation,
                        pcm,
                        map,
                        last_audible: stamp,
                    });
                    self.enforce_pcm_budget();
                    return Ok(reconciliation);
                }
                // Send first; the mirror mutates only on success (see
                // switch_variation). The incoming map installs atomically
                // with the PCM (ADR-109).
                self.send(RtCommand::SwapVariation {
                    slot: meta.slot,
                    variation: meta.variation,
                    pcm: Arc::clone(&pcm),
                    map,
                })?;
                self.lru_clock += 1;
                let stamp = self.lru_clock;
                let SlotState::Loaded(l) = &mut self.slots[meta.slot as usize] else {
                    unreachable!()
                };
                // The outgoing current variation goes to the §4.11 cache; a
                // re-load of an already-cached variation replaces the entry.
                let outgoing = CachedVariation {
                    variation: l.variation,
                    pcm: Arc::clone(&l.current_pcm),
                    map: l.map,
                    last_audible: stamp,
                };
                l.cached
                    .retain(|c| c.variation != outgoing.variation && c.variation != meta.variation);
                l.cached.push(outgoing);
                l.current_pcm = pcm;
                l.map = map;
                l.variation = meta.variation;
                if l.pending_variation == Some(meta.variation) {
                    l.pending_variation = None;
                }
                self.enforce_pcm_budget();
                return Ok(reconciliation);
            }
            if meta.cache_only {
                // Cache-only load of the INSTALLED variation: already
                // resident — no-op success. This must never fall through to
                // the fresh-install path below, which resets the slot and
                // wipes the §4.11 cache (the prewarm/switch race hazard,
                // ADR-098).
                return Ok(reconciliation);
            }
            // FULL load of the INSTALLED variation — the async-switch
            // dedupe-loser (ADR-098's other half): a §4.11 park completed
            // the pending switch first and the platform's answer load lands
            // late, now targeting the current variation. Falling through to
            // the fresh-install path would replace the LIVE TrackState with
            // an inactive one: a playing track dies silently — no
            // Deactivated/Removed event ever fires, so the UI keeps showing
            // it as playing while it renders nothing (the "silent variation
            // switch" bake-watch item, ADR-096) — and the slot's §4.11 cache
            // is wiped. Refresh the content through the swap path instead:
            // identical derived position, live state (active/mute/solo/
            // pitch/live gain) preserved, one inaudible same-content
            // crossfade. Identity + map equality were validated above, so
            // the incoming buffer is interchangeable with the installed one.
            self.send(RtCommand::SwapVariation {
                slot: meta.slot,
                variation: meta.variation,
                pcm: Arc::clone(&pcm),
                map,
            })?;
            let SlotState::Loaded(l) = &mut self.slots[meta.slot as usize] else {
                unreachable!()
            };
            l.current_pcm = pcm;
            l.map = map;
            if l.pending_variation == Some(meta.variation) {
                l.pending_variation = None;
            }
            return Ok(reconciliation);
        }

        // §4.11 prewarm requires an installed track — there is nothing to
        // park against on an empty slot (ADR-098).
        if meta.cache_only {
            return Err(format!(
                "cache-only load requires an installed track on slot {} (spec §4.11 prewarm)",
                meta.slot
            ));
        }

        // Per-track stretcher (spec §4.3), preconfigured and preallocated here
        // so the audio thread never allocates.
        //
        // preset_default is REQUIRED. The spec's budget fallback (§4.3,
        // new_cheaper globally) was tried on-device 2026-06-10 and REJECTED:
        // at rate 2.0 it attenuates click transients ~3× (109 of 146 onsets
        // fell below the detection threshold) and shifts detected attacks a
        // constant ~5,515 samples (~115 ms) — audibly out of sync (caught by
        // ear before the analyzer ran; capture evidence in
        // documentation/validation-runs/). The C0 budget gate is instead met
        // via Plan B (spec §3.3) — see render pump.
        let stretcher = Box::new(Stretcher::new(2, ENGINE_SAMPLE_RATE));
        let state = Box::new(crate::audio_core::TrackState::new(
            map,
            Arc::clone(&pcm),
            meta.gain,
            meta.pan,
            meta.variation,
            stretcher,
        ));

        self.send(RtCommand::InstallTrack { slot: meta.slot, state })?;
        self.slots[meta.slot as usize] = SlotState::Loaded(Box::new(LoadedSlot {
            o_mbpm: meta.original_tempo_mbpm,
            variation: meta.variation,
            reconciliation,
            map,
            current_pcm: pcm,
            cached: Vec::new(),
            pending_variation: None,
        }));
        self.enforce_pcm_budget();
        Ok(reconciliation)
    }

    // ------------------------------------------------------------------------
    // Streaming load lifecycle (ADR-112 §5.5). Public wrappers own the event
    // surface (TrackLoaded fires at finalize only — preserving the existing
    // contract that trackLoaded ⇒ complete PCM; teardown failures fire
    // TrackLoadFailed); the `*_inner` functions are the shared internals the
    // bulk `load_pcm` composition also runs through.
    // ------------------------------------------------------------------------

    /// Begin a streaming load: everything identity-shaped validates here, in
    /// order, before any byte moves (ADR-112 §5.5). On success the track is
    /// installed (frontier 0, visible in the snapshot, inactive).
    pub fn begin_pcm_stream(&mut self, smeta: &StreamLoadMeta) -> Result<(), String> {
        let result = self.begin_stream_inner(smeta);
        if let Err(reason) = &result {
            self.pending_events
                .push(Event::TrackLoadFailed { slot: smeta.meta.slot, reason: reason.clone() });
        }
        result
    }

    fn begin_stream_inner(&mut self, smeta: &StreamLoadMeta) -> Result<(), String> {
        let meta = &smeta.meta;
        if meta.slot as usize >= MAX_TRACKS {
            return Err(format!("slot {} out of range (MAX_TRACKS = {MAX_TRACKS})", meta.slot));
        }
        // 1. v1 restrictions: the slot must be Empty; cache_only streaming
        //    parks are out of scope (rejected at begin, ADR-112 §5.5).
        if !matches!(self.slots[meta.slot as usize], SlotState::Empty) {
            return Err(format!(
                "slot {} is occupied — v1 streams install on empty slots only (ADR-112 §5.5)",
                meta.slot
            ));
        }
        if meta.cache_only {
            return Err("cache_only streaming parks are not supported (ADR-112 §5.5 v1)".into());
        }
        // SampleFormat::I16 (Tier 2a, ADR-112 §6.2) is accepted here: the
        // buffer stores i16 (halving residency and budget accounting) and
        // converts back to f32 in the read copy loop. Positions are
        // storage-blind; the f32-vs-i16 null test pins the audio residual.
        // 2. §3.4#3: 48 kHz exactly; the engine never resamples.
        if meta.sample_rate != ENGINE_SAMPLE_RATE {
            return Err(format!(
                "sample rate {} != {ENGINE_SAMPLE_RATE} — the engine never resamples; \
                 load fails (spec §3.4)",
                meta.sample_rate
            ));
        }
        validate_gain(meta.gain)?;
        validate_pan(meta.pan)?;
        // 3. Reconciliation case from the declared total (exact 3-way match).
        let reconciliation = classify_reconciliation(meta, smeta.declared_total_samples)?;
        // 4. Loop identity (§3.4#2).
        let map = TrackTimeline::new(meta.original_tempo_mbpm, meta.loop_beats, meta.original_samples)
            .map_err(|e| e.to_string())?;
        // 5. §4.8 rate clamp at the current tempo.
        rate_in_range(self.tempo_mbpm, meta.original_tempo_mbpm, meta.slot as usize)?;
        // 6. Exact-size allocation + install (frontier 0, inactive). The
        //    starvation predicate (ADR-112 §5.4) protects the render path
        //    from the not-yet-loaded region from the first block.
        //    TRUST MODEL: unlike bulk load (whose size is bounded by a Vec
        //    the host already decoded), this allocates from the DECLARED
        //    size before any bytes arrive. The loop identity (#4) forces
        //    internal consistency but still admits absurd sizes; the host
        //    is same-app JS/Swift, not a network peer, and §4.11 budget
        //    enforcement runs right after install. A failed allocation
        //    aborts — the same failure bulk load would hit one step later.
        let pcm = Arc::new(EnginePcm::new_streaming(meta.original_samples, smeta.sample_format));
        let stretcher = Box::new(Stretcher::new(2, ENGINE_SAMPLE_RATE));
        let state = Box::new(crate::audio_core::TrackState::new(
            map,
            Arc::clone(&pcm),
            meta.gain,
            meta.pan,
            meta.variation,
            stretcher,
        ));
        self.send(RtCommand::InstallTrack { slot: meta.slot, state })?;
        self.slots[meta.slot as usize] = SlotState::Streaming(Box::new(StreamingSlot {
            meta: meta.clone(),
            reconciliation,
            declared_total: smeta.declared_total_samples,
            received_total: 0,
            next_seq: 0,
            pcm,
            map,
        }));
        self.enforce_pcm_budget();
        Ok(())
    }

    /// Append one chunk to the slot's stream. `seq` is the per-stream
    /// 0-based append counter (ADR-112 §5.2): MessagePort delivery is
    /// ordered, but `seq` turns host bookkeeping bugs into a loud teardown
    /// instead of corrupt audio. The per-append size cap bounds the
    /// worklet-thread copy.
    pub fn append_pcm_chunk(
        &mut self,
        slot: u8,
        seq: u32,
        left: &[f32],
        right: &[f32],
    ) -> AppendResult {
        if left.len() > MAX_STREAM_APPEND_FRAMES {
            return AppendResult::Rejected {
                reason: format!(
                    "append of {} frames exceeds MAX_STREAM_APPEND_FRAMES ({MAX_STREAM_APPEND_FRAMES}) \
                     — chunk the load (ADR-112 §5.2)",
                    left.len()
                ),
            };
        }
        let expected_seq = match &self.slots.get(slot as usize) {
            Some(SlotState::Streaming(s)) => s.next_seq,
            _ => {
                return AppendResult::Rejected {
                    reason: format!("no stream in progress on slot {slot}"),
                }
            }
        };
        if seq != expected_seq {
            // Out-of-order append: the sample stream is unrecoverable —
            // teardown (ADR-112 §5.5 state machine).
            let reason = format!(
                "out-of-order append on slot {slot}: seq {seq}, expected {expected_seq} — \
                 stream torn down (ADR-112 §5.2)"
            );
            self.teardown_stream(slot);
            self.pending_events
                .push(Event::TrackLoadFailed { slot, reason: reason.clone() });
            return AppendResult::Rejected { reason };
        }
        match self.append_stream_inner(slot, left, right) {
            Ok((frontier, complete)) => {
                let SlotState::Streaming(s) = &mut self.slots[slot as usize] else {
                    unreachable!()
                };
                s.next_seq += 1;
                AppendResult::Ok { frontier, complete }
            }
            Err(reason) => AppendResult::Rejected { reason },
        }
    }

    /// The shared append internals: raw-stream bookkeeping + keep-window trim
    /// + frontier publication. Returns (frontier, received == declared).
    fn append_stream_inner(&mut self, slot: u8, left: &[f32], right: &[f32]) -> Result<(u64, bool), String> {
        let SlotState::Streaming(s) = &mut self.slots[slot as usize] else {
            return Err(format!("no stream in progress on slot {slot}"));
        };
        if left.len() != right.len() {
            return Err(format!("channel length mismatch: L={} R={}", left.len(), right.len()));
        }
        if left.is_empty() {
            return Err("empty append".into());
        }
        let n = left.len() as u64;
        // Over-delivery rejects the append immediately (ADR-112 §5.5): the
        // host declared its total at begin; exceeding it breaks attestation.
        if s.received_total + n > s.declared_total {
            return Err(format!(
                "over-delivery on slot {slot}: {} + {n} raw samples exceeds the declared \
                 total {} (ADR-112 §5.5)",
                s.received_total, s.declared_total
            ));
        }
        // Trim-on-the-fly (ADR-112 §5.5): raw stream index s ∈ [D, D + orig)
        // is kept; samples outside are counted but discarded (the trailing
        // end_padding must still be delivered — received_total is the
        // honesty check). The frontier advances only for kept samples.
        let d = keep_window_start(s.reconciliation, &s.meta);
        let keep_end = d + s.meta.original_samples;
        let chunk_start = s.received_total;
        let chunk_end = chunk_start + n;
        let kept_start = chunk_start.max(d);
        let kept_end = chunk_end.min(keep_end);
        if kept_end > kept_start {
            let src_from = (kept_start - chunk_start) as usize;
            let src_to = (kept_end - chunk_start) as usize;
            s.pcm.append(kept_start - d, &left[src_from..src_to], &right[src_from..src_to]);
        }
        s.received_total = chunk_end;
        Ok((s.pcm.frontier(), s.received_total == s.declared_total))
    }

    /// Finalize the slot's stream: verify the host delivered exactly the
    /// declared total (⇒ frontier == pcm_len), mark complete, transition to
    /// Loaded, fire `Event::TrackLoaded` with the begin-time reconciliation.
    /// A mismatch is a hard failure WITH track removal (No Fallbacks: the
    /// slot was audibly playing host-attested data the host has now failed
    /// to attest — ADR-112 §5.5).
    pub fn finalize_pcm_stream(&mut self, slot: u8) -> Result<Reconciliation, String> {
        let result = self.finalize_stream_inner(slot);
        match &result {
            Ok(rec) => {
                let variation = match &self.slots[slot as usize] {
                    SlotState::Loaded(l) => l.variation,
                    _ => 0,
                };
                self.pending_events.push(Event::TrackLoaded {
                    slot,
                    variation,
                    reconciliation: rec.as_str().to_string(),
                });
            }
            Err(reason) => self
                .pending_events
                .push(Event::TrackLoadFailed { slot, reason: reason.clone() }),
        }
        result
    }

    fn finalize_stream_inner(&mut self, slot: u8) -> Result<Reconciliation, String> {
        let SlotState::Streaming(s) = &self.slots[slot as usize] else {
            return Err(format!("no stream in progress on slot {slot}"));
        };
        if s.received_total != s.declared_total {
            let reason = format!(
                "finalize under-delivery on slot {slot}: received {} of the declared {} raw \
                 samples — track removed (ADR-112 §5.5, No Fallbacks)",
                s.received_total, s.declared_total
            );
            self.teardown_stream(slot);
            return Err(reason);
        }
        debug_assert_eq!(s.pcm.frontier(), s.map.pcm_len());
        s.pcm.mark_complete();
        let SlotState::Streaming(s) =
            std::mem::replace(&mut self.slots[slot as usize], SlotState::Empty)
        else {
            unreachable!()
        };
        let reconciliation = s.reconciliation;
        self.slots[slot as usize] = SlotState::Loaded(Box::new(LoadedSlot {
            o_mbpm: s.meta.original_tempo_mbpm,
            variation: s.meta.variation,
            reconciliation,
            map: s.map,
            current_pcm: s.pcm,
            cached: Vec::new(),
            pending_variation: None,
        }));
        Ok(reconciliation)
    }

    /// Abort the slot's stream: teardown + `Event::TrackLoadFailed` (the
    /// state machine's failure event; user-initiated `RemoveTrack` is the
    /// path that fires `TrackRemoved` instead — ADR-112 §5.5).
    pub fn abort_pcm_stream(&mut self, slot: u8) -> Result<(), String> {
        if !matches!(self.slots.get(slot as usize), Some(SlotState::Streaming(_))) {
            return Err(format!("no stream in progress on slot {slot}"));
        }
        self.teardown_stream(slot);
        self.pending_events.push(Event::TrackLoadFailed {
            slot,
            reason: "stream aborted by host (ADR-112 §5.5)".into(),
        });
        Ok(())
    }

    /// Stream teardown: slot → Empty + `RtCommand::RemoveTrack`. The buffers
    /// are reclaimed off-RT (the reclaim queue, spec §3.3) — never a dealloc
    /// on the audio thread.
    fn teardown_stream(&mut self, slot: u8) {
        if !matches!(self.slots.get(slot as usize), Some(SlotState::Streaming(_))) {
            return;
        }
        // Send first, mutate on success (the switch_variation discipline). A
        // full-queue rejection on wasm leaves the slot Streaming so the host
        // can retry the abort once the render drains.
        if self.send(RtCommand::RemoveTrack { slot }).is_ok() {
            self.slots[slot as usize] = SlotState::Empty;
        }
    }

    /// Read the seqlock'd snapshot and convert to the wire schema. The only
    /// state egress (spec §3.5).
    pub fn snapshot(&self) -> Snapshot {
        let raw = self.raw_snapshot();
        let tracks = raw
            .tracks
            .iter()
            .enumerate()
            .filter(|(_, t)| t.occupied)
            .map(|(slot, t)| TrackSnap {
                slot: slot as u8,
                status: if t.active { TrackStatus::Active } else { TrackStatus::Loaded },
                variation: t.variation,
                gain: t.gain,
                pan: t.pan,
                muted: t.muted,
                solo: t.solo,
                loop_beats: t.loop_beats,
                loop_count: t.loop_count,
                loop_progress: t.loop_progress,
                rms_l: t.rms_l,
                rms_r: t.rms_r,
                peak_l: t.peak_l,
                peak_r: t.peak_r,
                loaded_samples: t.loaded_samples,
                streaming: t.streaming,
                starved: t.starved,
            })
            .collect();
        Snapshot {
            engine_frame: raw.frame,
            tempo_mbpm: raw.tempo_mbpm,
            transport: if raw.playing && raw.paused {
                TransportState::Paused
            } else if raw.playing {
                TransportState::Playing
            } else {
                TransportState::Stopped
            },
            bar: raw.bar,
            beat_in_bar: raw.beat_in_bar,
            timeline_epoch: raw.timeline_epoch,
            underruns: raw.underruns,
            render_ns_max: raw.render_ns_max,
            callback_period_ns: raw.callback_period_ns,
            blocks_total: raw.blocks_total,
            blocks_over_half_period: raw.blocks_over_half_period,
            events_dropped: raw.events_dropped,
            pcm_bytes_in_use: self.pcm_bytes_in_use(),
            tracks,
        }
    }

    /// Drain RT events + control-plane events, and free any resources the
    /// audio thread handed back. Call this from the backend's poll loop.
    pub fn drain_events(&mut self) -> Vec<Event> {
        // Reclaim first: dropping here is the whole point (spec §3.3).
        while let Ok(reclaimed) = self.reclaim.pop() {
            drop(reclaimed);
        }
        // RT event-queue overflow is surfaced, never silent (spec §6.7).
        let dropped_total = self.raw_snapshot().events_dropped;
        if dropped_total > self.events_dropped_seen {
            let dropped = dropped_total - self.events_dropped_seen;
            self.events_dropped_seen = dropped_total;
            self.pending_events.push(Event::EventQueueOverflow { dropped });
        }
        let mut out = std::mem::take(&mut self.pending_events);
        while let Ok(e) = self.events.pop() {
            out.push(match e {
                RtEvent::Started => Event::Started,
                RtEvent::Stopped => Event::Stopped,
                RtEvent::Paused { at_frame } => Event::Paused { at_frame },
                RtEvent::Resumed { at_frame } => Event::Resumed { at_frame },
                RtEvent::TempoApplied { mbpm, at_frame } => Event::TempoApplied { mbpm, at_frame },
                RtEvent::Activated { slot, at_frame } => Event::TrackActivated { slot, at_frame },
                RtEvent::Deactivated { slot, at_frame } => {
                    Event::TrackDeactivated { slot, at_frame }
                }
                RtEvent::MuteApplied { slot, muted, at_frame } => {
                    Event::MuteApplied { slot, muted, at_frame }
                }
                RtEvent::SoloApplied { slot, solo, at_frame } => {
                    Event::SoloApplied { slot, solo, at_frame }
                }
                RtEvent::Removed { slot } => Event::TrackRemoved { slot },
                RtEvent::VariationApplied { slot, variation, at_frame } => {
                    Event::VariationApplied { slot, variation, at_frame }
                }
                RtEvent::LoopWrapped { slot, wrap_count } => Event::LoopWrapped { slot, wrap_count },
                RtEvent::StreamStarved { slot, at_frame, needed, frontier } => {
                    Event::StreamStarved { slot, at_frame, needed, frontier }
                }
                RtEvent::StreamResumed { slot, at_frame } => {
                    Event::StreamResumed { slot, at_frame }
                }
                RtEvent::BudgetExceeded { render_ns, period_ns } => {
                    Event::BudgetExceeded { render_ns, period_ns }
                }
            });
        }
        out
    }

    pub fn config(&self) -> &EngineConfig {
        &self.config
    }

    fn raw_snapshot(&self) -> RawSnapshot {
        self.snapshot.read()
    }

    fn expect_loaded(&self, slot: u8) -> Result<(), CommandResult> {
        // Streaming installs count as loaded (ADR-112 §5.5): the track is in
        // the audio core; starvation covers the not-yet-loaded region.
        if (slot as usize) < MAX_TRACKS
            && matches!(
                self.slots[slot as usize],
                SlotState::Loaded(_) | SlotState::Streaming(_)
            )
        {
            Ok(())
        } else {
            Err(CommandResult::Rejected { reason: format!("slot {slot} has no loaded track") })
        }
    }

    /// Enqueue an RT command. On native targets the control plane blocks its
    /// own (non-audio) thread if the queue is full (spec §3.3) — the audio
    /// thread drains at every block boundary. On `wasm32-unknown-unknown`
    /// the control plane and the renderer share the ONE worklet thread, so
    /// blocking would deadlock the audio thread: a full queue rejects the
    /// command instead — loudly, never silently (the §3.3 blocking choice
    /// presumed a second thread; ADR-095). 256 pending commands without a
    /// render in between only happens when the AudioContext is not running.
    fn send(&mut self, cmd: RtCommand) -> Result<(), String> {
        #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
        {
            self.commands.push(cmd).map_err(|_| {
                format!(
                    "RT command queue full ({COMMAND_QUEUE_CAPACITY} pending) — the worklet \
                     render is not draining (suspended AudioContext?); command rejected \
                     rather than deadlocking the audio thread (ADR-095)"
                )
            })
        }
        #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
        {
            let mut cmd = cmd;
            loop {
                match self.commands.push(cmd) {
                    Ok(()) => return Ok(()),
                    Err(rtrb::PushError::Full(returned)) => {
                        cmd = returned;
                        std::thread::yield_now();
                    }
                }
            }
        }
    }

    /// `send` shaped as a `CommandResult` for the `dispatch` arms.
    fn send_cmd(&mut self, cmd: RtCommand) -> CommandResult {
        match self.send(cmd) {
            Ok(()) => CommandResult::Ok,
            Err(reason) => CommandResult::Rejected { reason },
        }
    }
}
