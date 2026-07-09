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
//! State egress is exactly one call: [`ControlPlane::snapshot`] reads the
//! seqlock'd POD the audio thread publishes each block and converts it to the
//! `maestro-api` schema. JS holds render-only mirrors (spec §3.5).

use std::sync::Arc;

use maestro_api::{
    Command, CommandResult, EngineConfig, Event, LoadMeta, Snapshot, TrackSnap, TrackStatus,
    TransportPolicy, TransportState,
};

use crate::audio_core::{AudioCore, RawSnapshot, Reclaimable, RtCommand, RtEvent, MAX_TRACKS};
use crate::pcm_cache::PcmData;
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

/// Control-side mirror of an installed track: immutable musical identity (for
/// variation-switch and §4.8 validation), the installed PCM, and the §4.11
/// variation cache for this slot.
struct LoadedSlot {
    o_mbpm: u32,
    loop_beats: u32,
    original_samples: u64,
    variation: u8,
    /// §3.4#4: which reconciliation case fired for the first loaded file of
    /// this stem; later variations must match.
    reconciliation: Reconciliation,
    /// The PCM currently installed in the audio core (Arc-shared with it).
    current_pcm: Arc<PcmData>,
    /// Cached non-current variations: (variation, pcm, last-audible stamp).
    /// Stamped when the variation stops being current (or at load if never
    /// current) — the §4.11 eviction order.
    cached: Vec<(u8, Arc<PcmData>, u64)>,
    /// §4.11 async switch: a requested variation awaiting platform load.
    pending_variation: Option<u8>,
}

enum SlotState {
    Empty,
    Loaded(Box<LoadedSlot>),
}

fn pcm_bytes(p: &PcmData) -> u64 {
    p.length_samples * p.num_channels as u64 * 4
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
                    if let SlotState::Loaded(l) = s {
                        if let Err(reason) = rate_in_range(*mbpm, l.o_mbpm, slot) {
                            return CommandResult::Rejected { reason };
                        }
                    }
                }
                if let Err(reason) = self.send(RtCommand::SetTempo { mbpm: *mbpm }) {
                    return CommandResult::Rejected { reason };
                }
                // Mirror reflects only commands actually issued.
                self.tempo_mbpm = *mbpm;
            }
            Command::Play => return self.send_cmd(RtCommand::Play),
            Command::Stop => return self.send_cmd(RtCommand::Stop),
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
        let SlotState::Loaded(l) = &mut self.slots[slot as usize] else { unreachable!() };
        if l.variation == variation {
            l.pending_variation = None;
            return CommandResult::Ok; // already current — no-op
        }
        if let Some(idx) = l.cached.iter().position(|(v, _, _)| *v == variation) {
            // Send first, mutate the control-side mirror only on success —
            // a rejected send (wasm full-queue case) must leave the mirror
            // consistent with what the audio core actually received.
            let pcm = Arc::clone(&l.cached[idx].1);
            if let Err(reason) = self.send(RtCommand::SwapVariation {
                slot,
                variation,
                pcm: Arc::clone(&pcm),
            }) {
                return CommandResult::Rejected { reason };
            }
            self.lru_clock += 1;
            let stamp = self.lru_clock;
            let SlotState::Loaded(l) = &mut self.slots[slot as usize] else { unreachable!() };
            let idx = l
                .cached
                .iter()
                .position(|(v, _, _)| *v == variation)
                .expect("cached entry still present");
            l.cached.swap_remove(idx);
            let outgoing = (l.variation, Arc::clone(&l.current_pcm));
            l.current_pcm = pcm;
            l.variation = variation;
            l.pending_variation = None;
            l.cached.push((outgoing.0, outgoing.1, stamp));
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
                    for (i, (_, _, stamp)) in l.cached.iter().enumerate() {
                        if oldest.is_none_or(|(_, _, best)| *stamp < best) {
                            oldest = Some((s, i, *stamp));
                        }
                    }
                }
            }
            let Some((s, i, _)) = oldest else { return }; // nothing evictable
            let SlotState::Loaded(l) = &mut self.slots[s] else { unreachable!() };
            let (variation, _pcm, _) = l.cached.swap_remove(i);
            self.pending_events.push(Event::VariationEvicted { slot: s as u8, variation });
        }
    }

    /// PCM bytes resident in this engine instance (installed + cached) —
    /// reported in the snapshot; the §4.11 budget gate reads it.
    pub fn pcm_bytes_in_use(&self) -> u64 {
        self.slots
            .iter()
            .map(|s| match s {
                SlotState::Loaded(l) => {
                    pcm_bytes(&l.current_pcm)
                        + l.cached.iter().map(|(_, p, _)| pcm_bytes(p)).sum::<u64>()
                }
                SlotState::Empty => 0,
            })
            .sum()
    }

    /// The one bulk-data call (spec §3.6). Performs §3.4 length reconciliation
    /// and loop-identity validation; on success installs the track at its slot.
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

    fn load_pcm_inner(&mut self, meta: &LoadMeta, mut left: Vec<f32>, mut right: Vec<f32>) -> Result<Reconciliation, String> {
        if meta.slot as usize >= MAX_TRACKS {
            return Err(format!("slot {} out of range (MAX_TRACKS = {MAX_TRACKS})", meta.slot));
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

        // §3.4#4 decoder-agnostic length reconciliation — the double-trim killer.
        // Three EXACT cases (ADR-095 added the third); anything else fails.
        // Where the metadata makes cases coincide (delay or padding of 0)
        // the trims coincide too — order cannot change the outcome.
        let raw_len = meta.original_samples + meta.encoder_delay as u64 + meta.end_padding as u64;
        let padded_len = meta.original_samples + meta.end_padding as u64;
        let reconciliation = if l_dec == meta.original_samples {
            Reconciliation::AlreadyTrimmed
        } else if l_dec == raw_len {
            let delay = meta.encoder_delay as usize;
            let orig = meta.original_samples as usize;
            left.drain(..delay);
            left.truncate(orig);
            right.drain(..delay);
            right.truncate(orig);
            Reconciliation::TrimmedHere
        } else if l_dec == padded_len {
            let orig = meta.original_samples as usize;
            left.truncate(orig);
            right.truncate(orig);
            Reconciliation::TrimmedPadding
        } else {
            return Err(format!(
                "length mismatch: decoded {l_dec}, expected {} (already-trimmed), {raw_len} \
                 (raw decode), or {padded_len} (delay-only trim) — no heuristics, load fails \
                 (spec §3.4)",
                meta.original_samples
            ));
        };

        // Loop identity (§3.4#2): loop_beats is authoritative; PCM validates
        // against it, never the reverse.
        let map = TrackTimeline::new(meta.original_tempo_mbpm, meta.loop_beats, meta.original_samples)
            .map_err(|e| e.to_string())?;

        // §4.8: the track's implicit rate at the current engine tempo must be
        // inside [0.25, 4.0] — rejected here, never discovered mid-render.
        rate_in_range(self.tempo_mbpm, meta.original_tempo_mbpm, meta.slot as usize)?;

        let pcm = Arc::new(PcmData {
            length_samples: left.len() as u64,
            channel_data: vec![left, right],
            sample_rate: ENGINE_SAMPLE_RATE,
            num_channels: 2,
        });

        // Variation switch (spec §3.2): loading a different variation into an
        // occupied slot swaps the PCM engine-side (Arc swap + re-anchor +
        // one-block crossfade at a block boundary) — also the completion path
        // of an async SwitchVariation (§4.11). All variations of a stem must
        // declare identical musical identity AND reconcile the same way
        // (spec §3.4) — anything else is a hard failure, because loop length
        // can never change underneath the timeline (kills D9/D10).
        if let SlotState::Loaded(l) = &self.slots[meta.slot as usize] {
            // Cross-variation identity + reconciliation-stability validation
            // also applies to §4.11 cache-only prewarms (incl. re-loads of
            // the installed variation) — a mismatch means a mis-tagged or
            // mis-decoded file, never a curiosity.
            if l.variation != meta.variation || meta.cache_only {
                if l.o_mbpm != meta.original_tempo_mbpm
                    || l.loop_beats != meta.loop_beats
                    || l.original_samples != meta.original_samples
                {
                    return Err(format!(
                        "variation {} identity mismatch on slot {}: installed \
                         (O={}, loop_beats={}, original_samples={}) \
                         vs incoming (O={}, loop_beats={}, original_samples={}) — all variations \
                         of a stem must be musically identical (spec §3.4)",
                        meta.variation,
                        meta.slot,
                        l.o_mbpm,
                        l.loop_beats,
                        l.original_samples,
                        meta.original_tempo_mbpm,
                        meta.loop_beats,
                        meta.original_samples,
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
                    // the entry (fresh stamp, fresh PCM).
                    l.cached.retain(|(v, _, _)| *v != meta.variation);
                    l.cached.push((meta.variation, pcm, stamp));
                    self.enforce_pcm_budget();
                    return Ok(reconciliation);
                }
                // Send first; the mirror mutates only on success (see
                // switch_variation).
                self.send(RtCommand::SwapVariation {
                    slot: meta.slot,
                    variation: meta.variation,
                    pcm: Arc::clone(&pcm),
                })?;
                self.lru_clock += 1;
                let stamp = self.lru_clock;
                let SlotState::Loaded(l) = &mut self.slots[meta.slot as usize] else {
                    unreachable!()
                };
                // The outgoing current variation goes to the §4.11 cache; a
                // re-load of an already-cached variation replaces the entry.
                let outgoing = (l.variation, Arc::clone(&l.current_pcm), stamp);
                l.cached.retain(|(v, _, _)| *v != outgoing.0 && *v != meta.variation);
                l.cached.push(outgoing);
                l.current_pcm = pcm;
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
            loop_beats: meta.loop_beats,
            original_samples: meta.original_samples,
            variation: meta.variation,
            reconciliation,
            current_pcm: pcm,
            cached: Vec::new(),
            pending_variation: None,
        }));
        self.enforce_pcm_budget();
        Ok(reconciliation)
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
                loop_count: t.loop_count,
                loop_progress: t.loop_progress,
                rms_l: t.rms_l,
                rms_r: t.rms_r,
                peak_l: t.peak_l,
                peak_r: t.peak_r,
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
        if (slot as usize) < MAX_TRACKS
            && matches!(self.slots[slot as usize], SlotState::Loaded(_))
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
