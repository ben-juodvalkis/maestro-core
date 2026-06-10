//! # control_plane — the non-RT half of the engine (spec §3.3 / §3.4 / §3.5)
//!
//! Runs on the caller's thread behind the FFI. Validates commands, performs
//! decode-length reconciliation and loop-identity validation, allocates and
//! preconfigures every resource the audio thread will touch (PCM buffers,
//! stretchers, scratch), then transfers them across the SPSC command queue.
//! It also drains the event and reclaim queues — the audio thread never frees
//! memory.
//!
//! State egress is exactly one call: [`ControlPlane::snapshot`] reads the
//! seqlock'd POD the audio thread publishes each block and converts it to the
//! `maestro-api` schema. JS holds render-only mirrors (spec §3.5).

use std::sync::Arc;

use maestro_api::{Command, CommandResult, EngineConfig, Event, LoadMeta, Snapshot, TrackSnap, TrackStatus, TransportState};

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
/// load event and asserted stable across variations of a stem (M2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reconciliation {
    /// Decoded length == original_samples: decoder already applied gapless trim.
    AlreadyTrimmed,
    /// Decoded length == original + delay + padding: raw decode; we trim exactly.
    TrimmedHere,
}

impl Reconciliation {
    fn as_str(self) -> &'static str {
        match self {
            Self::AlreadyTrimmed => "alreadyTrimmed",
            Self::TrimmedHere => "trimmedHere",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum SlotState {
    #[default]
    Empty,
    Loaded,
}

/// The control-plane handle. One per engine instance; not `Sync` — the FFI
/// layer (UniFFI/WASM) serializes access on its own terms.
pub struct ControlPlane {
    commands: rtrb::Producer<RtCommand>,
    events: rtrb::Consumer<RtEvent>,
    reclaim: rtrb::Consumer<Reclaimable>,
    snapshot: Arc<SeqLock<RawSnapshot>>,
    config: EngineConfig,
    /// Control-side mirror for validation only (is the slot loaded?); the
    /// authoritative runtime state is the audio core's.
    slots: [SlotState; MAX_TRACKS],
    /// Events originated control-side (loads), merged into `drain_events`.
    pending_events: Vec<Event>,
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
        config,
        slots: [SlotState::default(); MAX_TRACKS],
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
                // M0: live tempo changes are M1 scope. The core applies tempo
                // at block boundaries already, but the product gate lives here.
                if self.raw_snapshot().playing {
                    return CommandResult::Rejected {
                        reason: "live tempo changes land in M1; stop the transport first".into(),
                    };
                }
                self.send(RtCommand::SetTempo { mbpm: *mbpm });
            }
            Command::Play => self.send(RtCommand::Play),
            Command::Stop => self.send(RtCommand::Stop),
            Command::Activate { slot } => {
                if let Err(r) = self.expect_loaded(*slot) {
                    return r;
                }
                self.send(RtCommand::Activate { slot: *slot });
            }
            Command::Deactivate { slot } => {
                if let Err(r) = self.expect_loaded(*slot) {
                    return r;
                }
                self.send(RtCommand::Deactivate { slot: *slot });
            }
            Command::RemoveTrack { slot } => {
                if let Err(r) = self.expect_loaded(*slot) {
                    return r;
                }
                self.slots[*slot as usize] = SlotState::Empty;
                self.send(RtCommand::RemoveTrack { slot: *slot });
            }
        }
        CommandResult::Ok
    }

    /// The one bulk-data call (spec §3.6). Performs §3.4 length reconciliation
    /// and loop-identity validation; on success installs the track at its slot.
    ///
    /// `left`/`right` are the decoded PCM channels at 48 kHz. The platform shim
    /// guarantees no resampling and no channel conversion happened (spec §3.4#3);
    /// this function guarantees the length is exactly consistent with the
    /// stem's musical identity, or the load fails loudly.
    pub fn load_pcm(&mut self, meta: &LoadMeta, left: Vec<f32>, right: Vec<f32>) -> Result<Reconciliation, String> {
        let slot = meta.slot;
        let result = self.load_pcm_inner(meta, left, right);
        match &result {
            Ok(rec) => self.pending_events.push(Event::TrackLoaded {
                slot,
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
        if left.len() != right.len() {
            return Err(format!("channel length mismatch: L={} R={}", left.len(), right.len()));
        }
        let l_dec = left.len() as u64;

        // §3.4#4 decoder-agnostic length reconciliation — the double-trim killer.
        let raw_len = meta.original_samples + meta.encoder_delay as u64 + meta.end_padding as u64;
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
        } else {
            return Err(format!(
                "length mismatch: decoded {l_dec}, expected {} (already-trimmed) or {raw_len} \
                 (raw decode) — no heuristics, load fails (spec §3.4)",
                meta.original_samples
            ));
        };

        // Loop identity (§3.4#2): loop_beats is authoritative; PCM validates
        // against it, never the reverse.
        let map = TrackTimeline::new(meta.original_tempo_mbpm, meta.loop_beats, meta.original_samples)
            .map_err(|e| e.to_string())?;

        let pcm = Arc::new(PcmData {
            length_samples: left.len() as u64,
            channel_data: vec![left, right],
            sample_rate: ENGINE_SAMPLE_RATE,
            num_channels: 2,
        });

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
            pcm,
            meta.gain,
            meta.pan,
            stretcher,
        ));

        self.send(RtCommand::InstallTrack { slot: meta.slot, state });
        self.slots[meta.slot as usize] = SlotState::Loaded;
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
            transport: if raw.playing { TransportState::Playing } else { TransportState::Stopped },
            bar: raw.bar,
            beat_in_bar: raw.beat_in_bar,
            timeline_epoch: raw.timeline_epoch,
            underruns: raw.underruns,
            render_ns_max: raw.render_ns_max,
            callback_period_ns: raw.callback_period_ns,
            blocks_total: raw.blocks_total,
            blocks_over_half_period: raw.blocks_over_half_period,
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
        let mut out = std::mem::take(&mut self.pending_events);
        while let Ok(e) = self.events.pop() {
            out.push(match e {
                RtEvent::Started => Event::Started,
                RtEvent::Stopped => Event::Stopped,
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
        if (slot as usize) < MAX_TRACKS && self.slots[slot as usize] == SlotState::Loaded {
            Ok(())
        } else {
            Err(CommandResult::Rejected { reason: format!("slot {slot} has no loaded track") })
        }
    }

    /// Enqueue, blocking this (non-audio) thread if the queue is full
    /// (spec §3.3) — the audio thread drains at every block boundary.
    fn send(&mut self, cmd: RtCommand) {
        let mut cmd = cmd;
        loop {
            match self.commands.push(cmd) {
                Ok(()) => return,
                Err(rtrb::PushError::Full(returned)) => {
                    cmd = returned;
                    std::thread::yield_now();
                }
            }
        }
    }
}
