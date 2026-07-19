//! # audio_core — the audio-thread half of the engine (spec §3.2 / §3.3)
//!
//! Owned **exclusively** by the audio thread. Renders directly inside the OS
//! audio callback; no wall-clock timer exists anywhere in the audio path.
//!
//! ## RT rules (enforced, not aspirational)
//!
//! Inside [`AudioCore::render`]: no locks, no allocation or deallocation, no
//! `Arc` refcount drops to zero, no syscalls (timestamps excepted — reading a
//! timestamp is RT-safe), no logging, no panics (`debug_assert!` only). The
//! `assert_no_alloc` test in `tests/rt_safety.rs` and the clippy
//! `disallowed-types` config keep this true in CI.
//!
//! ## Ownership and queues
//!
//! - `rt_commands` (SPSC in): POD-ish commands applied at block boundaries.
//! - `rt_events` (SPSC out): edge events; dropped (Copy) if the queue is full —
//!   events are cosmetic, the snapshot is authoritative (spec §3.5).
//! - `reclaim` (SPSC out): the audio thread never frees memory; replaced
//!   resources are pushed here and dropped by the control plane. If `reclaim`
//!   is momentarily full, command draining pauses for the block (commands stay
//!   queued) rather than ever dropping a buffer on the audio thread.
//! - `snapshot` (seqlock): written once per block.
//!
//! ## Positions are computed, never advanced
//!
//! The render loop computes `u_i` at block edges only; per-block stretcher
//! feed counts are first differences that telescope to `u_i(F)` exactly
//! (spec §3.1). There is no per-track cursor anywhere in this file — the
//! `drift-injection` sabotage modes reintroduce the historic defect classes
//! on purpose so CI can prove T-INV detects them.

#![allow(clippy::too_many_arguments)]

use std::sync::Arc;
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
use std::time::Instant;

use maestro_api::QuantizePolicy;

use crate::engine_pcm::EnginePcm;
use crate::seqlock::SeqLock;
use crate::stretch::Stretcher;
use crate::timeline::{beats_display, Timeline, TrackTimeline, BEAT_UNITS_PER_BEAT};

pub const MAX_TRACKS: usize = 16;
/// Largest block size the engine preallocates for (spec §3.2). The OS chooses
/// the actual callback size; anything ≤ this renders allocation-free.
pub const MAX_BLOCK_FRAMES: usize = 4096;
/// Rate clamp upper bound (spec §4.8): max source samples consumed per output
/// frame, hence the feed-scratch sizing (+2 for floor/ceil block-edge slack).
const MAX_RATE: usize = 4;
const MAX_FEED_FRAMES: usize = MAX_BLOCK_FRAMES * MAX_RATE + 2;
/// Activation gain ramp (spec §3.2): ≤ 10 ms; masks the anchor's one-block
/// smear and avoids clicks when a track joins mid-waveform.
pub const ACTIVATION_RAMP_FRAMES: u32 = 480;
/// ADR-112 §5.4 resume hysteresis: a starved track re-enters only once the
/// frontier passes the failed requirement by this margin (1 s of source ≈
/// 250 ms of output at the 4× rate clamp). Without it, a frontier advancing
/// at exactly playback speed flutters audibly. Engine-owned, not
/// configurable in v1.
pub const STREAM_RESUME_MARGIN_FRAMES: u64 = 48_000;

/// Commands crossing control plane → audio core. Applied at block boundaries.
pub enum RtCommand {
    Play,
    /// Transport stop: freezes `F`, resets the timeline epoch (spec §3.1).
    Stop,
    /// Freeze `F` without an epoch reset (spec §3.1) — all tracks at once;
    /// positions are derived, so there is nothing per-track to freeze.
    Pause,
    /// Continue from `Pause`. Nothing is re-anchored.
    Resume,
    /// Applied at the next block boundary, live or stopped (M1). `bu` is
    /// continuous across the change by construction; the application frame is
    /// echoed via [`RtEvent::TempoApplied`].
    SetTempo { mbpm: u32 },
    InstallTrack { slot: u8, state: Box<TrackState> },
    RemoveTrack { slot: u8 },
    /// `auto_start` = TransportPolicy::Auto (ADR-092): also start a stopped
    /// transport. The engine decides; JS never makes timing decisions.
    Activate { slot: u8, quantize: QuantizePolicy, auto_start: bool },
    /// `auto_stop` = TransportPolicy::Auto: if this deactivation leaves zero
    /// active tracks, also stop the transport (epoch reset, echoed `Stopped`).
    Deactivate { slot: u8, auto_stop: bool },
    /// Per-track gain target (validated [0, 2] at the control plane). Applied
    /// with a one-block ramp; effective gain 0 makes the track skip its DSP
    /// entirely (spec §3.2).
    SetGain { slot: u8, gain: f32 },
    /// Per-track pan target (validated [-1, 1]); one-block ramp, audio-only.
    SetPan { slot: u8, pan: f32 },
    /// Mute/unmute (spec §3.2): mute ramps to silence then skips DSP; unmute
    /// re-anchors under the activation ramp. Echoed with the exact frame.
    SetMute { slot: u8, muted: bool },
    /// Solo on/off: while any loaded track is soloed, non-soloed tracks are
    /// inaudible (and skip DSP) — same re-anchor rule on restore.
    SetSolo { slot: u8, solo: bool },
    /// Per-track pitch in semitones (validated ±12 at the control plane,
    /// spec §4.8). Audio-only: promotes a unity track to the stretch path
    /// under the one-block crossfade; positions are unaffected.
    SetPitch { slot: u8, semitones: f32 },
    /// Variation switch (spec §3.2): Arc PCM swap + re-anchor + one-block
    /// equal-power crossfade at the block boundary. The control plane has
    /// already validated the shared identity (`O`, reconciliation case) and
    /// each file's own loop identity (spec §3.4, ADR-109). `map` carries the
    /// incoming variation's timing map so a length change installs atomically
    /// with the PCM — loop length and read length can never be two
    /// hand-synchronized values (the D9 guard, spec §2.3).
    SwapVariation { slot: u8, variation: u8, pcm: Arc<EnginePcm>, map: TrackTimeline },
    /// ADR-113: epoch re-base with a nonzero musical origin. Applied only
    /// while the transport is stopped (validated control-side; ignored here
    /// if racing a start — the host owns transport sequencing).
    SetTimelineOrigin { bu: u64 },
}

/// Edge events audio core → control plane. `Copy` so dropping on overflow
/// never deallocates on the audio thread.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RtEvent {
    Started,
    Stopped,
    Paused { at_frame: u64 },
    Resumed { at_frame: u64 },
    /// Tempo echo (spec §3.1): `mbpm` took effect at exactly `at_frame`.
    TempoApplied { mbpm: u32, at_frame: u64 },
    /// A track became audible at exactly `at_frame` (immediate, or the
    /// quantized activation frame `F_q`).
    Activated { slot: u8, at_frame: u64 },
    Deactivated { slot: u8, at_frame: u64 },
    /// Mute/solo echoes (spec §6.4 methodology): the exact block-boundary
    /// frame, so the device analyzer can predict mute silent windows.
    MuteApplied { slot: u8, muted: bool, at_frame: u64 },
    SoloApplied { slot: u8, solo: bool, at_frame: u64 },
    Removed { slot: u8 },
    VariationApplied { slot: u8, variation: u8, at_frame: u64 },
    /// Display/diagnostic only. `wrap_count = floor(bu/Λ)` re-bases (jumps)
    /// across a length-changing variation switch — Λ is per-variation
    /// (ADR-109), so the count is comparable only within one variation's
    /// tenure.
    LoopWrapped { slot: u8, wrap_count: u32 },
    /// ADR-112 §5.4: a streaming track's render was starved at the load
    /// frontier — it takes the inaudible path (silence) from `at_frame`.
    StreamStarved { slot: u8, at_frame: u64, needed: u64, frontier: u64 },
    /// The starved track re-entered (re-anchor + activation ramp).
    StreamResumed { slot: u8, at_frame: u64 },
    BudgetExceeded { render_ns: u64, period_ns: u64 },
}

/// Monotonic render-budget clock (spec §3.3 — the audio thread times itself).
/// `std::time::Instant` unconditionally panics on `wasm32-unknown-unknown`
/// (the target has no clock); the AudioWorklet quantum also has no OS
/// callback period to gate against, so budget telemetry reports zeros on the
/// web module (ADR-095) — the C4 web gates are the offline suites + parity +
/// E2E, never render-time telemetry. Every other target keeps real timing.
struct RenderClock(#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))] Instant);

impl RenderClock {
    #[inline]
    fn start() -> Self {
        #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
        {
            Self(Instant::now())
        }
        #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
        {
            Self()
        }
    }

    #[inline]
    fn elapsed_ns(&self) -> u64 {
        #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
        {
            self.0.elapsed().as_nanos() as u64
        }
        #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
        {
            0
        }
    }
}

/// Push an edge event, counting the drop if the queue is full (spec §6.7 —
/// overflow is surfaced via the snapshot's `events_dropped`, never silent).
/// Free function so it composes with split borrows of `AudioCore` fields.
#[inline]
fn push_event(events: &mut rtrb::Producer<RtEvent>, dropped: &mut u32, e: RtEvent) {
    if events.push(e).is_err() {
        *dropped = dropped.saturating_add(1);
    }
}

/// Per-block level meter values (display only — floats never feed timing).
#[derive(Debug, Clone, Copy, Default)]
pub struct TrackMeter {
    pub rms_l: f32,
    pub rms_r: f32,
    pub peak_l: f32,
    pub peak_r: f32,
}

/// Equal-power pan + gain mix coefficients for a (gain, pan) pair.
#[inline]
fn mix_coefficients(gain: f32, pan: f32) -> (f32, f32) {
    let theta = (pan.clamp(-1.0, 1.0) + 1.0) * std::f32::consts::FRAC_PI_4;
    (theta.cos() * gain, theta.sin() * gain)
}

/// Resources the audio thread hands back for deallocation off-thread.
pub enum Reclaimable {
    Track(Box<TrackState>),
    /// A variation's outgoing PCM after its crossfade block (spec §3.2).
    Pcm(Arc<EnginePcm>),
}

/// POD per-track snapshot slice (display only).
#[derive(Debug, Clone, Copy, Default)]
pub struct RawTrackSnap {
    pub occupied: bool,
    pub active: bool,
    pub variation: u8,
    pub gain: f32,
    pub pan: f32,
    pub muted: bool,
    pub solo: bool,
    /// The installed variation's loop length in beats — per-variation since
    /// ADR-109 (flips together with `variation` at a switch). Display only.
    pub loop_beats: u32,
    pub loop_count: u32,
    pub loop_progress: f32,
    pub rms_l: f32,
    pub rms_r: f32,
    pub peak_l: f32,
    pub peak_r: f32,
    /// Load frontier in samples (ADR-112); == pcm_len once complete.
    pub loaded_samples: u64,
    /// True from begin until finalize/abort.
    pub streaming: bool,
    /// True while the render is starved at the frontier (silent).
    pub starved: bool,
}

/// POD snapshot written by the audio thread once per block (spec §3.5).
#[derive(Debug, Clone, Copy)]
pub struct RawSnapshot {
    pub frame: u64,
    pub tempo_mbpm: u32,
    pub playing: bool,
    /// `F` frozen without epoch reset (spec §3.1). Only meaningful while
    /// `playing` — pause does not stop the transport.
    pub paused: bool,
    pub timeline_epoch: u32,
    /// Display-only beat position derived from bu(F).
    pub bar: u32,
    pub beat_in_bar: f32,
    /// Render-budget telemetry (spec §3.3): the audio thread times itself.
    pub underruns: u32,
    pub render_ns_max: u64,
    pub callback_period_ns: u64,
    pub blocks_total: u64,
    pub blocks_over_half_period: u64,
    /// Edge events dropped on RT event-queue overflow (spec §6.7 — surfaced,
    /// never silent).
    pub events_dropped: u32,
    pub tracks: [RawTrackSnap; MAX_TRACKS],
}

impl RawSnapshot {
    pub fn empty(tempo_mbpm: u32) -> Self {
        Self {
            frame: 0,
            tempo_mbpm,
            playing: false,
            paused: false,
            timeline_epoch: 0,
            bar: 1,
            beat_in_bar: 0.0,
            underruns: 0,
            render_ns_max: 0,
            callback_period_ns: 0,
            blocks_total: 0,
            blocks_over_half_period: 0,
            events_dropped: 0,
            tracks: [RawTrackSnap::default(); MAX_TRACKS],
        }
    }
}

/// Everything the audio thread needs to render one track. Built entirely by
/// the control plane (all allocation happens there), then moved across the
/// SPSC command queue. Holds **no positional state** — see module docs.
pub struct TrackState {
    map: TrackTimeline,
    pcm: Arc<EnginePcm>,
    active: bool,
    /// Target gain/pan (control-plane validated). The render loop ramps the
    /// applied mix coefficients toward these over one block — audio-only.
    gain: f32,
    pan: f32,
    muted: bool,
    solo: bool,
    /// Mix coefficients actually applied at the end of the last rendered
    /// block (post gain/pan/mute/solo), lerped per block for click-free
    /// parameter changes. `None` after (re)entry — the activation ramp masks
    /// the initial step.
    applied_mix: Option<(f32, f32)>,
    /// Whether the track rendered (was audible) last block — the spec §3.2
    /// DSP-skip transition state: inaudible → audible re-anchors under the
    /// activation ramp, exactly like a restart.
    was_audible: bool,
    /// Per-track level metering over the last rendered block (post gain/pan;
    /// the track's contribution to the mix). Display only.
    meter: TrackMeter,
    /// Currently rendering variation (display/echo only).
    variation: u8,
    /// User pitch shift in semitones (±12, validated control-side). Nonzero
    /// pitch forces the stretch path even at unity tempo (spec §3.2).
    pitch_semitones: f32,
    /// Last transpose actually set on the stretcher (effective, i.e. after
    /// the combined-ratio clamp), so the render loop only re-sets on change.
    applied_pitch: Option<f32>,
    /// Quantized activation target (spec §3.1): the track becomes audible at
    /// the first frame where `bu(F) ≥ pending_bu`; the render loop splits the
    /// block there so activation is sample-exact within a block.
    pending_bu: Option<u128>,
    /// Path rendered last block (`Some(true)` = bypass): a flip triggers the
    /// one-block equal-power crossfade (spec §3.2). `None` after (re)entry —
    /// entries are masked by the activation ramp, not a crossfade.
    prev_unity: Option<bool>,
    /// Outgoing (PCM, map) during a variation switch: the fade-out source for
    /// the crossfade block, then the PCM is handed to the reclaim queue (never
    /// dropped here). The outgoing map rides along because variations may have
    /// different loop lengths (ADR-109) — the fade-out read must wrap at the
    /// OUTGOING variation's Λ and bound by ITS pcm_len. `TrackTimeline` is
    /// `Copy` with no heap, so dropping it here is RT-safe.
    xfade_old: Option<(Arc<EnginePcm>, TrackTimeline)>,
    /// A retired PCM waiting for reclaim-queue space (overflow fallback).
    retired_pcm: Option<Arc<EnginePcm>>,
    /// Always provided (M1 tempo changes can move any track off unity).
    /// Preallocated by the control plane; `anchor` and `process` never allocate.
    stretcher: Box<Stretcher>,
    /// Cached completeness of `pcm` (ADR-112): refreshed with an Acquire
    /// load only while false, so complete/bulk tracks pay one branch per
    /// block and never touch the atomics again.
    pcm_complete: bool,
    /// True while the render is starved at the load frontier (ADR-112 §5.4).
    /// Edge in/out fires StreamStarved/StreamResumed.
    starved: bool,
    needs_anchor: bool,
    /// Set with `needs_anchor` by a tempo change that re-anchors a warm
    /// mid-stream stretch track (ADR-094): the OLD tempo, letting the next
    /// render fade out one final block of the old feed schedule from the
    /// still-warm stretcher (waveform-continuous) against the fresh anchor.
    /// Consumed (taken) on the very next render of the track.
    reanchor_fade_tempo: Option<u32>,
    ramp_remaining: u32,
    last_wrap_count: u64,
    /// Interleaved scratch, capacity `MAX_FEED_FRAMES * 2`.
    feed_scratch: Vec<f32>,
    /// Interleaved scratch, capacity `MAX_BLOCK_FRAMES * 2`.
    out_scratch: Vec<f32>,
    /// Interleaved crossfade fade-out scratch, capacity `MAX_BLOCK_FRAMES * 2`.
    xfade_scratch: Vec<f32>,
    /// Interleaved anchor window, capacity `(input_latency + output_latency) * 2`.
    anchor_scratch: Vec<f32>,
    /// Sabotage state for the banned accumulating-cursor class (spec §6.2
    /// negative control). f32 rather than the dossier's f64: a constant-rate
    /// f64 Bresenham cursor is the corrected-dossier case (§2.3#1) whose error
    /// is below 1 sample within an offline run, which would make the negative
    /// control nondeterministic. f32 exhibits the same *class* of defect at a
    /// deterministic, detectable magnitude (ulp ≈ 4 samples at 5×10⁷).
    #[cfg(feature = "drift-injection")]
    float_cursor: Option<f32>,
}

impl TrackState {
    /// Control-plane constructor: allocates every buffer the render path will
    /// ever touch for this track. `pcm` must be stereo with both channels of
    /// length `map.pcm_len()` (validated in the control plane).
    pub fn new(
        map: TrackTimeline,
        pcm: Arc<EnginePcm>,
        gain: f32,
        pan: f32,
        variation: u8,
        mut stretcher: Box<Stretcher>,
    ) -> Self {
        debug_assert_eq!(pcm.len(), map.pcm_len());
        let anchor_frames = stretcher.input_latency() + stretcher.output_latency();
        stretcher.preallocate(MAX_FEED_FRAMES.max(anchor_frames), MAX_BLOCK_FRAMES);
        let pcm_complete = pcm.is_complete();
        Self {
            map,
            pcm,
            pcm_complete,
            starved: false,
            active: false,
            gain,
            pan,
            muted: false,
            solo: false,
            applied_mix: None,
            was_audible: false,
            meter: TrackMeter::default(),
            variation,
            pitch_semitones: 0.0,
            applied_pitch: None,
            pending_bu: None,
            prev_unity: None,
            xfade_old: None,
            retired_pcm: None,
            stretcher,
            needs_anchor: true,
            reanchor_fade_tempo: None,
            ramp_remaining: 0,
            last_wrap_count: 0,
            feed_scratch: vec![0.0; MAX_FEED_FRAMES * 2],
            out_scratch: vec![0.0; MAX_BLOCK_FRAMES * 2],
            xfade_scratch: vec![0.0; MAX_BLOCK_FRAMES * 2],
            anchor_scratch: vec![0.0; anchor_frames * 2],
            #[cfg(feature = "drift-injection")]
            float_cursor: None,
        }
    }

    /// The dictated unwrapped source sample at musical position `bu`
    /// (spec §3.1 `u_i`), including the position-level sabotage hooks.
    #[inline]
    fn dictated_u(&self, bu: u128) -> u64 {
        let u = self.map.u(bu);
        #[cfg(feature = "drift-injection")]
        {
            use crate::sabotage::{mode, Sabotage};
            if mode() == Sabotage::PcmLengthLoop {
                // Defect D2: loop at integer PCM length. Each wrap contributes
                // pcm_len samples instead of the exact rational λ_w, so musical
                // phase drifts by (Λ/O − pcm_len) per wrap — the reproduced
                // 24-samples-in-10-minutes failure (spec Appendix A.1).
                let w = self.map.wrap_of_sample(u);
                let idx = (u - self.map.wrap_start_sample(w)).min(self.map.pcm_len() - 1);
                return w * self.map.pcm_len() + idx;
            }
        }
        u
    }

    /// Read `frames` source samples starting at unwrapped sample `n` into
    /// `dest` (interleaved stereo), following the beat-space Bresenham wrap.
    fn read_source(&self, n: u64, frames: usize, dest: &mut [f32]) {
        read_pcm(&self.map, &self.pcm, n, frames, dest);
    }

    /// One-time anchor (spec §3.2): prime the stretcher with a seek window of
    /// `input_latency + output_latency` source frames ending at
    /// `n0 + input_latency`, then stream. `n0` is the compensated feed origin
    /// `u(bu(F₀) + E·output_latency)` (ADR-094) — leading the playhead by the
    /// vocoder's constant output latency so stretched output lands on the
    /// derived grid, not 60 ms behind it. Runs on the audio thread but
    /// touches only preallocated scratch.
    fn anchor(&mut self, n0: u64, rate: f64) {
        self.stretcher.reset();
        let in_lat = self.stretcher.input_latency() as u64;
        let out_lat = self.stretcher.output_latency() as u64;
        let win = (in_lat + out_lat) as usize;
        if win == 0 {
            return;
        }
        let end = n0 + in_lat;
        let start = end.saturating_sub(win as u64);
        let lead_zeros = (win as u64 - (end - start)) as usize;
        let window_samples = win * 2;
        self.anchor_scratch[..lead_zeros * 2].fill(0.0);
        if end > start {
            // Split borrow: read_source needs &self, scratch is &mut. Take the
            // buffer out and put it back — no allocation either way.
            let mut scratch = std::mem::take(&mut self.anchor_scratch);
            self.read_source(start, (end - start) as usize, &mut scratch[lead_zeros * 2..window_samples]);
            self.anchor_scratch = scratch;
        }
        let scratch = std::mem::take(&mut self.anchor_scratch);
        let _ = self.stretcher.seek_interleaved(&scratch[..window_samples], rate);
        self.anchor_scratch = scratch;
    }
}

/// The audio-thread engine. Create with [`create_engine`](crate::control_plane::create_engine);
/// hand the `Box<AudioCore>` to the platform render callback.
pub struct AudioCore {
    frame: u64,
    playing: bool,
    /// `F` frozen, epoch intact (spec §3.1). Only meaningful while `playing`.
    paused: bool,
    timeline: Timeline,
    beats_per_bar: u32,
    tracks: [Option<Box<TrackState>>; MAX_TRACKS],
    commands: rtrb::Consumer<RtCommand>,
    events: rtrb::Producer<RtEvent>,
    reclaim: rtrb::Producer<Reclaimable>,
    snapshot: Arc<SeqLock<RawSnapshot>>,
    // Budget telemetry.
    underruns: u32,
    render_ns_max: u64,
    blocks_total: u64,
    blocks_over_half_period: u64,
    /// Edge events dropped on RT event-queue overflow (spec §6.7 — surfaced
    /// via the snapshot, never silent).
    events_dropped: u32,
    /// Plan B (render pump): ring-starvation underruns counted by the audio
    /// callback, merged into the snapshot's `underruns` (the C0 gate). Reading
    /// an atomic in `write_snapshot` is RT-safe.
    external_underruns: Option<std::sync::Arc<std::sync::atomic::AtomicU32>>,
    /// Deterministic LCG driving the timer_clock sabotage jitter.
    #[cfg(feature = "drift-injection")]
    lcg: u64,
}

impl AudioCore {
    pub(crate) fn new(
        timeline: Timeline,
        beats_per_bar: u32,
        commands: rtrb::Consumer<RtCommand>,
        events: rtrb::Producer<RtEvent>,
        reclaim: rtrb::Producer<Reclaimable>,
        snapshot: Arc<SeqLock<RawSnapshot>>,
    ) -> Self {
        Self {
            frame: 0,
            playing: false,
            paused: false,
            timeline,
            beats_per_bar: beats_per_bar.max(1),
            tracks: std::array::from_fn(|_| None),
            commands,
            events,
            reclaim,
            snapshot,
            underruns: 0,
            render_ns_max: 0,
            blocks_total: 0,
            blocks_over_half_period: 0,
            events_dropped: 0,
            external_underruns: None,
            #[cfg(feature = "drift-injection")]
            lcg: 0x9E37_79B9_7F4A_7C15,
        }
    }

    /// Master clock value `F` — frames actually rendered while playing.
    pub fn frame(&self) -> u64 {
        self.frame
    }

    /// Wire the ring-starvation counter (Plan B render pump) into the
    /// snapshot's `underruns`. Called once before the core moves to the
    /// producer thread.
    pub fn set_external_underruns(&mut self, counter: std::sync::Arc<std::sync::atomic::AtomicU32>) {
        self.external_underruns = Some(counter);
    }

    /// The dictated unwrapped source position of a track **as the renderer
    /// uses it** (sabotage modes included) — T-INV compares this against the
    /// exact rational reference (spec §6.2).
    pub fn dictated_unwrapped_position(&self, slot: usize) -> Option<u64> {
        let ts = self.tracks.get(slot)?.as_ref()?;
        #[cfg(feature = "drift-injection")]
        if crate::sabotage::mode() == crate::sabotage::Sabotage::FloatCursor {
            if let Some(c) = ts.float_cursor {
                return Some(c as u64);
            }
        }
        Some(ts.dictated_u(self.timeline.bu(self.frame)))
    }

    /// Render one block into non-interleaved stereo buffers. This is the body
    /// of the OS audio callback (spec §3.3) — see module docs for RT rules.
    pub fn render(&mut self, left: &mut [f32], right: &mut [f32]) {
        debug_assert_eq!(left.len(), right.len());
        let b = left.len().min(right.len()).min(MAX_BLOCK_FRAMES);
        let t0 = RenderClock::start();
        left.fill(0.0);
        right.fill(0.0);
        if b == 0 {
            return;
        }

        self.apply_commands();

        // Hand back any PCM retired by a variation switch — even while
        // stopped/paused (the reclaim queue may have been full last block).
        for ts in self.tracks.iter_mut().flatten() {
            if let Some(old) = ts.retired_pcm.take() {
                if let Err(rtrb::PushError::Full(Reclaimable::Pcm(back))) =
                    self.reclaim.push(Reclaimable::Pcm(old))
                {
                    ts.retired_pcm = Some(back);
                }
            }
        }

        if self.playing && !self.paused {
            let f0 = self.frame;
            let bu0 = self.timeline.bu(f0);
            let bu1 = self.timeline.bu(f0 + b as u64);
            let tempo = self.timeline.tempo_mbpm_at(f0);
            // Solo semantics (spec §3.2): while any loaded track is soloed,
            // non-soloed tracks are inaudible (and skip their DSP entirely).
            let any_solo = self.tracks.iter().flatten().any(|t| t.solo);
            for slot in 0..MAX_TRACKS {
                if let Some(ts) = self.tracks[slot].as_deref_mut() {
                    // Quantized activation (spec §3.1): split the block at the
                    // first frame F_q where bu(F_q) ≥ pending_bu — sample-exact
                    // within the block, by exact integer inversion of bu.
                    let mut offset = 0usize;
                    if let Some(bu_q) = ts.pending_bu {
                        if bu_q <= bu1 {
                            let d = if bu_q <= bu0 {
                                0
                            } else {
                                // Smallest d with bu0 + d·E ≥ bu_q. A block
                                // spans exactly one tempo segment (changes
                                // apply at block boundaries), so E is constant.
                                (bu_q - bu0).div_ceil(tempo as u128) as usize
                            };
                            if d < b {
                                ts.pending_bu = None;
                                ts.active = true;
                                ts.needs_anchor = true;
                                ts.prev_unity = None;
                                ts.ramp_remaining = ACTIVATION_RAMP_FRAMES;
                                let f_q = f0 + d as u64;
                                ts.last_wrap_count = ts.map.wrap_count(self.timeline.bu(f_q));
                                push_event(
                                    &mut self.events,
                                    &mut self.events_dropped,
                                    RtEvent::Activated { slot: slot as u8, at_frame: f_q },
                                );
                                offset = d;
                            }
                            // d == b: F_q is the next block's first frame —
                            // activation happens there (bu_q == its bu0).
                        }
                    }
                    // Audibility (spec §3.2): a muted / soloed-out / gain-0
                    // track skips its read/stretch DSP entirely — there is no
                    // cursor to keep warm. `had_signal` lets a track that just
                    // became inaudible render ONE more block, ramping its mix
                    // coefficients to zero (click-free), before skipping.
                    let soloed_out = any_solo && !ts.solo;
                    let audible_target =
                        ts.active && !ts.muted && !soloed_out && ts.gain > 0.0;
                    let target_gain = if audible_target { ts.gain } else { 0.0 };
                    let had_signal = ts.applied_mix.is_some_and(|(l, r)| l != 0.0 || r != 0.0);
                    if !ts.active {
                        ts.was_audible = false;
                        ts.applied_mix = None;
                        ts.meter = TrackMeter::default();
                        continue;
                    }
                    if !audible_target && !had_signal {
                        // Fully silent: skip DSP. Positions stay derived from
                        // F; on becoming audible the track re-anchors under
                        // the activation ramp, exactly like a restart —
                        // silence costs nothing and cannot desync (spec §3.2).
                        ts.was_audible = false;
                        ts.meter = TrackMeter::default();
                        ts.last_wrap_count = ts.map.wrap_count(bu1);
                        continue;
                    }
                    if audible_target && !ts.was_audible {
                        // Inaudible → audible: re-anchor under the ramp.
                        ts.needs_anchor = true;
                        ts.prev_unity = None;
                        ts.ramp_remaining = ACTIVATION_RAMP_FRAMES;
                        ts.applied_mix = None;
                        ts.last_wrap_count = ts.map.wrap_count(bu0);
                    }
                    let bu_start =
                        if offset == 0 { bu0 } else { self.timeline.bu(f0 + offset as u64) };
                    // §5.4 starvation gate (ADR-112): an incomplete track
                    // whose reads this block would touch at/above the load
                    // frontier takes exactly the inaudible path instead —
                    // DSP skipped, meters zeroed, wrap bookkeeping re-based,
                    // positions untouched (the clock does not care). Placed
                    // AFTER the re-entry mutations above so the predicate
                    // sees exactly the state render_track will see.
                    if !ts.pcm_complete {
                        if ts.pcm.is_complete() {
                            ts.pcm_complete = true;
                        } else if let Some(touched) =
                            required_max_touched(ts, bu_start, bu1, tempo, b - offset)
                        {
                            let needed = touched + 1;
                            let frontier = ts.pcm.frontier();
                            // Hysteresis (§5.4): re-entry needs the resume
                            // margin past the requirement that failed.
                            let resume_at = needed
                                .saturating_add(STREAM_RESUME_MARGIN_FRAMES)
                                .min(ts.map.pcm_len());
                            let can_render =
                                frontier >= if ts.starved { resume_at } else { needed };
                            if !can_render {
                                if !ts.starved {
                                    ts.starved = true;
                                    push_event(
                                        &mut self.events,
                                        &mut self.events_dropped,
                                        RtEvent::StreamStarved {
                                            slot: slot as u8,
                                            at_frame: f0 + offset as u64,
                                            needed,
                                            frontier,
                                        },
                                    );
                                }
                                ts.was_audible = false;
                                ts.applied_mix = None;
                                ts.meter = TrackMeter::default();
                                ts.last_wrap_count = ts.map.wrap_count(bu1);
                                continue;
                            }
                        }
                        // Reaching here means this block renders: the demand
                        // cleared the resume margin, the stream just completed
                        // (everything readable — no margin applies), or the
                        // block touches nothing. A set flag is therefore the
                        // resume edge — clearing it ONLY on the margin path
                        // left a track that finalized while starved reporting
                        // `starved: true` forever (the completion branch above
                        // stops this whole gate from running again).
                        if ts.starved {
                            ts.starved = false;
                            push_event(
                                &mut self.events,
                                &mut self.events_dropped,
                                RtEvent::StreamResumed {
                                    slot: slot as u8,
                                    at_frame: f0 + offset as u64,
                                },
                            );
                        }
                    }
                    ts.was_audible = audible_target;
                    render_track(
                        ts,
                        bu_start,
                        bu1,
                        tempo,
                        target_gain,
                        b - offset,
                        &mut left[offset..b],
                        &mut right[offset..b],
                    );
                    // Loop-wrap edge events (display/diagnostics only).
                    let w = ts.map.wrap_count(bu1);
                    if w > ts.last_wrap_count {
                        ts.last_wrap_count = w;
                        push_event(
                            &mut self.events,
                            &mut self.events_dropped,
                            RtEvent::LoopWrapped { slot: slot as u8, wrap_count: w as u32 },
                        );
                    }
                }
            }
            self.advance_clock(b);
        }

        // Budget telemetry (spec §3.3): the audio thread times itself; reading
        // a timestamp is RT-safe.
        let render_ns = t0.elapsed_ns();
        let period_ns = b as u64 * 1_000_000_000 / crate::timeline::ENGINE_SAMPLE_RATE as u64;
        self.blocks_total += 1;
        self.render_ns_max = self.render_ns_max.max(render_ns);
        if render_ns * 2 > period_ns {
            self.blocks_over_half_period += 1;
        }
        if render_ns > period_ns {
            self.underruns += 1;
            push_event(
                &mut self.events,
                &mut self.events_dropped,
                RtEvent::BudgetExceeded { render_ns, period_ns },
            );
        }

        self.write_snapshot(period_ns);
    }

    #[inline]
    fn advance_clock(&mut self, b: usize) {
        #[cfg(feature = "drift-injection")]
        if crate::sabotage::mode() == crate::sabotage::Sabotage::TimerClock {
            // Defect D8: the clock claims what a jittery wall timer says
            // instead of what was actually rendered. ±2 frames per block.
            self.lcg = self.lcg.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let jitter = (self.lcg >> 33) % 5; // 0..=4
            self.frame = self.frame + b as u64 + jitter - 2;
            return;
        }
        self.frame += b as u64;
    }

    /// Drain control-plane commands at the block boundary. Stops early if the
    /// reclaim queue cannot take another item — the audio thread never drops
    /// a heap resource (spec §3.3); commands simply stay queued for the next
    /// block.
    fn apply_commands(&mut self) {
        while self.reclaim.slots() > 0 {
            let Ok(cmd) = self.commands.pop() else { return };
            match cmd {
                RtCommand::Play => self.do_play(),
                RtCommand::Stop => self.do_stop(),
                RtCommand::Pause => {
                    // F freezes for all tracks at once; no epoch reset, no
                    // per-track state to touch (positions are derived).
                    if self.playing && !self.paused {
                        self.paused = true;
                        push_event(
                            &mut self.events,
                            &mut self.events_dropped,
                            RtEvent::Paused { at_frame: self.frame },
                        );
                    }
                }
                RtCommand::Resume => {
                    if self.playing && self.paused {
                        self.paused = false;
                        // Audio-only: a short ramp masks the resume edge (the
                        // waveform was cut at the pause boundary). Positions
                        // and stretcher streams continue exactly where they
                        // froze — nothing is re-anchored.
                        for ts in self.tracks.iter_mut().flatten() {
                            if ts.active {
                                ts.ramp_remaining = ts.ramp_remaining.max(ACTIVATION_RAMP_FRAMES);
                            }
                        }
                        push_event(
                            &mut self.events,
                            &mut self.events_dropped,
                            RtEvent::Resumed { at_frame: self.frame },
                        );
                    }
                }
                RtCommand::SetTempo { mbpm } => {
                    let prev = self.timeline.tempo_mbpm_at(self.frame);
                    if self.timeline.set_tempo(self.frame, mbpm).is_ok() {
                        if mbpm != prev {
                            // ADR-094: the stretch path's compensated feed
                            // schedule is a function of the current tempo
                            // (bu + E·out_lat), so a tempo change re-derives
                            // it discontinuously — feeding across that seam
                            // steps audibly mid-vocoder-window (measured
                            // 3.5× the T-SEAM bound on a pure sine). Tracks
                            // rendering the stretch path at the NEW tempo
                            // re-anchor instead; render_track masks the
                            // mid-stream re-anchor with the same one-block
                            // 1:1 crossfade as the unity↔stretch
                            // transitions. Tracks entering unity keep their
                            // warm stretcher for the stretch→unity fade.
                            for ts in self.tracks.iter_mut().flatten() {
                                if ts.map.original_tempo_mbpm() != mbpm
                                    || ts.pitch_semitones != 0.0
                                {
                                    ts.needs_anchor = true;
                                    if ts.prev_unity == Some(false) {
                                        // Warm mid-stream stretch: the next
                                        // render fades out one final block
                                        // of the OLD feed schedule.
                                        // get_or_insert: with two changes
                                        // at one boundary the stream is
                                        // still on the OLDEST schedule.
                                        ts.reanchor_fade_tempo.get_or_insert(prev);
                                    }
                                }
                            }
                        }
                        // Tempo echo (spec §3.1): the exact application frame,
                        // for the device analyzer's predicted onset grid.
                        push_event(
                            &mut self.events,
                            &mut self.events_dropped,
                            RtEvent::TempoApplied { mbpm, at_frame: self.frame },
                        );
                    }
                }
                RtCommand::InstallTrack { slot, state } => {
                    let slot = slot as usize;
                    if slot < MAX_TRACKS {
                        if let Some(old) = self.tracks[slot].replace(state) {
                            let _ = self.reclaim.push(Reclaimable::Track(old));
                        }
                    } else {
                        let _ = self.reclaim.push(Reclaimable::Track(state));
                    }
                }
                RtCommand::RemoveTrack { slot } => {
                    if let Some(old) = self.tracks.get_mut(slot as usize).and_then(Option::take) {
                        let _ = self.reclaim.push(Reclaimable::Track(old));
                        push_event(
                            &mut self.events,
                            &mut self.events_dropped,
                            RtEvent::Removed { slot },
                        );
                    }
                }
                RtCommand::Activate { slot, quantize, auto_start } => {
                    let bu = self.timeline.bu(self.frame);
                    let frame = self.frame;
                    let bar_units = self.beats_per_bar as u128 * BEAT_UNITS_PER_BEAT;
                    if let Some(ts) = self.tracks.get_mut(slot as usize).and_then(Option::as_deref_mut) {
                        if !ts.active {
                            match quantize {
                                QuantizePolicy::Immediate => {
                                    ts.active = true;
                                    ts.needs_anchor = true;
                                    ts.prev_unity = None;
                                    ts.ramp_remaining = ACTIVATION_RAMP_FRAMES;
                                    ts.last_wrap_count = ts.map.wrap_count(bu);
                                    ts.pending_bu = None;
                                    push_event(
                                        &mut self.events,
                                        &mut self.events_dropped,
                                        RtEvent::Activated { slot, at_frame: frame },
                                    );
                                }
                                QuantizePolicy::NextBar => {
                                    // Target the next bar boundary in beat-units
                                    // (exactly-on-boundary activates this block:
                                    // the render loop sees bu_q ≤ bu0). The
                                    // frame F_q is inverted from bu by the
                                    // render loop, against the segment in
                                    // effect when the boundary is reached —
                                    // robust to tempo changes in between.
                                    ts.pending_bu = Some(bu.div_ceil(bar_units) * bar_units);
                                }
                            }
                        }
                    }
                    // Safety preset Auto (ADR-092): activating with a stopped
                    // transport starts it — engine-decided, echoed `Started`.
                    if auto_start && !self.playing {
                        self.do_play();
                    }
                }
                RtCommand::Deactivate { slot, auto_stop } => {
                    let frame = self.frame;
                    if let Some(ts) = self.tracks.get_mut(slot as usize).and_then(Option::as_deref_mut) {
                        if ts.active {
                            push_event(
                                &mut self.events,
                                &mut self.events_dropped,
                                RtEvent::Deactivated { slot, at_frame: frame },
                            );
                        }
                        ts.active = false;
                        ts.pending_bu = None;
                    }
                    // Safety preset Auto (ADR-092): deactivating the LAST
                    // active track stops the transport (epoch reset). Without
                    // Auto the §3.1 default stands: the timeline keeps running.
                    if auto_stop
                        && self.playing
                        && !self.tracks.iter().flatten().any(|t| t.active || t.pending_bu.is_some())
                    {
                        self.do_stop();
                    }
                }
                RtCommand::SetGain { slot, gain } => {
                    if let Some(ts) = self.tracks.get_mut(slot as usize).and_then(Option::as_deref_mut) {
                        // Audio-only target; the render loop ramps the applied
                        // mix over one block. Gain 0 ⇒ DSP skip (spec §3.2).
                        ts.gain = gain;
                    }
                }
                RtCommand::SetPan { slot, pan } => {
                    if let Some(ts) = self.tracks.get_mut(slot as usize).and_then(Option::as_deref_mut) {
                        ts.pan = pan;
                    }
                }
                RtCommand::SetMute { slot, muted } => {
                    let frame = self.frame;
                    if let Some(ts) = self.tracks.get_mut(slot as usize).and_then(Option::as_deref_mut) {
                        if ts.muted != muted {
                            ts.muted = muted;
                            push_event(
                                &mut self.events,
                                &mut self.events_dropped,
                                RtEvent::MuteApplied { slot, muted, at_frame: frame },
                            );
                        }
                    }
                }
                RtCommand::SetSolo { slot, solo } => {
                    let frame = self.frame;
                    if let Some(ts) = self.tracks.get_mut(slot as usize).and_then(Option::as_deref_mut) {
                        if ts.solo != solo {
                            ts.solo = solo;
                            push_event(
                                &mut self.events,
                                &mut self.events_dropped,
                                RtEvent::SoloApplied { slot, solo, at_frame: frame },
                            );
                        }
                    }
                }
                RtCommand::SetPitch { slot, semitones } => {
                    if let Some(ts) = self.tracks.get_mut(slot as usize).and_then(Option::as_deref_mut) {
                        // Audio-only: render_track applies it to the stretcher
                        // (with the §4.8 combined-ratio clamp) and handles the
                        // unity→stretch promotion crossfade.
                        ts.pitch_semitones = semitones;
                    }
                }
                RtCommand::SetTimelineOrigin { bu } => {
                    // ADR-113: exactly the Stop epoch reset with a nonzero
                    // start_bu. Positions derive from `bu` at the next Play;
                    // per-track bookkeeping re-bases like do_stop's.
                    // A racing Play that won the queue makes applying unsafe
                    // (it would re-base a running clock) — dropping is the
                    // only sound RT choice, but it means the host's earlier
                    // Ok was optimistic. The host contract (one origin, then
                    // one scheduled play, one in flight) makes this
                    // unreachable; loud in debug if that ever breaks.
                    debug_assert!(
                        !self.playing,
                        "SetTimelineOrigin raced a Play — host must sequence transport (ADR-113)"
                    );
                    if !self.playing {
                        let tempo = self.timeline.tempo_mbpm_at(self.frame);
                        let _ = self.timeline.reset_with_origin(self.frame, tempo, bu as u128);
                        let bu0 = self.timeline.bu(self.frame);
                        for ts in self.tracks.iter_mut().flatten() {
                            ts.needs_anchor = true;
                            ts.prev_unity = None;
                            ts.last_wrap_count = ts.map.wrap_count(bu0);
                            // The epoch reset invalidates any quantized
                            // activation target (it was a bu of the old epoch).
                            ts.pending_bu = None;
                            #[cfg(feature = "drift-injection")]
                            {
                                ts.float_cursor = None;
                            }
                        }
                    }
                }
                RtCommand::SwapVariation { slot, variation, pcm, map } => {
                    let frame = self.frame;
                    let bu = self.timeline.bu(frame);
                    let rendering = self.playing && !self.paused;
                    if let Some(ts) = self.tracks.get_mut(slot as usize).and_then(Option::as_deref_mut) {
                        let old_pcm = std::mem::replace(&mut ts.pcm, pcm);
                        let old_map = std::mem::replace(&mut ts.map, map);
                        // Swaps only install complete PCM (streaming slots
                        // reject SwitchVariation, ADR-112 §5.5) — refresh the
                        // cached completeness for the incoming buffer.
                        ts.pcm_complete = ts.pcm.is_complete();
                        ts.starved = false;
                        ts.variation = variation;
                        ts.needs_anchor = true;
                        // Λ may have changed (per-variation lengths, ADR-109):
                        // re-base the wrap bookkeeping on the incoming map, or
                        // a long→short switch burst-fires stale LoopWrapped
                        // events and short→long silences them. Display/event
                        // only — positions stay derived.
                        ts.last_wrap_count = ts.map.wrap_count(bu);
                        if ts.active && rendering && ts.was_audible {
                            // Crossfade against the outgoing (PCM, map) this
                            // block (spec §3.2); the PCM is reclaimed after
                            // the fade.
                            if let Some((prev, _)) = ts.xfade_old.replace((old_pcm, old_map)) {
                                // Double swap inside one block: the earlier
                                // outgoing PCM never got its fade — reclaim it
                                // (the loop guard guarantees a free slot).
                                let _ = self.reclaim.push(Reclaimable::Pcm(prev));
                            }
                        } else {
                            // Not audible: no crossfade — hand it back now.
                            let _ = self.reclaim.push(Reclaimable::Pcm(old_pcm));
                        }
                        push_event(
                            &mut self.events,
                            &mut self.events_dropped,
                            RtEvent::VariationApplied { slot, variation, at_frame: frame },
                        );
                    } else {
                        // No track at that slot: the new PCM itself goes back.
                        let _ = self.reclaim.push(Reclaimable::Pcm(pcm));
                    }
                }
            }
        }
    }

    /// The Play flow (transport start). Shared by `RtCommand::Play` and the
    /// `Activate` auto-start policy.
    fn do_play(&mut self) {
        if !self.playing {
            self.playing = true;
            self.paused = false;
            for ts in self.tracks.iter_mut().flatten() {
                ts.needs_anchor = true;
                ts.prev_unity = None;
                if ts.active {
                    ts.ramp_remaining = ACTIVATION_RAMP_FRAMES;
                }
            }
            push_event(&mut self.events, &mut self.events_dropped, RtEvent::Started);
        }
    }

    /// The Stop flow: freeze `F` and reset the timeline epoch — one of the
    /// two epoch-reset flows (spec §3.1). Shared by `RtCommand::Stop` and the
    /// `Deactivate` auto-stop policy.
    fn do_stop(&mut self) {
        if self.playing {
            self.playing = false;
            self.paused = false;
            let tempo = self.timeline.tempo_mbpm_at(self.frame);
            let _ = self.timeline.reset(self.frame, tempo);
            for ts in self.tracks.iter_mut().flatten() {
                ts.needs_anchor = true;
                ts.prev_unity = None;
                ts.last_wrap_count = 0;
                // The epoch reset invalidates any quantized
                // activation target (it was a bu of the old epoch).
                ts.pending_bu = None;
                #[cfg(feature = "drift-injection")]
                {
                    ts.float_cursor = None;
                }
            }
            push_event(&mut self.events, &mut self.events_dropped, RtEvent::Stopped);
        }
    }

    fn write_snapshot(&mut self, period_ns: u64) {
        let bu = self.timeline.bu(self.frame);
        let (whole_beats, frac) = beats_display(bu);
        let mut tracks = [RawTrackSnap::default(); MAX_TRACKS];
        for (slot, t) in self.tracks.iter().enumerate() {
            if let Some(ts) = t.as_deref() {
                tracks[slot] = RawTrackSnap {
                    occupied: true,
                    active: ts.active,
                    variation: ts.variation,
                    gain: ts.gain,
                    pan: ts.pan,
                    muted: ts.muted,
                    solo: ts.solo,
                    loop_beats: ts.map.loop_beats(),
                    loop_count: ts.map.wrap_count(bu) as u32,
                    loop_progress: crate::timeline::loop_progress_display(&ts.map, bu),
                    rms_l: ts.meter.rms_l,
                    rms_r: ts.meter.rms_r,
                    peak_l: ts.meter.peak_l,
                    peak_r: ts.meter.peak_r,
                    // Live atomics, not the render-path cache: a stream can
                    // complete while paused/stopped and the snapshot must
                    // say so (ADR-112 §5.1).
                    loaded_samples: ts.pcm.frontier(),
                    streaming: !ts.pcm.is_complete(),
                    starved: ts.starved,
                };
            }
        }
        self.snapshot.write(RawSnapshot {
            frame: self.frame,
            tempo_mbpm: self.timeline.tempo_mbpm_at(self.frame),
            playing: self.playing,
            paused: self.paused,
            timeline_epoch: self.timeline.epoch(),
            bar: (whole_beats / self.beats_per_bar as u64) as u32 + 1,
            beat_in_bar: (whole_beats % self.beats_per_bar as u64) as f32 + frac + 1.0,
            underruns: self.underruns
                + self
                    .external_underruns
                    .as_ref()
                    .map(|c| c.load(std::sync::atomic::Ordering::Relaxed))
                    .unwrap_or(0),
            render_ns_max: self.render_ns_max,
            callback_period_ns: period_ns,
            blocks_total: self.blocks_total,
            blocks_over_half_period: self.blocks_over_half_period,
            events_dropped: self.events_dropped,
            tracks,
        });
    }
}

/// The C ABI render entry (spec §3.6): called by the OS audio callback —
/// iOS `AVAudioSourceNode` render block, later the AudioWorklet (M4). One
/// call renders the full mix into non-interleaved stereo buffers.
///
/// # Safety
///
/// `core` must be the live `Box<AudioCore>` pointer from `create_engine`,
/// called from exactly one thread (the audio thread); `left`/`right` must be
/// valid for `frames` writes each.
#[no_mangle]
pub unsafe extern "C" fn maestro_render(
    core: *mut AudioCore,
    left: *mut f32,
    right: *mut f32,
    frames: u32,
) {
    if core.is_null() || left.is_null() || right.is_null() {
        return;
    }
    let core = &mut *core;
    let l = std::slice::from_raw_parts_mut(left, frames as usize);
    let r = std::slice::from_raw_parts_mut(right, frames as usize);
    core.render(l, r);
}

/// Select a drift-injection sabotage mode from the device harness
/// (spec §6.4 negative controls). **No-op in shipping builds** — the symbol
/// always exists (so the Swift harness links unconditionally), but only a
/// `drift-injection` build has a sabotage selector to set.
#[no_mangle]
pub extern "C" fn maestro_set_sabotage(mode: u8) {
    #[cfg(feature = "drift-injection")]
    crate::sabotage::set_mode_raw(mode);
    #[cfg(not(feature = "drift-injection"))]
    let _ = mode;
}

/// Read `frames` source samples of `pcm` starting at unwrapped sample `n`
/// into `dest` (interleaved stereo), following the beat-space Bresenham wrap.
/// Copies contiguous runs — cost is O(wraps in span), not O(samples). Free
/// function so the variation crossfade can read the *outgoing* PCM through
/// the *outgoing* variation's map (per-variation lengths, ADR-109); `pcm`
/// must be the PCM this `map` was validated against.
fn read_pcm(map: &TrackTimeline, pcm: &EnginePcm, n: u64, frames: usize, dest: &mut [f32]) {
    debug_assert!(dest.len() >= frames * 2);
    let pcm_len = map.pcm_len();

    #[cfg(feature = "drift-injection")]
    if crate::sabotage::mode() == crate::sabotage::Sabotage::PcmLengthLoop {
        // Defect D2's audio half: modulo PCM length, ignoring beat space.
        for k in 0..frames {
            let idx = ((n + k as u64) % pcm_len) as usize;
            pcm.copy_interleaved(idx, 1, &mut dest[k * 2..k * 2 + 2]);
        }
        return;
    }

    let mut n = n;
    let mut filled = 0usize;
    while filled < frames {
        let remaining = (frames - filled) as u64;
        let run = map.run_remaining(n).min(remaining);
        let idx = map.pcm_index(n);
        // The Bresenham long wrap can dictate one index past PCM that
        // decoded at the floor length; clamp duplicates the last sample
        // (sub-sample seam detail, never accumulated — spec §3.1).
        let avail = pcm_len - idx;
        let copy = run.min(avail) as usize;
        let (idx, run) = (idx as usize, run as usize);
        if copy > 0 {
            pcm.copy_interleaved(idx, copy, &mut dest[filled * 2..(filled + copy) * 2]);
        }
        if copy < run {
            pcm.fill_interleaved(
                pcm_len as usize - 1,
                run - copy,
                &mut dest[(filled + copy) * 2..(filled + run) * 2],
            );
        }
        filled += run;
        n += run as u64;
    }
}

/// The highest PCM index the span `[lo, hi)` touches, under the Bresenham
/// wrap (ADR-112 §5.4): a span crossing a wrap seam reads the loop TAIL
/// (`read_pcm`'s clamp arm reads `pcm_len − 1`); within one wrap the index
/// is monotone, so the maximum is at `hi − 1`.
fn max_touched(map: &TrackTimeline, lo: u64, hi: u64) -> u64 {
    debug_assert!(hi > lo);
    if map.wrap_of_sample(hi - 1) > map.wrap_of_sample(lo) {
        map.pcm_len() - 1
    } else {
        map.pcm_index(hi - 1)
    }
}

/// ADR-112 §5.4: the highest PCM index of every source read `render_track`
/// would issue this block, or `None` when no read occurs. MIRRORS
/// `render_track`'s branch structure — any read added there must be added
/// here (the fade arms are over-approximated conservatively where the two
/// diverge only in transition bookkeeping; a false starve is one silent
/// block behind the resume margin, a false render would read uninitialized
/// memory). The mirror claim covers PRODUCTION reads only: the test-only
/// `drift-injection` sabotage paths (FloatCursor especially) deliberately
/// diverge from dictated positions and are not modeled here.
fn required_max_touched(
    ts: &TrackState,
    bu0: u128,
    bu1: u128,
    tempo_mbpm: u32,
    b: usize,
) -> Option<u64> {
    let map = &ts.map;
    let in_lat = ts.stretcher.input_latency() as u64;
    let out_lat = ts.stretcher.output_latency() as u64;
    let unity = map.original_tempo_mbpm() == tempo_mbpm && ts.pitch_semitones == 0.0;
    let n0 = ts.dictated_u(bu0);
    let lead = out_lat as u128 * tempo_mbpm as u128;
    let n0_s = ts.dictated_u(bu0 + lead);
    let span_s = ((ts.dictated_u(bu1 + lead) - n0_s) as usize).min(MAX_FEED_FRAMES) as u64;
    let mut worst: Option<u64> = None;
    let mut add = |lo: u64, hi: u64| {
        if hi > lo {
            let m = max_touched(map, lo, hi);
            worst = Some(worst.map_or(m, |w| w.max(m)));
        }
    };
    if unity {
        // Bit-exact bypass: exactly [n0, n0 + b).
        add(n0, n0 + b as u64);
        // stretch → unity fade: one more streamed span_s block.
        if ts.prev_unity == Some(false) && span_s > 0 && !ts.needs_anchor {
            add(n0_s + in_lat, n0_s + in_lat + span_s);
        }
    } else {
        // Stretch feed: [n0_s + input_latency, + span_s) (ADR-094 lead).
        add(n0_s + in_lat, n0_s + in_lat + span_s);
        if ts.needs_anchor {
            // Anchor window: input+output latency frames ending at
            // n0_s + input_latency (TrackState::anchor).
            add(n0_s.saturating_sub(out_lat), n0_s + in_lat);
        }
        // Fade-out sources (one-block transitions, spec §3.2):
        match (ts.prev_unity, ts.reanchor_fade_tempo) {
            (Some(false), Some(old_tempo)) if ts.needs_anchor => {
                // ADR-094 old-schedule fade: one final block at the OLD tempo.
                let old_lead = out_lat as u128 * old_tempo as u128;
                let a = ts.dictated_u(bu0 + old_lead);
                let span_old = ((ts
                    .dictated_u(bu0 + old_tempo as u128 * b as u128 + old_lead)
                    - a) as usize)
                    .min(MAX_FEED_FRAMES) as u64;
                add(a + in_lat, a + in_lat + span_old);
            }
            (Some(prev), _) if prev || ts.needs_anchor => {
                // 1:1 fade-out continuation from n0 (bypass→stretch, or a
                // re-anchor with no warm stretch stream).
                add(n0, n0 + b as u64);
            }
            _ => {}
        }
    }
    worst
}

/// Render one active track into the mix. Free function so the borrow on the
/// track is independent of `self` (the event queue is handled by the caller).
/// Effective transpose under the §4.8 combined-ratio clamp: the combined
/// time-pitch ratio must stay inside Signalsmith's clean-stretch bound
/// [0.25, 4.0] so its RNG path (audio-value nondeterminism, spec §2.3#4) is
/// never entered. Audio-only — floats never feed timing.
#[inline]
fn effective_transpose(pitch_semitones: f32, rate: f64) -> f32 {
    if pitch_semitones == 0.0 {
        return 0.0;
    }
    let lo = 12.0 * (0.25 / rate).log2();
    let hi = 12.0 * (4.0 / rate).log2();
    (pitch_semitones as f64).clamp(lo, hi) as f32
}

fn render_track(
    ts: &mut TrackState,
    bu0: u128,
    bu1: u128,
    tempo_mbpm: u32,
    target_gain: f32,
    b: usize,
    left: &mut [f32],
    right: &mut [f32],
) {
    if b == 0 {
        return;
    }
    // Pending mid-stream re-anchor fade (ADR-094): valid for exactly the
    // first render after the tempo change that set it — taken
    // unconditionally so it can never go stale across silence or restarts.
    let reanchor_fade_tempo = ts.reanchor_fade_tempo.take();
    // Unity means O == E **exactly** (integer compare, spec §3.2) and no
    // pitch shift — a pitch-shifted track is promoted to the stretch path.
    #[allow(unused_mut)]
    let mut unity = ts.map.original_tempo_mbpm() == tempo_mbpm && ts.pitch_semitones == 0.0;

    // Dictated span for this block (first difference — telescopes exactly).
    #[allow(unused_mut)]
    let (mut n0, mut span) = {
        let n0 = ts.dictated_u(bu0);
        let n1 = ts.dictated_u(bu1);
        (n0, (n1 - n0) as usize)
    };

    // Stretch-path feed positions — output-latency compensation (ADR-094).
    // The vocoder delays its output by a constant `output_latency` OUTPUT
    // frames (deterministic per spec §2.3#4: 2880 at 48 kHz preset_default),
    // so a stretch feed evaluated at bu lands every onset 60 ms behind the
    // bypass grid — audible whenever a stretched stem plays against a unity
    // stem (cross-song ghost stems; the C3 finding, ADR-093). The stretch
    // feed therefore evaluates the SAME derived map `output_latency` output
    // frames ahead: bu + E·out_lat. Recomputed each block from the current
    // tempo, the lead is exactly out_lat output frames at every rate (never
    // a baked-in rate-scaled source count); spans telescope at constant
    // tempo. The schedule is a function of E, so a tempo change re-derives
    // it discontinuously — the SetTempo handler re-anchors stretch tracks
    // under the one-block crossfade (see the fade arm below) rather than
    // feeding across the seam. Positions are untouched: this shifts only
    // which CONTENT the stretcher is fed, never what u(F) reports.
    #[allow(unused_mut)]
    let (mut n0_s, mut span_s) = {
        let lead = ts.stretcher.output_latency() as u128 * tempo_mbpm as u128;
        let a = ts.dictated_u(bu0 + lead);
        (a, (ts.dictated_u(bu1 + lead) - a) as usize)
    };

    #[cfg(feature = "drift-injection")]
    if crate::sabotage::mode() == crate::sabotage::Sabotage::StretchDisabled {
        // Defect class D5: stretch silently disabled — the track plays 1:1
        // from the timeline epoch regardless of tempo (source sample = output
        // frame). Reported positions stay correct (the defect was *silent*);
        // the device analyzer must catch it from the AUDIO (C0 negative
        // control: cross-tempo onsets land at wrong spacing).
        unity = true;
        n0 = (bu0 / tempo_mbpm as u128) as u64;
        span = ((bu1 - bu0) / tempo_mbpm as u128) as usize;
    }

    #[cfg(feature = "drift-injection")]
    if crate::sabotage::mode() == crate::sabotage::Sabotage::FloatCursor {
        // Defect class D1: a per-track position *advanced* each block. The
        // cursor is seeded from the derived position on first use, then
        // free-runs — the sabotaged feed follows it verbatim (no lead: the
        // defect being simulated predates the compensation).
        let rate = tempo_mbpm as f32 / ts.map.original_tempo_mbpm() as f32;
        let c0 = ts.float_cursor.unwrap_or(ts.map.u(bu0) as f32);
        let c1 = c0 + b as f32 * rate;
        ts.float_cursor = Some(c1);
        n0 = c0 as u64;
        span = (c1 as u64 - c0 as u64) as usize;
        n0_s = n0;
        span_s = span;
    }

    debug_assert!(span <= MAX_FEED_FRAMES);
    debug_assert!(span_s <= MAX_FEED_FRAMES);
    let span_s = span_s.min(MAX_FEED_FRAMES);
    // The un-led dictated span only backs the unity bypass identity check
    // below, which the drift-injection build disables.
    #[cfg(feature = "drift-injection")]
    let _ = span;

    // ---- Crossfade fade-out source (spec §3.2: one-block, equal-power). ----
    // A variation swap takes priority (its discontinuity subsumes any path
    // transition this block); otherwise a unity↔stretch flip fades between
    // the old and new render paths. Audio-only — positions are derived and
    // identical on both sides of every fade.
    let mut fade_out = false;
    if let Some((old_pcm, old_map)) = ts.xfade_old.as_ref() {
        // Outgoing variation, read 1:1 through the OUTGOING map — variations
        // may differ in loop length (ADR-109), and `read_pcm` bounds by the
        // map's pcm_len, so the old PCM must never be read through the new
        // (possibly longer) map. `n0` is map-independent (shared O): the fade
        // continues exactly what the old variation was playing. At stretch
        // rates these ≤ 10 ms play at source rate under the half-power tail —
        // masked, like the anchor smear (spec §3.2).
        let mut scratch = std::mem::take(&mut ts.xfade_scratch);
        read_pcm(old_map, old_pcm, n0, b, &mut scratch);
        ts.xfade_scratch = scratch;
        fade_out = true;
    } else {
        match ts.prev_unity {
            Some(prev_was_unity) if !unity && (prev_was_unity || ts.needs_anchor) => {
                match (prev_was_unity, reanchor_fade_tempo) {
                    (false, Some(old_tempo)) => {
                        // Mid-stream stretch re-anchor (tempo change,
                        // ADR-094): fade out one FINAL block of the old feed
                        // schedule from the still-warm stretcher — waveform-
                        // continuous with the previous block (a 1:1 source
                        // read here would step against the vocoder's output;
                        // measured 3.5× the T-SEAM seam bound). The main
                        // path below then re-anchors on the new schedule
                        // under the fade-in — the same one-block shape as
                        // the stretch→unity transition.
                        let lead =
                            ts.stretcher.output_latency() as u128 * old_tempo as u128;
                        let a = ts.dictated_u(bu0 + lead);
                        let span_old = (ts
                            .dictated_u(bu0 + old_tempo as u128 * b as u128 + lead)
                            - a) as usize;
                        let span_old = span_old.min(MAX_FEED_FRAMES);
                        if span_old > 0 {
                            let read_from = a + ts.stretcher.input_latency() as u64;
                            let mut feed = std::mem::take(&mut ts.feed_scratch);
                            ts.read_source(read_from, span_old, &mut feed);
                            ts.feed_scratch = feed;
                            let mut scratch = std::mem::take(&mut ts.xfade_scratch);
                            let _ = ts.stretcher.process_interleaved(
                                &ts.feed_scratch[..span_old * 2],
                                &mut scratch[..b * 2],
                            );
                            ts.xfade_scratch = scratch;
                            fade_out = true;
                        }
                    }
                    _ => {
                        // bypass → stretch (or a re-anchor whose predecessor
                        // was not a warm stretch stream — e.g. a variation
                        // swap applied while paused, masked by the resume
                        // ramp): fade out the 1:1 continuation from n0 —
                        // content-continuous with the previous block. At
                        // stretch rates these ≤ 10 ms play at source rate
                        // under the half-power tail — masked, like the
                        // variation-swap fade.
                        let mut scratch = std::mem::take(&mut ts.xfade_scratch);
                        ts.read_source(n0, b, &mut scratch);
                        ts.xfade_scratch = scratch;
                        fade_out = true;
                    }
                }
            }
            Some(false) if unity && span_s > 0 && !ts.needs_anchor => {
                // stretch → unity: fade out one more streamed stretch block
                // (at unity the dictated span == b, so the ratio is already
                // 1:1); the stretcher is then left until the next anchor.
                // The compensated feed keeps its content grid-aligned with
                // the incoming bypass — the crossfade joins two signals at
                // the SAME musical position (ADR-094).
                let read_from = n0_s + ts.stretcher.input_latency() as u64;
                let mut feed = std::mem::take(&mut ts.feed_scratch);
                ts.read_source(read_from, span_s, &mut feed);
                ts.feed_scratch = feed;
                let mut scratch = std::mem::take(&mut ts.xfade_scratch);
                let _ = ts
                    .stretcher
                    .process_interleaved(&ts.feed_scratch[..span_s * 2], &mut scratch[..b * 2]);
                ts.xfade_scratch = scratch;
                fade_out = true;
            }
            _ => {}
        }
    }

    // Mix parameters lifted out before `rendered` borrows the scratch buffers.
    // The applied coefficients ramp toward the target over this block
    // (click-free gain/pan/mute changes); identical endpoints lerp to the
    // exact constant, so the unity-bypass bit-exactness guarantee holds.
    let (gl1, gr1) = mix_coefficients(target_gain, ts.pan);
    let (gl0, gr0) = ts.applied_mix.unwrap_or((gl1, gr1));
    let mut ramp_remaining = ts.ramp_remaining;

    let rendered: &[f32] = if unity {
        // Bit-exact bypass (spec §3.2): span == B identically at unity.
        #[cfg(not(feature = "drift-injection"))]
        debug_assert_eq!(span, b);
        let mut scratch = std::mem::take(&mut ts.feed_scratch);
        ts.read_source(n0, b, &mut scratch);
        ts.feed_scratch = scratch;
        ts.needs_anchor = true; // leaving unity later must re-anchor
        &ts.feed_scratch[..b * 2]
    } else {
        let rate = tempo_mbpm as f64 / ts.map.original_tempo_mbpm() as f64;
        if ts.needs_anchor {
            ts.anchor(n0_s, rate);
            ts.needs_anchor = false;
        }
        // Pitch (spec §3.2): set_transpose_semitones, re-applied only when the
        // effective value changes (tempo changes move the §4.8 clamp window).
        let desired = effective_transpose(ts.pitch_semitones, rate);
        if ts.applied_pitch != Some(desired) {
            let _ = ts.stretcher.set_transpose_semitones(desired);
            ts.applied_pitch = Some(desired);
        }
        // The seek window already primed input up to the compensated feed
        // origin + input_latency; streaming continues from there. Re-feeding
        // from the origin would double-feed the overlap and delay every
        // output by input_latency/rate — a rate-DEPENDENT lag that misaligns
        // cross-tempo stems (measured: 720 samples between rates 2.0 and
        // 1.33). The shift is a constant inside the position→content
        // mapping; positions remain derived.
        let read_from = n0_s + ts.stretcher.input_latency() as u64;
        if span_s == 0 {
            // Sub-frame source advance (only possible at extreme rate × tiny
            // block combinations outside M0's envelope). Hold silence for the
            // block rather than calling the stretcher with no input.
            ts.out_scratch[..b * 2].fill(0.0);
        } else {
            let mut feed = std::mem::take(&mut ts.feed_scratch);
            ts.read_source(read_from, span_s, &mut feed);
            ts.feed_scratch = feed;
            let mut out = std::mem::take(&mut ts.out_scratch);
            let _ = ts
                .stretcher
                .process_interleaved(&ts.feed_scratch[..span_s * 2], &mut out[..b * 2]);
            ts.out_scratch = out;
        }
        &ts.out_scratch[..b * 2]
    };
    ts.prev_unity = Some(unity);

    // Equal-power pan + gain + activation ramp (+ one-block equal-power
    // crossfade when a fade-out source was rendered), mixed into the bus.
    // Track contribution is metered here (post everything, display only).
    let ramp_total = ACTIVATION_RAMP_FRAMES as f32;
    let inv_b = 1.0 / b as f32;
    let mut sumsq_l = 0.0f32;
    let mut sumsq_r = 0.0f32;
    let mut peak_l = 0.0f32;
    let mut peak_r = 0.0f32;
    for i in 0..b {
        let g = if ramp_remaining > 0 {
            let g = 1.0 - ramp_remaining as f32 / ramp_total;
            ramp_remaining -= 1;
            g
        } else {
            1.0
        };
        let (l_s, r_s) = if fade_out {
            let t = (i as f32 + 0.5) * inv_b * std::f32::consts::FRAC_PI_2;
            let (g_in, g_out) = (t.sin(), t.cos());
            (
                rendered[i * 2] * g_in + ts.xfade_scratch[i * 2] * g_out,
                rendered[i * 2 + 1] * g_in + ts.xfade_scratch[i * 2 + 1] * g_out,
            )
        } else {
            (rendered[i * 2], rendered[i * 2 + 1])
        };
        let t = (i + 1) as f32 * inv_b;
        let gl = gl0 + (gl1 - gl0) * t;
        let gr = gr0 + (gr1 - gr0) * t;
        let c_l = l_s * gl * g;
        let c_r = r_s * gr * g;
        left[i] += c_l;
        right[i] += c_r;
        sumsq_l += c_l * c_l;
        sumsq_r += c_r * c_r;
        peak_l = peak_l.max(c_l.abs());
        peak_r = peak_r.max(c_r.abs());
    }
    ts.ramp_remaining = ramp_remaining;
    ts.applied_mix = Some((gl1, gr1));
    // Meter ballistics (ADR-095, display only — floats never feed timing):
    // attack instant, release −20 dB per 300 ms. A raw per-block value is a
    // 2.67 ms sample that a 15–30 Hz snapshot poll aliases straight past
    // (the M4 web E2E read near-zero meters on demonstrably loud content —
    // peaks BETWEEN polls are unrecoverable downstream, so the hold lives
    // here where every block is seen). Steady-state values are identical to
    // the previous per-block definition (max with a decayed copy), and the
    // DSP-skip path still zeroes instantly.
    let decay = 10f32.powf(-(b as f32) / 14_400.0); // 20 dB / (0.3 s · 48 kHz)
    ts.meter = TrackMeter {
        rms_l: (sumsq_l * inv_b).sqrt().max(ts.meter.rms_l * decay),
        rms_r: (sumsq_r * inv_b).sqrt().max(ts.meter.rms_r * decay),
        peak_l: peak_l.max(ts.meter.peak_l * decay),
        peak_r: peak_r.max(ts.meter.peak_r * decay),
    };

    // The variation fade has consumed its outgoing PCM: stage it for the
    // reclaim queue (drained at the top of every render — never dropped on
    // the audio thread). If a previous retiree is still waiting (reclaim
    // queue jammed), keep the fade source in place; the fade repeats next
    // block — degraded audio, never a drop.
    if fade_out && ts.xfade_old.is_some() && ts.retired_pcm.is_none() {
        ts.retired_pcm = ts.xfade_old.take().map(|(pcm, _)| pcm);
    }
}
