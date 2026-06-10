//! Shared test utilities for the native-rebuild validation suite (spec §6).
//!
//! - `reference` — exact `num-bigint` mirror of the §3.1 formulas
//! - WAV fixture loading (32-bit float RIFF, Logic Pro JUNK-chunk tolerant)
//! - `detect_click_onsets` — promoted from the abandoned branch per the
//!   spec §7 reuse map; also used by the `maestro-validate` device CLI
//! - the T-INV scenario runner driving the real `AudioCore` render entry

#![allow(dead_code)] // each integration-test binary uses a subset

pub mod reference;
pub mod validation_thresholds;

use std::path::PathBuf;

use maestro_api::{Command, EngineConfig, Event, LoadMeta, QuantizePolicy};
use maestro_core::audio_core::AudioCore;
use maestro_core::control_plane::{create_engine, ControlPlane};

pub const SAMPLE_RATE: u32 = 48_000;
pub const BEATS_PER_BAR: u32 = 4;

// ---------------------------------------------------------------------------
// WAV fixtures + onset detection: shared with maestro-validate via
// maestro_core::validation (spec §7 — promoted from the branch's test code).
// ---------------------------------------------------------------------------

#[allow(unused_imports)] // each integration-test binary uses a subset
pub use maestro_core::validation::{detect_click_onsets, load_float_wav, LoadedWav};

pub fn fixture_path(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push(name);
    p
}

// ---------------------------------------------------------------------------
// T-INV scenario runner (spec §6.2)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
pub struct ScenarioTrack {
    pub o_mbpm: u32,
    pub loop_beats: u32,
    pub pcm_len: u64,
    /// Sine frequency for the synthetic content (anti-stub RMS floor, §6.1).
    pub sine_hz: f32,
    pub pan: f32,
}

/// The §6.2 track set: O = {60k, 90k, 113k, 116k} mBPM with loop lengths from
/// real metadata **including the non-integer musical-length cases (§2.2)** —
/// Humboldt bass (815_575.221… exact) and San Francisco voice (2_085_517.241…
/// exact). The 60/90 tracks use the metronome fixtures' exact-integer geometry.
pub const T_INV_TRACKS: [ScenarioTrack; 4] = [
    ScenarioTrack { o_mbpm: 60_000, loop_beats: 16, pcm_len: 768_000, sine_hz: 220.0, pan: -1.0 },
    ScenarioTrack { o_mbpm: 90_000, loop_beats: 16, pcm_len: 512_000, sine_hz: 330.0, pan: 1.0 },
    ScenarioTrack { o_mbpm: 113_000, loop_beats: 32, pcm_len: 815_576, sine_hz: 440.0, pan: -0.5 },
    ScenarioTrack { o_mbpm: 116_000, loop_beats: 84, pcm_len: 2_085_518, sine_hz: 550.0, pan: 0.5 },
];

/// Block sizes alternate per render call (proves block-size independence).
pub const T_INV_BLOCKS: [usize; 3] = [128, 480, 1024];

pub struct Checkpoint {
    /// True rendered frame count (tracked by the harness, NOT read from the
    /// engine — so a sabotaged engine clock cannot fool the reference).
    pub rendered_frames: u64,
    /// Engine-reported dictated unwrapped positions per track.
    pub positions: Vec<u64>,
    /// Exact rational reference positions at `rendered_frames`.
    pub reference: Vec<u64>,
}

pub struct ScenarioResult {
    pub checkpoints: Vec<Checkpoint>,
    /// Checkpoint entries where engine position != reference position.
    pub position_mismatches: usize,
    /// RMS of the rendered mix (anti-stub: silence fails, spec §6.1).
    pub mix_rms: f64,
    /// All events drained during the run (TempoApplied / TrackActivated / …) —
    /// the M1 legs assert exact application frames against these.
    pub events: Vec<Event>,
}

impl ScenarioResult {
    /// Stable digest of the engine-reported checkpoint positions, for the
    /// cross-platform parity comparison (spec §6.6): byte-for-byte equality
    /// of these dumps is asserted across x86/ARM (and wasm32 from M4).
    pub fn position_digest(&self) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325; // FNV-1a
        for cp in &self.checkpoints {
            for &p in std::iter::once(&cp.rendered_frames).chain(cp.positions.iter()) {
                for byte in p.to_le_bytes() {
                    h ^= byte as u64;
                    h = h.wrapping_mul(0x1000_0000_01b3);
                }
            }
        }
        h
    }
}

pub fn make_sine_pcm(len: u64, hz: f32) -> (Vec<f32>, Vec<f32>) {
    let mut l = Vec::with_capacity(len as usize);
    for n in 0..len {
        let t = n as f32 / SAMPLE_RATE as f32;
        l.push(0.25 * (2.0 * std::f32::consts::PI * hz * t).sin());
    }
    (l.clone(), l)
}

pub fn build_engine(tempo_mbpm: u32, tracks: &[ScenarioTrack]) -> (ControlPlane, Box<AudioCore>) {
    build_engine_select(tempo_mbpm, tracks, &(0..tracks.len()).collect::<Vec<_>>())
}

/// Build an engine with all tracks loaded but only `active_slots` activated —
/// the M1 mid-stream-arm leg starts with a track loaded and silent.
pub fn build_engine_select(
    tempo_mbpm: u32,
    tracks: &[ScenarioTrack],
    active_slots: &[usize],
) -> (ControlPlane, Box<AudioCore>) {
    let (mut plane, core) = create_engine(EngineConfig {
        sample_rate: SAMPLE_RATE,
        initial_tempo_mbpm: tempo_mbpm,
        beats_per_bar: BEATS_PER_BAR,
    })
    .expect("engine config valid");
    for (slot, t) in tracks.iter().enumerate() {
        let (l, r) = make_sine_pcm(t.pcm_len, t.sine_hz);
        plane
            .load_pcm(
                &LoadMeta {
                    slot: slot as u8,
                    original_tempo_mbpm: t.o_mbpm,
                    loop_beats: t.loop_beats,
                    encoder_delay: 0,
                    end_padding: 0,
                    original_samples: t.pcm_len,
                    variation: 0,
                    gain: 1.0,
                    pan: t.pan,
                },
                l,
                r,
            )
            .expect("synthetic track loads");
        if active_slots.contains(&slot) {
            let res = plane.dispatch(Command::Activate {
                slot: slot as u8,
                quantize: QuantizePolicy::Immediate,
            });
            assert!(matches!(res, maestro_api::CommandResult::Ok), "activate rejected: {res:?}");
        }
    }
    (plane, core)
}

/// Drive the real render entry for `total_frames`, checkpointing every
/// `checkpoint_interval` frames against the bigint reference. The M0 shape:
/// fixed tempo, all tracks active, no script.
pub fn run_t_inv_scenario(
    total_frames: u64,
    tempo_mbpm: u32,
    checkpoint_interval: u64,
) -> ScenarioResult {
    run_scripted_scenario(
        total_frames,
        tempo_mbpm,
        checkpoint_interval,
        &(0..T_INV_TRACKS.len()).collect::<Vec<_>>(),
        &[],
    )
}

// ---------------------------------------------------------------------------
// Scripted T-INV runner (spec §6.2, M1 legs): tempo changes, pause/resume,
// mid-stream arming, track stop → tempo-change-while-silent → restart.
// ---------------------------------------------------------------------------

/// One scripted action, fired when the harness's own engine-frame mirror
/// reaches `at_frame`. Actions at the same frame fire in script order.
pub enum ScriptAction {
    /// Dispatch a `maestro-api` command. It lands at exactly `at_frame`: the
    /// runner clamps block sizes so a block boundary falls on every scripted
    /// frame, and commands apply at the next block boundary.
    Cmd(Command),
    /// Render `n` blocks while the clock is expected frozen (paused). The
    /// runner asserts every track's dictated position is identical before and
    /// after — F must not advance, for all tracks at once (spec §3.1).
    IdleBlocksFrozen(usize),
}

pub struct ScriptEvent {
    pub at_frame: u64,
    pub action: ScriptAction,
}

/// Drive the real render entry through a command script, mirroring the tempo
/// segment table in the exact `num-bigint` reference. Checkpoints land every
/// `checkpoint_interval` engine frames AND at every scripted frame (so legs
/// like re-entry are checked at their exact boundary).
///
/// The engine-frame mirror is tracked by the harness itself (never read from
/// the engine), so a sabotaged clock cannot fool the reference.
pub fn run_scripted_scenario(
    total_frames: u64,
    initial_tempo_mbpm: u32,
    checkpoint_interval: u64,
    active_slots: &[usize],
    script: &[ScriptEvent],
) -> ScenarioResult {
    let tracks = T_INV_TRACKS;
    let (mut plane, mut core) = build_engine_select(initial_tempo_mbpm, &tracks, active_slots);
    assert!(matches!(plane.dispatch(Command::Play), maestro_api::CommandResult::Ok));

    let max_block = *T_INV_BLOCKS.iter().max().unwrap();
    let mut left = vec![0.0f32; max_block];
    let mut right = vec![0.0f32; max_block];

    // Harness-owned mirrors. `engine_f` advances only while the mirror says
    // the clock runs — the engine must agree through its positions.
    let mut ref_timeline = reference::RefTimeline::new(initial_tempo_mbpm);
    let mut engine_f: u64 = 0;
    let mut playing = true;
    let mut paused = false;

    let mut script_idx = 0usize;
    let mut next_checkpoint = checkpoint_interval;
    let mut checkpoints = Vec::new();
    let mut position_mismatches = 0usize;
    let mut sumsq: f64 = 0.0;
    let mut weight: u64 = 0;
    let mut block_i = 0usize;
    let mut events: Vec<Event> = Vec::new();

    let checkpoint =
        |core: &AudioCore, engine_f: u64, ref_timeline: &reference::RefTimeline,
         checkpoints: &mut Vec<Checkpoint>, position_mismatches: &mut usize| {
            let bu = ref_timeline.bu(engine_f);
            let mut positions = Vec::with_capacity(tracks.len());
            let mut refs = Vec::with_capacity(tracks.len());
            for (slot, t) in tracks.iter().enumerate() {
                let engine_pos =
                    core.dictated_unwrapped_position(slot).expect("track installed");
                let ref_pos = reference::u(&bu, t.o_mbpm);
                if engine_pos != ref_pos {
                    *position_mismatches += 1;
                }
                positions.push(engine_pos);
                refs.push(ref_pos);
            }
            checkpoints.push(Checkpoint { rendered_frames: engine_f, positions, reference: refs });
        };

    while engine_f < total_frames {
        // Fire due script actions (commands land at exactly engine_f).
        while let Some(ev) = script.get(script_idx) {
            if ev.at_frame > engine_f {
                break;
            }
            // A checkpoint at every scripted frame: legs are verified at
            // their exact boundaries, not just on the 1M grid.
            checkpoint(&core, engine_f, &ref_timeline, &mut checkpoints, &mut position_mismatches);
            match &ev.action {
                ScriptAction::Cmd(cmd) => {
                    // Mirror the timing-relevant state BEFORE dispatch; the
                    // command applies at the next render's block boundary,
                    // which is exactly engine_f.
                    match cmd {
                        Command::SetTempo { mbpm } => ref_timeline.set_tempo(engine_f, *mbpm),
                        Command::Pause => paused = true,
                        Command::Resume => paused = false,
                        Command::Stop => {
                            playing = false;
                            paused = false;
                            let tempo = ref_timeline.tempo_at(engine_f);
                            ref_timeline.reset(engine_f, tempo);
                        }
                        Command::Play => playing = true,
                        _ => {}
                    }
                    let res = plane.dispatch(cmd.clone());
                    assert!(
                        matches!(res, maestro_api::CommandResult::Ok),
                        "scripted command rejected at frame {engine_f}: {res:?}"
                    );
                }
                ScriptAction::IdleBlocksFrozen(n) => {
                    // Render with the clock frozen: every dictated position
                    // must be identical before and after (spec §3.1 — pause
                    // freezes F for all tracks at once).
                    let before: Vec<u64> = (0..tracks.len())
                        .map(|s| core.dictated_unwrapped_position(s).expect("installed"))
                        .collect();
                    for k in 0..*n {
                        let b = T_INV_BLOCKS[(block_i + k) % T_INV_BLOCKS.len()];
                        core.render(&mut left[..b], &mut right[..b]);
                    }
                    let after: Vec<u64> = (0..tracks.len())
                        .map(|s| core.dictated_unwrapped_position(s).expect("installed"))
                        .collect();
                    assert_eq!(
                        before, after,
                        "clock advanced during pause/stop at frame {engine_f} — F must freeze \
                         for all tracks at once"
                    );
                }
            }
            script_idx += 1;
        }

        // Block size: the T_INV rotation, clamped so the next scripted frame
        // lands exactly on a block boundary.
        let mut b = T_INV_BLOCKS[block_i % T_INV_BLOCKS.len()];
        block_i += 1;
        if playing && !paused {
            if let Some(ev) = script.get(script_idx) {
                b = b.min((ev.at_frame - engine_f) as usize);
            }
        }

        core.render(&mut left[..b], &mut right[..b]);
        if playing && !paused {
            engine_f += b as u64;
            for i in 0..b {
                sumsq += (left[i] as f64).powi(2) + (right[i] as f64).powi(2);
            }
            weight += b as u64;
        }

        if engine_f >= next_checkpoint {
            next_checkpoint += checkpoint_interval;
            checkpoint(&core, engine_f, &ref_timeline, &mut checkpoints, &mut position_mismatches);
        }
        if block_i.is_multiple_of(64) {
            events.extend(plane.drain_events());
        }
    }

    events.extend(plane.drain_events());
    let mix_rms = (sumsq / (weight.max(1) as f64 * 2.0)).sqrt();
    drop(plane);
    ScenarioResult { checkpoints, position_mismatches, mix_rms, events }
}
