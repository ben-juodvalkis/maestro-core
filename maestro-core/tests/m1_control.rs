//! # M1 control-surface tests (spec §5 M1, §3.2, §4.8)
//!
//! Focused coverage of the M1 operations that T-INV exercises only for
//! position correctness:
//!
//! - **Variation switch**: engine-side Arc PCM swap + one-block crossfade;
//!   the snapshot and event echo the applied variation; the audio actually
//!   changes content (anti-stub: dominant frequency flips); identity
//!   mismatches are hard load failures (spec §3.4).
//! - **Pitch**: ±12 clamp validation (rejected, never silently clamped);
//!   a pitch shift on a unity track promotes it to the stretch path and
//!   audibly transposes (anti-stub: measured frequency ratio); returning to
//!   pitch 0 restores the bit-exact bypass.
//! - **Rate clamp** (§4.8): SetTempo and load_pcm reject combinations that
//!   would push any track's implicit rate outside [0.25, 4.0].

mod common;

use common::{make_sine_pcm, SAMPLE_RATE};
use maestro_api::{Command, CommandResult, EngineConfig, Event, LoadMeta, QuantizePolicy};
use maestro_core::control_plane::{create_engine, ControlPlane};
use maestro_core::audio_core::AudioCore;

const O_MBPM: u32 = 120_000;
const LOOP_BEATS: u32 = 16;
const PCM_LEN: u64 = 384_000;
const BLOCK: usize = 480;

fn engine_with_sine(tempo: u32, hz: f32) -> (ControlPlane, Box<AudioCore>) {
    let (mut plane, core) = create_engine(EngineConfig {
        sample_rate: SAMPLE_RATE,
        initial_tempo_mbpm: tempo,
        beats_per_bar: 4,
    })
    .unwrap();
    let (l, r) = make_sine_pcm(PCM_LEN, hz);
    plane
        .load_pcm(&meta(0, 0), l, r)
        .expect("base variation loads");
    assert!(matches!(
        plane.dispatch(Command::Activate { slot: 0, quantize: QuantizePolicy::Immediate , transport: Default::default() }),
        CommandResult::Ok
    ));
    assert!(matches!(plane.dispatch(Command::Play), CommandResult::Ok));
    (plane, core)
}

fn meta(slot: u8, variation: u8) -> LoadMeta {
    LoadMeta {
        sample_rate: 48_000,
        slot,
        original_tempo_mbpm: O_MBPM,
        loop_beats: LOOP_BEATS,
        encoder_delay: 0,
        end_padding: 0,
        original_samples: PCM_LEN,
        variation,
        gain: 1.0,
        pan: 0.0,
    }
}

fn render_blocks(core: &mut AudioCore, n: usize) -> (Vec<f32>, u64) {
    let mut l = vec![0.0f32; BLOCK];
    let mut r = vec![0.0f32; BLOCK];
    let mut out = Vec::with_capacity(n * BLOCK);
    for _ in 0..n {
        core.render(&mut l, &mut r);
        out.extend_from_slice(&l);
    }
    (out, (n * BLOCK) as u64)
}

/// Dominant frequency estimate via zero crossings (good to ~1% on a sine).
fn measured_hz(signal: &[f32]) -> f64 {
    let crossings = signal.windows(2).filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0)).count();
    crossings as f64 / 2.0 * SAMPLE_RATE as f64 / signal.len() as f64
}

// ---------------------------------------------------------------------------
// Variation switch
// ---------------------------------------------------------------------------

#[test]
fn variation_switch_swaps_content_and_echoes() {
    let (mut plane, mut core) = engine_with_sine(O_MBPM, 220.0);
    let (_, mut frames) = render_blocks(&mut core, 50);

    // Engine-side switch: same musical identity, different content (440 Hz).
    let (l, r) = make_sine_pcm(PCM_LEN, 440.0);
    plane.load_pcm(&meta(0, 1), l, r).expect("variation 1 loads");
    let (_, f2) = render_blocks(&mut core, 50);
    frames += f2;

    // Echo: VariationApplied carries the slot, variation, and a frame inside
    // the rendered range (the block boundary where the swap landed).
    let events = plane.drain_events();
    let applied = events.iter().find_map(|e| match e {
        Event::VariationApplied { slot: 0, variation, at_frame } => Some((*variation, *at_frame)),
        _ => None,
    });
    let (var, at) = applied.expect("VariationApplied event missing");
    assert_eq!(var, 1);
    assert!(at >= 50 * BLOCK as u64 && at < frames, "swap frame {at} outside dispatch window");

    // Snapshot echoes the applied variation.
    let snap = plane.snapshot();
    assert_eq!(snap.tracks[0].variation, 1);

    // Anti-stub: the audio is now the NEW content (440 Hz), past the fade.
    let (tail, _) = render_blocks(&mut core, 50);
    let hz = measured_hz(&tail);
    assert!(
        (hz - 440.0).abs() < 10.0,
        "expected 440 Hz after variation switch, measured {hz:.1} Hz"
    );

    // Positions were never touched: at unity, u(F) == F identically, and the
    // bypass output must be the bit-exact source at the dictated position.
    let pos = core.dictated_unwrapped_position(0).expect("track installed");
    assert_eq!(pos, frames + 50 * BLOCK as u64, "variation switch perturbed the dictated position");
}

#[test]
fn variation_identity_mismatch_is_a_hard_failure() {
    let (mut plane, _core) = engine_with_sine(O_MBPM, 220.0);

    // Different loop_beats: must fail loudly (spec §3.4 — loop length can
    // never change underneath the timeline).
    let (l, r) = make_sine_pcm(PCM_LEN, 440.0);
    let mut bad = meta(0, 1);
    bad.loop_beats = 8;
    bad.original_samples = PCM_LEN; // length no longer matches 8 beats either
    let err = plane.load_pcm(&bad, l, r).unwrap_err();
    assert!(
        err.contains("identity") || err.contains("inconsistent"),
        "expected identity/loop-identity failure, got: {err}"
    );

    // The failure is also echoed as a TrackLoadFailed event.
    let events = plane.drain_events();
    assert!(
        events.iter().any(|e| matches!(e, Event::TrackLoadFailed { slot: 0, .. })),
        "missing TrackLoadFailed event"
    );
}

// ---------------------------------------------------------------------------
// Pitch
// ---------------------------------------------------------------------------

#[test]
fn pitch_out_of_range_is_rejected_never_clamped() {
    let (mut plane, _core) = engine_with_sine(O_MBPM, 220.0);
    for bad in [12.5f32, -100.0, f32::NAN, f32::INFINITY] {
        let res = plane.dispatch(Command::SetPitch { slot: 0, semitones: bad });
        assert!(
            matches!(res, CommandResult::Rejected { .. }),
            "pitch {bad} must be rejected (spec §4.8)"
        );
    }
    // Boundary value is accepted.
    assert!(matches!(
        plane.dispatch(Command::SetPitch { slot: 0, semitones: 12.0 }),
        CommandResult::Ok
    ));
}

#[test]
fn pitch_promotes_unity_track_and_transposes() {
    // Track at its own tempo: bypass. +12 semitones must promote it to the
    // stretch path (positions unchanged) and double the sounding frequency.
    let (mut plane, mut core) = engine_with_sine(O_MBPM, 220.0);
    render_blocks(&mut core, 50);

    assert!(matches!(
        plane.dispatch(Command::SetPitch { slot: 0, semitones: 12.0 }),
        CommandResult::Ok
    ));
    // Settle past the crossfade block + stretcher latency.
    render_blocks(&mut core, 30);
    let (signal, _) = render_blocks(&mut core, 100);
    let hz = measured_hz(&signal);
    assert!(
        (hz - 440.0).abs() < 15.0,
        "expected ~440 Hz (220 Hz + 12 semitones), measured {hz:.1} Hz"
    );

    // Positions are untouched by pitch: still u(F) == F at unity tempo.
    let rendered = (50 + 30 + 100) * BLOCK as u64;
    assert_eq!(core.dictated_unwrapped_position(0), Some(rendered));

    // Back to 0: the track returns to the bit-exact bypass. After the
    // one-block crossfade, output samples must equal source · pan gain
    // exactly (bit-for-bit — this is the bypass guarantee, spec §3.2).
    assert!(matches!(
        plane.dispatch(Command::SetPitch { slot: 0, semitones: 0.0 }),
        CommandResult::Ok
    ));
    render_blocks(&mut core, 2); // command block + crossfade block
    let (sine_l, _) = make_sine_pcm(PCM_LEN, 220.0);
    let gl = (std::f32::consts::FRAC_PI_4).cos(); // pan 0.0, gain 1.0
    let mut l = vec![0.0f32; BLOCK];
    let mut r = vec![0.0f32; BLOCK];
    let n0 = (50 + 30 + 100 + 2) * BLOCK as u64;
    core.render(&mut l, &mut r);
    for i in 0..BLOCK {
        let src = sine_l[((n0 + i as u64) % PCM_LEN) as usize];
        assert_eq!(
            l[i].to_bits(),
            (src * gl).to_bits(),
            "bypass not bit-exact at frame {} after pitch returned to 0",
            n0 + i as u64
        );
    }
    drop(plane);
}

// ---------------------------------------------------------------------------
// Rate clamp (§4.8)
// ---------------------------------------------------------------------------

#[test]
fn tempo_and_load_reject_out_of_range_rates() {
    let (mut plane, _core) = engine_with_sine(120_000, 220.0);

    // 120k-original track: tempo 30k would be rate 0.25 exactly (ok),
    // 29k would be < 0.25 — but both are below the engine tempo clamp
    // anyway; use the upper side: 4× of 120k is 480k > MAX_TEMPO. So
    // exercise the clamp with a second, slower track: O = 60k.
    let (l, r) = make_sine_pcm(768_000, 110.0);
    plane
        .load_pcm(
            &LoadMeta {
                sample_rate: 48_000,
                slot: 1,
                original_tempo_mbpm: 60_000,
                loop_beats: 16,
                encoder_delay: 0,
                end_padding: 0,
                original_samples: 768_000,
                variation: 0,
                gain: 1.0,
                pan: 0.0,
            },
            l,
            r,
        )
        .expect("60k track loads at tempo 120k (rate 2.0)");

    // 250k would put the 60k track at rate 4.17 — rejected, with the slot
    // named in the reason.
    let res = plane.dispatch(Command::SetTempo { mbpm: 250_000 });
    match res {
        CommandResult::Rejected { reason } => {
            assert!(reason.contains("slot 1"), "reason should name the violating slot: {reason}")
        }
        other => panic!("SetTempo 250k must be rejected with a 60k track loaded: {other:?}"),
    }
    // 240k is rate 4.0 exactly — allowed (closed interval).
    assert!(matches!(plane.dispatch(Command::SetTempo { mbpm: 240_000 }), CommandResult::Ok));

    // Loading a track whose rate at the CURRENT tempo is out of range fails.
    let (l2, r2) = make_sine_pcm(192_000, 880.0);
    let res = plane.load_pcm(
        &LoadMeta {
            sample_rate: 48_000,
            slot: 2,
            original_tempo_mbpm: 240_000, // rate at tempo 240k is 1.0 — fine…
            loop_beats: 16,
            encoder_delay: 0,
            end_padding: 0,
            original_samples: 192_000,
            variation: 0,
            gain: 1.0,
            pan: 0.0,
        },
        l2,
        r2,
    );
    assert!(res.is_ok(), "in-range load must pass: {res:?}");
    let (l3, r3) = make_sine_pcm(1_536_000, 55.0);
    let res = plane.load_pcm(
        &LoadMeta {
            sample_rate: 48_000,
            slot: 3,
            original_tempo_mbpm: 30_000, // rate at tempo 240k would be 8.0
            loop_beats: 16,
            encoder_delay: 0,
            end_padding: 0,
            original_samples: 1_536_000,
            variation: 0,
            gain: 1.0,
            pan: 0.0,
        },
        l3,
        r3,
    );
    assert!(res.is_err(), "load outside the rate clamp must fail (spec §4.8)");
}

// ---------------------------------------------------------------------------
// Quantized activation edge: exactly on a bar boundary activates immediately
// ---------------------------------------------------------------------------

#[test]
fn next_bar_activation_on_boundary_is_immediate() {
    // 120 BPM, 4/4: one bar = 4 beats = 96_000 frames. Render exactly one
    // bar, then arm with NextBar — bu is exactly on the boundary, so the
    // activation must land at the current frame, not a bar later.
    let (mut plane, mut core) = create_engine(EngineConfig {
        sample_rate: SAMPLE_RATE,
        initial_tempo_mbpm: 120_000,
        beats_per_bar: 4,
    })
    .unwrap();
    let (l, r) = make_sine_pcm(PCM_LEN, 220.0);
    plane.load_pcm(&meta(0, 0), l, r).unwrap();
    assert!(matches!(plane.dispatch(Command::Play), CommandResult::Ok));

    let bar_frames = 96_000usize;
    let mut lbuf = vec![0.0f32; BLOCK];
    let mut rbuf = vec![0.0f32; BLOCK];
    for _ in 0..bar_frames / BLOCK {
        core.render(&mut lbuf, &mut rbuf);
    }
    assert!(matches!(
        plane.dispatch(Command::Activate { slot: 0, quantize: QuantizePolicy::NextBar , transport: Default::default() }),
        CommandResult::Ok
    ));
    core.render(&mut lbuf, &mut rbuf);
    let events = plane.drain_events();
    let at = events
        .iter()
        .find_map(|e| match e {
            Event::TrackActivated { slot: 0, at_frame } => Some(*at_frame),
            _ => None,
        })
        .expect("activation event missing");
    assert_eq!(at, bar_frames as u64, "on-boundary NextBar must activate immediately");
}
