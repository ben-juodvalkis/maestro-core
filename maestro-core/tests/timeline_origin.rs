//! # SetTimelineOrigin (ADR-113) — epoch re-base with a nonzero musical origin
//!
//! The invariant under test: an origin is nothing but the Stop epoch reset
//! with `start_bu = bu` — positions stay derived, so a track activated from
//! timeline start ENTERS its PCM at `u(bu)`, bit-identically to the same
//! content played from zero once the activation ramp has passed.

mod common;

use common::{make_sine_pcm, SAMPLE_RATE};
use maestro_api::{Command, CommandResult, EngineConfig, LoadMeta, QuantizePolicy, SampleFormat};
use maestro_core::audio_core::{AudioCore, ACTIVATION_RAMP_FRAMES};
use maestro_core::control_plane::{create_engine, ControlPlane};
use maestro_core::timeline::BEAT_UNITS_PER_BEAT;

const O_120: u32 = 120_000;
const BEATS_16: u32 = 16;
const PCM_120: u64 = 384_000; // 16 beats @ 120 BPM @ 48 kHz, integer loop

fn engine_with_track() -> (ControlPlane, Box<AudioCore>) {
    let (mut plane, core) = create_engine(EngineConfig {
        sample_rate: SAMPLE_RATE,
        initial_tempo_mbpm: O_120,
        beats_per_bar: 4,
        pcm_budget_bytes: None,
    })
    .expect("engine config valid");
    let (l, r) = make_sine_pcm(PCM_120, 330.0);
    plane
        .load_pcm(
            &LoadMeta {
                slot: 0,
                sample_rate: SAMPLE_RATE,
                original_tempo_mbpm: O_120,
                loop_beats: BEATS_16,
                encoder_delay: 0,
                end_padding: 0,
                original_samples: PCM_120,
                variation: 0,
                cache_only: false, sample_format: SampleFormat::F32,
                gain: 1.0,
                pan: 0.0,
            },
            l,
            r,
        )
        .expect("track loads");
    (plane, core)
}

fn ok(result: CommandResult) {
    assert!(matches!(result, CommandResult::Ok), "rejected: {result:?}");
}

fn activate_and_play(plane: &mut ControlPlane) {
    ok(plane.dispatch(Command::Activate {
        slot: 0,
        quantize: QuantizePolicy::Immediate,
        transport: Default::default(),
    }));
    ok(plane.dispatch(Command::Play));
}

fn render(core: &mut AudioCore, frames: usize) -> (Vec<f32>, Vec<f32>) {
    let mut out_l = vec![0.0f32; frames];
    let mut out_r = vec![0.0f32; frames];
    for at in (0..frames).step_by(128) {
        let b = 128.min(frames - at);
        core.render(&mut out_l[at..at + b], &mut out_r[at..at + b]);
    }
    (out_l, out_r)
}

/// Positions derive from the origin: at unity, dictated position after F
/// rendered frames is exactly `bu/O + F`.
#[test]
fn origin_offsets_derived_positions_exactly() {
    let (mut plane, mut core) = engine_with_track();
    // 2.5 beats in: bu = 2.5·K, entry sample = bu/O = 2.5 × 24_000 = 60_000.
    let bu = (5 * BEAT_UNITS_PER_BEAT / 2) as u64;
    ok(plane.dispatch(Command::SetTimelineOrigin { bu }));
    activate_and_play(&mut plane);
    let mut l = vec![0.0f32; 128];
    let mut r = vec![0.0f32; 128];
    let mut rendered = 0u64;
    for _ in 0..100 {
        core.render(&mut l, &mut r);
        rendered += 128;
        let expect = bu as u128 / O_120 as u128 + rendered as u128;
        assert_eq!(
            core.dictated_unwrapped_position(0).unwrap() as u128,
            expect,
            "position must be u(origin) + F at unity"
        );
    }
    // The snapshot reflects the origin: 2.5 beats (origin) + 12_800/24_000
    // beats rendered = 3.033 elapsed → bar 1, beatInBar 4.033 (1-based).
    let snap = plane.snapshot();
    assert_eq!(snap.bar, 1);
    assert!((snap.beat_in_bar - 4.0333).abs() < 0.001, "beatInBar {}", snap.beat_in_bar);
}

/// The audio itself enters mid-PCM: past the activation ramp, an origin run
/// is byte-exact with the same content played from zero at the same PCM
/// offset (positions are computed; entry is not a special case).
#[test]
fn origin_entry_audio_is_byte_exact_past_the_ramp() {
    let bu = (4 * BEAT_UNITS_PER_BEAT) as u64; // 4 beats = 96_000 samples in
    let entry = (bu / O_120 as u64) as usize;
    let frames = 48_000usize;

    let (mut plane_a, mut core_a) = engine_with_track();
    ok(plane_a.dispatch(Command::SetTimelineOrigin { bu }));
    activate_and_play(&mut plane_a);
    let (a_l, _) = render(&mut core_a, frames);

    let (mut plane_b, mut core_b) = engine_with_track();
    activate_and_play(&mut plane_b);
    let (b_l, _) = render(&mut core_b, entry + frames);

    let ramp = ACTIVATION_RAMP_FRAMES as usize;
    assert_eq!(
        a_l[ramp..frames]
            .iter()
            .map(|s| s.to_bits())
            .collect::<Vec<_>>(),
        b_l[entry + ramp..entry + frames]
            .iter()
            .map(|s| s.to_bits())
            .collect::<Vec<_>>(),
        "origin entry must be byte-exact with the from-zero run past the ramp"
    );
    let peak = a_l.iter().fold(0.0f32, |m, s| m.max(s.abs()));
    assert!(peak > 0.0, "anti-stub: origin run rendered silence");
}

/// Wrap bookkeeping re-bases on the origin: entering near the loop tail
/// wraps once shortly after start, with no stale LoopWrapped burst.
#[test]
fn origin_near_loop_tail_wraps_cleanly() {
    let (mut plane, mut core) = engine_with_track();
    // 15.5 beats in — half a beat (12_000 frames) before the wrap.
    let bu = (31 * BEAT_UNITS_PER_BEAT / 2) as u64;
    ok(plane.dispatch(Command::SetTimelineOrigin { bu }));
    activate_and_play(&mut plane);
    let _ = render(&mut core, 24_000);
    let wraps: Vec<u32> = plane
        .drain_events()
        .into_iter()
        .filter_map(|e| match e {
            maestro_api::Event::LoopWrapped { wrap_count, .. } => Some(wrap_count),
            _ => None,
        })
        .collect();
    assert_eq!(wraps, vec![1], "exactly one wrap, correctly numbered");
    // And the position keeps deriving: 24_000 frames past 372_000 = 12_000
    // into wrap 1 → unwrapped 396_000.
    assert_eq!(core.dictated_unwrapped_position(0).unwrap(), 396_000);
}

/// The §3.6 guard rails: rejected while the transport is playing; a zero
/// origin behaves exactly like the status quo.
#[test]
fn origin_rejected_while_playing_and_zero_is_identity() {
    let (mut plane, mut core) = engine_with_track();
    activate_and_play(&mut plane);
    let mut l = vec![0.0f32; 128];
    let mut r = vec![0.0f32; 128];
    core.render(&mut l, &mut r); // snapshot now says playing
    let result = plane.dispatch(Command::SetTimelineOrigin { bu: 42 });
    assert!(
        matches!(result, CommandResult::Rejected { .. }),
        "origin must reject on a playing transport"
    );

    let (mut plane_z, mut core_z) = engine_with_track();
    ok(plane_z.dispatch(Command::SetTimelineOrigin { bu: 0 }));
    activate_and_play(&mut plane_z);
    core_z.render(&mut l, &mut r);
    assert_eq!(core_z.dictated_unwrapped_position(0).unwrap(), 128);
}

/// Regression (review finding): the stopped-transport validation must track
/// DISPATCH order, not the render-refreshed snapshot. A Play immediately
/// followed by SetTimelineOrigin — no render in between, so the snapshot
/// still says stopped — used to be accepted here and then dropped SILENTLY
/// by the RT arm (queue order applies Play first): an Ok the engine never
/// honored. It must reject at dispatch. A Stop clears the flag just as
/// synchronously: stop-then-origin with no render is accepted.
#[test]
fn origin_rejects_after_unrendered_play_and_accepts_after_unrendered_stop() {
    let (mut plane, mut core) = engine_with_track();
    activate_and_play(&mut plane);
    // NO render between Play and the origin dispatch — the snapshot is stale.
    let result = plane.dispatch(Command::SetTimelineOrigin { bu: 42 });
    assert!(
        matches!(result, CommandResult::Rejected { .. }),
        "origin after an unrendered Play must reject, got {result:?}"
    );

    // Stop, still with no render: the mirror clears in dispatch order, and
    // the RT queue applies Play → Stop → origin — stopped when it lands.
    ok(plane.dispatch(Command::Stop));
    ok(plane.dispatch(Command::SetTimelineOrigin { bu: 42 }));
    let mut l = vec![0.0f32; 128];
    let mut r = vec![0.0f32; 128];
    core.render(&mut l, &mut r); // drain the queue; transport ends stopped
    assert!(l.iter().all(|s| *s == 0.0), "stopped transport renders silence");
}
