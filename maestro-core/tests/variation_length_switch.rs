//! # Per-variation loop lengths (ADR-109)
//!
//! Variations of a stem may declare DIFFERENT integer loop lengths (e.g.
//! simple = 1 bar, default = 4, complex = 24); the shared identity is the
//! original tempo `O` (plus §3.4#4 reconciliation stability). A length-
//! changing switch swaps PCM + timing map atomically and the incoming
//! variation resumes at the derived position `bu(F) mod Λ_new` — as if it
//! had been playing since timeline zero. Pinned here:
//!
//! - (a)/(b) Post-crossfade audio after a short→long (and long→short) switch
//!   is BIT-IDENTICAL to a reference engine that had the target variation
//!   installed from frame 0 — the derived-resume semantics, tolerance zero.
//! - (c) A different-length park + cached switch behaves identically (no
//!   `VariationLoadRequired` round-trip).
//! - (d) The crossfade block reads the OUTGOING PCM through the OUTGOING
//!   map: hand-computed equal-power mix matches bit-for-bit. A regression
//!   that reads the old PCM through the new (longer) map indexes the old
//!   channel `Vec` out of bounds — this test is the panic tripwire.
//! - (e) `LoopWrapped`/`loop_count` re-base across a length change: no stale
//!   event at the swap block; the next event fires at the first genuine
//!   Λ_new boundary; snapshot `loop_beats`/`loop_count` flip together.
//! - (f) Length switches under stretch (non-unity rates) stay seamless:
//!   derived positions match the exact reference math, no discontinuity
//!   above the crossfade bound, and a tempo change immediately after the
//!   switch composes cleanly.
//! - (g) Heterogeneous per-variation byte footprints are accounted per
//!   entry; `TrimPcmCache` evicts least-recently-audible first by ACTUAL
//!   size.
//!
//! Same thread-free/fs-free idioms as `variation_prewarm.rs` (wasm32-clean).
//! Geometry: O = 120 BPM → exactly 24 000 samples/beat at 48 kHz, so every
//! variation's Λ/O is integral and wraps land on exact sample boundaries.

mod common;

use common::{make_sine_pcm, SAMPLE_RATE};
use maestro_api::{Command, CommandResult, EngineConfig, Event, LoadMeta, QuantizePolicy, TransportPolicy, SampleFormat};
use maestro_core::audio_core::AudioCore;
use maestro_core::control_plane::{create_engine, ControlPlane};

const O_MBPM: u32 = 120_000;
const BLOCK: usize = 480;
/// Variation loop lengths in beats: 1 / 4 / 24 bars at 4/4 — the motivating
/// Cups-Drums geometry (natural bounces 4/16/96 beats).
const VAR_BEATS: [u32; 3] = [4, 16, 96];
/// 24 000 samples/beat at O = 120 BPM: Λ/O is exactly integral.
const VAR_LEN: [u64; 3] = [96_000, 384_000, 2_304_000];
const VAR_HZ: [f32; 3] = [220.0, 330.0, 440.0];

fn var_bytes(v: usize) -> u64 {
    VAR_LEN[v] * 2 * 4 // stereo f32
}

fn meta(slot: u8, variation: u8, cache_only: bool) -> LoadMeta {
    LoadMeta {
        slot,
        sample_rate: SAMPLE_RATE,
        original_tempo_mbpm: O_MBPM,
        loop_beats: VAR_BEATS[variation as usize],
        encoder_delay: 0,
        end_padding: 0,
        original_samples: VAR_LEN[variation as usize],
        variation,
        cache_only,
        sample_format: SampleFormat::F32,
        gain: 1.0,
        pan: 0.0,
    }
}

fn pcm(variation: u8) -> (Vec<f32>, Vec<f32>) {
    make_sine_pcm(VAR_LEN[variation as usize], VAR_HZ[variation as usize])
}

/// Engine at `tempo` with `variation` installed on slot 0, active, playing.
fn engine_with(tempo_mbpm: u32, variation: u8) -> (ControlPlane, Box<AudioCore>) {
    let (mut plane, core) = create_engine(EngineConfig {
        sample_rate: SAMPLE_RATE,
        initial_tempo_mbpm: tempo_mbpm,
        beats_per_bar: 4,
        pcm_budget_bytes: None,
    })
    .unwrap();
    let (l, r) = pcm(variation);
    plane.load_pcm(&meta(0, variation, false), l, r).expect("variation installs");
    assert!(matches!(
        plane.dispatch(Command::Activate {
            slot: 0,
            quantize: QuantizePolicy::Immediate,
            transport: TransportPolicy::Manual
        }),
        CommandResult::Ok
    ));
    assert!(matches!(plane.dispatch(Command::Play), CommandResult::Ok));
    (plane, core)
}

fn render_blocks(core: &mut AudioCore, n: usize) -> Vec<f32> {
    let mut l = vec![0.0f32; BLOCK];
    let mut r = vec![0.0f32; BLOCK];
    let mut out = Vec::with_capacity(n * BLOCK);
    for _ in 0..n {
        core.render(&mut l, &mut r);
        out.extend_from_slice(&l);
    }
    out
}

fn rms(signal: &[f32]) -> f32 {
    (signal.iter().map(|s| s * s).sum::<f32>() / signal.len() as f32).sqrt()
}

/// Post-crossfade audio after a length-changing switch must be bit-identical
/// to a reference engine that had the TARGET variation installed from frame 0
/// (positions are derived from the shared F; the map swap only changes which
/// content the same `u(bu)` addresses).
fn assert_switch_bitexact_vs_from_start(from: u8, to: u8, pre_blocks: usize, post_blocks: usize) {
    let (mut plane, mut core) = engine_with(O_MBPM, from);
    render_blocks(&mut core, pre_blocks);
    plane.drain_events();

    let (l, r) = pcm(to);
    plane.load_pcm(&meta(0, to, false), l, r).expect("different-length variation swaps in");
    render_blocks(&mut core, 1); // the crossfade block — excluded from the compare
    let after = render_blocks(&mut core, post_blocks);

    let events = plane.drain_events();
    assert!(
        events
            .iter()
            .any(|e| matches!(e, Event::VariationApplied { slot: 0, variation, .. } if *variation == to)),
        "the swap must be acked: {events:?}"
    );
    let snap = plane.snapshot();
    assert_eq!(snap.tracks[0].variation, to);
    assert_eq!(
        snap.tracks[0].loop_beats, VAR_BEATS[to as usize],
        "snapshot loop_beats must flip with the variation"
    );

    let (_ref_plane, mut ref_core) = engine_with(O_MBPM, to);
    let reference = render_blocks(&mut ref_core, pre_blocks + 1 + post_blocks);
    assert_eq!(
        &reference[(pre_blocks + 1) * BLOCK..],
        &after[..],
        "post-fade audio must be bit-identical to the from-start reference \
         ({from}→{to}, {} beats → {} beats)",
        VAR_BEATS[from as usize],
        VAR_BEATS[to as usize],
    );
    // Anti-stub: the compared region is audible signal, not silence.
    assert!(rms(&after) > 0.05, "compared region must be audible: rms {}", rms(&after));
}

#[test]
fn switch_to_longer_is_bitexact_vs_long_from_start() {
    // 100 blocks in (frame 48 000, mid short-loop), then 300 blocks compared.
    assert_switch_bitexact_vs_from_start(0, 2, 100, 300);
}

#[test]
fn switch_back_to_shorter_is_bitexact_vs_short_from_start() {
    // The 300 compared blocks (144 000 frames) cross the 96 000-frame short
    // loop's wrap — Bresenham wrap equality is part of the compare.
    assert_switch_bitexact_vs_from_start(2, 0, 100, 300);
}

#[test]
fn park_then_cached_switch_with_different_length() {
    let (mut plane, mut core) = engine_with(O_MBPM, 0);
    render_blocks(&mut core, 10);

    // Park the 24-bar variation (different length, same O) — must not apply.
    let (l, r) = pcm(2);
    plane.load_pcm(&meta(0, 2, true), l, r).expect("different-length park succeeds");
    let events = plane.drain_events();
    assert!(
        events.iter().any(|e| matches!(e, Event::TrackLoaded { slot: 0, variation: 2, .. })),
        "park acks via TrackLoaded: {events:?}"
    );
    assert!(
        !events.iter().any(|e| matches!(
            e,
            Event::VariationApplied { .. } | Event::VariationLoadRequired { .. }
        )),
        "a park must not apply or request anything: {events:?}"
    );
    assert_eq!(plane.pcm_bytes_in_use(), var_bytes(0) + var_bytes(2));

    render_blocks(&mut core, 90); // 100 blocks total before the switch

    // The switch takes the cached path (no platform round-trip) and applies
    // the parked entry's OWN map atomically with its PCM.
    assert!(matches!(
        plane.dispatch(Command::SwitchVariation { slot: 0, variation: 2 }),
        CommandResult::Ok
    ));
    render_blocks(&mut core, 1); // crossfade block
    let after = render_blocks(&mut core, 300);
    let events = plane.drain_events();
    assert!(
        !events.iter().any(|e| matches!(e, Event::VariationLoadRequired { .. })),
        "cached switch must not request a load: {events:?}"
    );
    assert!(events
        .iter()
        .any(|e| matches!(e, Event::VariationApplied { slot: 0, variation: 2, .. })));

    let (_ref_plane, mut ref_core) = engine_with(O_MBPM, 2);
    let reference = render_blocks(&mut ref_core, 401);
    assert_eq!(
        &reference[101 * BLOCK..],
        &after[..],
        "cached-path switch must land on the identical derived schedule"
    );
}

#[test]
fn crossfade_reads_outgoing_through_outgoing_map() {
    // Play the 4-beat (96 000-sample) variation well PAST its PCM length in
    // unwrapped samples: at the swap frame, u(bu) = 120 000 > 96 000. A
    // regression that reads the outgoing PCM through the incoming 24-bar map
    // (pcm_len 2 304 000) computes pcm_index 120 000+ and indexes the old
    // 96 000-sample channel Vec out of bounds — an immediate panic here.
    let (a_l, _) = pcm(0);
    let (c_l, _) = pcm(2);
    let (mut plane, mut core) = engine_with(O_MBPM, 0);
    render_blocks(&mut core, 250); // frame 120 000 at unity
    plane.drain_events();

    let (l, r) = pcm(2);
    plane.load_pcm(&meta(0, 2, false), l, r).expect("swap to the long variation");
    let fade = render_blocks(&mut core, 1);

    // Hand-computed expectation, mirroring the render mix bit-for-bit:
    // gain 1 / pan 0 → gl = cos(π/4); ramp done; applied_mix converged, so
    // c_l = (new·g_in + old·g_out) · gl with g_in/g_out the equal-power pair.
    // Both maps are exactly integral here (Λ/O = pcm_len), so pcm_index is a
    // plain modulo: old = A[(120 000 + i) % 96 000], new = C[120 000 + i].
    let gl = std::f32::consts::FRAC_PI_4.cos() * 1.0;
    let inv_b = 1.0 / BLOCK as f32;
    for (i, &got) in fade.iter().enumerate() {
        let t = (i as f32 + 0.5) * inv_b * std::f32::consts::FRAC_PI_2;
        let (g_in, g_out) = (t.sin(), t.cos());
        let old = a_l[(120_000 + i) % 96_000];
        let new = c_l[120_000 + i];
        let expected = (new * g_in + old * g_out) * gl;
        assert_eq!(
            got.to_bits(),
            expected.to_bits(),
            "crossfade sample {i}: got {got}, expected {expected} — the fade-out must read \
             the OUTGOING PCM through the OUTGOING map"
        );
    }

    // And the block after the fade is pure new content at the derived index.
    let after = render_blocks(&mut core, 1);
    for (i, &got) in after.iter().enumerate() {
        let expected = c_l[120_480 + i] * gl;
        assert_eq!(got.to_bits(), expected.to_bits(), "post-fade sample {i} must be new content");
    }
}

#[test]
fn wrap_events_rebase_across_switch() {
    // 24-bar variation installed; the 1-bar variation parked. At frame
    // 240 000 (bu/Λ_short = 2.5) switch long→short: without the handler
    // re-base, last_wrap_count is still the long map's 0 and the swap block
    // fires a stale LoopWrapped(2).
    let (mut plane, mut core) = engine_with(O_MBPM, 2);
    render_blocks(&mut core, 1);
    let (l, r) = pcm(0);
    plane.load_pcm(&meta(0, 0, true), l, r).expect("short variation parks");
    render_blocks(&mut core, 499); // 500 blocks total → frame 240 000
    let snap = plane.snapshot();
    assert_eq!(snap.tracks[0].loop_beats, VAR_BEATS[2]);
    assert_eq!(snap.tracks[0].loop_count, 0, "the 24-bar loop has not wrapped yet");
    plane.drain_events();

    assert!(matches!(
        plane.dispatch(Command::SwitchVariation { slot: 0, variation: 0 }),
        CommandResult::Ok
    ));
    render_blocks(&mut core, 1); // swap block: frames 240 000..240 480
    let events = plane.drain_events();
    assert!(
        events.iter().any(|e| matches!(e, Event::VariationApplied { slot: 0, variation: 0, .. })),
        "swap acked: {events:?}"
    );
    assert!(
        !events.iter().any(|e| matches!(e, Event::LoopWrapped { .. })),
        "no stale LoopWrapped at the swap block — the count re-bases: {events:?}"
    );

    // The next genuine Λ_short boundary is frame 288 000 (wrap 3). 100 more
    // blocks reach exactly frame 288 480.
    render_blocks(&mut core, 100);
    let events = plane.drain_events();
    let wraps: Vec<u32> = events
        .iter()
        .filter_map(|e| match e {
            Event::LoopWrapped { slot: 0, wrap_count } => Some(*wrap_count),
            _ => None,
        })
        .collect();
    assert_eq!(
        wraps,
        vec![3],
        "exactly one wrap event at the first genuine short-loop boundary: {events:?}"
    );
    let snap = plane.snapshot();
    assert_eq!(snap.tracks[0].loop_beats, VAR_BEATS[0], "loop_beats flips with the variation");
    assert_eq!(snap.tracks[0].loop_count, 3, "loop_count re-based onto the short map");
}

#[test]
fn length_switch_under_stretch_is_seamless() {
    // Non-unity rates engage the stretch path: the incoming side re-anchors
    // (one-block seek smear) under the equal-power fade. A from-start
    // bit-compare is invalid here (the reference's stretcher never
    // re-anchored), so assert the derived invariants instead: dictated
    // unwrapped positions match the exact reference math across the switch
    // (they are Λ-independent), the fade block has no discontinuity above
    // the crossfade bound, and the snapshot re-bases onto the new map.
    for tempo in [90_000u32, 180_000] {
        let (mut plane, mut core) = engine_with(tempo, 1);
        render_blocks(&mut core, 100);
        let (l, r) = pcm(2);
        plane.load_pcm(&meta(0, 2, true), l, r).expect("long variation parks");
        plane.drain_events();

        assert!(matches!(
            plane.dispatch(Command::SwitchVariation { slot: 0, variation: 2 }),
            CommandResult::Ok
        ));
        let fade_and_after = {
            let mut out = render_blocks(&mut core, 1);
            out.extend(render_blocks(&mut core, 99));
            out
        };
        let events = plane.drain_events();
        assert!(
            events
                .iter()
                .any(|e| matches!(e, Event::VariationApplied { slot: 0, variation: 2, .. })),
            "tempo {tempo}: swap acked: {events:?}"
        );

        // Derived positions are exact at every rate: u = floor(bu/O) with
        // bu = frame·E (single segment) — Λ-independent, so the switch must
        // not move them.
        let frames = 200 * BLOCK as u64;
        let expected_u = (frames as u128 * tempo as u128 / O_MBPM as u128) as u64;
        assert_eq!(
            core.dictated_unwrapped_position(0),
            Some(expected_u),
            "tempo {tempo}: dictated position must match the exact reference math"
        );

        // Anti-stub + seam bound: audible, and no step above the crossfade
        // envelope (sine content at ±0.177 post-gain steps ~0.008/sample;
        // a naked splice discontinuity would step up to ~0.35).
        assert!(rms(&fade_and_after) > 0.03, "tempo {tempo}: must be audible");
        let max_step = fade_and_after
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_step < 0.1,
            "tempo {tempo}: discontinuity {max_step} above the crossfade bound"
        );

        // Snapshot re-based onto the new map.
        let snap = plane.snapshot();
        assert_eq!(snap.tracks[0].loop_beats, VAR_BEATS[2]);
        let lambda = VAR_BEATS[2] as u128 * 2_880_000_000u128;
        let bu = frames as u128 * tempo as u128;
        let expected_progress = ((bu % lambda) as f64 / lambda as f64) as f32;
        assert!(
            (snap.tracks[0].loop_progress - expected_progress).abs() < 1e-3,
            "tempo {tempo}: loop_progress {} vs expected {expected_progress}",
            snap.tracks[0].loop_progress
        );

        // Sanity leg: a tempo change right after a length switch composes —
        // the ADR-094 re-anchor fade and the swap fade are separate blocks.
        let other = if tempo == 90_000 { 180_000 } else { 90_000 };
        assert!(matches!(plane.dispatch(Command::SetTempo { mbpm: other }), CommandResult::Ok));
        let post_tempo = render_blocks(&mut core, 50);
        let events = plane.drain_events();
        assert!(
            events.iter().any(|e| matches!(e, Event::TempoApplied { mbpm, .. } if *mbpm == other)),
            "tempo {tempo}→{other}: echo expected: {events:?}"
        );
        assert!(rms(&post_tempo) > 0.03, "tempo {tempo}→{other}: still audible");
    }
}

#[test]
fn heterogeneous_sizes_account_per_entry() {
    // Installed 1-bar (768 000 B) + parked 4-bar (3 072 000 B) + parked
    // 24-bar (18 432 000 B): the budget math must use each entry's ACTUAL
    // footprint, not any per-stem assumption.
    let (mut plane, mut core) = engine_with(O_MBPM, 0);
    render_blocks(&mut core, 4);
    let (l, r) = pcm(1);
    plane.load_pcm(&meta(0, 1, true), l, r).expect("4-bar variation parks");
    let (l, r) = pcm(2);
    plane.load_pcm(&meta(0, 2, true), l, r).expect("24-bar variation parks");
    plane.drain_events();
    assert_eq!(plane.pcm_bytes_in_use(), var_bytes(0) + var_bytes(1) + var_bytes(2));

    // Trim to installed + the 24-bar park: evicting the LRU 4-bar entry
    // (parked first, never audible) reaches the target exactly; the newer,
    // 6× larger park survives.
    let target = var_bytes(0) + var_bytes(2);
    assert!(matches!(
        plane.dispatch(Command::TrimPcmCache { target_bytes: target }),
        CommandResult::Ok
    ));
    assert_eq!(plane.pcm_bytes_in_use(), target);
    let events = plane.drain_events();
    assert!(
        events.iter().any(|e| matches!(e, Event::VariationEvicted { slot: 0, variation: 1 })),
        "LRU entry (variation 1) evicted: {events:?}"
    );
    assert!(
        !events.iter().any(|e| matches!(e, Event::VariationEvicted { slot: 0, variation: 2 })),
        "the newer park survives regardless of its larger size: {events:?}"
    );
}
