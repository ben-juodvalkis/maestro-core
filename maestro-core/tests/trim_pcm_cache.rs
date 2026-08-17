//! # FR17 memory-pressure trim tests (ADR-104)
//!
//! `Command::TrimPcmCache { target_bytes }` is a one-shot eviction of PARKED
//! (cached, non-installed) variation PCM, least-recently-audible first,
//! down to the target. Pinned here (the Phase-4 gates):
//!
//! - parked-only eviction to the target; INSTALLED PCM survives even at
//!   target 0 (it is structurally unevictable);
//! - nothing reaches the RT thread: the rendered audio is bit-unchanged
//!   across the trim, no `VariationApplied` fires, and the snapshot's
//!   current variation is untouched;
//! - the snapshot's `pcm_bytes_in_use` reflects the trim;
//! - each eviction echoes `VariationEvicted` (the backend's residency
//!   bookkeeping depends on it, ADR-103);
//! - the CONFIGURED budget is unchanged: a later park may fill the cache
//!   back up to the budget;
//! - a trimmed variation switches via the async `VariationLoadRequired`
//!   path again; a surviving one stays on the cached path.
//!
//! Same thread-free/fs-free idioms as `variation_prewarm.rs` (wasm32-clean).

mod common;

use common::{make_sine_pcm, SAMPLE_RATE};
use maestro_api::{Command, CommandResult, Event, EngineConfig, LoadMeta, QuantizePolicy, TransportPolicy, SampleFormat};
use maestro_core::audio_core::AudioCore;
use maestro_core::control_plane::{create_engine, ControlPlane};

const O_MBPM: u32 = 120_000;
const LOOP_BEATS: u32 = 16;
const PCM_LEN: u64 = 384_000;
const BLOCK: usize = 480;
const BYTES_PER_VARIATION: u64 = PCM_LEN * 2 * 4; // stereo f32

fn meta(slot: u8, variation: u8, cache_only: bool) -> LoadMeta {
    LoadMeta {
        slot,
        sample_rate: SAMPLE_RATE,
        original_tempo_mbpm: O_MBPM,
        loop_beats: LOOP_BEATS,
        encoder_delay: 0,
        end_padding: 0,
        original_samples: PCM_LEN,
        variation,
        cache_only,
        sample_format: SampleFormat::F32,
        gain: 1.0,
        pan: 0.0,
    }
}

/// Engine with variation 0 installed + playing and variations 1, 2 PARKED
/// on slot 0 (three variations resident).
fn engine_with_parks() -> (ControlPlane, Box<AudioCore>) {
    let (mut plane, core) = create_engine(EngineConfig {
        sample_rate: SAMPLE_RATE,
        initial_tempo_mbpm: O_MBPM,
        beats_per_bar: 4,
        pcm_budget_bytes: None,
    })
    .unwrap();
    let (l, r) = make_sine_pcm(PCM_LEN, 220.0);
    plane.load_pcm(&meta(0, 0, false), l, r).expect("variation 0 loads");
    assert!(matches!(
        plane.dispatch(Command::Activate {
            slot: 0,
            quantize: QuantizePolicy::Immediate,
            transport: TransportPolicy::Manual
        }),
        CommandResult::Ok
    ));
    assert!(matches!(plane.dispatch(Command::Play), CommandResult::Ok));
    let (l, r) = make_sine_pcm(PCM_LEN, 440.0);
    plane.load_pcm(&meta(0, 1, true), l, r).expect("variation 1 parks");
    let (l, r) = make_sine_pcm(PCM_LEN, 880.0);
    plane.load_pcm(&meta(0, 2, true), l, r).expect("variation 2 parks");
    plane.drain_events();
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

#[test]
fn trims_parked_entries_lru_first_down_to_target() {
    let (mut plane, _core) = engine_with_parks();
    assert_eq!(plane.pcm_bytes_in_use(), 3 * BYTES_PER_VARIATION);

    // Target = installed + one variation: exactly one park must go — the
    // least-recently-audible one (variation 1, parked before 2).
    let target = 2 * BYTES_PER_VARIATION;
    assert!(matches!(
        plane.dispatch(Command::TrimPcmCache { target_bytes: target }),
        CommandResult::Ok
    ));
    assert_eq!(plane.pcm_bytes_in_use(), target);
    let events = plane.drain_events();
    assert!(
        events
            .iter()
            .any(|e| matches!(e, Event::VariationEvicted { slot: 0, variation: 1 })),
        "the older park (variation 1) is evicted first: {events:?}"
    );
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, Event::VariationEvicted { slot: 0, variation: 2 })),
        "the newer park survives a partial trim: {events:?}"
    );
}

#[test]
fn installed_pcm_survives_a_trim_to_zero_and_rt_is_untouched() {
    let (mut plane, mut core) = engine_with_parks();
    let before = render_blocks(&mut core, 10);

    assert!(matches!(
        plane.dispatch(Command::TrimPcmCache { target_bytes: 0 }),
        CommandResult::Ok
    ));

    // Installed PCM is structurally unevictable: resident = exactly it.
    assert_eq!(plane.pcm_bytes_in_use(), BYTES_PER_VARIATION);
    // The snapshot reflects both the trim and the untouched current variation.
    let snap = plane.snapshot();
    assert_eq!(snap.pcm_bytes_in_use, BYTES_PER_VARIATION);
    assert_eq!(snap.tracks[0].variation, 0);
    // Nothing reached the RT thread: no apply events, and the rendered
    // audio continues bit-for-bit as the same installed 220 Hz loop
    // (positions derive from F, so equal blocks ⇒ untouched schedule).
    let events = plane.drain_events();
    assert!(
        !events.iter().any(|e| matches!(
            e,
            Event::VariationApplied { .. } | Event::VariationLoadRequired { .. }
        )),
        "a trim must not apply or request anything: {events:?}"
    );
    let after = render_blocks(&mut core, 10);
    assert_eq!(before.len(), after.len());
    // The loop wraps every PCM_LEN frames; blocks 10..20 continue the same
    // deterministic schedule — compare against a fresh reference engine
    // rendered 20 blocks straight.
    let (mut ref_plane, mut ref_core) = engine_with_parks();
    let reference = render_blocks(&mut ref_core, 20);
    ref_plane.drain_events();
    assert_eq!(&reference[10 * BLOCK..], &after[..], "audio must be bit-unchanged across a trim");
}

#[test]
fn trim_is_one_shot_the_configured_budget_is_unchanged() {
    let (mut plane, _core) = engine_with_parks();
    assert!(matches!(
        plane.dispatch(Command::TrimPcmCache { target_bytes: 0 }),
        CommandResult::Ok
    ));
    plane.drain_events();
    assert_eq!(plane.pcm_bytes_in_use(), BYTES_PER_VARIATION);

    // Re-parking after the pressure moment works up to the CONFIGURED
    // budget — the trim did not shrink it.
    let (l, r) = make_sine_pcm(PCM_LEN, 440.0);
    plane.load_pcm(&meta(0, 1, true), l, r).expect("re-park after trim");
    assert_eq!(plane.pcm_bytes_in_use(), 2 * BYTES_PER_VARIATION);
    let events = plane.drain_events();
    assert!(
        !events.iter().any(|e| matches!(e, Event::VariationEvicted { .. })),
        "well under budget — nothing evicts: {events:?}"
    );
}

#[test]
fn trimmed_variation_switches_async_again_survivor_stays_cached() {
    let (mut plane, _core) = engine_with_parks();
    // Trim away variation 1 (LRU), keep 2.
    assert!(matches!(
        plane.dispatch(Command::TrimPcmCache { target_bytes: 2 * BYTES_PER_VARIATION }),
        CommandResult::Ok
    ));
    plane.drain_events();

    // Survivor: cached path — applies without a load request.
    assert!(matches!(
        plane.dispatch(Command::SwitchVariation { slot: 0, variation: 2 }),
        CommandResult::Ok
    ));
    let events = plane.drain_events();
    assert!(
        !events.iter().any(|e| matches!(e, Event::VariationLoadRequired { .. })),
        "surviving park takes the cached path: {events:?}"
    );

    // Trimmed: async path — the platform is asked to reload it.
    assert!(matches!(
        plane.dispatch(Command::SwitchVariation { slot: 0, variation: 1 }),
        CommandResult::Ok
    ));
    let events = plane.drain_events();
    assert!(
        events
            .iter()
            .any(|e| matches!(e, Event::VariationLoadRequired { slot: 0, variation: 1 })),
        "trimmed variation must reload via the async path: {events:?}"
    );
}

#[test]
fn trim_above_resident_is_a_no_op() {
    let (mut plane, _core) = engine_with_parks();
    let resident = plane.pcm_bytes_in_use();
    assert!(matches!(
        plane.dispatch(Command::TrimPcmCache { target_bytes: u64::MAX }),
        CommandResult::Ok
    ));
    assert_eq!(plane.pcm_bytes_in_use(), resident);
    assert!(
        !plane
            .drain_events()
            .iter()
            .any(|e| matches!(e, Event::VariationEvicted { .. })),
        "nothing to trim — nothing evicts"
    );
}
