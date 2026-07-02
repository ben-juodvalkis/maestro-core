//! # §4.11 variation prewarm tests (ADR-098)
//!
//! `LoadMeta.cache_only` parks a variation's PCM in the slot's §4.11 cache
//! WITHOUT switching to it, so a later `SwitchVariation` takes the cached
//! path (swap at the next block boundary) instead of the async
//! `VariationLoadRequired` round-trip. Pinned here:
//!
//! - A park never touches the RT thread: no `VariationApplied`, the rendered
//!   audio and the snapshot's current variation are unchanged.
//! - A park while a switch to exactly that variation is PENDING completes the
//!   switch instead (the user asked for it; parking would strand it).
//! - A cache-only load of the INSTALLED variation is a no-op success — it
//!   must never reach the fresh-install path, which resets the slot and
//!   wipes the cache (the prewarm/switch race hazard, ADR-098).
//! - Identity + reconciliation-stability validation (spec §3.4) applies to
//!   cache-only loads unchanged.
//! - `EngineConfig.pcm_budget_bytes` overrides the 96 MiB default; parked
//!   entries are evicted least-recently-audible first, installed PCM never.

mod common;

use common::{make_sine_pcm, SAMPLE_RATE};
use maestro_api::{
    Command, CommandResult, EngineConfig, Event, LoadMeta, QuantizePolicy, TransportPolicy,
};
use maestro_core::control_plane::{create_engine, ControlPlane, EngineError, PCM_BUDGET_BYTES};
use maestro_core::audio_core::AudioCore;

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
        gain: 1.0,
        pan: 0.0,
    }
}

fn engine(budget: Option<u64>) -> (ControlPlane, Box<AudioCore>) {
    create_engine(EngineConfig {
        sample_rate: SAMPLE_RATE,
        initial_tempo_mbpm: O_MBPM,
        beats_per_bar: 4,
        pcm_budget_bytes: budget,
    })
    .unwrap()
}

/// Engine with variation 0 (`hz` sine) installed on slot 0, active, playing.
fn engine_with_sine(budget: Option<u64>, hz: f32) -> (ControlPlane, Box<AudioCore>) {
    let (mut plane, core) = engine(budget);
    let (l, r) = make_sine_pcm(PCM_LEN, hz);
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

fn measured_hz(signal: &[f32]) -> f64 {
    let crossings = signal.windows(2).filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0)).count();
    crossings as f64 / 2.0 * SAMPLE_RATE as f64 / signal.len() as f64
}

// ---------------------------------------------------------------------------
// Park semantics
// ---------------------------------------------------------------------------

#[test]
fn cache_only_parks_without_applying() {
    let (mut plane, mut core) = engine_with_sine(None, 220.0);
    render_blocks(&mut core, 10);
    plane.drain_events();

    let (l, r) = make_sine_pcm(PCM_LEN, 440.0);
    plane.load_pcm(&meta(0, 1, true), l, r).expect("prewarm parks");

    // The load is acked, but nothing is applied and nothing was requested.
    let events = plane.drain_events();
    assert!(
        events
            .iter()
            .any(|e| matches!(e, Event::TrackLoaded { slot: 0, variation: 1, .. })),
        "a park must still ack via TrackLoaded: {events:?}"
    );
    assert!(
        !events.iter().any(|e| matches!(
            e,
            Event::VariationApplied { .. } | Event::VariationLoadRequired { .. }
        )),
        "a park must not apply or request anything: {events:?}"
    );

    // RT untouched: the installed variation keeps rendering, snapshot
    // unchanged, both variations resident.
    let audio = render_blocks(&mut core, 10);
    let hz = measured_hz(&audio);
    assert!((hz - 220.0).abs() < 10.0, "installed variation must keep playing: {hz:.1}");
    let snap = plane.snapshot();
    assert_eq!(snap.tracks[0].variation, 0, "park must not change the current variation");
    assert_eq!(snap.pcm_bytes_in_use, 2 * BYTES_PER_VARIATION);
}

#[test]
fn switch_after_prewarm_takes_the_cached_path() {
    let (mut plane, mut core) = engine_with_sine(None, 220.0);
    render_blocks(&mut core, 10);
    let (l, r) = make_sine_pcm(PCM_LEN, 440.0);
    plane.load_pcm(&meta(0, 1, true), l, r).expect("prewarm parks");
    plane.drain_events();

    // The whole point: the switch is cached — applied at a block boundary,
    // no platform load round-trip.
    assert!(matches!(
        plane.dispatch(Command::SwitchVariation { slot: 0, variation: 1 }),
        CommandResult::Ok
    ));
    render_blocks(&mut core, 3);
    let audio = render_blocks(&mut core, 10);
    let hz = measured_hz(&audio);
    assert!((hz - 440.0).abs() < 12.0, "prewarmed switch must apply immediately: {hz:.1}");
    let events = plane.drain_events();
    assert!(
        !events.iter().any(|e| matches!(e, Event::VariationLoadRequired { .. })),
        "prewarmed switch must not request a load: {events:?}"
    );
    assert!(events
        .iter()
        .any(|e| matches!(e, Event::VariationApplied { slot: 0, variation: 1, .. })));
    assert_eq!(plane.snapshot().tracks[0].variation, 1);
}

#[test]
fn cache_only_same_variation_is_a_noop() {
    let (mut plane, mut core) = engine_with_sine(None, 220.0);
    render_blocks(&mut core, 10);
    let (l, r) = make_sine_pcm(PCM_LEN, 440.0);
    plane.load_pcm(&meta(0, 1, true), l, r).expect("prewarm parks variation 1");
    plane.drain_events();

    // Cache-only re-load of the INSTALLED variation: no-op success. The
    // hazard this guards: falling through to the fresh-install path would
    // reset the slot and wipe the parked variation 1.
    let (l, r) = make_sine_pcm(PCM_LEN, 220.0);
    plane.load_pcm(&meta(0, 0, true), l, r).expect("same-variation prewarm no-ops");
    let events = plane.drain_events();
    assert!(
        !events.iter().any(|e| matches!(
            e,
            Event::VariationApplied { .. }
                | Event::VariationLoadRequired { .. }
                | Event::VariationEvicted { .. }
        )),
        "same-variation park must not touch anything: {events:?}"
    );
    let snap = plane.snapshot();
    assert_eq!(snap.tracks[0].variation, 0);
    assert_eq!(
        snap.pcm_bytes_in_use,
        2 * BYTES_PER_VARIATION,
        "the parked variation must survive a same-variation prewarm (no slot reset)"
    );

    // And the parked entry is still usable: switching to it is cached.
    assert!(matches!(
        plane.dispatch(Command::SwitchVariation { slot: 0, variation: 1 }),
        CommandResult::Ok
    ));
    let events = plane.drain_events();
    assert!(
        !events.iter().any(|e| matches!(e, Event::VariationLoadRequired { .. })),
        "cache must have survived: {events:?}"
    );
}

#[test]
fn cache_only_on_empty_slot_is_rejected() {
    let (mut plane, _core) = engine(None);
    let (l, r) = make_sine_pcm(PCM_LEN, 220.0);
    let err = plane.load_pcm(&meta(3, 1, true), l, r).expect_err("nothing to park against");
    assert!(err.contains("cache-only load requires an installed track"), "got: {err}");
    let events = plane.drain_events();
    assert!(
        events
            .iter()
            .any(|e| matches!(e, Event::TrackLoadFailed { slot: 3, .. })),
        "the failure must be surfaced: {events:?}"
    );
}

#[test]
fn cache_only_enforces_identity_and_reconciliation() {
    let (mut plane, _core) = engine_with_sine(None, 220.0);

    // Identity mismatch fails a prewarm exactly like a switch-completion
    // load would (spec §3.4). The incoming meta must be SELF-consistent
    // (loop identity validates before the cross-variation check), so declare
    // a different but coherent identity over the same sample count:
    // 8 beats @ 60 BPM @ 48 kHz = 384_000 samples, vs installed 16 @ 120.
    let mut bad_identity = meta(0, 1, true);
    bad_identity.original_tempo_mbpm = 60_000;
    bad_identity.loop_beats = 8;
    let (l, r) = make_sine_pcm(PCM_LEN, 440.0);
    let err = plane.load_pcm(&bad_identity, l, r).expect_err("identity mismatch fails");
    assert!(err.contains("identity mismatch"), "got: {err}");

    // Reconciliation-stability mismatch: the installed stem reconciled as
    // AlreadyTrimmed (delay/padding 0); an incoming raw decode (TrimmedHere)
    // flips the case — mis-tagged file, hard failure (spec §3.4#4).
    let mut flipped = meta(0, 1, true);
    flipped.encoder_delay = 2112;
    flipped.end_padding = 960;
    let raw_len = PCM_LEN + 2112 + 960;
    let (l, r) = make_sine_pcm(raw_len, 440.0);
    let err = plane.load_pcm(&flipped, l, r).expect_err("reconciliation flip fails");
    assert!(err.contains("reconciled as"), "got: {err}");

    // Nothing was parked by either failure.
    assert_eq!(plane.snapshot().pcm_bytes_in_use, BYTES_PER_VARIATION);
}

#[test]
fn cache_only_completes_a_pending_switch() {
    let (mut plane, mut core) = engine_with_sine(None, 220.0);
    render_blocks(&mut core, 10);

    // User switches to a not-yet-loaded variation: async protocol starts.
    assert!(matches!(
        plane.dispatch(Command::SwitchVariation { slot: 0, variation: 1 }),
        CommandResult::Ok
    ));
    let events = plane.drain_events();
    assert!(events
        .iter()
        .any(|e| matches!(e, Event::VariationLoadRequired { slot: 0, variation: 1 })));

    // A prewarm for exactly that variation lands while the switch is
    // pending: it must COMPLETE the switch, not park (parking would strand
    // the user's request behind a duplicate load).
    let (l, r) = make_sine_pcm(PCM_LEN, 440.0);
    plane.load_pcm(&meta(0, 1, true), l, r).expect("prewarm completes the pending switch");
    render_blocks(&mut core, 3);
    let audio = render_blocks(&mut core, 10);
    let hz = measured_hz(&audio);
    assert!((hz - 440.0).abs() < 12.0, "pending switch must complete audibly: {hz:.1}");
    let events = plane.drain_events();
    assert!(events
        .iter()
        .any(|e| matches!(e, Event::VariationApplied { slot: 0, variation: 1, .. })));
    assert_eq!(plane.snapshot().tracks[0].variation, 1);

    // The dedupe-loser simulation: a straggler cache-only load of the SAME
    // (now installed) variation no-ops instead of resetting the slot.
    let (l, r) = make_sine_pcm(PCM_LEN, 440.0);
    plane.load_pcm(&meta(0, 1, true), l, r).expect("straggler no-ops");
    let snap = plane.snapshot();
    assert_eq!(snap.tracks[0].variation, 1);
    assert_eq!(
        snap.pcm_bytes_in_use,
        2 * BYTES_PER_VARIATION,
        "the straggler must not wipe the cache (variation 0 stays resident)"
    );
}

// ---------------------------------------------------------------------------
// Configurable budget
// ---------------------------------------------------------------------------

#[test]
fn configured_budget_governs_prewarm_eviction() {
    // Budget fits the installed variation + ONE parked entry, not two.
    let budget = 2 * BYTES_PER_VARIATION + BYTES_PER_VARIATION / 2;
    assert!(budget < PCM_BUDGET_BYTES, "this test must exercise the OVERRIDE");
    let (mut plane, mut core) = engine_with_sine(Some(budget), 220.0);
    render_blocks(&mut core, 4);

    let (l, r) = make_sine_pcm(PCM_LEN, 440.0);
    plane.load_pcm(&meta(0, 1, true), l, r).expect("variation 1 parks");
    plane.drain_events();
    let (l, r) = make_sine_pcm(PCM_LEN, 880.0);
    plane.load_pcm(&meta(0, 2, true), l, r).expect("variation 2 parks");

    // Least-recently-audible parked entry (variation 1 — older stamp, never
    // audible) is evicted; the installed variation is untouchable.
    let events = plane.drain_events();
    assert!(
        events
            .iter()
            .any(|e| matches!(e, Event::VariationEvicted { slot: 0, variation: 1 })),
        "over-budget park must evict the oldest parked entry: {events:?}"
    );
    let snap = plane.snapshot();
    assert!(snap.pcm_bytes_in_use <= budget, "configured budget enforced");
    assert_eq!(snap.tracks[0].variation, 0, "installed variation never evicted");

    // Observable consequences: variation 2 is cached, variation 1 is gone.
    assert!(matches!(
        plane.dispatch(Command::SwitchVariation { slot: 0, variation: 2 }),
        CommandResult::Ok
    ));
    let events = plane.drain_events();
    assert!(
        !events.iter().any(|e| matches!(e, Event::VariationLoadRequired { .. })),
        "variation 2 must be cached: {events:?}"
    );
    assert!(matches!(
        plane.dispatch(Command::SwitchVariation { slot: 0, variation: 1 }),
        CommandResult::Ok
    ));
    let events = plane.drain_events();
    assert!(
        events
            .iter()
            .any(|e| matches!(e, Event::VariationLoadRequired { slot: 0, variation: 1 })),
        "the evicted variation must load async again: {events:?}"
    );
}

#[test]
fn omitted_budget_equals_the_default_constant() {
    // None behaves exactly like the historical constant: under 96 MiB no
    // eviction fires for two resident variations of the standard test stem.
    let (mut plane, _core) = engine_with_sine(None, 220.0);
    let (l, r) = make_sine_pcm(PCM_LEN, 440.0);
    plane.load_pcm(&meta(0, 1, true), l, r).expect("parks");
    let events = plane.drain_events();
    assert!(
        !events.iter().any(|e| matches!(e, Event::VariationEvicted { .. })),
        "2 × {BYTES_PER_VARIATION} B is far under the 96 MiB default: {events:?}"
    );
    assert_eq!(plane.snapshot().pcm_bytes_in_use, 2 * BYTES_PER_VARIATION);
}

#[test]
fn zero_budget_config_rejected() {
    let result = create_engine(EngineConfig {
        sample_rate: SAMPLE_RATE,
        initial_tempo_mbpm: O_MBPM,
        beats_per_bar: 4,
        pcm_budget_bytes: Some(0),
    });
    let Err(err) = result else {
        panic!("a zero budget must be rejected, never clamped");
    };
    assert!(matches!(err, EngineError::InvalidConfig { .. }), "got: {err:?}");
}
