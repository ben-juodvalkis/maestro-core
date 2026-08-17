//! Repro attempt for the reported bug: Lagos playing (autostart), switch the
//! guitar's complexity in Mix mode → the stem goes silent until toggled
//! off/on. Drives the engine exactly as the web host does: multi-track
//! Lagos geometry, TrimmedPadding-shaped loads (the Chrome decode case),
//! `Activate{immediate, auto}` autostart, 128-frame worklet quanta, then a
//! `SwitchVariation` answered async with interleaved FR15 micro-warm parks.

mod common;

use common::SAMPLE_RATE;
use maestro_api::{Command, CommandResult, EngineConfig, Event, LoadMeta, QuantizePolicy, TransportPolicy, SampleFormat};
use maestro_core::audio_core::AudioCore;
use maestro_core::control_plane::{create_engine, ControlPlane};

const O_MBPM: u32 = 100_000; // Lagos: 100 BPM
const BLOCK: usize = 128; // worklet quantum
const ENCODER_DELAY: u32 = 2112;
const END_PADDING: u32 = 960;

/// Lagos stems: (loop_beats, original_samples, sine hz per variation base).
const STEMS: [(u32, u64, f32); 5] = [
    (160, 4_608_000, 200.0), // Pad
    (16, 460_800, 300.0),    // Guitar  ← the reported stem (slot 1)
    (8, 230_400, 400.0),     // Drum
    (160, 4_608_000, 500.0), // Piano
    (8, 230_400, 600.0),     // Bass
];
const GUITAR: u8 = 1;

fn meta(slot: u8, variation: u8, cache_only: bool) -> LoadMeta {
    let (loop_beats, original_samples, _) = STEMS[slot as usize];
    LoadMeta {
        slot,
        sample_rate: SAMPLE_RATE,
        original_tempo_mbpm: O_MBPM,
        loop_beats,
        encoder_delay: ENCODER_DELAY,
        end_padding: END_PADDING,
        original_samples,
        variation,
        cache_only,
        sample_format: SampleFormat::F32,
        gain: 1.0,
        pan: 0.0,
    }
}

/// TrimmedPadding-shaped decode (Chrome): decoded = original + end_padding.
/// Real signal in the keep window, zeros in the padding tail.
fn decoded_pcm(slot: u8, variation: u8) -> (Vec<f32>, Vec<f32>) {
    let (_, original, base_hz) = STEMS[slot as usize];
    let hz = base_hz + variation as f32 * 37.0;
    let total = original + END_PADDING as u64;
    let mut l = Vec::with_capacity(total as usize);
    for n in 0..original {
        let t = n as f32 / SAMPLE_RATE as f32;
        l.push(0.25 * (2.0 * std::f32::consts::PI * hz * t).sin());
    }
    l.extend(std::iter::repeat_n(0.0, END_PADDING as usize));
    (l.clone(), l)
}

/// Bring up the engine the way the app does: load default (variation 1) on
/// every slot, then autostart each track (`Activate{immediate, auto}` — the
/// first activation starts the transport; no explicit Play).
fn lagos_engine() -> (ControlPlane, Box<AudioCore>) {
    let (mut plane, core) = create_engine(EngineConfig {
        sample_rate: SAMPLE_RATE,
        initial_tempo_mbpm: O_MBPM,
        beats_per_bar: 4,
        pcm_budget_bytes: None,
    })
    .unwrap();
    for slot in 0..STEMS.len() as u8 {
        let (l, r) = decoded_pcm(slot, 1);
        plane
            .load_pcm(&meta(slot, 1, false), l, r)
            .unwrap_or_else(|e| panic!("slot {slot} default load: {e}"));
        assert!(matches!(
            plane.dispatch(Command::Activate {
                slot,
                quantize: QuantizePolicy::Immediate,
                transport: TransportPolicy::Auto,
            }),
            CommandResult::Ok
        ));
    }
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

/// The guitar's own post-mix level from the snapshot meter (per-track
/// contribution — the same signal the UI's tile meter shows).
fn guitar_rms(plane: &ControlPlane) -> f32 {
    plane.snapshot().tracks[GUITAR as usize].rms_l
}

/// Host-order A (the common real ordering): user switches; the FR15
/// micro-warm parks the OTHER variation first (its target list is built from
/// a stale snapshot); the §4.11 answer load lands after.
#[test]
fn async_switch_with_interleaved_microwarm_stays_audible() {
    let (mut plane, mut core) = lagos_engine();
    render_blocks(&mut core, 1000); // ~2.7 s of playback
    assert!(guitar_rms(&plane) > 0.05, "precondition: guitar audible before the switch");
    plane.drain_events();

    // User taps "complex" (variation 2) on the guitar.
    assert!(matches!(
        plane.dispatch(Command::SwitchVariation { slot: GUITAR, variation: 2 }),
        CommandResult::Ok
    ));
    let events = plane.drain_events();
    assert!(
        events
            .iter()
            .any(|e| matches!(e, Event::VariationLoadRequired { slot: 1, variation: 2 })),
        "not parked yet → async protocol: {events:?}"
    );

    // Playback continues while the host fetches/decodes.
    render_blocks(&mut core, 200);

    // FR15 micro-warm parks variation 0 (stale-snapshot target list).
    let (l, r) = decoded_pcm(GUITAR, 0);
    plane.load_pcm(&meta(GUITAR, 0, true), l, r).expect("micro-warm park of v0");
    render_blocks(&mut core, 200);

    // The §4.11 answer: full load of variation 2 completes the switch.
    let (l, r) = decoded_pcm(GUITAR, 2);
    plane.load_pcm(&meta(GUITAR, 2, false), l, r).expect("answer load of v2");
    render_blocks(&mut core, 2); // swap + crossfade blocks

    let events = plane.drain_events();
    assert!(
        events
            .iter()
            .any(|e| matches!(e, Event::VariationApplied { slot: 1, variation: 2, .. })),
        "swap must be acked: {events:?}"
    );

    // The reported symptom: the guitar goes silent after the switch.
    let audio = render_blocks(&mut core, 400);
    let snap = plane.snapshot();
    assert_eq!(snap.tracks[GUITAR as usize].variation, 2);
    assert_eq!(snap.tracks[GUITAR as usize].status, maestro_api::TrackStatus::Active);
    assert!(
        guitar_rms(&plane) > 0.05,
        "guitar went SILENT after the async switch (meter {})",
        guitar_rms(&plane)
    );
    let mix_rms = (audio.iter().map(|s| s * s).sum::<f32>() / audio.len() as f32).sqrt();
    assert!(mix_rms > 0.05, "mix went silent: {mix_rms}");
}

/// Host-order B: the poll answered the switch BEFORE the micro-warm got to
/// the target (park of v0 lands after the swap).
#[test]
fn async_switch_answer_first_park_after_stays_audible() {
    let (mut plane, mut core) = lagos_engine();
    render_blocks(&mut core, 1000);
    plane.drain_events();

    assert!(matches!(
        plane.dispatch(Command::SwitchVariation { slot: GUITAR, variation: 2 }),
        CommandResult::Ok
    ));
    render_blocks(&mut core, 100);

    let (l, r) = decoded_pcm(GUITAR, 2);
    plane.load_pcm(&meta(GUITAR, 2, false), l, r).expect("answer load of v2");
    render_blocks(&mut core, 2);

    let (l, r) = decoded_pcm(GUITAR, 0);
    plane.load_pcm(&meta(GUITAR, 0, true), l, r).expect("late park of v0");
    render_blocks(&mut core, 400);

    assert_eq!(plane.snapshot().tracks[GUITAR as usize].variation, 2);
    assert!(
        guitar_rms(&plane) > 0.05,
        "guitar went SILENT (order B, meter {})",
        guitar_rms(&plane)
    );
}

/// A pending switch completed BY the micro-warm's cache-only load of the same
/// variation (the sanctioned ADR-098 race), Lagos geometry.
#[test]
fn pending_switch_completed_by_microwarm_park_stays_audible() {
    let (mut plane, mut core) = lagos_engine();
    render_blocks(&mut core, 1000);
    plane.drain_events();

    assert!(matches!(
        plane.dispatch(Command::SwitchVariation { slot: GUITAR, variation: 0 }),
        CommandResult::Ok
    ));
    render_blocks(&mut core, 100);

    // Micro-warm target list is [v0, v2]; v0 completes the pending switch,
    // v2 parks normally.
    let (l, r) = decoded_pcm(GUITAR, 0);
    plane.load_pcm(&meta(GUITAR, 0, true), l, r).expect("park completes the pending switch");
    render_blocks(&mut core, 2);
    let (l, r) = decoded_pcm(GUITAR, 2);
    plane.load_pcm(&meta(GUITAR, 2, true), l, r).expect("v2 parks");
    render_blocks(&mut core, 400);

    let snap = plane.snapshot();
    assert_eq!(snap.tracks[GUITAR as usize].variation, 0);
    assert!(
        guitar_rms(&plane) > 0.05,
        "guitar went SILENT (park-completion, meter {})",
        guitar_rms(&plane)
    );
}

/// Async-answered swap while the track renders the STRETCH path (tempo nudged
/// off the stems' original) — the combination no existing test covers:
/// `variation_length_switch` swaps under stretch via the CACHED path only,
/// and its async-path test runs at unity.
#[test]
fn async_switch_under_stretch_stays_audible() {
    for tempo in [96_000u32, 104_000] {
        let (mut plane, mut core) = lagos_engine();
        render_blocks(&mut core, 400);
        // User nudges the global tempo stepper: every stem goes stretch-path.
        assert!(matches!(plane.dispatch(Command::SetTempo { mbpm: tempo }), CommandResult::Ok));
        render_blocks(&mut core, 1000);
        assert!(
            guitar_rms(&plane) > 0.03,
            "tempo {tempo}: precondition — guitar audible under stretch"
        );
        plane.drain_events();

        assert!(matches!(
            plane.dispatch(Command::SwitchVariation { slot: GUITAR, variation: 2 }),
            CommandResult::Ok
        ));
        // Fetch+decode delay: seconds of rendering before the answer lands.
        render_blocks(&mut core, 800);

        let (l, r) = decoded_pcm(GUITAR, 2);
        plane.load_pcm(&meta(GUITAR, 2, false), l, r).expect("answer load of v2");
        render_blocks(&mut core, 2);

        let events = plane.drain_events();
        assert!(
            events
                .iter()
                .any(|e| matches!(e, Event::VariationApplied { slot: 1, variation: 2, .. })),
            "tempo {tempo}: swap must be acked: {events:?}"
        );
        render_blocks(&mut core, 800);
        assert_eq!(plane.snapshot().tracks[GUITAR as usize].variation, 2);
        assert!(
            guitar_rms(&plane) > 0.03,
            "tempo {tempo}: guitar went SILENT after async switch under stretch (meter {})",
            guitar_rms(&plane)
        );
    }
}

/// Same, with a nonzero per-stem Tune (pitch) — the other stretch-path
/// promoter reachable from the detail card.
#[test]
fn async_switch_with_pitch_stays_audible() {
    let (mut plane, mut core) = lagos_engine();
    render_blocks(&mut core, 400);
    assert!(matches!(
        plane.dispatch(Command::SetPitch { slot: GUITAR, semitones: 3.0 }),
        CommandResult::Ok
    ));
    render_blocks(&mut core, 1000);
    assert!(guitar_rms(&plane) > 0.03, "precondition: audible with pitch");
    plane.drain_events();

    assert!(matches!(
        plane.dispatch(Command::SwitchVariation { slot: GUITAR, variation: 0 }),
        CommandResult::Ok
    ));
    render_blocks(&mut core, 800);
    let (l, r) = decoded_pcm(GUITAR, 0);
    plane.load_pcm(&meta(GUITAR, 0, false), l, r).expect("answer load of v0");
    render_blocks(&mut core, 802);
    assert_eq!(plane.snapshot().tracks[GUITAR as usize].variation, 0);
    assert!(
        guitar_rms(&plane) > 0.03,
        "guitar went SILENT after async switch with pitch (meter {})",
        guitar_rms(&plane)
    );
}

/// THE REPORTED BUG's engine-side mechanism: a full (non-cache-only) load of
/// the variation that is ALREADY INSTALLED, landing on an ACTIVE, PLAYING
/// slot. Reachable from the host as the async-switch dedupe loser (ADR-098):
/// a §4.11 park completes the pending switch first, then the answer load
/// arrives for the now-installed variation. Today this falls through to the
/// fresh-install path: InstallTrack replaces the playing TrackState with an
/// inactive one — sustained silence, NO Deactivated/Removed event (the UI
/// keeps showing "playing"), §4.11 cache wiped. Toggling the tile off/on
/// re-activates — exactly the user-reported workaround.
#[test]
fn full_reload_of_installed_variation_must_not_kill_a_playing_track() {
    let (mut plane, mut core) = lagos_engine();
    render_blocks(&mut core, 1000);

    // The user's switch: pending → answered concurrently by park AND load.
    assert!(matches!(
        plane.dispatch(Command::SwitchVariation { slot: GUITAR, variation: 2 }),
        CommandResult::Ok
    ));
    render_blocks(&mut core, 200);

    // The park (racing executor) lands first and completes the pending switch.
    let (l, r) = decoded_pcm(GUITAR, 2);
    plane.load_pcm(&meta(GUITAR, 2, true), l, r).expect("park completes the pending switch");
    render_blocks(&mut core, 200);
    assert_eq!(plane.snapshot().tracks[GUITAR as usize].variation, 2);
    assert!(guitar_rms(&plane) > 0.05, "precondition: audible on v2 after park-completion");
    plane.drain_events();

    // The dedupe-loser: the §4.11 answer load arrives late — a full load of
    // the NOW-INSTALLED variation 2.
    let (l, r) = decoded_pcm(GUITAR, 2);
    plane.load_pcm(&meta(GUITAR, 2, false), l, r).expect("late answer load must not fail");
    render_blocks(&mut core, 400);

    let events = plane.drain_events();
    let snap = plane.snapshot();
    // The track must survive: still ACTIVE, still AUDIBLE, no silent
    // deactivation (a Deactivated/Removed event would at least tell the UI).
    assert!(
        !events.iter().any(|e| matches!(e, Event::TrackDeactivated { slot: 1, .. })),
        "no deactivation may fire: {events:?}"
    );
    assert_eq!(
        snap.tracks[GUITAR as usize].status,
        maestro_api::TrackStatus::Active,
        "the playing track was silently deactivated by the same-variation reload"
    );
    assert!(
        guitar_rms(&plane) > 0.05,
        "guitar went SILENT: the same-variation reload reset the slot (meter {})",
        guitar_rms(&plane)
    );
}
