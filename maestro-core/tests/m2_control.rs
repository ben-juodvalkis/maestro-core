//! # M2 control-surface tests (spec §5 M2, §3.2, §4.11, ADR-092)
//!
//! - **Mute/solo/gain-0 skip DSP entirely** (spec §3.2): meters read zero,
//!   the bus stays silent, and on becoming audible again the track re-anchors
//!   under the activation ramp — verified bit-exact through the unity bypass
//!   (the strongest possible "cannot desync" assertion: the audio equals the
//!   source at the derived position, to the bit).
//! - **Level metering**: rms/peak in the snapshot reflect the actual rendered
//!   contribution (anti-stub for the M3 level-meter UI).
//! - **Transport safety policy** (ADR-092): Activate+Auto starts a stopped
//!   transport; Deactivate+Auto on the last active track stops it (epoch
//!   reset); Manual never touches the transport — codifying §3.1's
//!   armed-tracks model.
//! - **§4.11 variation cache**: async switch (acknowledged → load-required →
//!   applied-on-completion), cached switch (immediate), eviction above the
//!   96 MB budget (never the installed variation), `pcm_bytes_in_use`
//!   reporting.

mod common;

use common::{make_sine_pcm, SAMPLE_RATE};
use maestro_api::{
    Command, CommandResult, EngineConfig, Event, LoadMeta, QuantizePolicy, TransportPolicy,
    TransportState,
};
use maestro_core::audio_core::AudioCore;
use maestro_core::control_plane::{create_engine, ControlPlane, PCM_BUDGET_BYTES};

const O_MBPM: u32 = 120_000;
const LOOP_BEATS: u32 = 16;
const PCM_LEN: u64 = 384_000;
const BLOCK: usize = 480;

fn meta(slot: u8, variation: u8) -> LoadMeta {
    LoadMeta {
        slot,
        sample_rate: SAMPLE_RATE,
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

fn engine_with_sine(hz: f32) -> (ControlPlane, Box<AudioCore>) {
    let (mut plane, core) = create_engine(EngineConfig {
        sample_rate: SAMPLE_RATE,
        initial_tempo_mbpm: O_MBPM,
        beats_per_bar: 4,
    })
    .unwrap();
    let (l, r) = make_sine_pcm(PCM_LEN, hz);
    plane.load_pcm(&meta(0, 0), l, r).expect("loads");
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

fn rms(s: &[f32]) -> f32 {
    (s.iter().map(|x| x * x).sum::<f32>() / s.len() as f32).sqrt()
}

// ---------------------------------------------------------------------------
// Mute / solo / gain-0: DSP skip + bit-exact re-entry
// ---------------------------------------------------------------------------

#[test]
fn mute_silences_skips_dsp_and_reenters_bit_exact() {
    let (mut plane, mut core) = engine_with_sine(220.0);
    render_blocks(&mut core, 20);

    assert!(matches!(plane.dispatch(Command::SetMute { slot: 0, muted: true }), CommandResult::Ok));
    render_blocks(&mut core, 2); // command block + ramp-to-zero block
    let silent = render_blocks(&mut core, 20);
    assert_eq!(rms(&silent), 0.0, "muted track must contribute exact silence");
    let snap = plane.snapshot();
    assert!(snap.tracks[0].muted);
    assert_eq!(snap.tracks[0].rms_l, 0.0, "meters must read zero while DSP is skipped");
    assert_eq!(snap.tracks[0].peak_l, 0.0);

    // Unmute: re-anchor under the activation ramp; past the ramp, the unity
    // bypass must be the bit-exact source at the DERIVED position — silence
    // cost nothing and cannot desync (spec §3.2).
    let mute_applied = plane
        .drain_events()
        .iter()
        .find_map(|e| match e {
            Event::MuteApplied { slot: 0, muted: true, at_frame } => Some(*at_frame),
            _ => None,
        })
        .expect("MuteApplied echo missing");
    assert_eq!(mute_applied, 20 * BLOCK as u64, "mute echo must carry the exact frame");

    assert!(matches!(
        plane.dispatch(Command::SetMute { slot: 0, muted: false }),
        CommandResult::Ok
    ));
    let blocks_so_far = 20 + 2 + 20;
    render_blocks(&mut core, 2); // command + ramp block
    let (sine_l, _) = make_sine_pcm(PCM_LEN, 220.0);
    let gl = (std::f32::consts::FRAC_PI_4).cos();
    let mut l = vec![0.0f32; BLOCK];
    let mut r = vec![0.0f32; BLOCK];
    let n0 = (blocks_so_far + 2) as u64 * BLOCK as u64;
    core.render(&mut l, &mut r);
    for i in 0..BLOCK {
        let src = sine_l[((n0 + i as u64) % PCM_LEN) as usize];
        assert_eq!(
            l[i].to_bits(),
            (src * gl).to_bits(),
            "post-unmute bypass not bit-exact at frame {}",
            n0 + i as u64
        );
    }
    drop(plane);
}

#[test]
fn gain_zero_skips_dsp_and_gain_restores() {
    let (mut plane, mut core) = engine_with_sine(220.0);
    render_blocks(&mut core, 10);
    assert!(matches!(plane.dispatch(Command::SetGain { slot: 0, gain: 0.0 }), CommandResult::Ok));
    render_blocks(&mut core, 2);
    let silent = render_blocks(&mut core, 10);
    assert_eq!(rms(&silent), 0.0, "gain-0 track must skip to exact silence");

    assert!(matches!(plane.dispatch(Command::SetGain { slot: 0, gain: 0.5 }), CommandResult::Ok));
    render_blocks(&mut core, 3);
    let audible = render_blocks(&mut core, 10);
    let expected = 0.25 * (std::f32::consts::FRAC_PI_4).cos() * 0.5 / 2f32.sqrt();
    assert!(
        (rms(&audible) - expected).abs() < expected * 0.05,
        "gain 0.5 sine RMS {} vs expected {expected}",
        rms(&audible)
    );
    drop(plane);
}

#[test]
fn gain_and_pan_out_of_range_rejected() {
    let (mut plane, _core) = engine_with_sine(220.0);
    for bad in [-0.1f32, 2.1, f32::NAN, f32::INFINITY] {
        assert!(matches!(
            plane.dispatch(Command::SetGain { slot: 0, gain: bad }),
            CommandResult::Rejected { .. }
        ));
    }
    for bad in [-1.1f32, 1.1, f32::NAN] {
        assert!(matches!(
            plane.dispatch(Command::SetPan { slot: 0, pan: bad }),
            CommandResult::Rejected { .. }
        ));
    }
    // Boundaries accepted.
    assert!(matches!(plane.dispatch(Command::SetGain { slot: 0, gain: 2.0 }), CommandResult::Ok));
    assert!(matches!(plane.dispatch(Command::SetPan { slot: 0, pan: -1.0 }), CommandResult::Ok));
}

#[test]
fn solo_silences_others_and_restores_under_ramp() {
    // Two tracks: solo slot 1 → slot 0 silent; clear solo → slot 0 returns.
    let (mut plane, mut core) = engine_with_sine(220.0);
    let (l, r) = make_sine_pcm(PCM_LEN, 330.0);
    plane.load_pcm(&meta(1, 0), l, r).expect("slot 1 loads");
    assert!(matches!(
        plane.dispatch(Command::Activate {
            slot: 1,
            quantize: QuantizePolicy::Immediate,
            transport: TransportPolicy::Manual
        }),
        CommandResult::Ok
    ));
    render_blocks(&mut core, 10);

    assert!(matches!(plane.dispatch(Command::SetSolo { slot: 1, solo: true }), CommandResult::Ok));
    render_blocks(&mut core, 4);
    let snap = plane.snapshot();
    assert_eq!(snap.tracks[0].rms_l, 0.0, "soloed-out track must be skipped (meters zero)");
    assert!(snap.tracks[1].rms_l > 0.0, "soloed track must keep playing");

    assert!(matches!(plane.dispatch(Command::SetSolo { slot: 1, solo: false }), CommandResult::Ok));
    render_blocks(&mut core, 4);
    let snap = plane.snapshot();
    assert!(snap.tracks[0].rms_l > 0.0, "track must return when the last solo clears");
    let events = plane.drain_events();
    assert!(events.iter().any(|e| matches!(e, Event::SoloApplied { slot: 1, solo: true, .. })));
    drop(plane);
}

#[test]
fn metering_reports_rendered_contribution() {
    let (plane, mut core) = engine_with_sine(220.0);
    render_blocks(&mut core, 20);
    let snap = plane.snapshot();
    // Sine amplitude 0.25, gain 1, pan 0 → per-channel amplitude 0.25·cos(π/4);
    // RMS = amplitude/√2; peak ≈ amplitude.
    let amp = 0.25 * (std::f32::consts::FRAC_PI_4).cos();
    let t = &snap.tracks[0];
    assert!((t.rms_l - amp / 2f32.sqrt()).abs() < 0.02, "rms_l {} vs {}", t.rms_l, amp / 2f32.sqrt());
    assert!((t.peak_l - amp).abs() < 0.02, "peak_l {} vs {amp}", t.peak_l);
    assert!((t.rms_r - t.rms_l).abs() < 1e-3, "centered pan must meter equally");
    drop(plane);
}

// ---------------------------------------------------------------------------
// Transport safety policy (ADR-092)
// ---------------------------------------------------------------------------

#[test]
fn activate_auto_starts_stopped_transport_and_manual_does_not() {
    let (mut plane, core) = create_engine(EngineConfig {
        sample_rate: SAMPLE_RATE,
        initial_tempo_mbpm: O_MBPM,
        beats_per_bar: 4,
    })
    .unwrap();
    let mut core = core;
    let (l, r) = make_sine_pcm(PCM_LEN, 220.0);
    plane.load_pcm(&meta(0, 0), l, r).unwrap();

    // Manual: arming never starts the transport.
    assert!(matches!(
        plane.dispatch(Command::Activate {
            slot: 0,
            quantize: QuantizePolicy::Immediate,
            transport: TransportPolicy::Manual
        }),
        CommandResult::Ok
    ));
    render_blocks(&mut core, 4);
    assert_eq!(plane.snapshot().transport, TransportState::Stopped);
    assert_eq!(plane.snapshot().engine_frame, 0);

    // Auto: arming starts it (engine-decided; Started echoed).
    assert!(matches!(
        plane.dispatch(Command::Deactivate { slot: 0, transport: TransportPolicy::Manual }),
        CommandResult::Ok
    ));
    assert!(matches!(
        plane.dispatch(Command::Activate {
            slot: 0,
            quantize: QuantizePolicy::Immediate,
            transport: TransportPolicy::Auto
        }),
        CommandResult::Ok
    ));
    render_blocks(&mut core, 4);
    assert_eq!(plane.snapshot().transport, TransportState::Playing);
    assert!(plane.snapshot().engine_frame > 0);
    assert!(plane.drain_events().iter().any(|e| matches!(e, Event::Started)));
    drop(plane);
}

#[test]
fn deactivate_auto_stops_only_when_last_and_manual_never_stops() {
    let (mut plane, mut core) = engine_with_sine(220.0);
    let (l, r) = make_sine_pcm(PCM_LEN, 330.0);
    plane.load_pcm(&meta(1, 0), l, r).unwrap();
    assert!(matches!(
        plane.dispatch(Command::Activate {
            slot: 1,
            quantize: QuantizePolicy::Immediate,
            transport: TransportPolicy::Manual
        }),
        CommandResult::Ok
    ));
    render_blocks(&mut core, 4);

    // Auto-deactivate slot 1 while slot 0 is still active: must NOT stop.
    assert!(matches!(
        plane.dispatch(Command::Deactivate { slot: 1, transport: TransportPolicy::Auto }),
        CommandResult::Ok
    ));
    render_blocks(&mut core, 2);
    assert_eq!(plane.snapshot().transport, TransportState::Playing);

    // Manual-deactivate the LAST active track: §3.1 default — the timeline
    // keeps running (armed-tracks model), no epoch reset.
    let epoch_before = plane.snapshot().timeline_epoch;
    let f_before = plane.snapshot().engine_frame;
    assert!(matches!(
        plane.dispatch(Command::Deactivate { slot: 0, transport: TransportPolicy::Manual }),
        CommandResult::Ok
    ));
    render_blocks(&mut core, 4);
    let snap = plane.snapshot();
    assert_eq!(snap.transport, TransportState::Playing, "Manual must never stop the transport");
    assert!(snap.engine_frame > f_before, "F must keep advancing through silence (spec §3.1)");
    assert_eq!(snap.timeline_epoch, epoch_before);

    // Re-activate, then Auto-deactivate the last one: engine stops, epoch resets.
    assert!(matches!(
        plane.dispatch(Command::Activate {
            slot: 0,
            quantize: QuantizePolicy::Immediate,
            transport: TransportPolicy::Manual
        }),
        CommandResult::Ok
    ));
    render_blocks(&mut core, 2);
    assert!(matches!(
        plane.dispatch(Command::Deactivate { slot: 0, transport: TransportPolicy::Auto }),
        CommandResult::Ok
    ));
    render_blocks(&mut core, 2);
    let snap = plane.snapshot();
    assert_eq!(snap.transport, TransportState::Stopped, "Auto on the last track must stop");
    assert_eq!(snap.timeline_epoch, epoch_before + 1, "stop is an epoch-reset flow (spec §3.1)");
    assert!(plane.drain_events().iter().any(|e| matches!(e, Event::Stopped)));
    drop(plane);
}

// ---------------------------------------------------------------------------
// §4.11 variation cache: async switch, cached switch, eviction, reporting
// ---------------------------------------------------------------------------

/// Dominant frequency estimate via zero crossings (good to ~1% on a sine).
fn measured_hz(signal: &[f32]) -> f64 {
    let crossings = signal.windows(2).filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0)).count();
    crossings as f64 / 2.0 * SAMPLE_RATE as f64 / signal.len() as f64
}

#[test]
fn variation_switch_async_then_cached() {
    let (mut plane, mut core) = engine_with_sine(220.0);
    render_blocks(&mut core, 10);

    // Uncached switch: acknowledged; the platform is asked to load; the OLD
    // variation keeps playing until completion (no gap).
    assert!(matches!(
        plane.dispatch(Command::SwitchVariation { slot: 0, variation: 1 }),
        CommandResult::Ok
    ));
    let events = plane.drain_events();
    assert!(
        events
            .iter()
            .any(|e| matches!(e, Event::VariationLoadRequired { slot: 0, variation: 1 })),
        "uncached switch must request a load: {events:?}"
    );
    let still_old = render_blocks(&mut core, 10);
    let hz = measured_hz(&still_old);
    assert!((hz - 220.0).abs() < 10.0, "old variation must keep playing while loading: {hz:.1}");

    // Load completion applies the swap at a block boundary.
    let (l, r) = make_sine_pcm(PCM_LEN, 440.0);
    plane.load_pcm(&meta(0, 1), l, r).expect("variation 1 loads");
    render_blocks(&mut core, 3);
    let now_new = render_blocks(&mut core, 10);
    let hz = measured_hz(&now_new);
    assert!((hz - 440.0).abs() < 12.0, "variation 1 must be audible after load: {hz:.1}");
    let events = plane.drain_events();
    assert!(events
        .iter()
        .any(|e| matches!(e, Event::VariationApplied { slot: 0, variation: 1, .. })));

    // Switching BACK is now a cached switch: applied immediately, no load
    // request, content flips within a few blocks.
    assert!(matches!(
        plane.dispatch(Command::SwitchVariation { slot: 0, variation: 0 }),
        CommandResult::Ok
    ));
    render_blocks(&mut core, 3);
    let back = render_blocks(&mut core, 10);
    let hz = measured_hz(&back);
    assert!((hz - 220.0).abs() < 10.0, "cached switch must apply immediately: {hz:.1}");
    let events = plane.drain_events();
    assert!(
        !events.iter().any(|e| matches!(e, Event::VariationLoadRequired { .. })),
        "cached switch must not request a load: {events:?}"
    );
    assert!(events
        .iter()
        .any(|e| matches!(e, Event::VariationApplied { slot: 0, variation: 0, .. })));

    // Snapshot reports both resident variations.
    let snap = plane.snapshot();
    let expected_bytes = 2 * PCM_LEN * 2 * 4; // two variations, stereo f32
    assert_eq!(snap.pcm_bytes_in_use, expected_bytes);
    drop(plane);
}

#[test]
fn eviction_above_budget_never_touches_the_installed_variation() {
    // A long loop big enough that two variations exceed the 96 MB budget:
    // 1250 beats @ 60 BPM @ 48 kHz = 60 M samples = 480 MB?? No — keep it
    // real: 16 beats @ 60 BPM = 768_000 samples ≈ 6.1 MB per variation means
    // eviction needs ~16 variations; instead use a 208-beat loop @ 60 BPM
    // (9.984 M samples ≈ 79.9 MB per variation): the SECOND variation pushes
    // the slot past 96 MB and must evict the cached (not installed) one.
    const LONG_LEN: u64 = 9_984_000;
    const LONG_BEATS: u32 = 208;
    let bytes_per_variation = LONG_LEN * 2 * 4;
    assert!(bytes_per_variation < PCM_BUDGET_BYTES && 2 * bytes_per_variation > PCM_BUDGET_BYTES);

    let (mut plane, mut core) = create_engine(EngineConfig {
        sample_rate: SAMPLE_RATE,
        initial_tempo_mbpm: 60_000,
        beats_per_bar: 4,
    })
    .unwrap();
    let long_meta = |variation: u8| LoadMeta {
        slot: 0,
        sample_rate: SAMPLE_RATE,
        original_tempo_mbpm: 60_000,
        loop_beats: LONG_BEATS,
        encoder_delay: 0,
        end_padding: 0,
        original_samples: LONG_LEN,
        variation,
        gain: 1.0,
        pan: 0.0,
    };
    let (l, r) = make_sine_pcm(LONG_LEN, 110.0);
    plane.load_pcm(&long_meta(0), l, r).expect("variation 0 loads");
    assert!(matches!(
        plane.dispatch(Command::Activate {
            slot: 0,
            quantize: QuantizePolicy::Immediate,
            transport: TransportPolicy::Manual
        }),
        CommandResult::Ok
    ));
    assert!(matches!(plane.dispatch(Command::Play), CommandResult::Ok));
    render_blocks(&mut core, 4);
    assert!(plane.snapshot().pcm_bytes_in_use <= PCM_BUDGET_BYTES);

    // Switching to variation 1 caches variation 0 → over budget → the CACHED
    // variation 0 is evicted; the installed variation 1 is untouchable.
    let (l, r) = make_sine_pcm(LONG_LEN, 220.0);
    plane.load_pcm(&long_meta(1), l, r).expect("variation 1 swaps in");
    render_blocks(&mut core, 4);
    let snap = plane.snapshot();
    assert!(
        snap.pcm_bytes_in_use <= PCM_BUDGET_BYTES,
        "budget enforced: {} > {PCM_BUDGET_BYTES}",
        snap.pcm_bytes_in_use
    );
    assert_eq!(snap.tracks[0].variation, 1, "installed variation must survive eviction");
    let events = plane.drain_events();
    assert!(
        events
            .iter()
            .any(|e| matches!(e, Event::VariationEvicted { slot: 0, variation: 0 })),
        "eviction must be surfaced: {events:?}"
    );

    // Switching back to 0 is async again (it was evicted) — the eviction is
    // observable behavior, not bookkeeping.
    assert!(matches!(
        plane.dispatch(Command::SwitchVariation { slot: 0, variation: 0 }),
        CommandResult::Ok
    ));
    let events = plane.drain_events();
    assert!(events
        .iter()
        .any(|e| matches!(e, Event::VariationLoadRequired { slot: 0, variation: 0 })));

    // Audio kept playing across all of it (no gap on the playing stem).
    let tail = render_blocks(&mut core, 10);
    assert!(rms(&tail) > 0.05, "stem must keep sounding through switch + eviction");
    drop(plane);
}
