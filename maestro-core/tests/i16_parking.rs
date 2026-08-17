//! ADR-119 §6.1 — i16 storage on the PARK/SWAP path.
//!
//! ADR-112 Tier 2a shipped `SampleFormat::I16` but wired it only to the
//! STREAMING path (`StreamLoadMeta.sample_format`); `build_complete_pcm`
//! hardcoded F32, so the §4.11 park/swap path — the one every variation
//! switch takes — could not use it. ADR-119 FR1/FR2 open that door, and these
//! tests are the ones that make it safe to walk through:
//!
//!  * the null tests, extended from streams to parks: what a listener would
//!    hear after a switch INTO an i16-parked take, differenced against the
//!    f32 control;
//!  * the mixed-format state (f32 installed + i16 parked), which is not a
//!    design we ship but IS transiently real during any flag flip or partial
//!    rollout, so it must be correct in both directions (PRD §3.1 residue);
//!  * budget accounting halving for parks, extending
//!    `t_stream::i16_budget_accounting_halves` off the streaming path;
//!  * ADR-109 per-variation loop lengths surviving in i16 — format and length
//!    are orthogonal, and the map-swaps-atomically-with-the-PCM contract has
//!    to hold in both formats.
//!
//! FR3 (the f32 default is byte-unchanged) is pinned by the rest of the
//! suite staying green, not here.
mod common;

use common::{make_sine_pcm, SAMPLE_RATE};
use maestro_api::{
    Command, CommandResult, EngineConfig, LoadMeta, QuantizePolicy, SampleFormat, TrackStatus,
    TransportPolicy,
};
use maestro_core::audio_core::AudioCore;
use maestro_core::control_plane::{create_engine, ControlPlane};

const O_MBPM: u32 = 120_000;
const LOOP_BEATS: u32 = 16;
const PCM_LEN: u64 = 384_000;
const BLOCK: usize = 480;

fn meta_fmt(slot: u8, variation: u8, cache_only: bool, format: SampleFormat) -> LoadMeta {
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
        sample_format: format,
        gain: 1.0,
        pan: 0.0,
    }
}

/// Same, with an explicit per-file loop length (ADR-109).
fn meta_len(slot: u8, variation: u8, format: SampleFormat, loop_beats: u32, samples: u64) -> LoadMeta {
    LoadMeta {
        slot,
        sample_rate: SAMPLE_RATE,
        original_tempo_mbpm: O_MBPM,
        loop_beats,
        encoder_delay: 0,
        end_padding: 0,
        original_samples: samples,
        variation,
        cache_only: true,
        sample_format: format,
        gain: 1.0,
        pan: 0.0,
    }
}

fn engine_at(tempo_mbpm: u32) -> (ControlPlane, Box<AudioCore>) {
    create_engine(EngineConfig {
        sample_rate: SAMPLE_RATE,
        initial_tempo_mbpm: tempo_mbpm,
        beats_per_bar: 4,
        pcm_budget_bytes: None,
    })
    .unwrap()
}

fn ok(r: CommandResult) {
    assert!(matches!(r, CommandResult::Ok), "command rejected: {r:?}");
}

fn render(core: &mut AudioCore, blocks: usize) -> (Vec<f32>, Vec<f32>) {
    let mut l = vec![0.0f32; BLOCK];
    let mut r = vec![0.0f32; BLOCK];
    let (mut ol, mut or) = (Vec::new(), Vec::new());
    for _ in 0..blocks {
        core.render(&mut l, &mut r);
        ol.extend_from_slice(&l);
        or.extend_from_slice(&r);
    }
    (ol, or)
}

/// Install variation 0 (f32, 220 Hz), park variation 1 (`park_format`,
/// 440 Hz), switch to it, then render `blocks` blocks of the RESULT.
///
/// The switch is what makes this a park-path test rather than a load test:
/// the engine installs the parked `Arc<EnginePcm>` directly (no re-decode,
/// no re-expansion), so whatever format it was parked in is the format that
/// then renders.
fn park_switch_render(
    tempo_mbpm: u32,
    park_format: SampleFormat,
    blocks: usize,
) -> (Vec<f32>, Vec<f32>) {
    let (mut plane, mut core) = engine_at(tempo_mbpm);
    let (l0, r0) = make_sine_pcm(PCM_LEN, 220.0);
    plane.load_pcm(&meta_fmt(0, 0, false, SampleFormat::F32), l0, r0).expect("install v0");
    let (l1, r1) = make_sine_pcm(PCM_LEN, 440.0);
    plane.load_pcm(&meta_fmt(0, 1, true, park_format), l1, r1).expect("park v1");

    ok(plane.dispatch(Command::Activate {
        slot: 0,
        quantize: QuantizePolicy::Immediate,
        transport: TransportPolicy::Manual,
    }));
    ok(plane.dispatch(Command::Play));
    // Let the crossfade block pass so we difference steady-state audio, not
    // the one-block fade the swap schedules.
    render(&mut core, 2);
    ok(plane.dispatch(Command::SwitchVariation { slot: 0, variation: 1 }));
    render(&mut core, 2);
    render(&mut core, blocks)
}

/// Residual RMS (dBFS) between an f32-parked and an i16-parked switch.
fn park_residual_dbfs(tempo_mbpm: u32) -> f64 {
    let (fl, fr) = park_switch_render(tempo_mbpm, SampleFormat::F32, 200);
    let (il, ir) = park_switch_render(tempo_mbpm, SampleFormat::I16, 200);
    let peak = fl.iter().fold(0.0f32, |a, s| a.max(s.abs()));
    assert!(peak > 0.0, "anti-stub: the null test rendered silence");

    let mut sumsq = 0.0f64;
    let mut n = 0u64;
    for (a, b) in [(&fl, &il), (&fr, &ir)] {
        assert_eq!(a.len(), b.len(), "control and i16 runs must be the same length");
        for (x, y) in a.iter().zip(b.iter()) {
            let d = (*x - *y) as f64;
            sumsq += d * d;
            n += 1;
        }
    }
    20.0 * (sumsq / n as f64).sqrt().log10()
}

// ---------------------------------------------------------------------------
// Null tests on the park/swap path (§6.1)
// ---------------------------------------------------------------------------

/// Unity (bit-exact bypass): switching into an i16-parked take differs from
/// the f32 control only by the once-at-load quantization floor. Same bound as
/// the streaming unity null test.
#[test]
fn i16_park_switch_null_test_unity_path() {
    let residual = park_residual_dbfs(O_MBPM);
    // Printed, not just asserted: the NUMBER is the evidence this bound is
    // honest, and it is what the ADR records (cargo test -- --nocapture).
    eprintln!("i16 park/swap residual, unity path: {residual:.1} dBFS");
    assert!(residual <= -90.0, "unity park residual {residual:.1} dBFS > −90 dBFS");
}

/// Stretch path, at the worst rate ADR-112 measured (1.5×). Reuses that
/// deviation's bound and rationale verbatim: a phase vocoder does not
/// preserve small input deltas, so the −96 dBFS load floor emerges
/// rate-dependently as REDISTRIBUTED vocoder noise, never a signal-level
/// artifact. The perceptual gate is the listening pass, not this number.
#[test]
fn i16_park_switch_null_test_stretch_path() {
    let residual = park_residual_dbfs(180_000);
    eprintln!("i16 park/swap residual, stretch path (1.5x): {residual:.1} dBFS");
    assert!(residual <= -45.0, "stretch park residual {residual:.1} dBFS > −45 dBFS");
}

// ---------------------------------------------------------------------------
// Mixed format (PRD §3.1 residue) — never shipped as a design, transiently
// real during a flag flip, therefore must be correct.
// ---------------------------------------------------------------------------

/// f32 installed → switch to an i16 park → switch BACK to the f32 original.
/// Both directions must sound like the take they name, and the track must
/// stay active throughout (the ADR-115 property: a switch never silently
/// deactivates).
#[test]
fn mixed_format_swaps_correctly_in_both_directions() {
    let (mut plane, mut core) = engine_at(O_MBPM);
    let (l0, r0) = make_sine_pcm(PCM_LEN, 220.0);
    plane.load_pcm(&meta_fmt(0, 0, false, SampleFormat::F32), l0, r0).expect("install f32 v0");
    let (l1, r1) = make_sine_pcm(PCM_LEN, 440.0);
    plane.load_pcm(&meta_fmt(0, 1, true, SampleFormat::I16), l1, r1).expect("park i16 v1");

    ok(plane.dispatch(Command::Activate {
        slot: 0,
        quantize: QuantizePolicy::Immediate,
        transport: TransportPolicy::Manual,
    }));
    ok(plane.dispatch(Command::Play));
    render(&mut core, 2);

    // f32 → i16
    ok(plane.dispatch(Command::SwitchVariation { slot: 0, variation: 1 }));
    render(&mut core, 2);
    let (l, _) = render(&mut core, 40);
    assert!((measured_hz(&l) - 440.0).abs() < 5.0, "i16 park did not become audible as 440 Hz");
    assert_eq!(
        plane.snapshot().tracks[0].status,
        TrackStatus::Active,
        "the swap silently deactivated the track (ADR-115)"
    );

    // i16 → f32 (the outgoing i16 buffer is now the parked one)
    ok(plane.dispatch(Command::SwitchVariation { slot: 0, variation: 0 }));
    render(&mut core, 2);
    let (l, _) = render(&mut core, 40);
    assert!((measured_hz(&l) - 220.0).abs() < 5.0, "switch back to f32 did not restore 220 Hz");
    assert_eq!(
        plane.snapshot().tracks[0].status,
        TrackStatus::Active,
        "the swap back silently deactivated the track (ADR-115)"
    );
}

// ---------------------------------------------------------------------------
// Budget accounting (§6.1) — the whole point of the change
// ---------------------------------------------------------------------------

/// `pcm_bytes_in_use` after an i16 park is exactly half the f32 figure.
/// Extends `t_stream::i16_budget_accounting_halves` from begin/stream to the
/// park path, which is where the library's residency actually lives.
#[test]
fn i16_parks_halve_budget_accounting() {
    let installed_f32 = PCM_LEN * 2 * 4;

    let (mut plane, _core) = engine_at(O_MBPM);
    let (l0, r0) = make_sine_pcm(PCM_LEN, 220.0);
    plane.load_pcm(&meta_fmt(0, 0, false, SampleFormat::F32), l0, r0).expect("install");
    assert_eq!(plane.pcm_bytes_in_use(), installed_f32);

    let (l1, r1) = make_sine_pcm(PCM_LEN, 440.0);
    plane.load_pcm(&meta_fmt(0, 1, true, SampleFormat::I16), l1, r1).expect("park i16");
    assert_eq!(
        plane.pcm_bytes_in_use(),
        installed_f32 + PCM_LEN * 2 * 2,
        "an i16 park must cost exactly half an f32 park"
    );

    // And the f32 control, for the same content.
    let (mut plane2, _core2) = engine_at(O_MBPM);
    let (a, b) = make_sine_pcm(PCM_LEN, 220.0);
    plane2.load_pcm(&meta_fmt(0, 0, false, SampleFormat::F32), a, b).expect("install");
    let (c, d) = make_sine_pcm(PCM_LEN, 440.0);
    plane2.load_pcm(&meta_fmt(0, 1, true, SampleFormat::F32), c, d).expect("park f32");
    assert_eq!(plane2.pcm_bytes_in_use(), installed_f32 * 2);
}

/// Option B is UNIFORM: a whole slot loaded i16 — the install AND every park
/// — costs exactly half the same slot loaded f32.
///
/// This is the test that catches the real bug the first version of this file
/// missed. `i16_parks_halve_budget_accounting` above installs f32 and parks
/// i16, so it passes happily while INSTALLS are still f32 — which is exactly
/// what happened: a fresh install on an empty slot is re-expressed over the
/// streaming core (ADR-112 §5.3) and that branch hardcoded `F32`, so the app
/// only half-halved and silently sat in PRD §3.1's rejected Option A. Only an
/// end-to-end ratio over install + parks sees it.
#[test]
fn a_whole_slot_in_i16_costs_exactly_half_the_same_slot_in_f32() {
    let bytes_for = |format: SampleFormat| -> u64 {
        let (mut plane, _core) = engine_at(O_MBPM);
        let (l0, r0) = make_sine_pcm(PCM_LEN, 220.0);
        plane.load_pcm(&meta_fmt(0, 0, false, format), l0, r0).expect("install");
        for (variation, hz) in [(1u8, 440.0f32), (2, 660.0)] {
            let (l, r) = make_sine_pcm(PCM_LEN, hz);
            plane.load_pcm(&meta_fmt(0, variation, true, format), l, r).expect("park");
        }
        plane.pcm_bytes_in_use()
    };

    let f32_bytes = bytes_for(SampleFormat::F32);
    let i16_bytes = bytes_for(SampleFormat::I16);
    assert_eq!(f32_bytes, PCM_LEN * 2 * 4 * 3, "three f32 variations");
    assert_eq!(
        i16_bytes,
        f32_bytes / 2,
        "i16 must halve the WHOLE slot (install included), not just the parks — \
         got {i16_bytes} vs f32 {f32_bytes} (ratio {:.3})",
        f32_bytes as f64 / i16_bytes as f64
    );
}

// ---------------------------------------------------------------------------
// ADR-109 orthogonality
// ---------------------------------------------------------------------------

/// Variations of a stem may differ in whole-bar length (ADR-109). Format and
/// length are orthogonal: an i16 park of a DIFFERENT-length variation must
/// install its own timing map atomically with its PCM, exactly as f32 does.
/// Geometry mirrors Cups' 4/16-beat bounces.
#[test]
fn i16_park_of_a_different_length_variation_installs_its_own_map() {
    let quarter_len = PCM_LEN / 4; // 4 beats where the installed take is 16
    let (mut plane, mut core) = engine_at(O_MBPM);

    let (l0, r0) = make_sine_pcm(PCM_LEN, 220.0);
    let mut install = meta_len(0, 0, SampleFormat::F32, LOOP_BEATS, PCM_LEN);
    install.cache_only = false;
    plane.load_pcm(&install, l0, r0).expect("install 16-beat f32");

    let (l1, r1) = make_sine_pcm(quarter_len, 440.0);
    plane
        .load_pcm(&meta_len(0, 1, SampleFormat::I16, LOOP_BEATS / 4, quarter_len), l1, r1)
        .expect("park 4-beat i16");

    ok(plane.dispatch(Command::Activate {
        slot: 0,
        quantize: QuantizePolicy::Immediate,
        transport: TransportPolicy::Manual,
    }));
    ok(plane.dispatch(Command::Play));
    render(&mut core, 2);
    ok(plane.dispatch(Command::SwitchVariation { slot: 0, variation: 1 }));
    render(&mut core, 2);

    let (l, _) = render(&mut core, 40);
    assert!(
        (measured_hz(&l) - 440.0).abs() < 5.0,
        "the shorter i16 variation did not become audible"
    );
    assert_eq!(
        plane.snapshot().tracks[0].loop_beats,
        LOOP_BEATS / 4,
        "the i16 park's own loop length did not install with its PCM (ADR-109)"
    );
}

fn measured_hz(signal: &[f32]) -> f64 {
    let crossings = signal.windows(2).filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0)).count();
    crossings as f64 / 2.0 * SAMPLE_RATE as f64 / signal.len() as f64
}
