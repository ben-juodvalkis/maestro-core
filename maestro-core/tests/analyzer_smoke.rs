//! Offline smoke captures for the maestro-validate analyzer (spec §6.4's
//! negative-control requirement, exercised without hardware).
//!
//! Renders the two C0 scenarios through the real engine and writes
//! analyzer-compatible stereo WAVs under `target/validation-smoke/`:
//!
//! - `unity.wav` — metronome_120 hard-panned L and R at tempo 120 (bypass)
//! - `cross.wav` — metronome_60 (L) + metronome_90 (R) at tempo 120 (stretch)
//! - `cross_stretch_disabled.wav` — same, with the StretchDisabled sabotage
//!   (only when built with `--features drift-injection`)
//!
//! CI / device-validation.sh then runs `maestro-validate` against these:
//! the first two must PASS, the sabotaged one must FAIL (`--expect-fail`).

mod common;

use common::{fixture_path, load_float_wav, SAMPLE_RATE};
use maestro_api::{Command, CommandResult, EngineConfig, LoadMeta};
use maestro_core::control_plane::create_engine;
use maestro_core::validation::write_float_wav_stereo;
use std::path::PathBuf;

const TEMPO: u32 = 120_000;
const CAPTURE_SECONDS: u64 = 70; // ≥ 60 s and ≥ 8 wraps of the 8 s loops (C0)
const BLOCK: usize = 480;

fn out_dir() -> PathBuf {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../target/validation-smoke");
    std::fs::create_dir_all(&p).unwrap();
    p
}

struct Stem {
    fixture: &'static str,
    o_mbpm: u32,
    pan: f32,
}

fn render_scenario(stems: &[Stem], out_name: &str) {
    let (mut plane, mut core) = create_engine(EngineConfig {
        sample_rate: SAMPLE_RATE,
        initial_tempo_mbpm: TEMPO,
        beats_per_bar: 4,
    })
    .unwrap();

    for (slot, stem) in stems.iter().enumerate() {
        let wav = load_float_wav(&fixture_path(stem.fixture)).unwrap();
        assert_eq!(wav.sample_rate, SAMPLE_RATE);
        plane
            .load_pcm(
                &LoadMeta {
                    slot: slot as u8,
                    original_tempo_mbpm: stem.o_mbpm,
                    loop_beats: 16,
                    encoder_delay: 0,
                    end_padding: 0,
                    original_samples: wav.frames as u64,
                    variation: 0,
                    gain: 1.0,
                    pan: stem.pan,
                },
                wav.channels[0].clone(),
                wav.channels[1].clone(),
            )
            .unwrap();
        assert!(matches!(plane.dispatch(Command::Activate { slot: slot as u8, quantize: maestro_api::QuantizePolicy::Immediate }), CommandResult::Ok));
    }
    assert!(matches!(plane.dispatch(Command::Play), CommandResult::Ok));

    let total = CAPTURE_SECONDS * SAMPLE_RATE as u64;
    let mut cap_l = Vec::with_capacity(total as usize);
    let mut cap_r = Vec::with_capacity(total as usize);
    let mut l = vec![0.0f32; BLOCK];
    let mut r = vec![0.0f32; BLOCK];
    let mut rendered = 0u64;
    while rendered < total {
        core.render(&mut l, &mut r);
        cap_l.extend_from_slice(&l);
        cap_r.extend_from_slice(&r);
        rendered += BLOCK as u64;
    }

    let path = out_dir().join(out_name);
    write_float_wav_stereo(&path, &cap_l, &cap_r, SAMPLE_RATE).unwrap();
    println!("wrote {}", path.display());
}

#[test]
fn write_unity_capture() {
    render_scenario(
        &[
            Stem { fixture: "metronome_120.wav", o_mbpm: 120_000, pan: -1.0 },
            Stem { fixture: "metronome_120.wav", o_mbpm: 120_000, pan: 1.0 },
        ],
        "unity.wav",
    );
}

#[test]
fn write_cross_capture() {
    render_scenario(
        &[
            Stem { fixture: "metronome_60.wav", o_mbpm: 60_000, pan: -1.0 },
            Stem { fixture: "metronome_90.wav", o_mbpm: 90_000, pan: 1.0 },
        ],
        "cross.wav",
    );
}

/// The C0 stretch-disabled negative control, offline edition: the analyzer
/// must FAIL this capture (driven via --expect-fail by the caller).
#[cfg(feature = "drift-injection")]
#[test]
fn write_cross_stretch_disabled_capture() {
    maestro_core::sabotage::set_mode(maestro_core::sabotage::Sabotage::StretchDisabled);
    render_scenario(
        &[
            Stem { fixture: "metronome_60.wav", o_mbpm: 60_000, pan: -1.0 },
            Stem { fixture: "metronome_90.wav", o_mbpm: 90_000, pan: 1.0 },
        ],
        "cross_stretch_disabled.wav",
    );
    maestro_core::sabotage::set_mode(maestro_core::sabotage::Sabotage::None);
}

// ---------------------------------------------------------------------------
// C1 `timing` scenario (spec §6.4 scripted sequence), offline edition:
// renders the same command schedule the device harness runs and writes both
// the capture WAV and a harness-shaped report JSON (events + finalSnapshot),
// so maestro-validate's command-log grid analysis is smoke-tested without
// hardware. The stale_reanchor edition must FAIL the analysis.
// ---------------------------------------------------------------------------

/// Frames at 48 kHz for the §6.4 wall-clock script (all multiples of BLOCK).
const T_B_STOP: u64 = 20 * SAMPLE_RATE as u64;
const T_TEMPO: u64 = 30 * SAMPLE_RATE as u64;
const T_B_RESTART: u64 = 35 * SAMPLE_RATE as u64;
const T_VARIATION: u64 = 40 * SAMPLE_RATE as u64;
const TIMING_TEMPO_AFTER: u32 = 100_000;

fn render_timing_scenario(out_name: &str, report_name: &str) {
    let (mut plane, mut core) = create_engine(EngineConfig {
        sample_rate: SAMPLE_RATE,
        initial_tempo_mbpm: TEMPO,
        beats_per_bar: 4,
    })
    .unwrap();

    let stems = [
        Stem { fixture: "metronome_60.wav", o_mbpm: 60_000, pan: -1.0 },
        Stem { fixture: "metronome_90.wav", o_mbpm: 90_000, pan: 1.0 },
    ];
    for (slot, stem) in stems.iter().enumerate() {
        let wav = load_float_wav(&fixture_path(stem.fixture)).unwrap();
        plane
            .load_pcm(
                &LoadMeta {
                    slot: slot as u8,
                    original_tempo_mbpm: stem.o_mbpm,
                    loop_beats: 16,
                    encoder_delay: 0,
                    end_padding: 0,
                    original_samples: wav.frames as u64,
                    variation: 0,
                    gain: 1.0,
                    pan: stem.pan,
                },
                wav.channels[0].clone(),
                wav.channels[1].clone(),
            )
            .unwrap();
        assert!(matches!(
            plane.dispatch(Command::Activate {
                slot: slot as u8,
                quantize: maestro_api::QuantizePolicy::Immediate
            }),
            CommandResult::Ok
        ));
    }
    assert!(matches!(plane.dispatch(Command::Play), CommandResult::Ok));

    let var_wav = load_float_wav(&fixture_path("metronome_60_var.wav")).unwrap();

    let total = CAPTURE_SECONDS * SAMPLE_RATE as u64;
    let mut cap_l = Vec::with_capacity(total as usize);
    let mut cap_r = Vec::with_capacity(total as usize);
    let mut l = vec![0.0f32; BLOCK];
    let mut r = vec![0.0f32; BLOCK];
    let mut rendered = 0u64;
    let mut events = Vec::new();
    while rendered < total {
        match rendered {
            T_B_STOP => {
                assert!(matches!(
                    plane.dispatch(Command::Deactivate { slot: 1 }),
                    CommandResult::Ok
                ));
            }
            T_TEMPO => {
                assert!(matches!(
                    plane.dispatch(Command::SetTempo { mbpm: TIMING_TEMPO_AFTER }),
                    CommandResult::Ok
                ));
            }
            T_B_RESTART => {
                assert!(matches!(
                    plane.dispatch(Command::Activate {
                        slot: 1,
                        quantize: maestro_api::QuantizePolicy::Immediate
                    }),
                    CommandResult::Ok
                ));
            }
            T_VARIATION => {
                plane
                    .load_pcm(
                        &LoadMeta {
                            slot: 0,
                            original_tempo_mbpm: 60_000,
                            loop_beats: 16,
                            encoder_delay: 0,
                            end_padding: 0,
                            original_samples: var_wav.frames as u64,
                            variation: 1,
                            gain: 1.0,
                            pan: -1.0,
                        },
                        var_wav.channels[0].clone(),
                        var_wav.channels[1].clone(),
                    )
                    .expect("variation switch loads");
            }
            _ => {}
        }
        core.render(&mut l, &mut r);
        cap_l.extend_from_slice(&l);
        cap_r.extend_from_slice(&r);
        rendered += BLOCK as u64;
        events.extend(plane.drain_events());
    }

    let snapshot = plane.snapshot();
    events.extend(plane.drain_events());

    let dir = out_dir();
    write_float_wav_stereo(&dir.join(out_name), &cap_l, &cap_r, SAMPLE_RATE).unwrap();
    // Harness-shaped report: the same fields the device writes (spec §6.4).
    let report = serde_json::json!({
        "scenario": "timing",
        "scenarioSeconds": CAPTURE_SECONDS,
        "finalSnapshot": snapshot,
        "events": events,
        "snapshots": [],
    });
    std::fs::write(dir.join(report_name), serde_json::to_string_pretty(&report).unwrap())
        .unwrap();
    println!("wrote {} + {}", dir.join(out_name).display(), dir.join(report_name).display());
}

#[test]
fn write_timing_capture() {
    render_timing_scenario("timing.wav", "timing_report.json");
}

/// The C1 stale-reanchor negative control, offline edition (defect D7: the
/// tempo change anchored from a stale clock). The timing analysis must FAIL
/// this capture+report pair (--expect-fail by the caller): the post-change
/// offsets step by ~1024 frames' worth of musical position.
#[cfg(feature = "drift-injection")]
#[test]
fn write_timing_stale_reanchor_capture() {
    maestro_core::sabotage::set_mode(maestro_core::sabotage::Sabotage::StaleReanchor);
    render_timing_scenario("timing_stale_reanchor.wav", "timing_stale_reanchor_report.json");
    maestro_core::sabotage::set_mode(maestro_core::sabotage::Sabotage::None);
}
