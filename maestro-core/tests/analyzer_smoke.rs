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
//! - `unitycross.wav` — metronome_120 stems through the §6.4 timing script
//!   with a 120→117 change (unity-crossing — the ADR-094 pin; see below)
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
                    sample_rate: 48_000,
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
        assert!(matches!(plane.dispatch(Command::Activate { slot: slot as u8, quantize: maestro_api::QuantizePolicy::Immediate , transport: Default::default() }), CommandResult::Ok));
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


/// The sabotage mode is process-global (`maestro_core::sabotage`), so the
/// capture-writing tests must not run concurrently: a clean render racing a
/// sabotaged one inherits its mode and the clean capture fails analysis
/// (observed as an order-dependent flake at M3). Every test in this file
/// takes this lock first; a poisoned lock is fine (the panicking test
/// already failed the suite).
/// Test-orchestration lock, never engine code — the §3.3 no-locks rule (and
/// the clippy disallowed-types gate enforcing it) protects the render path.
#[allow(clippy::disallowed_types)]
static RENDER_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[allow(clippy::disallowed_types)]
fn render_serial() -> std::sync::MutexGuard<'static, ()> {
    RENDER_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

#[test]
fn write_unity_capture() {
    let _serial = render_serial();
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
    let _serial = render_serial();
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
    let _serial = render_serial();
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

/// One §6.4 timing-script parameterization (the same command schedule, two
/// stem/tempo geometries): the C1 `timing` scenario (60/90-BPM stems, change
/// 120→100 — both sides stretch) and the ADR-094 `unitycross` scenario
/// (metronome_120 stems, change 120→117 — the change crosses the unity
/// boundary, the C3 finding's geometry).
struct TimingScript {
    stems: [Stem; 2],
    tempo_after: u32,
    var_fixture: &'static str,
    var_o_mbpm: u32,
    scenario_label: &'static str,
}

fn c1_timing_script() -> TimingScript {
    TimingScript {
        stems: [
            Stem { fixture: "metronome_60.wav", o_mbpm: 60_000, pan: -1.0 },
            Stem { fixture: "metronome_90.wav", o_mbpm: 90_000, pan: 1.0 },
        ],
        tempo_after: TIMING_TEMPO_AFTER,
        var_fixture: "metronome_60_var.wav",
        var_o_mbpm: 60_000,
        scenario_label: "timing",
    }
}

fn render_timing_scenario(out_name: &str, report_name: &str) {
    render_timing_script(&c1_timing_script(), out_name, report_name);
}

fn render_timing_script(script: &TimingScript, out_name: &str, report_name: &str) {
    let (mut plane, mut core) = create_engine(EngineConfig {
        sample_rate: SAMPLE_RATE,
        initial_tempo_mbpm: TEMPO,
        beats_per_bar: 4,
    })
    .unwrap();

    let stems = &script.stems;
    for (slot, stem) in stems.iter().enumerate() {
        let wav = load_float_wav(&fixture_path(stem.fixture)).unwrap();
        plane
            .load_pcm(
                &LoadMeta {
                    sample_rate: 48_000,
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
            , transport: Default::default() }),
            CommandResult::Ok
        ));
    }
    assert!(matches!(plane.dispatch(Command::Play), CommandResult::Ok));

    let var_wav = load_float_wav(&fixture_path(script.var_fixture)).unwrap();

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
                    plane.dispatch(Command::Deactivate { slot: 1, transport: Default::default() }),
                    CommandResult::Ok
                ));
            }
            T_TEMPO => {
                assert!(matches!(
                    plane.dispatch(Command::SetTempo { mbpm: script.tempo_after }),
                    CommandResult::Ok
                ));
            }
            T_B_RESTART => {
                assert!(matches!(
                    plane.dispatch(Command::Activate {
                        slot: 1,
                        quantize: maestro_api::QuantizePolicy::Immediate
                    , transport: Default::default() }),
                    CommandResult::Ok
                ));
            }
            T_VARIATION => {
                plane
                    .load_pcm(
                        &LoadMeta {
                            sample_rate: 48_000,
                            slot: 0,
                            original_tempo_mbpm: script.var_o_mbpm,
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
        "scenario": script.scenario_label,
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
    let _serial = render_serial();
    render_timing_scenario("timing.wav", "timing_report.json");
}

/// ADR-094 device-loop smoke: the same §6.4 script with the tempo change
/// crossing the UNITY boundary — metronome_120 stems (O = engine tempo) and
/// 120→117, the UI tempo button's single step (ratio 40/39: the offset
/// pattern is aperiodic; the analyzer's window-spread fallback covers it,
/// ADR-093). Analyzed with `--stem-tempo-mbpm 120000`: the expected step
/// across the change is STRETCH_PATH_PREROLL = 0 — the compensated engine
/// passes, an uncompensated one measures 2880 and fails.
#[test]
fn write_unitycross_capture() {
    let _serial = render_serial();
    render_timing_script(
        &TimingScript {
            stems: [
                Stem { fixture: "metronome_120.wav", o_mbpm: 120_000, pan: -1.0 },
                Stem { fixture: "metronome_120.wav", o_mbpm: 120_000, pan: 1.0 },
            ],
            tempo_after: 117_000,
            var_fixture: "metronome_120_var.wav",
            var_o_mbpm: 120_000,
            scenario_label: "unitycross",
        },
        "unitycross.wav",
        "unitycross_report.json",
    );
}

/// The C1 stale-reanchor negative control, offline edition (defect D7: the
/// tempo change anchored from a stale clock). The timing analysis must FAIL
/// this capture+report pair (--expect-fail by the caller): the post-change
/// offsets step by ~1024 frames' worth of musical position.
#[cfg(feature = "drift-injection")]
#[test]
fn write_timing_stale_reanchor_capture() {
    let _serial = render_serial();
    maestro_core::sabotage::set_mode(maestro_core::sabotage::Sabotage::StaleReanchor);
    render_timing_scenario("timing_stale_reanchor.wav", "timing_stale_reanchor_report.json");
    maestro_core::sabotage::set_mode(maestro_core::sabotage::Sabotage::None);
}

// ---------------------------------------------------------------------------
// C2 `control` scenario (the full dispatch-only flow), offline edition:
// mute/unmute B, solo/unsolo A, tempo change, async variation switch
// (VariationLoadRequired answered via load_pcm), pause/resume. The capture
// mirrors the device shape: paused blocks still append (silent) capture
// frames while F is frozen, so the analyzer's two-segment alignment is
// exercised offline. The stale_reanchor edition must FAIL the analysis.
// ---------------------------------------------------------------------------

const C_MUTE_ON: u64 = 12 * SAMPLE_RATE as u64;
const C_MUTE_OFF: u64 = 18 * SAMPLE_RATE as u64;
const C_SOLO_ON: u64 = 24 * SAMPLE_RATE as u64;
const C_SOLO_OFF: u64 = 30 * SAMPLE_RATE as u64;
const C_TEMPO: u64 = 36 * SAMPLE_RATE as u64;
const C_VARIATION: u64 = 42 * SAMPLE_RATE as u64;
const C_PAUSE: u64 = 52 * SAMPLE_RATE as u64;
const C_RESUME: u64 = 56 * SAMPLE_RATE as u64;

fn render_control_scenario(out_name: &str, report_name: &str) {
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
                    sample_rate: 48_000,
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
                quantize: maestro_api::QuantizePolicy::Immediate,
                transport: Default::default()
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
    let mut captured = 0u64; // capture frames (wall-clock shape, incl. pause)
    let mut events = Vec::new();
    while captured < total {
        match captured {
            C_MUTE_ON => {
                assert!(matches!(
                    plane.dispatch(Command::SetMute { slot: 1, muted: true }),
                    CommandResult::Ok
                ));
            }
            C_MUTE_OFF => {
                assert!(matches!(
                    plane.dispatch(Command::SetMute { slot: 1, muted: false }),
                    CommandResult::Ok
                ));
            }
            C_SOLO_ON => {
                assert!(matches!(
                    plane.dispatch(Command::SetSolo { slot: 0, solo: true }),
                    CommandResult::Ok
                ));
            }
            C_SOLO_OFF => {
                assert!(matches!(
                    plane.dispatch(Command::SetSolo { slot: 0, solo: false }),
                    CommandResult::Ok
                ));
            }
            C_TEMPO => {
                assert!(matches!(
                    plane.dispatch(Command::SetTempo { mbpm: 100_000 }),
                    CommandResult::Ok
                ));
            }
            C_VARIATION => {
                // §4.11 async switch through the command surface.
                assert!(matches!(
                    plane.dispatch(Command::SwitchVariation { slot: 0, variation: 1 }),
                    CommandResult::Ok
                ));
            }
            C_PAUSE => {
                assert!(matches!(plane.dispatch(Command::Pause), CommandResult::Ok));
            }
            C_RESUME => {
                assert!(matches!(plane.dispatch(Command::Resume), CommandResult::Ok));
            }
            _ => {}
        }
        core.render(&mut l, &mut r);
        cap_l.extend_from_slice(&l);
        cap_r.extend_from_slice(&r);
        captured += BLOCK as u64;
        for e in plane.drain_events() {
            // The platform's job (§4.11): answer the async load request.
            if let maestro_api::Event::VariationLoadRequired { slot, variation } = e {
                plane
                    .load_pcm(
                        &LoadMeta {
                            sample_rate: 48_000,
                            slot,
                            original_tempo_mbpm: 60_000,
                            loop_beats: 16,
                            encoder_delay: 0,
                            end_padding: 0,
                            original_samples: var_wav.frames as u64,
                            variation,
                            gain: 1.0,
                            pan: -1.0,
                        },
                        var_wav.channels[0].clone(),
                        var_wav.channels[1].clone(),
                    )
                    .expect("async variation load completes");
            }
            events.push(e);
        }
    }

    let snapshot = plane.snapshot();
    events.extend(plane.drain_events());

    let dir = out_dir();
    write_float_wav_stereo(&dir.join(out_name), &cap_l, &cap_r, SAMPLE_RATE).unwrap();
    let report = serde_json::json!({
        "scenario": "control",
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
fn write_control_capture() {
    let _serial = render_serial();
    render_control_scenario("control.wav", "control_report.json");
}

/// The C2 negative control, offline edition: stale_reanchor corrupts the
/// scripted tempo change — the control analysis must FAIL this pair
/// (--expect-fail by the caller).
#[cfg(feature = "drift-injection")]
#[test]
fn write_control_stale_reanchor_capture() {
    let _serial = render_serial();
    maestro_core::sabotage::set_mode(maestro_core::sabotage::Sabotage::StaleReanchor);
    render_control_scenario("control_stale_reanchor.wav", "control_stale_reanchor_report.json");
    maestro_core::sabotage::set_mode(maestro_core::sabotage::Sabotage::None);
}
