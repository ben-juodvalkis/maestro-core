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
                    gain: 1.0,
                    pan: stem.pan,
                },
                wav.channels[0].clone(),
                wav.channels[1].clone(),
            )
            .unwrap();
        assert!(matches!(plane.dispatch(Command::Activate { slot: slot as u8 }), CommandResult::Ok));
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
