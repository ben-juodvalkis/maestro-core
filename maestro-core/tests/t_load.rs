//! # T-LOAD — decode/trim validation (spec §6.5, every commit)
//!
//! The §3.4 invariant under test: **by the time PCM reaches the engine, its
//! length is exactly consistent with the stem's musical identity, or the load
//! fails loudly.** No code path silently pads, trims heuristically, or
//! resamples — the last three fixtures below are the negative controls that
//! prove the assertions can fail (spec §6.1).
//!
//! Fixture matrix (spec §6.5):
//! 1. correctly-tagged M4A (real AAC, decoded here with **symphonia** — a
//!    third decoder, so reconciliation is proven decoder-agnostic, not tuned
//!    to AVAudioFile or ffmpeg);
//! 2. decoder-pre-trimmed (simulated by length) → `AlreadyTrimmed`;
//! 3. raw decode (+delay+padding length) → `TrimmedHere`, with the trim
//!    WINDOW (not just the length) verified bit-exact through the render path;
//! 3. (b) delay-only trim (+padding length) → `TrimmedPadding` — Chrome's
//!    `decodeAudioData` behavior, found by the gate firing on the first M4
//!    in-app WASM run (ADR-095); same bit-exact window verification;
//! 4. wrong-length file → `TrackLoadFailed` naming every expected length;
//! 5. variation set with one mismatched length → hard identity failure;
//! 6. non-48k file → hard failure (the engine never resamples).
//!
//! Plus the M2 §3.4#4 extension: the reconciliation case is asserted STABLE
//! across variations of a stem — same decoder + same encoder must reconcile
//! the same way; a flip is a mis-tagged file, hard failure.

mod common;

use std::fs::File;
use std::path::Path;

use common::{detect_click_onsets, fixture_path, load_float_wav, make_sine_pcm, SAMPLE_RATE};
use maestro_api::{Command, CommandResult, EngineConfig, Event, LoadMeta, QuantizePolicy};
use maestro_core::control_plane::{create_engine, ControlPlane, Reconciliation};
use maestro_core::audio_core::AudioCore;

use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

/// Measured packet table of `tests/fixtures/metronome_60.m4a` (afinfo
/// read-back at fixture creation — per-FILE measured values, spec §3.4#1):
/// `audio 768000 valid frames + 2112 priming + 960 remainder = 771072`.
const M4A_ORIGINAL_SAMPLES: u64 = 768_000;
const M4A_ENCODER_DELAY: u32 = 2_112;
const M4A_END_PADDING: u32 = 960;
/// The stem's musical identity: 16 beats @ 60 BPM @ 48 kHz = exactly 768 000.
const M4A_O_MBPM: u32 = 60_000;
const M4A_LOOP_BEATS: u32 = 16;

/// Measured packet table of `tests/fixtures/metronome_60_44k.m4a`
/// (the non-48k negative control): 705600 valid + 2112 + 896 @ 44100 Hz.
const M4A_44K_VALID: u64 = 705_600;
const M4A_44K_DELAY: u32 = 2_112;
const M4A_44K_PADDING: u32 = 896;
const M4A_44K_RATE: u32 = 44_100;

fn engine() -> (ControlPlane, Box<AudioCore>) {
    create_engine(EngineConfig { pcm_budget_bytes: None,
        sample_rate: SAMPLE_RATE,
        initial_tempo_mbpm: 120_000,
        beats_per_bar: 4,
    })
    .unwrap()
}

fn meta(slot: u8, variation: u8, original_samples: u64) -> LoadMeta {
    LoadMeta { cache_only: false,
        slot,
        sample_rate: SAMPLE_RATE,
        original_tempo_mbpm: M4A_O_MBPM,
        loop_beats: M4A_LOOP_BEATS,
        encoder_delay: M4A_ENCODER_DELAY,
        end_padding: M4A_END_PADDING,
        original_samples,
        variation,
        gain: 1.0,
        pan: 0.0,
    }
}

/// Decode an M4A fixture with symphonia. `gapless` selects symphonia's own
/// edit-list handling — T-LOAD runs BOTH settings through reconciliation,
/// which must classify each decode by its length alone (decoder-agnostic).
fn decode_m4a(path: &Path, gapless: bool) -> (Vec<f32>, Vec<f32>, u32) {
    let file = File::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    hint.with_extension("m4a");
    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions { enable_gapless: gapless, ..Default::default() },
            &MetadataOptions::default(),
        )
        .expect("probe m4a");
    let mut format = probed.format;
    let track = format.default_track().expect("audio track");
    let track_id = track.id;
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .expect("aac decoder");
    let mut left = Vec::new();
    let mut right = Vec::new();
    let mut rate = 0u32;
    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(SymphoniaError::IoError(e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break
            }
            Err(e) => panic!("packet read: {e}"),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = decoder.decode(&packet).expect("aac decode");
        let spec = *decoded.spec();
        rate = spec.rate;
        assert_eq!(spec.channels.count(), 2, "fixture must be stereo");
        let mut buf = symphonia::core::audio::SampleBuffer::<f32>::new(
            decoded.capacity() as u64,
            spec,
        );
        buf.copy_interleaved_ref(decoded);
        for frame in buf.samples().chunks_exact(2) {
            left.push(frame[0]);
            right.push(frame[1]);
        }
    }
    (left, right, rate)
}

fn expect_load_failed_event(plane: &mut ControlPlane, slot: u8, needle: &str) {
    let events = plane.drain_events();
    let reason = events
        .iter()
        .find_map(|e| match e {
            Event::TrackLoadFailed { slot: s, reason } if *s == slot => Some(reason.clone()),
            _ => None,
        })
        .unwrap_or_else(|| panic!("missing TrackLoadFailed for slot {slot}; events: {events:?}"));
    assert!(
        reason.contains(needle),
        "TrackLoadFailed reason must name the precise cause (`{needle}`): {reason}"
    );
}

// ---------------------------------------------------------------------------
// 1. Correctly-tagged M4A — real AAC through a third decoder.
// ---------------------------------------------------------------------------

#[test]
fn correctly_tagged_m4a_reconciles_for_both_decoder_behaviors() {
    let path = fixture_path("metronome_60.m4a");
    let wav = load_float_wav(&fixture_path("metronome_60.wav")).expect("source fixture");

    for gapless in [false, true] {
        let (left, right, rate) = decode_m4a(&path, gapless);
        assert_eq!(rate, SAMPLE_RATE);
        let l_dec = left.len() as u64;

        // The decode must land on one of the two §3.4 lengths — which one is
        // the DECODER's business; reconciliation classifies by length alone.
        let raw_len = M4A_ORIGINAL_SAMPLES + (M4A_ENCODER_DELAY + M4A_END_PADDING) as u64;
        assert!(
            l_dec == M4A_ORIGINAL_SAMPLES || l_dec == raw_len,
            "symphonia (gapless={gapless}) produced {l_dec} frames — neither §3.4 case"
        );

        // Anti-stub (spec §6.1): the trimmed audio must contain the exact
        // metronome click train at the exact positions of the source WAV —
        // this fails if the trim window is off by even one AAC packet.
        let trim = if l_dec == raw_len { M4A_ENCODER_DELAY as usize } else { 0 };
        let trimmed = &left[trim..trim + M4A_ORIGINAL_SAMPLES as usize];
        let min_gap = SAMPLE_RATE as usize / 2;
        let dec_onsets = detect_click_onsets(trimmed, 0.10, min_gap);
        let wav_onsets = detect_click_onsets(&wav.channels[0], 0.10, min_gap);
        assert_eq!(dec_onsets.len(), wav_onsets.len(), "click count mismatch after trim");
        for (d, w) in dec_onsets.iter().zip(&wav_onsets) {
            let diff = (*d as i64 - *w as i64).abs();
            assert!(
                diff <= 48,
                "decoded click at {d} vs source {w} (off {diff} samples) — trim window wrong"
            );
        }

        // Through the engine: the load must succeed and report the case that
        // matches the decode length.
        let (mut plane, _core) = engine();
        let rec = plane
            .load_pcm(&meta(0, 0, M4A_ORIGINAL_SAMPLES), left, right)
            .expect("correctly-tagged fixture must load");
        let expected = if l_dec == raw_len {
            Reconciliation::TrimmedHere
        } else {
            Reconciliation::AlreadyTrimmed
        };
        assert_eq!(rec, expected, "reconciliation case must match decode length (gapless={gapless})");
        let events = plane.drain_events();
        assert!(
            events.iter().any(|e| matches!(
                e,
                Event::TrackLoaded { slot: 0, variation: 0, reconciliation } if !reconciliation.is_empty()
            )),
            "TrackLoaded event must carry the reconciliation case"
        );
    }
}

// ---------------------------------------------------------------------------
// 2./3. Length-simulated decoder behaviors, with the trim WINDOW verified
// bit-exact through the real render path (not just the length).
// ---------------------------------------------------------------------------

#[test]
fn pre_trimmed_length_reconciles_already_trimmed() {
    let (mut plane, _core) = engine();
    let (l, r) = make_sine_pcm(M4A_ORIGINAL_SAMPLES, 220.0);
    let rec = plane.load_pcm(&meta(0, 0, M4A_ORIGINAL_SAMPLES), l, r).expect("loads");
    assert_eq!(rec, Reconciliation::AlreadyTrimmed);
}

#[test]
fn raw_length_trims_exactly_the_measured_window() {
    let (mut plane, mut core) = engine();
    // Ramp content makes any off-by-one trim visible: pcm[i] = (i % 9601)/9600
    // scaled small. After the exact §3.4 trim, sample k of the loop must be
    // ramp[encoder_delay + k] — verified bit-exact through the unity bypass.
    let raw_len = M4A_ORIGINAL_SAMPLES + (M4A_ENCODER_DELAY + M4A_END_PADDING) as u64;
    let ramp: Vec<f32> = (0..raw_len).map(|i| (i % 9601) as f32 / 96_000.0).collect();
    let mut m = meta(0, 0, M4A_ORIGINAL_SAMPLES);
    m.original_tempo_mbpm = 120_000; // unity at the engine tempo → bypass
    m.loop_beats = 32; // 32 beats @ 120 = 768_000 exactly
    let rec = plane.load_pcm(&m, ramp.clone(), ramp.clone()).expect("raw-length loads");
    assert_eq!(rec, Reconciliation::TrimmedHere);

    assert!(matches!(
        plane.dispatch(Command::Activate {
            slot: 0,
            quantize: QuantizePolicy::Immediate,
            transport: Default::default()
        }),
        CommandResult::Ok
    ));
    assert!(matches!(plane.dispatch(Command::Play), CommandResult::Ok));
    let mut left = vec![0.0f32; 480];
    let mut right = vec![0.0f32; 480];
    core.render(&mut left, &mut right); // applies commands; first audible block
    // Past the activation ramp:
    for _ in 0..10 {
        core.render(&mut left, &mut right);
    }
    let gl = (std::f32::consts::FRAC_PI_4).cos(); // pan 0, gain 1
    let n0 = 10 * 480u64; // dictated source sample of this block at unity
    for i in 0..480usize {
        let src = ramp[(M4A_ENCODER_DELAY as u64 + n0 + i as u64) as usize];
        assert_eq!(
            left[i].to_bits(),
            (src * gl).to_bits(),
            "trim window wrong at loop sample {} — §3.4 demands the exact span",
            n0 + i as u64
        );
    }
    drop(plane);
}

#[test]
fn padded_length_trims_exactly_the_end_padding() {
    // Chrome's decodeAudioData: priming stripped, end padding left in place
    // (decoded == original + end_padding). Found in production geometry by
    // the §3.4#4 gate itself on the first in-app M4 WASM run (ADR-095): the
    // Metronome Test default decoded 384960 = 384000 + 960. Same anti-stub
    // shape as the raw-length case: ramp content, bit-exact window through
    // the unity bypass — sample k of the loop must be ramp[k] (NO delay
    // offset; the decoder already removed it).
    let (mut plane, mut core) = engine();
    let padded_len = M4A_ORIGINAL_SAMPLES + M4A_END_PADDING as u64;
    let ramp: Vec<f32> = (0..padded_len).map(|i| (i % 9601) as f32 / 96_000.0).collect();
    let mut m = meta(0, 0, M4A_ORIGINAL_SAMPLES);
    m.original_tempo_mbpm = 120_000; // unity at the engine tempo → bypass
    m.loop_beats = 32; // 32 beats @ 120 = 768_000 exactly
    let rec = plane.load_pcm(&m, ramp.clone(), ramp.clone()).expect("padded-length loads");
    assert_eq!(rec, Reconciliation::TrimmedPadding);

    assert!(matches!(
        plane.dispatch(Command::Activate {
            slot: 0,
            quantize: QuantizePolicy::Immediate,
            transport: Default::default()
        }),
        CommandResult::Ok
    ));
    assert!(matches!(plane.dispatch(Command::Play), CommandResult::Ok));
    let mut left = vec![0.0f32; 480];
    let mut right = vec![0.0f32; 480];
    core.render(&mut left, &mut right); // applies commands; first audible block
    for _ in 0..10 {
        core.render(&mut left, &mut right);
    }
    let gl = (std::f32::consts::FRAC_PI_4).cos(); // pan 0, gain 1
    let n0 = 10 * 480u64;
    for i in 0..480usize {
        let src = ramp[(n0 + i as u64) as usize];
        assert_eq!(
            left[i].to_bits(),
            (src * gl).to_bits(),
            "padding-only trim window wrong at loop sample {} — §3.4 demands the exact span",
            n0 + i as u64
        );
    }
    drop(plane);
}

// ---------------------------------------------------------------------------
// 4./5./6. The negative controls (spec §6.1: assertions must be able to fail).
// ---------------------------------------------------------------------------

#[test]
fn wrong_length_fails_with_both_expected_lengths_named() {
    let (mut plane, _core) = engine();
    // The historic double-trim corruption: 3072 samples short (defect D4).
    let (l, r) = make_sine_pcm(M4A_ORIGINAL_SAMPLES - 3_072, 220.0);
    let err = plane.load_pcm(&meta(0, 0, M4A_ORIGINAL_SAMPLES), l, r).unwrap_err();
    assert!(err.contains("length mismatch"), "precise reason required: {err}");
    assert!(
        err.contains(&M4A_ORIGINAL_SAMPLES.to_string())
            && err.contains(&(M4A_ORIGINAL_SAMPLES + 3_072).to_string()),
        "both §3.4 expected lengths must be named: {err}"
    );
    expect_load_failed_event(&mut plane, 0, "length mismatch");

    // One sample off fails too — no heuristic pad/trim anywhere.
    let (l, r) = make_sine_pcm(M4A_ORIGINAL_SAMPLES + 1, 220.0);
    assert!(plane.load_pcm(&meta(0, 0, M4A_ORIGINAL_SAMPLES), l, r).is_err());

    // One sample past the delay-only-trim case (ADR-095) fails identically —
    // the third case is an exact equality, not a tolerance.
    let (l, r) = make_sine_pcm(M4A_ORIGINAL_SAMPLES + M4A_END_PADDING as u64 + 1, 220.0);
    assert!(plane.load_pcm(&meta(0, 0, M4A_ORIGINAL_SAMPLES), l, r).is_err());
}

#[test]
fn variation_set_with_mismatched_length_fails() {
    let (mut plane, _core) = engine();
    let (l, r) = make_sine_pcm(M4A_ORIGINAL_SAMPLES, 220.0);
    plane.load_pcm(&meta(0, 0, M4A_ORIGINAL_SAMPLES), l, r).expect("base variation loads");

    // Variation 1 declares a self-consistent but DIFFERENT identity
    // (8 beats @ 60 BPM = exactly 384_000): variation identity mismatch,
    // hard failure (loop length can never change underneath the timeline).
    let (l, r) = make_sine_pcm(384_000, 440.0);
    let mut m = meta(0, 1, 384_000);
    m.loop_beats = 8;
    let err = plane.load_pcm(&m, l, r).unwrap_err();
    assert!(err.contains("identity mismatch"), "{err}");
    expect_load_failed_event(&mut plane, 0, "identity");

    // Variation 1 whose declared length contradicts its own musical identity
    // (768_000 − 1_024 against 16 beats @ 60): loop-identity failure — the
    // other precise §3.4 reason.
    let (l, r) = make_sine_pcm(M4A_ORIGINAL_SAMPLES - 1_024, 440.0);
    let err = plane
        .load_pcm(&meta(0, 1, M4A_ORIGINAL_SAMPLES - 1_024), l, r)
        .unwrap_err();
    assert!(err.contains("inconsistent with musical loop length"), "{err}");

    // Variation 1 declares the right identity but its PCM is short: §3.4
    // length mismatch, also hard.
    let (l, r) = make_sine_pcm(M4A_ORIGINAL_SAMPLES - 1_024, 440.0);
    let err = plane.load_pcm(&meta(0, 1, M4A_ORIGINAL_SAMPLES), l, r).unwrap_err();
    assert!(err.contains("length mismatch"), "{err}");
}

#[test]
fn non_48k_file_fails_loudly() {
    // The real 44.1 kHz fixture through the real decoder: the reported rate
    // must hard-fail the load. Nothing resamples, ever (spec §3.4#3 / §4.7).
    let (left, right, rate) = decode_m4a(&fixture_path("metronome_60_44k.m4a"), false);
    assert_eq!(rate, M4A_44K_RATE, "fixture must decode at its native 44.1 kHz");
    let l_dec = left.len() as u64;
    assert!(
        l_dec == M4A_44K_VALID || l_dec == M4A_44K_VALID + (M4A_44K_DELAY + M4A_44K_PADDING) as u64,
        "decode length {l_dec} matches neither case — fixture metadata stale"
    );

    let (mut plane, _core) = engine();
    let mut m = LoadMeta { cache_only: false,
        slot: 0,
        sample_rate: rate,
        original_tempo_mbpm: M4A_O_MBPM,
        loop_beats: M4A_LOOP_BEATS,
        encoder_delay: M4A_44K_DELAY,
        end_padding: M4A_44K_PADDING,
        original_samples: M4A_44K_VALID,
        variation: 0,
        gain: 1.0,
        pan: 0.0,
    };
    let err = plane.load_pcm(&m, left, right).unwrap_err();
    assert!(
        err.contains("sample rate") && err.contains("44100"),
        "precise reason required: {err}"
    );
    expect_load_failed_event(&mut plane, 0, "sample rate");

    // Even a well-formed 48k payload is rejected if the shim REPORTS another
    // rate — the assertion is on the report, not on luck.
    let (l, r) = make_sine_pcm(M4A_ORIGINAL_SAMPLES, 220.0);
    m.sample_rate = 44_100;
    m.original_samples = M4A_ORIGINAL_SAMPLES;
    m.encoder_delay = M4A_ENCODER_DELAY;
    m.end_padding = M4A_END_PADDING;
    assert!(plane.load_pcm(&m, l, r).is_err());
}

// ---------------------------------------------------------------------------
// M2 §3.4#4 extension: reconciliation stability across variations.
// ---------------------------------------------------------------------------

#[test]
fn reconciliation_must_be_stable_across_variations() {
    let (mut plane, _core) = engine();
    // Variation 0 arrives pre-trimmed (AlreadyTrimmed)…
    let (l, r) = make_sine_pcm(M4A_ORIGINAL_SAMPLES, 220.0);
    let rec = plane.load_pcm(&meta(0, 0, M4A_ORIGINAL_SAMPLES), l, r).expect("loads");
    assert_eq!(rec, Reconciliation::AlreadyTrimmed);

    // …variation 1 arrives raw-shaped (TrimmedHere): same decoder + same
    // encoder cannot legitimately flip cases — mis-tagged file, hard failure.
    let raw_len = M4A_ORIGINAL_SAMPLES + (M4A_ENCODER_DELAY + M4A_END_PADDING) as u64;
    let (l, r) = make_sine_pcm(raw_len, 440.0);
    let err = plane.load_pcm(&meta(0, 1, M4A_ORIGINAL_SAMPLES), l, r).unwrap_err();
    assert!(err.contains("stable"), "stability violation must be named: {err}");
    expect_load_failed_event(&mut plane, 0, "stable");

    // A consistent variation 1 still loads fine afterwards.
    let (l, r) = make_sine_pcm(M4A_ORIGINAL_SAMPLES, 440.0);
    assert!(plane.load_pcm(&meta(0, 1, M4A_ORIGINAL_SAMPLES), l, r).is_ok());
}
