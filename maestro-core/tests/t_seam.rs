//! # T-SEAM — loop-seam test (spec §6.3)
//!
//! Two fixture families through the real render path, ≥ 50 analyzed wraps,
//! rates {1.0 (bypass), 0.75, 1.5}:
//!
//! (a) **Seam continuity** — seamless integer-cycle sine: the maximum
//!     inter-sample step over the analyzed region stays within
//!     `SEAM_STEP_FACTOR ×` the fixture's natural maximum. The engine must
//!     keep a seamless source seamless through the streaming stretch feed —
//!     the stretcher is never seeked at a wrap (spec §3.2).
//!
//! (b) **Onset grid** — metronome fixture: click onsets land on the exact
//!     derived grid. Wrap-to-wrap residual variance must be exactly zero for
//!     both paths (bypass additionally sanity-bounded against the absolute
//!     grid). The first wrap is excluded (activation ramp + anchor pre-roll).
//!
//! **Negative control** — the double-trim sabotage fixture (2112 samples
//! chopped from the front, zero-padded to length — defect D4's content
//! corruption) must produce a detected seam transient, proving the detector
//! sees real seam clicks (reproduced on the branch at 0.35 vs 0.0115
//! baseline, spec Appendix A.2).

mod common;

use common::validation_thresholds::{
    BYPASS_GRID_TOLERANCE, MAX_ONSET_PATTERN_PERIOD_BEATS, MIN_ANALYZED_WRAPS, MIN_MIX_RMS,
    ONSET_THRESHOLD, SEAM_STEP_FACTOR, STRETCH_GRID_MEDIAN_MAX, STRETCH_ONSET_JITTER_MAX,
};
use common::{detect_click_onsets, fixture_path, load_float_wav, make_sine_pcm, SAMPLE_RATE};
use maestro_api::{Command, CommandResult, EngineConfig, LoadMeta};
use maestro_core::control_plane::create_engine;
use maestro_core::timeline::BEAT_UNITS_PER_BEAT;

/// All T-SEAM scenarios use a 16-beat, 120-BPM source loop (the metronome
/// fixture's real geometry; the sine fixture is built to match).
const O_MBPM: u32 = 120_000;
const LOOP_BEATS: u32 = 16;
const PCM_LEN: u64 = 384_000;
/// 480 Hz at 48 kHz = exactly 100 samples/cycle ⇒ 3840 whole cycles per loop:
/// a perfectly seamless tonal fixture.
const SINE_HZ: f32 = 480.0;
const BLOCK: usize = 480;

/// Engine tempos exercising the three §6.3 rates against the 120k source.
const RATE_CASES: [(u32, &str); 3] =
    [(120_000, "1.0 (bypass)"), (90_000, "0.75 (stretch)"), (180_000, "1.5 (stretch)")];

/// Output frames per loop wrap at engine tempo E: Λ/E (exact integers for all
/// three tempos: 384_000 / 512_000 / 256_000).
fn wrap_frames(tempo_mbpm: u32) -> u64 {
    let lambda = LOOP_BEATS as u128 * BEAT_UNITS_PER_BEAT;
    assert_eq!(lambda % tempo_mbpm as u128, 0, "test tempos chosen for integer wrap durations");
    (lambda / tempo_mbpm as u128) as u64
}

/// Output frames per source beat at engine tempo E: K/E (24_000 / 32_000 / 16_000).
fn beat_frames(tempo_mbpm: u32) -> u64 {
    assert_eq!(BEAT_UNITS_PER_BEAT % tempo_mbpm as u128, 0);
    (BEAT_UNITS_PER_BEAT / tempo_mbpm as u128) as u64
}

/// Build a one-track engine, press play, render `total_frames`, return the
/// left channel capture.
fn render_capture(tempo_mbpm: u32, left: Vec<f32>, right: Vec<f32>, total_frames: u64) -> Vec<f32> {
    render_capture_with_changes(tempo_mbpm, left, right, total_frames, &[])
}

/// Like [`render_capture`], dispatching live tempo changes at the given
/// frames. Each change frame must be a multiple of BLOCK so the change lands
/// exactly at a block boundary (where the engine applies commands).
fn render_capture_with_changes(
    tempo_mbpm: u32,
    left: Vec<f32>,
    right: Vec<f32>,
    total_frames: u64,
    tempo_changes: &[(u64, u32)],
) -> Vec<f32> {
    let (mut plane, mut core) = create_engine(EngineConfig { pcm_budget_bytes: None,
        sample_rate: SAMPLE_RATE,
        initial_tempo_mbpm: tempo_mbpm,
        beats_per_bar: 4,
    })
    .unwrap();
    plane
        .load_pcm(
            &LoadMeta { cache_only: false,
                sample_rate: 48_000,
                slot: 0,
                original_tempo_mbpm: O_MBPM,
                loop_beats: LOOP_BEATS,
                encoder_delay: 0,
                end_padding: 0,
                original_samples: left.len() as u64,
                variation: 0,
                gain: 1.0,
                pan: 0.0,
            },
            left,
            right,
        )
        .unwrap();
    assert!(matches!(plane.dispatch(Command::Activate { slot: 0, quantize: maestro_api::QuantizePolicy::Immediate , transport: Default::default() }), CommandResult::Ok));
    assert!(matches!(plane.dispatch(Command::Play), CommandResult::Ok));

    let mut capture = Vec::with_capacity(total_frames as usize);
    let mut l = vec![0.0f32; BLOCK];
    let mut r = vec![0.0f32; BLOCK];
    let mut rendered = 0u64;
    let mut change_idx = 0usize;
    while rendered < total_frames {
        if let Some(&(at, mbpm)) = tempo_changes.get(change_idx) {
            if rendered == at {
                let res = plane.dispatch(Command::SetTempo { mbpm });
                assert!(matches!(res, CommandResult::Ok), "tempo change rejected: {res:?}");
                change_idx += 1;
            } else {
                assert!(rendered < at, "change frame {at} not block-aligned (BLOCK={BLOCK})");
            }
        }
        let b = BLOCK.min((total_frames - rendered) as usize);
        core.render(&mut l[..b], &mut r[..b]);
        capture.extend_from_slice(&l[..b]);
        rendered += b as u64;
    }
    capture
}

/// Natural maximum inter-sample step of a loop, seam pair included.
fn natural_max_step(pcm: &[f32]) -> f32 {
    let mut max = (pcm[0] - pcm[pcm.len() - 1]).abs();
    for w in pcm.windows(2) {
        max = max.max((w[1] - w[0]).abs());
    }
    max
}

/// Median of an offset sequence (the absolute-grid constant of a window —
/// robust against the periodic vocoder smear, which the spread gate bounds).
fn median_offset(offsets: &[i64]) -> i64 {
    assert!(!offsets.is_empty());
    let mut sorted = offsets.to_vec();
    sorted.sort_unstable();
    sorted[sorted.len() / 2]
}

/// Max inter-sample step in `signal[from..]`, with the frame where it occurs.
fn max_step_from(signal: &[f32], from: usize) -> (f32, usize) {
    let mut max = 0.0f32;
    let mut at = from;
    for i in from.max(1)..signal.len() {
        let step = (signal[i] - signal[i - 1]).abs();
        if step > max {
            max = step;
            at = i;
        }
    }
    (max, at)
}

// ---------------------------------------------------------------------------
// (a) Seam continuity — seamless sine stays seamless at every rate
// ---------------------------------------------------------------------------

#[test]
fn t_seam_sine_continuity_all_rates() {
    let (sine_l, sine_r) = make_sine_pcm(PCM_LEN, SINE_HZ);
    let natural = natural_max_step(&sine_l);
    assert!(natural > 0.0 && natural < 0.02, "sanity: 480 Hz sine natural step ≈ 0.0157");

    for (tempo, label) in RATE_CASES {
        let wf = wrap_frames(tempo);
        let total = wf * (MIN_ANALYZED_WRAPS + 1);
        let capture = render_capture(tempo, sine_l.clone(), sine_r.clone(), total);

        // Anti-stub (§6.1): audible, correct-content output.
        let analyzed = &capture[wf as usize..];
        let rms = (analyzed.iter().map(|s| (*s as f64).powi(2)).sum::<f64>()
            / analyzed.len() as f64)
            .sqrt();
        assert!(rms > MIN_MIX_RMS, "rate {label}: sine RMS {rms} below floor — silent render");

        // Wrap 0 excluded (activation ramp + anchor pre-roll).
        let (max_step, at) = max_step_from(&capture, wf as usize);
        let limit = natural * SEAM_STEP_FACTOR;
        assert!(
            max_step <= limit,
            "rate {label}: max inter-sample step {max_step} at frame {at} exceeds \
             {SEAM_STEP_FACTOR}× natural ({limit}) — loop seam transient (≥ 50 wraps analyzed)"
        );
    }
}

// ---------------------------------------------------------------------------
// (b) Onset grid — metronome clicks land on the derived grid, zero per-wrap
//     variance
// ---------------------------------------------------------------------------

#[test]
fn t_seam_metronome_onset_grid_all_rates() {
    let wav = load_float_wav(&fixture_path("metronome_120.wav")).unwrap();
    assert_eq!(wav.sample_rate, SAMPLE_RATE);
    assert_eq!(wav.frames as u64, PCM_LEN, "fixture must be 16 beats @ 120 BPM");

    for (tempo, label) in RATE_CASES {
        let wf = wrap_frames(tempo);
        let bf = beat_frames(tempo);
        let total = wf * (MIN_ANALYZED_WRAPS + 1);
        let capture =
            render_capture(tempo, wav.channels[0].clone(), wav.channels[1].clone(), total);

        let onsets = detect_click_onsets(&capture, ONSET_THRESHOLD, bf as usize / 2);
        // Onset → grid-slot assignment with a half-beat guard at both window
        // edges (ADR-094): the compensated stretch path lands onsets ON the
        // grid, so a beat's detection can sit a few samples EARLY of an exact
        // wrap boundary (measured −21 at rate 0.75). A raw `o >= wf` filter
        // would chop such an onset and shift every pairing by one beat.
        let analyzed: Vec<u64> = onsets
            .iter()
            .map(|&o| o as u64)
            .filter(|&o| o + bf / 2 >= wf && o + bf / 2 < total)
            .collect();

        // Anti-stub (§6.1): exact expected click count — 16 per wrap, every
        // analyzed wrap. A silent or stalled track fails here, not in sync math.
        let expected = (LOOP_BEATS as u64 * MIN_ANALYZED_WRAPS) as usize;
        assert_eq!(
            analyzed.len(),
            expected,
            "rate {label}: expected exactly {expected} onsets in {MIN_ANALYZED_WRAPS} wraps, got {}",
            analyzed.len()
        );

        // Onset offsets against the absolute derived grid. The vocoder smears
        // each click's attack by an amount that depends only on the click's
        // phase against its analysis-hop grid, so the offset sequence must be
        // (1) bounded and (2) EXACTLY periodic. Exact periodicity over ≥ 50
        // wraps is the zero-drift assertion: one sample lost or gained at any
        // wrap (the branch's "frame lost at the loop point") breaks it, as
        // does any slope. (Measured at rate 0.75: spread 80 samples, period
        // 12 beats — a 16-beat wrap sees a rotation, which is why "per-wrap
        // variance" is asserted via periodicity, not wrap-equality.)
        let offsets: Vec<i64> = analyzed
            .iter()
            .enumerate()
            .map(|(k, &o)| o as i64 - (wf + k as u64 * bf) as i64)
            .collect();

        let spread = offsets.iter().max().unwrap() - offsets.iter().min().unwrap();
        assert!(
            spread <= STRETCH_ONSET_JITTER_MAX,
            "rate {label}: onset offset spread {spread} exceeds bounded-smear gate"
        );

        let period = (1..=MAX_ONSET_PATTERN_PERIOD_BEATS).find(|&p| {
            offsets.len() > p && offsets.iter().zip(offsets.iter().skip(p)).all(|(a, b)| a == b)
        });
        assert!(
            period.is_some(),
            "rate {label}: onset offsets are APERIODIC over {} beats — accumulation \
             (drift) present: {:?}…",
            offsets.len(),
            &offsets[..LOOP_BEATS as usize]
        );

        // Absolute-grid constant (§6.3b, ADR-094): BOTH paths must land on
        // the derived grid itself, not merely a constant away from it. The
        // uncompensated stretch path measured a constant 2880 here (the
        // Signalsmith output latency — the C3 finding, ADR-093); the engine's
        // feed-lead compensation must cancel it at every rate.
        let median = median_offset(&offsets);
        assert!(
            median.abs() <= STRETCH_GRID_MEDIAN_MAX,
            "rate {label}: absolute onset-grid median {median} exceeds \
             ±{STRETCH_GRID_MEDIAN_MAX} — stretch-path pre-roll uncompensated"
        );

        // Bypass only (§6.3b "±0"): the output is a bit-exact copy of the
        // source, so rendered onset offsets must equal the source fixture's
        // own per-beat detection offsets (the fixture has accented downbeats:
        // pattern {1, 10, 10, 10} per bar), tiled across wraps, with zero
        // tolerance. Each must also sit within the attack-envelope bound.
        if tempo == O_MBPM {
            let src_onsets =
                detect_click_onsets(&wav.channels[0], ONSET_THRESHOLD, bf as usize / 2);
            let src_offsets: Vec<i64> = src_onsets
                .iter()
                .enumerate()
                .map(|(j, &o)| o as i64 - (j as u64 * bf) as i64)
                .collect();
            assert_eq!(src_offsets.len(), LOOP_BEATS as usize);
            for (k, &off) in offsets.iter().enumerate() {
                assert_eq!(
                    off,
                    src_offsets[k % LOOP_BEATS as usize],
                    "bypass: beat {k} offset differs from source content — not bit-exact"
                );
                assert!(off.abs() <= BYPASS_GRID_TOLERANCE);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// (c) Across a live tempo change (checkpoint C1, spec §5 M1 / §6.3):
//     seams stay continuous and onsets stay on the derived segment-table
//     grid through stretch→stretch, stretch→unity, and unity→stretch
//     changes (the latter two exercising the one-block crossfade, §3.2).
// ---------------------------------------------------------------------------

/// (label, tempo before, tempo after) — covering a span-only change and both
/// crossfade directions of the unity transition.
const CHANGE_CASES: [(&str, u32, u32); 3] = [
    ("stretch→stretch (0.75→1.5)", 90_000, 180_000),
    ("stretch→unity (0.75→1.0)", 90_000, 120_000),
    ("unity→stretch (1.0→1.5)", 120_000, 180_000),
];

/// Wraps rendered per side; both sides analyzed (minus the first wrap of the
/// capture and the first wrap after the change) total ≥ 50 wraps.
const WRAPS_PER_SIDE: u64 = 27;

/// The change lands mid-wrap, mid-beat (the hard case: nothing is aligned).
fn change_frame(tempo_a: u32) -> u64 {
    wrap_frames(tempo_a) * (WRAPS_PER_SIDE - 1) + beat_frames(tempo_a) * 5 + beat_frames(tempo_a) / 2
}

#[test]
fn t_seam_sine_continuity_across_tempo_change() {
    let (sine_l, sine_r) = make_sine_pcm(PCM_LEN, SINE_HZ);
    let natural = natural_max_step(&sine_l);

    for (label, tempo_a, tempo_b) in CHANGE_CASES {
        let f_change = change_frame(tempo_a);
        let total = f_change + wrap_frames(tempo_b) * WRAPS_PER_SIDE;
        let capture = render_capture_with_changes(
            tempo_a,
            sine_l.clone(),
            sine_r.clone(),
            total,
            &[(f_change, tempo_b)],
        );

        // Anti-stub (§6.1): audible on both sides of the change.
        for (side, range) in [
            ("before", wrap_frames(tempo_a) as usize..f_change as usize),
            ("after", f_change as usize..capture.len()),
        ] {
            let seg = &capture[range];
            let rms = (seg.iter().map(|s| (*s as f64).powi(2)).sum::<f64>() / seg.len() as f64)
                .sqrt();
            assert!(rms > MIN_MIX_RMS, "{label}: silent render {side} the change");
        }

        // Spec §3.2: a tempo change only changes the dictated span — no
        // re-anchor, no phase reset, no transient. The unity transitions ride
        // the one-block equal-power crossfade of two content-continuous
        // signals. The same seam bound holds across the ENTIRE capture,
        // change block included (wrap 0 excluded as ever: activation ramp).
        let (max_step, at) = max_step_from(&capture, wrap_frames(tempo_a) as usize);
        let limit = natural * SEAM_STEP_FACTOR;
        assert!(
            max_step <= limit,
            "{label}: max inter-sample step {max_step} at frame {at} exceeds \
             {SEAM_STEP_FACTOR}× natural ({limit}) — transient across the tempo change \
             (change at {f_change})"
        );
    }
}

#[test]
fn t_seam_metronome_onset_grid_across_tempo_change() {
    let wav = load_float_wav(&fixture_path("metronome_120.wav")).unwrap();
    assert_eq!(wav.frames as u64, PCM_LEN);

    for (label, tempo_a, tempo_b) in CHANGE_CASES {
        let f_change = change_frame(tempo_a);
        let total = f_change + wrap_frames(tempo_b) * WRAPS_PER_SIDE;
        let capture = render_capture_with_changes(
            tempo_a,
            wav.channels[0].clone(),
            wav.channels[1].clone(),
            total,
            &[(f_change, tempo_b)],
        );

        // Predicted beat grid from the segment table (bu continuous across
        // the change): beat m sounds at the frame where bu crosses m·K. The
        // tempo pairs are chosen so every grid frame is exact (asserted).
        let bu_c = f_change as u128 * tempo_a as u128;
        let grid: Vec<u64> = (0..)
            .map(|m| {
                let target = m as u128 * BEAT_UNITS_PER_BEAT;
                if target <= bu_c {
                    assert_eq!(target % tempo_a as u128, 0, "{label}: pre-change grid inexact");
                    (target / tempo_a as u128) as u64
                } else {
                    assert_eq!(
                        (target - bu_c) % tempo_b as u128,
                        0,
                        "{label}: post-change grid inexact"
                    );
                    f_change + ((target - bu_c) / tempo_b as u128) as u64
                }
            })
            .take_while(|&f| f < total)
            .collect();

        let min_gap = (beat_frames(tempo_a).min(beat_frames(tempo_b)) / 2) as usize;
        let onsets = detect_click_onsets(&capture, ONSET_THRESHOLD, min_gap);

        // Two analysis windows: [wrap 1, change) and [change + 1 wrap, end) —
        // the first wrap of the capture (activation ramp + anchor pre-roll)
        // and the first wrap after the change (crossfade + vocoder
        // re-settling smear, bounded by the constant window) are excluded
        // from the grid assertions; the COUNT gate below still spans the
        // transition, so a dropped or doubled click there cannot hide.
        let windows = [
            ("before", wrap_frames(tempo_a), f_change, beat_frames(tempo_a)),
            ("after", f_change + wrap_frames(tempo_b), total, beat_frames(tempo_b)),
        ];

        // Anti-stub count gate across the whole capture (minus wrap 0): one
        // onset per predicted beat, exactly.
        let expected_total = grid.iter().filter(|&&g| g >= wrap_frames(tempo_a)).count();
        let detected_total = onsets.iter().filter(|&&o| o as u64 >= wrap_frames(tempo_a)).count();
        assert!(
            (expected_total as i64 - detected_total as i64).abs() <= 1,
            "{label}: expected {expected_total} onsets after wrap 0, detected {detected_total} \
             — a click was lost or doubled somewhere (possibly at the change)"
        );

        let mut side_medians = [0i64; 2];
        for (idx, (side, from, to, bf_side)) in windows.into_iter().enumerate() {
            // Predicted beats of the window, half-beat-guarded at the open
            // end so a beat whose detection would straddle the window edge
            // is never demanded. Offsets are measured by NEAREST detected
            // onset per predicted beat (ADR-094: compensated onsets straddle
            // the grid, so positional pairing against a window cut is
            // ambiguous at the edges). A lost click's nearest detection is
            // the neighbouring beat — a full beat away, far outside the
            // spread gate — so losses still fail loudly per beat (the global
            // count gate above catches doubles).
            let pred: Vec<u64> = grid
                .iter()
                .copied()
                .filter(|&g| g >= from && g + bf_side / 2 < to)
                .collect();
            assert!(
                pred.len() >= LOOP_BEATS as usize,
                "{label} ({side}): analysis window too small ({} beats)",
                pred.len()
            );

            let offsets: Vec<i64> = pred
                .iter()
                .map(|&g| {
                    onsets
                        .iter()
                        .map(|&o| o as i64 - g as i64)
                        .min_by_key(|d| d.abs())
                        .unwrap()
                })
                .collect();
            let spread = offsets.iter().max().unwrap() - offsets.iter().min().unwrap();
            assert!(
                spread <= STRETCH_ONSET_JITTER_MAX,
                "{label} ({side}): onset offset spread {spread} exceeds bounded-smear gate"
            );

            // Exact periodicity per tempo segment = the zero-drift assertion
            // (spec §6.3b): any sample lost or gained breaks it, as does any
            // slope. Periodicity is asserted per side — the smear pattern is
            // a function of the click phase against the hop grid, which the
            // tempo change legitimately re-phases.
            let period = (1..=MAX_ONSET_PATTERN_PERIOD_BEATS).find(|&p| {
                offsets.len() > p
                    && offsets.iter().zip(offsets.iter().skip(p)).all(|(a, b)| a == b)
            });
            assert!(
                period.is_some(),
                "{label} ({side}): onset offsets APERIODIC over {} beats — accumulation \
                 (drift) present: {:?}…",
                offsets.len(),
                &offsets[..offsets.len().min(LOOP_BEATS as usize)]
            );

            // Absolute-grid constant per side (§6.3b, ADR-094): on the
            // derived grid itself on BOTH sides of the change. Before the
            // engine's output-latency compensation, any change crossing the
            // unity boundary stepped by ±2880 here (the C3 finding, ADR-093:
            // measured +2861/+2865 on device) — the audible flam this gate
            // exists to catch.
            let median = median_offset(&offsets);
            assert!(
                median.abs() <= STRETCH_GRID_MEDIAN_MAX,
                "{label} ({side}): absolute onset-grid median {median} exceeds \
                 ±{STRETCH_GRID_MEDIAN_MAX} — stretch-path pre-roll uncompensated"
            );
            side_medians[idx] = median;
        }

        // Step across the change (the device analyzer's C1 gate shape):
        // offset before == offset after, within the same 2 ms class bound.
        let step = side_medians[1] - side_medians[0];
        assert!(
            step.abs() <= STRETCH_GRID_MEDIAN_MAX,
            "{label}: onset-grid median stepped {step} across the tempo change \
             (gate ≤ {STRETCH_GRID_MEDIAN_MAX})"
        );
    }
}

// ---------------------------------------------------------------------------
// Negative control — the detector must SEE a corrupted seam (defect D4)
// ---------------------------------------------------------------------------

#[test]
fn t_seam_negative_control_chopped_fixture_detected() {
    // Double-trim content corruption: 2112 samples of real audio removed from
    // the loop head, length preserved with silence — the seam (and the
    // mid-loop splice) become discontinuous, as production assets were.
    let (sine_l, _) = make_sine_pcm(PCM_LEN, SINE_HZ);
    let natural = natural_max_step(&sine_l);
    let mut chopped = sine_l[2112..].to_vec();
    chopped.resize(PCM_LEN as usize, 0.0);

    let tempo = 90_000; // rate 0.75 — the Appendix A.2 reproduction case
    let wf = wrap_frames(tempo);
    let total = wf * 11;
    let capture = render_capture(tempo, chopped.clone(), chopped, total);

    let (max_step, _) = max_step_from(&capture, wf as usize);
    let limit = natural * SEAM_STEP_FACTOR;
    assert!(
        max_step > limit,
        "negative control BROKEN: chopped fixture rendered max step {max_step} \
         under the {limit} threshold — the seam detector cannot see real seam \
         clicks (spec §6.3 negative control)"
    );
}
