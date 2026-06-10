//! # maestro-validate — device-capture analyzer (spec §6.4)
//!
//! Analyzes a mixer-tap capture from the device harness: detects click onsets
//! per channel (the two metronome stems are hard-panned L/R), measures the
//! cross-stem onset offset series, and applies the C0 gates:
//!
//! - **Onset counts exact** (anti-stub §6.1): each stem must produce exactly
//!   the expected number of clicks for the capture length — silence, a dead
//!   stem, or wrong-rate playback (the stretch-disabled negative control)
//!   fails here first.
//! - **Drift = 0**: the least-squares slope of offset-vs-onset-index,
//!   extrapolated over the whole capture, must round to 0 samples. Any
//!   accumulating step fails.
//! - **Bounded offset**: max−min of the offset series stays within the
//!   vocoder's bounded-smear window (unity captures: exactly 0 — bypass is
//!   bit-exact, both stems identical content ⇒ identical detection).
//! - **Budget gates** (from the harness snapshot log): underruns = 0,
//!   blocks_over_half_period / blocks_total ≤ 1% (p99 < 50% of period),
//!   render_ns_max < 90% of the callback period.
//!
//! Exit code 0 = all gates pass; 1 = any gate fails; the JSON report is
//! written either way (committed under documentation/validation-runs/).
//!
//! Negative-control duty (§6.4): fed a capture from a sabotaged build
//! (pcm_length_loop / stretch_disabled), this analyzer MUST report failure.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use maestro_core::validation::{detect_click_onsets, linear_slope, load_float_wav};
use serde::Serialize;

/// Mirror of tests/common/validation_thresholds.rs values used on-device.
/// VALIDATION-THRESHOLD markers apply here too (CI greps both files).
mod thresholds {
    /// VALIDATION-THRESHOLD: onset detection level (fixture peak ≈ 0.285).
    pub const ONSET_THRESHOLD: f32 = 0.10;
    /// VALIDATION-THRESHOLD: total extrapolated drift over the capture must
    /// round to zero samples (C0: "any nonzero integer step fails").
    pub const MAX_TOTAL_DRIFT_SAMPLES: f64 = 0.5;
    /// VALIDATION-THRESHOLD: cross-stem offset spread bound on the stretch
    /// path (both stems smear inside the constant vocoder window; the
    /// per-stem pattern is periodic — see T-SEAM). Unity captures: exactly 0.
    pub const STRETCH_OFFSET_SPREAD_MAX: i64 = 960;
    /// VALIDATION-THRESHOLD: p99 budget gate — fraction of blocks whose
    /// render exceeded half the callback period (p99 < 50% ⇔ ≤ 1%).
    pub const MAX_OVER_HALF_PERIOD_FRACTION: f64 = 0.01;
    /// VALIDATION-THRESHOLD: render_ns_max must stay under 90% of period.
    pub const MAX_RENDER_PERIOD_FRACTION: f64 = 0.90;
    /// VALIDATION-THRESHOLD: leading onsets skipped per channel (activation
    /// ramp + anchor pre-roll make the first beats unreliable for detection).
    pub const SKIP_LEADING_ONSETS: usize = 2;
    /// VALIDATION-THRESHOLD: minimum onsets that must remain after skip/align
    /// for the analysis to be meaningful (≥ 60 s at 120 BPM ⇒ ~120 beats).
    pub const MIN_ANALYZED_ONSETS: usize = 50;
}

#[derive(Serialize)]
struct GateResult {
    name: String,
    passed: bool,
    detail: String,
}

#[derive(Serialize)]
struct Report {
    capture: String,
    scenario: String,
    sample_rate: u32,
    capture_frames: usize,
    onsets_left: usize,
    onsets_right: usize,
    analyzed_pairs: usize,
    offset_first: i64,
    offset_min: i64,
    offset_max: i64,
    drift_slope_samples_per_onset: f64,
    total_drift_samples: f64,
    gates: Vec<GateResult>,
    passed: bool,
}

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let usage = "usage: maestro-validate --capture <wav> --scenario <unity|cross> \
                 --tempo-mbpm <120000> [--harness-report <json>] [--report <out.json>] [--expect-fail]";

    let Some(capture_path) = arg(&args, "--capture") else {
        eprintln!("{usage}");
        return ExitCode::from(2);
    };
    let scenario = arg(&args, "--scenario").unwrap_or_else(|| "cross".into());
    let tempo_mbpm: u64 = arg(&args, "--tempo-mbpm").unwrap_or_else(|| "120000".into()).parse().expect("--tempo-mbpm must be an integer (milli-BPM)");
    let harness_report = arg(&args, "--harness-report");
    let report_out = arg(&args, "--report");
    // Negative-control mode (§6.4): the analysis must FAIL for the run to pass.
    let expect_fail = args.iter().any(|a| a == "--expect-fail");

    let mut gates: Vec<GateResult> = Vec::new();
    let mut gate = |name: &str, passed: bool, detail: String| {
        println!("  [{}] {name}: {detail}", if passed { "PASS" } else { "FAIL" });
        gates.push(GateResult { name: name.into(), passed, detail });
    };

    // ---------------------------------------------------------------- capture
    let wav = match load_float_wav(Path::new(&capture_path)) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("cannot read capture: {e}");
            return ExitCode::from(2);
        }
    };
    println!(
        "capture: {} — {} Hz, {} ch, {} frames ({:.1} s)",
        capture_path,
        wav.sample_rate,
        wav.num_channels,
        wav.frames,
        wav.frames as f64 / wav.sample_rate as f64
    );
    assert!(wav.num_channels >= 2, "capture must be stereo (stems hard-panned L/R)");

    // Both metronome stems click once per engine beat (they are metronomes at
    // their own tempo, stretched to the engine tempo): beat period in capture
    // frames = 60_000 / tempo_mbpm seconds.
    let beat_seconds = 60_000.0 / tempo_mbpm as f64;
    let beat_frames = beat_seconds * wav.sample_rate as f64;
    let min_gap = (beat_frames / 2.0) as usize;

    let onsets_l = detect_click_onsets(&wav.channels[0], thresholds::ONSET_THRESHOLD, min_gap);
    let onsets_r = detect_click_onsets(&wav.channels[1], thresholds::ONSET_THRESHOLD, min_gap);

    // ------------------------------------------------------------ count gate
    // The capture begins when the harness starts the transport (capture tap
    // installed before Play), so expected clicks ≈ duration / beat. Allow ±2
    // for capture-edge truncation; wrong-rate playback (stretch disabled at
    // rate 2.0 ⇒ half the clicks) is far outside this window.
    let expected = (wav.frames as f64 / beat_frames).floor() as i64;
    let count_ok = |n: usize| (n as i64 - expected).abs() <= 2;
    gate(
        "onset-count",
        count_ok(onsets_l.len()) && count_ok(onsets_r.len()),
        format!("expected ≈{expected} clicks/stem, got L={} R={}", onsets_l.len(), onsets_r.len()),
    );

    // ------------------------------------------------------------ pair align
    let skip = thresholds::SKIP_LEADING_ONSETS;
    let mut offsets: Vec<i64> = Vec::new();
    let half_beat = (beat_frames / 2.0) as i64;
    let mut ri = 0usize;
    for &l in onsets_l.iter().skip(skip) {
        // advance right index to the onset nearest this left onset
        while ri + 1 < onsets_r.len()
            && (onsets_r[ri + 1] as i64 - l as i64).abs() < (onsets_r[ri] as i64 - l as i64).abs()
        {
            ri += 1;
        }
        if ri < onsets_r.len() {
            let d = l as i64 - onsets_r[ri] as i64;
            if d.abs() < half_beat {
                offsets.push(d);
            }
        }
    }
    gate(
        "analyzed-pairs",
        offsets.len() >= thresholds::MIN_ANALYZED_ONSETS,
        format!("{} cross-stem onset pairs (min {})", offsets.len(), thresholds::MIN_ANALYZED_ONSETS),
    );

    // ------------------------------------------------------------ drift gate
    let (slope, total_drift, omin, omax) = if offsets.len() >= 2 {
        let slope = linear_slope(&offsets);
        let total = slope * offsets.len() as f64;
        (slope, total, *offsets.iter().min().unwrap(), *offsets.iter().max().unwrap())
    } else {
        (0.0, f64::INFINITY, 0, 0)
    };
    gate(
        "zero-drift",
        total_drift.abs() < thresholds::MAX_TOTAL_DRIFT_SAMPLES,
        format!(
            "offset slope {slope:.6} samples/onset ⇒ {total_drift:.3} samples over the capture \
             (gate: |drift| < {})",
            thresholds::MAX_TOTAL_DRIFT_SAMPLES
        ),
    );

    // ----------------------------------------------------------- spread gate
    let spread = omax - omin;
    let spread_limit = match scenario.as_str() {
        // Bypass is bit-exact and both stems are identical content: the
        // cross-stem offset must be exactly constant.
        "unity" => 0,
        _ => thresholds::STRETCH_OFFSET_SPREAD_MAX,
    };
    gate(
        "offset-bounded",
        spread <= spread_limit,
        format!("offset range [{omin}, {omax}] spread {spread} (limit {spread_limit}, scenario {scenario})"),
    );

    // ----------------------------------------------------------- budget gates
    if let Some(hr_path) = &harness_report {
        match std::fs::read_to_string(hr_path).map_err(|e| e.to_string()).and_then(|s| {
            serde_json::from_str::<serde_json::Value>(&s).map_err(|e| e.to_string())
        }) {
            Ok(hr) => {
                let snap = hr.get("finalSnapshot").cloned().unwrap_or_default();
                let u64of = |k: &str| snap.get(k).and_then(|v| v.as_u64()).unwrap_or(u64::MAX);
                let underruns = u64of("underruns");
                let blocks = u64of("blocksTotal");
                let over_half = u64of("blocksOverHalfPeriod");
                let max_ns = u64of("renderNsMax");
                let period_ns = u64of("callbackPeriodNs");
                gate("underruns", underruns == 0, format!("{underruns} underruns (gate: 0)"));
                let frac = over_half as f64 / blocks.max(1) as f64;
                gate(
                    "budget-p99",
                    blocks > 0 && frac <= thresholds::MAX_OVER_HALF_PERIOD_FRACTION,
                    format!("{over_half}/{blocks} blocks over half period ({:.3}%, gate ≤ 1%)", frac * 100.0),
                );
                gate(
                    "budget-max",
                    period_ns > 0
                        && (max_ns as f64) < thresholds::MAX_RENDER_PERIOD_FRACTION * period_ns as f64,
                    format!("render max {max_ns} ns vs period {period_ns} ns (gate < 90%)"),
                );
            }
            Err(e) => gate("harness-report", false, format!("unreadable: {e}")),
        }
    }

    // ---------------------------------------------------------------- report
    let passed = gates.iter().all(|g| g.passed);
    let report = Report {
        capture: capture_path.clone(),
        scenario: scenario.clone(),
        sample_rate: wav.sample_rate,
        capture_frames: wav.frames,
        onsets_left: onsets_l.len(),
        onsets_right: onsets_r.len(),
        analyzed_pairs: offsets.len(),
        offset_first: offsets.first().copied().unwrap_or(0),
        offset_min: omin,
        offset_max: omax,
        drift_slope_samples_per_onset: slope,
        total_drift_samples: total_drift,
        gates,
        passed,
    };
    if let Some(out) = report_out {
        std::fs::create_dir_all(PathBuf::from(&out).parent().unwrap_or(Path::new("."))).ok();
        std::fs::write(&out, serde_json::to_string_pretty(&report).unwrap()).expect("write report");
        println!("report → {out}");
    }

    if expect_fail {
        if passed {
            println!("NEGATIVE CONTROL BROKEN: sabotaged capture PASSED analysis (spec §6.4)");
            ExitCode::from(1)
        } else {
            println!("negative control OK: sabotaged capture correctly failed analysis");
            ExitCode::SUCCESS
        }
    } else if passed {
        println!("ALL GATES PASS");
        ExitCode::SUCCESS
    } else {
        println!("GATES FAILED");
        ExitCode::from(1)
    }
}
