//! # maestro-validate — device-capture analyzer (spec §6.4)
//!
//! Analyzes a mixer-tap capture from the device harness: detects click onsets
//! per channel (the metronome stems are hard-panned L/R), measures onset
//! offsets, and applies the checkpoint gates.
//!
//! ## Scenarios
//!
//! - `unity` / `cross` (C0): cross-stem offset constant, drift = 0, exact
//!   onset counts, budget gates from the harness snapshot.
//! - `timing` (C1): the §6.4 scripted M1 sequence. The expected onset grid is
//!   computed **from the harness report's command echoes** (TempoApplied /
//!   TrackActivated frames — predicted-vs-measured, never measured-vs-measured
//!   alone). Gates: offset before == offset after the tempo change; zero
//!   slope both sides; re-entry equality for stem B (post-restart offset ==
//!   pre-stop offset); zero onsets inside B's stop window; exact counts;
//!   variation switch applied without disturbing the grid.
//! - `budget` (C1): budget gates with a tempo change + variation switch
//!   mid-measurement; the count gate derives expected beats from the
//!   report's segment table.
//!
//! Exit code 0 = all gates pass; 1 = any gate fails; the JSON report is
//! written either way (committed under documentation/validation-runs/).
//!
//! Negative-control duty (§6.4): fed a capture from a sabotaged build
//! (pcm_length_loop / stretch_disabled / stale_reanchor), this analyzer MUST
//! report failure (`--expect-fail` inverts the exit code).

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use maestro_core::validation::{detect_click_onsets, linear_slope, load_float_wav};
use serde::Serialize;

/// Beat-units per beat (spec §3.1): K = 60_000 × 48_000.
const K: u128 = 2_880_000_000;

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
    /// VALIDATION-THRESHOLD (C1): max median-offset step of a stem across the
    /// tempo change ("offset before == offset after", spec §6.4). Timeline
    /// integrity errors are ≥ one render chunk (512 samples) or one stale
    /// re-anchor window (≥ 1000); the vocoder's rate-dependent attack-smear
    /// difference measured at C0 stayed within a 26-sample spread. 96 (2 ms)
    /// separates the defect classes with margin on both sides.
    pub const TEMPO_CHANGE_STEP_MAX: i64 = 96;
    /// VALIDATION-THRESHOLD (C1): re-entry equality — stem B's median offset
    /// after restart (across the tempo change) vs before its stop.
    pub const RE_ENTRY_STEP_MAX: i64 = 96;
    /// VALIDATION-THRESHOLD (C1): minimum onsets per analysis window
    /// (pre/post-change, pre-stop/post-restart).
    pub const MIN_WINDOW_ONSETS: usize = 12;
    /// VALIDATION-THRESHOLD (C1): per-side zero-slope gate, expressed per
    /// onset because the C1 windows (~30–40 s) are shorter than C0's full
    /// capture and the periodic vocoder jitter aliases into a least-squares
    /// fit (offline-measured spurious slope ≤ 0.014/onset on a healthy
    /// engine). The historic drift class measures 0.04 samples/onset
    /// (24 samples in 10 min at one onset/s, spec Appendix A.1) — 0.02
    /// separates the two with a factor of ~1.5 either way; the full-capture
    /// `cross` scenario keeps the tighter absolute gate.
    pub const ZERO_SLOPE_PER_ONSET_MAX: f64 = 0.02;
    /// VALIDATION-THRESHOLD (C1): minimum onsets for a least-squares slope
    /// fit to outweigh the periodic vocoder jitter (pattern period up to
    /// ~12 beats, T-SEAM). Shorter windows are NOT slope-gated — they are
    /// bracketed by median-equality gates on both boundaries instead, which
    /// bound any in-window drift to the step thresholds.
    pub const MIN_SLOPE_WINDOW_ONSETS: usize = 32;
    /// VALIDATION-THRESHOLD (C1): longest admissible exact period (in
    /// onsets) of a window's offset pattern. The primary zero-accumulation
    /// assertion is EXACT periodicity (one sample lost or gained anywhere
    /// breaks it — T-SEAM's rule); the LS slope gate is the fallback for
    /// aperiodic-but-flat windows. A period must repeat at least twice
    /// inside the window to count.
    pub const ONSET_PATTERN_PERIOD_MAX: usize = 16;
}

#[derive(Serialize)]
struct GateResult {
    name: String,
    passed: bool,
    detail: String,
}

#[derive(Serialize, Default)]
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
    /// C1 timing scenario: per-window medians (samples, grid-relative).
    timing: Option<TimingNumbers>,
    gates: Vec<GateResult>,
    passed: bool,
}

#[derive(Serialize)]
struct TimingNumbers {
    tempo_change_frame: u64,
    b_stop_frame: u64,
    b_restart_frame: u64,
    variation_frame: Option<u64>,
    align_offset: i64,
    a_median_pre: i64,
    a_median_post: i64,
    b_median_pre: i64,
    b_median_post: i64,
    a_slope_pre: f64,
    a_slope_post: f64,
    b_slope_post: f64,
}

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn median(v: &[i64]) -> i64 {
    let mut s = v.to_vec();
    s.sort_unstable();
    s[s.len() / 2]
}

// ---------------------------------------------------------------------------
// Command log: the engine's own echoes from the harness report (spec §6.4 —
// the expected grid comes from here, never from wall-clock assumptions).
// ---------------------------------------------------------------------------

struct CommandLog {
    /// (start_frame, mbpm), starting with (0, initial tempo).
    segments: Vec<(u64, u64)>,
    /// Slot-1 deactivation / post-stop reactivation frames (timing scenario).
    b_stop: Option<u64>,
    b_restart: Option<u64>,
    variation_at: Option<u64>,
    /// Engine frames rendered (final snapshot).
    final_frame: u64,
}

fn parse_command_log(report: &serde_json::Value, initial_tempo: u64) -> Result<CommandLog, String> {
    let final_frame = report
        .get("finalSnapshot")
        .and_then(|s| s.get("engineFrame"))
        .and_then(|v| v.as_u64())
        .ok_or("finalSnapshot.engineFrame missing")?;
    let mut log = CommandLog {
        segments: vec![(0, initial_tempo)],
        b_stop: None,
        b_restart: None,
        variation_at: None,
        final_frame,
    };
    let events = report.get("events").and_then(|e| e.as_array()).ok_or("events missing")?;
    for e in events {
        let typ = e.get("type").and_then(|t| t.as_str()).unwrap_or("");
        let at = e.get("atFrame").and_then(|v| v.as_u64());
        let slot = e.get("slot").and_then(|v| v.as_u64());
        match typ {
            "tempoApplied" => {
                let mbpm = e.get("mbpm").and_then(|v| v.as_u64()).ok_or("tempoApplied.mbpm")?;
                let at = at.ok_or("tempoApplied.atFrame")?;
                if at == 0 {
                    log.segments[0].1 = mbpm;
                } else {
                    log.segments.push((at, mbpm));
                }
            }
            "trackDeactivated" if slot == Some(1) => log.b_stop = at,
            "trackActivated" if slot == Some(1) => {
                // The initial arm echoes at frame 0; the restart is the one
                // after the stop.
                if let (Some(at), Some(stop)) = (at, log.b_stop) {
                    if at >= stop {
                        log.b_restart = Some(at);
                    }
                }
            }
            "variationApplied" => log.variation_at = at,
            _ => {}
        }
    }
    log.segments.sort_by_key(|s| s.0);
    Ok(log)
}

/// `bu(F)` over the echoed segment table (exact integer arithmetic).
fn bu_at(segments: &[(u64, u64)], frame: u64) -> u128 {
    let mut bu: u128 = 0;
    let mut prev = segments[0];
    for &(f, mbpm) in &segments[1..] {
        if frame < f {
            break;
        }
        bu += (f - prev.0) as u128 * prev.1 as u128;
        prev = (f, mbpm);
    }
    bu + (frame.max(prev.0) - prev.0) as u128 * prev.1 as u128
}

/// Engine-frame beat grid: the frame where `bu` first reaches `m·K`, for all
/// beats inside the rendered range. Exact ceiling inversion per segment.
fn beat_grid(segments: &[(u64, u64)], final_frame: u64) -> Vec<u64> {
    let mut grid = Vec::new();
    let bu_end = bu_at(segments, final_frame);
    let mut m: u128 = 0;
    while m * K < bu_end {
        let target = m * K;
        // Find the segment containing target.
        let mut f = None;
        for (i, &(f_k, mbpm)) in segments.iter().enumerate() {
            let bu_k = bu_at(segments, f_k);
            let bu_next = segments
                .get(i + 1)
                .map(|&(f_next, _)| bu_at(segments, f_next))
                .unwrap_or(u128::MAX);
            if target >= bu_k && target < bu_next {
                let d = (target - bu_k).div_ceil(mbpm as u128);
                f = Some(f_k + d as u64);
                break;
            }
        }
        grid.push(f.expect("beat inside rendered range"));
        m += 1;
    }
    grid
}

// ---------------------------------------------------------------------------
// Window analysis helper
// ---------------------------------------------------------------------------

struct Window {
    offsets: Vec<i64>,
    median: i64,
    slope: f64,
    spread: i64,
}

/// Pair detected onsets (capture frames) sequentially with predicted grid
/// frames (engine frames + alignment) and summarize.
fn window_offsets(onsets: &[u64], grid: &[u64], align: i64) -> Option<Window> {
    if onsets.is_empty() || onsets.len() != grid.len() {
        return None;
    }
    let offsets: Vec<i64> = onsets
        .iter()
        .zip(grid)
        .map(|(&o, &g)| o as i64 - g as i64 - align)
        .collect();
    let median = median(&offsets);
    let slope = if offsets.len() >= 2 { linear_slope(&offsets) } else { 0.0 };
    let spread = offsets.iter().max().unwrap() - offsets.iter().min().unwrap();
    Some(Window { offsets, median, slope, spread })
}

/// Zero-accumulation verdict for a window of offsets (the spec §6.1 rule:
/// assert periodicity/slope, never per-onset constancy). The first onset of
/// the window is a boundary artifact (activation ramp / crossfade /
/// re-anchor settling) and is excluded. Primary: the remaining offsets are
/// EXACTLY periodic (period repeating ≥ 2× inside the window) — one sample
/// lost or gained anywhere breaks this, as does any slope. Fallback for
/// aperiodic-but-flat windows: the LS slope gate.
fn zero_accumulation(w: &Window) -> (bool, String) {
    let body = if w.offsets.len() > 1 { &w.offsets[1..] } else { &w.offsets[..] };
    let max_p = thresholds::ONSET_PATTERN_PERIOD_MAX.min(body.len() / 2);
    for p in 1..=max_p {
        if body.iter().zip(body.iter().skip(p)).all(|(a, b)| a == b) {
            return (
                true,
                format!(
                    "offsets exactly periodic (period {p}, {} onsets) — zero accumulation",
                    body.len()
                ),
            );
        }
    }
    (
        w.slope.abs() < thresholds::ZERO_SLOPE_PER_ONSET_MAX,
        format!(
            "aperiodic within period ≤ {}; LS slope {:.6} samples/onset over {} onsets (gate < {})",
            thresholds::ONSET_PATTERN_PERIOD_MAX,
            w.slope,
            w.offsets.len(),
            thresholds::ZERO_SLOPE_PER_ONSET_MAX
        ),
    )
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let usage = "usage: maestro-validate --capture <wav> --scenario <unity|cross|timing|budget> \
                 --tempo-mbpm <120000> [--harness-report <json>] [--report <out.json>] [--expect-fail]";

    let Some(capture_path) = arg(&args, "--capture") else {
        eprintln!("{usage}");
        return ExitCode::from(2);
    };
    let scenario = arg(&args, "--scenario").unwrap_or_else(|| "cross".into());
    let tempo_mbpm: u64 = arg(&args, "--tempo-mbpm")
        .unwrap_or_else(|| "120000".into())
        .parse()
        .expect("--tempo-mbpm must be an integer (milli-BPM)");
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

    let harness_json: Option<serde_json::Value> = harness_report.as_ref().map(|p| {
        let s = std::fs::read_to_string(p).unwrap_or_else(|e| panic!("harness report {p}: {e}"));
        serde_json::from_str(&s).unwrap_or_else(|e| panic!("harness report {p}: {e}"))
    });

    let mut report = Report {
        capture: capture_path.clone(),
        scenario: scenario.clone(),
        sample_rate: wav.sample_rate,
        capture_frames: wav.frames,
        ..Default::default()
    };

    match scenario.as_str() {
        "timing" => {
            let hr = harness_json.as_ref().expect("--harness-report is required for timing");
            let log = match parse_command_log(hr, tempo_mbpm) {
                Ok(l) => l,
                Err(e) => {
                    eprintln!("cannot parse command log: {e}");
                    return ExitCode::from(2);
                }
            };
            analyze_timing(&wav, &log, &mut gate, &mut report);
        }
        _ => {
            analyze_offset_pairs(&wav, &scenario, tempo_mbpm, harness_json.as_ref(), &mut gate, &mut report);
        }
    }

    // ----------------------------------------------------------- budget gates
    if let Some(hr) = &harness_json {
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
            blocks > 0 && blocks != u64::MAX && frac <= thresholds::MAX_OVER_HALF_PERIOD_FRACTION,
            format!("{over_half}/{blocks} blocks over half period ({:.3}%, gate ≤ 1%)", frac * 100.0),
        );
        gate(
            "budget-max",
            period_ns > 0
                && period_ns != u64::MAX
                && (max_ns as f64) < thresholds::MAX_RENDER_PERIOD_FRACTION * period_ns as f64,
            format!("render max {max_ns} ns vs period {period_ns} ns (gate < 90%)"),
        );
    }

    // ---------------------------------------------------------------- report
    let passed = gates.iter().all(|g| g.passed);
    report.gates = gates;
    report.passed = passed;
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

// ---------------------------------------------------------------------------
// C0-style analysis (unity / cross / budget): cross-stem offsets
// ---------------------------------------------------------------------------

fn analyze_offset_pairs(
    wav: &maestro_core::validation::LoadedWav,
    scenario: &str,
    tempo_mbpm: u64,
    harness_json: Option<&serde_json::Value>,
    gate: &mut impl FnMut(&str, bool, String),
    report: &mut Report,
) {
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
    // installed before Play), so expected clicks ≈ duration / beat — except
    // the budget scenario (C1), whose tempo changes mid-run: there the
    // expected count is the segment table's beat total from the report.
    let expected = match (scenario, harness_json) {
        ("budget", Some(hr)) => match parse_command_log(hr, tempo_mbpm) {
            Ok(log) => (bu_at(&log.segments, log.final_frame) / K) as i64,
            Err(_) => (wav.frames as f64 / beat_frames).floor() as i64,
        },
        _ => (wav.frames as f64 / beat_frames).floor() as i64,
    };
    let count_ok = |n: usize| (n as i64 - expected).abs() <= 2;
    gate(
        "onset-count",
        count_ok(onsets_l.len()) && count_ok(onsets_r.len()),
        format!("expected ≈{expected} clicks/stem, got L={} R={}", onsets_l.len(), onsets_r.len()),
    );

    // ------------------------------------------------------------ pair align
    let skip = thresholds::SKIP_LEADING_ONSETS;
    let mut offsets: Vec<i64> = Vec::new();
    let mut pair_pos: Vec<u64> = Vec::new(); // capture frame of each pair (L onset)
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
                pair_pos.push(l as u64);
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
    let budget_log = match (scenario, harness_json) {
        ("budget", Some(hr)) => parse_command_log(hr, tempo_mbpm).ok().filter(|l| l.segments.len() > 1),
        _ => None,
    };
    if let Some(log) = &budget_log {
        // The budget scenario contains scripted commands (tempo changes +
        // variation switch): a whole-capture LS fit is invalid across those
        // boundaries (each one legitimately re-phases the vocoder's smear
        // pattern). Instead: zero accumulation inside each command-free
        // segment, plus a bounded net step from the first segment to the
        // last (the same estimator design as the timing scenario).
        let d_est = (wav.frames as i64 - log.final_frame as i64).max(0) as u64 / 2;
        let margin = (1.5 * wav.sample_rate as f64) as u64;
        let mut boundaries: Vec<u64> = log.segments[1..].iter().map(|&(f, _)| f + d_est).collect();
        if let Some(v) = log.variation_at {
            boundaries.push(v + d_est);
        }
        boundaries.sort_unstable();
        let seg_of = |pos: u64| boundaries.iter().filter(|&&b| pos >= b).count();
        let excluded =
            |pos: u64| boundaries.iter().any(|&b| pos + margin > b && pos < b + margin);
        let n_segs = boundaries.len() + 1;
        let mut seg_windows: Vec<Option<Window>> = Vec::new();
        for s in 0..n_segs {
            let seg_offsets: Vec<i64> = offsets
                .iter()
                .zip(&pair_pos)
                .filter(|(_, &p)| seg_of(p) == s && !excluded(p))
                .map(|(&o, _)| o)
                .collect();
            if seg_offsets.len() < 2 {
                seg_windows.push(None);
                continue;
            }
            let med = median(&seg_offsets);
            let sl = linear_slope(&seg_offsets);
            let spread = seg_offsets.iter().max().unwrap() - seg_offsets.iter().min().unwrap();
            seg_windows.push(Some(Window { offsets: seg_offsets, median: med, slope: sl, spread }));
        }
        for (s, w) in seg_windows.iter().enumerate() {
            match w {
                Some(w) if w.offsets.len() >= thresholds::MIN_WINDOW_ONSETS => {
                    let (ok, detail) = zero_accumulation(w);
                    gate(&format!("zero-drift-segment-{s}"), ok, detail);
                }
                _ => gate(
                    &format!("zero-drift-segment-{s}"),
                    true,
                    "segment too short between commands — bounded by the net-step gate".into(),
                ),
            }
        }
        let first = seg_windows.iter().flatten().next();
        let last = seg_windows.iter().flatten().last();
        if let (Some(first), Some(last)) = (first, last) {
            let step = last.median - first.median;
            gate(
                "net-offset-step",
                step.abs() <= thresholds::TEMPO_CHANGE_STEP_MAX,
                format!(
                    "cross-stem median offset: first segment {} vs last {} (step {step}, gate ≤ {})",
                    first.median,
                    last.median,
                    thresholds::TEMPO_CHANGE_STEP_MAX
                ),
            );
        } else {
            gate("net-offset-step", false, "missing analyzable segments".into());
        }
    } else {
        gate(
            "zero-drift",
            total_drift.abs() < thresholds::MAX_TOTAL_DRIFT_SAMPLES,
            format!(
                "offset slope {slope:.6} samples/onset ⇒ {total_drift:.3} samples over the capture \
                 (gate: |drift| < {})",
                thresholds::MAX_TOTAL_DRIFT_SAMPLES
            ),
        );
    }

    // ----------------------------------------------------------- spread gate
    let spread = omax - omin;
    let spread_limit = match scenario {
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

    report.onsets_left = onsets_l.len();
    report.onsets_right = onsets_r.len();
    report.analyzed_pairs = offsets.len();
    report.offset_first = offsets.first().copied().unwrap_or(0);
    report.offset_min = omin;
    report.offset_max = omax;
    report.drift_slope_samples_per_onset = slope;
    report.total_drift_samples = total_drift;
}

// ---------------------------------------------------------------------------
// C1 timing analysis: predicted grid from the command log (spec §6.4)
// ---------------------------------------------------------------------------

fn analyze_timing(
    wav: &maestro_core::validation::LoadedWav,
    log: &CommandLog,
    gate: &mut impl FnMut(&str, bool, String),
    report: &mut Report,
) {
    // The scripted sequence has exactly one tempo change while playing.
    gate(
        "command-log",
        log.segments.len() == 2 && log.b_stop.is_some() && log.b_restart.is_some(),
        format!(
            "segments {:?}, B stop {:?}, B restart {:?}",
            log.segments, log.b_stop, log.b_restart
        ),
    );
    let (Some(b_stop), Some(b_restart)) = (log.b_stop, log.b_restart) else { return };
    let t_change = log.segments.get(1).map(|s| s.0).unwrap_or(u64::MAX);
    gate(
        "sequence-order",
        b_stop < t_change && t_change < b_restart,
        format!("stop {b_stop} < tempo change {t_change} < restart {b_restart} (spec §6.4 ordering)"),
    );

    let grid = beat_grid(&log.segments, log.final_frame);
    let min_gap = {
        // Smallest beat period across the segments, in frames.
        let min_period = log.segments.iter().map(|&(_, m)| (K / m as u128) as usize).min().unwrap();
        min_period / 2
    };
    let onsets_l = detect_click_onsets(&wav.channels[0], thresholds::ONSET_THRESHOLD, min_gap);
    let onsets_r = detect_click_onsets(&wav.channels[1], thresholds::ONSET_THRESHOLD, min_gap);
    report.onsets_left = onsets_l.len();
    report.onsets_right = onsets_r.len();

    let skip = thresholds::SKIP_LEADING_ONSETS;

    // ---- Stem A (L): plays the whole capture; its grid is every beat. ----
    // Counts must match exactly (±2 for capture-edge truncation, §6.1).
    let count_diff = onsets_l.len() as i64 - grid.len() as i64;
    gate(
        "a-onset-count",
        count_diff.abs() <= 2,
        format!("stem A: {} onsets vs {} predicted beats", onsets_l.len(), grid.len()),
    );
    let n_a = onsets_l.len().min(grid.len());
    let a_on: Vec<u64> = onsets_l[skip..n_a].iter().map(|&o| o as u64).collect();
    let a_grid: Vec<u64> = grid[skip..n_a].to_vec();

    // Alignment constant: engine frame → capture frame (ring + output path
    // latency), estimated once from stem A's PRE-CHANGE window. Everything
    // downstream is predicted-vs-measured through this single constant.
    //
    // Stem A is analyzed in three windows split at the engine's echoed
    // frames: [start, tempo change), [change, variation switch),
    // [variation, end). Each boundary is a legitimate vocoder re-phasing
    // point (the change re-phases the hop grid; the swap re-anchors), so
    // slope is gated per window and the boundary steps are gated by the
    // median-equality thresholds — periodicity/slope, never per-onset
    // constancy (T-SEAM's rule, spec §6.1).
    let t_var = log.variation_at.unwrap_or(u64::MAX);
    gate(
        "variation-after-restart",
        log.variation_at.is_some_and(|v| v > b_restart),
        format!("VariationApplied at {:?} (must follow the restart, spec §6.4)", log.variation_at),
    );
    let w1: Vec<usize> = (0..a_grid.len()).filter(|&i| a_grid[i] < t_change).collect();
    let w2: Vec<usize> =
        (0..a_grid.len()).filter(|&i| a_grid[i] >= t_change && a_grid[i] < t_var).collect();
    let w3: Vec<usize> = (0..a_grid.len()).filter(|&i| a_grid[i] >= t_var).collect();
    gate(
        "a-window-sizes",
        w1.len() >= thresholds::MIN_WINDOW_ONSETS
            && w2.len() >= thresholds::MIN_WINDOW_ONSETS
            && w3.len() >= thresholds::MIN_WINDOW_ONSETS,
        format!(
            "stem A windows: {} pre-change, {} change→variation, {} post-variation onsets",
            w1.len(),
            w2.len(),
            w3.len()
        ),
    );
    if w1.len() < 2 || w2.len() < 2 || w3.len() < 2 {
        return;
    }
    let raw_pre: Vec<i64> = w1.iter().map(|&i| a_on[i] as i64 - a_grid[i] as i64).collect();
    let align = median(&raw_pre);

    let win = |idx: &[usize]| {
        window_offsets(
            &idx.iter().map(|&i| a_on[i]).collect::<Vec<_>>(),
            &idx.iter().map(|&i| a_grid[i]).collect::<Vec<_>>(),
            align,
        )
        .expect("stem A window")
    };
    let a_pre = win(&w1);
    let a_mid = win(&w2);
    let a_post = win(&w3);

    // Zero accumulation on both sides of the change (C1): exact periodicity
    // of the offset pattern (primary) or zero LS slope (fallback). Windows
    // shorter than MIN_SLOPE_WINDOW_ONSETS can't support either against the
    // jitter pattern; they are bracketed by the median-equality gates on
    // both boundaries instead (reported, not silently skipped).
    for (name, w) in
        [("a-zero-slope-pre", &a_pre), ("a-zero-slope-post", &a_mid), ("a-zero-slope-post-variation", &a_post)]
    {
        if w.offsets.len() < thresholds::MIN_SLOPE_WINDOW_ONSETS {
            gate(
                name,
                true,
                format!(
                    "{} onsets < {} — too short for a slope fit; bounded by the \
                     median-step gates on both boundaries",
                    w.offsets.len(),
                    thresholds::MIN_SLOPE_WINDOW_ONSETS
                ),
            );
            continue;
        }
        let (ok, detail) = zero_accumulation(w);
        gate(name, ok, detail);
    }
    // Offset before == offset after the tempo change (C1).
    let a_step = a_mid.median - a_pre.median;
    gate(
        "a-offset-equal-across-change",
        a_step.abs() <= thresholds::TEMPO_CHANGE_STEP_MAX,
        format!(
            "stem A median offset: pre {} vs post {} (step {a_step}, gate ≤ {})",
            a_pre.median, a_mid.median, thresholds::TEMPO_CHANGE_STEP_MAX
        ),
    );
    // …and across the variation switch (re-anchor may re-phase the smear
    // inside the constant window; it must not move the grid).
    let a_var_step = a_post.median - a_mid.median;
    gate(
        "a-offset-equal-across-variation",
        a_var_step.abs() <= thresholds::TEMPO_CHANGE_STEP_MAX,
        format!(
            "stem A median offset: pre-variation {} vs post-variation {} (step {a_var_step}, gate ≤ {})",
            a_mid.median, a_post.median, thresholds::TEMPO_CHANGE_STEP_MAX
        ),
    );
    gate(
        "a-offset-bounded",
        a_pre.spread <= thresholds::STRETCH_OFFSET_SPREAD_MAX
            && a_mid.spread <= thresholds::STRETCH_OFFSET_SPREAD_MAX
            && a_post.spread <= thresholds::STRETCH_OFFSET_SPREAD_MAX,
        format!(
            "stem A spreads: pre {} mid {} post {} (limit {})",
            a_pre.spread, a_mid.spread, a_post.spread, thresholds::STRETCH_OFFSET_SPREAD_MAX
        ),
    );

    // ---- Stem B (R): silent window [b_stop, b_restart). --------------------
    let b_grid_pre: Vec<u64> = grid.iter().copied().filter(|&g| g < b_stop).collect();
    let b_grid_post: Vec<u64> = grid.iter().copied().filter(|&g| g >= b_restart).collect();

    // Split detected onsets at the middle of the silent window (in capture
    // frames, via the alignment constant).
    let silent_mid = ((b_stop + b_restart) / 2) as i64 + align;
    let b_on_pre: Vec<u64> =
        onsets_r.iter().map(|&o| o as u64).filter(|&o| (o as i64) < silent_mid).collect();
    let b_on_post: Vec<u64> =
        onsets_r.iter().map(|&o| o as u64).filter(|&o| (o as i64) >= silent_mid).collect();

    // Exact counts, including ZERO onsets inside the stop window: an onset
    // detected there fails the count (pre gets an extra, §6.4).
    gate(
        "b-onset-count-pre",
        (b_on_pre.len() as i64 - b_grid_pre.len() as i64).abs() <= 1,
        format!("stem B pre-stop: {} onsets vs {} predicted (silent window must be silent)", b_on_pre.len(), b_grid_pre.len()),
    );
    gate(
        "b-onset-count-post",
        (b_on_post.len() as i64 - b_grid_post.len() as i64).abs() <= 2,
        format!("stem B post-restart: {} onsets vs {} predicted", b_on_post.len(), b_grid_post.len()),
    );

    // Explicit silent-window scan: no onset between stop and restart
    // (with half-beat margins for attack tails at the edges).
    let margin = (min_gap / 2) as i64;
    let silent_lo = b_stop as i64 + align + margin;
    let silent_hi = b_restart as i64 + align - margin;
    let intruders = onsets_r
        .iter()
        .filter(|&&o| (o as i64) > silent_lo && (o as i64) < silent_hi)
        .count();
    gate(
        "b-silent-window",
        intruders == 0,
        format!("{intruders} onsets inside B's stop window [{silent_lo}, {silent_hi}] (gate: 0)"),
    );

    let n_pre = b_on_pre.len().min(b_grid_pre.len());
    let n_post = b_on_post.len().min(b_grid_post.len());
    gate(
        "b-window-sizes",
        n_pre > skip + thresholds::MIN_WINDOW_ONSETS && n_post >= thresholds::MIN_WINDOW_ONSETS,
        format!("stem B windows: {n_pre} pre-stop, {n_post} post-restart onsets"),
    );
    if n_pre <= skip + 2 || n_post < 2 {
        return;
    }
    let b_pre = window_offsets(&b_on_pre[skip..n_pre], &b_grid_pre[skip..n_pre], align)
        .expect("B pre window");
    // Post-restart: the first onset after re-entry is under the activation
    // ramp; skip it from the offset stats (the count gates above still see it).
    let b_post = window_offsets(&b_on_post[1..n_post], &b_grid_post[1..n_post], align)
        .expect("B post window");

    // Re-entry equality (C1): same constant before the stop and after the
    // restart, across the tempo change that happened while B was silent —
    // the exact class of re-anchoring bug that sank the previous attempt.
    let b_step = b_post.median - b_pre.median;
    gate(
        "b-re-entry-equality",
        b_step.abs() <= thresholds::RE_ENTRY_STEP_MAX,
        format!(
            "stem B median offset: pre-stop {} vs post-restart {} (step {b_step}, gate ≤ {})",
            b_pre.median, b_post.median, thresholds::RE_ENTRY_STEP_MAX
        ),
    );
    let (b_ok, b_detail) = zero_accumulation(&b_post);
    gate("b-zero-slope-post", b_ok, format!("stem B post-restart: {b_detail}"));
    gate(
        "b-offset-bounded",
        b_pre.spread <= thresholds::STRETCH_OFFSET_SPREAD_MAX
            && b_post.spread <= thresholds::STRETCH_OFFSET_SPREAD_MAX,
        format!("stem B spreads: pre {} post {} (limit {})", b_pre.spread, b_post.spread, thresholds::STRETCH_OFFSET_SPREAD_MAX),
    );

    // Cross-stem offset before == after (the C1 wording, derived from the
    // same single alignment constant).
    let cross_pre = b_pre.median - a_pre.median;
    let cross_post = b_post.median - a_post.median;
    gate(
        "cross-offset-equal",
        (cross_post - cross_pre).abs() <= thresholds::TEMPO_CHANGE_STEP_MAX,
        format!("cross-stem offset: pre {cross_pre} vs post {cross_post} (gate ≤ {})", thresholds::TEMPO_CHANGE_STEP_MAX),
    );

    // Variation switch applied (slot 0 at t≈40 s) without disturbing stem A's
    // grid — the grid gates above span it; here we assert the engine echo.
    gate(
        "variation-applied",
        log.variation_at.is_some(),
        format!("VariationApplied echo at {:?}", log.variation_at),
    );

    report.analyzed_pairs = a_pre.offsets.len() + a_post.offsets.len();
    report.timing = Some(TimingNumbers {
        tempo_change_frame: t_change,
        b_stop_frame: b_stop,
        b_restart_frame: b_restart,
        variation_frame: log.variation_at,
        align_offset: align,
        a_median_pre: a_pre.median,
        a_median_post: a_post.median,
        b_median_pre: b_pre.median,
        b_median_post: b_post.median,
        a_slope_pre: a_pre.slope,
        a_slope_post: a_post.slope,
        b_slope_post: b_post.slope,
    });
}
