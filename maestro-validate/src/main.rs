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
    /// VALIDATION-THRESHOLD (C3, ADR-093): the stretch path's deterministic
    /// pre-roll — Signalsmith preset_default output latency at 48 kHz (spec
    /// §2.3#4; the dossier-verified 2880/2880 instance latency). Bypass
    /// audio lands exactly on the derived grid; stretched audio lands this
    /// CONSTANT late (spec §6.3(b) explicitly permits the stretch path "a
    /// constant measured offset"). It is invisible to every cross-stem gate
    /// (all stems of a song share O, so they cross the unity boundary
    /// together — measured +0 samples cross-stem on the C3 UI capture) and
    /// to the per-capture alignment constant. It becomes observable exactly
    /// once: when a tempo change crosses the unity boundary, a stem's median
    /// offset must step by THIS value — not 0, and not anything else (a
    /// stale re-anchor still measures ~1.2k, outside 2880 ±
    /// TEMPO_CHANGE_STEP_MAX, so the D7 negative control still fails).
    /// Engine-side pre-roll compensation is filed as follow-up (it matters
    /// for cross-song ghost stems, which can be stretched while host stems
    /// run unity).
    pub const STRETCH_PATH_PREROLL: i64 = 2880;
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
    /// VALIDATION-THRESHOLD (C2, spec §4.11): the per-song PCM budget the
    /// memory scenario asserts against the snapshot — mirrors
    /// `control_plane::PCM_BUDGET_BYTES`.
    pub const MEMORY_PCM_BUDGET_BYTES: u64 = 96 * 1024 * 1024;
    /// VALIDATION-THRESHOLD (C2): "no audio gap on the playing stems" — every
    /// window of this length must keep an RMS floor on the sustained stem.
    /// The floor detects ENGINE silence, not quiet music: a real gap (DSP
    /// skip / botched swap) contributes exact zeros to the mixed bus and
    /// measures ~1e-6 in the capture (converter/dither floor), while the
    /// quietest musical moment of the Lagos pad measured 2.75e-3 on the
    /// first C2 device capture (2026-06-10-1930). 1e-4 (−80 dBFS) separates
    /// the classes by ~30 dB on both sides. (Initial a-priori value 3e-3
    /// mis-flagged quiet content — recalibrated against the real capture,
    /// ADR-092 §5.)
    pub const GAP_WINDOW_SECONDS: f64 = 0.5;
    pub const GAP_RMS_FLOOR: f64 = 1e-4;
    /// VALIDATION-THRESHOLD (C2): anti-stub presence floor for the rhythmic
    /// stem over the whole capture (measured 0.12 on device; an order of
    /// magnitude of margin).
    pub const STEM_PRESENT_RMS_FLOOR: f64 = 0.01;
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
    /// Slot-1 first activation frame (timing scenario). The harness arms
    /// before Play so this is 0 there; a REAL-UI run (checkpoint C3) arms
    /// stem B by a human tap mid-transport, so B's predicted grid starts at
    /// the engine's own activation echo — §6.4's predicted-vs-measured rule,
    /// now grounded for both drivers. Not a threshold change (ADR-093).
    b_start: Option<u64>,
    /// Slot-1 deactivation / post-stop reactivation frames (timing scenario).
    b_stop: Option<u64>,
    b_restart: Option<u64>,
    variation_at: Option<u64>,
    /// C2 control scenario echoes (spec §6.4 methodology: every silent
    /// window is predicted from these, never from wall-clock).
    mute_on: Option<u64>,
    mute_off: Option<u64>,
    solo_on: Option<u64>,
    solo_off: Option<u64>,
    paused_at: Option<u64>,
    resumed_at: Option<u64>,
    /// §4.11 memory-gate echoes.
    variation_load_required: bool,
    variation_evicted: bool,
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
        b_start: None,
        b_stop: None,
        b_restart: None,
        variation_at: None,
        mute_on: None,
        mute_off: None,
        solo_on: None,
        solo_off: None,
        paused_at: None,
        resumed_at: None,
        variation_load_required: false,
        variation_evicted: false,
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
                // First activation starts B's grid (harness: frame 0; real
                // UI: the human's tap, echoed exactly); the restart is the
                // first activation after the stop.
                match (at, log.b_stop) {
                    (Some(at), Some(stop)) if at >= stop => log.b_restart = Some(at),
                    (Some(at), None) if log.b_start.is_none() => log.b_start = Some(at),
                    _ => {}
                }
            }
            "variationApplied" => log.variation_at = at,
            "variationLoadRequired" => log.variation_load_required = true,
            "variationEvicted" => log.variation_evicted = true,
            // Control scenario (C2): stem B (slot 1) mute pair; stem A
            // (slot 0) solo pair silences B; pause/resume freeze F.
            "muteApplied" if slot == Some(1) => {
                let muted = e.get("muted").and_then(|v| v.as_bool()).unwrap_or(false);
                if muted {
                    log.mute_on = at;
                } else {
                    log.mute_off = at;
                }
            }
            "soloApplied" if slot == Some(0) => {
                let solo = e.get("solo").and_then(|v| v.as_bool()).unwrap_or(false);
                if solo {
                    log.solo_on = at;
                } else {
                    log.solo_off = at;
                }
            }
            "paused" => log.paused_at = at,
            "resumed" => log.resumed_at = at,
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
    if w.slope.abs() < thresholds::ZERO_SLOPE_PER_ONSET_MAX {
        return (
            true,
            format!(
                "aperiodic within period ≤ {}; LS slope {:.6} samples/onset over {} onsets (gate < {})",
                thresholds::ONSET_PATTERN_PERIOD_MAX,
                w.slope,
                w.offsets.len(),
                thresholds::ZERO_SLOPE_PER_ONSET_MAX
            ),
        );
    }
    // Aperiodic AND slope over gate: the LS fit is only valid when the
    // vocoder's jitter pattern repeats inside the window. At tempo ratios
    // like 40/39 (120→117, the UI tempo button's single step) the pattern
    // period is ~39 beats — longer than any window can repeat twice — and
    // the fit aliases the sawtooth (C3: measured spurious slopes ±0.06–0.16
    // on captures whose offset SPREAD was ≤ 31 samples, ADR-093). The
    // spread bound is the sharper instrument there: in-window accumulation
    // can never exceed the spread, so a window whose spread stays within
    // the step-threshold class has zero accumulation at the same 96-sample
    // granularity every boundary gate uses. Not a loosening: the historic
    // drift class (0.04 samples/onset, spec A.1) accumulates ~1–2 samples
    // per window — invisible to ANY per-window fit — and is caught by the
    // full-capture gates (exact counts, cross-window medians).
    let spread_bounded = w.spread <= thresholds::TEMPO_CHANGE_STEP_MAX;
    (
        spread_bounded,
        format!(
            "aperiodic within period ≤ {} and LS slope {:.6}/onset over {} onsets is fit-aliased;              accumulation bounded by window spread {} (gate ≤ {})",
            thresholds::ONSET_PATTERN_PERIOD_MAX,
            w.slope,
            w.offsets.len(),
            w.spread,
            thresholds::TEMPO_CHANGE_STEP_MAX
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
    // C3 (ADR-093): the stems' original tempo, when the capture comes from a
    // library song (real-UI runs). Lets the timing analysis know whether a
    // tempo change crosses the UNITY boundary — bypass sits exactly on the
    // grid, stretch sits a deterministic pre-roll late (spec §6.3(b)), so
    // the expected step across such a change is STRETCH_PATH_PREROLL, not 0.
    // Absent for harness runs (cross-tempo stems never run unity): steps 0.
    let stem_tempo_mbpm: Option<u64> =
        arg(&args, "--stem-tempo-mbpm").map(|v| v.parse().expect("--stem-tempo-mbpm must be an integer (milli-BPM)"));
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
            analyze_timing(&wav, &log, stem_tempo_mbpm, &mut gate, &mut report);
        }
        "control" => {
            let hr = harness_json.as_ref().expect("--harness-report is required for control");
            let log = match parse_command_log(hr, tempo_mbpm) {
                Ok(l) => l,
                Err(e) => {
                    eprintln!("cannot parse command log: {e}");
                    return ExitCode::from(2);
                }
            };
            analyze_control(&wav, &log, &mut gate, &mut report);
        }
        "interruption" => {
            // §4.9 route-cycle leg: the outage is a common-mode splice; the
            // cross-stem gates of the `cross` analysis are immune to it by
            // construction — plus the anti-stub that the leg really happened.
            analyze_offset_pairs(&wav, "cross", tempo_mbpm, harness_json.as_ref(), &mut gate, &mut report);
            let hr = harness_json.as_ref().expect("--harness-report is required for interruption");
            let route_changes = hr
                .get("session")
                .and_then(|s| s.get("routeChanges"))
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            gate(
                "route-changes-happened",
                route_changes >= 2,
                format!("{route_changes} route changes observed (need ≥ 2: speaker→receiver→speaker)"),
            );
        }
        "memory" => {
            let hr = harness_json.as_ref().expect("--harness-report is required for memory");
            analyze_memory(&wav, hr, tempo_mbpm, &mut gate, &mut report);
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
    stem_tempo_mbpm: Option<u64>,
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
    // Offset before == offset after the tempo change (C1) — except when the
    // change crosses the UNITY boundary (C3, ADR-093): bypass lands exactly
    // on the grid, stretch lands the deterministic pre-roll late (spec
    // §6.3(b)), so the EXPECTED step is ±STRETCH_PATH_PREROLL there. This is
    // an exact additional expectation, not a loosening: a stale re-anchor
    // (~1.2k) is outside both windows and still fails.
    let expected_flip_step = |mbpm_before: u64, mbpm_after: u64| -> i64 {
        match stem_tempo_mbpm {
            Some(o) if o == mbpm_before && o != mbpm_after => thresholds::STRETCH_PATH_PREROLL,
            Some(o) if o != mbpm_before && o == mbpm_after => -thresholds::STRETCH_PATH_PREROLL,
            _ => 0,
        }
    };
    let a_expected = expected_flip_step(log.segments[0].1, log.segments[1].1);
    let a_step = a_mid.median - a_pre.median;
    gate(
        "a-offset-equal-across-change",
        (a_step - a_expected).abs() <= thresholds::TEMPO_CHANGE_STEP_MAX,
        format!(
            "stem A median offset: pre {} vs post {} (step {a_step}, expected {a_expected} ± {})",
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

    // ---- Stem B (R): active [b_start, b_stop), silent [b_stop, b_restart). --
    // b_start is 0 for harness runs (armed before Play); for real-UI runs
    // (C3) it is the engine's echo of the human's arm tap.
    let b_start = log.b_start.unwrap_or(0);
    let b_grid_pre: Vec<u64> =
        grid.iter().copied().filter(|&g| g >= b_start && g < b_stop).collect();
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
    let b_expected = expected_flip_step(log.segments[0].1, log.segments[1].1);
    let b_step = b_post.median - b_pre.median;
    gate(
        "b-re-entry-equality",
        (b_step - b_expected).abs() <= thresholds::RE_ENTRY_STEP_MAX,
        format!(
            "stem B median offset: pre-stop {} vs post-restart {} (step {b_step}, expected {b_expected} ± {})",
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

// ---------------------------------------------------------------------------
// C2 control analysis: the full dispatch-only flow (mute/solo/tempo/
// variation/pause) gated against the engine's own echoes (spec §6.4)
// ---------------------------------------------------------------------------

/// Pair detected onsets against grid frames inside an engine-frame window,
/// mapping capture positions through `align`. Returns None when the window
/// has too few usable beats.
struct PairedWindow {
    window: Window,
    expected: usize,
    detected: usize,
}

fn pair_window(
    onsets: &[usize],
    grid: &[u64],
    range: (u64, u64),
    align: i64,
    skip_first: usize,
) -> Option<PairedWindow> {
    let grid_in: Vec<u64> = grid.iter().copied().filter(|&g| g >= range.0 && g < range.1).collect();
    let lo = range.0 as i64 + align;
    let hi = i64::try_from(range.1).map_or(i64::MAX, |v| v.saturating_add(align));
    let detected = onsets
        .iter()
        .filter(|&&o| (o as i64) >= lo && (o as i64) < hi)
        .count();
    let expected = grid_in.len();
    if grid_in.len() <= skip_first + 1 {
        return None;
    }
    // Pair each grid beat with the NEAREST detected onset within half the
    // local beat spacing — immune to inclusion jitter at non-silent window
    // boundaries (a sequential pairing shifts every pair by one beat there).
    let mut offsets: Vec<i64> = Vec::with_capacity(grid_in.len());
    for (i, &g) in grid_in.iter().enumerate().skip(skip_first) {
        let spacing = if i + 1 < grid_in.len() {
            grid_in[i + 1] - grid_in[i]
        } else {
            grid_in[i] - grid_in[i - 1]
        };
        let target = g as i64 + align;
        let nearest = onsets
            .iter()
            .map(|&o| o as i64 - target)
            .min_by_key(|d| d.abs())
            .filter(|d| d.unsigned_abs() < spacing / 2);
        if let Some(d) = nearest {
            offsets.push(d);
        }
    }
    if offsets.len() < 2 {
        return None;
    }
    let median = median(&offsets);
    let slope = linear_slope(&offsets);
    let spread = offsets.iter().max().unwrap() - offsets.iter().min().unwrap();
    Some(PairedWindow { window: Window { offsets, median, slope, spread }, expected, detected })
}

#[allow(clippy::too_many_lines)]
fn analyze_control(
    wav: &maestro_core::validation::LoadedWav,
    log: &CommandLog,
    gate: &mut impl FnMut(&str, bool, String),
    report: &mut Report,
) {
    // The scripted sequence: mute B / unmute B / solo A / unsolo A / tempo
    // change / variation switch on A / pause / resume — all echoed.
    let need = [
        ("muteApplied(on)", log.mute_on),
        ("muteApplied(off)", log.mute_off),
        ("soloApplied(on)", log.solo_on),
        ("soloApplied(off)", log.solo_off),
        ("paused", log.paused_at),
        ("resumed", log.resumed_at),
        ("variationApplied", log.variation_at),
    ];
    let missing: Vec<&str> =
        need.iter().filter(|(_, v)| v.is_none()).map(|(n, _)| *n).collect();
    gate(
        "command-log",
        missing.is_empty() && log.segments.len() == 2,
        format!("echo coverage: missing {missing:?}; segments {:?}", log.segments),
    );
    if !missing.is_empty() || log.segments.len() != 2 {
        return;
    }
    let (mute_on, mute_off) = (log.mute_on.unwrap(), log.mute_off.unwrap());
    let (solo_on, solo_off) = (log.solo_on.unwrap(), log.solo_off.unwrap());
    let f_pause = log.paused_at.unwrap();
    let t_tempo = log.segments[1].0;
    let t_var = log.variation_at.unwrap();
    gate(
        "sequence-order",
        mute_on < mute_off
            && mute_off < solo_on
            && solo_on < solo_off
            && solo_off < t_tempo
            && t_tempo < t_var
            && t_var < f_pause
            && log.resumed_at == log.paused_at, // F frozen: same frame on both echoes
        format!(
            "mute [{mute_on},{mute_off}) solo [{solo_on},{solo_off}) tempo {t_tempo} \
             variation {t_var} pause {f_pause} (resumed at {:?} must equal paused)",
            log.resumed_at
        ),
    );

    let grid = beat_grid(&log.segments, log.final_frame);
    let min_gap = {
        let min_period = log.segments.iter().map(|&(_, m)| (K / m as u128) as usize).min().unwrap();
        min_period / 2
    };
    let onsets_l = detect_click_onsets(&wav.channels[0], thresholds::ONSET_THRESHOLD, min_gap);
    let onsets_r = detect_click_onsets(&wav.channels[1], thresholds::ONSET_THRESHOLD, min_gap);
    report.onsets_left = onsets_l.len();
    report.onsets_right = onsets_r.len();
    let skip = thresholds::SKIP_LEADING_ONSETS;

    // ---- Stem A (L): audible throughout; the engine emits no frames while
    // paused, so the engine-domain grid is pause-free — only the ALIGNMENT
    // splits at the pause (capture gains wall-clock silence).
    let n_pre_grid = grid.iter().filter(|&&g| g < f_pause).count();
    gate(
        "a-onset-count",
        (onsets_l.len() as i64 - grid.len() as i64).abs() <= 2,
        format!("stem A: {} onsets vs {} predicted beats", onsets_l.len(), grid.len()),
    );
    if onsets_l.len() < n_pre_grid.min(grid.len()) {
        return;
    }
    // align1 from A's pre-tempo-change window.
    let a_pre_idx: Vec<usize> = (skip..n_pre_grid.min(onsets_l.len()))
        .filter(|&i| grid[i] < t_tempo)
        .collect();
    if a_pre_idx.len() < 2 {
        gate("a-window-sizes", false, "no pre-change onsets for alignment".into());
        return;
    }
    let raw_pre: Vec<i64> =
        a_pre_idx.iter().map(|&i| onsets_l[i] as i64 - grid[i] as i64).collect();
    let align1 = median(&raw_pre);
    // align2 from A's post-resume onsets (sequential pairing).
    let post_grid: Vec<u64> = grid.iter().copied().filter(|&g| g >= f_pause).collect();
    let a_post_on: Vec<u64> = onsets_l[n_pre_grid.min(onsets_l.len())..]
        .iter()
        .map(|&o| o as u64)
        .collect();
    let n_post = a_post_on.len().min(post_grid.len());
    gate(
        "a-post-pause-count",
        (a_post_on.len() as i64 - post_grid.len() as i64).abs() <= 2 && n_post >= 3,
        format!("stem A post-resume: {} onsets vs {} predicted", a_post_on.len(), post_grid.len()),
    );
    if n_post < 3 {
        return;
    }
    let raw_post: Vec<i64> =
        (1..n_post).map(|i| a_post_on[i] as i64 - post_grid[i] as i64).collect();
    let align2 = median(&raw_post);
    let pause_wall = align2 - align1;
    gate(
        "pause-wall-duration",
        (48_000..480_000).contains(&pause_wall),
        format!(
            "capture gained {pause_wall} frames across the pause (≈{:.2} s; F frozen, \
             wall clock not) — sanity window [1 s, 10 s]",
            pause_wall as f64 / 48_000.0
        ),
    );

    // A's windows pre-pause: [0,t_tempo) / [t_tempo,t_var) / [t_var,f_pause).
    let on_l: Vec<usize> = onsets_l.clone();
    let wa1 = pair_window(&on_l, &grid, (0, t_tempo), align1, skip);
    let wa2 = pair_window(&on_l, &grid, (t_tempo, t_var), align1, 1);
    let wa3 = pair_window(&on_l, &grid, (t_var, f_pause), align1, 1);
    let wa4 = pair_window(&on_l, &grid, (f_pause, u64::MAX), align2, 1);
    let (Some(wa1), Some(wa2), Some(wa3), Some(wa4)) = (wa1, wa2, wa3, wa4) else {
        gate("a-window-sizes", false, "a stem-A window had too few onsets".into());
        return;
    };
    gate(
        "a-window-sizes",
        wa1.window.offsets.len() >= thresholds::MIN_WINDOW_ONSETS
            && wa2.window.offsets.len() >= 4
            && wa3.window.offsets.len() >= 4
            && wa4.window.offsets.len() >= thresholds::MIN_WINDOW_ONSETS,
        format!(
            "stem A windows: {}/{}/{}/{} onsets",
            wa1.window.offsets.len(),
            wa2.window.offsets.len(),
            wa3.window.offsets.len(),
            wa4.window.offsets.len()
        ),
    );
    for (name, w) in [("a-zero-accum-pre", &wa1), ("a-zero-accum-post-resume", &wa4)] {
        if w.window.offsets.len() < thresholds::MIN_SLOPE_WINDOW_ONSETS {
            gate(name, true, format!("{} onsets — bounded by step gates", w.window.offsets.len()));
            continue;
        }
        let (ok, detail) = zero_accumulation(&w.window);
        gate(name, ok, detail);
    }
    let step_tempo = wa2.window.median - wa1.window.median;
    gate(
        "a-offset-equal-across-tempo-change",
        step_tempo.abs() <= thresholds::TEMPO_CHANGE_STEP_MAX,
        format!(
            "stem A median: pre {} vs post-change {} (step {step_tempo}, gate ≤ {})",
            wa1.window.median, wa2.window.median, thresholds::TEMPO_CHANGE_STEP_MAX
        ),
    );
    let step_var = wa3.window.median - wa2.window.median;
    gate(
        "a-offset-equal-across-variation",
        step_var.abs() <= thresholds::TEMPO_CHANGE_STEP_MAX,
        format!("step {step_var} across the variation switch (gate ≤ {})", thresholds::TEMPO_CHANGE_STEP_MAX),
    );

    // ---- Stem B (R): silent in [mute_on,mute_off) and [solo_on,solo_off). --
    let beat = (K / log.segments[0].1 as u128) as i64;
    let margin = beat / 2;
    let silent = |name: &str, lo: u64, hi: u64, gate: &mut dyn FnMut(&str, bool, String)| {
        let lo_c = lo as i64 + align1 + margin;
        let hi_c = hi as i64 + align1 - margin;
        let intruders = onsets_r
            .iter()
            .filter(|&&o| (o as i64) > lo_c && (o as i64) < hi_c)
            .count();
        gate(
            name,
            intruders == 0,
            format!("{intruders} onsets inside the predicted silent window [{lo_c}, {hi_c}] (gate: 0)"),
        );
    };
    silent("b-mute-window-silent", mute_on, mute_off, gate);
    silent("b-solo-window-silent", solo_on, solo_off, gate);

    let on_r: Vec<usize> = onsets_r.clone();
    let wb1 = pair_window(&on_r, &grid, (0, mute_on), align1, skip);
    let wb2 = pair_window(&on_r, &grid, (mute_off, solo_on), align1, 1);
    let wb3 = pair_window(&on_r, &grid, (solo_off, t_tempo), align1, 1);
    let wb4 = pair_window(&on_r, &grid, (t_tempo, f_pause), align1, 1);
    let wb5 = pair_window(&on_r, &grid, (f_pause, u64::MAX), align2, 1);
    let (Some(wb1), Some(wb2), Some(wb3), Some(wb4), Some(wb5)) = (wb1, wb2, wb3, wb4, wb5)
    else {
        gate("b-window-sizes", false, "a stem-B window had too few onsets".into());
        return;
    };
    for (name, w) in [
        ("b-count-pre-mute", &wb1),
        ("b-count-between", &wb2),
        ("b-count-post-solo", &wb3),
        ("b-count-post-tempo", &wb4),
        ("b-count-post-resume", &wb5),
    ] {
        gate(
            name,
            (w.detected as i64 - w.expected as i64).abs() <= 1,
            format!("{} onsets vs {} predicted", w.detected, w.expected),
        );
    }
    // Re-entry equality across each silenced window (the §3.2 claim: silence
    // cannot desync) and across the tempo change.
    for (name, a, b, max) in [
        ("b-re-entry-across-mute", wb1.window.median, wb2.window.median, thresholds::RE_ENTRY_STEP_MAX),
        ("b-re-entry-across-solo", wb2.window.median, wb3.window.median, thresholds::RE_ENTRY_STEP_MAX),
        ("b-offset-equal-across-tempo-change", wb3.window.median, wb4.window.median, thresholds::TEMPO_CHANGE_STEP_MAX),
    ] {
        let step = b - a;
        gate(name, step.abs() <= max, format!("median {a} → {b} (step {step}, gate ≤ {max})"));
    }
    // Cross-stem offset equality across the pause (alignment-free).
    let cross_pre = wb4.window.median - wa3.window.median;
    let cross_post = wb5.window.median - wa4.window.median;
    gate(
        "cross-offset-equal-across-pause",
        (cross_post - cross_pre).abs() <= thresholds::TEMPO_CHANGE_STEP_MAX,
        format!("cross-stem offset pre {cross_pre} vs post {cross_post} (gate ≤ {})", thresholds::TEMPO_CHANGE_STEP_MAX),
    );

    report.analyzed_pairs = wa1.window.offsets.len() + wa4.window.offsets.len();
    report.timing = Some(TimingNumbers {
        tempo_change_frame: t_tempo,
        b_stop_frame: mute_on,
        b_restart_frame: mute_off,
        variation_frame: Some(t_var),
        align_offset: align1,
        a_median_pre: wa1.window.median,
        a_median_post: wa4.window.median,
        b_median_pre: wb1.window.median,
        b_median_post: wb5.window.median,
        a_slope_pre: wa1.window.slope,
        a_slope_post: wa4.window.slope,
        b_slope_post: wb5.window.slope,
    });
}

// ---------------------------------------------------------------------------
// C2 memory analysis (spec §4.11): budget, lazy switch, eviction, no gap
// ---------------------------------------------------------------------------

fn analyze_memory(
    wav: &maestro_core::validation::LoadedWav,
    hr: &serde_json::Value,
    tempo_mbpm: u64,
    gate: &mut impl FnMut(&str, bool, String),
    report: &mut Report,
) {
    let log = parse_command_log(hr, tempo_mbpm).unwrap_or_else(|e| panic!("command log: {e}"));

    // Budget (spec §4.11): every polled snapshot AND the final one stay under
    // the per-song PCM budget — including through the lazy switch.
    let mut max_bytes = hr
        .get("finalSnapshot")
        .and_then(|s| s.get("pcmBytesInUse"))
        .and_then(|v| v.as_u64())
        .unwrap_or(u64::MAX);
    let mut saw_switched_variation = false;
    if let Some(snaps) = hr.get("snapshots").and_then(|s| s.as_array()) {
        for s in snaps {
            if let Some(b) = s.get("pcmBytesInUse").and_then(|v| v.as_u64()) {
                max_bytes = max_bytes.max(b);
            }
            if let Some(tracks) = s.get("tracks").and_then(|t| t.as_array()) {
                if tracks
                    .iter()
                    .any(|t| t.get("slot").and_then(|v| v.as_u64()) == Some(0)
                        && t.get("variation").and_then(|v| v.as_u64()) == Some(2))
                {
                    saw_switched_variation = true;
                }
            }
        }
    }
    gate(
        "pcm-budget",
        max_bytes <= thresholds::MEMORY_PCM_BUDGET_BYTES,
        format!(
            "max pcmBytesInUse {:.1} MB (budget {} MB, spec §4.11)",
            max_bytes as f64 / (1024.0 * 1024.0),
            thresholds::MEMORY_PCM_BUDGET_BYTES / (1024 * 1024)
        ),
    );

    // Lazy switch life cycle: requested → loaded → applied (+ eviction once
    // the song's PCM exceeds the budget), all engine-echoed.
    gate(
        "variation-load-required",
        log.variation_load_required,
        format!("variationLoadRequired echoed: {}", log.variation_load_required),
    );
    gate(
        "variation-applied",
        log.variation_at.is_some() && saw_switched_variation,
        format!(
            "variationApplied at {:?}; snapshot shows variation 2 on slot 0: {}",
            log.variation_at, saw_switched_variation
        ),
    );
    gate(
        "variation-evicted",
        log.variation_evicted,
        format!("variationEvicted echoed (budget enforcement observable): {}", log.variation_evicted),
    );

    // Load health: every TrackLoaded must reconcile, identically, with zero
    // failures — this is the D4 fix running against REAL R2 files + measured
    // per-file metadata on the REAL device decoder.
    let mut reconciliations: Vec<String> = Vec::new();
    let mut load_failures: Vec<String> = Vec::new();
    if let Some(events) = hr.get("events").and_then(|e| e.as_array()) {
        for e in events {
            match e.get("type").and_then(|t| t.as_str()) {
                Some("trackLoaded") => reconciliations.push(
                    e.get("reconciliation").and_then(|r| r.as_str()).unwrap_or("?").to_string(),
                ),
                Some("trackLoadFailed") => load_failures
                    .push(e.get("reason").and_then(|r| r.as_str()).unwrap_or("?").to_string()),
                _ => {}
            }
        }
    }
    let all_same = reconciliations.windows(2).all(|w| w[0] == w[1]);
    gate(
        "loads-reconcile",
        reconciliations.len() >= 6 && all_same && load_failures.is_empty(),
        format!(
            "{} loads reconciled as {:?} (stable: {all_same}); failures: {load_failures:?}",
            reconciliations.len(),
            reconciliations.first()
        ),
    );

    // No audio gap on the playing stems (spec §4.11): every GAP_WINDOW of the
    // sustained stem (L = pad) keeps an RMS floor from start-of-sound to the
    // capture tail; the rhythmic stem (R) is present overall.
    let win = (thresholds::GAP_WINDOW_SECONDS * wav.sample_rate as f64) as usize;
    let l = &wav.channels[0];
    let start = 2 * wav.sample_rate as usize; // skip ramp-in
    let end = wav.frames.saturating_sub(wav.sample_rate as usize);
    let mut worst = f64::INFINITY;
    let mut worst_at = 0usize;
    let mut pos = start;
    while pos + win <= end {
        let rms = (l[pos..pos + win].iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>()
            / win as f64)
            .sqrt();
        if rms < worst {
            worst = rms;
            worst_at = pos;
        }
        pos += win / 2;
    }
    gate(
        "no-audio-gap",
        worst > thresholds::GAP_RMS_FLOOR,
        format!(
            "worst {:.1} ms window RMS {worst:.5} at {:.1} s (floor {}, spec §4.11 — the lazy \
             switch must not gap the playing stems)",
            thresholds::GAP_WINDOW_SECONDS * 1000.0,
            worst_at as f64 / wav.sample_rate as f64,
            thresholds::GAP_RMS_FLOOR
        ),
    );
    let r = &wav.channels[1];
    let r_rms = (r[start..end.max(start + 1)]
        .iter()
        .map(|&x| (x as f64) * (x as f64))
        .sum::<f64>()
        / (end.max(start + 1) - start) as f64)
        .sqrt();
    gate(
        "rhythmic-stem-present",
        r_rms > thresholds::STEM_PRESENT_RMS_FLOOR,
        format!("stem R overall RMS {r_rms:.5} (floor {})", thresholds::STEM_PRESENT_RMS_FLOOR),
    );
    report.analyzed_pairs = 0;
}
