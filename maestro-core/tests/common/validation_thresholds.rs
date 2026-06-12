//! Single home for every validation threshold (spec §6.1).
//!
//! **Loosening any value here requires an ADR reference in the same commit.**
//! CI greps for the `VALIDATION-THRESHOLD` marker on changed lines to keep
//! threshold edits visible in review — the abandoned branch's "peak assertion
//! loosened to pass" failure class (D13) is structurally disallowed.

/// VALIDATION-THRESHOLD: max non-onset inter-sample step at loop seams, as a
/// multiple of the fixture's natural maximum step (spec §6.3a).
pub const SEAM_STEP_FACTOR: f32 = 1.5;

/// VALIDATION-THRESHOLD: click onset detection level. Metronome fixture peak
/// is ≈ 0.285; this sits well below it and well above the noise floor.
pub const ONSET_THRESHOLD: f32 = 0.10;

/// VALIDATION-THRESHOLD: minimum analyzed loop wraps for T-SEAM (spec §6.3: ≥ 50).
pub const MIN_ANALYZED_WRAPS: u64 = 50;

/// VALIDATION-THRESHOLD: anti-stub mix RMS floor (spec §6.1) — silence or a
/// dead track fails this before any sync assertion can pass vacuously.
pub const MIN_MIX_RMS: f64 = 0.05;

/// VALIDATION-THRESHOLD: bypass within-wrap onset-grid sanity bound. Covers
/// the click sample's few-frame attack envelope only; wrap-to-wrap variance
/// must still be exactly zero (the drift assertion proper).
pub const BYPASS_GRID_TOLERANCE: i64 = 20;

/// VALIDATION-THRESHOLD: max spread (max − min) of stretch-path onset
/// detection offsets across all analyzed beats. The vocoder smears a click's
/// attack differently depending on where it lands relative to the analysis
/// hop grid — bounded by the constant ~120 ms window and exactly periodic
/// (measured: 80-sample spread, period 12 beats at rate 0.75). This bounds
/// the smear; the zero-drift assertion proper is exact periodicity of the
/// offset sequence (spec §1.1.3: smear inside a constant window is bounded
/// and non-accumulating; repositioning is impossible).
pub const STRETCH_ONSET_JITTER_MAX: i64 = 480;

/// VALIDATION-THRESHOLD (ADR-094): absolute onset-grid median bound for the
/// stretch path — the median offset of detected onsets against the derived
/// grid, after the engine's output-latency compensation. The C3 finding
/// (ADR-093) measured the UNcompensated stretch path a constant 2880 samples
/// (Signalsmith preset_default output latency, spec §2.3#4) late; with the
/// §3.2 feed-lead compensation the constant must be ~0. 96 (2 ms) is the same
/// class-separation bound as the device analyzer's TEMPO_CHANGE_STEP_MAX:
/// the vocoder's attack-smear bias stays within tens of samples, while every
/// defect class (uncompensated pre-roll 2880, stale re-anchor ≥ 1000, lost
/// render chunk ≥ 512) sits far outside it.
pub const STRETCH_GRID_MEDIAN_MAX: i64 = 96;

/// VALIDATION-THRESHOLD: longest admissible period (in beats) of the
/// stretch-path onset-offset pattern (= lcm(beat grid, vocoder hop grid) /
/// beat). Aperiodicity within this horizon means accumulation — drift.
pub const MAX_ONSET_PATTERN_PERIOD_BEATS: usize = 256;
