//! Pitch control: cents, semitones, and ratio conversions.
//!
//! Provides pure conversion functions between pitch representations
//! and a per-track `PitchController` that manages pitch offset state.
//!
//! Ported from:
//! - `audio-engine/core/track.ts` (setPitchOffset / getPitchOffset)
//! - `audio-engine/wasm/signalsmith-stretch-worker-enhanced.js` (semitones↔ratio math)
//! - `src/domains/coordination/services/pitch/PitchControlServiceImpl.ts`
//! - `src/components/AudioPlayer/PitchControl.tsx` (UI range constants)
//!
//! Browser-specific concerns stripped: ConstantSourceNode, AudioParam,
//! module-seamless-di.ts wiring, randomness coordinator service.
//! The Rust core owns the pitch state; the platform layer reads it.

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Minimum pitch offset in semitones (−2 octaves).
pub const MIN_PITCH_SEMITONES: f64 = -24.0;

/// Maximum pitch offset in semitones (+2 octaves).
pub const MAX_PITCH_SEMITONES: f64 = 24.0;

/// UI pitch control range: minimum (−1 octave).
pub const UI_MIN_PITCH_SEMITONES: f64 = -12.0;

/// UI pitch control range: maximum (+1 octave).
pub const UI_MAX_PITCH_SEMITONES: f64 = 12.0;

/// UI pitch step size (semitones per button click).
pub const UI_PITCH_STEP: f64 = 1.0;

/// Number of semitones per octave.
pub const SEMITONES_PER_OCTAVE: f64 = 12.0;

/// Number of cents per semitone.
pub const CENTS_PER_SEMITONE: f64 = 100.0;

/// Number of cents per octave.
pub const CENTS_PER_OCTAVE: f64 = 1200.0;

/// Tolerance for floating-point pitch comparisons.
const PITCH_TOLERANCE: f64 = 0.001;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that can occur during pitch operations.
#[derive(Debug, Clone, PartialEq)]
pub enum PitchError {
    /// Pitch ratio must be positive and finite.
    InvalidRatio(f64),
    /// Cents value must be finite.
    InvalidCents(f64),
    /// Semitones value must be finite.
    InvalidSemitones(f64),
    /// Track not found for pitch operation.
    TrackNotFound(String),
}

impl std::fmt::Display for PitchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PitchError::InvalidRatio(v) => {
                write!(f, "Invalid pitch ratio: {v}. Must be positive and finite")
            }
            PitchError::InvalidCents(v) => {
                write!(f, "Invalid cents value: {v}. Must be finite")
            }
            PitchError::InvalidSemitones(v) => {
                write!(f, "Invalid semitones value: {v}. Must be finite")
            }
            PitchError::TrackNotFound(id) => {
                write!(f, "Track '{id}' not found for pitch operation")
            }
        }
    }
}

impl std::error::Error for PitchError {}

// ---------------------------------------------------------------------------
// Pure conversion functions
// ---------------------------------------------------------------------------

/// Convert semitones to a pitch ratio.
///
/// Formula: `2^(semitones / 12)`
///
/// Examples:
/// -  0 semitones → 1.0   (no change)
/// - +12 semitones → 2.0   (octave up)
/// - −12 semitones → 0.5   (octave down)
/// -  +1 semitone  → ~1.0595
pub fn semitones_to_ratio(semitones: f64) -> Result<f64, PitchError> {
    if !semitones.is_finite() {
        return Err(PitchError::InvalidSemitones(semitones));
    }
    Ok(2.0_f64.powf(semitones / SEMITONES_PER_OCTAVE))
}

/// Convert a pitch ratio to semitones.
///
/// Formula: `12 * log2(ratio)`
///
/// Examples:
/// - 1.0 → 0 semitones
/// - 2.0 → 12 semitones (octave up)
/// - 0.5 → −12 semitones (octave down)
pub fn ratio_to_semitones(ratio: f64) -> Result<f64, PitchError> {
    if ratio <= 0.0 || !ratio.is_finite() {
        return Err(PitchError::InvalidRatio(ratio));
    }
    Ok(SEMITONES_PER_OCTAVE * ratio.log2())
}

/// Convert cents to a pitch ratio.
///
/// Formula: `2^(cents / 1200)`
pub fn cents_to_ratio(cents: f64) -> Result<f64, PitchError> {
    if !cents.is_finite() {
        return Err(PitchError::InvalidCents(cents));
    }
    Ok(2.0_f64.powf(cents / CENTS_PER_OCTAVE))
}

/// Convert a pitch ratio to cents.
///
/// Formula: `1200 * log2(ratio)`
pub fn ratio_to_cents(ratio: f64) -> Result<f64, PitchError> {
    if ratio <= 0.0 || !ratio.is_finite() {
        return Err(PitchError::InvalidRatio(ratio));
    }
    Ok(CENTS_PER_OCTAVE * ratio.log2())
}

/// Convert cents to semitones.
pub fn cents_to_semitones(cents: f64) -> Result<f64, PitchError> {
    if !cents.is_finite() {
        return Err(PitchError::InvalidCents(cents));
    }
    Ok(cents / CENTS_PER_SEMITONE)
}

/// Convert semitones to cents.
pub fn semitones_to_cents(semitones: f64) -> Result<f64, PitchError> {
    if !semitones.is_finite() {
        return Err(PitchError::InvalidSemitones(semitones));
    }
    Ok(semitones * CENTS_PER_SEMITONE)
}

/// Clamp a pitch offset (semitones) to the engine-supported range.
pub fn clamp_semitones(semitones: f64) -> f64 {
    semitones.clamp(MIN_PITCH_SEMITONES, MAX_PITCH_SEMITONES)
}

/// Clamp a pitch offset (semitones) to the UI range.
pub fn clamp_ui_semitones(semitones: f64) -> f64 {
    semitones.clamp(UI_MIN_PITCH_SEMITONES, UI_MAX_PITCH_SEMITONES)
}

/// Validate that a semitone offset is within the engine range.
pub fn is_valid_semitones(semitones: f64) -> bool {
    semitones.is_finite() && (MIN_PITCH_SEMITONES..=MAX_PITCH_SEMITONES).contains(&semitones)
}

// ---------------------------------------------------------------------------
// PitchState — per-track pitch state
// ---------------------------------------------------------------------------

/// Per-track pitch state.
///
/// The platform layer reads [`offset_semitones`] and passes it to the
/// Signalsmith Stretch DSP (via `_setTransposeSemitones` on WASM, or the
/// native Rust Signalsmith integration).
///
/// [`offset_semitones`]: PitchState::offset_semitones
#[derive(Debug, Clone, PartialEq)]
pub struct PitchState {
    /// Current pitch offset in semitones.
    offset_semitones: f64,
}

impl PitchState {
    /// Create a new pitch state at neutral (0 semitones).
    pub fn new() -> Self {
        Self {
            offset_semitones: 0.0,
        }
    }

    /// Current pitch offset in semitones.
    pub fn offset_semitones(&self) -> f64 {
        self.offset_semitones
    }

    /// Current pitch offset as a frequency ratio.
    pub fn offset_ratio(&self) -> f64 {
        // Safe: offset_semitones is always finite (clamped on set)
        semitones_to_ratio(self.offset_semitones).unwrap_or(1.0)
    }

    /// Current pitch offset in cents.
    pub fn offset_cents(&self) -> f64 {
        self.offset_semitones * CENTS_PER_SEMITONE
    }

    /// Set pitch offset in semitones.
    ///
    /// The value is clamped to [`MIN_PITCH_SEMITONES`, `MAX_PITCH_SEMITONES`].
    /// Returns `true` if the offset actually changed.
    pub fn set_offset_semitones(&mut self, semitones: f64) -> bool {
        let clamped = clamp_semitones(if semitones.is_finite() {
            semitones
        } else {
            0.0
        });
        if (self.offset_semitones - clamped).abs() < PITCH_TOLERANCE {
            return false;
        }
        self.offset_semitones = clamped;
        true
    }

    /// Set pitch offset in cents.
    ///
    /// Converted to semitones internally; clamped to engine range.
    /// Returns `true` if the offset actually changed.
    pub fn set_offset_cents(&mut self, cents: f64) -> bool {
        let semitones = if cents.is_finite() {
            cents / CENTS_PER_SEMITONE
        } else {
            0.0
        };
        self.set_offset_semitones(semitones)
    }

    /// Reset pitch to neutral (0 semitones).
    ///
    /// Returns `true` if the offset was not already neutral.
    pub fn reset(&mut self) -> bool {
        self.set_offset_semitones(0.0)
    }

    /// Whether the pitch offset is at neutral (0).
    pub fn is_neutral(&self) -> bool {
        self.offset_semitones.abs() < PITCH_TOLERANCE
    }
}

impl Default for PitchState {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Conversion function tests
    // -----------------------------------------------------------------------

    mod conversions {
        use super::*;

        // -- semitones ↔ ratio --

        #[test]
        fn semitones_to_ratio_zero() {
            let r = semitones_to_ratio(0.0).unwrap();
            assert!((r - 1.0).abs() < 0.0001);
        }

        #[test]
        fn semitones_to_ratio_octave_up() {
            let r = semitones_to_ratio(12.0).unwrap();
            assert!((r - 2.0).abs() < 0.0001);
        }

        #[test]
        fn semitones_to_ratio_octave_down() {
            let r = semitones_to_ratio(-12.0).unwrap();
            assert!((r - 0.5).abs() < 0.0001);
        }

        #[test]
        fn semitones_to_ratio_one_semitone() {
            let r = semitones_to_ratio(1.0).unwrap();
            // 2^(1/12) ≈ 1.05946
            assert!((r - 1.05946).abs() < 0.001);
        }

        #[test]
        fn semitones_to_ratio_two_octaves_up() {
            let r = semitones_to_ratio(24.0).unwrap();
            assert!((r - 4.0).abs() < 0.0001);
        }

        #[test]
        fn semitones_to_ratio_two_octaves_down() {
            let r = semitones_to_ratio(-24.0).unwrap();
            assert!((r - 0.25).abs() < 0.0001);
        }

        #[test]
        fn semitones_to_ratio_nan() {
            assert!(semitones_to_ratio(f64::NAN).is_err());
        }

        #[test]
        fn semitones_to_ratio_infinity() {
            assert!(semitones_to_ratio(f64::INFINITY).is_err());
        }

        #[test]
        fn ratio_to_semitones_unity() {
            let s = ratio_to_semitones(1.0).unwrap();
            assert!(s.abs() < 0.0001);
        }

        #[test]
        fn ratio_to_semitones_double() {
            let s = ratio_to_semitones(2.0).unwrap();
            assert!((s - 12.0).abs() < 0.0001);
        }

        #[test]
        fn ratio_to_semitones_half() {
            let s = ratio_to_semitones(0.5).unwrap();
            assert!((s - (-12.0)).abs() < 0.0001);
        }

        #[test]
        fn ratio_to_semitones_invalid() {
            assert!(ratio_to_semitones(0.0).is_err());
            assert!(ratio_to_semitones(-1.0).is_err());
            assert!(ratio_to_semitones(f64::NAN).is_err());
            assert!(ratio_to_semitones(f64::INFINITY).is_err());
        }

        // -- cents ↔ ratio --

        #[test]
        fn cents_to_ratio_zero() {
            let r = cents_to_ratio(0.0).unwrap();
            assert!((r - 1.0).abs() < 0.0001);
        }

        #[test]
        fn cents_to_ratio_octave() {
            let r = cents_to_ratio(1200.0).unwrap();
            assert!((r - 2.0).abs() < 0.0001);
        }

        #[test]
        fn cents_to_ratio_negative_octave() {
            let r = cents_to_ratio(-1200.0).unwrap();
            assert!((r - 0.5).abs() < 0.0001);
        }

        #[test]
        fn cents_to_ratio_100_cents() {
            // 100 cents = 1 semitone
            let r = cents_to_ratio(100.0).unwrap();
            let r_semitone = semitones_to_ratio(1.0).unwrap();
            assert!((r - r_semitone).abs() < 0.0001);
        }

        #[test]
        fn cents_to_ratio_nan() {
            assert!(cents_to_ratio(f64::NAN).is_err());
        }

        #[test]
        fn ratio_to_cents_unity() {
            let c = ratio_to_cents(1.0).unwrap();
            assert!(c.abs() < 0.01);
        }

        #[test]
        fn ratio_to_cents_octave() {
            let c = ratio_to_cents(2.0).unwrap();
            assert!((c - 1200.0).abs() < 0.01);
        }

        #[test]
        fn ratio_to_cents_invalid() {
            assert!(ratio_to_cents(0.0).is_err());
            assert!(ratio_to_cents(-1.0).is_err());
        }

        // -- cents ↔ semitones --

        #[test]
        fn cents_to_semitones_basic() {
            let s = cents_to_semitones(100.0).unwrap();
            assert!((s - 1.0).abs() < 0.0001);
        }

        #[test]
        fn cents_to_semitones_octave() {
            let s = cents_to_semitones(1200.0).unwrap();
            assert!((s - 12.0).abs() < 0.0001);
        }

        #[test]
        fn cents_to_semitones_negative() {
            let s = cents_to_semitones(-600.0).unwrap();
            assert!((s - (-6.0)).abs() < 0.0001);
        }

        #[test]
        fn cents_to_semitones_nan() {
            assert!(cents_to_semitones(f64::NAN).is_err());
        }

        #[test]
        fn semitones_to_cents_basic() {
            let c = semitones_to_cents(1.0).unwrap();
            assert!((c - 100.0).abs() < 0.001);
        }

        #[test]
        fn semitones_to_cents_negative() {
            let c = semitones_to_cents(-7.0).unwrap();
            assert!((c - (-700.0)).abs() < 0.001);
        }

        #[test]
        fn semitones_to_cents_nan() {
            assert!(semitones_to_cents(f64::NAN).is_err());
        }

        // -- round-trip tests --

        #[test]
        fn semitones_ratio_round_trip() {
            for s in [-24.0, -12.0, -7.0, -1.0, 0.0, 1.0, 5.0, 12.0, 24.0] {
                let ratio = semitones_to_ratio(s).unwrap();
                let back = ratio_to_semitones(ratio).unwrap();
                assert!(
                    (back - s).abs() < 0.0001,
                    "Round trip failed for {s} semitones: got {back}"
                );
            }
        }

        #[test]
        fn cents_ratio_round_trip() {
            for c in [-1200.0, -600.0, -100.0, 0.0, 100.0, 700.0, 1200.0] {
                let ratio = cents_to_ratio(c).unwrap();
                let back = ratio_to_cents(ratio).unwrap();
                assert!(
                    (back - c).abs() < 0.01,
                    "Round trip failed for {c} cents: got {back}"
                );
            }
        }

        #[test]
        fn cents_semitones_round_trip() {
            for s in [-12.0, -1.0, 0.0, 3.5, 12.0] {
                let cents = semitones_to_cents(s).unwrap();
                let back = cents_to_semitones(cents).unwrap();
                assert!(
                    (back - s).abs() < 0.0001,
                    "Round trip failed for {s} semitones: got {back}"
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Clamping and validation tests
    // -----------------------------------------------------------------------

    mod clamping {
        use super::*;

        #[test]
        fn clamp_semitones_within_range() {
            assert_eq!(clamp_semitones(0.0), 0.0);
            assert_eq!(clamp_semitones(-24.0), -24.0);
            assert_eq!(clamp_semitones(24.0), 24.0);
        }

        #[test]
        fn clamp_semitones_below_min() {
            assert_eq!(clamp_semitones(-30.0), MIN_PITCH_SEMITONES);
        }

        #[test]
        fn clamp_semitones_above_max() {
            assert_eq!(clamp_semitones(30.0), MAX_PITCH_SEMITONES);
        }

        #[test]
        fn clamp_ui_semitones_range() {
            assert_eq!(clamp_ui_semitones(0.0), 0.0);
            assert_eq!(clamp_ui_semitones(-12.0), -12.0);
            assert_eq!(clamp_ui_semitones(12.0), 12.0);
            assert_eq!(clamp_ui_semitones(-20.0), UI_MIN_PITCH_SEMITONES);
            assert_eq!(clamp_ui_semitones(20.0), UI_MAX_PITCH_SEMITONES);
        }

        #[test]
        fn is_valid_semitones_in_range() {
            assert!(is_valid_semitones(0.0));
            assert!(is_valid_semitones(-24.0));
            assert!(is_valid_semitones(24.0));
            assert!(is_valid_semitones(7.5));
        }

        #[test]
        fn is_valid_semitones_out_of_range() {
            assert!(!is_valid_semitones(-24.1));
            assert!(!is_valid_semitones(24.1));
            assert!(!is_valid_semitones(f64::NAN));
            assert!(!is_valid_semitones(f64::INFINITY));
        }
    }

    // -----------------------------------------------------------------------
    // PitchState tests
    // -----------------------------------------------------------------------

    mod pitch_state {
        use super::*;

        #[test]
        fn new_is_neutral() {
            let ps = PitchState::new();
            assert!(ps.is_neutral());
            assert!((ps.offset_semitones() - 0.0).abs() < 0.0001);
            assert!((ps.offset_ratio() - 1.0).abs() < 0.0001);
            assert!((ps.offset_cents() - 0.0).abs() < 0.01);
        }

        #[test]
        fn set_offset_semitones() {
            let mut ps = PitchState::new();
            let changed = ps.set_offset_semitones(5.0);
            assert!(changed);
            assert!((ps.offset_semitones() - 5.0).abs() < 0.0001);
            assert!(!ps.is_neutral());
        }

        #[test]
        fn set_offset_semitones_no_change() {
            let mut ps = PitchState::new();
            let changed = ps.set_offset_semitones(0.0);
            assert!(!changed);
        }

        #[test]
        fn set_offset_semitones_clamped() {
            let mut ps = PitchState::new();
            ps.set_offset_semitones(30.0);
            assert!((ps.offset_semitones() - MAX_PITCH_SEMITONES).abs() < 0.0001);

            ps.set_offset_semitones(-30.0);
            assert!((ps.offset_semitones() - MIN_PITCH_SEMITONES).abs() < 0.0001);
        }

        #[test]
        fn set_offset_semitones_nan_resets() {
            let mut ps = PitchState::new();
            ps.set_offset_semitones(5.0);
            ps.set_offset_semitones(f64::NAN);
            assert!(ps.is_neutral());
        }

        #[test]
        fn set_offset_cents() {
            let mut ps = PitchState::new();
            let changed = ps.set_offset_cents(700.0);
            assert!(changed);
            assert!((ps.offset_semitones() - 7.0).abs() < 0.0001);
            assert!((ps.offset_cents() - 700.0).abs() < 0.01);
        }

        #[test]
        fn set_offset_cents_nan_resets() {
            let mut ps = PitchState::new();
            ps.set_offset_semitones(5.0);
            ps.set_offset_cents(f64::NAN);
            assert!(ps.is_neutral());
        }

        #[test]
        fn offset_ratio_octave_up() {
            let mut ps = PitchState::new();
            ps.set_offset_semitones(12.0);
            assert!((ps.offset_ratio() - 2.0).abs() < 0.0001);
        }

        #[test]
        fn offset_ratio_octave_down() {
            let mut ps = PitchState::new();
            ps.set_offset_semitones(-12.0);
            assert!((ps.offset_ratio() - 0.5).abs() < 0.0001);
        }

        #[test]
        fn reset() {
            let mut ps = PitchState::new();
            ps.set_offset_semitones(7.0);
            let changed = ps.reset();
            assert!(changed);
            assert!(ps.is_neutral());
        }

        #[test]
        fn reset_already_neutral() {
            let mut ps = PitchState::new();
            let changed = ps.reset();
            assert!(!changed);
        }

        #[test]
        fn default_is_neutral() {
            let ps = PitchState::default();
            assert!(ps.is_neutral());
        }

        #[test]
        fn set_offset_semitones_small_change_within_tolerance() {
            let mut ps = PitchState::new();
            ps.set_offset_semitones(5.0);
            // A change smaller than tolerance should not register
            let changed = ps.set_offset_semitones(5.0005);
            assert!(!changed);
            assert!((ps.offset_semitones() - 5.0).abs() < 0.001);
        }

        #[test]
        fn cents_and_ratio_consistency() {
            let mut ps = PitchState::new();
            ps.set_offset_semitones(7.0);

            // cents = semitones * 100
            assert!((ps.offset_cents() - 700.0).abs() < 0.01);

            // ratio = 2^(7/12) ≈ 1.4983
            let expected_ratio = 2.0_f64.powf(7.0 / 12.0);
            assert!((ps.offset_ratio() - expected_ratio).abs() < 0.0001);
        }
    }
}
