//! # T-INV negative controls (spec §6.2)
//!
//! Built only with `--features drift-injection`. Each sabotage mode
//! reintroduces a historic defect class; **T-INV must detect all three** —
//! if a sabotage mode produces zero position mismatches, the suite cannot
//! catch drift at the 1-sample level and this test FAILS the build.
//!
//! All three modes run inside one `#[test]` because the sabotage selector is
//! process-global; parallel test threads would interfere.

#![cfg(feature = "drift-injection")]

mod common;

use common::run_t_inv_scenario;
use maestro_core::sabotage::{set_mode, Sabotage};

const TEMPO: u32 = 120_000;

#[test]
fn t_inv_detects_all_sabotage_modes() {
    // 6 M frames: ≥ 7 wraps of the non-integer loops — enough for every mode
    // to exceed one sample of drift (pcm_length_loop drifts ~0.78 samples per
    // wrap of the Humboldt geometry; float_cursor and timer_clock are faster).
    let frames = 6_000_000;

    for (mode, name) in [
        (Sabotage::FloatCursor, "float_cursor"),
        (Sabotage::PcmLengthLoop, "pcm_length_loop"),
        (Sabotage::TimerClock, "timer_clock"),
    ] {
        set_mode(mode);
        let result = run_t_inv_scenario(frames, TEMPO, 1_000_000);
        set_mode(Sabotage::None);
        assert!(
            result.position_mismatches > 0,
            "T-INV FAILED TO DETECT the `{name}` sabotage — the invariant suite \
             is broken and cannot catch drift at the 1-sample level (spec §6.2)"
        );
        println!("{name}: detected ({} checkpoint mismatches)", result.position_mismatches);
    }

    // Control of the control: with sabotage off, the same short scenario is clean.
    let clean = run_t_inv_scenario(frames, TEMPO, 1_000_000);
    assert_eq!(clean.position_mismatches, 0, "clean run must be drift-free");
}
