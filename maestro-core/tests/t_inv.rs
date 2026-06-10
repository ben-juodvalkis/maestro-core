//! # T-INV — zero-drift invariant test (spec §6.2), M0 legs
//!
//! Drives the **real** `AudioCore` render entry (not internal helpers) with
//! the §6.2 track set — O = {60k, 90k, 113k, 116k} mBPM, loop lengths from
//! real metadata including the non-integer musical-length cases (§2.2) — at
//! engine tempo 120_000, block sizes alternating {128, 480, 1024}, for ≥ 50 M
//! output frames (≈ 17 min of audio) spanning > 100 wraps of the shortest
//! loop.
//!
//! **Tolerance: zero.** Engine positions must equal the `num-bigint` rational
//! reference bit-for-bit at every checkpoint.
//!
//! M1 adds: tempo-change legs (one at a loop boundary, one mid-bar),
//! pause/resume, mid-stream arming, and the stop → tempo-change-while-stopped
//! → restart leg. The negative controls live in `t_inv_negative.rs` (built
//! with `--features drift-injection`; CI asserts they DETECT each sabotage).

mod common;

use common::{run_t_inv_scenario, T_INV_TRACKS};

const TEMPO: u32 = 120_000;
const CHECKPOINT: u64 = 1_000_000;

/// The full M0 invariant run. ≥ 50 M frames ⇒ 130 wraps of the shortest loop
/// (384_000 output frames at tempo 120) and ~25 wraps of the longest.
#[test]
fn t_inv_50m_frames_zero_drift_bit_exact() {
    let result = run_t_inv_scenario(50_000_000, TEMPO, CHECKPOINT);

    // Anti-stub (spec §6.1): the mix must be audible, not silent.
    assert!(
        result.mix_rms > 0.05,
        "anti-stub: mix RMS {} below floor — green-over-silence is forbidden",
        result.mix_rms
    );

    assert!(result.checkpoints.len() >= 50, "expected ≥ 50 checkpoints");

    // Zero tolerance: every engine position equals the rational reference.
    for cp in &result.checkpoints {
        assert_eq!(
            cp.positions, cp.reference,
            "drift detected at rendered frame {} — engine vs exact rational reference",
            cp.rendered_frames
        );
    }
    assert_eq!(result.position_mismatches, 0);

    // Pairwise musical-phase bound (spec §1.1): u_i·O_i ∈ (bu − O_i, bu] for
    // every pair at every checkpoint — constant-bounded forever, no growth.
    for cp in &result.checkpoints {
        let bu = cp.rendered_frames as u128 * TEMPO as u128;
        for (i, t) in T_INV_TRACKS.iter().enumerate() {
            let phase = bu - cp.positions[i] as u128 * t.o_mbpm as u128;
            assert!(
                phase < t.o_mbpm as u128,
                "track {i} musical phase {phase} outside [0, O) at frame {}",
                cp.rendered_frames
            );
        }
    }

    // Cross-platform parity digest (spec §6.6): printed so CI can compare the
    // dump byte-for-byte across x86 / ARM (and wasm32 from M4).
    println!("T-INV position digest: {:#018x}", result.position_digest());
}

/// Determinism (spec §1.1 #1): two replays of the same command history produce
/// bit-identical position sequences.
#[test]
fn t_inv_replay_is_bit_identical() {
    let a = run_t_inv_scenario(5_000_000, TEMPO, CHECKPOINT);
    let b = run_t_inv_scenario(5_000_000, TEMPO, CHECKPOINT);
    assert_eq!(a.position_digest(), b.position_digest());
    for (ca, cb) in a.checkpoints.iter().zip(&b.checkpoints) {
        assert_eq!(ca.positions, cb.positions);
    }
}
