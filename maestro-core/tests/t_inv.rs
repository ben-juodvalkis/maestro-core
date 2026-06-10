//! # T-INV — zero-drift invariant test (spec §6.2), full M1 surface
//!
//! Drives the **real** `AudioCore` render entry (not internal helpers) with
//! the §6.2 track set — O = {60k, 90k, 113k, 116k} mBPM, loop lengths from
//! real metadata including the non-integer musical-length cases (§2.2) — for
//! ≥ 50 M output frames, block sizes alternating {128, 480, 1024}, spanning
//! > 100 wraps of the shortest loop, with the full M1 leg schedule:
//!
//! - a tempo change **exactly at a loop boundary** of the shortest loop;
//! - a tempo change **mid-bar** (which also moves the 116k track onto the
//!   unity bypass — the unity↔stretch crossfade may never touch positions);
//! - a **pause/resume** (F freezes for all tracks at once; no epoch reset);
//! - a **mid-stream arm** with `NextBar` quantization (activation frame `F_q`
//!   by exact integer inversion of `bu`, asserted against the engine's echo);
//! - the **stop → tempo-change-while-stopped → restart** leg: a track is
//!   deactivated, the tempo changes twice while it is silent (once mid-bar,
//!   once exactly at the silent track's would-be loop boundary), then the
//!   track re-enters — its positions must equal the reference dictate
//!   bit-for-bit and its pairwise offsets to the tracks that played through
//!   must be exactly the reference's.
//!
//! **Tolerance: zero.** Engine positions must equal the `num-bigint` rational
//! reference bit-for-bit at every checkpoint (1 M grid + every scripted
//! frame). The negative controls live in `t_inv_negative.rs` (built with
//! `--features drift-injection`; CI asserts they DETECT each sabotage,
//! including the M1 `stale_reanchor` tempo-change mode).

mod common;

use common::reference::RefTimeline;
use common::{
    run_scripted_scenario, run_t_inv_scenario, ScriptAction, ScriptEvent, BEATS_PER_BAR,
    T_INV_TRACKS,
};
use maestro_api::{Command, Event, QuantizePolicy};
use num_bigint::BigUint;

const TEMPO0: u32 = 120_000;
const CHECKPOINT: u64 = 1_000_000;

// ---- The M1 leg schedule (engine frames; tempo changes land exactly here
// because the runner clamps block sizes onto scripted frames). ----
const F_TEMPO_LOOP_BOUNDARY: u64 = 3_840_000; // exactly 10 wraps of slot 0 @ 120k
const TEMPO_LOOP_BOUNDARY: u32 = 87_000;
const F_TEMPO_MIDBAR: u64 = 6_000_000; // mid-bar; 116k ⇒ slot 3 goes unity
const TEMPO_MIDBAR: u32 = 116_000;
const F_PAUSE: u64 = 9_000_000;
const F_ARM: u64 = 12_000_000; // Activate slot 1, NextBar, dispatched mid-bar
const F_DEACT: u64 = 20_000_000; // Deactivate slot 0
const F_TEMPO_MIDBAR_SILENT: u64 = 22_000_020; // mid-bar, slot 0 silent; chosen so
const TEMPO_MIDBAR_SILENT: u32 = 90_000; // slot 0's next loop boundary lands on a frame
const F_REACT: u64 = 26_000_000; // Reactivate slot 0 (immediate) — re-entry
const TEMPO_FINAL: u32 = 120_000; // applied exactly at slot 0's silent loop boundary
const TOTAL_FRAMES: u64 = 55_000_000;

/// Λ of slot 0 (16 beats): the shortest loop, whose wraps gate the schedule.
fn lambda_slot0() -> BigUint {
    BigUint::from(T_INV_TRACKS[0].loop_beats) * common::reference::k()
}

/// Build the reference mirror of the scripted tempo history and derive the
/// computed schedule values: the silent-loop-boundary frame for the final
/// tempo change, and the quantized activation frame F_q for the arm leg.
/// Everything is asserted exact — a mis-chosen constant fails loudly here,
/// not silently downstream.
fn m1_schedule() -> (RefTimeline, u64, u64) {
    let mut tl = RefTimeline::new(TEMPO0);
    let bar = BigUint::from(BEATS_PER_BAR) * common::reference::k();
    let lambda0 = lambda_slot0();

    // Leg 1: tempo change exactly at a loop boundary of slot 0.
    assert_eq!(
        tl.bu(F_TEMPO_LOOP_BOUNDARY) % &lambda0,
        BigUint::ZERO,
        "schedule error: F_TEMPO_LOOP_BOUNDARY is not a slot-0 loop boundary"
    );
    tl.set_tempo(F_TEMPO_LOOP_BOUNDARY, TEMPO_LOOP_BOUNDARY);

    // Leg 2: mid-bar tempo change (must NOT be a bar boundary).
    assert_ne!(
        tl.bu(F_TEMPO_MIDBAR) % &bar,
        BigUint::ZERO,
        "schedule error: F_TEMPO_MIDBAR is a bar boundary"
    );
    tl.set_tempo(F_TEMPO_MIDBAR, TEMPO_MIDBAR);

    // Leg 4: quantized arm — dispatched mid-bar; F_q = exact inversion of the
    // next bar boundary in beat-units.
    let bu_arm = tl.bu(F_ARM);
    assert_ne!(&bu_arm % &bar, BigUint::ZERO, "schedule error: F_ARM is a bar boundary");
    let bu_q = (&bu_arm / &bar + 1u32) * &bar;
    let f_q = tl.invert_bu(F_ARM, &bu_q);
    assert!(tl.bu(f_q) >= bu_q && tl.bu(f_q - 1) < bu_q, "F_q inversion not exact");
    assert!(f_q > F_ARM && f_q < F_DEACT, "schedule error: F_q outside its leg window");

    // Leg 5: mid-bar tempo change while slot 0 is silent…
    assert_ne!(
        tl.bu(F_TEMPO_MIDBAR_SILENT) % &bar,
        BigUint::ZERO,
        "schedule error: F_TEMPO_MIDBAR_SILENT is a bar boundary"
    );
    tl.set_tempo(F_TEMPO_MIDBAR_SILENT, TEMPO_MIDBAR_SILENT);

    // …then a tempo change EXACTLY at slot 0's would-be loop boundary. The
    // 90k segment is chosen so the boundary lands on an integer frame; both
    // facts are asserted exactly.
    let bu_now = tl.bu(F_TEMPO_MIDBAR_SILENT);
    let target = (&bu_now / &lambda0 + 1u32) * &lambda0;
    let f_boundary = tl.invert_bu(F_TEMPO_MIDBAR_SILENT, &target);
    assert_eq!(
        tl.bu(f_boundary),
        target,
        "schedule error: slot 0's loop boundary does not land on an integer frame \
         under the 90k segment"
    );
    assert!(f_boundary < F_REACT, "schedule error: boundary change must precede re-entry");
    tl.set_tempo(f_boundary, TEMPO_FINAL);

    (tl, f_boundary, f_q)
}

fn m1_script(f_boundary: u64) -> Vec<ScriptEvent> {
    vec![
        ScriptEvent {
            at_frame: F_TEMPO_LOOP_BOUNDARY,
            action: ScriptAction::Cmd(Command::SetTempo { mbpm: TEMPO_LOOP_BOUNDARY }),
        },
        ScriptEvent {
            at_frame: F_TEMPO_MIDBAR,
            action: ScriptAction::Cmd(Command::SetTempo { mbpm: TEMPO_MIDBAR }),
        },
        ScriptEvent { at_frame: F_PAUSE, action: ScriptAction::Cmd(Command::Pause) },
        ScriptEvent { at_frame: F_PAUSE, action: ScriptAction::IdleBlocksFrozen(7) },
        ScriptEvent { at_frame: F_PAUSE, action: ScriptAction::Cmd(Command::Resume) },
        ScriptEvent {
            at_frame: F_ARM,
            action: ScriptAction::Cmd(Command::Activate {
                slot: 1,
                quantize: QuantizePolicy::NextBar,
            }),
        },
        ScriptEvent {
            at_frame: F_DEACT,
            action: ScriptAction::Cmd(Command::Deactivate { slot: 0 }),
        },
        ScriptEvent {
            at_frame: F_TEMPO_MIDBAR_SILENT,
            action: ScriptAction::Cmd(Command::SetTempo { mbpm: TEMPO_MIDBAR_SILENT }),
        },
        ScriptEvent {
            at_frame: f_boundary,
            action: ScriptAction::Cmd(Command::SetTempo { mbpm: TEMPO_FINAL }),
        },
        ScriptEvent {
            at_frame: F_REACT,
            action: ScriptAction::Cmd(Command::Activate {
                slot: 0,
                quantize: QuantizePolicy::Immediate,
            }),
        },
    ]
}

/// The full §6.2 invariant run with every M1 leg. ≥ 50 M engine frames,
/// > 100 wraps of the shortest loop across four tempo segments.
#[test]
fn t_inv_full_m1_scenario_zero_drift_bit_exact() {
    let (ref_tl, f_boundary, f_q) = m1_schedule();
    let script = m1_script(f_boundary);

    // Slot 1 starts loaded-but-silent: it is the mid-stream arm.
    let result = run_scripted_scenario(TOTAL_FRAMES, TEMPO0, CHECKPOINT, &[0, 2, 3], &script);

    // Anti-stub (spec §6.1): the mix must be audible, not silent.
    assert!(
        result.mix_rms > common::validation_thresholds::MIN_MIX_RMS,
        "anti-stub: mix RMS {} below floor — green-over-silence is forbidden",
        result.mix_rms
    );

    assert!(result.checkpoints.len() >= 50, "expected ≥ 50 checkpoints");

    // > 100 wraps of the shortest loop across the whole run (spec §6.2).
    let final_wraps = ref_tl.bu(TOTAL_FRAMES) / lambda_slot0();
    assert!(
        final_wraps >= BigUint::from(100u32),
        "scenario spans only {final_wraps} wraps of the shortest loop"
    );

    // Zero tolerance: every engine position equals the rational reference —
    // through both tempo changes, the pause, the arm, and the re-entry.
    for cp in &result.checkpoints {
        assert_eq!(
            cp.positions, cp.reference,
            "drift detected at engine frame {} — engine vs exact rational reference",
            cp.rendered_frames
        );
    }
    assert_eq!(result.position_mismatches, 0);

    // Pairwise musical-phase bound (spec §1.1): bu − u_i·O_i ∈ [0, O_i) for
    // every track at every checkpoint — constant-bounded forever, no growth.
    for cp in &result.checkpoints {
        let bu = ref_tl.bu(cp.rendered_frames);
        for (i, t) in T_INV_TRACKS.iter().enumerate() {
            let phase = &bu - BigUint::from(cp.positions[i]) * BigUint::from(t.o_mbpm);
            assert!(
                phase < BigUint::from(t.o_mbpm),
                "track {i} musical phase {phase} outside [0, O) at frame {}",
                cp.rendered_frames
            );
        }
    }

    // Re-entry equality (spec §6.2): from the re-entry frame on, slot 0's
    // pairwise offsets to the tracks that played through the silent-window
    // tempo changes equal the reference's exactly (zero tolerance). The
    // offsets are signed bu-scale differences u_0·O_0 − u_j·O_j.
    let mut saw_post_reentry_checkpoint = false;
    for cp in result.checkpoints.iter().filter(|cp| cp.rendered_frames >= F_REACT) {
        saw_post_reentry_checkpoint = true;
        for j in [2usize, 3] {
            let eng = i256_diff(
                cp.positions[0],
                T_INV_TRACKS[0].o_mbpm,
                cp.positions[j],
                T_INV_TRACKS[j].o_mbpm,
            );
            let refr = i256_diff(
                cp.reference[0],
                T_INV_TRACKS[0].o_mbpm,
                cp.reference[j],
                T_INV_TRACKS[j].o_mbpm,
            );
            assert_eq!(
                eng, refr,
                "re-entry pairwise offset (slot 0 vs {j}) diverged at frame {}",
                cp.rendered_frames
            );
        }
    }
    assert!(saw_post_reentry_checkpoint, "no checkpoint after re-entry — scenario broken");

    // The engine's command echoes carry the exact application frames the
    // device analyzer will rely on (spec §6.4): assert each leg's echo.
    let tempo_echoes: Vec<(u32, u64)> = result
        .events
        .iter()
        .filter_map(|e| match e {
            Event::TempoApplied { mbpm, at_frame } => Some((*mbpm, *at_frame)),
            _ => None,
        })
        .collect();
    assert_eq!(
        tempo_echoes,
        vec![
            (TEMPO_LOOP_BOUNDARY, F_TEMPO_LOOP_BOUNDARY),
            (TEMPO_MIDBAR, F_TEMPO_MIDBAR),
            (TEMPO_MIDBAR_SILENT, F_TEMPO_MIDBAR_SILENT),
            (TEMPO_FINAL, f_boundary),
        ],
        "tempo echo events must carry the exact application frames"
    );
    assert!(
        result.events.iter().any(|e| matches!(e, Event::Paused { at_frame } if *at_frame == F_PAUSE)),
        "missing Paused echo at {F_PAUSE}"
    );
    assert!(
        result.events.iter().any(|e| matches!(e, Event::Resumed { at_frame } if *at_frame == F_PAUSE)),
        "missing Resumed echo at {F_PAUSE} (F frozen across the pause)"
    );

    // Quantized arm: the engine's activation echo must equal the exact
    // integer inversion F_q — sample-exact "start at next bar" (spec §3.1).
    let activations: Vec<(u8, u64)> = result
        .events
        .iter()
        .filter_map(|e| match e {
            Event::TrackActivated { slot, at_frame } => Some((*slot, *at_frame)),
            _ => None,
        })
        .collect();
    assert!(
        activations.contains(&(1, f_q)),
        "slot 1 quantized activation: expected F_q = {f_q}, got activations {activations:?}"
    );
    assert!(
        activations.contains(&(0, F_REACT)),
        "slot 0 re-entry: expected immediate activation at {F_REACT}, got {activations:?}"
    );
    assert!(
        result.events.iter().any(
            |e| matches!(e, Event::TrackDeactivated { slot: 0, at_frame } if *at_frame == F_DEACT)
        ),
        "missing slot 0 deactivation echo at {F_DEACT}"
    );

    // Cross-platform parity digest (spec §6.6): printed so CI can compare the
    // dump byte-for-byte across x86 / ARM (and wasm32 from M4).
    println!("T-INV position digest: {:#018x}", result.position_digest());
}

/// Signed difference u_a·O_a − u_b·O_b without leaving exact integers.
fn i256_diff(u_a: u64, o_a: u32, u_b: u64, o_b: u32) -> (bool, BigUint) {
    let a = BigUint::from(u_a) * BigUint::from(o_a);
    let b = BigUint::from(u_b) * BigUint::from(o_b);
    if a >= b {
        (true, a - b)
    } else {
        (false, b - a)
    }
}

/// The M0 fixed-tempo run is kept as the cross-arch digest baseline and the
/// block-size-independence witness (spec §6.2): 50 M frames, no script.
#[test]
fn t_inv_50m_frames_zero_drift_bit_exact() {
    let result = run_t_inv_scenario(50_000_000, TEMPO0, CHECKPOINT);

    assert!(
        result.mix_rms > common::validation_thresholds::MIN_MIX_RMS,
        "anti-stub: mix RMS {} below floor — green-over-silence is forbidden",
        result.mix_rms
    );
    assert!(result.checkpoints.len() >= 50, "expected ≥ 50 checkpoints");
    for cp in &result.checkpoints {
        assert_eq!(
            cp.positions, cp.reference,
            "drift detected at rendered frame {} — engine vs exact rational reference",
            cp.rendered_frames
        );
    }
    assert_eq!(result.position_mismatches, 0);

    for cp in &result.checkpoints {
        let bu = cp.rendered_frames as u128 * TEMPO0 as u128;
        for (i, t) in T_INV_TRACKS.iter().enumerate() {
            let phase = bu - cp.positions[i] as u128 * t.o_mbpm as u128;
            assert!(
                phase < t.o_mbpm as u128,
                "track {i} musical phase {phase} outside [0, O) at frame {}",
                cp.rendered_frames
            );
        }
    }

    println!("T-INV position digest: {:#018x}", result.position_digest());
}

/// Determinism (spec §1.1 #1): two replays of the same command history —
/// including the full M1 script — produce bit-identical position sequences.
#[test]
fn t_inv_replay_is_bit_identical() {
    let (_, f_boundary, _) = m1_schedule();
    let script_a = m1_script(f_boundary);
    let script_b = m1_script(f_boundary);
    let a = run_scripted_scenario(30_000_000, TEMPO0, CHECKPOINT, &[0, 2, 3], &script_a);
    let b = run_scripted_scenario(30_000_000, TEMPO0, CHECKPOINT, &[0, 2, 3], &script_b);
    assert_eq!(a.position_digest(), b.position_digest());
    for (ca, cb) in a.checkpoints.iter().zip(&b.checkpoints) {
        assert_eq!(ca.positions, cb.positions);
    }
}
