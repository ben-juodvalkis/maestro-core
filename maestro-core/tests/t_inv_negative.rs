//! # T-INV negative controls (spec §6.2)
//!
//! Built only with `--features drift-injection`. Each sabotage mode
//! reintroduces a historic defect class; **T-INV must detect all of them** —
//! if a sabotage mode produces zero position mismatches, the suite cannot
//! catch drift at the 1-sample level and this test FAILS the build.
//!
//! M1 extends the controls across the new legs: every mode now runs through
//! a scenario containing a live tempo change, a pause/resume, and a
//! deactivate → tempo-change-while-silent → reactivate sequence; and the new
//! `stale_reanchor` mode sabotages the tempo change itself (defect D7 — the
//! exact control-plane re-anchoring class that produced the abandoned
//! branch's post-"done" failures).
//!
//! All modes run inside one `#[test]` because the sabotage selector is
//! process-global; parallel test threads would interfere.

#![cfg(feature = "drift-injection")]

mod common;

use common::{run_scripted_scenario, run_t_inv_scenario, ScriptAction, ScriptEvent};
use maestro_api::{Command, QuantizePolicy};
use maestro_core::sabotage::{set_mode, Sabotage};

const TEMPO: u32 = 120_000;

/// Compact M1 leg script inside a 6 M-frame window: tempo change at a slot-0
/// loop boundary (3.84 M), pause/resume, deactivate slot 0, tempo change
/// while it is silent, reactivate. Every sabotage mode must be detected
/// running THROUGH these legs — not only on the M0 straight-line path.
fn m1_negative_script() -> Vec<ScriptEvent> {
    vec![
        ScriptEvent {
            at_frame: 3_840_000,
            action: ScriptAction::Cmd(Command::SetTempo { mbpm: 87_000 }),
        },
        ScriptEvent { at_frame: 4_200_000, action: ScriptAction::Cmd(Command::Pause) },
        ScriptEvent { at_frame: 4_200_000, action: ScriptAction::IdleBlocksFrozen(5) },
        ScriptEvent { at_frame: 4_200_000, action: ScriptAction::Cmd(Command::Resume) },
        ScriptEvent {
            at_frame: 4_600_000,
            action: ScriptAction::Cmd(Command::Deactivate { slot: 0, transport: Default::default() }),
        },
        ScriptEvent {
            at_frame: 4_800_000,
            action: ScriptAction::Cmd(Command::SetTempo { mbpm: 96_000 }),
        },
        ScriptEvent {
            at_frame: 5_000_000,
            action: ScriptAction::Cmd(Command::Activate {
                slot: 0,
                quantize: QuantizePolicy::Immediate,
             transport: Default::default(), }),
        },
    ]
}

#[test]
fn t_inv_detects_all_sabotage_modes() {
    // 6 M frames: ≥ 7 wraps of the non-integer loops — enough for every mode
    // to exceed one sample of drift (pcm_length_loop drifts ~0.78 samples per
    // wrap of the Humboldt geometry; float_cursor and timer_clock are faster;
    // stale_reanchor jumps ~2000 samples at the sabotaged tempo change).
    let frames = 6_000_000;

    for (mode, name) in [
        (Sabotage::FloatCursor, "float_cursor"),
        (Sabotage::PcmLengthLoop, "pcm_length_loop"),
        (Sabotage::TimerClock, "timer_clock"),
        (Sabotage::StaleReanchor, "stale_reanchor"),
    ] {
        // The M1 legs: tempo change, pause/resume, silent-window tempo
        // change, re-entry. stale_reanchor can ONLY be caught here (it
        // corrupts tempo changes; the M0 straight line has none).
        set_mode(mode);
        let result = std::panic::catch_unwind(|| {
            run_scripted_scenario(frames, TEMPO, 1_000_000, &[0, 1, 2, 3], &m1_negative_script())
        });
        set_mode(Sabotage::None);
        let detected = match result {
            // timer_clock can desync the harness's frame mirror from the
            // engine enough to trip the frozen-pause assertion — that IS a
            // detection (the clock moved when it must not).
            Err(_) => true,
            Ok(r) => r.position_mismatches > 0,
        };
        assert!(
            detected,
            "T-INV FAILED TO DETECT the `{name}` sabotage on the M1 legs — the invariant \
             suite is broken and cannot catch drift at the 1-sample level (spec §6.2)"
        );
        println!("{name}: detected on M1 legs");
    }

    // The three M0 modes must also still be caught on the straight-line path.
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
            "T-INV FAILED TO DETECT the `{name}` sabotage (M0 path) — spec §6.2"
        );
        println!("{name}: detected ({} checkpoint mismatches)", result.position_mismatches);
    }

    // Control of the control: with sabotage off, both scenarios are clean.
    let clean = run_t_inv_scenario(frames, TEMPO, 1_000_000);
    assert_eq!(clean.position_mismatches, 0, "clean M0 run must be drift-free");
    let clean_m1 =
        run_scripted_scenario(frames, TEMPO, 1_000_000, &[0, 1, 2, 3], &m1_negative_script());
    assert_eq!(clean_m1.position_mismatches, 0, "clean M1-legs run must be drift-free");

    // Parity-lane negative control (spec §6.6 / §6.1, M4): the cross-platform
    // position digest must MOVE under drift. The CI parity lane compares
    // digests byte-for-byte across x86 / arm64 / wasm32 — a digest that
    // hashed something insensitive to the defect classes would wave a
    // sabotaged build straight through that comparison. This runs on every
    // lane, INCLUDING wasm32, so the wasm parity path itself is proven to
    // fail on a drift-injection build.
    set_mode(Sabotage::FloatCursor);
    let sabotaged = run_t_inv_scenario(frames, TEMPO, 1_000_000);
    set_mode(Sabotage::None);
    assert_ne!(
        clean.position_digest(),
        sabotaged.position_digest(),
        "the §6.6 parity digest did not change under float_cursor sabotage — \
         the parity lane cannot detect drift and is broken"
    );
    println!(
        "parity negative control: sabotaged digest {:#018x} != clean {:#018x}",
        sabotaged.position_digest(),
        clean.position_digest()
    );
}
