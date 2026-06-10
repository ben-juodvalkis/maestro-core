//! # RT-safety test (spec §3.3): the render loop performs ZERO heap allocation
//!
//! Wraps full renders — including the first playing block, where Play/Install
//! command application and the stretcher anchor procedure run — in
//! `assert_no_alloc`, which panics on any Rust-allocator activity inside the
//! guarded scope. Together with the clippy `disallowed-types` lint (no locks)
//! this enforces the §3.3 RT rules in CI rather than aspiring to them.
//!
//! Known boundary: Signalsmith's C++ internals don't route through the Rust
//! allocator, so this guard can't see them. The wrapper is configured fully at
//! construction (control-plane side) and its `process`/`seek` calls are
//! documented allocation-free; render-time CPU is separately gated by the
//! device budget telemetry (C0).

mod common;

use assert_no_alloc::{assert_no_alloc, AllocDisabler};
use common::{build_engine, T_INV_TRACKS};
use maestro_api::{Command, CommandResult};

#[cfg(debug_assertions)]
#[global_allocator]
static A: AllocDisabler = AllocDisabler;

#[test]
fn render_loop_never_allocates() {
    // All allocation happens here: engine creation, PCM load, scratch prep.
    let (mut plane, mut core) = build_engine(120_000, &T_INV_TRACKS);
    assert!(matches!(plane.dispatch(Command::Play), CommandResult::Ok));

    let mut left = vec![0.0f32; 4096];
    let mut right = vec![0.0f32; 4096];

    // Every block — including the very first, which applies the queued
    // Install/Activate/Play commands and runs the anchor procedure — must be
    // allocation-free.
    assert_no_alloc(|| {
        for (i, &b) in [128usize, 480, 1024, 4096, 333, 480, 128, 1024].iter().cycle().take(400).enumerate() {
            let _ = i;
            core.render(&mut left[..b], &mut right[..b]);
        }
    });

    // The engine actually played (anti-stub): nonzero output reached the bus.
    let peak = left.iter().chain(right.iter()).fold(0.0f32, |a, s| a.max(s.abs()));
    assert!(peak > 0.0, "render produced silence — RT test exercised nothing");
}

/// Deactivation/reactivation mid-stream (re-anchor under ramp) is also
/// allocation-free — the reactivation path is exactly a restart (spec §3.2).
#[test]
fn reactivation_render_never_allocates() {
    let (mut plane, mut core) = build_engine(120_000, &T_INV_TRACKS);
    assert!(matches!(plane.dispatch(Command::Play), CommandResult::Ok));
    let mut left = vec![0.0f32; 1024];
    let mut right = vec![0.0f32; 1024];
    for _ in 0..50 {
        core.render(&mut left, &mut right);
    }
    assert!(matches!(plane.dispatch(Command::Deactivate { slot: 0 }), CommandResult::Ok));
    assert!(matches!(plane.dispatch(Command::Activate { slot: 0 }), CommandResult::Ok));
    assert_no_alloc(|| {
        for _ in 0..100 {
            core.render(&mut left, &mut right);
        }
    });
}
