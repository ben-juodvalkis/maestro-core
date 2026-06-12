//! # T-PUMP — Plan B render pump equivalence (spec §3.3)
//!
//! Plan B's whole claim is "sync unaffected by construction; only latency
//! changes." This suite holds it to that:
//!
//! - the pumped sample stream is **bit-identical** to direct rendering of the
//!   same engine configuration with the same chunking (the pump adds a ring,
//!   never arithmetic);
//! - a consumer draining at varied, callback-like sizes sees zero underruns
//!   when the producer keeps up (occupancy pacing works);
//! - the callback-side `maestro_pump_consume` is allocation-free.

// Plan B is iOS delivery (ADR-090); the web engine renders directly in the
// worklet quantum and render_pump is not compiled for wasm32 (spec §3.8).
#![cfg(not(target_arch = "wasm32"))]

mod common;

use assert_no_alloc::{assert_no_alloc, AllocDisabler};
use common::{build_engine, T_INV_TRACKS};
use maestro_api::{Command, CommandResult};
use maestro_core::render_pump::{maestro_pump_consume, RenderPump, PUMP_CHUNK_FRAMES};

#[cfg(debug_assertions)]
#[global_allocator]
static A: AllocDisabler = AllocDisabler;

const TEST_FRAMES: usize = 480_000; // 10 s

/// Render the reference stream directly, in pump-sized chunks (the stretch
/// path's audio depends on block sizes, so equivalence is asserted against
/// identical chunking — positions never depend on it, per T-INV).
fn direct_reference() -> (Vec<f32>, Vec<f32>) {
    let (mut plane, mut core) = build_engine(120_000, &T_INV_TRACKS);
    assert!(matches!(plane.dispatch(Command::Play), CommandResult::Ok));
    let mut left = vec![0.0f32; PUMP_CHUNK_FRAMES];
    let mut right = vec![0.0f32; PUMP_CHUNK_FRAMES];
    let mut out_l = Vec::with_capacity(TEST_FRAMES);
    let mut out_r = Vec::with_capacity(TEST_FRAMES);
    while out_l.len() < TEST_FRAMES {
        core.render(&mut left, &mut right);
        out_l.extend_from_slice(&left);
        out_r.extend_from_slice(&right);
    }
    (out_l, out_r)
}

#[test]
fn pump_stream_is_bit_identical_to_direct_render() {
    let (ref_l, ref_r) = direct_reference();

    let (mut plane, core) = build_engine(120_000, &T_INV_TRACKS);
    assert!(matches!(plane.dispatch(Command::Play), CommandResult::Ok));
    let mut pump = RenderPump::start(core);
    let consumer = pump.consumer_ptr();

    // Callback-like consumption at varied block sizes (the OS picks; the ring
    // must not care). A real callback is paced by the hardware clock and the
    // producer stays ahead; here we pace on ring occupancy instead — pop only
    // when the ring holds a full block, so the test runs at full speed without
    // manufacturing starvation.
    let sizes = [240usize, 480, 128, 1024, 240, 333];
    let mut got_l = Vec::with_capacity(TEST_FRAMES);
    let mut got_r = Vec::with_capacity(TEST_FRAMES);
    let mut l = vec![0.0f32; 1024];
    let mut r = vec![0.0f32; 1024];
    let mut i = 0usize;
    while got_l.len() < TEST_FRAMES {
        let b = sizes[i % sizes.len()].min(TEST_FRAMES - got_l.len());
        let mut spins = 0u64;
        while pump.available_frames() < b {
            spins += 1;
            assert!(spins < 100_000_000, "producer stalled — ring never refilled");
            std::thread::yield_now();
        }
        unsafe { maestro_pump_consume(consumer, l.as_mut_ptr(), r.as_mut_ptr(), b as u32) };
        got_l.extend_from_slice(&l[..b]);
        got_r.extend_from_slice(&r[..b]);
        i += 1;
    }
    assert_eq!(pump.underruns(), 0, "paced consumption must never starve");

    assert_eq!(got_l, ref_l[..got_l.len()], "left stream diverged from direct render");
    assert_eq!(got_r, ref_r[..got_r.len()], "right stream diverged from direct render");

    // Anti-stub: the streams carry audio, not agreement-in-silence.
    let peak = got_l.iter().fold(0.0f32, |a, s| a.max(s.abs()));
    assert!(peak > 0.1, "pump equivalence proved over silence");
    drop(plane);
}

#[test]
fn pump_consume_is_allocation_free() {
    let (mut plane, core) = build_engine(120_000, &T_INV_TRACKS);
    assert!(matches!(plane.dispatch(Command::Play), CommandResult::Ok));
    let mut pump = RenderPump::start(core);
    let consumer = pump.consumer_ptr();
    // Let the producer fill the ring before the guarded consumption.
    std::thread::sleep(std::time::Duration::from_millis(100));

    let mut l = vec![0.0f32; 512];
    let mut r = vec![0.0f32; 512];
    assert_no_alloc(|| {
        for _ in 0..200 {
            unsafe { maestro_pump_consume(consumer, l.as_mut_ptr(), r.as_mut_ptr(), 512) };
        }
    });
    drop(plane);
}
