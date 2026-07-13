//! # T-CTRL — control-plane storm test (spec §6.7, every commit)
//!
//! Seeded randomized command fuzz against the REAL render entry: arm/disarm,
//! tempo, gain/pan/mute/solo, pitch, variation switches (cached AND async),
//! pause/resume and stop/restart storms at bursts up to 60 commands/s for
//! 10 M frames. At every checkpoint the T-INV invariants must hold — every
//! track's dictated position equals the exact `num-bigint` rational reference
//! **bit-for-bit** (tolerance zero); RT queue overflow is surfaced, never
//! silent; and snapshot fields stay consistent (`loop_progress` monotone
//! within a wrap, tempo mirror exact).
//!
//! Anti-stub pairing (spec §6.1): the run must be audible (mix RMS floor),
//! must actually execute its command budget, and must observe real variation
//! switches and tempo echoes — a storm that no-ops cannot pass. The negative
//! control for the overflow gate lives in `event_queue_overflow_is_surfaced`:
//! it deliberately overflows the RT event queue and asserts the drop counter
//! and `EventQueueOverflow` event fire — proving the §6.7 gate can fail.

mod common;

use common::reference::RefTimeline;
use common::{build_engine_select, make_sine_pcm, T_INV_TRACKS};
use maestro_api::{Command, CommandResult, Event, LoadMeta, QuantizePolicy, TransportPolicy};
use num_bigint::BigUint;

const TOTAL_FRAMES: u64 = 10_000_000;
const TEMPO0: u32 = 120_000;
/// 60 commands/s at 48 kHz = one command per 800 frames (the burst floor).
const MIN_SPACING_FRAMES: u64 = 800;
const BLOCKS: [usize; 3] = [128, 480, 1024];

/// Deterministic LCG (same constants as the engine's sabotage jitter) — the
/// storm is seeded and replayable.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 11
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn f32_unit(&mut self) -> f32 {
        (self.next() % 10_000) as f32 / 10_000.0
    }
}

fn storm(seed: u64) {
    let tracks = T_INV_TRACKS;
    let (mut plane, mut core) = build_engine_select(TEMPO0, &tracks, &[0, 1, 2, 3]);
    assert!(matches!(plane.dispatch(Command::Play), CommandResult::Ok));

    let mut rng = Lcg(seed);
    let max_block = *BLOCKS.iter().max().unwrap();
    let mut left = vec![0.0f32; max_block];
    let mut right = vec![0.0f32; max_block];

    // Harness-owned mirrors (never read back from the engine).
    let mut ref_tl = RefTimeline::new(TEMPO0);
    let mut engine_f: u64 = 0;
    // Pause/stop windows render inline (frozen) and restart before the loop
    // continues, so the outer loop only ever runs in the playing state.
    let playing = true;
    let paused = false;
    let mut tempo_now = TEMPO0;

    // Bookkeeping for the snapshot-consistency gate.
    // (epoch, loop_beats, count, progress) — loop_beats scopes the
    // monotonicity check to one variation's tenure: a length-changing switch
    // (ADR-109) legitimately re-bases count AND progress.
    let mut last_loop: Vec<Option<(u32, u32, u32, f32)>> = vec![None; tracks.len()];

    // Anti-stub counters.
    let mut commands_accepted = 0usize;
    let mut rejections_seen = 0usize;
    let mut variations_applied = 0usize;
    let mut tempo_echoes = 0usize;
    let mut overflow_events = 0usize;
    let mut sumsq = 0.0f64;
    let mut weight = 0u64;
    let mut mismatches = 0usize;

    let mut next_command_at: u64 = 4_800;
    let mut next_checkpoint: u64 = 500_000;
    let mut block_i = 0usize;

    // Respond to async variation loads (spec §4.11): decode is the platform's
    // job — here the platform is this closure. Variations carry DIFFERENT
    // per-variation loop lengths (ADR-109) around the installed baseline
    // (÷4 / baseline / ×2, all self-consistent against the shared O), so the
    // storm soaks length-changing swaps under the full T-INV checkpointing —
    // dictated positions are Λ-independent, so the reference math is
    // unchanged.
    let var_geometry = |slot: usize, variation: u8| -> (u32, u64) {
        let t = &tracks[slot];
        let beats = match variation {
            0 => t.loop_beats,
            1 => t.loop_beats / 4,
            _ => t.loop_beats * 2,
        };
        // Exact rounded loop identity: pcm_len = round(beats·K/O).
        const K: u128 = 2_880_000_000;
        let o = t.o_mbpm as u128;
        let len = ((beats as u128 * K + o / 2) / o) as u64;
        (beats, len)
    };
    let respond_load = |plane: &mut maestro_core::control_plane::ControlPlane,
                        slot: usize,
                        variation: u8| {
        let t = &tracks[slot];
        let (beats, len) = var_geometry(slot, variation);
        let (l, r) = make_sine_pcm(len, 220.0 * (variation as f32 + 1.0));
        plane
            .load_pcm(
                &LoadMeta { cache_only: false,
                    slot: slot as u8,
                    sample_rate: 48_000,
                    original_tempo_mbpm: t.o_mbpm,
                    loop_beats: beats,
                    encoder_delay: 0,
                    end_padding: 0,
                    original_samples: len,
                    variation,
                    gain: 1.0,
                    pan: t.pan,
                },
                l,
                r,
            )
            .expect("async variation load completes");
    };

    let checkpoint = |core: &maestro_core::audio_core::AudioCore,
                      engine_f: u64,
                      ref_tl: &RefTimeline,
                      mismatches: &mut usize| {
        let bu = ref_tl.bu(engine_f);
        for (slot, t) in tracks.iter().enumerate() {
            let engine_pos = core.dictated_unwrapped_position(slot).expect("installed");
            let ref_pos = common::reference::u(&bu, t.o_mbpm);
            if engine_pos != ref_pos {
                *mismatches += 1;
            }
        }
    };

    while engine_f < TOTAL_FRAMES {
        // ---- Fire a command burst when due (commands land at exactly this
        // block boundary because block sizes are clamped below). ----
        if playing && !paused && engine_f >= next_command_at {
            checkpoint(&core, engine_f, &ref_tl, &mut mismatches);
            let burst = 1 + rng.below(3);
            for _ in 0..burst {
                match rng.below(14) {
                    0 => {
                        // Valid tempo for the §6.2 track set under the §4.8
                        // clamp: [40k, 240k].
                        let mbpm = 40_000 + rng.below(200_001) as u32;
                        ref_tl.set_tempo(engine_f, mbpm);
                        tempo_now = mbpm;
                        let res = plane.dispatch(Command::SetTempo { mbpm });
                        assert!(matches!(res, CommandResult::Ok), "valid tempo rejected: {res:?}");
                        commands_accepted += 1;
                    }
                    1 => {
                        // Out-of-range tempo MUST be rejected and must not
                        // perturb anything (the mirror is not updated).
                        let mbpm = if rng.below(2) == 0 { 250_000 + rng.below(60_000) as u32 } else { 30_000 };
                        let res = plane.dispatch(Command::SetTempo { mbpm });
                        assert!(
                            matches!(res, CommandResult::Rejected { .. }),
                            "out-of-range tempo accepted: {mbpm}"
                        );
                        rejections_seen += 1;
                    }
                    2 | 3 => {
                        let slot = rng.below(4) as u8;
                        let quantize = if rng.below(2) == 0 {
                            QuantizePolicy::Immediate
                        } else {
                            QuantizePolicy::NextBar
                        };
                        let res = plane.dispatch(Command::Activate {
                            slot,
                            quantize,
                            transport: TransportPolicy::Manual,
                        });
                        assert!(matches!(res, CommandResult::Ok));
                        commands_accepted += 1;
                    }
                    4 | 5 => {
                        let slot = rng.below(4) as u8;
                        let res = plane.dispatch(Command::Deactivate {
                            slot,
                            transport: TransportPolicy::Manual,
                        });
                        assert!(matches!(res, CommandResult::Ok));
                        commands_accepted += 1;
                    }
                    6 => {
                        let slot = rng.below(4) as u8;
                        let gain = rng.f32_unit() * 2.0;
                        assert!(matches!(
                            plane.dispatch(Command::SetGain { slot, gain }),
                            CommandResult::Ok
                        ));
                        commands_accepted += 1;
                    }
                    7 => {
                        let slot = rng.below(4) as u8;
                        let pan = rng.f32_unit() * 2.0 - 1.0;
                        assert!(matches!(
                            plane.dispatch(Command::SetPan { slot, pan }),
                            CommandResult::Ok
                        ));
                        commands_accepted += 1;
                    }
                    8 => {
                        let slot = rng.below(4) as u8;
                        let muted = rng.below(2) == 0;
                        assert!(matches!(
                            plane.dispatch(Command::SetMute { slot, muted }),
                            CommandResult::Ok
                        ));
                        commands_accepted += 1;
                    }
                    9 => {
                        let slot = rng.below(4) as u8;
                        let solo = rng.below(3) == 0; // mostly off, occasionally on
                        assert!(matches!(
                            plane.dispatch(Command::SetSolo { slot, solo }),
                            CommandResult::Ok
                        ));
                        commands_accepted += 1;
                    }
                    10 => {
                        let slot = rng.below(4) as u8;
                        let semitones = (rng.f32_unit() - 0.5) * 8.0;
                        assert!(matches!(
                            plane.dispatch(Command::SetPitch { slot, semitones }),
                            CommandResult::Ok
                        ));
                        commands_accepted += 1;
                    }
                    11 => {
                        // Variation switch — async on first touch, cached after
                        // (§4.11). The load response below completes pending ones.
                        let slot = rng.below(4) as u8;
                        let variation = rng.below(3) as u8;
                        assert!(matches!(
                            plane.dispatch(Command::SwitchVariation { slot, variation }),
                            CommandResult::Ok
                        ));
                        commands_accepted += 1;
                    }
                    12 => {
                        // Pause storm: freeze, render frozen blocks, resume.
                        assert!(matches!(plane.dispatch(Command::Pause), CommandResult::Ok));
                        let before: Vec<u64> = (0..4)
                            .map(|s| core.dictated_unwrapped_position(s).unwrap())
                            .collect();
                        for k in 0..(1 + rng.below(6) as usize) {
                            let b = BLOCKS[(block_i + k) % BLOCKS.len()];
                            core.render(&mut left[..b], &mut right[..b]);
                        }
                        let after: Vec<u64> = (0..4)
                            .map(|s| core.dictated_unwrapped_position(s).unwrap())
                            .collect();
                        assert_eq!(before, after, "F advanced during pause at {engine_f}");
                        assert!(matches!(plane.dispatch(Command::Resume), CommandResult::Ok));
                        commands_accepted += 2;
                    }
                    _ => {
                        // Stop → frozen render → restart (epoch reset flow).
                        assert!(matches!(plane.dispatch(Command::Stop), CommandResult::Ok));
                        let tempo = ref_tl.tempo_at(engine_f);
                        ref_tl.reset(engine_f, tempo);
                        for k in 0..(1 + rng.below(4) as usize) {
                            let b = BLOCKS[(block_i + k) % BLOCKS.len()];
                            core.render(&mut left[..b], &mut right[..b]);
                        }
                        assert!(matches!(plane.dispatch(Command::Play), CommandResult::Ok));
                        // Reactivate everything so the storm stays audible.
                        for slot in 0..4u8 {
                            let _ = plane.dispatch(Command::Activate {
                                slot,
                                quantize: QuantizePolicy::Immediate,
                                transport: TransportPolicy::Manual,
                            });
                        }
                        // Clear mute/solo so the anti-stub RMS floor is fair.
                        for slot in 0..4u8 {
                            let _ = plane.dispatch(Command::SetMute { slot, muted: false });
                            let _ = plane.dispatch(Command::SetSolo { slot, solo: false });
                            let _ = plane.dispatch(Command::SetGain { slot, gain: 1.0 });
                        }
                        commands_accepted += 2;
                        // Loop-progress tracking restarts with the epoch.
                        last_loop.iter_mut().for_each(|s| *s = None);
                    }
                }
            }
            next_command_at = engine_f + MIN_SPACING_FRAMES + rng.below(30_000);
        }

        // ---- Render one block, clamped onto the next command frame. ----
        let mut b = BLOCKS[block_i % BLOCKS.len()];
        block_i += 1;
        if playing && !paused {
            b = b.min((next_command_at.saturating_sub(engine_f)).max(1) as usize);
        }
        core.render(&mut left[..b], &mut right[..b]);
        if playing && !paused {
            engine_f += b as u64;
            for i in 0..b {
                sumsq += (left[i] as f64).powi(2) + (right[i] as f64).powi(2);
            }
            weight += b as u64;
        }

        // ---- Drain events; answer async variation loads; count echoes. ----
        if block_i.is_multiple_of(16) {
            for e in plane.drain_events() {
                match e {
                    Event::VariationLoadRequired { slot, variation } => {
                        respond_load(&mut plane, slot as usize, variation);
                    }
                    Event::VariationApplied { .. } => variations_applied += 1,
                    Event::TempoApplied { .. } => tempo_echoes += 1,
                    Event::EventQueueOverflow { .. } => overflow_events += 1,
                    _ => {}
                }
            }
        }

        // ---- Snapshot consistency (spec §6.7). ----
        if block_i.is_multiple_of(64) {
            let snap = plane.snapshot();
            assert!(
                snap.tempo_mbpm >= 40_000 && snap.tempo_mbpm <= 240_000,
                "snapshot tempo out of storm range: {}",
                snap.tempo_mbpm
            );
            if playing && !paused && snap.engine_frame == engine_f {
                // At an exactly-known frame the snapshot tempo must equal the
                // mirror (commands land at scripted boundaries).
                assert_eq!(snap.tempo_mbpm, tempo_now, "snapshot tempo mirror diverged");
            }
            for t in &snap.tracks {
                let slot = t.slot as usize;
                // loop_progress monotone within a wrap (same epoch + map +
                // count — a length-changing variation switch re-bases both,
                // ADR-109).
                if let Some((epoch, beats, count, progress)) = last_loop[slot] {
                    if epoch == snap.timeline_epoch
                        && beats == t.loop_beats
                        && count == t.loop_count
                        && playing
                        && !paused
                    {
                        assert!(
                            t.loop_progress >= progress - 1e-6,
                            "loop_progress went backwards within wrap {} of slot {slot}: {} → {}",
                            count,
                            progress,
                            t.loop_progress
                        );
                    }
                }
                last_loop[slot] =
                    Some((snap.timeline_epoch, t.loop_beats, t.loop_count, t.loop_progress));
                assert!((0.0..1.0 + 1e-6).contains(&t.loop_progress));
            }
        }

        if engine_f >= next_checkpoint {
            next_checkpoint += 500_000;
            checkpoint(&core, engine_f, &ref_tl, &mut mismatches);
        }
    }

    // Final drain + gates.
    for e in plane.drain_events() {
        match e {
            Event::VariationApplied { .. } => variations_applied += 1,
            Event::TempoApplied { .. } => tempo_echoes += 1,
            Event::EventQueueOverflow { .. } => overflow_events += 1,
            _ => {}
        }
    }
    checkpoint(&core, engine_f, &ref_tl, &mut mismatches);

    // T-INV invariant: zero position mismatches, tolerance zero (spec §6.7).
    assert_eq!(mismatches, 0, "storm produced position drift (seed {seed})");

    // RT queue overflow is surfaced, never silent — and a healthy storm with
    // regular drains must not overflow at all.
    let snap = plane.snapshot();
    assert_eq!(
        snap.events_dropped, 0,
        "RT event queue overflowed during the storm (seed {seed}) — drains every 16 blocks \
         must keep up at 60 commands/s"
    );
    assert_eq!(overflow_events, 0);

    // Anti-stub (spec §6.1): the storm really happened and was audible.
    let mix_rms = (sumsq / (weight.max(1) as f64 * 2.0)).sqrt();
    assert!(
        mix_rms > 0.01,
        "storm mix RMS {mix_rms} ≈ silence — the fuzz exercised nothing (seed {seed})"
    );
    assert!(commands_accepted >= 400, "only {commands_accepted} commands ran (seed {seed})");
    assert!(rejections_seen >= 10, "rejection path untested (seed {seed})");
    assert!(variations_applied >= 5, "no real variation switches observed (seed {seed})");
    assert!(tempo_echoes >= 20, "no tempo echoes observed (seed {seed})");

    // The pairwise musical-phase bound (spec §1.1) at the end, for good measure.
    let bu = ref_tl.bu(engine_f);
    for (slot, t) in tracks.iter().enumerate() {
        let pos = core.dictated_unwrapped_position(slot).unwrap();
        let phase = &bu - BigUint::from(pos) * BigUint::from(t.o_mbpm);
        assert!(phase < BigUint::from(t.o_mbpm));
    }
    drop(plane);
}

#[test]
fn t_ctrl_storm_seed_1() {
    storm(0xC0FFEE);
}

#[test]
fn t_ctrl_storm_seed_2() {
    storm(0xDEAD_BEEF_CAFE);
}

/// Negative control (spec §6.1/§6.7): the overflow gate must be able to fail.
/// Without any drains, a long run generates more edge events (loop wraps +
/// echoes) than the 256-slot RT queue holds; the drop counter must surface in
/// the snapshot and `EventQueueOverflow` must fire on the next drain.
#[test]
fn event_queue_overflow_is_surfaced() {
    let tracks = T_INV_TRACKS;
    let (mut plane, mut core) = build_engine_select(TEMPO0, &tracks, &[0, 1, 2, 3]);
    assert!(matches!(plane.dispatch(Command::Play), CommandResult::Ok));
    let mut left = vec![0.0f32; 1024];
    let mut right = vec![0.0f32; 1024];

    // Slot 0 wraps every 384_000 frames at tempo 120k (16 beats @ 60k → ×2
    // rate). Mute toggles generate two echoes per toggle. NO drains happen.
    let mut toggles = 0u32;
    for i in 0..40_000usize {
        if i % 100 == 0 {
            let muted = (i / 100) % 2 == 0;
            let _ = plane.dispatch(Command::SetMute { slot: 1, muted });
            toggles += 1;
        }
        core.render(&mut left, &mut right);
    }
    assert!(toggles > 300);
    let snap = plane.snapshot();
    assert!(
        snap.events_dropped > 0,
        "queue must have overflowed without drains — the §6.7 gate cannot fail, so it \
         proves nothing"
    );
    let events = plane.drain_events();
    let surfaced: u32 = events
        .iter()
        .filter_map(|e| match e {
            Event::EventQueueOverflow { dropped } => Some(*dropped),
            _ => None,
        })
        .sum();
    assert_eq!(surfaced, snap.events_dropped, "every drop must be surfaced exactly once");
}
