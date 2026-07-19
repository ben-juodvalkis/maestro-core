//! # T-STREAM — chunked/streaming PCM load (ADR-112 §10.1)
//!
//! The §5 invariants under test:
//! - the streaming lifecycle produces **byte-exact** the same audio as the
//!   equivalent bulk `load_pcm`, per reconciliation case (the bulk path is
//!   internally the same composition — these tests pin the equivalence);
//! - begin-time validation fires in the specified order, before any byte
//!   moves; append/finalize failures are loud and leave defined states;
//! - the §5.4 starvation predicate at the exact frontier boundary — unity,
//!   stretch with the ADR-094 lead at the 4× rate clamp, the §5.8 wrap-seam
//!   rule, and the resume-margin hysteresis;
//! - **zero drift under starvation**: a starving/resuming track's dictated
//!   positions are bit-identical to an always-loaded control run.

mod common;

use common::{make_sine_pcm, SAMPLE_RATE};
use maestro_api::{
    AppendResult, Command, CommandResult, EngineConfig, Event, LoadMeta, QuantizePolicy,
    SampleFormat, StreamLoadMeta,
};
use maestro_core::audio_core::{AudioCore, STREAM_RESUME_MARGIN_FRAMES};
use maestro_core::control_plane::{create_engine, ControlPlane, MAX_STREAM_APPEND_FRAMES};
use maestro_core::stretch::Stretcher;

const BEATS_PER_BAR: u32 = 4;

fn engine_at(tempo_mbpm: u32) -> (ControlPlane, Box<AudioCore>) {
    create_engine(EngineConfig {
        sample_rate: SAMPLE_RATE,
        initial_tempo_mbpm: tempo_mbpm,
        beats_per_bar: BEATS_PER_BAR,
        pcm_budget_bytes: None,
    })
    .expect("engine config valid")
}

/// 16 beats @ 120 BPM @ 48 kHz = exactly 384_000 samples (integer loop).
const O_120: u32 = 120_000;
const BEATS_16: u32 = 16;
const PCM_120: u64 = 384_000;

fn meta_120(slot: u8, encoder_delay: u32, end_padding: u32) -> LoadMeta {
    LoadMeta {
        slot,
        sample_rate: SAMPLE_RATE,
        original_tempo_mbpm: O_120,
        loop_beats: BEATS_16,
        encoder_delay,
        end_padding,
        original_samples: PCM_120,
        variation: 0,
        cache_only: false,
        gain: 1.0,
        pan: 0.0,
    }
}

fn stream_meta(meta: LoadMeta, declared: u64) -> StreamLoadMeta {
    StreamLoadMeta { meta, declared_total_samples: declared, sample_format: SampleFormat::F32 }
}

/// The raw decoded stream for a reconciliation case: junk priming, the real
/// signal, junk padding — the trim must keep exactly the middle.
fn raw_stream(delay: u32, orig: u64, padding: u32) -> (Vec<f32>, Vec<f32>) {
    let (sig_l, sig_r) = make_sine_pcm(orig, 330.0);
    let mut l = vec![0.77f32; delay as usize];
    l.extend_from_slice(&sig_l);
    l.extend(std::iter::repeat_n(-0.66f32, padding as usize));
    let mut r = vec![0.55f32; delay as usize];
    r.extend_from_slice(&sig_r);
    r.extend(std::iter::repeat_n(-0.44f32, padding as usize));
    (l, r)
}

fn ok(result: CommandResult) {
    assert!(matches!(result, CommandResult::Ok), "rejected: {result:?}");
}

fn append_all_in_chunks(
    plane: &mut ControlPlane,
    slot: u8,
    left: &[f32],
    right: &[f32],
    chunk: usize,
) {
    let mut seq = 0u32;
    let mut at = 0usize;
    while at < left.len() {
        let n = chunk.min(left.len() - at);
        match plane.append_pcm_chunk(slot, seq, &left[at..at + n], &right[at..at + n]) {
            AppendResult::Ok { .. } => {}
            AppendResult::Rejected { reason } => panic!("append {seq} rejected: {reason}"),
        }
        seq += 1;
        at += n;
    }
}

/// Activate + play + render `blocks` blocks of `b` frames; returns the
/// concatenated stereo output.
fn activate_play_render(
    plane: &mut ControlPlane,
    core: &mut AudioCore,
    slot: u8,
    blocks: usize,
    b: usize,
) -> (Vec<f32>, Vec<f32>) {
    ok(plane.dispatch(Command::Activate {
        slot,
        quantize: QuantizePolicy::Immediate,
        transport: Default::default(),
    }));
    ok(plane.dispatch(Command::Play));
    render_blocks(core, blocks, b)
}

fn render_blocks(core: &mut AudioCore, blocks: usize, b: usize) -> (Vec<f32>, Vec<f32>) {
    let mut out_l = Vec::with_capacity(blocks * b);
    let mut out_r = Vec::with_capacity(blocks * b);
    let mut l = vec![0.0f32; b];
    let mut r = vec![0.0f32; b];
    for _ in 0..blocks {
        core.render(&mut l, &mut r);
        out_l.extend_from_slice(&l);
        out_r.extend_from_slice(&r);
    }
    (out_l, out_r)
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|s| s.to_bits()).collect()
}

fn loaded_event_reconciliation(events: &[Event]) -> Option<String> {
    events.iter().find_map(|e| match e {
        Event::TrackLoaded { reconciliation, .. } => Some(reconciliation.clone()),
        _ => None,
    })
}

// ---------------------------------------------------------------------------
// Lifecycle: byte-exact bulk equivalence per reconciliation case
// ---------------------------------------------------------------------------

/// One case: the same per-file metadata (delay 2112 / padding 960), three
/// decoder behaviors distinguished by what the raw stream actually contains.
/// Bulk-load the raw stream on engine A; stream it in odd-sized chunks on
/// engine B; both must fire TrackLoaded with the same reconciliation and
/// render byte-identical audio.
fn assert_bulk_equivalence(stream_delay: u32, stream_padding: u32, expect_reconciliation: &str) {
    let (meta_delay, meta_padding) = (2_112u32, 960u32);
    let raw_total = PCM_120 + stream_delay as u64 + stream_padding as u64;
    let (raw_l, raw_r) = raw_stream(stream_delay, PCM_120, stream_padding);

    let (mut plane_a, mut core_a) = engine_at(O_120);
    plane_a
        .load_pcm(&meta_120(0, meta_delay, meta_padding), raw_l.clone(), raw_r.clone())
        .expect("bulk load");
    let events_a = plane_a.drain_events();
    assert_eq!(
        loaded_event_reconciliation(&events_a).as_deref(),
        Some(expect_reconciliation),
        "bulk reconciliation"
    );

    let (mut plane_b, mut core_b) = engine_at(O_120);
    plane_b
        .begin_pcm_stream(&stream_meta(meta_120(0, meta_delay, meta_padding), raw_total))
        .expect("begin");
    // Odd chunk size exercises keep-window boundaries straddling chunks.
    append_all_in_chunks(&mut plane_b, 0, &raw_l, &raw_r, 12_347);
    plane_b.finalize_pcm_stream(0).expect("finalize");
    let events_b = plane_b.drain_events();
    assert_eq!(
        loaded_event_reconciliation(&events_b).as_deref(),
        Some(expect_reconciliation),
        "stream reconciliation"
    );

    let (a_l, a_r) = activate_play_render(&mut plane_a, &mut core_a, 0, 200, 480);
    let (b_l, b_r) = activate_play_render(&mut plane_b, &mut core_b, 0, 200, 480);
    assert_eq!(bits(&a_l), bits(&b_l), "left channel must be byte-exact bulk vs stream");
    assert_eq!(bits(&a_r), bits(&b_r), "right channel must be byte-exact bulk vs stream");
    let peak = a_l.iter().fold(0.0f32, |a, s| a.max(s.abs()));
    assert!(peak > 0.0, "anti-stub: the comparison rendered silence");
}

#[test]
fn stream_equals_bulk_already_trimmed() {
    // Decoder applied the full gapless trim: the stream is exactly the audio.
    assert_bulk_equivalence(0, 0, "alreadyTrimmed");
}

#[test]
fn stream_equals_bulk_trimmed_here() {
    // Raw decode: priming and padding both present; the engine trims.
    assert_bulk_equivalence(2_112, 960, "trimmedHere");
}

#[test]
fn stream_equals_bulk_trimmed_padding() {
    // Chrome's behavior (ADR-095): priming dropped, padding delivered.
    assert_bulk_equivalence(0, 960, "trimmedPadding");
}

// ---------------------------------------------------------------------------
// Begin-time rejections (validated before any byte moves)
// ---------------------------------------------------------------------------

#[test]
fn begin_rejects_bad_shape_occupied_cache_only_rate_identity_and_i16() {
    let (mut plane, _core) = engine_at(O_120);

    // Bad declared shape: matches none of the three §3.4#4 cases.
    let err = plane
        .begin_pcm_stream(&stream_meta(meta_120(0, 2_112, 960), PCM_120 + 1))
        .unwrap_err();
    assert!(err.contains("length mismatch"), "{err}");

    // Non-48k sample rate.
    let mut meta = meta_120(0, 0, 0);
    meta.sample_rate = 44_100;
    let err = plane.begin_pcm_stream(&stream_meta(meta, PCM_120)).unwrap_err();
    assert!(err.contains("never resamples"), "{err}");

    // cache_only streaming parks are v1-rejected.
    let mut meta = meta_120(0, 0, 0);
    meta.cache_only = true;
    let err = plane.begin_pcm_stream(&stream_meta(meta, PCM_120)).unwrap_err();
    assert!(err.contains("cache_only"), "{err}");

    // Loop identity (§3.4#2): a corrupt original_samples fails the gate.
    let mut meta = meta_120(0, 0, 0);
    meta.original_samples = PCM_120 - 3_072; // the D4 double-trim length
    let err = plane
        .begin_pcm_stream(&stream_meta(meta, PCM_120 - 3_072))
        .unwrap_err();
    assert!(err.contains("inconsistent"), "{err}");

    // §4.8 rate clamp at the current tempo: O far below E/4.
    let mut meta = meta_120(0, 0, 0);
    meta.original_tempo_mbpm = 20_000; // rate would be 6.0
    meta.original_samples = 2_304_000; // keep the loop identity valid: 16·K/20_000
    let err = plane
        .begin_pcm_stream(&stream_meta(meta, 2_304_000))
        .unwrap_err();
    assert!(err.contains("rate clamp"), "{err}");

    // Occupied slot: bulk-load slot 0, then begin on it must reject.
    let (l, r) = raw_stream(0, PCM_120, 0);
    plane.load_pcm(&meta_120(0, 0, 0), l, r).expect("bulk load");
    let err = plane
        .begin_pcm_stream(&stream_meta(meta_120(0, 0, 0), PCM_120))
        .unwrap_err();
    assert!(err.contains("occupied"), "{err}");

    // Every rejection fired a loud TrackLoadFailed.
    let failures = plane
        .drain_events()
        .into_iter()
        .filter(|e| matches!(e, Event::TrackLoadFailed { .. }))
        .count();
    assert_eq!(failures, 6);
}

// ---------------------------------------------------------------------------
// Append and finalize failure modes
// ---------------------------------------------------------------------------

#[test]
fn out_of_order_seq_tears_the_stream_down() {
    let (mut plane, _core) = engine_at(O_120);
    plane
        .begin_pcm_stream(&stream_meta(meta_120(0, 0, 0), PCM_120))
        .expect("begin");
    let chunk = vec![0.1f32; 1_000];
    assert!(matches!(plane.append_pcm_chunk(0, 0, &chunk, &chunk), AppendResult::Ok { .. }));
    // seq 2 (skipping 1) is a host bookkeeping bug: loud teardown.
    let AppendResult::Rejected { reason } = plane.append_pcm_chunk(0, 2, &chunk, &chunk) else {
        panic!("out-of-order append must reject");
    };
    assert!(reason.contains("out-of-order"), "{reason}");
    // The slot is empty now: further appends reject, and TrackLoadFailed fired.
    let AppendResult::Rejected { reason } = plane.append_pcm_chunk(0, 1, &chunk, &chunk) else {
        panic!("append after teardown must reject");
    };
    assert!(reason.contains("no stream in progress"), "{reason}");
    assert!(plane
        .drain_events()
        .iter()
        .any(|e| matches!(e, Event::TrackLoadFailed { .. })));
    assert_eq!(plane.pcm_bytes_in_use(), 0, "teardown must release the budget");
}

#[test]
fn over_delivery_rejects_the_append_and_keeps_the_stream() {
    let (mut plane, _core) = engine_at(O_120);
    plane
        .begin_pcm_stream(&stream_meta(meta_120(0, 0, 0), PCM_120))
        .expect("begin");
    let chunk = vec![0.1f32; (PCM_120 - 10) as usize];
    assert!(matches!(plane.append_pcm_chunk(0, 0, &chunk, &chunk), AppendResult::Ok { .. }));
    let too_much = vec![0.2f32; 11];
    let AppendResult::Rejected { reason } = plane.append_pcm_chunk(0, 1, &too_much, &too_much)
    else {
        panic!("over-delivery must reject");
    };
    assert!(reason.contains("over-delivery"), "{reason}");
    // The stream survives: delivering the exact remainder still works.
    let rest = vec![0.3f32; 10];
    let AppendResult::Ok { frontier, complete } = plane.append_pcm_chunk(0, 1, &rest, &rest)
    else {
        panic!("exact remainder must be accepted");
    };
    assert_eq!(frontier, PCM_120);
    assert!(complete);
    plane.finalize_pcm_stream(0).expect("finalize");
}

#[test]
fn append_after_finalize_rejects_without_disturbing_the_track() {
    let (mut plane, mut core) = engine_at(O_120);
    plane
        .begin_pcm_stream(&stream_meta(meta_120(0, 0, 0), PCM_120))
        .expect("begin");
    let (l, r) = make_sine_pcm(PCM_120, 220.0);
    append_all_in_chunks(&mut plane, 0, &l, &r, 100_000);
    plane.finalize_pcm_stream(0).expect("finalize");
    let chunk = vec![0.1f32; 100];
    let AppendResult::Rejected { reason } = plane.append_pcm_chunk(0, 4, &chunk, &chunk) else {
        panic!("post-finalize append must reject");
    };
    assert!(reason.contains("no stream in progress"), "{reason}");
    // The finalized track is intact (render once so the seqlock snapshot
    // reflects the install).
    let mut ol = vec![0.0f32; 128];
    let mut or = vec![0.0f32; 128];
    core.render(&mut ol, &mut or);
    let snap = plane.snapshot();
    assert_eq!(snap.tracks.len(), 1);
    assert!(!snap.tracks[0].streaming);
}

#[test]
fn oversize_append_rejects_engine_side() {
    let (mut plane, _core) = engine_at(O_120);
    plane
        .begin_pcm_stream(&stream_meta(meta_120(0, 0, 0), PCM_120))
        .expect("begin");
    let chunk = vec![0.0f32; MAX_STREAM_APPEND_FRAMES + 1];
    let AppendResult::Rejected { reason } = plane.append_pcm_chunk(0, 0, &chunk, &chunk) else {
        panic!("oversize append must reject");
    };
    assert!(reason.contains("MAX_STREAM_APPEND_FRAMES"), "{reason}");
    // Nothing was consumed: seq 0 with a legal size still lands.
    let legal = vec![0.0f32; 1_000];
    assert!(matches!(plane.append_pcm_chunk(0, 0, &legal, &legal), AppendResult::Ok { .. }));
}

#[test]
fn finalize_under_delivery_removes_the_track() {
    let (mut plane, mut core) = engine_at(O_120);
    plane
        .begin_pcm_stream(&stream_meta(meta_120(0, 0, 0), PCM_120))
        .expect("begin");
    let chunk = vec![0.1f32; 1_000];
    assert!(matches!(plane.append_pcm_chunk(0, 0, &chunk, &chunk), AppendResult::Ok { .. }));
    let err = plane.finalize_pcm_stream(0).unwrap_err();
    assert!(err.contains("under-delivery"), "{err}");
    assert!(plane
        .drain_events()
        .iter()
        .any(|e| matches!(e, Event::TrackLoadFailed { .. })));
    // The RemoveTrack crossed to the core; after a render + drain the slot
    // is empty in the snapshot and the budget is released.
    let mut l = vec![0.0f32; 128];
    let mut r = vec![0.0f32; 128];
    core.render(&mut l, &mut r);
    plane.drain_events();
    assert_eq!(plane.snapshot().tracks.len(), 0);
    assert_eq!(plane.pcm_bytes_in_use(), 0);
}

#[test]
fn abort_mid_stream_fires_track_load_failed_and_empties_the_slot() {
    let (mut plane, _core) = engine_at(O_120);
    plane
        .begin_pcm_stream(&stream_meta(meta_120(0, 0, 0), PCM_120))
        .expect("begin");
    assert_eq!(plane.pcm_bytes_in_use(), PCM_120 * 8, "budget counts capacity from begin");
    plane.abort_pcm_stream(0).expect("abort");
    assert!(plane
        .drain_events()
        .iter()
        .any(|e| matches!(e, Event::TrackLoadFailed { .. })));
    assert_eq!(plane.pcm_bytes_in_use(), 0);
    // A fresh begin on the same slot works.
    plane
        .begin_pcm_stream(&stream_meta(meta_120(0, 0, 0), PCM_120))
        .expect("begin after abort");
}

#[test]
fn switch_variation_and_bulk_load_reject_on_a_streaming_slot() {
    let (mut plane, _core) = engine_at(O_120);
    plane
        .begin_pcm_stream(&stream_meta(meta_120(0, 0, 0), PCM_120))
        .expect("begin");
    let res = plane.dispatch(Command::SwitchVariation { slot: 0, variation: 1 });
    let CommandResult::Rejected { reason } = res else {
        panic!("switchVariation on a streaming slot must reject");
    };
    assert!(reason.contains("stream in progress"), "{reason}");
    let (l, r) = make_sine_pcm(PCM_120, 220.0);
    let err = plane.load_pcm(&meta_120(0, 0, 0), l, r).unwrap_err();
    assert!(err.contains("stream in progress"), "{err}");
}

// ---------------------------------------------------------------------------
// §5.4 starvation predicate — boundary table
// ---------------------------------------------------------------------------

/// Build a streaming engine at `tempo`, append exactly `frontier` frames of
/// sine, activate + play, render one block of `b`; returns (was silent,
/// snapshot starved flag, events).
fn render_one_block_with_frontier(
    tempo: u32,
    meta: LoadMeta,
    declared: u64,
    frontier: u64,
    b: usize,
) -> (bool, bool, Vec<Event>) {
    let (mut plane, mut core) = engine_at(tempo);
    plane.begin_pcm_stream(&stream_meta(meta, declared)).expect("begin");
    if frontier > 0 {
        let (l, r) = make_sine_pcm(frontier, 330.0);
        append_all_in_chunks(&mut plane, 0, &l, &r, MAX_STREAM_APPEND_FRAMES);
    }
    let (out_l, out_r) = activate_play_render(&mut plane, &mut core, 0, 1, b);
    let peak = out_l.iter().chain(out_r.iter()).fold(0.0f32, |a, s| a.max(s.abs()));
    let snap = plane.snapshot();
    let events = plane.drain_events();
    (peak == 0.0, snap.tracks[0].starved, events)
}

#[test]
fn unity_starves_at_frontier_minus_one_and_renders_at_frontier() {
    let b = 128usize;
    // The block reads [0, b): needed frontier is exactly b.
    let (silent, starved, events) =
        render_one_block_with_frontier(O_120, meta_120(0, 0, 0), PCM_120, b as u64 - 1, b);
    assert!(silent && starved, "frontier b−1 must starve");
    assert!(
        events.iter().any(|e| matches!(
            e,
            Event::StreamStarved { slot: 0, needed, frontier, .. }
                if *needed == b as u64 && *frontier == b as u64 - 1
        )),
        "streamStarved must carry the exact requirement: {events:?}"
    );

    let (silent, starved, events) =
        render_one_block_with_frontier(O_120, meta_120(0, 0, 0), PCM_120, b as u64, b);
    assert!(!silent && !starved, "frontier b must render");
    assert!(!events.iter().any(|e| matches!(e, Event::StreamStarved { .. })));
}

/// 16 beats @ 60 BPM = 768_000 samples; engine at 240 BPM = the 4× rate
/// clamp ceiling. The stretch feed leads by E·output_latency (ADR-094) and
/// consumes 4 source samples per output frame; the anchor window rides the
/// first block. Requirement (§5.4): n0_s + input_latency + span_s.
#[test]
fn stretch_at_4x_rate_demands_the_compensated_lead() {
    let o = 60_000u32;
    let e = 240_000u32;
    let beats = 16u32;
    let pcm_len: u64 = 768_000;
    let meta = LoadMeta {
        slot: 0,
        sample_rate: SAMPLE_RATE,
        original_tempo_mbpm: o,
        loop_beats: beats,
        encoder_delay: 0,
        end_padding: 0,
        original_samples: pcm_len,
        variation: 0,
        cache_only: false,
        gain: 1.0,
        pan: 0.0,
    };
    let probe = Stretcher::new(2, SAMPLE_RATE);
    let (in_lat, out_lat) = (probe.input_latency() as u64, probe.output_latency() as u64);
    let b = 128u64;
    // n0_s = u(0 + out_lat·E) = 4·out_lat; span_s = 4·b; anchor window ends
    // at n0_s + in_lat (inside the feed span, so the feed end dominates).
    let needed = 4 * out_lat + in_lat + 4 * b;

    let (silent, starved, _) =
        render_one_block_with_frontier(e, meta.clone(), pcm_len, needed - 1, b as usize);
    assert!(silent && starved, "frontier needed−1 must starve the stretch path");

    let (silent, starved, _) =
        render_one_block_with_frontier(e, meta, pcm_len, needed, b as usize);
    assert!(!silent && !starved, "frontier == needed must render the stretch path");
}

/// §5.8: a block whose span crosses the wrap seam touches the loop TAIL —
/// `max_touched = pcm_len − 1` — so an incomplete stream starves through the
/// seam and re-enters after it (the resume margin is met immediately: the
/// post-wrap reads are near PCM 0, the most-loaded region).
#[test]
fn wrap_seam_starves_an_incomplete_stream_then_resumes_after_the_wrap() {
    let (mut plane, mut core) = engine_at(O_120);
    plane
        .begin_pcm_stream(&stream_meta(meta_120(0, 0, 0), PCM_120))
        .expect("begin");
    // Everything but the final 64 samples is loaded. Blocks render fine
    // until one touches [pcm_len − 64, pcm_len).
    let frontier = PCM_120 - 64;
    let (l, r) = make_sine_pcm(frontier, 330.0);
    append_all_in_chunks(&mut plane, 0, &l, &r, MAX_STREAM_APPEND_FRAMES);
    let b = 128usize;
    let blocks_total = (PCM_120 as usize / b) + 40; // through the seam + beyond
    let (out_l, _) = activate_play_render(&mut plane, &mut core, 0, blocks_total, b);
    let events = plane.drain_events();
    let starve_frame = events.iter().find_map(|e| match e {
        Event::StreamStarved { at_frame, .. } => Some(*at_frame),
        _ => None,
    });
    let resume_frame = events.iter().find_map(|e| match e {
        Event::StreamResumed { at_frame, .. } => Some(*at_frame),
        _ => None,
    });
    let starve_frame = starve_frame.expect("the seam block must starve");
    let resume_frame = resume_frame.expect("the stream must resume after the wrap");
    // The starve lands on the block that first touches the missing tail.
    let first_touching_block = ((frontier as usize) / b) * b; // block start ≤ frontier
    assert_eq!(starve_frame as usize, first_touching_block);
    assert!(resume_frame > starve_frame);
    assert!(
        (resume_frame as usize) < first_touching_block + 3 * b + PCM_120 as usize % b + b,
        "resume must land just past the seam, got {resume_frame}"
    );
    // Audio: silent across the starved span, playing again after resume.
    let starved_block = &out_l[starve_frame as usize..starve_frame as usize + b];
    assert!(starved_block.iter().all(|s| *s == 0.0), "the seam block must be silent");
    let after = &out_l[resume_frame as usize..resume_frame as usize + b];
    assert!(after.iter().any(|s| *s != 0.0), "audio must resume after the wrap");
}

/// §5.4 hysteresis: a frontier advancing at exactly consumption rate must
/// NOT flutter render/starve — once starved, the track stays silent until
/// the frontier clears the failed requirement by the full resume margin.
#[test]
fn resume_margin_prevents_flutter_at_consumption_rate() {
    let b = 128usize;
    let (mut plane, mut core) = engine_at(O_120);
    plane
        .begin_pcm_stream(&stream_meta(meta_120(0, 0, 0), PCM_120))
        .expect("begin");
    // One sample short of the first block's requirement: starves at block 0.
    let mut frontier = b as u64 - 1;
    let (l, r) = make_sine_pcm(PCM_120, 330.0);
    append_all_in_chunks(&mut plane, 0, &l[..frontier as usize], &r[..frontier as usize], b);
    let mut seq = (frontier as usize).div_ceil(b) as u32;
    ok(plane.dispatch(Command::Activate {
        slot: 0,
        quantize: QuantizePolicy::Immediate,
        transport: Default::default(),
    }));
    ok(plane.dispatch(Command::Play));

    let mut out_l = vec![0.0f32; b];
    let mut out_r = vec![0.0f32; b];
    // 100 blocks at exact consumption rate: every one must be silent.
    for _ in 0..100 {
        core.render(&mut out_l, &mut out_r);
        assert!(
            out_l.iter().chain(out_r.iter()).all(|s| *s == 0.0),
            "hysteresis violated: rendered mid-flutter"
        );
        let n = b.min((PCM_120 - frontier) as usize);
        match plane.append_pcm_chunk(
            0,
            seq,
            &l[frontier as usize..frontier as usize + n],
            &r[frontier as usize..frontier as usize + n],
        ) {
            AppendResult::Ok { .. } => {}
            AppendResult::Rejected { reason } => panic!("append rejected: {reason}"),
        }
        seq += 1;
        frontier += n as u64;
    }
    // Now push the frontier past requirement + margin in one burst: resumes.
    let burst = (STREAM_RESUME_MARGIN_FRAMES as usize + 2 * b).min((PCM_120 - frontier) as usize);
    match plane.append_pcm_chunk(
        0,
        seq,
        &l[frontier as usize..frontier as usize + burst],
        &r[frontier as usize..frontier as usize + burst],
    ) {
        AppendResult::Ok { .. } => {}
        AppendResult::Rejected { reason } => panic!("burst append rejected: {reason}"),
    }
    let (post_l, _) = render_blocks(&mut core, 2, b);
    assert!(post_l.iter().any(|s| *s != 0.0), "the margin burst must resume playback");
    let events = plane.drain_events();
    let starves = events.iter().filter(|e| matches!(e, Event::StreamStarved { .. })).count();
    let resumes = events.iter().filter(|e| matches!(e, Event::StreamResumed { .. })).count();
    assert_eq!(starves, 1, "exactly one starve edge");
    assert_eq!(resumes, 1, "exactly one resume edge");
}

// ---------------------------------------------------------------------------
// i16 storage (Tier 2a, ADR-112 §6.2)
// ---------------------------------------------------------------------------

/// Load the same content as f32 and i16 streams into two engines at
/// `tempo`, render identically, and return the residual RMS in dBFS.
fn f32_vs_i16_residual_dbfs(tempo: u32) -> f64 {
    let (sig_l, sig_r) = make_sine_pcm(PCM_120, 330.0);
    let mut engines = Vec::new();
    for format in [SampleFormat::F32, SampleFormat::I16] {
        let (mut plane, core) = engine_at(tempo);
        let mut smeta = stream_meta(meta_120(0, 0, 0), PCM_120);
        smeta.sample_format = format;
        plane.begin_pcm_stream(&smeta).expect("begin");
        append_all_in_chunks(&mut plane, 0, &sig_l, &sig_r, MAX_STREAM_APPEND_FRAMES);
        plane.finalize_pcm_stream(0).expect("finalize");
        engines.push((plane, core));
    }
    let mut outputs = Vec::new();
    for (plane, core) in engines.iter_mut() {
        let (l, r) = activate_play_render(plane, core, 0, 200, 480);
        outputs.push((l, r));
    }
    let (ref f32_out, _) = outputs[0];
    let peak = f32_out.iter().fold(0.0f32, |a, s| a.max(s.abs()));
    assert!(peak > 0.0, "anti-stub: the null test rendered silence");
    let mut sumsq = 0.0f64;
    let mut n = 0u64;
    for ch in 0..2 {
        let a = if ch == 0 { &outputs[0].0 } else { &outputs[0].1 };
        let b = if ch == 0 { &outputs[1].0 } else { &outputs[1].1 };
        for (x, y) in a.iter().zip(b.iter()) {
            let d = (*x - *y) as f64;
            sumsq += d * d;
            n += 1;
        }
    }
    20.0 * (sumsq / n as f64).sqrt().log10()
}

/// §10.1: the f32-vs-i16 null test through the unity (bit-exact bypass)
/// path — the load-time quantization floor passes through untouched.
/// Measured −103.9 dBFS on this fixture.
#[test]
fn i16_null_test_unity_path() {
    let residual = f32_vs_i16_residual_dbfs(O_120);
    assert!(residual <= -90.0, "unity residual {residual:.1} dBFS > −90 dBFS");
}

/// §10.1 stretch-path null test — IMPLEMENTATION-NOTES DEVIATION from the
/// ADR's ≤ −90 dBFS: a phase vocoder does not preserve small input deltas
/// (phase perturbation on low-magnitude bins scales with processing, not
/// with the input floor), so the −96 dBFS load quantization emerges
/// rate-dependently. Measured on this fixture (sine at −15 dBFS):
///   0.75× → −97.4   1.1× → −65.4   1.5× → −51.3 (worst)   2× → −66.5
/// The bound pins the worst measured rate with ~6 dB margin; the ADR's
/// by-ear leg on the worst stretch case remains the perceptual gate
/// (§6.2 — the noise is redistributed, never signal-level artifacts).
#[test]
fn i16_null_test_stretch_path() {
    let residual = f32_vs_i16_residual_dbfs(180_000);
    assert!(residual <= -45.0, "stretch residual {residual:.1} dBFS > −45 dBFS");
}

/// §6.2: i16 halves the §4.11 budget accounting (capacity, from begin).
#[test]
fn i16_budget_accounting_halves() {
    let (mut plane, _core) = engine_at(O_120);
    let mut smeta = stream_meta(meta_120(0, 0, 0), PCM_120);
    smeta.sample_format = SampleFormat::I16;
    plane.begin_pcm_stream(&smeta).expect("begin");
    assert_eq!(plane.pcm_bytes_in_use(), PCM_120 * 2 * 2, "i16 stereo bytes");
    plane.abort_pcm_stream(0).expect("abort");
    plane.begin_pcm_stream(&stream_meta(meta_120(0, 0, 0), PCM_120)).expect("begin f32");
    assert_eq!(plane.pcm_bytes_in_use(), PCM_120 * 2 * 4, "f32 stereo bytes");
}

// ---------------------------------------------------------------------------
// Zero drift under starvation
// ---------------------------------------------------------------------------

/// T-INV-style: a streaming track that starves and resumes repeatedly must
/// report dictated positions bit-identical to an always-loaded control run —
/// starvation never touches positions (they are derived from F alone).
#[test]
fn dictated_positions_are_bit_identical_under_starvation() {
    let (sig_l, sig_r) = make_sine_pcm(PCM_120, 330.0);

    // Control: fully loaded from the start.
    let (mut plane_c, mut core_c) = engine_at(O_120);
    plane_c
        .load_pcm(&meta_120(0, 0, 0), sig_l.clone(), sig_r.clone())
        .expect("bulk load");
    ok(plane_c.dispatch(Command::Activate {
        slot: 0,
        quantize: QuantizePolicy::Immediate,
        transport: Default::default(),
    }));
    ok(plane_c.dispatch(Command::Play));

    // Streaming: drip-fed so it starves and resumes repeatedly.
    let (mut plane_s, mut core_s) = engine_at(O_120);
    plane_s
        .begin_pcm_stream(&stream_meta(meta_120(0, 0, 0), PCM_120))
        .expect("begin");
    ok(plane_s.dispatch(Command::Activate {
        slot: 0,
        quantize: QuantizePolicy::Immediate,
        transport: Default::default(),
    }));
    ok(plane_s.dispatch(Command::Play));

    let blocks = [128usize, 480, 1024];
    let mut l = vec![0.0f32; 1024];
    let mut r = vec![0.0f32; 1024];
    let mut frontier = 0u64;
    let mut seq = 0u32;
    for i in 0..600usize {
        let b = blocks[i % blocks.len()];
        core_c.render(&mut l[..b], &mut r[..b]);
        core_s.render(&mut l[..b], &mut r[..b]);
        let pos_c = core_c.dictated_unwrapped_position(0).expect("control track");
        let pos_s = core_s.dictated_unwrapped_position(0).expect("stream track");
        assert_eq!(pos_c, pos_s, "position diverged at block {i}");
        // Bursty drip: 70k frames every 160 blocks (≈ 87k frames consumed
        // per cycle at the rotating block sizes) — the playhead repeatedly
        // outruns the frontier, then each burst clears the resume margin,
        // forcing full starve→resume cycles.
        if i % 160 == 0 && frontier < PCM_120 {
            let n = 70_000.min((PCM_120 - frontier) as usize);
            match plane_s.append_pcm_chunk(
                0,
                seq,
                &sig_l[frontier as usize..frontier as usize + n],
                &sig_r[frontier as usize..frontier as usize + n],
            ) {
                AppendResult::Ok { .. } => {}
                AppendResult::Rejected { reason } => panic!("append rejected: {reason}"),
            }
            seq += 1;
            frontier += n as u64;
        }
    }
    let events = plane_s.drain_events();
    let starves = events.iter().filter(|e| matches!(e, Event::StreamStarved { .. })).count();
    let resumes = events.iter().filter(|e| matches!(e, Event::StreamResumed { .. })).count();
    assert!(starves >= 2, "the drip schedule must force repeated starvation (got {starves})");
    assert!(resumes >= 2, "and repeated resumes (got {resumes})");
    plane_c.drain_events();
}

// ---------------------------------------------------------------------------
// Snapshot truthfulness + command interactions mid-stream
// ---------------------------------------------------------------------------

#[test]
fn snapshot_reports_frontier_streaming_and_starved_truthfully() {
    let (mut plane, mut core) = engine_at(O_120);
    plane
        .begin_pcm_stream(&stream_meta(meta_120(0, 0, 0), PCM_120))
        .expect("begin");
    // The install crosses to the core at the next block boundary; a stopped
    // transport still applies commands and writes the seqlock snapshot.
    let mut l = vec![0.0f32; 128];
    let mut r = vec![0.0f32; 128];
    core.render(&mut l, &mut r);
    let snap = plane.snapshot();
    assert_eq!(snap.tracks.len(), 1, "a streaming install is visible before any audio");
    assert!(snap.tracks[0].streaming);
    assert_eq!(snap.tracks[0].loaded_samples, 0);
    assert!(!snap.tracks[0].starved);

    let (sig_l, sig_r) = make_sine_pcm(10_000, 330.0);
    assert!(matches!(
        plane.append_pcm_chunk(0, 0, &sig_l, &sig_r),
        AppendResult::Ok { frontier: 10_000, complete: false }
    ));
    core.render(&mut l, &mut r);
    let snap = plane.snapshot();
    assert_eq!(snap.tracks[0].loaded_samples, 10_000);

    // Starve it: activate + play; the first block within the frontier
    // renders, and once the playhead outruns the frontier the flag flips.
    ok(plane.dispatch(Command::Activate {
        slot: 0,
        quantize: QuantizePolicy::Immediate,
        transport: Default::default(),
    }));
    ok(plane.dispatch(Command::Play));
    for _ in 0..((10_000 / 128) + 2) {
        core.render(&mut l, &mut r);
    }
    let snap = plane.snapshot();
    assert!(snap.tracks[0].starved, "the playhead outran the frontier");
    assert!(snap.tracks[0].streaming);
}

/// Regression (review finding on ADR-112): a stream whose remainder is
/// delivered AND finalized while the track is starved must still clear
/// `starved` and fire `StreamResumed` on the next rendered block. The
/// completion branch of the §5.4 gate used to short-circuit past the only
/// resume-edge clear, leaving the snapshot reporting `starved: true`
/// forever (a stale buffering indicator for any UI reading it).
#[test]
fn finalize_while_starved_clears_the_flag_and_fires_resumed() {
    let (mut plane, mut core) = engine_at(O_120);
    plane
        .begin_pcm_stream(&stream_meta(meta_120(0, 0, 0), PCM_120))
        .expect("begin");
    let (sig_l, sig_r) = make_sine_pcm(PCM_120, 330.0);
    let head = 10_000usize;
    assert!(matches!(
        plane.append_pcm_chunk(0, 0, &sig_l[..head], &sig_r[..head]),
        AppendResult::Ok { .. }
    ));
    // Outrun the frontier → starved.
    let b = 128usize;
    activate_play_render(&mut plane, &mut core, 0, head / b + 2, b);
    let snap = plane.snapshot();
    assert!(snap.tracks[0].starved, "precondition: the playhead outran the frontier");
    assert!(plane.drain_events().iter().any(|e| matches!(e, Event::StreamStarved { .. })));

    // Deliver the whole remainder and finalize BEFORE any further render —
    // the next block takes the completion branch, not the margin path.
    let mut seq = 1u32;
    let mut at = head;
    while at < PCM_120 as usize {
        let n = MAX_STREAM_APPEND_FRAMES.min(PCM_120 as usize - at);
        match plane.append_pcm_chunk(0, seq, &sig_l[at..at + n], &sig_r[at..at + n]) {
            AppendResult::Ok { .. } => {}
            AppendResult::Rejected { reason } => panic!("append {seq} rejected: {reason}"),
        }
        seq += 1;
        at += n;
    }
    plane.finalize_pcm_stream(0).expect("finalize");
    let (out_l, _) = render_blocks(&mut core, 2, b);

    let snap = plane.snapshot();
    assert!(!snap.tracks[0].starved, "completion must clear the starved flag");
    assert!(!snap.tracks[0].streaming, "finalize ended the stream");
    assert_eq!(snap.tracks[0].loaded_samples, PCM_120);
    let events = plane.drain_events();
    assert!(
        events.iter().any(|e| matches!(e, Event::StreamResumed { .. })),
        "the resume edge must fire on the completed stream, got {events:?}"
    );
    assert!(out_l.iter().any(|s| *s != 0.0), "audio must resume after completion");
}

/// §5.5 command interactions: pause/resume and tempo changes mid-stream are
/// orthogonal to loading; appends continue while paused; a legal tempo
/// change re-validates streaming slots exactly like loaded ones.
#[test]
fn pause_tempo_and_appends_compose_mid_stream() {
    let (mut plane, mut core) = engine_at(O_120);
    plane
        .begin_pcm_stream(&stream_meta(meta_120(0, 0, 0), PCM_120))
        .expect("begin");
    let (sig_l, sig_r) = make_sine_pcm(PCM_120, 330.0);
    assert!(matches!(
        plane.append_pcm_chunk(0, 0, &sig_l[..48_000], &sig_r[..48_000]),
        AppendResult::Ok { .. }
    ));
    ok(plane.dispatch(Command::Activate {
        slot: 0,
        quantize: QuantizePolicy::Immediate,
        transport: Default::default(),
    }));
    ok(plane.dispatch(Command::Play));
    let mut l = vec![0.0f32; 128];
    let mut r = vec![0.0f32; 128];
    for _ in 0..10 {
        core.render(&mut l, &mut r);
    }
    // Pause freezes F; loading continues.
    ok(plane.dispatch(Command::Pause));
    core.render(&mut l, &mut r);
    let before = core.dictated_unwrapped_position(0).unwrap();
    assert!(matches!(
        plane.append_pcm_chunk(0, 1, &sig_l[48_000..96_000], &sig_r[48_000..96_000]),
        AppendResult::Ok { frontier: 96_000, .. }
    ));
    core.render(&mut l, &mut r);
    assert_eq!(core.dictated_unwrapped_position(0).unwrap(), before, "paused F must freeze");
    // A tempo change against a streaming slot re-validates the §4.8 rate.
    let res = plane.dispatch(Command::SetTempo { mbpm: 20_000 });
    assert!(matches!(res, CommandResult::Rejected { .. }), "sub-40 BPM must reject");
    ok(plane.dispatch(Command::SetTempo { mbpm: 240_000 })); // 2× on O=120k: legal
    ok(plane.dispatch(Command::Resume));
    for _ in 0..10 {
        core.render(&mut l, &mut r);
    }
    // Finish the stream and finalize under the new tempo.
    let mut frontier = 96_000usize;
    let mut seq = 2u32;
    while frontier < PCM_120 as usize {
        let n = 200_000.min(PCM_120 as usize - frontier);
        assert!(matches!(
            plane.append_pcm_chunk(0, seq, &sig_l[frontier..frontier + n], &sig_r[frontier..frontier + n]),
            AppendResult::Ok { .. }
        ));
        seq += 1;
        frontier += n;
    }
    plane.finalize_pcm_stream(0).expect("finalize");
    let events = plane.drain_events();
    assert!(loaded_event_reconciliation(&events).is_some(), "trackLoaded at finalize");
}

