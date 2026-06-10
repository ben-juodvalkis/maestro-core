//! # render_pump — Plan B delivery (spec §3.3)
//!
//! The pre-decided fallback for when direct in-callback rendering cannot meet
//! the render-budget gate: the §3.2 render loop runs on a **dedicated
//! high-priority producer thread** feeding an SPSC sample ring; the OS audio
//! callback only pops. Invoked for M0 after C0 measured Signalsmith
//! preset_default hop-bursts at ~61% of a 5 ms callback period with 4 tracks
//! at 1.5× (p99 gate ≤ 50%), and preset_cheaper was rejected for audible
//! transient collapse at rate 2.0 (see ADR-090 and
//! documentation/validation-runs/2026-06-10-c0-partial/).
//!
//! **Sync is unaffected by construction.** The master clock remains `F` as
//! counted by frames *rendered* (by the producer, into the ring); positions
//! derive from `F` regardless of who advances it. The producer is paced by
//! **ring occupancy, never by a timer** — when the ring is full it parks
//! briefly; it can never inject, skip, or time-warp a frame. Only latency
//! changes: output lags by the ring occupancy (≈ 85 ms at the default sizes),
//! which quantized starts make musically irrelevant (spec §4.10).
//!
//! A callback finding the ring short fills the remainder with silence and
//! counts an underrun (atomic, surfaced via the snapshot — the C0 gate).
//! The clock does NOT advance for unrendered frames, so starvation is a
//! glitch, never drift.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use crate::audio_core::AudioCore;

/// Producer render chunk (frames). Larger than typical OS callbacks so the
/// Signalsmith hop-burst cost amortizes; small enough to keep command-to-ear
/// latency bounded (1024 frames ≈ 21.3 ms render cadence, within the §4.10
/// accepted control-feel envelope).
///
/// Raised 512 → 1024 for checkpoint C1 (ADR-091): once a variation switch
/// re-anchors one track, its hop bursts de-phase from the tracks anchored at
/// Play, spreading bursts across MORE chunks — at 512 frames the
/// blocks-over-half-period budget proxy measured ~1.2% (gate ≤ 1%) with
/// max and underruns unchanged (53%, 0). Doubling the chunk averages ~2× the
/// hops per measurement window, pulling the proxy back under the gate
/// without touching any threshold. Sync is unaffected by construction
/// (positions derive from F; T-PUMP asserts bit-identity).
pub const PUMP_CHUNK_FRAMES: usize = 1024;
/// Ring capacity (frames, stereo interleaved underneath). Worst-case added
/// output latency = ring occupancy ≈ 85 ms at 48 kHz.
pub const PUMP_RING_FRAMES: usize = 4096;

/// Consumer half handed to the audio callback. Heap-pinned (`Box`) so its
/// address is stable for the C ABI; the owning [`RenderPump`] must outlive
/// every callback invocation (same contract as `maestro_render`).
pub struct PumpConsumer {
    ring: rtrb::Consumer<f32>,
    /// Ring-starvation events (callback got fewer frames than it needed).
    underruns: Arc<AtomicU32>,
}

/// The Plan B engine driver: owns the producer thread and the consumer half.
pub struct RenderPump {
    consumer: Box<PumpConsumer>,
    stop: Arc<AtomicBool>,
    join: Option<JoinHandle<()>>,
}

impl RenderPump {
    /// Move the audio core onto a dedicated producer thread. The core's
    /// snapshot `underruns` field is wired to the ring-starvation counter
    /// before the move (the C0 underrun gate reads the snapshot).
    pub fn start(mut core: Box<AudioCore>) -> Self {
        let (producer, ring_consumer) = rtrb::RingBuffer::<f32>::new(PUMP_RING_FRAMES * 2);
        let underruns = Arc::new(AtomicU32::new(0));
        core.set_external_underruns(Arc::clone(&underruns));
        let stop = Arc::new(AtomicBool::new(false));
        let join = std::thread::Builder::new()
            .name("maestro-render-pump".into())
            .spawn({
                let stop = Arc::clone(&stop);
                move || producer_loop(core, producer, stop)
            })
            .expect("spawn render pump thread");
        Self {
            consumer: Box::new(PumpConsumer { ring: ring_consumer, underruns }),
            stop,
            join: Some(join),
        }
    }

    /// Stable pointer for the audio callback (`maestro_pump_consume`).
    pub fn consumer_ptr(&mut self) -> *mut PumpConsumer {
        self.consumer.as_mut() as *mut PumpConsumer
    }

    pub fn underruns(&self) -> u32 {
        self.consumer.underruns.load(Ordering::Relaxed)
    }

    /// Whole frames currently buffered (test/diagnostic use; the audio
    /// callback never needs this — it pops and the shortfall policy applies).
    pub fn available_frames(&self) -> usize {
        self.consumer.ring.slots() / 2
    }
}

impl Drop for RenderPump {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

/// The producer loop: render a chunk whenever the ring has room, park briefly
/// when it doesn't. Occupancy-paced — the clock is advanced exclusively by
/// `AudioCore::render` for frames actually produced.
fn producer_loop(mut core: Box<AudioCore>, mut ring: rtrb::Producer<f32>, stop: Arc<AtomicBool>) {
    promote_thread_priority();

    let mut left = vec![0.0f32; PUMP_CHUNK_FRAMES];
    let mut right = vec![0.0f32; PUMP_CHUNK_FRAMES];
    let mut interleaved = vec![0.0f32; PUMP_CHUNK_FRAMES * 2];

    while !stop.load(Ordering::Relaxed) {
        if ring.slots() >= PUMP_CHUNK_FRAMES * 2 {
            core.render(&mut left, &mut right);
            for i in 0..PUMP_CHUNK_FRAMES {
                interleaved[i * 2] = left[i];
                interleaved[i * 2 + 1] = right[i];
            }
            let mut written = 0;
            while written < interleaved.len() {
                match ring.write_chunk_uninit(interleaved.len() - written) {
                    Ok(chunk) => {
                        let n = chunk.fill_from_iter(interleaved[written..].iter().copied());
                        written += n;
                    }
                    Err(_) => break, // can't happen: slots checked above
                }
            }
        } else {
            // Ring full: wait for the callback to drain. Pacing only — the
            // timeline cannot advance here.
            std::thread::park_timeout(Duration::from_millis(2));
        }
    }
}

/// Best-effort priority promotion for the producer thread. USER_INTERACTIVE
/// QoS keeps the pump ahead of UI work on Apple platforms; a hard realtime
/// time-constraint policy is an M5 hardening option if soak tests want it.
fn promote_thread_priority() {
    #[cfg(any(target_os = "ios", target_os = "macos"))]
    unsafe {
        // QOS_CLASS_USER_INTERACTIVE = 0x21
        pthread_set_qos_class_self_np(0x21, 0);
    }
}

#[cfg(any(target_os = "ios", target_os = "macos"))]
extern "C" {
    fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
}

/// The Plan B audio-callback entry (C ABI): pop `frames` stereo frames into
/// non-interleaved buffers. On ring starvation the remainder is silence and
/// one underrun is counted — the clock does not advance for unrendered
/// frames, so this can glitch but never desync.
///
/// # Safety
///
/// `consumer` must be the live pointer from [`RenderPump::consumer_ptr`],
/// called from exactly one thread (the audio thread); `left`/`right` must be
/// valid for `frames` writes each. RT-safe: no locks, no allocation.
#[no_mangle]
pub unsafe extern "C" fn maestro_pump_consume(
    consumer: *mut PumpConsumer,
    left: *mut f32,
    right: *mut f32,
    frames: u32,
) {
    if consumer.is_null() || left.is_null() || right.is_null() {
        return;
    }
    let c = &mut *consumer;
    let l = std::slice::from_raw_parts_mut(left, frames as usize);
    let r = std::slice::from_raw_parts_mut(right, frames as usize);

    let want = frames as usize * 2;
    let have = c.ring.slots().min(want) & !1; // whole frames only
    let got_frames = have / 2;
    if let Ok(chunk) = c.ring.read_chunk(have) {
        let (a, b) = chunk.as_slices();
        for (i, s) in a.iter().chain(b.iter()).enumerate() {
            if i % 2 == 0 {
                l[i / 2] = *s;
            } else {
                r[i / 2] = *s;
            }
        }
        chunk.commit_all();
    }
    if got_frames < frames as usize {
        l[got_frames..].fill(0.0);
        r[got_frames..].fill(0.0);
        c.underruns.fetch_add(1, Ordering::Relaxed);
    }
}
