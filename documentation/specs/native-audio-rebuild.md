# Native Audio Rebuild — Implementation Specification

**Status:** Authoritative. Supersedes ADRs 083–089 (which document the abandoned
`feature/milestone-5-render-stretch` approach and are retained as history only).
**Baseline:** `main`. The abandoned branch is a parts/evidence source, not a starting point.
**Author's mandate:** rebuild the native audio engine and its integration so that
sample-perfect multi-track sync is impossible to violate by construction, prove it by test,
and converge native and web onto one Rust core.

---

## 1. Mission and the Zero-Drift Law

> **There can be no drift. Ever. Zero deviation from sample-perfect multi-track sync.**

### 1.1 Precise definition

Let `F` be the index of an output frame produced by the engine (the engine's master clock,
defined in §3.1). For every active track `i`, the engine defines a **dictated source sample**
`u_i(F)` — which sample of track `i`'s loop must be sounding at output frame `F`. The law:

1. **Determinism.** `u_i(F)` is a pure function of `F` and the command history (tempo
   segments, loop metadata). It contains no mutable per-track state and no floating-point
   arithmetic. Two replays of the same command history produce bit-identical `u_i(F)`
   sequences on any architecture.
2. **Sample lock.** For any two active tracks `i, j` and any frame `F`, the musical phase
   difference between `u_i(F)` and `u_j(F)` is bounded by strictly less than one source
   sample, and this bound holds for all `F` — across loop wraps, tempo changes, pitch
   changes, arming/disarming, variation switches, pause/resume, interruptions, and
   arbitrarily long playback. There is no term anywhere in the system that grows with time,
   block count, or wrap count.
3. **Audio fidelity to the dictate.** The rendered audio for track `i` at frame `F` is the
   stretcher/reader output for exactly the source span the timeline dictated. The DSP can
   smear a waveform inside its constant analysis window (~120 ms, identical for all tracks
   and all time); it can never reposition a track, because it is never asked where it is —
   it is told what to consume (§3.2).

Measurable corollary (this is what the device harness asserts, §6.4): the click onsets of
two metronome stems in the captured output differ by a **constant** offset with **zero
slope** over the entire capture.

### 1.2 Why drift is impossible by construction (summary; full model in §3.1)

- There is exactly **one mutable timing variable** in the engine: the master frame counter
  `F: u64`, advanced by the audio callback by exactly the block size, only while playing.
- Tempo state is an **append-only segment table** of integers. Musical position is an exact
  integer function `bu(F)` (beat-units) of `F` and that table.
- Every track position is **derived**: `u_i(F) = floor(bu(F) / O_i)` with integer `O_i`
  (milli-BPM). Loops wrap in **beat space** (`bu mod Λ_i`, both exact integers), never at an
  integer PCM length, so non-integer musical loop lengths (the real-world case — see §2.2)
  cannot accumulate per-wrap error.
- Per-block stretcher feed counts are first differences `u_i(F+B) − u_i(F)`, which
  **telescope**: the sum of all feeds after any number of blocks equals `u_i(F)` exactly.
  No remainder is carried in mutable state; there is nothing to accumulate.
- Floats appear only in sample values (gain, pan, DSP). No float ever feeds a position,
  a wrap, a feed count, or a clock.

A per-track position that is *advanced* each block (by any increment, float or int) is
**forbidden in this codebase**. Positions are *computed*, never *advanced*. CI enforces the
test in §6.2 whose negative controls fail if an accumulating cursor is ever reintroduced.

---

## 2. Failure analysis — what actually broke, with evidence

This section re-grounds the commissioning dossier's findings. Items marked **[REPRO]** were
reproduced by running the abandoned branch engine offline (method and numbers in Appendix A);
**[VERIFIED]** were confirmed by direct code reading; **[CORRECTED]** revise the dossier.

### 2.1 Confirmed defects of the abandoned branch

| # | Defect | Evidence |
|---|--------|----------|
| D1 | Per-track free-running cursors: `playback_position` + `cursor_fractions` advanced per block, never re-anchored | [VERIFIED] branch `engine.rs:1019-1026`, `track.rs:378-383` |
| D2 | Loops wrap at integer PCM length while exact musical lengths are non-integer → unbounded inter-stem drift even at rate 1.0 | **[REPRO]** 24 samples drift in 10 min for two same-tempo stems with real-world loop lengths (84-beat and 16-beat loops @116 BPM). Appendix A.1 |
| D3 | Loop-point glitch: a discontinuous loop seam fed through the per-block re-seek vocoder emits a loud transient at **every** wrap | **[REPRO]** 0.35 inter-sample step (30× signal max) at every wrap once the seam is discontinuous. Appendix A.2 |
| D4 | The production trim pipeline makes seams discontinuous: (a) `AVAudioFile` already strips AAC priming via iTunSMPB, then Rust trims another 2112/960 (**double-trim**, −3072 samples of real audio per loop); (b) the recorded 2112/960 are **hardcoded constants never measured** — and the variation encoder is ffmpeg's native AAC whose real priming is typically 1024, so the constants are likely wrong for every file | [VERIFIED] branch `MaestroAudioPlugin.swift:364-430`, `uniffi_api.rs:621-651`; `scripts/audio/process-stems.ts:243-260` hardcodes `2112/960`; `ffmpeg-encoder.ts` uses native `aac` |
| D5 | Rate silently falls back to 1.0 when a track's original tempo is missing (`Track::original_tempo` was always 0.0) — stretch silently disabled, gross drift | [VERIFIED] branch `engine.rs:881-884`; ADR-089 "Discovered deviations" |
| D6 | Bypass gated on `engine_tempo.is_none()`: the app always sets a tempo, so **every** track runs the fragile re-seek vocoder path on device, even at rate 1.0 | [VERIFIED] `engine.rs:893` + ADR-089 |
| D7 | Beat clock derived from render ticks (float accumulation `increment_beat_position`), stalls when render stalls; control plane re-anchors tracks from it through a 5–20 ms async bridge — three post-M5 sync bugs were all this | [VERIFIED] `engine.rs:1066-1076`; ADR-089 Post-M5 |
| D8 | Timing authority = 5 ms GCD wall-clock timer producing into the ring; never profiled on device; xcframework shipped debug-only | [VERIFIED/DOC] ADR-087/089 |
| D9 | `sync_length_samples` overwritten by the **last-loaded variation**; loop length and read length are two hand-synchronized values | [VERIFIED] `engine.rs:565-580` |
| D10 | Variation metadata is per-stem (one `encoder_delay/end_padding/original_samples` shared by three independently encoded files); variation switch never revalidates length | [VERIFIED] `music-library.json`, `AudioInitializationManager.ts:576-581` |
| D11 | iOS load path unconditionally routes through `AVAudioConverter` to 48 kHz; if any asset is not 48 kHz, trims land at wrong offsets and loop lengths are perturbed per-file (latent: current assets are 48 kHz) | [VERIFIED] `MaestroAudioPlugin.swift:366-397` |
| D12 | 5×-hand-mirrored API surface (~2,384 lines of UniFFI+WASM mirrors), `NativeEventBridge` impedance layer, ~12 `if (nativeAdapter)` coordinator forks, no `WebAudioAdapter` | [VERIFIED] branch `ffi/`, adapters |
| D13 | Silent-stub milestone discipline: Phases 1–2 declared done over a render path emitting zeros; cross-tempo test passed by fixture coincidence; peak assertion loosened to pass | [DOC] ADR-087/089 |

### 2.2 The decisive finding the dossier missed: non-integer musical loop lengths

Real stem metadata (`src/data/music-library.json`):

- San Francisco voice: 84 beats @ 116 BPM @ 48 kHz = **2,085,517.24** samples; asset is 2,085,518.
- Humboldt bass: 32 beats @ 113 BPM = **815,575.22** samples; asset is 815,576.

An exact musical loop length is generally **not an integer number of samples**. Any engine
that loops PCM at its integer sample length injects up to ±0.5 sample of musical error per
wrap, with a different residual per stem. Two stems of the same song with different loop
lengths therefore drift apart linearly forever — ~2.4 samples/min for the San Francisco
pair (reproduced, Appendix A.1), faster for shorter loops, and catastrophic once trim errors
change lengths by thousands of samples (D4). **This is the structural root of the audible
drift**, and it exists at rate 1.0, with no floats involved. The cure is wrapping in beat
space (§3.1), not better rounding.

### 2.3 Corrections to the dossier (auditable)

1. **§5.1 over-attributes the drift to float accumulation.** The branch's carried-fraction
   cursor (`floor(frame_count·rate + frac)`) is actually a correct Bresenham accumulator at
   constant rate; its float error is ~10⁻⁷ samples/hour. The dominant verified drift vectors
   are D2 (integer-length looping), D4 (trim corrupting lengths), D5 (silent rate fallback),
   and D7 (control-plane re-anchoring from a float beat clock). The *class ban* on advanced
   cursors stands — because the class is unprovable and invites exactly D5/D7-style patches —
   but the spec targets the real mechanisms, not only the symbolic one.
2. **§5.2 is wrong that a modulo-wrapped seek window inherently glitches the vocoder.**
   Reproduced: with a perfectly seamless loop, the branch's own re-seek path renders the
   seam cleanly even at rate 0.75 (Appendix A.2). The audible glitch appears precisely when
   the seam content is discontinuous — which production assets are, because of D4. Loop-aware
   feeding is still specified (§3.2), but asset trim correctness (§3.4) is the load-bearing
   fix.
3. **§5.4 "non-deterministic across architectures" is refuted** for the position math in
   question: `f64` `*`, `/`, `%`, `.round()` are IEEE-754-deterministic across ARM/x86; Rust
   applies no FMA contraction or fast-math. Genuine platform variance exists only in `powf`
   (pitch ratio) and FMA contraction inside the C++ stretcher — both affect audio sample
   values, never positions. The spec still eliminates floats from timing entirely (§3.1), for
   provability rather than for ARM-vs-x86 determinism.
4. **§5.4 per-instance stretcher latency: deterministic — confirmed.** `signalsmith-stretch`
   0.1.3 (C++ v1.3.1 via FFI): `inputLatency/outputLatency` are pure functions of
   `configure()` (preset default @48 kHz: 2880/2880); no runtime state affects them.
   Reproduced: measured uniform 2880-frame pre-roll. Caveat: the C++ default constructor
   seeds an RNG used only when the stretch ratio exceeds the "clean stretch" bound — audio
   values (not positions) become nondeterministic at extreme ratios; §4.8 bounds the ratio.
5. **§5.4 per-variation encoder delay: structurally confirmed, practically refined.** All
   three variation files of a stem share one hardcoded trim pair. Same encoder + settings ⇒
   genuinely different priming between variations of one stem is unlikely; the real exposure
   is that the recorded values are asserted, not measured, and are likely wrong for every
   file (ffmpeg native AAC primes ~1024, not 2112). Resolution in §3.4: per-file measured
   metadata + load-time length reconciliation with hard failure.
6. **Click-onset detector exists** on the branch (`render_pipeline_integration.rs::detect_click_onsets`)
   — the dossier's reuse pointer is valid; promote it to a shared test utility.

---

## 3. Target architecture

```
                       ┌────────────────────── JS / React (unchanged UI) ──────────────────────┐
                       │  hooks/stores → AudioEngineCoordinator → AudioBackend (one interface) │
                       └──────────────┬───────────────────────────────┬───────────────────────-┘
                                      │ NativeBackend                 │ WebBackend (Phase W)
                                      │ (Capacitor plugin)            │ (AudioWorklet port)
                       ┌──────────────▼───────────────┐ ┌─────────────▼───────────────┐
                       │ Swift thin shim:             │ │ worklet shim:               │
                       │ dispatch/snapshot/events,    │ │ same dispatch/snapshot/     │
                       │ decode+load, AVAudioEngine   │ │ events over MessagePort     │
                       └──────────────┬───────────────┘ └─────────────┬───────────────┘
                                      │            maestro-api (one serde schema:        
                                      │            Command / Event / Snapshot → UniFFI + TS codegen)
                       ┌──────────────▼──────────────────────────────────────────────┐
                       │ maestro-core ControlPlane (any thread, non-RT):             │
                       │ validates commands, prepares PCM/stretchers, owns metadata  │
                       └──────────────┬───────────────▲──────────────────────────────┘
                          RT command  │ SPSC          │ SPSC event + reclaim queues
                          queue (POD) │               │   + snapshot seqlock
                       ┌──────────────▼───────────────┴──────────────┐
                       │ maestro-core AudioCore (audio thread only): │
                       │ master clock F, segment table, derived      │
                       │ positions, readers, stretchers, mixer       │
                       │ — renders directly in the OS audio callback │
                       └─────────────────────────────────────────────┘
```

### 3.1 The timeline model: derived integer positions

**Constants and units.**

- `SR` — engine sample rate, **48,000 Hz**, fixed. Every asset must decode to exactly `SR`
  (§3.4). The engine never resamples; the OS converts the *mixed* output to the hardware
  rate downstream of the engine (one converter after the mix cannot affect inter-track sync).
- Tempo — integer **milli-BPM** (`u32`), e.g. 116 BPM = `116_000`. The UI's f64 BPM is
  quantized once at the backend boundary (round-half-even; max error 0.0005 BPM, inaudible)
  and echoed back in the snapshot; JS never converts back.
- Beat-units — the engine's musical position unit. `K = 60_000 × SR = 2_880_000_000`
  beat-units per beat. All beat-unit arithmetic is `u128`.

**Master clock.** `F: u64` counts output frames rendered while the transport runs. It is
owned by the audio thread and incremented by exactly the callback's frame count per render.
It does not advance when paused/stopped/interrupted. At 48 kHz, `u64` overflows after
12 million years; `u128` products below stay under 2¹²⁷ for > 10⁵ years (§4.1).

**Tempo segments.** An append-only table `[(F_k, bu_k, E_k)]`: at frame `F_k` the engine
tempo became `E_k` mBPM with accumulated musical position `bu_k`. Musical position:

```
bu(F) = bu_k + (F − F_k) · E_k        // u128; segment k is the last with F_k ≤ F
beats(F) = bu(F) / K                  // display only
```

Tempo changes append a segment at the next block boundary: `(F_c, bu(F_c), E_new)`.
`bu` is continuous across the change by construction — every track's musical phase is
preserved simultaneously; there is nothing per-track to re-anchor.

**Track mapping.** Track `i` has integer original tempo `O_i` (mBPM) and integer loop length
in beats `loop_beats_i` (validated at load, §3.4). Define `Λ_i = loop_beats_i × K` (u128).

```
u_i(F)   = floor(bu(F) / O_i)                      // unwrapped dictated source sample
w_i(F)   = floor(bu(F) / Λ_i)                      // loop (wrap) count
pcm_i(n) = n − ceil(w(n)·Λ_i / O_i),  w(n) = floor(n·O_i / Λ_i)
                                                   // PCM index of unwrapped source sample n
```

Derivations and properties (all integer, all stateless):

- **Rate is implicit and exact.** At constant tempo `E`, `u_i` advances `E/O_i` source
  samples per output frame in exact rational terms. A track at its own tempo (`O_i = E`)
  advances exactly 1:1 — `u_i(F+B) − u_i(F) = B` identically, enabling a bit-exact bypass
  (§3.2).
- **Beat-space wrap (Bresenham looping).** The per-wrap PCM read length
  `λ_w = ceil((w+1)Λ_i/O_i) − ceil(wΛ_i/O_i)` alternates between `⌊Λ_i/O_i⌋` and `⌈Λ_i/O_i⌉`
  so that the **average loop length equals the exact rational musical length with zero
  cumulative error**. This eliminates D2 by construction. (For the San Francisco voice, the
  read length alternates 2,085,517/2,085,518; the ±1-sample final read clamps to
  `min(idx, L_pcm−1)` — a sub-sample waveform detail at the seam, never accumulated.)
- **Sample lock.** For tracks `i, j`: `u_i(F)·O_i` and `u_j(F)·O_j` each lie within
  `[bu(F) − O, bu(F)]` — their musical phase difference is bounded by < 1 source sample of
  the slower track, for every `F`, forever. That bound is an algebraic identity of `floor`,
  not a property maintained by code.
- **No state to corrupt.** Arming a track mid-bar sets an `active` flag; its position is
  `u_i(F)` like everyone else's. Pause freezes `F` for all tracks at once. A stalled render
  stalls the clock and the music identically — there is no separate beat clock to diverge
  (kills D7). Quantized launch ("start at next bar") is an activation frame `F_q` computed
  by inverting `bu` (exact integer ceiling division); the render loop splits the block at
  `F_q` so activation is sample-exact within a block.
- **Timeline epoch (explicit product semantics).** `bu` epoch (bar 1 beat 1) resets only
  via an explicit `ResetTimeline` command. That command is issued in exactly two flows:
  explicit transport stop, and song switch. Track start/stop **never** resets it (kills the
  branch's Bug 3 class), and — codifying the armed-tracks product model — **`F` keeps
  advancing while the transport runs even if zero stems are audible**: stop every stem,
  wait, restart one, and it enters at the musical position the timeline reached in the
  silence. The UI's stop-all button maps to transport stop (epoch reset); individually
  stopping the last stem does not.

**Reference implementation note.** The render loop computes `u_i` at block edges only (two
u128 divisions per track per block — negligible). A `#[cfg(test)]` mirror implements the same
formulas in `num-bigint` rationals; §6.2's invariant test cross-checks them bit-for-bit.

### 3.2 Render path and loop-aware stretch

Per render callback (block of `B` frames, `B` set by the OS, any value):

```
for each active track i:
    span = u_i(F+B) − u_i(F)                 // integer source samples dictated this block
    if O_i == E (unity):                     // bypass: span == B exactly
        read B source samples via pcm_i(·) (circular, Bresenham wrap) → track buffer
    else:                                    // stretch path
        span_L = u_i(bu(F+B) + E·L_out) − u_i(bu(F) + E·L_out)   // L_out = stretcher output latency
        feed exactly `span_L` source samples (read via pcm_i(·) at the lead
        positions) into the track's Signalsmith stretcher
        (streaming process(input[span_L], output[B])) → track buffer
        // The feed leads the playhead by exactly L_out OUTPUT frames of
        // musical time (E·L_out beat-units), cancelling the vocoder's
        // constant output latency: stretched output lands ON the derived
        // grid, not 60 ms behind it (ADR-094). Rate-independent; spans
        // telescope within a tempo segment exactly as before.
    apply gain/pan (+ activation ramp if just armed) and mix into the output
F += B
write snapshot (seqlock); drain RT commands at block boundary
```

Design decisions, each closing a §2 defect:

- **Streaming, not re-seek (closes D3's engine half).** Within a tempo segment the
  stretcher is **never seeked**; it consumes the circular source stream. Loop seams are
  ordinary audio (assets are validated seamless, §3.4) — Signalsmith accepts arbitrary
  in/out counts per call (`process(input, output)`, ratio implied per call). A tempo
  change is a feed-schedule boundary (the compensated lead is a function of E, ADR-094):
  stretch-path tracks re-anchor at the change block, fading out one final block of the
  OLD schedule from the still-warm stretcher (waveform-continuous) under the one-block
  equal-power crossfade — no transient (T-SEAM-gated across the change), positions
  untouched.
- **The stretcher is a slave, never an authority.** Cumulative input fed after any number of
  blocks is exactly `u_i(F) − u_i(F_anchor)` (telescoping). The stretcher cannot reposition
  a track; the worst it can do is smear inside its constant ~120 ms window — bounded,
  constant, identical for all tracks, non-accumulating.
- **Unity bypass without phase split (closes D6).** Bypass output is bit-exact PCM. When the
  tempo moves off unity, the track anchors a stretcher (below) and crossfades bypass→stretch
  over one block (equal-power, ≤ 10 ms); positions are unaffected (they are derived).
  Hysteresis: unity means `O_i == E` exactly (integer compare).
- **Anchoring (start, variation switch, unity transition, stretch-path tempo change,
  post-`ResetTimeline`).** One-time procedure: `reset()`; build a seek window of
  `input_latency + output_latency` source frames ending at the compensated feed origin
  `pcm_i(u_i(bu(F₀) + E·output_latency) + input_latency)` (ADR-094); `seek(window, rate)`;
  then stream.
  The known one-block smear of `seek()` (ADR-089's empirical finding) is masked by the
  ≤ 10 ms activation gain ramp and occurs only at musically chosen entries — never at loop
  seams. Latency values are read from the instance (deterministic per §2.3#4; correctness
  does not depend on uniformity since each anchor uses its own instance's values).
- **Variation switch.** PCM pointer swap + re-anchor + one-block crossfade at the block
  boundary. Loop length cannot change (all variations validated to identical
  `loop_beats`/length, §3.4) — D9/D10 die: `loop_beats` is the **single source of truth**
  for loop length; PCM lengths are validated against it, never the reverse.
- **Pitch.** `set_transpose_semitones` on the stretcher (and pitch-compensation as today);
  affects no position math. A pitch shift on a unity track promotes it to the stretch path
  (same crossfade rule).
- **Mute/solo and silent tracks skip DSP.** Because positions are derived, an inaudible
  track (muted, soloed-out, or effective gain 0) safely skips its read/stretch work
  entirely — there is no cursor to keep warm. On becoming audible again it re-anchors under
  the activation ramp, exactly like a restart (§3.1). This is both the CPU win and the
  correctness statement: silence costs nothing and cannot desync.
- **RT-safe buffers.** All scratch (read buffer, stretch in/out, mix bus) is preallocated at
  track-add time for `B_max = 4096`; the render loop allocates nothing (§3.3).

### 3.3 Timing authority, threading, and real-time safety

**The audio callback is the timing authority.** No wall-clock timer exists anywhere in the
audio path (kills D8). The engine renders **directly inside** the OS audio callback:

- iOS: `AVAudioSourceNode` (format: 48 kHz stereo float32 non-interleaved) whose render
  block calls `maestro_render(core_ptr, buffers, frame_count)` — a `#[no_mangle] extern "C"`
  function that runs the §3.2 loop. One source node carries the full mix.
- Web (Phase W): `AudioWorkletProcessor.process()` calls the same engine compiled to WASM.

**Two-plane ownership.**

- `AudioCore` — owned by the audio thread exclusively. Contains: `F`, segment table, track
  render states (PCM `Arc` handles, stretchers, scratch), mixer. No mutex anywhere on it.
- `ControlPlane` — runs on the caller's thread behind the FFI. Validates commands, decodes
  nothing itself (decode is the platform shim's job, §3.4), allocates/configures stretchers
  and PCM buffers, then transfers them to the AudioCore via the RT command queue.

**Queues (all lock-free, fixed-capacity, preallocated).**

- `rt_commands: spsc<RtCommand>` — POD commands applied at block boundaries
  (SetTempo{mbpm}, Activate{track, at_frame, ramp}, Deactivate, SetGain/Pan/Pitch,
  SwapVariation{track, Arc<Pcm>}, InstallTrack{...}, ResetTimeline, …). Capacity 256;
  control plane blocks (its thread, not audio) if full.
- `rt_events: spsc<RtEvent>` — POD edge events out (Started, Stopped, LoopWrapped,
  Underrun, BudgetExceeded). Drained by ControlPlane, merged with control-plane events,
  exposed via `drain_events()`.
- `reclaim: spsc<Reclaimable>` — the audio thread never drops an `Arc`/buffer; replaced
  resources are pushed here and freed by ControlPlane (no dealloc on the audio thread).
- `snapshot` — a seqlock'd POD struct written by the audio thread once per block, read
  lock-free by `snapshot()` (§3.5).

**RT rules (enforced, not aspirational).** In `maestro_render`: no locks, no allocation or
deallocation, no `Arc` refcount drops to zero, no syscalls, no logging, no panics
(`panic = abort` + the render loop is panic-free by construction; `debug_assert` only).
Enforcement: (a) unit tests wrap a full render in `assert_no_alloc`; (b) a CI job runs the
invariant test under it; (c) `clippy.toml` denies `std::sync::Mutex` and friends in the
`audio_core` module.

**Render budget gate (M0/M1 checkpoint, §5).** Measured on a physical device, release
build, 4 tracks at 1.5×: per-callback render time p99 < 50% and max < 90% of the callback
period. The audio thread records its own timing into the snapshot (host-time delta around
the render call; reading a timestamp is RT-safe).

**Plan B (pre-decided fallback, only if the gate fails on the oldest supported device):**
move the §3.2 loop to a dedicated realtime-priority producer thread feeding an SPSC sample
ring (the branch's one correct invariant), with the callback popping. **The clock remains
`F` as counted by rendered frames; the producer is paced by ring occupancy, never by a
timer.** Sync is unaffected by construction (positions derive from `F` regardless of who
advances it); only latency changes. No other architecture changes.

### 3.4 Decode, trim, and loop-length validation (single source of truth)

Closes D4, D9, D10, D11. The invariant: **by the time PCM reaches the engine, its length is
exactly consistent with the stem's musical identity, or the load fails loudly.**

1. **Metadata (per file, not per stem).** The encoding pipeline
   (`scripts/audio/process-stems.ts`) is extended to emit, for **each variation file**:
   `encoder_delay`, `end_padding` **measured from the actual encoded file** (iTunSMPB /
   packet-table read-back, as the existing `afconvert-encoder.ts` path already does — the
   hardcoded `2112/960` constants are removed), `original_samples`, and the stem-level
   `loop_beats` (integer) + `original_tempo_mbpm`. `music-library.json` schema gains these
   fields; existing entries are regenerated by re-running metadata extraction against the
   files on R2 (a one-time script; no re-encode required unless validation fails).
2. **Loop identity check (build time + load time).**
   `|original_samples − loop_beats·K/O| < 24 samples` (0.5 ms) — else the asset is rejected
   at pipeline time. `loop_beats` is authoritative thereafter; `Λ = loop_beats·K` drives all
   wrapping (§3.1). All variations of a stem must declare identical `loop_beats` and
   `original_samples`.
3. **Decode (platform shim).** iOS: `AVAudioFile` decode **with no resampling and no
   channel conversion** — assert `processingFormat.sampleRate == 48000` and channels == 2,
   else `TrackLoadFailed` (no fallback, per project principle). Web: `decodeAudioData` on a
   48 kHz context. Tests/CLI: `symphonia`.
4. **Length reconciliation (in Rust, decoder-agnostic — the double-trim killer).** Given
   decoded length `L_dec`:
   - `L_dec == original_samples` → decoder already applied gapless trim; **do not trim**.
   - `L_dec == original_samples + encoder_delay + end_padding` → raw decode; trim exactly.
   - anything else → `TrackLoadFailed { reason: LengthMismatch {…} }`. No heuristics.
   The reconciliation result (which case fired, per file) is reported in the load event and
   asserted stable across variations of a stem.
5. **Seam validation (build time).** The pipeline's existing decode step additionally checks
   seam continuity per variation: `|pcm[0] − pcm[L−1]|` and short-window cross-correlation
   end→start within thresholds; failures are fixed at authoring/encode time (e.g. trim-to-
   zero-crossing or authored crossfade), **never at runtime**.

### 3.5 State authority and egress

**Rust owns all transport/armed/tempo/beat/position state.** JS holds render-only mirrors.

- `snapshot() → Snapshot` — the **only** state egress, one bridge call, default poll 15 Hz
  (UI), 30 Hz while level meters visible (kills the 200-call/s polling storm; one call
  returns everything):

```rust
struct Snapshot {                      // POD, seqlock'd, serde+ts-rs generated
  engine_frame: u64, tempo_mbpm: u32, transport: TransportState,
  bar: u32, beat_in_bar: f32,          // derived from bu(F); display only
  timeline_epoch: u32,
  tracks: [TrackSnap; MAX_TRACKS],     // MAX_TRACKS = 16
}
struct TrackSnap {
  slot: u8,                            // stable slot index assigned by InstallTrack
  status: TrackStatus, armed: bool, variation: u8,
  loop_count: u32, loop_progress: f32, // derived from bu(F) mod Λ; display only
  rms_l: f32, rms_r: f32, peak_l: f32, peak_r: f32,
}
```

Track identity across the bridge is the **slot index** (assigned at `InstallTrack`, freed at
`RemoveTrack`, never reused while occupied); the backend owns the `trackId ↔ slot` map. No
string hashing in the snapshot — no collision class, no per-poll string traffic.

- `drain_events() → Vec<Event>` — edge-triggered only (loaded, failed, started, stopped,
  underrun, tempo-applied, variation-applied). No state is reconstructed from events; missing
  an event can never desync JS because JS state is cosmetic.
- **JS makes no timing decisions.** `toggleTrack` etc. send commands; the engine decides
  activation frames (quantize policy is a command parameter). The branch's Bug 1/2 class
  (bridge round-trip in the timing path) is structurally gone — there is no beat capture to
  race because there is no JS-side anchoring at all.

### 3.6 One API schema, generated bindings

Crate `maestro-api` (new, ~300 lines of types): `Command`, `Query`, `Event`, `Snapshot`,
`Config` as serde enums/structs, `#[serde(tag = "type", rename_all = "camelCase")]`, with
`ts-rs` derives generating `maestro-api.d.ts` (committed; CI fails if regeneration differs —
binding drift is a build error, kills D12).

Transport surface (total, for every platform):

```
init(config) ; dispatch(command_json: String) -> result_json   // control plane, ≤ 60 Hz
snapshot_json() -> String ; drain_events_json() -> String      // egress
load_pcm(track, variation, bytes/ptr, meta) -> result          // the one bulk-data call
maestro_render(core, buffers, frames)                          // C ABI, audio thread only
```

- **Native:** UniFFI proc-macros export exactly these six (no UDL, no mirror types — the
  1,216-line `uniffi_api.rs` and 1,168-line `wasm_bindings.rs` mirrors are not ported).
  Swift plugin = passthrough + decode/load + AVAudioEngine setup; it holds no state.
- **WASM:** the same six via `wasm-bindgen`. Same JSON in both transports; the TS layer has
  **one** codec, generated. Events arrive correctly shaped — `NativeEventBridge` is replaced
  by `JSON.parse` + a generated type guard.
- Adding an operation = one enum variant + one `match` arm in ControlPlane. CI's
  `cargo run -p maestro-api --bin gen-ts && git diff --exit-code` keeps all layers honest.

**Schema-evolution law (ADR-100, from the prewarm-parks-as-swaps production
bug):** a mode flag that flips a call between destructive and non-destructive
behavior must ride a surface an old engine *rejects* — a new `Command`
variant (unknown variants fail loudly at parse) or a handshake-gated field —
never a serde-defaulted field an old engine silently *reinterprets*.
`LoadMeta.cache_only` predates the law and stays, because the
**schema-fingerprint handshake** now guards the whole class: the engine
embeds, at its build time, a hash over the committed `bindings/` content
(`maestro_api::SCHEMA_FINGERPRINT`, computed by build.rs; gen-ts writes the
same value to `bindings/fingerprint.ts` for the app bundle); both transports
surface it (wasm: the worklet `ready` message; native: the plugin's
`getSchemaFingerprint`) and the backends assert equality at init — mismatch
or absence is a hard init failure. Committed engine artifacts
(`public/maestro/*`, `worklet/pkg/*`) are additionally gated in CI
(`scripts/check-engine-artifact-freshness.sh`, fingerprint-compare — the
wasm is not byte-reproducible across environments), and `BackendBase`
carries a runtime park tripwire: a `cache_only` load that changes the
audible variation without a recorded user switch intent aborts the parking
loop and raises an error-grade event on the ADR-096 net.

### 3.7 `AudioBackend` and coordinator integration

One TS interface, two implementations, zero platform forks in the coordinator:

```ts
interface AudioBackend {
  init(config: EngineConfig): Promise<void>;
  dispatch(cmd: Command): Promise<CommandResult>;
  getSnapshot(): Promise<Snapshot>;          // NativeBackend: plugin call; WebBackend: SAB/port
  onEvent(cb: (e: EngineEvent) => void): Unsubscribe;
  loadTrack(req: LoadRequest): Promise<void>; // native: URL→Swift decode; web: decodeAudioData→WASM
}
```

`AudioEngineCoordinator` routes through `this.backend` only; the ~12 `if (this.nativeAdapter)`
forks are removed in M3 (the migration commit deletes `NativeAudioAdapter`/`NativeEventBridge`
and asserts via ESLint `no-restricted-imports` that nothing outside the backends imports the
plugin). The coordinator's init path is identical on both platforms (kills the dual-init
ADR-086 bug class): `new Coordinator(backend)` → `backend.init()` → ready.
Level meters read from the snapshot poll (no separate polling path).

### 3.8 Web convergence plan

- **Until Phase W ships, web runs the current TS engine untouched.** No shared logic is
  duplicated in the interim: the Rust core is the only implementation of the new model, and
  it ships on native first.
- **Phase W:** compile `maestro-core` to WASM; the engine runs inside an
  `AudioWorkletProcessor` (audio thread = worklet thread; `F` advances per 128-frame
  quantum). Commands/events cross via MessagePort into the worklet's dispatch; snapshot via
  a SharedArrayBuffer seqlock (fallback: port-posted at 15 Hz where COOP/COEP is
  unavailable). `WebBackend` implements `AudioBackend`; the coordinator does not change.
- **Parity gate:** the §6 invariant suite runs against the native build and the WASM build
  in CI; positions must match **bit-for-bit** for identical command scripts (they are
  integers — this is achievable and asserted). A/B bake on web (TS engine vs WASM behind a
  flag) for two weeks before deleting `audio-engine/` (~55 files).
- Stretch on web: Signalsmith compiles into the WASM module (same C++ via Emscripten in the
  crate's build); the separate signalsmith-stretch.wasm and its worker pipeline retire with
  the TS engine.

---

## 4. Decisions resolved

| # | Decision | Resolution |
|---|----------|-----------|
| 4.1 | Numeric model | Integers only in timing: `F: u64`; tempos `u32` mBPM; beat-units `u128` (`K = 2.88e9/beat`). Overflow: `bu` after 100 years of continuous 999-BPM playback ≈ 2⁹⁰ ≪ 2¹²⁸; `u_i` products bounded by `bu` — analysis in §3.1. Floats never feed timing; f64 appears in UI conversion (one-way) and DSP only. Cross-platform: integer ops are bit-identical on ARM/x86/WASM; position parity is asserted bit-for-bit in CI (§6.6). |
| 4.2 | Per-variation trim | Per-file measured `encoder_delay`/`end_padding` + decoder-agnostic length reconciliation with hard failure (§3.4). Variations validated to identical `loop_beats`/`original_samples`. The double-trim class is killed by reconciliation, not by trusting any decoder's behavior. |
| 4.3 | Shared vs per-track stretcher | **Per-track instances** (different pitch/variation/anchor states), identically configured; latency is deterministic per config (verified, §2.3#4) and correctness does not depend on uniformity (anchors use instance-reported values). `new_cheaper()` is reserved as a budget fallback (§5 M1 gate), applied globally if ever (never mixed presets across tracks). |
| 4.4 | Timing authority | The OS audio callback owns the clock; `F` = frames actually rendered. No timers in the audio path. Plan B (producer thread + ring) is pre-decided in §3.3 and changes latency only, never sync. |
| 4.5 | Snapshot & events | Defined once in `maestro-api`, serde+ts-rs generated, seqlock egress, 15 Hz poll (§3.5/3.6). |
| 4.6 | Loop length authority | Integer `loop_beats` per stem; `Λ = loop_beats·K` drives beat-space wrapping; PCM lengths validate against it (§3.1, §3.4). |
| 4.7 | Engine sample rate | Fixed 48 kHz; assets must be 48 kHz (validated); hardware-rate conversion happens after the mix in the OS graph and cannot affect inter-track sync. A future non-48k asset requires a pipeline change, not an engine change. |
| 4.8 | Tempo/pitch ranges | Engine tempo clamp 40–300 BPM; rate `E/O_i` clamped to [0.25, 4.0]; pitch ±12 semitones. Combined time-pitch ratio kept inside Signalsmith's clean-stretch bound to avoid its RNG path (audio-value nondeterminism, §2.3#4). Out-of-range commands are rejected with an error event. |
| 4.9 | Interruptions / routes | Interruption → `Pause` command (F freezes; all tracks together; resume continues — positions derived, nothing to re-anchor). Route change → AVAudioEngine restart, F continuity preserved; media-services-reset → rebuild graph, engine state survives in-process. All handled in Swift→command, never in JS. |
| 4.10 | Capacitor bridge latency | Accepted for control feel (commands apply at the next block boundary, ≤ ~30 ms end-to-end). Quantized starts make this musically irrelevant; nothing timing-critical crosses the bridge (§3.5). |
| 4.11 | PCM memory policy | Worst-case naive load is real (~17 MB per variation per long stem; 5 stems × 3 variations ≈ 250 MB — jetsam territory). Policy: **eager-load the default variation only**; simple/complex load lazily on first variation interaction (prefetch on UI hover/long-press), Arc-swapped per §3.2 with eviction of the variation least recently audible once a **per-song PCM budget of 96 MB** is exceeded (never evicting an audible variation). Variation switch to a not-yet-loaded variation is asynchronous: command is acknowledged, swap applies on load completion at a block boundary (event signals it). Memory in use is reported in the snapshot; C2 and C5 gate on the budget. f32 storage stands; int16 storage (−50%) is the pre-decided contingency if the budget fails on the largest song. **Amended (ADR-098)**: the budget is `EngineConfig.pcm_budget_bytes` (omitted → the 96 MB default above); the composition roots pass a device-RAM-tiered value (288 MiB ≥ 6 GiB RAM / web desktop, 192 MiB ≥ 4 GiB, default otherwise — `pcm-budget-policy.ts`). On prewarm-enabled tiers, song load additionally fetches+decodes every non-default variation in the background and parks it via `LoadMeta.cache_only` loads (park in the slot's cache, NO swap, RT untouched; a park while a switch to that variation is pending completes the switch; a cache-only load of the installed variation is a no-op; empty slot rejected), so variation switches take the cached path (<10 ms) instead of the async round-trip. Default-tier devices keep the lazy behavior above verbatim. |
| 4.12 | Library lint (build-time) | The pipeline validates every song before publish: stem count ≤ MAX_TRACKS (16); `loop_beats`, `original_tempo_mbpm`, per-file gapless metadata present; loop-identity check (§3.4#2); seam check (§3.4#5); `beats_per_bar` taken from the song's time-signature metadata (no hardcoded 4/4 in the engine — bar display derives from it). A song failing lint cannot ship; the engine still re-validates at load (defense in depth). |
| 4.13 | Lock screen / remote controls, background-audio UX | Out of scope for this program (product feature, not engine). The session category (`.playback`) already permits background playback; MPRemoteCommandCenter wiring is backlog. Noted so its absence is a decision, not an oversight. |

---

## 5. Sequencing — hardest first, a checkpoint on every milestone

Rules: a milestone is **done** only when its named checkpoint passes — on a **physical
iPhone** in a **release** build wherever the milestone touches render/delivery. By-ear,
simulator, and debug evidence are not acceptance. No milestone closes on a stub: every
checkpoint includes an assertion that fails on silence or on disabled stretch (§6.1).
Checkpoint evidence (harness reports, capture analyses) is committed under
`documentation/validation-runs/`.

### M0 — Walking skeleton: two cross-tempo stems, sample-locked, out of a physical iPhone
*Scope:* `maestro-api` skeleton (init/dispatch/load/snapshot only, ~6 commands); AudioCore
with timeline (§3.1), bypass + streaming stretch path (§3.2, fixed tempo set at init, no
live changes yet); RT queues; Swift shim with AVAudioSourceNode callback render; device
harness (§6.4) with mixer-tap capture + onset analysis; metronome fixtures (60/90/120 WAVs
reused from the branch) + Metronome Test song; offline invariant test (§6.2, no tempo-change
leg yet) + loop-seam test (§6.3) running in CI from day one.
*Explicitly absent:* volume/pan/pitch/variations/arming UX/web — breadth waits.
*Checkpoint C0:* On a physical iPhone, release build, two captured scenarios:
(1) the Metronome Test song (two 120-BPM stems, L/R) at engine tempo 120 — unity-path lock;
(2) harness-bundled **60-BPM and 90-BPM** metronome stems (L/R) at engine tempo 120 — both
stretched, no fixture coincidence (per §6.1), with the stretch-disabled negative control
failing as required. Each capture: cross-stem onset offset **constant, drift = 0 samples
over ≥ 60 s and ≥ 8 wraps**; **underruns = 0**; render budget p99 < 50% of callback period
with 4 tracks at 1.5×. Offline: invariant + seam suites green, negative controls red.

### M1 — Full timing surface
*Scope:* live tempo changes (segment table), pause/resume, `ResetTimeline`, quantized and
immediate activation (mid-bar arming), unity↔stretch transitions with crossfade, variation
switch (engine-side swap+anchor), Bresenham wrap clamp, pitch + compensation.
*Checkpoint C1:* full §6.2 invariant test (≥ 50 M frames, ≥ 100 wraps, ≥ 2 tempo changes —
one at a loop boundary, one mid-bar — plus a pause/resume, a mid-stream arm, and the
stop → tempo-change-while-stopped → restart leg) — drift **0 samples, bit-exact
positions**; §6.3 seam test across a tempo change; on device: C0 measurements repeated
**plus** a tempo change mid-capture (offset before == offset after, zero slope both sides)
**and** the §6.4 stop/tempo-change/restart sequence (re-entry equality threshold); budget
gate re-passed (4 tracks, 1.5×, with one variation switch and one tempo change during
measurement).

### M2 — Full control plane on the generated schema
*Scope:* complete `Command`/`Event`/`Snapshot` per §3.6 (volume/pan/mute/solo, full track
lifecycle, safety presets); ts-rs codegen + CI drift check; decode/trim reconciliation
(§3.4) in the Swift shim + Rust; per-file metadata pipeline change + regenerated
`music-library.json`; interruption/route handling (§4.9); event/reclaim queues complete.
*Checkpoint C2:* §6.5 load-validation suite (including a deliberately mis-trimmed fixture →
hard failure, and a variation-length-mismatch fixture → hard failure); §6.7 control-plane
storm test; binding-drift CI check green; device: full Metronome Test flow driven through
`dispatch` only (no test scaffolding), C1 measurements re-passed, plus an
interruption (timer-triggered route to speaker→receiver→speaker) with offset unchanged
across it; **memory gate**: loading the largest library song stays under the §4.11 per-song
PCM budget (snapshot-reported, asserted by the harness), and a lazy variation switch
applies cleanly (event observed, no audio gap on the playing stems).

### M3 — App integration behind `AudioBackend`
*Scope:* `AudioBackend` interface + `NativeBackend`; coordinator de-fork (delete
`NativeAudioAdapter`, `NativeEventBridge`, all `if (nativeAdapter)` branches); level meters
and beat UI from snapshot poll; `AudioInitializationManager` native path = web path; UI
unchanged (zero component diffs outside the backend layer).
*Checkpoint C3:* full E2E suite green on iOS build (existing Playwright + console-error
detection where applicable, plus a native smoke checklist); grep-gate: no platform
conditionals in coordinator (`if.*nativeAdapter` count = 0 enforced by ESLint rule);
**no-dual-engine gate:** on native, the WebView must construct no `AudioContext` and load
no worklet (runtime assertion in the native build + lint rule) — the TS engine must be
provably inert on iOS, not merely unused, so it cannot contend for the audio session;
device re-run of C2 measurements from the real UI (tap play in the app, capture, analyze);
web regression suite untouched and green (TS engine still serving web).

### M4 — Web convergence (Phase W)
*Scope:* WASM build in AudioWorklet; `WebBackend`; SAB/port snapshot; parity suite; A/B
flag; two-week bake; delete `audio-engine/` TS engine + signalsmith worker pipeline.
*Checkpoint C4:* §6.6 native↔WASM position parity **bit-for-bit** in CI; §6.2/§6.3 suites
green compiled to wasm32; web E2E green on the WASM backend; A/B bake exit criteria (no
sync/glitch regressions reported, error rates ≤ TS engine baseline) before deletion lands.

### M5 — Hardening and release
*Scope:* release xcframework pipeline (release slice is the only artifact CI publishes —
debug-only shipping is structurally prevented); 4-hour device soak; Sentry breadcrumbs from
engine events; dashboard counters (underruns, budget, reconciliation outcomes).
*Checkpoint C5:* 4-hour soak on the oldest supported device: drift = 0 samples start→end,
underruns = 0, no memory growth (RSS slope ≈ 0), budget headroom unchanged hour 1 vs hour 4.

Dependency note: M0/M1 are the project; everything after is breadth on a proven core. If C0
or C1 cannot be made to pass, the model is wrong and **the spec fails loudly here** — there
is no path where later milestones proceed on a red core checkpoint.

---

## 6. Validation plan

### 6.1 Structural anti-stub policy

Every sync assertion in this plan is paired with an assertion that the audio is *present and
correct in content*, so "green over silence" is impossible:

- Onset-grid tests assert the **exact expected click count** and an RMS floor — silence or a
  dead track fails the count.
- Cross-tempo tests use **60 vs 90 BPM** fixtures (no shared-click coincidence as with
  60/120 — the branch's fixture-coincidence trap is excluded by fixture choice), and include
  a **stretch-disabled negative control**: with stretch forced off, the test must fail.
- Thresholds live in one `validation_thresholds.rs` constants file; loosening any value
  requires an ADR reference in the same commit (CI greps for the marker).

### 6.2 T-INV: zero-drift invariant test (offline, exact, every commit)

Drive the real AudioCore (the render entry, not internal helpers) with 4 tracks —
`O = {60_000, 90_000, 113_000, 116_000}` mBPM, loop lengths from real metadata including the
non-integer-musical-length cases (§2.2) — at engine tempo 120_000, block sizes alternating
{128, 480, 1024} (proves block-size independence), for ≥ 50 M output frames (≈ 17 min audio)
spanning ≥ 100 wraps of the shortest loop, with: a tempo change to 87_000 exactly at a loop
boundary, another mid-bar, a pause/resume, a track armed mid-stream, and an explicit
**stop → tempo-change-while-stopped → restart** leg: one track is deactivated, the engine
tempo changes (once mid-bar, once at the stopped track's would-be loop boundary) while it
is silent, and the track is then reactivated — its positions after re-entry must equal the
reference function's dictate bit-for-bit, and its pairwise musical-phase offsets to the
tracks that played through the change must be identical to their pre-stop values.

Assertions (tolerance **zero**):

- At every 1 M-frame checkpoint and at the end: each track's reported `u_i(F)` equals the
  `num-bigint` rational reference implementation **bit-for-bit**.
- For every pair: musical-phase difference `bu(F) − u_i(F)·O_i` is within `[0, O_i)`
  (the algebraic bound) and the *pairwise* offset sequence is **identical** at every
  checkpoint (constant forever).
- Bypass/stretch equivalence: a unity track's dictated positions are identical whether it
  renders via bypass or stretch path (paths differ in audio, never in positions).
- Runs in CI on x86 and ARM runners and (from M4) wasm32 — outputs must be identical.

**Negative controls (CI asserts these FAIL):** a `drift-injection` test feature provides
three sabotage modes — `float_cursor` (per-block f64 accumulate), `pcm_length_loop` (wrap at
integer PCM length), `timer_clock` (advance F from a simulated wall timer with jitter).
T-INV must detect all three. This proves the suite can catch drift at the 1-sample level;
the suite is broken (build fails) if a sabotage mode passes.

### 6.3 T-SEAM: loop-seam test (offline, every commit)

Seamless tonal fixture (integer-cycle sine) + metronome fixtures, rates {1.0, 0.75, 1.5},
≥ 50 wraps: (a) max non-onset inter-sample step within 1.5× the signal's natural maximum at
every wrap window (the branch engine fails this only when content is discontinuous — our
loader guarantees it isn't; the engine must keep it true through streaming feed); (b) click
onsets land on the exact derived grid (onset frame == predicted frame from `u(F)` inversion,
± 0 for bypass; for the stretch path, ON the same grid — absolute median within the 2 ms
class bound, ADR-094 — with zero per-wrap variance; the uncompensated output latency, a
constant 2880, is a gated regression). **Negative control:** the double-trim sabotage
fixture (2112 samples chopped)
must produce a detected seam transient — proving the detector sees real seam clicks
(reproduced at 0.35 vs 0.0115 baseline, Appendix A.2).

### 6.4 T-DEV: device delivery harness (physical iPhone, release build; gates every milestone touching render/delivery, and every release)

- Harness mode in the plugin (debug menu / URL param): installs an `AVAudioEngine`
  mainMixerNode tap, writes a CAF capture; a `maestro-validate` Rust CLI pulls the file and
  analyzes (reusing `detect_click_onsets`).
- Script: `scripts/device-validation.sh <device-udid>` builds release, installs, runs the
  scripted scenario (60 s, both stems playing from t=0): **stop stem B at t=20 s** (stem A
  keeps playing); **tempo change at t=30 s while B is silent**; **restart B at t=35 s**;
  variation switch at t=40 s; interruption at t=50 s. Pulls capture + snapshot log, runs
  the analyzer, emits a pass/fail report (committed to `documentation/validation-runs/`).
  This ordering deliberately validates re-entry sync after a tempo change that happened
  while the stem was stopped — the exact class of control-plane re-anchoring bug that
  produced the previous attempt's post-"done" failures.
- The analyzer computes the **expected onset grid from the pulled command/snapshot log**
  (activation frames, tempo segments), not from wall-clock assumptions — so the tempo
  change, stop window, and re-entry all have exact predicted frames, and the comparison is
  predicted-vs-measured, never measured-vs-measured alone.
- Thresholds: cross-stem onset offset **constant; drift = 0 samples** over the capture
  (slope of offset-vs-time = 0; any nonzero integer step fails); **re-entry equality**: stem
  B's onset offset to stem A in the post-restart window (t ≥ 35 s, across the tempo change)
  equals its pre-stop offset (t < 20 s) exactly — same constant, zero step at re-entry;
  expected onset count exact (including zero B onsets inside the stop window — an onset
  detected there fails the run);
  RMS floor met; **underruns = 0** (atomic counter via snapshot); render budget p99 < 50%,
  max < 90% of callback period at 4 tracks 1.5×.
- The analyzer has its own negative control: fed a capture from the sabotaged build
  (`pcm_length_loop`), it must report drift ≠ 0. (A captured pair exists from the branch
  reproduction: 24-sample drift — Appendix A.1.)

### 6.5 T-LOAD: decode/trim validation (every commit)

Fixtures: correctly-tagged M4A; M4A whose decoder pre-trims (simulated by length); raw
decode (+delay+padding length); wrong-length file; variation set with one mismatched length;
non-48k file. Assert: reconciliation picks the right case for the first three; the last
three produce `TrackLoadFailed` with the precise reason; no code path silently pads, trims
heuristically, or resamples.

### 6.6 T-PAR: cross-platform parity (CI matrix; M4 adds wasm)

Same command script → `u_i(F)` checkpoint dumps compared **byte-for-byte** across
x86-linux / arm64-macOS / wasm32. (Audio sample values are compared per-platform against
golden hashes for regression, but cross-platform audio equality is **not** claimed — FMA
contraction in the C++ stretcher may differ in LSBs; positions, not waveform bits, define
sync. This boundary is intentional and documented.)

### 6.7 T-CTRL: control-plane storm (every commit)

Randomized (seeded) command fuzz: arm/disarm/tempo/variation/pause storms at up to 60
commands/s for 10 M frames; assert T-INV invariants still hold at every checkpoint, the
RT queues never overflow silently (overflow = surfaced event + test failure), and snapshot
fields stay consistent (e.g. `loop_progress` monotone within a wrap).

### 6.8 CI policy

- Every commit: T-INV, T-SEAM, T-LOAD, T-CTRL + negative controls (cargo, both feature
  sets); ts-rs binding drift check; clippy RT lints; `assert_no_alloc` render test.
- Every milestone touching render/delivery + every release: T-DEV on physical hardware;
  the report is part of the PR.
- A milestone PR that changes any threshold or disables any negative control does not merge
  without an ADR in the same diff.

---

## 7. Reuse map (from `main` and the branch)

**Keep as-is (parts bin, used by the new core):**
`timing.rs` (pure conversion math — display/UI helpers), `pitch.rs`, `mixer.rs` (gain/pan/
level math), `pcm_cache.rs` (storage; gains length-validation hooks), `transport.rs` (state
machine), `events.rs` (collector; payloads move to `maestro-api`), `stretch.rs` (Signalsmith
wrapper; add a preallocated-buffer process variant and the anchor helper; `process_separated`'s
per-call allocation is banned from the render path), `track.rs` (state machine + controls;
**delete** `playback_position`/`advance_playback_position` — positions are not state).

**Refactor:** `tempo.rs` → integer mBPM + the segment table (new `timeline.rs` replaces
`beat_timeline.rs`'s float accumulators); `engine.rs` facade → split into
`control_plane.rs` / `audio_core.rs` per §3.3.

**Retire from the native path:** `position.rs`, `beat_timeline.rs`, `beat_tracker.rs`
(subsumed by derived `bu(F)`), `fifo_buffer.rs`, `frame_reader.rs` (web-era streaming;
the web TS engine keeps its own copies until M4 deletes them together).

**Do not port from the branch:** `render_to_buffer` + cursor model, `render_bridge.rs` /
`render_ffi.rs` as timing authority (Plan-B raw material only), `uniffi_api.rs` and
`wasm_bindings.rs` mirrors, `NativeEventBridge.ts`, `NativeAudioAdapter.ts`,
`start_track_at_beat` and every beat-capture patch.

**Reuse from the branch:** metronome WAV fixtures (60/90/120), `detect_click_onsets`
(promote to a shared test util + the `maestro-validate` CLI), the Metronome Test song +
library entry, `AudioSessionManager.swift` (session/interruption plumbing; delegate now
issues commands), the AVAudioSourceNode setup in `AudioOutputBridge.swift` (rewired to
direct render), the SPSC discipline (as Plan B and for the command/event/reclaim queues),
ADR-088's harness methodology.

---

## 8. Non-goals and accepted risks

**Non-goals (this program):** Android delivery (the architecture is Oboe-compatible — same
C ABI render entry — but no Android milestone is scheduled here); offline export/recording;
> 16 simultaneous tracks; sub-block (< ~10 ms) control-action latency; networked/multi-device
sync; changing the UI.

**Accepted risks, eyes open:**

1. **Signalsmith CPU on the oldest device is unmeasured until M0.** Mitigations are
   pre-decided (global `new_cheaper()`, then Plan B §3.3); the budget gate is the very first
   checkpoint, so the exposure is days, not months.
2. **Asset metadata regeneration (per-file gapless values) may reveal mis-encoded files.**
   That is the system working — such files are exactly today's glitch sources; the pipeline
   fix (re-encode via the afconvert path) is mechanical.
3. **Stretch-path audio (not positions) is not bit-identical across architectures** (FMA in
   the C++ stretcher) and is nondeterministic past the clean-stretch ratio bound; §4.8
   clamps ranges and §6.6 scopes the parity claim to positions. Sync is unaffected.
4. **Capacitor bridge latency (~10–30 ms) for control feel** — accepted; quantized starts
   make it musically invisible. If product later demands tighter feel, the schema permits a
   binary fast-path without redesign.
5. **The web TS engine keeps its (small, sub-ms/hour) drift behaviors until M4.** Accepted:
   converging web onto the proven core is the cure; interim web behavior is today's status
   quo, unchanged.

---

## Appendix A — Reproduction evidence (abandoned branch engine, offline)

Method: `maestro-core/tests/drift_reproduction.rs` on `feature/milestone-5-render-stretch`
(run 2026-06-09, release build, x86/ARM-independent results). Synthetic stems built to real
metadata geometry; rendered through the branch's actual `render_to_buffer` in 480-frame
blocks; onsets measured with the branch's own detector logic.

**A.1 Inter-stem drift, same tempo, rate 1.0 (the user's "stems drift apart").**
Two click stems, both 116 BPM (as production: every stem of a song shares `trackBPM`), loop
lengths 2,085,517 (84 beats) and 397,241 (16 beats) — real-world non-ratio lengths. Engine
tempo 116 ⇒ rate 1.0, but (D6) both run the stretcher path, as on device. Result: pairwise
click offset **0 samples at start → −24 ± 1 samples after 10 minutes** (prediction from
integer-PCM-length wrapping: −24.2). Both stems also showed the uniform 2,880-frame
stretcher pre-roll (confirming deterministic latency). Linear, unbounded: ≈ −2.4 samples/min
for this pair; short loops drift proportionally faster; trim-induced length errors (D4)
scale it by orders of magnitude.

**A.2 Loop-point transient (the user's "frame lost at the loop point").**
A perfectly seamless integer-cycle sine loop at rate 0.75 renders **clean seams** (max
inter-sample step 0.0115 vs 0.0108 theoretical sine max; top-10 steps uncorrelated with wrap
positions). The same loop with 2,112 samples chopped from its start (simulating the device
double-trim, D4) produces a **0.29–0.35 step at every wrap** (30× the signal's natural
maximum), each landing at wrap ± the 2,880-frame pre-roll — an audible click at every loop.
Conclusion: the glitch is caused by discontinuous seam content (trim defects), smeared by
the vocoder; the fix is §3.4 (trim correctness) + §3.2 (streaming feed), and T-SEAM's
negative control encodes this exact experiment.

**Not verified here (deferred to M0 by design):** behavior on physical hardware — CPU
budget, underruns, session/route effects, and the end-to-end magnitude of D4 on real R2
assets. The first milestone exists to close exactly this gap with measurements, not ears.

---

## Appendix B — Traceability: dossier §5 → this spec

| Dossier item | Status | Where addressed |
|---|---|---|
| 5.1 free-running cursors | Confirmed (mechanism refined, §2.3#1) | §3.1 derived positions; T-INV negative control `float_cursor` |
| 5.2 loop-seam vocoder discontinuity | **Corrected**: content discontinuity, not modulo feeding (§2.3#2, A.2) | §3.4 trim policy; §3.2 streaming feed; T-SEAM |
| 5.3 rate-1.0 forced through stretcher | Confirmed | §3.2 unity bypass + crossfade transitions |
| 5.4 beat clock from render ticks | Confirmed | §3.1 single clock; no separate beat clock exists |
| 5.4 tempo-change desync | Confirmed | §3.1 segment table; T-INV tempo-change legs |
| 5.4 per-variation trim [HYPOTHESIS] | Confirmed structurally; refined (§2.3#5) | §3.4 per-file metadata + reconciliation; T-LOAD |
| 5.4 stretcher latency [HYPOTHESIS] | Refuted as risk: deterministic (§2.3#4) | §3.2 anchor procedure; §4.3 |
| 5.4 cross-tempo f64 rounding | **Corrected**: deterministic; real issue is the float model (§2.3#3) | §3.1 integer-only timing; §6.6 parity |
| 5.5 silent-stub milestones | Confirmed | §5 sequencing; §6.1 anti-stub policy |
| 5.6 false-confidence tests | Confirmed | §6.1 fixture choice (60/90), negative controls, threshold ADR rule |
| 5.7 timer as timing authority | Confirmed | §3.3 callback authority; Plan B keeps clock = rendered frames |
| 5.8 5× hand-written FFI | Confirmed | §3.6 one schema + codegen |
| 5.9 scattered state / coordinator fork | Confirmed | §3.5 Rust authority + snapshot; §3.7 AudioBackend |
| 5.10 event impedance layer | Confirmed | §3.6 generated event types; bridge = deserialize |
| 5.11 three engines / untested delivery | Confirmed | §3.8 convergence; §6.4 device harness gates |
| (new) non-integer musical loop lengths | **New finding** (§2.2, A.1) | §3.1 beat-space Bresenham wrap; T-INV `pcm_length_loop` control |
| (new) double-trim + hardcoded 2112/960 | **New finding** (§2.1 D4, A.2) | §3.4 reconciliation; T-LOAD; pipeline fix |
| (new) unconditional 48k resample path | **New finding** (D11) | §3.4 no-resample assertion; §4.7 |
