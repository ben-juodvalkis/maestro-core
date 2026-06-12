//! # maestro-core
//!
//! Shared Rust audio engine for Maestro.
//! Compiles to native (.dylib / .so) and WASM (.wasm) targets.
//!
//! ## Module porting order (dependency-driven)
//!
//! 1. `timing` + `beat_timeline`    — pure math, zero deps
//! 2. `fifo_buffer` + `frame_reader` — core data structures
//! 3. `position` + `beat_tracker`   — depends on 1+2
//! 4. `tempo` + `pitch`             — control parameters
//! 5. `pcm_cache` + `track`         — track data model + cache
//! 6. `mixer` + `transport`         — mixing, state machine
//! 7. `events` + `engine`           — orchestration layer
//! 8. Signalsmith Stretch           — native C++ linkage
//! 9. WASM compilation target       — wasm-bindgen exports

// Native rebuild (documentation/specs/native-audio-rebuild.md):
// the single timing authority — derived integer positions, no advanced cursors.
pub mod timeline;

// Lock-free single-writer snapshot cell (spec §3.5).
pub mod seqlock;

// The two engine planes (spec §3.3). Audio core renders inside the OS audio
// callback (or the AudioWorklet quantum on web, M4); control plane validates
// and prepares on the caller's thread. Both need the Signalsmith C++
// stretcher — `native-stretch` now builds it for native AND wasm32 targets
// (the vendored crate compiles the same C++ via wasi-sdk, spec §3.8).
#[cfg(feature = "native-stretch")]
pub mod audio_core;
#[cfg(feature = "native-stretch")]
pub mod control_plane;

// Plan B delivery (spec §3.3): dedicated producer thread + SPSC sample ring.
// iOS-only delivery (ADR-090); never compiled for wasm32 — the web engine
// renders directly in `AudioWorkletProcessor.process()` (spec §3.8, no
// render pump on web) and wasm32 has no std::thread to pump with.
#[cfg(all(feature = "native-stretch", not(target_arch = "wasm32")))]
pub mod render_pump;

// C runtime shims (malloc/free, libm) for the wasm32-unknown-unknown
// shipping module — the Signalsmith C++'s libc needs (M4, spec §3.8).
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
pub mod wasm_runtime;

// T-INV negative-control sabotage modes (spec §6.2). Test builds only.
#[cfg(feature = "drift-injection")]
pub mod sabotage;

// Shared analysis utilities (WAV loading, onset detection) for the offline
// suites and the maestro-validate device-capture analyzer (spec §6.4).
pub mod validation;

// Step 1: Pure math — timing conversions and beat timeline
pub mod timing;
pub mod beat_timeline;

// Step 2: Core data structures — FIFO buffer and frame reader
pub mod fifo_buffer;
pub mod frame_reader;

// Step 3: Position calculation and beat tracking (depends on timing + frame reader)
pub mod position;
pub mod beat_tracker;

// Step 4: Control parameters — tempo and pitch
pub mod tempo;
pub mod pitch;

// Step 5: Track data model + PCM cache
pub mod pcm_cache;
pub mod track;

// Step 6: Mixing and transport state machine
pub mod mixer;
pub mod transport;

// Step 7: Orchestration layer — events and engine
pub mod events;
pub mod engine;

// Step 8: Signalsmith Stretch — native C++ time-stretching and pitch-shifting
// Feature-gated: `native-stretch` for native targets, stub for WASM
pub mod stretch;

// Step 9: Foreign Function Interface — wasm-bindgen exports (web) and the
// UniFFI six-function transport (native, spec §3.6)
pub mod ffi;

// UniFFI scaffolding must be defined at the crate root.
#[cfg(feature = "uniffi-bindings")]
uniffi::setup_scaffolding!();
