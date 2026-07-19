//! # maestro-core
//!
//! Shared Rust audio engine for Maestro.
//! Compiles to native (.dylib / .so) and WASM (.wasm) targets.
//!
//! The native-rebuild architecture (documentation/specs/native-audio-rebuild.md):
//! `timeline` (derived integer positions) + `audio_core` (audio thread) +
//! `control_plane` (validation/prep) behind the transports
//! (`ffi::uniffi_api` native, `ffi::wasm_bindings` web).

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

// Engine-resident PCM behind a monotone load frontier (ADR-112 §5.3) — the
// one representation for bulk-loaded AND streaming track playback buffers.
pub mod engine_pcm;

// The Signalsmith C++ stretcher wrapper (native + wasm32 via the vendored
// crate, spec §3.8). The legacy step-1..7 modules of the abandoned-roadmap
// port (engine.rs facade, float beat timeline, FIFO frame readers, …) were
// deleted with the legacy TS engine (ADR-095) — the §3 architecture in
// timeline/audio_core/control_plane is the engine.
pub mod stretch;

// Step 9: Foreign Function Interface — wasm-bindgen exports (web) and the
// UniFFI transport (native, spec §3.6)
pub mod ffi;

// UniFFI scaffolding must be defined at the crate root.
#[cfg(feature = "uniffi-bindings")]
uniffi::setup_scaffolding!();

// ---------------------------------------------------------------------------
// D8 build-slice canaries (spec §5 M5, ADR-096). The abandoned branch
// shipped a debug-only xcframework for months because nothing STRUCTURAL
// prevented it. These symbols exist only in the slices that must never
// ship: the packaging script (`maestro-core/scripts/build-ios.sh`), the
// plugin pod's release-guard script phase, and the CI release pipeline all
// `nm`-grep for them — a debug slice cannot be packaged or linked into a
// Release app build, and the published artifact can carry neither debug
// asserts nor the sabotage surface.
// ---------------------------------------------------------------------------

/// Present only when compiled with debug assertions (a debug/dev profile).
#[cfg(debug_assertions)]
#[no_mangle]
pub static MAESTRO_DEBUG_BUILD_CANARY: u8 = 1;

/// Present only in drift-injection (sabotage) builds — valid for local
/// negative-control device runs (spec §6.4), never for a published artifact.
#[cfg(feature = "drift-injection")]
#[no_mangle]
pub static MAESTRO_DRIFT_INJECTION_CANARY: u8 = 1;
