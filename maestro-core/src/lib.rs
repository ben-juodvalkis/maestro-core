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

// pub mod music_library;
// pub mod ffi;
