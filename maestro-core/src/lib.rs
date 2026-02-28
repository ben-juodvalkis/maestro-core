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
// pub mod tempo;
// pub mod pitch;
// pub mod pcm_cache;
// pub mod track;
// pub mod mixer;
// pub mod transport;
// pub mod events;
// pub mod engine;
// pub mod music_library;
// pub mod ffi;
