//! Foreign Function Interface (FFI) bindings for maestro-core.
//!
//! ## Targets
//!
//! - **WASM** (`wasm` feature): `wasm-bindgen` exports for web consumption.
//!   See [`wasm_bindings`] for the JavaScript API.
//!
//! - **UniFFI** (future): Swift/Kotlin bindings for Capacitor plugin.
//!   Will be added in a later step.

#[cfg(feature = "wasm")]
pub mod wasm_bindings;
