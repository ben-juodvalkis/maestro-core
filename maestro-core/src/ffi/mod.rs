//! Foreign Function Interface (FFI) bindings for maestro-core.
//!
//! ## Targets
//!
//! - **WASM** (`wasm` feature): `wasm-bindgen` exports for web consumption.
//!   See [`wasm_bindings`] for the JavaScript API.
//!
//! - **UniFFI** (`uniffi-bindings` feature): the six-function native
//!   transport (spec §3.6). See [`uniffi_api`].

#[cfg(feature = "wasm")]
pub mod wasm_bindings;

#[cfg(feature = "uniffi-bindings")]
pub mod uniffi_api;
