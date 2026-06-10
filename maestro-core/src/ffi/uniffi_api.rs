//! # UniFFI surface — the six-function transport (spec §3.6)
//!
//! The native bridge exports exactly the schema transport and nothing else:
//!
//! ```text
//! MaestroEngine::new(config_json)          // init
//! dispatch(command_json) -> result_json    // control plane, ≤ 60 Hz
//! snapshot_json() / drain_events_json()    // egress
//! load_pcm(meta_json, left, right)         // the one bulk-data call
//! core_pointer() -> u64                    // handed to the render callback
//! ```
//!
//! plus the C ABI `maestro_render` (declared in `include/maestro_render.h`),
//! which the AVAudioSourceNode render block calls directly with the
//! `core_pointer` value. No UDL, no mirror types: every payload is JSON in
//! the `maestro-api` schema, identical to the WASM transport (kills D12).
//!
//! ## Lifetime contract (enforced by the Swift shim)
//!
//! The audio thread dereferences `core_pointer` without synchronization (RT
//! rule: no locks). The shim must therefore stop the AVAudioEngine (and thus
//! the render callback) **before** releasing its `MaestroEngine` reference.
//! `AudioOutputBridge.stop()` does exactly that.

use std::sync::Arc;
// FFI-boundary exception to the workspace no-locks rule (clippy.toml): these
// Mutexes serialize *control-plane* calls arriving from Swift/Kotlin threads.
// The audio thread never touches them — it works through the raw core pointer
// and the lock-free SPSC queues only (spec §3.3).
#[allow(clippy::disallowed_types)]
use std::sync::Mutex;

use crate::audio_core::AudioCore;
use crate::control_plane::{create_engine, ControlPlane};
// Note: `uniffi::setup_scaffolding!()` lives in lib.rs (crate root requirement).

#[derive(Debug, uniffi::Error)]
#[uniffi(flat_error)]
pub enum MaestroFfiError {
    InvalidConfig(String),
}

impl std::fmt::Display for MaestroFfiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig(reason) => write!(f, "invalid config: {reason}"),
        }
    }
}

impl std::error::Error for MaestroFfiError {}

#[derive(uniffi::Object)]
pub struct MaestroEngine {
    #[allow(clippy::disallowed_types)]
    control: Mutex<ControlPlane>,
    /// Owns the audio core so its address stays stable for the lifetime of
    /// the engine object; the render callback reaches it via `core_pointer`.
    #[allow(clippy::disallowed_types)]
    core: Mutex<Box<AudioCore>>,
}

fn json_result(result: maestro_api::CommandResult) -> String {
    serde_json::to_string(&result).expect("CommandResult serializes")
}

fn rejected(reason: String) -> String {
    json_result(maestro_api::CommandResult::Rejected { reason })
}

#[uniffi::export]
impl MaestroEngine {
    /// `init` (spec §3.6): `config_json` is a `maestro-api` `EngineConfig`.
    #[uniffi::constructor]
    pub fn new(config_json: String) -> Result<Arc<Self>, MaestroFfiError> {
        let config: maestro_api::EngineConfig = serde_json::from_str(&config_json)
            .map_err(|e| MaestroFfiError::InvalidConfig(e.to_string()))?;
        let (control, core) =
            create_engine(config).map_err(|e| MaestroFfiError::InvalidConfig(e.to_string()))?;
        Ok(Arc::new(Self { control: Mutex::new(control), core: Mutex::new(core) }))
    }

    /// `dispatch`: a `maestro-api` `Command` in, a `CommandResult` out.
    pub fn dispatch(&self, command_json: String) -> String {
        let cmd: maestro_api::Command = match serde_json::from_str(&command_json) {
            Ok(cmd) => cmd,
            Err(e) => return rejected(format!("malformed command: {e}")),
        };
        let mut control = self.control.lock().expect("control plane lock poisoned");
        json_result(control.dispatch(cmd))
    }

    /// `snapshot_json`: the only state egress (spec §3.5).
    pub fn snapshot_json(&self) -> String {
        let control = self.control.lock().expect("control plane lock poisoned");
        serde_json::to_string(&control.snapshot()).expect("Snapshot serializes")
    }

    /// `drain_events_json`: edge events as a JSON array; also reclaims any
    /// resources the audio thread handed back.
    pub fn drain_events_json(&self) -> String {
        let mut control = self.control.lock().expect("control plane lock poisoned");
        serde_json::to_string(&control.drain_events()).expect("events serialize")
    }

    /// `load_pcm`: the one bulk-data call. `meta_json` is a `maestro-api`
    /// `LoadMeta`; `left`/`right` are the decoded 48 kHz channels (the shim
    /// asserts no resample / no channel conversion happened, spec §3.4#3).
    /// Returns a `CommandResult` JSON.
    pub fn load_pcm(&self, meta_json: String, left: Vec<f32>, right: Vec<f32>) -> String {
        let meta: maestro_api::LoadMeta = match serde_json::from_str(&meta_json) {
            Ok(meta) => meta,
            Err(e) => return rejected(format!("malformed load meta: {e}")),
        };
        let mut control = self.control.lock().expect("control plane lock poisoned");
        match control.load_pcm(&meta, left, right) {
            Ok(_) => json_result(maestro_api::CommandResult::Ok),
            Err(reason) => rejected(reason),
        }
    }

    /// The `AudioCore` pointer for the render callback. Pass to
    /// `maestro_render` (C ABI) from the audio thread only. See the module
    /// docs for the lifetime contract.
    pub fn core_pointer(&self) -> u64 {
        let mut core = self.core.lock().expect("core storage lock poisoned");
        core.as_mut() as *mut AudioCore as u64
    }
}
