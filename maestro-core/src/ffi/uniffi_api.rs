//! # UniFFI surface — the transport (spec §3.6)
//!
//! The native bridge exports exactly the schema transport and nothing else:
//!
//! ```text
//! MaestroEngine::new(config_json)          // init
//! dispatch(command_json) -> result_json    // control plane, ≤ 60 Hz
//! snapshot_json() / drain_events_json()    // egress
//! load_pcm(meta_json, left, right)         // bulk-data lane
//! core_pointer() -> u64                    // handed to the render callback
//!
//! // the same lane, chunked (ADR-112) — NOT a second kind of lane:
//! begin_pcm_stream(meta_json)              // StreamLoadMeta; validates, installs
//! append_pcm_chunk(slot, seq, left, right) // -> AppendResult { frontier }
//! finalize_pcm_stream(slot)                // TrackLoaded fires here
//! abort_pcm_stream(slot)
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
use crate::render_pump::RenderPump;
// Note: `uniffi::setup_scaffolding!()` lives in lib.rs (crate root requirement).

/// How the engine is being driven (spec §3.3): direct render in the OS
/// callback, or Plan B (producer thread + ring) once `start_pump` is called.
enum Runtime {
    Direct(Box<AudioCore>),
    Pumped(RenderPump),
    /// Momentary state inside `start_pump` while the core moves between
    /// owners; never observable through the public surface.
    Transitioning,
}

/// FR2 (ADR-100): the maestro-api schema fingerprint this engine binary was
/// built against, embedded at build time. Surfaced by the plugin's additive
/// `getSchemaFingerprint`; `NativeBackend` hard-fails init on mismatch or
/// absence — a stale MaestroCore.xcframework must never silently
/// reinterpret the schema (the ADR-098 park-as-swap bug class, PRD §2).
/// Read-only; never touches the engine.
#[uniffi::export]
pub fn schema_fingerprint() -> String {
    maestro_api::SCHEMA_FINGERPRINT.to_string()
}

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
    /// Owns the engine driver; addresses handed to the render callback stay
    /// stable for the lifetime of the engine object.
    #[allow(clippy::disallowed_types)]
    runtime: Mutex<Runtime>,
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
        Ok(Arc::new(Self {
            control: Mutex::new(control),
            runtime: Mutex::new(Runtime::Direct(core)),
        }))
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

    /// `begin_pcm_stream` (ADR-112 §5.2): `meta_json` is a `maestro-api`
    /// `StreamLoadMeta`; mirrors the wasm transport 1:1.
    pub fn begin_pcm_stream(&self, meta_json: String) -> String {
        let smeta: maestro_api::StreamLoadMeta = match serde_json::from_str(&meta_json) {
            Ok(m) => m,
            Err(e) => return rejected(format!("malformed stream meta: {e}")),
        };
        let mut control = self.control.lock().expect("control plane lock poisoned");
        match control.begin_pcm_stream(&smeta) {
            Ok(()) => json_result(maestro_api::CommandResult::Ok),
            Err(reason) => rejected(reason),
        }
    }

    /// `append_pcm_chunk` (ADR-112 §5.2): returns an `AppendResult` JSON;
    /// the audio thread never touches the control-plane lock.
    pub fn append_pcm_chunk(&self, slot: u8, seq: u32, left: Vec<f32>, right: Vec<f32>) -> String {
        let mut control = self.control.lock().expect("control plane lock poisoned");
        serde_json::to_string(&control.append_pcm_chunk(slot, seq, &left, &right))
            .expect("AppendResult serializes")
    }

    /// `finalize_pcm_stream` (ADR-112 §5.5).
    pub fn finalize_pcm_stream(&self, slot: u8) -> String {
        let mut control = self.control.lock().expect("control plane lock poisoned");
        match control.finalize_pcm_stream(slot) {
            Ok(_) => json_result(maestro_api::CommandResult::Ok),
            Err(reason) => rejected(reason),
        }
    }

    /// `abort_pcm_stream` (ADR-112 §5.5).
    pub fn abort_pcm_stream(&self, slot: u8) -> String {
        let mut control = self.control.lock().expect("control plane lock poisoned");
        match control.abort_pcm_stream(slot) {
            Ok(()) => json_result(maestro_api::CommandResult::Ok),
            Err(reason) => rejected(reason),
        }
    }

    /// The `AudioCore` pointer for direct rendering (`maestro_render`).
    /// Returns 0 once the pump has started — the two modes are exclusive.
    pub fn core_pointer(&self) -> u64 {
        let mut runtime = self.runtime.lock().expect("runtime lock poisoned");
        match &mut *runtime {
            Runtime::Direct(core) => core.as_mut() as *mut AudioCore as u64,
            _ => 0,
        }
    }

    /// Switch to Plan B delivery (spec §3.3): move the audio core onto a
    /// dedicated producer thread feeding an SPSC ring. Idempotent. After this,
    /// the render callback must use `maestro_pump_consume(pumpPointer(), …)`.
    pub fn start_pump(&self) {
        let mut runtime = self.runtime.lock().expect("runtime lock poisoned");
        let prev = std::mem::replace(&mut *runtime, Runtime::Transitioning);
        *runtime = match prev {
            Runtime::Direct(core) => Runtime::Pumped(RenderPump::start(core)),
            other => other,
        };
    }

    /// The `PumpConsumer` pointer for the Plan B render callback
    /// (`maestro_pump_consume`). Returns 0 unless the pump is running. Same
    /// lifetime contract as `core_pointer`: stop the AVAudioEngine before
    /// releasing the engine reference.
    pub fn pump_pointer(&self) -> u64 {
        let mut runtime = self.runtime.lock().expect("runtime lock poisoned");
        match &mut *runtime {
            Runtime::Pumped(pump) => pump.consumer_ptr() as u64,
            _ => 0,
        }
    }
}
