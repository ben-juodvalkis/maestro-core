//! # wasm-bindgen surface — the six-function transport (spec §3.6, M4)
//!
//! The web transport is IDENTICAL in shape to the UniFFI one
//! ([`super::uniffi_api`]): exactly the schema transport, nothing else.
//! Every payload is JSON in the `maestro-api` schema; no mirror types exist
//! here (the 1,168-line wasm mirror this file used to be is the D12 defect
//! class — spec §3.6 kills it).
//!
//! ```text
//! new WasmMaestroEngine(config_json)        // init
//! dispatch(command_json) -> result_json     // control plane, ≤ 60 Hz
//! snapshot_json() / drain_events_json()     // egress
//! load_pcm(meta_json, left, right)          // the one bulk-data call
//! render(frames) + left_ptr()/right_ptr()   // audio thread only
//! ```
//!
//! ## Threading model (spec §3.8)
//!
//! The whole engine lives INSIDE the `AudioWorkletProcessor`: the worklet
//! thread is the audio thread, `process()` calls [`WasmMaestroEngine::render`]
//! directly (no render pump — Plan B is iOS delivery, ADR-090), and `F`
//! advances per 128-frame quantum. Commands and events cross from the main
//! thread via MessagePort into the worklet's `dispatch`/`drain_events_json`,
//! which run on the SAME thread between render quanta — so unlike the UniFFI
//! surface there are no locks here at all, and a full command queue rejects
//! instead of blocking (see `ControlPlane::send`).
//!
//! ## Output / snapshot transfer
//!
//! `render()` writes into engine-owned preallocated buffers; the worklet
//! views them through `left_ptr()`/`right_ptr()` (`Float32Array` over wasm
//! memory — recreate the view per quantum, memory growth invalidates it) and
//! copies into the `process()` outputs. `write_snapshot_json()` +
//! `snapshot_buf_ptr()` expose the snapshot as UTF-8 bytes in wasm memory
//! for the SharedArrayBuffer seqlock lane — the AudioWorkletGlobalScope has
//! no TextEncoder, so the bytes go main-thread-ward raw and are decoded
//! there (spec §3.8; port-posted JSON strings are the fallback lane).

use wasm_bindgen::prelude::*;

use crate::audio_core::{AudioCore, MAX_BLOCK_FRAMES};
use crate::control_plane::{create_engine, ControlPlane};

fn json_result(result: maestro_api::CommandResult) -> String {
    serde_json::to_string(&result).expect("CommandResult serializes")
}

fn rejected(reason: String) -> String {
    json_result(maestro_api::CommandResult::Rejected { reason })
}

/// The engine pair (control plane + audio core) behind the wasm transport.
/// Owned by the worklet; `&mut self` exclusivity is enforced by wasm-bindgen
/// at the boundary and everything runs on the one worklet thread.
#[wasm_bindgen]
pub struct WasmMaestroEngine {
    control: ControlPlane,
    core: Box<AudioCore>,
    /// Preallocated render output (non-interleaved), `MAX_BLOCK_FRAMES` each:
    /// the render path allocates nothing (spec §3.3).
    out_left: Vec<f32>,
    out_right: Vec<f32>,
    /// Retained snapshot JSON for the SAB byte lane.
    snapshot_buf: String,
}

#[wasm_bindgen]
impl WasmMaestroEngine {
    /// `init` (spec §3.6): `config_json` is a `maestro-api` `EngineConfig`.
    #[wasm_bindgen(constructor)]
    pub fn new(config_json: &str) -> Result<WasmMaestroEngine, JsError> {
        let config: maestro_api::EngineConfig = serde_json::from_str(config_json)
            .map_err(|e| JsError::new(&format!("invalid config: {e}")))?;
        let (control, core) =
            create_engine(config).map_err(|e| JsError::new(&format!("invalid config: {e}")))?;
        Ok(Self {
            control,
            core,
            out_left: vec![0.0; MAX_BLOCK_FRAMES],
            out_right: vec![0.0; MAX_BLOCK_FRAMES],
            snapshot_buf: String::new(),
        })
    }

    /// `dispatch`: a `maestro-api` `Command` in, a `CommandResult` out.
    pub fn dispatch(&mut self, command_json: &str) -> String {
        let cmd: maestro_api::Command = match serde_json::from_str(command_json) {
            Ok(cmd) => cmd,
            Err(e) => return rejected(format!("malformed command: {e}")),
        };
        json_result(self.control.dispatch(cmd))
    }

    /// `snapshot_json`: the only state egress (spec §3.5).
    pub fn snapshot_json(&self) -> String {
        serde_json::to_string(&self.control.snapshot()).expect("Snapshot serializes")
    }

    /// `drain_events_json`: edge events as a JSON array; also reclaims any
    /// resources the audio thread handed back.
    pub fn drain_events_json(&mut self) -> String {
        serde_json::to_string(&self.control.drain_events()).expect("events serialize")
    }

    /// `load_pcm`: the one bulk-data call. `meta_json` is a `maestro-api`
    /// `LoadMeta`; `left`/`right` are the `decodeAudioData` channel buffers
    /// at 48 kHz (the backend asserts no resample / no channel conversion
    /// happened, spec §3.4#3). Returns a `CommandResult` JSON.
    pub fn load_pcm(&mut self, meta_json: &str, left: Vec<f32>, right: Vec<f32>) -> String {
        let meta: maestro_api::LoadMeta = match serde_json::from_str(meta_json) {
            Ok(meta) => meta,
            Err(e) => return rejected(format!("malformed load meta: {e}")),
        };
        match self.control.load_pcm(&meta, left, right) {
            Ok(_) => json_result(maestro_api::CommandResult::Ok),
            Err(reason) => rejected(reason),
        }
    }

    // ------------------------------------------------------------ audio thread

    /// Render one block into the engine-owned output buffers. Called from
    /// `AudioWorkletProcessor.process()` — the worklet thread IS the audio
    /// thread (spec §3.8); `F` advances by exactly `frames`.
    pub fn render(&mut self, frames: usize) {
        let n = frames.min(MAX_BLOCK_FRAMES);
        self.core.render(&mut self.out_left[..n], &mut self.out_right[..n]);
    }

    /// Pointer (into wasm linear memory) to the left output of the last
    /// `render`. View with `new Float32Array(memory.buffer, ptr, frames)` —
    /// recreate the view every quantum (memory growth detaches buffers).
    pub fn left_ptr(&self) -> *const f32 {
        self.out_left.as_ptr()
    }

    /// Right-channel counterpart of [`Self::left_ptr`].
    pub fn right_ptr(&self) -> *const f32 {
        self.out_right.as_ptr()
    }

    // ------------------------------------------------------------ SAB lane

    /// Serialize the current snapshot into the retained buffer and return its
    /// byte length. The worklet copies `snapshot_buf_ptr()[..len]` into the
    /// SharedArrayBuffer under a seq counter; the main thread TextDecodes +
    /// parses (spec §3.8 SAB seqlock lane; no TextEncoder exists in the
    /// worklet scope).
    pub fn write_snapshot_json(&mut self) -> usize {
        self.snapshot_buf.clear();
        // Snapshot is small (≤ 16 tracks of POD); reuse the String's capacity.
        let json = serde_json::to_string(&self.control.snapshot()).expect("Snapshot serializes");
        self.snapshot_buf.push_str(&json);
        self.snapshot_buf.len()
    }

    /// Pointer to the bytes written by the last [`Self::write_snapshot_json`].
    pub fn snapshot_buf_ptr(&self) -> *const u8 {
        self.snapshot_buf.as_ptr()
    }
}
