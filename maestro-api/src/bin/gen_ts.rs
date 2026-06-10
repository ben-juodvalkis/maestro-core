//! Regenerates the committed TypeScript bindings under `maestro-api/bindings/`.
//!
//! CI runs `cargo run -p maestro-api --bin gen-ts && git diff --exit-code` so a
//! schema change that isn't reflected in the committed bindings fails the build
//! (spec §3.6 — binding drift is a build error).

use ts_rs::TS;

fn main() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/bindings");
    maestro_api::EngineConfig::export_all_to(dir).unwrap();
    maestro_api::Command::export_all_to(dir).unwrap();
    maestro_api::CommandResult::export_all_to(dir).unwrap();
    maestro_api::LoadMeta::export_all_to(dir).unwrap();
    maestro_api::Event::export_all_to(dir).unwrap();
    maestro_api::Snapshot::export_all_to(dir).unwrap();
    println!("TypeScript bindings written to {dir}");
}
