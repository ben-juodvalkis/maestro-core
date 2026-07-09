//! Embeds the FR2 schema fingerprint (ADR-100) at build time: a hash over
//! the committed `bindings/` content (which the CI regen gate keeps honest),
//! exposed as `maestro_api::SCHEMA_FINGERPRINT`. Every engine artifact —
//! wasm and native — carries it; both backends assert it against the app
//! bundle's `bindings/fingerprint.ts` at init, so a stale committed engine
//! artifact is a hard init failure (PRD §2: the pre-ADR-098 wasm silently
//! reinterpreted prewarm parks as variation swaps).

// The hashing code is shared verbatim with the crate (gen-ts recomputes the
// same value for fingerprint.ts); `#[path]` keeps one implementation.
#[path = "src/fingerprint.rs"]
mod fingerprint;

fn main() {
    println!("cargo:rerun-if-changed=bindings");
    println!("cargo:rerun-if-changed=src/fingerprint.rs");
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR");
    let bindings = std::path::Path::new(&manifest_dir).join("bindings");
    let fp = fingerprint::fingerprint_bindings(&bindings).unwrap_or_else(|e| {
        panic!(
            "cannot fingerprint {} (FR2, ADR-100): {e} — the committed ts-rs \
             bindings are the schema surface; regenerate with \
             `cargo run -p maestro-api --features ts --bin gen-ts`",
            bindings.display()
        )
    });
    println!("cargo:rustc-env=MAESTRO_SCHEMA_FINGERPRINT={fp}");
}
