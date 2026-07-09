// FR2 schema fingerprint (ADR-100): a mechanical hash over the committed
// ts-rs bindings under `maestro-api/bindings/` — the schema surface both
// transports carry as JSON. The engine embeds it at ITS build time
// (build.rs → `maestro_api::SCHEMA_FINGERPRINT`); gen-ts writes the same
// value into `bindings/fingerprint.ts` for the app bundle; the backends
// assert equality at init. Skew between a committed engine artifact and the
// TS that drives it is therefore a hard init failure, never a silent
// reinterpretation (the ADR-098 park-as-swap bug class, PRD §2).
//
// This file is `include!`-free plain std so build.rs can share it via
// `#[path]` — it must not reference other crate items. Plain `//` comments
// only (an inner `//!` doc would break the build-script inclusion).

use std::fs;
use std::io;
use std::path::Path;

/// Magic prefix. The full string lands verbatim in every engine artifact's
/// data section, so the FR3 freshness gate can extract it with a binary grep
/// — no need to instantiate the module.
pub const SCHEMA_FINGERPRINT_PREFIX: &str = "maestro-schema-fp:v1:";

/// The generated TS module carrying the fingerprint — excluded from its own
/// hash (it is derived from the others, not part of the schema surface).
pub const FINGERPRINT_BINDING_FILE: &str = "fingerprint.ts";

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn absorb(hash: u64, bytes: &[u8]) -> u64 {
    bytes
        .iter()
        .fold(hash, |h, &b| (h ^ u64::from(b)).wrapping_mul(FNV_PRIME))
}

/// FNV-1a 64 over `(name, content)` pairs, each absorbed as
/// `name \0 content \0`. Callers pass the pairs already sorted by name.
/// Not cryptographic — it distinguishes schema versions across honest
/// builds; it does not defend against an adversary.
pub fn fingerprint_files(files: &[(String, Vec<u8>)]) -> String {
    let mut hash = FNV_OFFSET;
    for (name, content) in files {
        hash = absorb(hash, name.as_bytes());
        hash = absorb(hash, &[0]);
        hash = absorb(hash, content);
        hash = absorb(hash, &[0]);
    }
    format!("{SCHEMA_FINGERPRINT_PREFIX}{hash:016x}")
}

/// Fingerprint the committed bindings directory: every `.ts` file except
/// [`FINGERPRINT_BINDING_FILE`], sorted by filename. Errors if the directory
/// is unreadable or holds no binding files — a fingerprint over nothing
/// would green-light exactly the skew this exists to catch.
pub fn fingerprint_bindings(dir: &Path) -> io::Result<String> {
    let mut names: Vec<String> = fs::read_dir(dir)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .filter(|name| name.ends_with(".ts") && name != FINGERPRINT_BINDING_FILE)
        .collect();
    names.sort();
    if names.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("no .ts binding files in {}", dir.display()),
        ));
    }
    let mut files = Vec::with_capacity(names.len());
    for name in names {
        let content = fs::read(dir.join(&name))?;
        files.push((name, content));
    }
    Ok(fingerprint_files(&files))
}
