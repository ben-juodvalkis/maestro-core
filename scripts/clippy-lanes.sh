#!/usr/bin/env bash
# The four clippy lanes of the Rust validation suite — ONE definition, two
# callers: `.github/workflows/rust-core.yml` (the "Clippy (RT lints, deny
# warnings)" step) and `scripts/validate-local.sh`. They used to be four
# literal lines inside the workflow, which meant a local run could only match
# CI by transcribing them; anything transcribed drifts.
#
# Usage: scripts/clippy-lanes.sh [--list]
#   --list  print the lanes (one per line, `name<TAB>args…`) and exit 0
#           without compiling anything.
#
# Every lane is `-D warnings`: clippy.toml carries the RT-safety lints (no
# allocation, no locks on the audio thread), and a warning there is a real
# RT hazard, not style.
#
# WHY FOUR — each lane compiles code the others cannot see:
#
#   1. default            The ordinary surface: workspace + all targets.
#
#   2. drift-injection    The sabotage builds behind maestro-core's
#                         drift-injection feature. They are only compiled
#                         when the feature is on, so lane 1 never lints the
#                         negative controls that the T-INV suite depends on.
#
#   3. all-features       Both FFI surfaces are feature-gated OFF by default
#                         (ffi/mod.rs), so lanes 1–2 never compile
#                         ffi/uniffi_api.rs or ffi/wasm_bindings.rs: the
#                         no-locks lint was blind to precisely the code most
#                         likely to need a lock, and an unannotated Mutex sat
#                         in the UniFFI constructor unflagged. --all-features
#                         rather than an explicit --features list so that a
#                         future feature-gated module is linted the day it
#                         lands, not the day someone remembers to extend this
#                         list.
#
#   4. no-default-features
#                         --all-features can only turn features ON, so no
#                         lane above builds maestro-core with native-stretch
#                         OFF — and that configuration is NOT hypothetical:
#                         maestro-validate declares
#                         `maestro-core = { ..., default-features = false }`,
#                         so every `cargo run -p maestro-validate` (in
#                         rust-core.yml and in device-validation.sh /
#                         device-soak.sh / c3-ui-validation.sh) builds the
#                         crate without the Signalsmith C++. This lane gates
#                         that build. It catches an ungated module acquiring
#                         a dependency on stretch-gated code, which would
#                         break the analyzer while all three lanes above
#                         stayed green; negative control in ADR-114. Keeping
#                         native-stretch optional is what keeps this check at
#                         ~8 s instead of a full C++ build.
#                         No --all-targets on purpose — the integration
#                         suites legitimately assume the real stretcher and
#                         do not compile without it; the lib is the target.
set -euo pipefail
cd "$(dirname "$0")/.."

# name<TAB>cargo-clippy arguments (before the `-- -D warnings` separator).
LANES=(
  $'default\t--workspace --all-targets'
  $'drift-injection\t--workspace --all-targets --features maestro-core/drift-injection'
  $'all-features\t--workspace --all-targets --all-features'
  $'no-default-features\t--workspace --no-default-features'
)

if [[ "${1:-}" == "--list" ]]; then
  printf '%s\n' "${LANES[@]}"
  exit 0
fi

for lane in "${LANES[@]}"; do
  name="${lane%%$'\t'*}"
  args="${lane#*$'\t'}"
  echo "--- clippy lane: $name (cargo clippy $args -- -D warnings)"
  # shellcheck disable=SC2086 # $args is an intentional argument list.
  cargo clippy $args -- -D warnings
done

echo "clippy: all ${#LANES[@]} lanes clean"
