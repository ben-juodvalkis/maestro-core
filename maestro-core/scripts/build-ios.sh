#!/usr/bin/env bash
#
# Build maestro-core for iOS, generate Swift bindings (UniFFI library mode),
# and package MaestroCore.xcframework for the Capacitor plugin.
# (native-audio-rebuild spec §3.6; adapted from the abandoned branch's script
# per the §7 reuse map — workspace target dir, six-function surface.)
#
# Usage:
#   ./scripts/build-ios.sh                 # Release build (the only artifact
#                                          # the harness/app may ship — debug
#                                          # delivery is structurally banned)
#   ./scripts/build-ios.sh debug           # local development only
#   SABOTAGE=1 ./scripts/build-ios.sh      # drift-injection build for the
#                                          # §6.4 analyzer negative controls
#
# Output:
#   capacitor-plugin-maestro-audio/MaestroCore.xcframework/
#   capacitor-plugin-maestro-audio/ios/Sources/MaestroAudioPlugin/maestro_core.swift

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$CORE_DIR/.." && pwd)"
PLUGIN_DIR="$REPO_DIR/capacitor-plugin-maestro-audio"
BINDGEN_DIR="$CORE_DIR/uniffi-bindgen"
BUILD_DIR="$REPO_DIR/target" # workspace-level target dir
BINDINGS_DIR="$BUILD_DIR/swift-bindings"

PROFILE="${1:-release}"
if [ "$PROFILE" = "debug" ]; then
    CARGO_FLAGS=""
    PROFILE_DIR="debug"
else
    CARGO_FLAGS="--release"
    PROFILE_DIR="release"
fi

FEATURES="uniffi-bindings"
if [ "${SABOTAGE:-0}" = "1" ]; then
    FEATURES="uniffi-bindings,drift-injection"
    echo "!!! SABOTAGE build (drift-injection) — negative-control use only !!!"
fi

echo "=== Building maestro-core for iOS ($PROFILE, features: $FEATURES) ==="

cd "$REPO_DIR"

echo "--- Step 1: iOS targets ---"
# Match the app's deployment target so the static lib links without
# min-version warnings (the SDK default would stamp the current iOS version).
export IPHONEOS_DEPLOYMENT_TARGET=14.0
cargo build $CARGO_FLAGS -p maestro-core --target aarch64-apple-ios --features "$FEATURES"

# signalsmith-stretch's bindgen needs help with the -sim triple.
SIM_SDK=$(xcrun --sdk iphonesimulator --show-sdk-path)
BINDGEN_EXTRA_CLANG_ARGS="--target=arm64-apple-ios-simulator --sysroot=$SIM_SDK" \
    cargo build $CARGO_FLAGS -p maestro-core --target aarch64-apple-ios-sim --features "$FEATURES"

echo "--- Step 2: host dylib for binding generation ---"
cargo build -p maestro-core --features "$FEATURES"

echo "--- Step 3: Swift bindings (library mode) ---"
rm -rf "$BINDINGS_DIR"
mkdir -p "$BINDINGS_DIR"
(cd "$BINDGEN_DIR" && cargo run -q -- generate \
    --library "$BUILD_DIR/debug/libmaestro_core.dylib" \
    --language swift \
    --out-dir "$BINDINGS_DIR")
mv "$BINDINGS_DIR/maestro_coreFFI.modulemap" "$BINDINGS_DIR/module.modulemap"

echo "--- Step 4: XCFramework ---"
XCFRAMEWORK_PATH="$PLUGIN_DIR/MaestroCore.xcframework"
rm -rf "$XCFRAMEWORK_PATH"

DEVICE_HEADERS="$BUILD_DIR/headers-ios-device"
SIM_HEADERS="$BUILD_DIR/headers-ios-sim"
rm -rf "$DEVICE_HEADERS" "$SIM_HEADERS"
mkdir -p "$DEVICE_HEADERS" "$SIM_HEADERS"

for H in "$DEVICE_HEADERS" "$SIM_HEADERS"; do
    cp "$BINDINGS_DIR/maestro_coreFFI.h" "$H/"
    cp "$BINDINGS_DIR/module.modulemap" "$H/"
    cp "$CORE_DIR/include/maestro_render.h" "$H/"
    # Expose the raw render entry alongside the UniFFI header.
    sed -i '' 's|header "maestro_coreFFI.h"|header "maestro_coreFFI.h"\n    header "maestro_render.h"|' "$H/module.modulemap"
done

# D8 release-slice guard (spec §5 M5, ADR-096): a release packaging run must
# contain no debug slice — the canary symbol exists only under
# debug_assertions, so even a profile override (e.g. a stray
# CARGO_PROFILE_RELEASE_DEBUG_ASSERTIONS) is caught here, not on a device
# months later. Sabotage (drift-injection) slices stay legitimate for local
# negative-control runs; the CI release pipeline rejects BOTH canaries
# before publishing.
if [ "$PROFILE_DIR" = "release" ]; then
    for LIB in "$BUILD_DIR/aarch64-apple-ios/$PROFILE_DIR/libmaestro_core.a" \
               "$BUILD_DIR/aarch64-apple-ios-sim/$PROFILE_DIR/libmaestro_core.a"; do
        if nm "$LIB" 2>/dev/null | grep -q MAESTRO_DEBUG_BUILD_CANARY; then
            echo "ERROR: $LIB contains MAESTRO_DEBUG_BUILD_CANARY — a debug slice in a"
            echo "release packaging run (spec D8 / ADR-096). Check for profile overrides."
            exit 1
        fi
    done
fi

xcodebuild -create-xcframework \
    -library "$BUILD_DIR/aarch64-apple-ios/$PROFILE_DIR/libmaestro_core.a" \
    -headers "$DEVICE_HEADERS" \
    -library "$BUILD_DIR/aarch64-apple-ios-sim/$PROFILE_DIR/libmaestro_core.a" \
    -headers "$SIM_HEADERS" \
    -output "$XCFRAMEWORK_PATH"

# Provenance stamp (read back by the CI release pipeline and by humans
# wondering what a device is running).
cat > "$XCFRAMEWORK_PATH/BUILD_INFO" << INFO
profile=$PROFILE_DIR
features=$FEATURES
git_rev=$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || echo unknown)
rustc=$(rustc --version)
built_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
INFO

echo "--- Step 5: Swift bindings → plugin sources ---"
SWIFT_DEST="$PLUGIN_DIR/ios/Sources/MaestroAudioPlugin"
mkdir -p "$SWIFT_DEST"
cp "$BINDINGS_DIR/maestro_core.swift" "$SWIFT_DEST/"

echo "=== Done: $XCFRAMEWORK_PATH ($PROFILE, $FEATURES) ==="
