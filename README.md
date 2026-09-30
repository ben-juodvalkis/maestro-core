# maestro-core

A multi-track loop engine in Rust, built so that tracks **cannot drift apart**: not
after an hour, not across tempo changes, not across loop wraps. It is the audio engine
behind **Maestro Move**, a stem player that plays several stems of a
song in sync, with live tempo and pitch control and per-stem variations.

One engine runs on both platforms. On the web it runs as WebAssembly inside an
AudioWorklet; on iOS it is an XCFramework called through UniFFI.

## The idea

Most loop players move a floating-point playhead forward one block at a time. Every
block adds a tiny rounding error, and over time the tracks separate.

This engine keeps no running playhead. For every output frame, the sample each track
should be playing is **computed from scratch**: it is a pure integer function of the
frame number and the history of commands (tempo changes, loop lengths). Because nothing
accumulates, there is nothing to drift. The time-stretcher is never asked where a track
is. It is told which span of source audio to consume.

The full design is in [`documentation/specs/native-audio-rebuild.md`](documentation/specs/native-audio-rebuild.md).
Code comments cite it as "spec §N".

## What the tests prove

- **T-INV:** every frame's position is checked bit-for-bit against an exact
  rational-arithmetic reference, and the result is digested. The digest must be
  **identical on x86-64 and wasm32**.
- **Negative controls:** a `drift-injection` feature deliberately breaks the engine
  (float cursor, PCM-length looping, timer clock). The suite must *catch* each one; a
  test that can't detect sabotage doesn't count.
- **T-SEAM, T-CTRL, T-LOAD, T-STREAM:** loop seams, control changes, decoded-file
  length reconciliation, and streaming loads that start before the whole file arrives.
- **Real-time safety:** a test proves the render loop allocates nothing, and a lint
  (`clippy.toml`) bans locks anywhere in the engine.

## Layout

| Path | What it is |
|---|---|
| `maestro-core/` | The engine: timeline, audio core, control plane, WASM and UniFFI bindings |
| `maestro-api/` | The command/event/snapshot schema, with generated TypeScript bindings |
| `maestro-validate/` | Analyzer for captured device audio: measures cross-stem click sync |
| `maestro-core/vendor/signalsmith-stretch/` | [Signalsmith Stretch](https://github.com/Signalsmith-Audio/signalsmith-stretch) Rust bindings, patched so the same C++ also compiles to wasm32 |

## Building and testing

```bash
cargo test --workspace
```

```bash
scripts/clippy-lanes.sh
```

The wasm32 suites need [wasi-sdk](https://github.com/WebAssembly/wasi-sdk) 25
(`WASI_SDK_PATH`) and [wasmtime](https://wasmtime.dev):

```bash
cargo test -p maestro-core --target wasm32-wasip1 --test t_inv
```

## Status

This repository is a **read-only mirror**. Development happens inside the Maestro app
repository, and the engine's history is synced here. Issues are welcome; pull requests
can't be merged directly, but good ideas will be carried over by hand.

## License

MIT. See [LICENSE](LICENSE). Signalsmith Stretch (© Geraint Luff / Signalsmith
Audio) and its Rust bindings (© Colin Marc) are also MIT; their license files are in
`maestro-core/vendor/signalsmith-stretch/`.
