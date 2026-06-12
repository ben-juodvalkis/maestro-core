// Vendored build.rs (native audio rebuild M4, spec §3.8).
//
// Native targets: byte-identical to upstream 0.1.3 behavior (cc + bindgen).
//
// wasm32 targets: the SAME C++ compiles into the module via wasi-sdk's
// clang++ (`WASI_SDK_PATH` must point at a wasi-sdk install). Objects are
// compiled with `--target=wasm32-wasip1` regardless of the Rust target —
// wasm32 object files are link-compatible across wasm32-wasip1 (the offline
// test lane, which links Rust's self-contained wasi-libc) and
// wasm32-unknown-unknown (the shipping AudioWorklet module, where
// maestro-core provides malloc/free + libm shims in Rust). `wasm_shims.cpp`
// supplies the few libc++ runtime symbols that are never linked on either
// wasm target: operator new/delete, std::random_device (deterministic seed —
// the engine never enters Signalsmith's randomized path, spec §4.8/§2.3#4),
// and __libcpp_verbose_abort.

use std::env;
use std::path::{Path, PathBuf};

fn main() {
    let src_path = Path::new("src");

    println!("cargo::rerun-if-changed=src/wrapper.h");
    println!("cargo::rerun-if-changed=src/wrapper.cpp");
    println!("cargo::rerun-if-changed=src/wasm_shims.cpp");
    println!("cargo::rerun-if-changed=signalsmith-stretch/signalsmith-stretch.h");
    println!("cargo::rerun-if-env-changed=WASI_SDK_PATH");

    let target = env::var("TARGET").unwrap_or_default();
    let is_wasm = target.starts_with("wasm32");

    let mut build = cc::Build::new();
    build
        .file(src_path.join("wrapper.cpp"))
        .include(Path::new("signalsmith-stretch"))
        .include(Path::new("."))
        .cpp(true)
        .std("c++14");

    let mut bindgen_builder = bindgen::Builder::default()
        .header(src_path.join("wrapper.h").as_os_str().to_str().unwrap())
        .allowlist_item("signalsmith_stretch_.*");

    if is_wasm {
        let wasi_sdk = PathBuf::from(env::var("WASI_SDK_PATH").expect(
            "WASI_SDK_PATH must point at a wasi-sdk install to build the \
             Signalsmith C++ for wasm32 (native audio rebuild spec §3.8). \
             Download: https://github.com/WebAssembly/wasi-sdk/releases",
        ));
        let clangxx = wasi_sdk.join("bin/clang++");
        assert!(
            clangxx.exists(),
            "WASI_SDK_PATH={} has no bin/clang++",
            wasi_sdk.display()
        );

        build
            .compiler(&clangxx)
            .archiver(wasi_sdk.join("bin/ar"))
            // wasip1 selects the full libc/libc++ header set baked into the
            // wasi-sdk driver; the resulting wasm32 objects link into both
            // Rust wasm32 targets (only the runtime symbols differ — shims
            // below / in maestro-core cover the gap).
            .target("wasm32-wasip1")
            .flag("-fno-exceptions")
            .flag("-fno-rtti")
            .flag("-fvisibility=hidden")
            // Runtime symbols libc++.a would normally provide (never linked
            // on either wasm target).
            .file(src_path.join("wasm_shims.cpp"))
            // No `-lstdc++`/`-lc++` emission — there is no C++ stdlib archive
            // in the wasm link; everything needed is header-only + shims.
            .cpp_link_stdlib(None);

        // bindgen parses the plain-C wrapper.h; only the data layout matters
        // (identical across wasm32 triples). `-fvisibility=default` is
        // load-bearing: clang marks wasm32 declarations hidden by default and
        // bindgen 0.70 then silently generates NO function bindings
        // (empirically verified; the struct/typedefs survive, which is why
        // the failure mode is 16 missing `sys::` functions, not a parse
        // error).
        bindgen_builder = bindgen_builder
            .clang_arg("--target=wasm32-unknown-unknown")
            .clang_arg("-fvisibility=default");
    }

    build.compile("signalsmith-stretch");

    let bindings = bindgen_builder
        .generate()
        .expect("Unable to generate bindings");

    let out_path = Path::new(&env::var("OUT_DIR").unwrap()).join("bindings.rs");
    bindings
        .write_to_file(out_path)
        .expect("Couldn't write bindings!");
}
