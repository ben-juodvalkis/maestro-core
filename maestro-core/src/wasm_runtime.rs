//! C runtime shims for the `wasm32-unknown-unknown` shipping module
//! (native audio rebuild M4, spec §3.8).
//!
//! The Signalsmith C++ compiles into the module via wasi-sdk (see the
//! vendored crate's build.rs). On `wasm32-wasip1` (the offline test lane)
//! its libc symbol needs are met by Rust's self-contained wasi-libc; on
//! `wasm32-unknown-unknown` there is no libc at all, so the C side's
//! `malloc`/`free` and libm calls resolve to these definitions instead.
//!
//! - `malloc`/`free` delegate to the Rust global allocator with a 16-byte
//!   size header (alignment preserved for the C++ payload).
//! - The libm functions delegate to Rust's own float implementations (std's
//!   wasm32 math is a compiled-in libm port under internal symbol names, so
//!   these `extern "C"` names cannot collide with it).
//!
//! `memcpy`/`memset`/`memmove` come from compiler-builtins as usual. None of
//! this is reachable from the audio path on native targets — the module is
//! compiled only for `wasm32-unknown-unknown`.

use std::alloc::{alloc, dealloc, Layout};

/// Header bytes prepended to every C allocation to remember its size.
/// 16 keeps the payload at the strictest alignment C++ may assume here.
const HEADER: usize = 16;

/// # Safety
/// C ABI `malloc`. The returned pointer is valid for `size` bytes or null.
#[no_mangle]
pub extern "C" fn malloc(size: usize) -> *mut u8 {
    let Some(total) = size.checked_add(HEADER) else {
        return std::ptr::null_mut();
    };
    let Ok(layout) = Layout::from_size_align(total, HEADER) else {
        return std::ptr::null_mut();
    };
    unsafe {
        let base = alloc(layout);
        if base.is_null() {
            return base;
        }
        (base as *mut usize).write(total);
        base.add(HEADER)
    }
}

/// # Safety
/// C ABI `free`. `ptr` must be null or a pointer previously returned by the
/// `malloc` above.
#[no_mangle]
pub unsafe extern "C" fn free(ptr: *mut u8) {
    if ptr.is_null() {
        return;
    }
    let base = ptr.sub(HEADER);
    let total = (base as *const usize).read();
    if let Ok(layout) = Layout::from_size_align(total, HEADER) {
        dealloc(base, layout);
    }
}

macro_rules! libm_shim_f32 {
    ($($name:ident => $method:ident),* $(,)?) => {$(
        #[no_mangle]
        pub extern "C" fn $name(x: f32) -> f32 { x.$method() }
    )*};
}

macro_rules! libm_shim_f64 {
    ($($name:ident => $method:ident),* $(,)?) => {$(
        #[no_mangle]
        pub extern "C" fn $name(x: f64) -> f64 { x.$method() }
    )*};
}

libm_shim_f32! {
    cosf => cos, sinf => sin, tanf => tan,
    expf => exp, exp2f => exp2, logf => ln, log2f => log2, log10f => log10,
    roundf => round, cbrtf => cbrt,
    asinf => asin, acosf => acos, atanf => atan,
}

libm_shim_f64! {
    cos => cos, sin => sin, tan => tan,
    exp => exp, exp2 => exp2, log => ln, log2 => log2, log10 => log10,
    round => round, cbrt => cbrt,
    asin => asin, acos => acos, atan => atan,
}

#[no_mangle]
pub extern "C" fn powf(x: f32, y: f32) -> f32 {
    x.powf(y)
}

#[no_mangle]
pub extern "C" fn pow(x: f64, y: f64) -> f64 {
    x.powf(y)
}

#[no_mangle]
pub extern "C" fn fmodf(x: f32, y: f32) -> f32 {
    x % y
}

#[no_mangle]
pub extern "C" fn fmod(x: f64, y: f64) -> f64 {
    x % y
}

#[no_mangle]
pub extern "C" fn atan2f(y: f32, x: f32) -> f32 {
    y.atan2(x)
}

#[no_mangle]
pub extern "C" fn atan2(y: f64, x: f64) -> f64 {
    y.atan2(x)
}

#[no_mangle]
pub extern "C" fn abort() -> ! {
    // No I/O exists to report through; trap so the failure is loud (the
    // worklet host sees the wasm trap). Mirrors `panic = "abort"`.
    std::process::abort()
}
