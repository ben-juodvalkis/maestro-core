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
//!
//! `memcpy`/`memset`/`memmove` AND the libm symbols (`cosf`, `expf`, …) come
//! from compiler-builtins, which provides them (weakly) on
//! wasm32-unknown-unknown precisely so float math works without a libc.
//! Defining our own libm shims here is a trap, found the hard way: a shim
//! like `cosf(x) = x.cos()` compiles to LLVM's `llvm.cos.f32`, which the
//! wasm backend lowers BACK to a call to `cosf` — the strong shim overrides
//! the weak compiler-builtins symbol and recurses until the stack dies (the
//! M4 worklet smoke failed with "Maximum call stack size exceeded" at the
//! first Signalsmith `Stretcher::new`). Let compiler-builtins resolve them.
//!
//! None of this is reachable on native targets — the module is compiled only
//! for `wasm32-unknown-unknown`.

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

#[no_mangle]
pub extern "C" fn abort() -> ! {
    // No I/O exists to report through; trap so the failure is loud (the
    // worklet host sees the wasm trap). Mirrors `panic = "abort"`.
    std::process::abort()
}
