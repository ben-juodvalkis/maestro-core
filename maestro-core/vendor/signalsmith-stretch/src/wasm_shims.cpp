// wasm32 runtime shims for the Signalsmith C++ (native audio rebuild M4,
// spec §3.8). Neither wasm32 Rust target links a C++ standard library
// archive, so the handful of out-of-line libc++ symbols the header-only
// code references are defined here instead:
//
//   - operator new/delete        → malloc/free (wasi-libc on wasm32-wasip1;
//                                  maestro-core's Rust allocator shims on
//                                  wasm32-unknown-unknown)
//   - std::random_device         → deterministic seed. Only ever called to
//                                  seed SignalsmithStretch's randomEngine,
//                                  whose randomized path is unreachable under
//                                  the engine's clean-stretch clamp
//                                  (spec §4.8 / §2.3#4); a fixed seed also
//                                  keeps the wasm module fully deterministic
//                                  (upstream ships an equivalent Emscripten
//                                  workaround in web/emscripten/pre.js).
//   - __libcpp_verbose_abort     → trap (no I/O exists to print to).
//
// Compiled only for wasm32 targets (see build.rs).

#include <cstddef>
#include <new>
#include <random>
#include <string>

extern "C" {
void *malloc(size_t size);
void free(void *ptr);
}

void *operator new(std::size_t size) {
    void *p = malloc(size);
    if (p == nullptr) {
        __builtin_trap(); // allocation failure: fail loudly, never corrupt
    }
    return p;
}

void *operator new[](std::size_t size) { return operator new(size); }

void operator delete(void *ptr) noexcept { free(ptr); }
void operator delete[](void *ptr) noexcept { free(ptr); }
void operator delete(void *ptr, std::size_t) noexcept { free(ptr); }
void operator delete[](void *ptr, std::size_t) noexcept { free(ptr); }

_LIBCPP_BEGIN_NAMESPACE_STD

random_device::random_device(const string & /*token*/) {}

random_device::~random_device() {}

unsigned random_device::operator()() {
    // Deterministic: see header comment. The value itself is irrelevant —
    // the seeded engine is never consulted inside the clean-stretch bound.
    return 0x6d616573; // "maes"
}

[[noreturn]] void __libcpp_verbose_abort(const char * /*format*/, ...) {
    __builtin_trap();
}

_LIBCPP_END_NAMESPACE_STD
