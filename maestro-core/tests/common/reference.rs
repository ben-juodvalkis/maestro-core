//! Exact rational reference implementation of the spec §3.1 formulas in
//! `num-bigint` (spec §6.2: the T-INV invariant test cross-checks the engine
//! bit-for-bit against this mirror). Deliberately written from the formulas,
//! not from the engine code — a shared bug would have to be made twice in two
//! numeric systems.

use num_bigint::BigUint;

/// `K`: beat-units per beat (60_000 × 48_000).
pub fn k() -> BigUint {
    BigUint::from(60_000u32) * BigUint::from(48_000u32)
}

/// `bu(F)` at fixed tempo `E` (mBPM): `F · E`.
pub fn bu_fixed(frame: u64, e_mbpm: u32) -> BigUint {
    BigUint::from(frame) * BigUint::from(e_mbpm)
}

/// `u_i(F) = floor(bu / O_i)`.
pub fn u(bu: &BigUint, o_mbpm: u32) -> u64 {
    let q = bu / BigUint::from(o_mbpm);
    u64::try_from(&q).expect("reference position exceeds u64")
}

/// `Λ_i = loop_beats · K`.
pub fn lambda(loop_beats: u32) -> BigUint {
    BigUint::from(loop_beats) * k()
}

/// `w(n) = floor(n · O / Λ)`.
pub fn wrap_of_sample(n: u64, o_mbpm: u32, loop_beats: u32) -> u64 {
    let q = (BigUint::from(n) * BigUint::from(o_mbpm)) / lambda(loop_beats);
    u64::try_from(&q).expect("wrap count exceeds u64")
}

/// `wrap_start(w) = ceil(w · Λ / O)`.
pub fn wrap_start_sample(w: u64, o_mbpm: u32, loop_beats: u32) -> u64 {
    let num = BigUint::from(w) * lambda(loop_beats);
    let den = BigUint::from(o_mbpm);
    let q = if (&num % &den) == BigUint::ZERO { &num / &den } else { &num / &den + 1u32 };
    u64::try_from(&q).expect("wrap start exceeds u64")
}

/// `pcm(n) = n − wrap_start(w(n))`, clamped to `pcm_len − 1`.
pub fn pcm_index(n: u64, o_mbpm: u32, loop_beats: u32, pcm_len: u64) -> u64 {
    let w = wrap_of_sample(n, o_mbpm, loop_beats);
    (n - wrap_start_sample(w, o_mbpm, loop_beats)).min(pcm_len - 1)
}
