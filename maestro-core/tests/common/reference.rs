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

/// Exact rational mirror of the spec §3.1 tempo segment table:
/// `bu(F) = bu_k + (F − F_k) · E_k` over an append-only table
/// `[(F_k, bu_k, E_k)]`. Written from the formulas, not from
/// `timeline.rs` — the engine is validated against this, bit-for-bit
/// (spec §6.2, M1 tempo-change legs).
pub struct RefTimeline {
    /// (start_frame, start_bu, mbpm); never empty, sorted by start_frame.
    segments: Vec<(u64, BigUint, u32)>,
}

impl RefTimeline {
    pub fn new(initial_mbpm: u32) -> Self {
        Self { segments: vec![(0, BigUint::ZERO, initial_mbpm)] }
    }

    /// Append a tempo change effective at `at_frame` (a block boundary in the
    /// scenario runner). `bu` is continuous across the change by construction.
    /// Two changes at the same boundary: the later wins (mirrors the engine).
    pub fn set_tempo(&mut self, at_frame: u64, mbpm: u32) {
        let (last_frame, _, last_mbpm) = self.segments.last().expect("never empty");
        assert!(at_frame >= *last_frame, "reference table is append-only");
        if mbpm == *last_mbpm {
            return; // no-op change; mirrors Timeline::set_tempo
        }
        let start_bu = self.bu(at_frame);
        let last = self.segments.last_mut().expect("never empty");
        if at_frame == last.0 {
            last.2 = mbpm;
        } else {
            self.segments.push((at_frame, start_bu, mbpm));
        }
    }

    /// Epoch reset (explicit transport stop / song switch — spec §3.1):
    /// musical position restarts at zero from `at_frame`.
    pub fn reset(&mut self, at_frame: u64, mbpm: u32) {
        self.segments.clear();
        self.segments.push((at_frame, BigUint::ZERO, mbpm));
    }

    /// Musical position at `frame` (exact, arbitrary precision).
    pub fn bu(&self, frame: u64) -> BigUint {
        let (f_k, bu_k, e_k) = self
            .segments
            .iter()
            .rev()
            .find(|(f_k, _, _)| *f_k <= frame)
            .expect("frame precedes reference epoch");
        bu_k + BigUint::from(frame - f_k) * BigUint::from(*e_k)
    }

    /// Tempo in effect at `frame`.
    pub fn tempo_at(&self, frame: u64) -> u32 {
        self.segments
            .iter()
            .rev()
            .find(|(f_k, _, _)| *f_k <= frame)
            .expect("frame precedes reference epoch")
            .2
    }

    /// Smallest frame `F ≥ from_frame` with `bu(F) ≥ target_bu`, assuming no
    /// segment is appended after `from_frame`'s segment (exact integer
    /// inversion of `bu` — the quantized-activation `F_q` of spec §3.1).
    pub fn invert_bu(&self, from_frame: u64, target_bu: &BigUint) -> u64 {
        let bu_from = self.bu(from_frame);
        if &bu_from >= target_bu {
            return from_frame;
        }
        let e = BigUint::from(self.tempo_at(from_frame));
        // d = ceil((target − bu_from) / E)
        let need = target_bu - &bu_from;
        let d = (&need + &e - BigUint::from(1u8)) / &e;
        from_frame + u64::try_from(&d).expect("inversion distance exceeds u64")
    }
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
