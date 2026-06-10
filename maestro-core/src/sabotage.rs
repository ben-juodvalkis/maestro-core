//! # sabotage — drift-injection modes for T-INV negative controls (spec §6.2)
//!
//! Only compiled under the `drift-injection` feature; **never enabled in
//! shipping builds**. CI builds the test suite with this feature and asserts
//! that the invariant tests DETECT each mode — proving the suite can catch
//! drift at the 1-sample level. If a sabotage mode ever passes T-INV, the
//! suite is broken and the build fails.
//!
//! The three modes reproduce the abandoned branch's real defect classes:
//!
//! - `FloatCursor` — a per-track f64 position advanced per block (the banned
//!   accumulating-cursor class, dossier §5.1 / spec §2.3#1).
//! - `PcmLengthLoop` — wrap at integer PCM length instead of beat space
//!   (defect D2, the structural drift root — spec §2.2).
//! - `TimerClock` — advance `F` from a jittery simulated wall timer instead
//!   of rendered frames (defect D8 — spec §3.3).

use std::sync::atomic::{AtomicU8, Ordering};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Sabotage {
    None = 0,
    FloatCursor = 1,
    PcmLengthLoop = 2,
    TimerClock = 3,
    /// Defect D5/D6 class: stretch silently disabled — non-unity tracks read
    /// 1:1 (wrong rate). The C0 device checkpoint's cross-tempo scenario must
    /// FAIL its analysis against a build running this mode (spec §6.1).
    StretchDisabled = 4,
}

static MODE: AtomicU8 = AtomicU8::new(0);

pub fn set_mode(mode: Sabotage) {
    MODE.store(mode as u8, Ordering::SeqCst);
}

#[inline]
pub fn mode() -> Sabotage {
    match MODE.load(Ordering::Relaxed) {
        1 => Sabotage::FloatCursor,
        2 => Sabotage::PcmLengthLoop,
        3 => Sabotage::TimerClock,
        4 => Sabotage::StretchDisabled,
        _ => Sabotage::None,
    }
}

pub(crate) fn set_mode_raw(mode: u8) {
    MODE.store(if mode <= 4 { mode } else { 0 }, Ordering::SeqCst);
}
