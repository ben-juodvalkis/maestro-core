//! # timeline — the single timing authority (spec §3.1)
//!
//! Every musical position in the engine is **derived** from one mutable timing
//! variable: the master frame counter `F: u64`, advanced by the audio callback
//! by exactly the block size, only while playing. There is no per-track cursor
//! anywhere — a position that is *advanced* each block is forbidden in this
//! codebase; positions are *computed*. The T-INV negative controls (spec §6.2)
//! fail the build if an accumulating cursor is ever reintroduced.
//!
//! ## Units
//!
//! - **Frames** — output samples at the fixed engine rate (48 kHz, spec §4.7).
//! - **milli-BPM** (`u32`) — integer tempo unit; 116 BPM = `116_000`.
//! - **Beat-units** (`u128`) — the engine's musical position unit, with
//!   `K = 60_000 × SR = 2_880_000_000` beat-units per beat. Chosen so that at
//!   tempo `E` mBPM, musical position advances **exactly `E` beat-units per
//!   frame** — all timeline arithmetic is integer multiplication/division.
//!
//! ## The mapping (all integer, all stateless)
//!
//! ```text
//! bu(F)  = bu_k + (F − F_k) · E_k          // musical position (segment table)
//! u_i(F) = floor(bu(F) / O_i)              // dictated source sample of track i
//! Λ_i    = loop_beats_i · K                // loop length in beat-units
//! w(n)   = floor(n · O_i / Λ_i)            // wrap index of unwrapped sample n
//! pcm(n) = n − ceil(w(n) · Λ_i / O_i)      // PCM index (clamped, see below)
//! ```
//!
//! Loops wrap in **beat space**, never at an integer PCM length: the per-wrap
//! PCM read length alternates `⌊Λ/O⌋ / ⌈Λ/O⌉` (Bresenham) so the average loop
//! length equals the exact rational musical length with zero cumulative error.
//! This kills the structural drift root (spec §2.2): an exact musical loop
//! length is generally not an integer number of samples.
//!
//! ## Overflow bounds (spec §4.1)
//!
//! `F: u64` overflows after 12 million years at 48 kHz. `bu ≤ F · E_max`:
//! after 100 years of continuous 999-BPM playback `bu ≈ 2⁹⁰ ≪ 2¹²⁸`. The
//! products `n·O` and `w·Λ` are bounded by `bu`, so every intermediate stays
//! comfortably inside `u128` for > 10⁵ years.
//!
//! Floats appear **nowhere** in this module except the explicitly display-only
//! helpers at the bottom, which feed UI snapshots and never feed timing.

/// Fixed engine sample rate (spec §4.7). Every asset must decode to exactly
/// this rate; the engine never resamples.
pub const ENGINE_SAMPLE_RATE: u32 = 48_000;

/// Beat-units per beat: `K = 60_000 × SR`. At tempo `E` mBPM, musical position
/// advances exactly `E` beat-units per output frame.
pub const BEAT_UNITS_PER_BEAT: u128 = 60_000 * ENGINE_SAMPLE_RATE as u128; // 2_880_000_000

/// Loop-identity tolerance (spec §3.4#2): a track's decoded PCM length must be
/// within this many samples (0.5 ms) of its exact musical loop length
/// `loop_beats · K / O`, or the load fails loudly.
pub const LOOP_IDENTITY_TOLERANCE_SAMPLES: u64 = 24;

/// Engine tempo clamp (spec §4.8), in milli-BPM.
pub const MIN_TEMPO_MBPM: u32 = 40_000;
pub const MAX_TEMPO_MBPM: u32 = 300_000;

/// Ceiling division for u128 (no floats, no rounding modes).
#[inline(always)]
fn div_ceil_u128(a: u128, b: u128) -> u128 {
    debug_assert!(b > 0);
    if a == 0 {
        0
    } else {
        (a - 1) / b + 1
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TimelineError {
    /// Tempo outside the §4.8 clamp. Rejected, never silently clamped.
    TempoOutOfRange { mbpm: u32 },
    /// Track metadata invalid (zero tempo / zero loop length / empty PCM).
    InvalidTrackMetadata { reason: &'static str },
    /// Decoded PCM length is inconsistent with the track's musical identity
    /// (spec §3.4#2). `expected_*` bracket the exact rational length Λ/O.
    LoopIdentityMismatch {
        pcm_len: u64,
        expected_floor: u64,
        expected_ceil: u64,
    },
}

impl std::fmt::Display for TimelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TempoOutOfRange { mbpm } => {
                write!(f, "tempo {mbpm} mBPM outside [{MIN_TEMPO_MBPM}, {MAX_TEMPO_MBPM}]")
            }
            Self::InvalidTrackMetadata { reason } => write!(f, "invalid track metadata: {reason}"),
            Self::LoopIdentityMismatch { pcm_len, expected_floor, expected_ceil } => write!(
                f,
                "PCM length {pcm_len} inconsistent with musical loop length \
                 [{expected_floor}, {expected_ceil}] (±{LOOP_IDENTITY_TOLERANCE_SAMPLES})"
            ),
        }
    }
}

impl std::error::Error for TimelineError {}

/// One row of the append-only tempo segment table: at frame `start_frame` the
/// engine tempo became `mbpm`, with accumulated musical position `start_bu`.
/// `bu` is continuous across every change by construction — every track's
/// musical phase is preserved simultaneously; there is nothing per-track to
/// re-anchor (kills D7).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TempoSegment {
    pub start_frame: u64,
    pub start_bu: u128,
    pub mbpm: u32,
}

/// The engine timeline: an append-only tempo segment table over the master
/// frame clock. The clock itself (`F`) lives in the audio core; this struct
/// answers "what is the musical position at frame F" as a pure function.
#[derive(Debug, Clone)]
pub struct Timeline {
    /// Never empty. Sorted by `start_frame`, append-only between epoch resets.
    segments: Vec<TempoSegment>,
    /// Bumped on every epoch reset (explicit transport stop / song switch —
    /// spec §3.1 "Timeline epoch"). Display/diagnostic only.
    epoch: u32,
}

impl Timeline {
    /// Capacity for tempo segments preallocated up front. `set_tempo` is
    /// called from the audio thread at block boundaries; staying under this
    /// capacity means no allocation on the audio thread. 1024 live tempo
    /// changes between epoch resets is far beyond any real session; if it is
    /// ever exceeded Vec growth occurs (allocation), which `assert_no_alloc`
    /// render tests would catch in CI rather than corrupting timing.
    pub const SEGMENT_CAPACITY: usize = 1024;

    pub fn new(initial_mbpm: u32) -> Result<Self, TimelineError> {
        Self::validate_tempo(initial_mbpm)?;
        let mut segments = Vec::with_capacity(Self::SEGMENT_CAPACITY);
        segments.push(TempoSegment { start_frame: 0, start_bu: 0, mbpm: initial_mbpm });
        Ok(Self { segments, epoch: 0 })
    }

    pub fn validate_tempo(mbpm: u32) -> Result<(), TimelineError> {
        if !(MIN_TEMPO_MBPM..=MAX_TEMPO_MBPM).contains(&mbpm) {
            return Err(TimelineError::TempoOutOfRange { mbpm });
        }
        Ok(())
    }

    /// Musical position (beat-units) at output frame `frame`. Pure integer
    /// function of the segment table; bit-identical on every architecture.
    #[inline]
    pub fn bu(&self, frame: u64) -> u128 {
        let seg = self.segment_at(frame);
        seg.start_bu + (frame - seg.start_frame) as u128 * seg.mbpm as u128
    }

    /// Engine tempo in effect at `frame`.
    #[inline]
    pub fn tempo_mbpm_at(&self, frame: u64) -> u32 {
        self.segment_at(frame).mbpm
    }

    #[inline]
    fn segment_at(&self, frame: u64) -> &TempoSegment {
        debug_assert!(!self.segments.is_empty());
        debug_assert!(
            frame >= self.segments[0].start_frame,
            "query before timeline epoch start"
        );
        // Last segment with start_frame <= frame. The table is small and the
        // common case is the final segment, so scan from the back.
        self.segments
            .iter()
            .rev()
            .find(|s| s.start_frame <= frame)
            .expect("frame precedes timeline epoch")
    }

    /// Append a tempo change effective at `at_frame` (the render loop passes
    /// the next block boundary). `bu` is continuous across the change by
    /// construction. Append-only: `at_frame` must not precede the last segment.
    pub fn set_tempo(&mut self, at_frame: u64, mbpm: u32) -> Result<(), TimelineError> {
        Self::validate_tempo(mbpm)?;
        let last = *self.segments.last().expect("segments never empty");
        assert!(
            at_frame >= last.start_frame,
            "tempo segment table is append-only (at_frame {at_frame} < {})",
            last.start_frame
        );
        if mbpm == last.mbpm {
            return Ok(()); // no-op change; keep the table minimal
        }
        let start_bu = self.bu(at_frame);
        if at_frame == last.start_frame {
            // Two changes at the same boundary: the later one wins; start_bu
            // is unchanged because bu(at_frame) is segment-start exact.
            let s = self.segments.last_mut().expect("segments never empty");
            s.mbpm = mbpm;
        } else {
            self.segments.push(TempoSegment { start_frame: at_frame, start_bu, mbpm });
        }
        Ok(())
    }

    /// Explicit epoch reset (spec §3.1): issued in exactly two flows —
    /// transport stop and song switch. Musical position restarts at zero
    /// (bar 1 beat 1) from `at_frame`; earlier frames can no longer be queried.
    pub fn reset(&mut self, at_frame: u64, mbpm: u32) -> Result<(), TimelineError> {
        Self::validate_tempo(mbpm)?;
        self.segments.clear();
        self.segments.push(TempoSegment { start_frame: at_frame, start_bu: 0, mbpm });
        self.epoch = self.epoch.wrapping_add(1);
        Ok(())
    }

    pub fn epoch(&self) -> u32 {
        self.epoch
    }

    pub fn segments(&self) -> &[TempoSegment] {
        &self.segments
    }
}

/// A track's immutable musical identity (spec §3.1 "Track mapping"): original
/// tempo `O` (mBPM), loop length `Λ = loop_beats · K` (beat-units), and the
/// validated PCM length. Carries **no mutable state** — every position is a
/// pure function of `bu(F)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrackTimeline {
    o_mbpm: u32,
    loop_beats: u32,
    /// Λ = loop_beats · K, cached.
    lambda: u128,
    /// Decoded PCM length in frames, already validated against the loop
    /// identity (|pcm_len − Λ/O| < 24 samples).
    pcm_len: u64,
}

impl TrackTimeline {
    /// Validates the loop identity (spec §3.4#2) entirely in integers:
    /// `|pcm_len − Λ/O| < 24  ⇔  |pcm_len·O − Λ| < 24·O`.
    pub fn new(o_mbpm: u32, loop_beats: u32, pcm_len: u64) -> Result<Self, TimelineError> {
        if o_mbpm == 0 {
            return Err(TimelineError::InvalidTrackMetadata { reason: "original tempo is 0" });
        }
        if loop_beats == 0 {
            return Err(TimelineError::InvalidTrackMetadata { reason: "loop_beats is 0" });
        }
        if pcm_len == 0 {
            return Err(TimelineError::InvalidTrackMetadata { reason: "PCM length is 0" });
        }
        let lambda = loop_beats as u128 * BEAT_UNITS_PER_BEAT;
        let o = o_mbpm as u128;
        let scaled_pcm = pcm_len as u128 * o;
        let diff = scaled_pcm.abs_diff(lambda);
        if diff >= LOOP_IDENTITY_TOLERANCE_SAMPLES as u128 * o {
            return Err(TimelineError::LoopIdentityMismatch {
                pcm_len,
                expected_floor: (lambda / o) as u64,
                expected_ceil: div_ceil_u128(lambda, o) as u64,
            });
        }
        Ok(Self { o_mbpm, loop_beats, lambda, pcm_len })
    }

    #[inline]
    pub fn original_tempo_mbpm(&self) -> u32 {
        self.o_mbpm
    }

    #[inline]
    pub fn loop_beats(&self) -> u32 {
        self.loop_beats
    }

    #[inline]
    pub fn lambda(&self) -> u128 {
        self.lambda
    }

    #[inline]
    pub fn pcm_len(&self) -> u64 {
        self.pcm_len
    }

    /// `u(F)`: the dictated **unwrapped** source sample at musical position
    /// `bu`. Two tracks' `u·O` both lie within `[bu − O, bu]` — their musical
    /// phase difference is bounded by < 1 source sample of the slower track,
    /// for every frame, forever. That bound is an algebraic identity of
    /// `floor`, not a property maintained by code.
    #[inline]
    pub fn u(&self, bu: u128) -> u64 {
        let n = bu / self.o_mbpm as u128;
        debug_assert!(n <= u64::MAX as u128, "unwrapped position exceeds u64");
        n as u64
    }

    /// Loop (wrap) count at musical position `bu`: `floor(bu / Λ)`.
    #[inline]
    pub fn wrap_count(&self, bu: u128) -> u64 {
        (bu / self.lambda) as u64
    }

    /// Wrap index containing unwrapped source sample `n`: `w(n) = floor(n·O/Λ)`.
    #[inline]
    pub fn wrap_of_sample(&self, n: u64) -> u64 {
        ((n as u128 * self.o_mbpm as u128) / self.lambda) as u64
    }

    /// First unwrapped source sample of wrap `w`: `ceil(w·Λ/O)`. Per-wrap read
    /// lengths `wrap_start(w+1) − wrap_start(w)` alternate `⌊Λ/O⌋ / ⌈Λ/O⌉`
    /// (Bresenham) so the cumulative loop length telescopes to the exact
    /// rational musical length — zero accumulated error after any wrap count.
    #[inline]
    pub fn wrap_start_sample(&self, w: u64) -> u64 {
        div_ceil_u128(w as u128 * self.lambda, self.o_mbpm as u128) as u64
    }

    /// PCM index of unwrapped source sample `n` (spec §3.1 `pcm_i(n)`).
    ///
    /// The Bresenham long-wrap (`⌈Λ/O⌉`) can dictate one index past the end of
    /// PCM that decoded at the floor length; the final read clamps to
    /// `pcm_len − 1` — a sub-sample waveform detail at the seam, never
    /// accumulated (positions are derived; the clamp affects audio only).
    #[inline]
    pub fn pcm_index(&self, n: u64) -> u64 {
        let w = self.wrap_of_sample(n);
        let idx = n - self.wrap_start_sample(w);
        idx.min(self.pcm_len - 1)
    }

    /// Unwrapped samples from `n` (inclusive) to the end of its wrap
    /// (exclusive) — the length of the contiguous PCM run starting at
    /// `pcm_index(n)`. Lets readers copy contiguous spans instead of doing
    /// per-sample division.
    #[inline]
    pub fn run_remaining(&self, n: u64) -> u64 {
        self.wrap_start_sample(self.wrap_of_sample(n) + 1) - n
    }
}

// ---------------------------------------------------------------------------
// Display-only conversions (floats allowed here and only here — these feed
// UI snapshots, never timing).
// ---------------------------------------------------------------------------

/// Whole beats and fractional beat at `bu`. Display only.
#[inline]
pub fn beats_display(bu: u128) -> (u64, f32) {
    let whole = (bu / BEAT_UNITS_PER_BEAT) as u64;
    let frac_units = (bu % BEAT_UNITS_PER_BEAT) as u64;
    (whole, frac_units as f32 / BEAT_UNITS_PER_BEAT as f32)
}

/// Loop progress in [0, 1) at `bu` for a track. Display only.
#[inline]
pub fn loop_progress_display(track: &TrackTimeline, bu: u128) -> f32 {
    let in_loop = bu % track.lambda();
    // Scale down before the float conversion to keep precision reasonable.
    (in_loop as f64 / track.lambda() as f64) as f32
}

// ---------------------------------------------------------------------------
// Tests — real-world stem geometry (spec §2.2): exact musical loop lengths
// are generally NOT integer sample counts; these tests pin the Bresenham
// wrap behavior against the actual San Francisco / Humboldt metadata.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// San Francisco voice: 84 beats @ 116 BPM. Exact length
    /// 84·K/116000 = 2_085_517.241… samples; asset decodes to 2_085_518.
    fn sf_voice() -> TrackTimeline {
        TrackTimeline::new(116_000, 84, 2_085_518).unwrap()
    }

    /// Humboldt bass: 32 beats @ 113 BPM. Exact length 815_575.221… samples;
    /// asset decodes to 815_576.
    fn humboldt_bass() -> TrackTimeline {
        TrackTimeline::new(113_000, 32, 815_576).unwrap()
    }

    /// Metronome fixture: 16 beats @ 120 BPM = exactly 384_000 samples.
    fn metronome_120() -> TrackTimeline {
        TrackTimeline::new(120_000, 16, 384_000).unwrap()
    }

    #[test]
    fn beat_unit_constant_matches_spec() {
        assert_eq!(BEAT_UNITS_PER_BEAT, 2_880_000_000);
    }

    #[test]
    fn bu_is_linear_at_fixed_tempo() {
        let tl = Timeline::new(120_000).unwrap();
        assert_eq!(tl.bu(0), 0);
        assert_eq!(tl.bu(1), 120_000);
        assert_eq!(tl.bu(48_000), 48_000u128 * 120_000);
        // One beat at 120 BPM = 0.5 s = 24_000 frames.
        assert_eq!(tl.bu(24_000), BEAT_UNITS_PER_BEAT);
    }

    #[test]
    fn tempo_change_keeps_bu_continuous() {
        let mut tl = Timeline::new(120_000).unwrap();
        let f_c = 100_000u64;
        let bu_before = tl.bu(f_c);
        tl.set_tempo(f_c, 87_000).unwrap();
        assert_eq!(tl.bu(f_c), bu_before, "bu must be continuous at the change");
        assert_eq!(tl.bu(f_c + 1) - tl.bu(f_c), 87_000, "new slope after change");
        assert_eq!(tl.tempo_mbpm_at(f_c - 1), 120_000);
        assert_eq!(tl.tempo_mbpm_at(f_c), 87_000);
    }

    #[test]
    fn tempo_changes_at_same_boundary_last_wins() {
        let mut tl = Timeline::new(120_000).unwrap();
        tl.set_tempo(5_000, 90_000).unwrap();
        tl.set_tempo(5_000, 100_000).unwrap();
        assert_eq!(tl.tempo_mbpm_at(5_000), 100_000);
        assert_eq!(tl.segments().len(), 2, "same-boundary change replaces, not appends");
        // Continuity still holds.
        assert_eq!(tl.bu(5_000), 5_000u128 * 120_000);
    }

    #[test]
    fn tempo_out_of_range_rejected() {
        assert!(Timeline::new(39_999).is_err());
        assert!(Timeline::new(300_001).is_err());
        let mut tl = Timeline::new(120_000).unwrap();
        assert!(tl.set_tempo(0, 30_000).is_err());
    }

    #[test]
    fn epoch_reset_restarts_musical_position() {
        let mut tl = Timeline::new(120_000).unwrap();
        tl.set_tempo(50_000, 87_000).unwrap();
        let f_r = 200_000u64;
        tl.reset(f_r, 116_000).unwrap();
        assert_eq!(tl.epoch(), 1);
        assert_eq!(tl.bu(f_r), 0, "musical position restarts at bar 1 beat 1");
        assert_eq!(tl.bu(f_r + 10), 10 * 116_000);
        assert_eq!(tl.segments().len(), 1);
    }

    #[test]
    fn loop_identity_accepts_real_assets_and_rejects_corrupt_ones() {
        // The two real-world non-integer cases from spec §2.2.
        sf_voice();
        humboldt_bass();
        // Exact-integer case.
        metronome_120();
        // A double-trim corrupted length (−3072 samples, the D4 defect) must fail.
        let err = TrackTimeline::new(116_000, 84, 2_085_518 - 3_072).unwrap_err();
        assert!(matches!(err, TimelineError::LoopIdentityMismatch { .. }));
        // Off by exactly the tolerance must fail; one inside must pass.
        assert!(TrackTimeline::new(120_000, 16, 384_000 + 24).is_err());
        assert!(TrackTimeline::new(120_000, 16, 384_000 + 23).is_ok());
    }

    #[test]
    fn sf_voice_wrap_lengths_alternate_bresenham() {
        // Exact rational loop length: Λ/O = 84·2_880_000_000/116_000
        //   = 2_085_517.2413… → wraps must alternate 2_085_517 / 2_085_518
        // with zero cumulative error against the rational length.
        let t = sf_voice();
        let lo = (t.lambda() / t.original_tempo_mbpm() as u128) as u64;
        let hi = lo + 1;
        assert_eq!(lo, 2_085_517);
        let mut seen_lo = 0u64;
        let mut seen_hi = 0u64;
        for w in 0..10_000u64 {
            let len = t.wrap_start_sample(w + 1) - t.wrap_start_sample(w);
            assert!(
                len == lo || len == hi,
                "wrap {w}: read length {len} not in {{{lo}, {hi}}}"
            );
            if len == lo {
                seen_lo += 1;
            } else {
                seen_hi += 1;
            }
        }
        assert!(seen_lo > 0 && seen_hi > 0, "both Bresenham lengths must occur");
        // Zero cumulative error: |wrap_start(N)·O − N·Λ| < O for any N —
        // i.e. the integer wrap grid never deviates a full sample from the
        // exact rational grid, no matter how many wraps elapse.
        for n in [1u64, 7, 100, 9_999, 10_000] {
            let start = t.wrap_start_sample(n) as u128 * t.original_tempo_mbpm() as u128;
            let exact = n as u128 * t.lambda();
            assert!(start >= exact, "ceil property violated");
            assert!(
                start - exact < t.original_tempo_mbpm() as u128,
                "wrap grid drifted ≥ 1 sample after {n} wraps"
            );
        }
    }

    #[test]
    fn integer_length_loop_has_constant_wrap_length() {
        let t = metronome_120();
        for w in 0..1_000u64 {
            assert_eq!(t.wrap_start_sample(w + 1) - t.wrap_start_sample(w), 384_000);
        }
    }

    #[test]
    fn pcm_index_stays_in_bounds_and_zeroes_at_wrap_starts() {
        for t in [sf_voice(), humboldt_bass(), metronome_120()] {
            for w in [0u64, 1, 2, 57, 9_999] {
                let start = t.wrap_start_sample(w);
                assert_eq!(t.pcm_index(start), 0, "wrap {w} must start at PCM 0");
                if start > 0 {
                    let last = t.pcm_index(start - 1);
                    assert!(last <= t.pcm_len() - 1);
                    assert!(
                        last >= t.pcm_len() - 2,
                        "sample before wrap start should be near PCM end, got {last}"
                    );
                }
            }
            // Scan a full wrap boundary region densely.
            let b = t.wrap_start_sample(3);
            for n in (b - 5)..(b + 5) {
                assert!(t.pcm_index(n) < t.pcm_len());
            }
        }
    }

    #[test]
    fn run_remaining_lands_exactly_on_wrap_starts() {
        let t = sf_voice();
        let mut n = 0u64;
        for w in 1..=200u64 {
            n += t.run_remaining(n);
            assert_eq!(n, t.wrap_start_sample(w));
            assert_eq!(t.pcm_index(n), 0);
        }
    }

    #[test]
    fn unity_track_advances_exactly_one_to_one() {
        // O == E: u(F+B) − u(F) == B identically — the bit-exact bypass
        // precondition (spec §3.2).
        let tl = Timeline::new(116_000).unwrap();
        let t = sf_voice();
        let mut f = 0u64;
        for b in [128u64, 480, 1024, 4096, 1, 333] {
            let span = t.u(tl.bu(f + b)) - t.u(tl.bu(f));
            assert_eq!(span, b, "unity track must advance exactly B per block");
            f += b;
        }
    }

    #[test]
    fn feeds_telescope_to_exact_position() {
        // Sum of per-block feed counts u(F+B)−u(F) equals u(F_end) exactly —
        // no remainder carried in mutable state; nothing to accumulate.
        let tl = Timeline::new(120_000).unwrap();
        for t in [sf_voice(), humboldt_bass(), metronome_120()] {
            let mut f = 0u64;
            let mut fed = 0u64;
            let blocks = [128u64, 480, 1024];
            for i in 0..30_000u64 {
                let b = blocks[(i % 3) as usize];
                fed += t.u(tl.bu(f + b)) - t.u(tl.bu(f));
                f += b;
            }
            assert_eq!(fed, t.u(tl.bu(f)), "telescoping identity violated");
        }
    }

    #[test]
    fn pairwise_musical_phase_bound_holds_everywhere() {
        // bu − u_i·O_i ∈ [0, O_i): the sample-lock bound (spec §1.1) as an
        // algebraic identity, checked across tempo changes and odd frames.
        let mut tl = Timeline::new(120_000).unwrap();
        tl.set_tempo(1_000_000, 87_000).unwrap();
        tl.set_tempo(2_500_000, 174_000).unwrap();
        let tracks = [sf_voice(), humboldt_bass(), metronome_120()];
        let mut f = 1u64;
        while f < 10_000_000 {
            let bu = tl.bu(f);
            for t in &tracks {
                let phase = bu - t.u(bu) as u128 * t.original_tempo_mbpm() as u128;
                assert!(phase < t.original_tempo_mbpm() as u128);
            }
            f = f * 3 / 2 + 17; // irregular sweep
        }
    }

    #[test]
    fn wrap_count_consistent_with_wrap_of_sample() {
        let tl = Timeline::new(120_000).unwrap();
        let t = humboldt_bass();
        for f in [0u64, 12_345, 1_000_000, 50_000_000] {
            let bu = tl.bu(f);
            // floor(bu/Λ) and floor(u·O/Λ) may differ transiently only when u
            // truncation crosses a wrap edge; both must be within 1 of each other.
            let wc = t.wrap_count(bu);
            let ws = t.wrap_of_sample(t.u(bu));
            assert!(wc == ws || wc == ws + 1, "wrap counts inconsistent: {wc} vs {ws}");
        }
    }

    #[test]
    fn display_helpers_are_sane() {
        let tl = Timeline::new(120_000).unwrap();
        let (whole, frac) = beats_display(tl.bu(36_000)); // 1.5 beats at 120 BPM
        assert_eq!(whole, 1);
        assert!((frac - 0.5).abs() < 1e-6);
        let t = metronome_120();
        let p = loop_progress_display(&t, tl.bu(0));
        assert_eq!(p, 0.0);
    }
}
