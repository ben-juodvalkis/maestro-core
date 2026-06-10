#!/usr/bin/env python3
"""Generate metronome_60_var.wav — the variation-switch fixture (spec §6.4 M1).

Built FROM metronome_60.wav so the musical identity (16 beats @ 60 BPM @
48 kHz stereo = 768,000 frames) and — critically — every click's **attack**
are bit-identical to variation 0. The variation adds a clearly audible
1.8 kHz burst 30 ms after each beat start (inside the onset detector's
min-gap window, so detection still sees exactly one onset per beat at exactly
the original attack): the analyzer's onset offsets are content-invariant
across the variation switch, while the audio is unmistakably different.

Writes both copies:
  maestro-core/tests/fixtures/metronome_60_var.wav
  capacitor-plugin-maestro-audio/ios/Resources/metronome_60_var.wav
"""

import math
import os
import struct

SR = 48_000
BEATS = 16
BEAT_FRAMES = SR  # 60 BPM
TOTAL = BEATS * BEAT_FRAMES
BURST_HZ = 1_800.0
BURST_PEAK = 0.25
BURST_DECAY_S = 0.004
BURST_DELAY = int(0.030 * SR)
BURST_LEN = int(0.020 * SR)


def read_float_wav(path: str):
    """Minimal RIFF float32 reader, arbitrary-chunk tolerant (JUNK etc.)."""
    with open(path, "rb") as f:
        data = f.read()
    assert data[:4] == b"RIFF" and data[8:12] == b"WAVE", path
    pos = 12
    fmt = None
    samples = None
    while pos + 8 <= len(data):
        cid = data[pos : pos + 4]
        size = struct.unpack("<I", data[pos + 4 : pos + 8])[0]
        body = data[pos + 8 : pos + 8 + size]
        if cid == b"fmt ":
            fmt = struct.unpack("<HHIIHH", body[:16])
        elif cid == b"data":
            samples = body
        pos += 8 + size + (size & 1)
    assert fmt is not None and samples is not None, path
    audio_format, channels, rate, _, _, bits = fmt
    assert audio_format == 3 and bits == 32, f"{path}: not float32 (fmt {fmt})"
    assert rate == SR and channels == 2, f"{path}: {rate} Hz / {channels} ch"
    n = len(samples) // 8
    flat = struct.unpack(f"<{n * 2}f", samples)
    left = list(flat[0::2])
    right = list(flat[1::2])
    return left, right


def write_wav(path: str, left: list, right: list) -> None:
    n = len(left)
    data_bytes = n * 2 * 4
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 4 + 8 + 16 + 8 + data_bytes))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH", 16, 3, 2, SR, SR * 2 * 4, 8, 32))
        f.write(b"data")
        f.write(struct.pack("<I", data_bytes))
        out = bytearray()
        for l_s, r_s in zip(left, right):
            out += struct.pack("<ff", l_s, r_s)
        f.write(out)


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(os.path.dirname(here))
    src = os.path.join(repo, "maestro-core/tests/fixtures/metronome_60.wav")
    left, right = read_float_wav(src)
    assert len(left) == TOTAL, f"unexpected fixture length {len(left)}"

    for b in range(BEATS):
        start = b * BEAT_FRAMES + BURST_DELAY
        for k in range(BURST_LEN):
            t = k / SR
            s = BURST_PEAK * math.sin(2 * math.pi * BURST_HZ * t) * math.exp(-t / BURST_DECAY_S)
            left[start + k] += s
            right[start + k] += s

    for rel in (
        "maestro-core/tests/fixtures/metronome_60_var.wav",
        "capacitor-plugin-maestro-audio/ios/Resources/metronome_60_var.wav",
    ):
        path = os.path.join(repo, rel)
        write_wav(path, left, right)
        print(f"wrote {path} ({TOTAL} frames)")


if __name__ == "__main__":
    main()
