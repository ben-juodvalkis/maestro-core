//! # validation — shared analysis utilities (spec §7 reuse map)
//!
//! `detect_click_onsets` promoted from the abandoned branch's test code to a
//! shared utility, used by the offline suites (T-SEAM), and the
//! `maestro-validate` device-capture analyzer (spec §6.4). Plus a minimal
//! float-WAV reader for fixtures and device captures.
//!
//! Test/tooling code only — never on the render path.

use std::io::Read;
use std::path::Path;

/// A decoded float WAV: per-channel PCM.
pub struct LoadedWav {
    /// Per-channel PCM data, each of length `frames`.
    pub channels: Vec<Vec<f32>>,
    pub sample_rate: u32,
    pub num_channels: u32,
    pub frames: usize,
}

/// Minimal RIFF/WAVE reader for 32-bit IEEE float PCM (format tag 3, or
/// WAVE_FORMAT_EXTENSIBLE wrapping it). Walks chunks, so `JUNK` blocks
/// (Logic Pro) and appended chunks (AVAudioFile) are handled.
pub fn load_float_wav(path: &Path) -> std::io::Result<LoadedWav> {
    let mut f = std::fs::File::open(path)?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)?;
    let err = |msg: String| std::io::Error::new(std::io::ErrorKind::InvalidData, msg);

    if buf.len() < 12 || &buf[0..4] != b"RIFF" || &buf[8..12] != b"WAVE" {
        return Err(err(format!("{path:?}: not a RIFF/WAVE file")));
    }

    let mut pos: usize = 12;
    let mut fmt_tag: u16 = 0;
    let mut num_channels: u32 = 0;
    let mut sample_rate: u32 = 0;
    let mut bits_per_sample: u16 = 0;
    let mut data_offset: usize = 0;
    let mut data_size: usize = 0;

    while pos + 8 <= buf.len() {
        let cid = &buf[pos..pos + 4];
        let csize = u32::from_le_bytes(buf[pos + 4..pos + 8].try_into().unwrap()) as usize;
        let body = pos + 8;
        if body + csize > buf.len() && cid != b"data" {
            break;
        }
        match cid {
            b"fmt " => {
                fmt_tag = u16::from_le_bytes(buf[body..body + 2].try_into().unwrap());
                num_channels = u16::from_le_bytes(buf[body + 2..body + 4].try_into().unwrap()) as u32;
                sample_rate = u32::from_le_bytes(buf[body + 4..body + 8].try_into().unwrap());
                bits_per_sample = u16::from_le_bytes(buf[body + 14..body + 16].try_into().unwrap());
                // WAVE_FORMAT_EXTENSIBLE: the real format is in the SubFormat
                // GUID's first two bytes (3 = IEEE float).
                if fmt_tag == 0xFFFE && csize >= 40 {
                    fmt_tag = u16::from_le_bytes(buf[body + 24..body + 26].try_into().unwrap());
                }
            }
            b"data" => {
                data_offset = body;
                // AVAudioFile may leave a placeholder size; clamp to the file.
                data_size = csize.min(buf.len() - body);
                break;
            }
            _ => {}
        }
        pos = body + csize + (csize & 1);
    }

    if fmt_tag != 3 {
        return Err(err(format!("{path:?}: expected IEEE float (tag 3), got {fmt_tag}")));
    }
    if bits_per_sample != 32 {
        return Err(err(format!("{path:?}: expected 32-bit float, got {bits_per_sample}")));
    }
    if data_offset == 0 || data_size == 0 || num_channels == 0 {
        return Err(err(format!("{path:?}: missing fmt/data chunk")));
    }

    let bytes_per_sample = (bits_per_sample / 8) as usize;
    let frames = data_size / (num_channels as usize * bytes_per_sample);
    let mut channels: Vec<Vec<f32>> =
        (0..num_channels).map(|_| Vec::with_capacity(frames)).collect();
    for frame in 0..frames {
        for (ch, channel) in channels.iter_mut().enumerate() {
            let off = data_offset + (frame * num_channels as usize + ch) * bytes_per_sample;
            channel.push(f32::from_le_bytes(buf[off..off + 4].try_into().unwrap()));
        }
    }

    Ok(LoadedWav { channels, sample_rate, num_channels, frames })
}

/// Write a stereo float32 WAV (format tag 3). Used by offline smoke tests to
/// produce analyzer-compatible captures without device hardware.
pub fn write_float_wav_stereo(
    path: &Path,
    left: &[f32],
    right: &[f32],
    sample_rate: u32,
) -> std::io::Result<()> {
    use std::io::Write;
    assert_eq!(left.len(), right.len());
    let frames = left.len();
    let data_bytes = (frames * 2 * 4) as u32;
    let mut out = std::io::BufWriter::new(std::fs::File::create(path)?);
    out.write_all(b"RIFF")?;
    out.write_all(&(36 + data_bytes).to_le_bytes())?;
    out.write_all(b"WAVE")?;
    out.write_all(b"fmt ")?;
    out.write_all(&16u32.to_le_bytes())?;
    out.write_all(&3u16.to_le_bytes())?; // IEEE float
    out.write_all(&2u16.to_le_bytes())?;
    out.write_all(&sample_rate.to_le_bytes())?;
    out.write_all(&(sample_rate * 2 * 4).to_le_bytes())?;
    out.write_all(&8u16.to_le_bytes())?; // block align
    out.write_all(&32u16.to_le_bytes())?;
    out.write_all(b"data")?;
    out.write_all(&data_bytes.to_le_bytes())?;
    for i in 0..frames {
        out.write_all(&left[i].to_le_bytes())?;
        out.write_all(&right[i].to_le_bytes())?;
    }
    Ok(())
}

/// Detect click onset frames in one channel. An onset is the first frame
/// above `threshold` after a quiet gap of at least `min_gap_frames`.
pub fn detect_click_onsets(channel: &[f32], threshold: f32, min_gap_frames: usize) -> Vec<usize> {
    let mut onsets = Vec::new();
    let mut last_onset: i64 = -(min_gap_frames as i64);
    for (frame, s) in channel.iter().enumerate() {
        if s.abs() > threshold && (frame as i64) - last_onset >= min_gap_frames as i64 {
            onsets.push(frame);
            last_onset = frame as i64;
        }
    }
    onsets
}

/// Least-squares slope of `ys` against their indices, in units of
/// y-per-index. Used by the device analyzer to assert zero drift (slope of
/// cross-stem offset over onset index).
pub fn linear_slope(ys: &[i64]) -> f64 {
    let n = ys.len() as f64;
    if ys.len() < 2 {
        return 0.0;
    }
    let mean_x = (ys.len() as f64 - 1.0) / 2.0;
    let mean_y = ys.iter().map(|&y| y as f64).sum::<f64>() / n;
    let mut num = 0.0;
    let mut den = 0.0;
    for (x, &y) in ys.iter().enumerate() {
        let dx = x as f64 - mean_x;
        num += dx * (y as f64 - mean_y);
        den += dx * dx;
    }
    if den == 0.0 {
        0.0
    } else {
        num / den
    }
}
