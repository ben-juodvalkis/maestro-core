//! PCM cache — in-memory storage for decoded audio data.
//!
//! Stores decoded PCM (pulse-code modulation) audio data keyed by
//! `(track_id, variation_type)`.  The platform layer decodes audio files
//! (via symphonia on Rust, AVFoundation on iOS, Web Audio on browser)
//! and stores the resulting float samples here.  The frame reader then
//! reads from this cache during playback.
//!
//! Ported from:
//! - `src/domains/core/audio/pcm-cache-manager.ts` (PCMCacheManager)
//!
//! Browser-specific concerns stripped: IndexedDB persistence, AudioContext
//! buffer reconstruction, progressive network loading.  The Rust core uses
//! an in-memory HashMap; persistence is a platform-layer concern.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Cache version string for invalidation across releases.
pub const CACHE_VERSION: &str = "1.0.0";

// ---------------------------------------------------------------------------
// Variation type
// ---------------------------------------------------------------------------

/// Stem variation complexity level.
///
/// Each track can have up to three variations (simple, default, complex)
/// representing different levels of musical complexity for the same stem.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VariationType {
    /// Simplified arrangement — fewer notes / instruments.
    Simple,
    /// Standard arrangement.
    Default,
    /// Complex arrangement — more notes / embellishments.
    Complex,
}

impl VariationType {
    /// All variation types, in order.
    pub const ALL: [VariationType; 3] = [
        VariationType::Simple,
        VariationType::Default,
        VariationType::Complex,
    ];
}

impl std::fmt::Display for VariationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VariationType::Simple => write!(f, "simple"),
            VariationType::Default => write!(f, "default"),
            VariationType::Complex => write!(f, "complex"),
        }
    }
}

// ---------------------------------------------------------------------------
// Gapless metadata
// ---------------------------------------------------------------------------

/// Metadata for gapless playback — compensates for encoder delay and padding.
///
/// M4A/AAC encoders add silent samples at the start (encoder delay) and end
/// (end padding).  This metadata lets the frame reader skip those samples
/// for sample-accurate looping.
#[derive(Debug, Clone, PartialEq)]
pub struct GaplessMetadata {
    /// Number of original audio samples (excluding encoder artifacts).
    pub original_samples: u64,
    /// Encoder delay in samples (silence prepended by the encoder).
    pub encoder_delay: u32,
    /// End padding in samples (silence appended by the encoder).
    pub end_padding: u32,
}

// ---------------------------------------------------------------------------
// PCM data
// ---------------------------------------------------------------------------

/// Decoded audio data — interleaved per-channel float samples.
///
/// Layout: `channel_data[channel_index][sample_index]` where each sample
/// is a 32-bit float in the range `[-1.0, 1.0]`.
#[derive(Debug, Clone)]
pub struct PcmData {
    /// Per-channel sample data.  `channel_data[0]` = left, `[1]` = right.
    pub channel_data: Vec<Vec<f32>>,
    /// Sample rate in Hz (typically 48000).
    pub sample_rate: u32,
    /// Number of channels (typically 2 for stereo).
    pub num_channels: u32,
    /// Total number of samples per channel.
    pub length_samples: u64,
}

impl PcmData {
    /// Total size in bytes across all channels.
    pub fn size_bytes(&self) -> usize {
        self.channel_data
            .iter()
            .map(|ch| ch.len() * std::mem::size_of::<f32>())
            .sum()
    }

    /// Duration in seconds.
    pub fn duration_seconds(&self) -> f64 {
        if self.sample_rate == 0 {
            return 0.0;
        }
        self.length_samples as f64 / self.sample_rate as f64
    }
}

// ---------------------------------------------------------------------------
// Cache entry
// ---------------------------------------------------------------------------

/// A single cached PCM entry, keyed by `(track_id, variation_type)`.
#[derive(Debug, Clone)]
pub struct PcmCacheEntry {
    /// Track identifier.
    pub track_id: String,
    /// Variation type (simple / default / complex).
    pub variation_type: VariationType,
    /// Source URL this data was decoded from (for cache invalidation).
    pub url: String,
    /// Decoded audio data.
    pub pcm_data: PcmData,
    /// Optional gapless playback metadata.
    pub gapless_metadata: Option<GaplessMetadata>,
    /// Timestamp when this entry was stored (epoch millis).
    pub timestamp: u64,
    /// Cache version for invalidation.
    pub version: String,
}

// ---------------------------------------------------------------------------
// Cache statistics
// ---------------------------------------------------------------------------

/// Summary statistics for the PCM cache.
#[derive(Debug, Clone, PartialEq)]
pub struct CacheStats {
    /// Number of distinct tracks with at least one cached variation.
    pub total_tracks: usize,
    /// Total number of cached variations across all tracks.
    pub total_variations: usize,
    /// Total memory usage in bytes.
    pub total_bytes: usize,
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that can occur during PCM cache operations.
#[derive(Debug, Clone, PartialEq)]
pub enum PcmCacheError {
    /// No cached data for the requested track + variation.
    NotCached {
        track_id: String,
        variation_type: VariationType,
    },
    /// Channel index out of bounds.
    InvalidChannel {
        requested: u32,
        available: u32,
    },
    /// PCM data is invalid (e.g., mismatched channel count / length).
    InvalidData(String),
}

impl std::fmt::Display for PcmCacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PcmCacheError::NotCached {
                track_id,
                variation_type,
            } => {
                write!(
                    f,
                    "No cached PCM for track '{track_id}', variation '{variation_type}'"
                )
            }
            PcmCacheError::InvalidChannel {
                requested,
                available,
            } => {
                write!(
                    f,
                    "Channel {requested} out of bounds (available: {available})"
                )
            }
            PcmCacheError::InvalidData(msg) => {
                write!(f, "Invalid PCM data: {msg}")
            }
        }
    }
}

impl std::error::Error for PcmCacheError {}

// ---------------------------------------------------------------------------
// PcmCache
// ---------------------------------------------------------------------------

/// In-memory cache for decoded PCM audio data.
///
/// Keyed by `(track_id, variation_type)`.  Each track can have up to
/// three variations (simple, default, complex).
pub struct PcmCache {
    entries: HashMap<(String, VariationType), PcmCacheEntry>,
}

impl PcmCache {
    /// Create an empty cache.
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Store decoded PCM data in the cache.
    ///
    /// Overwrites any existing entry for the same `(track_id, variation_type)`.
    /// Returns `Err` if the PCM data is invalid (e.g., channel count mismatch).
    pub fn store(
        &mut self,
        track_id: &str,
        variation_type: VariationType,
        pcm_data: PcmData,
        url: &str,
        gapless_metadata: Option<GaplessMetadata>,
        timestamp: u64,
    ) -> Result<(), PcmCacheError> {
        // Validate channel data consistency
        if pcm_data.num_channels as usize != pcm_data.channel_data.len() {
            return Err(PcmCacheError::InvalidData(format!(
                "num_channels ({}) does not match channel_data length ({})",
                pcm_data.num_channels,
                pcm_data.channel_data.len()
            )));
        }
        for (i, ch) in pcm_data.channel_data.iter().enumerate() {
            if ch.len() as u64 != pcm_data.length_samples {
                return Err(PcmCacheError::InvalidData(format!(
                    "channel {} has {} samples, expected {}",
                    i,
                    ch.len(),
                    pcm_data.length_samples
                )));
            }
        }

        let entry = PcmCacheEntry {
            track_id: track_id.to_string(),
            variation_type,
            url: url.to_string(),
            pcm_data,
            gapless_metadata,
            timestamp,
            version: CACHE_VERSION.to_string(),
        };

        self.entries
            .insert((track_id.to_string(), variation_type), entry);
        Ok(())
    }

    /// Get a reference to cached PCM data.
    pub fn get(
        &self,
        track_id: &str,
        variation_type: VariationType,
    ) -> Result<&PcmData, PcmCacheError> {
        self.entries
            .get(&(track_id.to_string(), variation_type))
            .map(|entry| &entry.pcm_data)
            .ok_or_else(|| PcmCacheError::NotCached {
                track_id: track_id.to_string(),
                variation_type,
            })
    }

    /// Get a reference to the full cache entry (including metadata).
    pub fn get_entry(
        &self,
        track_id: &str,
        variation_type: VariationType,
    ) -> Result<&PcmCacheEntry, PcmCacheError> {
        self.entries
            .get(&(track_id.to_string(), variation_type))
            .ok_or_else(|| PcmCacheError::NotCached {
                track_id: track_id.to_string(),
                variation_type,
            })
    }

    /// Check if a variation is cached.
    pub fn has(&self, track_id: &str, variation_type: VariationType) -> bool {
        self.entries
            .contains_key(&(track_id.to_string(), variation_type))
    }

    /// Check if a variation is cached with a specific URL.
    ///
    /// Returns `false` if not cached or if the cached URL differs
    /// (indicating the source file changed and the cache is stale).
    pub fn has_with_url(
        &self,
        track_id: &str,
        variation_type: VariationType,
        url: &str,
    ) -> bool {
        self.entries
            .get(&(track_id.to_string(), variation_type))
            .is_some_and(|entry| entry.url == url)
    }

    /// Get a slice of channel data for direct reading.
    ///
    /// This is the primary access path for the frame reader.
    pub fn get_channel_data(
        &self,
        track_id: &str,
        variation_type: VariationType,
        channel: u32,
    ) -> Result<&[f32], PcmCacheError> {
        let pcm = self.get(track_id, variation_type)?;
        if channel >= pcm.num_channels {
            return Err(PcmCacheError::InvalidChannel {
                requested: channel,
                available: pcm.num_channels,
            });
        }
        Ok(&pcm.channel_data[channel as usize])
    }

    /// Remove all cached variations for a track.
    ///
    /// Returns the number of entries removed.
    pub fn clear_track(&mut self, track_id: &str) -> usize {
        let keys_to_remove: Vec<_> = self
            .entries
            .keys()
            .filter(|(id, _)| id == track_id)
            .cloned()
            .collect();
        let count = keys_to_remove.len();
        for key in keys_to_remove {
            self.entries.remove(&key);
        }
        count
    }

    /// Remove all entries from the cache.
    pub fn clear_all(&mut self) {
        self.entries.clear();
    }

    /// Get cache statistics.
    pub fn get_stats(&self) -> CacheStats {
        let mut track_ids = std::collections::HashSet::new();
        let mut total_bytes: usize = 0;

        for entry in self.entries.values() {
            track_ids.insert(&entry.track_id);
            total_bytes += entry.pcm_data.size_bytes();
        }

        CacheStats {
            total_tracks: track_ids.len(),
            total_variations: self.entries.len(),
            total_bytes,
        }
    }

    /// Get all cached variation types for a track.
    pub fn get_track_variations(&self, track_id: &str) -> Vec<VariationType> {
        self.entries
            .keys()
            .filter(|(id, _)| id == track_id)
            .map(|(_, vt)| *vt)
            .collect()
    }

    /// Total number of cached entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Default for PcmCache {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn make_stereo_pcm(length: u64, sample_rate: u32) -> PcmData {
        PcmData {
            channel_data: vec![vec![0.0_f32; length as usize]; 2],
            sample_rate,
            num_channels: 2,
            length_samples: length,
        }
    }

    fn make_mono_pcm(length: u64, sample_rate: u32) -> PcmData {
        PcmData {
            channel_data: vec![vec![0.0_f32; length as usize]],
            sample_rate,
            num_channels: 1,
            length_samples: length,
        }
    }

    fn make_gapless() -> GaplessMetadata {
        GaplessMetadata {
            original_samples: 48000,
            encoder_delay: 1024,
            end_padding: 512,
        }
    }

    // -----------------------------------------------------------------------
    // VariationType
    // -----------------------------------------------------------------------

    #[test]
    fn variation_type_display() {
        assert_eq!(VariationType::Simple.to_string(), "simple");
        assert_eq!(VariationType::Default.to_string(), "default");
        assert_eq!(VariationType::Complex.to_string(), "complex");
    }

    #[test]
    fn variation_type_all() {
        assert_eq!(VariationType::ALL.len(), 3);
        assert_eq!(VariationType::ALL[0], VariationType::Simple);
        assert_eq!(VariationType::ALL[1], VariationType::Default);
        assert_eq!(VariationType::ALL[2], VariationType::Complex);
    }

    #[test]
    fn variation_type_equality_and_hash() {
        let mut set = std::collections::HashSet::new();
        set.insert(VariationType::Simple);
        set.insert(VariationType::Simple); // duplicate
        assert_eq!(set.len(), 1);
        set.insert(VariationType::Default);
        assert_eq!(set.len(), 2);
    }

    // -----------------------------------------------------------------------
    // PcmData
    // -----------------------------------------------------------------------

    #[test]
    fn pcm_data_size_bytes() {
        let pcm = make_stereo_pcm(1000, 48000);
        // 2 channels × 1000 samples × 4 bytes/sample = 8000
        assert_eq!(pcm.size_bytes(), 8000);
    }

    #[test]
    fn pcm_data_duration() {
        let pcm = make_stereo_pcm(48000, 48000);
        assert!((pcm.duration_seconds() - 1.0).abs() < 0.001);
    }

    #[test]
    fn pcm_data_duration_zero_rate() {
        let pcm = PcmData {
            channel_data: vec![],
            sample_rate: 0,
            num_channels: 0,
            length_samples: 100,
        };
        assert_eq!(pcm.duration_seconds(), 0.0);
    }

    // -----------------------------------------------------------------------
    // Store and retrieve
    // -----------------------------------------------------------------------

    #[test]
    fn store_and_get() {
        let mut cache = PcmCache::new();
        let pcm = make_stereo_pcm(48000, 48000);
        cache
            .store("t1", VariationType::Default, pcm, "http://a.com/f.m4a", None, 1000)
            .unwrap();

        let data = cache.get("t1", VariationType::Default).unwrap();
        assert_eq!(data.length_samples, 48000);
        assert_eq!(data.num_channels, 2);
    }

    #[test]
    fn store_with_gapless() {
        let mut cache = PcmCache::new();
        let pcm = make_stereo_pcm(48000, 48000);
        let gapless = make_gapless();
        cache
            .store(
                "t1",
                VariationType::Simple,
                pcm,
                "http://a.com/f.m4a",
                Some(gapless.clone()),
                1000,
            )
            .unwrap();

        let entry = cache.get_entry("t1", VariationType::Simple).unwrap();
        assert_eq!(entry.gapless_metadata.as_ref().unwrap().encoder_delay, 1024);
        assert_eq!(entry.gapless_metadata.as_ref().unwrap().end_padding, 512);
    }

    #[test]
    fn store_overwrites_existing() {
        let mut cache = PcmCache::new();
        let pcm1 = make_stereo_pcm(1000, 48000);
        let pcm2 = make_stereo_pcm(2000, 48000);

        cache
            .store("t1", VariationType::Default, pcm1, "url1", None, 1000)
            .unwrap();
        cache
            .store("t1", VariationType::Default, pcm2, "url2", None, 2000)
            .unwrap();

        let data = cache.get("t1", VariationType::Default).unwrap();
        assert_eq!(data.length_samples, 2000);
    }

    #[test]
    fn store_invalid_channel_count_mismatch() {
        let mut cache = PcmCache::new();
        let pcm = PcmData {
            channel_data: vec![vec![0.0; 100]], // 1 channel
            sample_rate: 48000,
            num_channels: 2, // claims 2 channels
            length_samples: 100,
        };
        let result = cache.store("t1", VariationType::Default, pcm, "url", None, 0);
        assert!(matches!(result, Err(PcmCacheError::InvalidData(_))));
    }

    #[test]
    fn store_invalid_sample_length_mismatch() {
        let mut cache = PcmCache::new();
        let pcm = PcmData {
            channel_data: vec![vec![0.0; 50], vec![0.0; 50]],
            sample_rate: 48000,
            num_channels: 2,
            length_samples: 100, // claims 100 but channels have 50
        };
        let result = cache.store("t1", VariationType::Default, pcm, "url", None, 0);
        assert!(matches!(result, Err(PcmCacheError::InvalidData(_))));
    }

    // -----------------------------------------------------------------------
    // has / has_with_url
    // -----------------------------------------------------------------------

    #[test]
    fn has_returns_true_when_cached() {
        let mut cache = PcmCache::new();
        cache
            .store("t1", VariationType::Default, make_stereo_pcm(100, 48000), "url", None, 0)
            .unwrap();

        assert!(cache.has("t1", VariationType::Default));
        assert!(!cache.has("t1", VariationType::Simple));
        assert!(!cache.has("t2", VariationType::Default));
    }

    #[test]
    fn has_with_url_matching() {
        let mut cache = PcmCache::new();
        cache
            .store("t1", VariationType::Default, make_stereo_pcm(100, 48000), "url_a", None, 0)
            .unwrap();

        assert!(cache.has_with_url("t1", VariationType::Default, "url_a"));
        assert!(!cache.has_with_url("t1", VariationType::Default, "url_b"));
        assert!(!cache.has_with_url("t2", VariationType::Default, "url_a"));
    }

    // -----------------------------------------------------------------------
    // get_channel_data
    // -----------------------------------------------------------------------

    #[test]
    fn get_channel_data_valid() {
        let mut cache = PcmCache::new();
        let mut pcm = make_stereo_pcm(4, 48000);
        pcm.channel_data[0] = vec![0.1, 0.2, 0.3, 0.4];
        pcm.channel_data[1] = vec![0.5, 0.6, 0.7, 0.8];

        cache
            .store("t1", VariationType::Default, pcm, "url", None, 0)
            .unwrap();

        let left = cache.get_channel_data("t1", VariationType::Default, 0).unwrap();
        assert_eq!(left, &[0.1, 0.2, 0.3, 0.4]);

        let right = cache.get_channel_data("t1", VariationType::Default, 1).unwrap();
        assert_eq!(right, &[0.5, 0.6, 0.7, 0.8]);
    }

    #[test]
    fn get_channel_data_invalid_channel() {
        let mut cache = PcmCache::new();
        cache
            .store("t1", VariationType::Default, make_stereo_pcm(100, 48000), "url", None, 0)
            .unwrap();

        let result = cache.get_channel_data("t1", VariationType::Default, 2);
        assert!(matches!(
            result,
            Err(PcmCacheError::InvalidChannel {
                requested: 2,
                available: 2
            })
        ));
    }

    #[test]
    fn get_channel_data_not_cached() {
        let cache = PcmCache::new();
        let result = cache.get_channel_data("t1", VariationType::Default, 0);
        assert!(matches!(result, Err(PcmCacheError::NotCached { .. })));
    }

    // -----------------------------------------------------------------------
    // clear_track / clear_all
    // -----------------------------------------------------------------------

    #[test]
    fn clear_track_removes_all_variations() {
        let mut cache = PcmCache::new();
        for vt in VariationType::ALL {
            cache
                .store("t1", vt, make_stereo_pcm(100, 48000), "url", None, 0)
                .unwrap();
        }
        cache
            .store("t2", VariationType::Default, make_stereo_pcm(100, 48000), "url", None, 0)
            .unwrap();

        let removed = cache.clear_track("t1");
        assert_eq!(removed, 3);
        assert!(!cache.has("t1", VariationType::Default));
        assert!(cache.has("t2", VariationType::Default));
    }

    #[test]
    fn clear_track_nonexistent() {
        let mut cache = PcmCache::new();
        assert_eq!(cache.clear_track("ghost"), 0);
    }

    #[test]
    fn clear_all() {
        let mut cache = PcmCache::new();
        cache
            .store("t1", VariationType::Default, make_stereo_pcm(100, 48000), "url", None, 0)
            .unwrap();
        cache
            .store("t2", VariationType::Simple, make_stereo_pcm(100, 48000), "url", None, 0)
            .unwrap();

        cache.clear_all();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    // -----------------------------------------------------------------------
    // get_stats
    // -----------------------------------------------------------------------

    #[test]
    fn stats_empty() {
        let cache = PcmCache::new();
        let stats = cache.get_stats();
        assert_eq!(
            stats,
            CacheStats {
                total_tracks: 0,
                total_variations: 0,
                total_bytes: 0,
            }
        );
    }

    #[test]
    fn stats_with_entries() {
        let mut cache = PcmCache::new();
        // Track 1: 2 variations, stereo, 1000 samples each → 2×(2×1000×4) = 16000 bytes
        cache
            .store("t1", VariationType::Default, make_stereo_pcm(1000, 48000), "url", None, 0)
            .unwrap();
        cache
            .store("t1", VariationType::Simple, make_stereo_pcm(1000, 48000), "url", None, 0)
            .unwrap();
        // Track 2: 1 variation, mono, 500 samples → 1×500×4 = 2000 bytes
        cache
            .store("t2", VariationType::Complex, make_mono_pcm(500, 48000), "url", None, 0)
            .unwrap();

        let stats = cache.get_stats();
        assert_eq!(stats.total_tracks, 2);
        assert_eq!(stats.total_variations, 3);
        assert_eq!(stats.total_bytes, 18000);
    }

    // -----------------------------------------------------------------------
    // get_track_variations
    // -----------------------------------------------------------------------

    #[test]
    fn get_track_variations_returns_cached() {
        let mut cache = PcmCache::new();
        cache
            .store("t1", VariationType::Default, make_stereo_pcm(100, 48000), "url", None, 0)
            .unwrap();
        cache
            .store("t1", VariationType::Complex, make_stereo_pcm(100, 48000), "url", None, 0)
            .unwrap();

        let mut variations = cache.get_track_variations("t1");
        variations.sort_by_key(|v| format!("{v}"));
        assert_eq!(variations.len(), 2);
    }

    #[test]
    fn get_track_variations_empty() {
        let cache = PcmCache::new();
        assert!(cache.get_track_variations("nonexistent").is_empty());
    }

    // -----------------------------------------------------------------------
    // len / is_empty / Default
    // -----------------------------------------------------------------------

    #[test]
    fn len_and_is_empty() {
        let mut cache = PcmCache::new();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);

        cache
            .store("t1", VariationType::Default, make_stereo_pcm(100, 48000), "url", None, 0)
            .unwrap();
        assert!(!cache.is_empty());
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn default_impl() {
        let cache = PcmCache::default();
        assert!(cache.is_empty());
    }

    // -----------------------------------------------------------------------
    // Error display
    // -----------------------------------------------------------------------

    #[test]
    fn error_display() {
        let e = PcmCacheError::NotCached {
            track_id: "t1".to_string(),
            variation_type: VariationType::Default,
        };
        assert!(e.to_string().contains("t1"));
        assert!(e.to_string().contains("default"));

        let e = PcmCacheError::InvalidChannel {
            requested: 5,
            available: 2,
        };
        assert!(e.to_string().contains('5'));
        assert!(e.to_string().contains('2'));

        let e = PcmCacheError::InvalidData("bad".to_string());
        assert!(e.to_string().contains("bad"));
    }

    // -----------------------------------------------------------------------
    // get_entry (full entry with metadata)
    // -----------------------------------------------------------------------

    #[test]
    fn get_entry_includes_url_and_version() {
        let mut cache = PcmCache::new();
        cache
            .store(
                "t1",
                VariationType::Default,
                make_stereo_pcm(100, 48000),
                "http://cdn.example.com/track.m4a",
                None,
                42,
            )
            .unwrap();

        let entry = cache.get_entry("t1", VariationType::Default).unwrap();
        assert_eq!(entry.url, "http://cdn.example.com/track.m4a");
        assert_eq!(entry.timestamp, 42);
        assert_eq!(entry.version, CACHE_VERSION);
    }

    #[test]
    fn get_entry_not_cached() {
        let cache = PcmCache::new();
        let result = cache.get_entry("t1", VariationType::Default);
        assert!(matches!(result, Err(PcmCacheError::NotCached { .. })));
    }
}
