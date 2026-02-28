//! Integration tests: Engine + mixer interactions.
//!
//! Tests volume/mute/solo operations through the engine API, verifying
//! that the engine properly coordinates between the track controls and
//! the mixer sub-system.

use maestro_core::engine::{EngineError, MaestroEngine};
use maestro_core::events::EngineEvent;
use maestro_core::pcm_cache::{PcmData, VariationType};
use maestro_core::transport::SafetyPreset;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_pcm(channels: u32, frames: usize, sample_rate: u32) -> PcmData {
    let channel_data: Vec<Vec<f32>> = (0..channels)
        .map(|_| vec![0.0f32; frames])
        .collect();
    PcmData {
        channel_data,
        sample_rate,
        num_channels: channels,
        length_samples: frames as u64,
    }
}

fn engine_with_tracks(names: &[&str]) -> MaestroEngine {
    let mut e = MaestroEngine::new(48000);
    e.initialize().unwrap();
    for name in names {
        e.create_track(name, &format!("url/{name}"), None, Some(120.0))
            .unwrap();
        e.load_track_audio(
            name,
            VariationType::Default,
            make_pcm(2, 48000 * 5, 48000),
            &format!("url/{name}"),
        )
        .unwrap();
        e.arm_track(name).unwrap();
    }
    e.drain_events();
    e
}

// ---------------------------------------------------------------------------
// Volume control through engine
// ---------------------------------------------------------------------------

#[test]
fn set_volume_updates_both_track_and_mixer() {
    let mut engine = engine_with_tracks(&["drums"]);

    engine.set_track_volume("drums", 0.4).unwrap();

    // Track controls updated
    let track = engine.track("drums").unwrap();
    assert!((track.controls().volume - 0.4).abs() < 0.01);

    // Mixer updated
    let mixer_state = engine.mixer().track_state("drums").unwrap();
    assert!((mixer_state.volume - 0.4).abs() < 0.01);
}

#[test]
fn volume_change_emits_event_with_correct_value() {
    let mut engine = engine_with_tracks(&["drums"]);
    engine.set_track_volume("drums", 0.7).unwrap();

    let events = engine.drain_events();
    let vol_event = events.iter().find_map(|e| {
        if let EngineEvent::TrackVolumeChanged { track_id, volume } = e {
            if track_id == "drums" {
                return Some(*volume);
            }
        }
        None
    });

    assert!(vol_event.is_some());
    assert!((vol_event.unwrap() - 0.7).abs() < 0.01);
}

#[test]
fn volume_on_nonexistent_track_fails() {
    let mut engine = engine_with_tracks(&["drums"]);
    let result = engine.set_track_volume("nope", 0.5);
    assert!(matches!(result, Err(EngineError::TrackNotFound(_))));
}

// ---------------------------------------------------------------------------
// Mute/unmute through engine
// ---------------------------------------------------------------------------

#[test]
fn mute_unmute_updates_mixer_state() {
    let mut engine = engine_with_tracks(&["drums"]);

    engine.mute_track("drums").unwrap();
    assert!(engine.mixer().track_state("drums").unwrap().muted);

    engine.unmute_track("drums").unwrap();
    assert!(!engine.mixer().track_state("drums").unwrap().muted);
}

#[test]
fn mute_nonexistent_track_fails() {
    let mut engine = engine_with_tracks(&["drums"]);
    assert!(matches!(
        engine.mute_track("nope"),
        Err(EngineError::TrackNotFound(_))
    ));
    assert!(matches!(
        engine.unmute_track("nope"),
        Err(EngineError::TrackNotFound(_))
    ));
}

// ---------------------------------------------------------------------------
// Mixer solo interaction through engine
// ---------------------------------------------------------------------------

#[test]
fn solo_via_mixer_direct_access() {
    let mut engine = engine_with_tracks(&["drums", "bass", "guitar"]);

    // Solo drums via direct mixer access
    engine.mixer_mut().solo_track("drums").unwrap();

    // Drums should be solo'd
    assert!(engine.mixer().track_state("drums").unwrap().solo);
    assert!(!engine.mixer().track_state("bass").unwrap().solo);
    assert!(!engine.mixer().track_state("guitar").unwrap().solo);

    // Unsolo
    engine.mixer_mut().unsolo_track("drums").unwrap();
    assert!(!engine.mixer().track_state("drums").unwrap().solo);
}

// ---------------------------------------------------------------------------
// Volume during playback
// ---------------------------------------------------------------------------

#[test]
fn volume_changes_during_playback() {
    let mut engine = engine_with_tracks(&["drums", "bass"]);
    engine.play(SafetyPreset::Permissive).unwrap();
    engine.drain_events();

    // Change volumes while playing
    engine.set_track_volume("drums", 0.2).unwrap();
    engine.set_track_volume("bass", 0.8).unwrap();

    // Verify both updated
    assert!((engine.track("drums").unwrap().controls().volume - 0.2).abs() < 0.01);
    assert!((engine.track("bass").unwrap().controls().volume - 0.8).abs() < 0.01);

    let events = engine.drain_events();
    let vol_events: Vec<_> = events
        .iter()
        .filter(|e| matches!(e, EngineEvent::TrackVolumeChanged { .. }))
        .collect();
    assert_eq!(vol_events.len(), 2);
}

// ---------------------------------------------------------------------------
// Mixer state queried via track info
// ---------------------------------------------------------------------------

#[test]
fn track_info_reflects_mute_and_solo() {
    let mut engine = engine_with_tracks(&["drums", "bass"]);

    engine.mute_track("drums").unwrap();
    engine.mixer_mut().solo_track("bass").unwrap();

    let drums_info = engine.get_track_info("drums").unwrap();
    assert!(drums_info.muted);
    assert!(!drums_info.solo);

    let bass_info = engine.get_track_info("bass").unwrap();
    assert!(!bass_info.muted);
    assert!(bass_info.solo);
}

// ---------------------------------------------------------------------------
// Master volume via mixer
// ---------------------------------------------------------------------------

#[test]
fn master_volume_adjustable_via_mixer() {
    let mut engine = engine_with_tracks(&["drums"]);

    assert!((engine.mixer().master_volume() - 1.0).abs() < 0.01);

    engine.mixer_mut().set_master_volume(0.5).unwrap();
    assert!((engine.mixer().master_volume() - 0.5).abs() < 0.01);
}

// ---------------------------------------------------------------------------
// Track removal unregisters from mixer
// ---------------------------------------------------------------------------

#[test]
fn removing_track_unregisters_from_mixer() {
    let mut engine = engine_with_tracks(&["drums", "bass"]);

    engine.remove_track("drums").unwrap();
    assert!(engine.mixer().track_state("drums").is_err());
    assert!(engine.mixer().track_state("bass").is_ok());
}
