//! Integration tests: Engine lifecycle and multi-track playback scenarios.
//!
//! These tests exercise the full `MaestroEngine` API through realistic
//! multi-step workflows, verifying that sub-systems (transport, mixer,
//! tempo, beat tracker, PCM cache) interact correctly.

use maestro_core::engine::{EngineError, MaestroEngine};
use maestro_core::events::{EngineEvent, EngineState};
use maestro_core::pcm_cache::{PcmData, VariationType};
use maestro_core::track::TrackStatus;
use maestro_core::transport::{SafetyPreset, TransportState};

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

fn ready_engine() -> MaestroEngine {
    let mut e = MaestroEngine::new(48000);
    e.initialize().unwrap();
    e.drain_events();
    e
}

fn engine_with_loaded_track(id: &str, tempo: f64) -> MaestroEngine {
    let mut e = ready_engine();
    e.create_track(id, &format!("http://example.com/{id}.m4a"), None, Some(tempo))
        .unwrap();
    e.load_track_audio(
        id,
        VariationType::Default,
        make_pcm(2, 48000 * 10, 48000), // 10 seconds stereo
        &format!("http://example.com/{id}.m4a"),
    )
    .unwrap();
    e.arm_track(id).unwrap();
    e.drain_events();
    e
}

// ---------------------------------------------------------------------------
// Full lifecycle: create → load → arm → play → pause → resume → stop → destroy
// ---------------------------------------------------------------------------

#[test]
fn full_lifecycle_single_track() {
    let mut engine = ready_engine();

    // Create
    engine
        .create_track("drums", "url/drums.m4a", None, Some(120.0))
        .unwrap();
    assert_eq!(engine.state(), EngineState::EnginesReady);

    // Load
    engine
        .load_track_audio("drums", VariationType::Default, make_pcm(2, 48000 * 5, 48000), "url/drums.m4a")
        .unwrap();
    assert_eq!(engine.state(), EngineState::Ready);
    assert_eq!(engine.track("drums").unwrap().status(), TrackStatus::Ready);

    // Arm
    engine.arm_track("drums").unwrap();
    assert!(engine.track("drums").unwrap().controls().armed);
    engine.drain_events();

    // Play — Permissive allows auto-start
    engine.play(SafetyPreset::Permissive).unwrap();
    assert_eq!(engine.transport_state(), TransportState::Playing);
    assert_eq!(engine.track("drums").unwrap().status(), TrackStatus::Playing);
    assert_eq!(engine.active_track_count(), 1);
    engine.drain_events();

    // Pause
    engine.pause().unwrap();
    assert_eq!(engine.transport_state(), TransportState::Paused);
    assert_eq!(engine.track("drums").unwrap().status(), TrackStatus::Paused);
    engine.drain_events();

    // Resume (play from paused)
    engine.play(SafetyPreset::Permissive).unwrap();
    assert_eq!(engine.transport_state(), TransportState::Playing);
    assert_eq!(engine.track("drums").unwrap().status(), TrackStatus::Playing);
    engine.drain_events();

    // Stop
    engine.stop().unwrap();
    assert_eq!(engine.transport_state(), TransportState::Stopped);
    assert_eq!(engine.track("drums").unwrap().status(), TrackStatus::Ready);
    assert_eq!(engine.active_track_count(), 0);

    // Destroy
    engine.destroy();
    assert_eq!(engine.state(), EngineState::Destroying);
    assert_eq!(engine.track_count(), 0);
}

// ---------------------------------------------------------------------------
// Multi-track: create multiple, arm subset, play
// ---------------------------------------------------------------------------

#[test]
fn multi_track_selective_arming() {
    let mut engine = ready_engine();

    // Create 4 tracks
    for name in &["drums", "bass", "guitar", "keys"] {
        engine
            .create_track(name, &format!("url/{name}.m4a"), None, Some(120.0))
            .unwrap();
        engine
            .load_track_audio(
                name,
                VariationType::Default,
                make_pcm(2, 48000 * 8, 48000),
                &format!("url/{name}.m4a"),
            )
            .unwrap();
    }
    assert_eq!(engine.track_count(), 4);
    assert_eq!(engine.state(), EngineState::Ready);

    // Arm only drums and bass
    engine.arm_track("drums").unwrap();
    engine.arm_track("bass").unwrap();
    engine.drain_events();

    // Play — only armed tracks start
    engine.play(SafetyPreset::Permissive).unwrap();
    assert_eq!(engine.track("drums").unwrap().status(), TrackStatus::Playing);
    assert_eq!(engine.track("bass").unwrap().status(), TrackStatus::Playing);
    assert_eq!(engine.track("guitar").unwrap().status(), TrackStatus::Ready);
    assert_eq!(engine.track("keys").unwrap().status(), TrackStatus::Ready);
    assert_eq!(engine.active_track_count(), 2);

    // Stop
    engine.stop().unwrap();
    assert_eq!(engine.track("drums").unwrap().status(), TrackStatus::Ready);
    assert_eq!(engine.track("bass").unwrap().status(), TrackStatus::Ready);
    assert!(engine.is_stopped());
}

#[test]
fn arm_additional_tracks_mid_playback() {
    let mut engine = engine_with_loaded_track("drums", 120.0);

    // Add and load a second track
    engine
        .create_track("bass", "url/bass.m4a", None, Some(120.0))
        .unwrap();
    engine
        .load_track_audio("bass", VariationType::Default, make_pcm(2, 48000 * 10, 48000), "url/bass.m4a")
        .unwrap();
    engine.drain_events();

    // Start with only drums
    engine.play(SafetyPreset::Permissive).unwrap();
    assert_eq!(engine.active_track_count(), 1);
    engine.drain_events();

    // Arm bass and start it manually
    engine.arm_track("bass").unwrap();
    // Bass is Ready + Armed, but engine is already playing — need to call play again
    // or handle individually. The engine's play() starts all armed Ready/Paused tracks.
    engine.play(SafetyPreset::Permissive).unwrap();
    assert_eq!(engine.track("bass").unwrap().status(), TrackStatus::Playing);
    assert_eq!(engine.active_track_count(), 2);
}

// ---------------------------------------------------------------------------
// Track removal during playback
// ---------------------------------------------------------------------------

#[test]
fn remove_playing_track_cleans_up_transport() {
    let mut engine = ready_engine();

    engine
        .create_track("drums", "url/drums", None, Some(120.0))
        .unwrap();
    engine
        .load_track_audio("drums", VariationType::Default, make_pcm(2, 48000 * 5, 48000), "url/drums")
        .unwrap();
    engine
        .create_track("bass", "url/bass", None, Some(120.0))
        .unwrap();
    engine
        .load_track_audio("bass", VariationType::Default, make_pcm(2, 48000 * 5, 48000), "url/bass")
        .unwrap();
    engine.arm_track("drums").unwrap();
    engine.arm_track("bass").unwrap();
    engine.drain_events();

    engine.play(SafetyPreset::Permissive).unwrap();
    assert_eq!(engine.active_track_count(), 2);
    engine.drain_events();

    // Remove one playing track
    engine.remove_track("drums").unwrap();
    assert_eq!(engine.active_track_count(), 1);
    assert_eq!(engine.track_count(), 1);
    assert!(engine.is_playing()); // bass still playing

    // Remove the last playing track — transport should auto-stop
    engine.remove_track("bass").unwrap();
    assert_eq!(engine.active_track_count(), 0);
    assert_eq!(engine.track_count(), 0);
    assert!(engine.is_stopped());
}

// ---------------------------------------------------------------------------
// Event ordering: verify events arrive in correct order
// ---------------------------------------------------------------------------

#[test]
fn play_emits_all_expected_events() {
    let mut engine = engine_with_loaded_track("drums", 120.0);
    engine.play(SafetyPreset::Permissive).unwrap();

    let events = engine.drain_events();
    let event_types: Vec<&str> = events.iter().map(|e| match e {
        EngineEvent::TrackStateChanged { .. } => "TrackStateChanged",
        EngineEvent::TrackStarted { .. } => "TrackStarted",
        EngineEvent::TrackPlayingStateChanged { .. } => "TrackPlayingStateChanged",
        EngineEvent::TransportChanged { .. } => "TransportChanged",
        EngineEvent::PlaybackStarted => "PlaybackStarted",
        EngineEvent::PlaybackStateChange { .. } => "PlaybackStateChange",
        _ => "Other",
    }).collect();

    // All key event types should be present
    assert!(event_types.contains(&"TransportChanged"), "expected TransportChanged");
    assert!(event_types.contains(&"PlaybackStarted"), "expected PlaybackStarted");
    assert!(event_types.contains(&"PlaybackStateChange"), "expected PlaybackStateChange");
    assert!(event_types.contains(&"TrackStarted"), "expected TrackStarted");
    assert!(event_types.contains(&"TrackStateChanged"), "expected TrackStateChanged");
    assert!(event_types.contains(&"TrackPlayingStateChanged"), "expected TrackPlayingStateChanged");
}

#[test]
fn stop_event_ordering() {
    let mut engine = engine_with_loaded_track("drums", 120.0);
    engine.play(SafetyPreset::Permissive).unwrap();
    engine.drain_events();

    engine.stop().unwrap();
    let events = engine.drain_events();
    let event_types: Vec<&str> = events.iter().map(|e| match e {
        EngineEvent::TrackStateChanged { .. } => "TrackStateChanged",
        EngineEvent::TrackStopped { .. } => "TrackStopped",
        EngineEvent::TrackPlayingStateChanged { .. } => "TrackPlayingStateChanged",
        EngineEvent::TransportChanged { .. } => "TransportChanged",
        EngineEvent::PlaybackStopped => "PlaybackStopped",
        EngineEvent::PlaybackStateChange { .. } => "PlaybackStateChange",
        _ => "Other",
    }).collect();

    let track_stopped_idx = event_types.iter().position(|&e| e == "TrackStopped").unwrap();
    let playback_stopped_idx = event_types.iter().position(|&e| e == "PlaybackStopped").unwrap();
    assert!(track_stopped_idx < playback_stopped_idx,
        "TrackStopped should fire before PlaybackStopped");
}

// ---------------------------------------------------------------------------
// State gates: operations rejected in wrong states
// ---------------------------------------------------------------------------

#[test]
fn operations_rejected_before_init() {
    let mut engine = MaestroEngine::new(48000);

    assert!(engine.create_track("x", "url", None, None).is_err());
    assert!(engine.play(SafetyPreset::Permissive).is_err());
    assert!(engine.pause().is_err());
    assert!(engine.stop().is_err());
}

#[test]
fn play_rejected_without_tracks() {
    let mut engine = ready_engine();
    // ServicesReady state, no tracks → play requires EnginesReady minimum
    let result = engine.play(SafetyPreset::Permissive);
    assert!(matches!(result, Err(EngineError::InvalidState { .. })));
}

#[test]
fn double_init_rejected() {
    let mut engine = MaestroEngine::new(48000);
    engine.initialize().unwrap();
    assert!(engine.initialize().is_err());
}

// ---------------------------------------------------------------------------
// Suspend/resume preserves state
// ---------------------------------------------------------------------------

#[test]
fn suspend_resume_preserves_tracks_and_transport() {
    let mut engine = engine_with_loaded_track("drums", 120.0);

    // Play, then suspend
    engine.play(SafetyPreset::Permissive).unwrap();
    assert!(engine.is_playing());
    engine.drain_events();

    engine.suspend().unwrap();
    assert_eq!(engine.state(), EngineState::Suspended);
    // Track count preserved
    assert_eq!(engine.track_count(), 1);

    // Operations still work while suspended
    engine.set_track_volume("drums", 0.3).unwrap();
    assert!((engine.track("drums").unwrap().controls().volume - 0.3).abs() < 0.01);

    // Resume
    engine.resume().unwrap();
    assert_eq!(engine.state(), EngineState::Ready);
    assert_eq!(engine.track_count(), 1);
}

// ---------------------------------------------------------------------------
// Track variation loading
// ---------------------------------------------------------------------------

#[test]
fn load_multiple_variations_for_track() {
    let mut engine = ready_engine();
    engine
        .create_track("drums", "url/drums", None, Some(120.0))
        .unwrap();

    // Load Default variation
    engine
        .load_track_audio(
            "drums",
            VariationType::Default,
            make_pcm(2, 48000 * 5, 48000),
            "url/drums-default",
        )
        .unwrap();

    // Load Simple variation (additional)
    engine
        .load_track_audio(
            "drums",
            VariationType::Simple,
            make_pcm(2, 48000 * 5, 48000),
            "url/drums-simple",
        )
        .unwrap();

    // Load Complex variation (additional)
    engine
        .load_track_audio(
            "drums",
            VariationType::Complex,
            make_pcm(2, 48000 * 5, 48000),
            "url/drums-complex",
        )
        .unwrap();

    // All three should be cached
    let cache = engine.pcm_cache();
    assert!(cache.has("drums", VariationType::Default));
    assert!(cache.has("drums", VariationType::Simple));
    assert!(cache.has("drums", VariationType::Complex));

    // Switch variation
    engine.arm_track("drums").unwrap();
    engine.set_track_variation("drums", 2, "Complex").unwrap();
    assert_eq!(engine.track("drums").unwrap().controls().current_variation, 2);
}

// ---------------------------------------------------------------------------
// AllTracksLoaded event with staggered loading
// ---------------------------------------------------------------------------

#[test]
fn all_tracks_loaded_event_fires_only_when_all_ready() {
    let mut engine = ready_engine();

    engine
        .create_track("drums", "url1", None, Some(120.0))
        .unwrap();
    engine
        .create_track("bass", "url2", None, Some(120.0))
        .unwrap();
    engine
        .create_track("guitar", "url3", None, Some(120.0))
        .unwrap();
    engine.drain_events();

    // Load first — not all loaded yet
    engine
        .load_track_audio("drums", VariationType::Default, make_pcm(2, 48000, 48000), "url1")
        .unwrap();
    let events = engine.drain_events();
    assert!(!events.iter().any(|e| matches!(e, EngineEvent::AllTracksLoaded)));

    // Load second — still not all
    engine
        .load_track_audio("bass", VariationType::Default, make_pcm(2, 48000, 48000), "url2")
        .unwrap();
    let events = engine.drain_events();
    assert!(!events.iter().any(|e| matches!(e, EngineEvent::AllTracksLoaded)));

    // Load third — now all loaded
    engine
        .load_track_audio("guitar", VariationType::Default, make_pcm(2, 48000, 48000), "url3")
        .unwrap();
    let events = engine.drain_events();
    assert!(events.iter().any(|e| matches!(e, EngineEvent::AllTracksLoaded)));
}

// ---------------------------------------------------------------------------
// PCM cache cleanup on track removal
// ---------------------------------------------------------------------------

#[test]
fn removing_track_clears_pcm_cache() {
    let mut engine = engine_with_loaded_track("drums", 120.0);

    assert!(engine.pcm_cache().has("drums", VariationType::Default));

    engine.remove_track("drums").unwrap();

    assert!(!engine.pcm_cache().has("drums", VariationType::Default));
}
