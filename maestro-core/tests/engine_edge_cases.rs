//! Edge case tests for MaestroEngine.
//!
//! Tests error paths, boundary conditions, and unusual operation sequences
//! that the inline unit tests don't cover.

use maestro_core::engine::{EngineError, MaestroEngine};
use maestro_core::events::EngineState;
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

fn engine_with_loaded_track(id: &str) -> MaestroEngine {
    let mut e = ready_engine();
    e.create_track(id, &format!("url/{id}"), None, Some(120.0))
        .unwrap();
    e.load_track_audio(id, VariationType::Default, make_pcm(2, 48000 * 5, 48000), &format!("url/{id}"))
        .unwrap();
    e.arm_track(id).unwrap();
    e.drain_events();
    e
}

// ---------------------------------------------------------------------------
// Error paths: operations on nonexistent tracks
// ---------------------------------------------------------------------------

#[test]
fn arm_nonexistent_track() {
    let mut engine = ready_engine();
    assert!(matches!(
        engine.arm_track("ghost"),
        Err(EngineError::TrackNotFound(_))
    ));
}

#[test]
fn disarm_nonexistent_track() {
    let mut engine = ready_engine();
    assert!(matches!(
        engine.disarm_track("ghost"),
        Err(EngineError::TrackNotFound(_))
    ));
}

#[test]
fn toggle_nonexistent_track() {
    let mut engine = ready_engine();
    assert!(matches!(
        engine.toggle_track("ghost"),
        Err(EngineError::TrackNotFound(_))
    ));
}

#[test]
fn set_variation_nonexistent_track() {
    let mut engine = ready_engine();
    assert!(matches!(
        engine.set_track_variation("ghost", 1, "Simple"),
        Err(EngineError::TrackNotFound(_))
    ));
}

#[test]
fn load_audio_nonexistent_track() {
    let mut engine = ready_engine();
    let result = engine.load_track_audio(
        "ghost",
        VariationType::Default,
        make_pcm(2, 48000, 48000),
        "url",
    );
    assert!(matches!(result, Err(EngineError::TrackNotFound(_))));
}

// ---------------------------------------------------------------------------
// Rapid play/stop cycles
// ---------------------------------------------------------------------------

#[test]
fn rapid_play_stop_cycles() {
    let mut engine = engine_with_loaded_track("drums");

    for _ in 0..20 {
        engine.play(SafetyPreset::Permissive).unwrap();
        assert!(engine.is_playing());
        engine.stop().unwrap();
        assert!(engine.is_stopped());
        engine.drain_events();
    }

    // Engine should still be in a clean state
    assert_eq!(engine.active_track_count(), 0);
    assert_eq!(engine.track("drums").unwrap().status(), TrackStatus::Ready);
}

#[test]
fn rapid_play_pause_resume_cycles() {
    let mut engine = engine_with_loaded_track("drums");

    for _ in 0..10 {
        engine.play(SafetyPreset::Permissive).unwrap();
        engine.pause().unwrap();
        engine.play(SafetyPreset::Permissive).unwrap();
        engine.stop().unwrap();
        engine.drain_events();
    }

    assert!(engine.is_stopped());
    assert_eq!(engine.track("drums").unwrap().status(), TrackStatus::Ready);
}

// ---------------------------------------------------------------------------
// Stop when already stopped
// ---------------------------------------------------------------------------

#[test]
fn stop_when_already_stopped_is_safe() {
    let mut engine = engine_with_loaded_track("drums");

    // Stop without playing first
    engine.stop().unwrap();
    assert!(engine.is_stopped());

    // Double stop
    engine.play(SafetyPreset::Permissive).unwrap();
    engine.stop().unwrap();
    engine.stop().unwrap(); // Should not error
    assert!(engine.is_stopped());
}

#[test]
fn pause_when_not_playing_is_safe() {
    let mut engine = engine_with_loaded_track("drums");

    // Pause without playing
    engine.pause().unwrap();
    // Transport still stopped (pause is a no-op when not playing)
}

// ---------------------------------------------------------------------------
// Destroy from various states
// ---------------------------------------------------------------------------

#[test]
fn destroy_from_uninitialized() {
    let mut engine = MaestroEngine::new(48000);
    engine.destroy(); // Should not panic
    assert_eq!(engine.state(), EngineState::Destroying);
}

#[test]
fn destroy_from_playing() {
    let mut engine = engine_with_loaded_track("drums");
    engine.play(SafetyPreset::Permissive).unwrap();
    engine.destroy();
    assert_eq!(engine.state(), EngineState::Destroying);
    assert_eq!(engine.track_count(), 0);
}

#[test]
fn destroy_from_suspended() {
    let mut engine = engine_with_loaded_track("drums");
    engine.suspend().unwrap();
    engine.destroy();
    assert_eq!(engine.state(), EngineState::Destroying);
}

// ---------------------------------------------------------------------------
// Many tracks
// ---------------------------------------------------------------------------

#[test]
fn engine_handles_many_tracks() {
    let mut engine = ready_engine();

    // Create 20 tracks
    for i in 0..20 {
        let id = format!("track_{i}");
        engine
            .create_track(&id, &format!("url/{id}"), None, Some(120.0))
            .unwrap();
        engine
            .load_track_audio(&id, VariationType::Default, make_pcm(2, 4800, 48000), &format!("url/{id}"))
            .unwrap();
        engine.arm_track(&id).unwrap();
    }

    assert_eq!(engine.track_count(), 20);
    assert_eq!(engine.state(), EngineState::Ready);

    // Play all 20
    engine.play(SafetyPreset::Permissive).unwrap();
    assert_eq!(engine.active_track_count(), 20);

    // Stop all
    engine.stop().unwrap();
    assert_eq!(engine.active_track_count(), 0);
    engine.drain_events();

    // Remove them all
    for i in 0..20 {
        engine.remove_track(&format!("track_{i}")).unwrap();
    }
    assert_eq!(engine.track_count(), 0);
}

// ---------------------------------------------------------------------------
// Event count doesn't grow unboundedly
// ---------------------------------------------------------------------------

#[test]
fn events_dont_accumulate_when_drained() {
    let mut engine = engine_with_loaded_track("drums");

    for _ in 0..100 {
        engine.play(SafetyPreset::Permissive).unwrap();
        engine.stop().unwrap();
        let events = engine.drain_events();
        // Each cycle should produce a bounded number of events
        assert!(events.len() < 50, "too many events per cycle: {}", events.len());
    }

    // Final drain should be empty
    assert_eq!(engine.pending_event_count(), 0);
}

// ---------------------------------------------------------------------------
// Mono audio (1 channel)
// ---------------------------------------------------------------------------

#[test]
fn mono_audio_loads_correctly() {
    let mut engine = ready_engine();
    engine
        .create_track("mono", "url/mono", None, Some(120.0))
        .unwrap();

    // Load 1-channel audio
    engine
        .load_track_audio("mono", VariationType::Default, make_pcm(1, 48000, 48000), "url/mono")
        .unwrap();

    assert_eq!(engine.track("mono").unwrap().status(), TrackStatus::Ready);
}

// ---------------------------------------------------------------------------
// Multi-channel audio (6 channels)
// ---------------------------------------------------------------------------

#[test]
fn multichannel_audio_loads_correctly() {
    let mut engine = ready_engine();
    engine
        .create_track("surround", "url/surround", None, Some(120.0))
        .unwrap();

    // Load 6-channel audio (5.1 surround)
    engine
        .load_track_audio("surround", VariationType::Default, make_pcm(6, 48000, 48000), "url/surround")
        .unwrap();

    assert_eq!(engine.track("surround").unwrap().status(), TrackStatus::Ready);
}

// ---------------------------------------------------------------------------
// Very short audio
// ---------------------------------------------------------------------------

#[test]
fn very_short_audio_loads() {
    let mut engine = ready_engine();
    engine
        .create_track("click", "url/click", None, Some(120.0))
        .unwrap();

    // 1 frame of audio
    engine
        .load_track_audio("click", VariationType::Default, make_pcm(2, 1, 48000), "url/click")
        .unwrap();

    assert_eq!(engine.track("click").unwrap().status(), TrackStatus::Ready);
    let duration = engine.track("click").unwrap().duration();
    assert!(duration > 0.0 && duration < 0.001);
}

// ---------------------------------------------------------------------------
// Sample rate variations
// ---------------------------------------------------------------------------

#[test]
fn different_sample_rates() {
    for sr in [22050, 44100, 48000, 96000] {
        let mut engine = MaestroEngine::new(sr);
        engine.initialize().unwrap();
        assert_eq!(engine.sample_rate(), sr);
    }
}

// ---------------------------------------------------------------------------
// Transport safety presets
// ---------------------------------------------------------------------------

#[test]
fn manual_control_does_not_auto_start_transport() {
    let mut engine = engine_with_loaded_track("drums");

    // ManualControl has allow_auto_start: false
    engine.play(SafetyPreset::ManualControl).unwrap();

    // Track moves to Playing, but transport shouldn't auto-start
    assert_eq!(engine.track("drums").unwrap().status(), TrackStatus::Playing);
    assert_eq!(engine.transport_state(), TransportState::Stopped);
}

#[test]
fn slider_operation_preserves_transport_on_stop() {
    let mut engine = engine_with_loaded_track("drums");
    engine.play(SafetyPreset::Permissive).unwrap();
    assert!(engine.is_playing());
    engine.drain_events();

    // SliderOperation has allow_auto_stop: false
    // Use transport directly to test the preset behavior
    engine.transport_mut().set_operation_context(
        maestro_core::transport::OperationContext {
            source: maestro_core::transport::OperationSource::LiveMixing,
            safety_preset: SafetyPreset::SliderOperation,
            reason: Some("tempo drag".into()),
            track_id: None,
        },
    );
    let action = engine.transport_mut().on_track_stopped("drums");
    // With SliderOperation, auto-stop is denied
    assert_eq!(action, maestro_core::transport::TransportAction::None);
}

// ---------------------------------------------------------------------------
// Track info query on empty engine
// ---------------------------------------------------------------------------

#[test]
fn track_info_returns_none_for_missing() {
    let engine = ready_engine();
    assert!(engine.get_track_info("nonexistent").is_none());
    assert!(engine.track("nonexistent").is_none());
}

#[test]
fn get_all_tracks_empty() {
    let engine = ready_engine();
    assert!(engine.get_all_tracks().is_empty());
    assert!(engine.track_ids().is_empty());
}

// ---------------------------------------------------------------------------
// Beat info before tempo configured
// ---------------------------------------------------------------------------

#[test]
fn beat_info_safe_before_tempo_set() {
    let engine = ready_engine();
    let info = engine.get_beat_info();
    // bpm should be 0 (safe default) when no tempo configured
    assert!((info.bpm - 0.0).abs() < 0.01);
    assert_eq!(info.time_signature_numerator, 4);
    assert_eq!(info.time_signature_denominator, 4);
}

// ---------------------------------------------------------------------------
// Engine error Display coverage
// ---------------------------------------------------------------------------

#[test]
fn all_engine_error_variants_display() {
    let errors = vec![
        EngineError::InvalidState {
            current: EngineState::Uninitialized,
            operation: "test",
        },
        EngineError::InvalidStateTransition {
            from: EngineState::Uninitialized,
            to: EngineState::Ready,
        },
        EngineError::TrackNotFound("x".into()),
        EngineError::TrackAlreadyExists("x".into()),
        EngineError::Tempo("bad".into()),
        EngineError::Pitch("bad".into()),
        EngineError::Mixer("bad".into()),
        EngineError::Track("bad".into()),
    ];

    for err in &errors {
        let msg = err.to_string();
        assert!(!msg.is_empty(), "error Display should not be empty");
    }
}
