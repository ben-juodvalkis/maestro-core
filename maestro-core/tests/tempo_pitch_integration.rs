//! Integration tests: Engine + tempo/pitch control during playback.
//!
//! Tests that tempo and pitch changes coordinate correctly between the
//! engine, tempo controller, timeline, and track controls.

use maestro_core::engine::MaestroEngine;
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

fn playing_engine_with_tempo(tempo: f64) -> MaestroEngine {
    let mut e = MaestroEngine::new(48000);
    e.initialize().unwrap();
    e.create_track("drums", "url/drums", None, Some(tempo))
        .unwrap();
    e.load_track_audio("drums", VariationType::Default, make_pcm(2, 48000 * 10, 48000), "url/drums")
        .unwrap();
    e.arm_track("drums").unwrap();
    e.set_tempo(tempo).unwrap();
    e.play(SafetyPreset::Permissive).unwrap();
    e.drain_events();
    e
}

// ---------------------------------------------------------------------------
// Tempo control
// ---------------------------------------------------------------------------

#[test]
fn set_tempo_updates_bpm() {
    let mut engine = playing_engine_with_tempo(120.0);

    engine.set_tempo(140.0).unwrap();
    assert!((engine.current_bpm() - 140.0).abs() < 0.01);
}

#[test]
fn set_tempo_emits_tempo_changed_event() {
    let mut engine = playing_engine_with_tempo(120.0);
    engine.set_tempo(95.0).unwrap();

    let events = engine.drain_events();
    let tempo_event = events.iter().find_map(|e| {
        if let EngineEvent::TempoChanged { tempo, previous_tempo } = e {
            Some((*tempo, *previous_tempo))
        } else {
            None
        }
    });

    assert!(tempo_event.is_some());
    let (new_tempo, prev_tempo) = tempo_event.unwrap();
    assert!((new_tempo - 95.0).abs() < 0.01);
    assert!((prev_tempo - 120.0).abs() < 0.01);
}

#[test]
fn small_tempo_change_no_event() {
    let mut engine = playing_engine_with_tempo(120.0);

    // Change by less than 0.01 — should not emit event
    engine.set_tempo(120.005).unwrap();

    let events = engine.drain_events();
    let has_tempo_event = events
        .iter()
        .any(|e| matches!(e, EngineEvent::TempoChanged { .. }));
    assert!(!has_tempo_event, "tiny tempo change should not emit event");
}

#[test]
fn tempo_ratio_control() {
    let mut engine = playing_engine_with_tempo(120.0);

    // set_tempo_ratio requires calculated_tempo to be set (the original/metadata tempo)
    engine.tempo_controller_mut().set_calculated_tempo(120.0).unwrap();

    // Default ratio is 1.0
    assert!((engine.tempo_ratio() - 1.0).abs() < 0.01);

    // Set ratio to 0.5 (half speed)
    engine.set_tempo_ratio(0.5).unwrap();

    // Ratio should reflect the change
    assert!((engine.tempo_ratio() - 0.5).abs() < 0.01);
}

#[test]
fn tempo_changes_during_playback() {
    let mut engine = playing_engine_with_tempo(120.0);

    // Simulate a DJ gradually increasing tempo
    for bpm in [125.0, 130.0, 135.0, 140.0] {
        engine.set_tempo(bpm).unwrap();
        assert!((engine.current_bpm() - bpm).abs() < 0.01);
    }

    let events = engine.drain_events();
    let tempo_events: Vec<_> = events
        .iter()
        .filter(|e| matches!(e, EngineEvent::TempoChanged { .. }))
        .collect();

    // Should have 4 tempo change events
    assert_eq!(tempo_events.len(), 4);
}

#[test]
fn beat_info_reflects_tempo_after_change() {
    let mut engine = playing_engine_with_tempo(100.0);
    engine.set_tempo(180.0).unwrap();

    let info = engine.get_beat_info();
    assert!((info.bpm - 180.0).abs() < 0.01);
}

// ---------------------------------------------------------------------------
// Pitch control
// ---------------------------------------------------------------------------

#[test]
fn set_pitch_cents_updates_track() {
    let mut engine = playing_engine_with_tempo(120.0);

    engine.set_track_pitch_cents("drums", 300.0).unwrap();

    let track = engine.track("drums").unwrap();
    assert!((track.controls().pitch_cents - 300.0).abs() < 0.01);
}

#[test]
fn set_pitch_emits_event_with_semitones() {
    let mut engine = playing_engine_with_tempo(120.0);
    engine.set_track_pitch_cents("drums", 200.0).unwrap();

    let events = engine.drain_events();
    let pitch_event = events.iter().find_map(|e| {
        if let EngineEvent::TrackPitchChanged {
            track_id,
            offset_cents,
            offset_semitones,
        } = e
        {
            if track_id == "drums" {
                return Some((*offset_cents, *offset_semitones));
            }
        }
        None
    });

    assert!(pitch_event.is_some());
    let (cents, semitones) = pitch_event.unwrap();
    assert!((cents - 200.0).abs() < 0.01);
    assert!((semitones - 2.0).abs() < 0.01); // 200 cents = 2 semitones
}

#[test]
fn pitch_on_nonexistent_track_fails() {
    let mut engine = playing_engine_with_tempo(120.0);
    let result = engine.set_track_pitch_cents("nope", 100.0);
    assert!(result.is_err());
}

#[test]
fn pitch_changes_reflected_in_track_info() {
    let mut engine = playing_engine_with_tempo(120.0);
    engine.set_track_pitch_cents("drums", -400.0).unwrap();

    let info = engine.get_track_info("drums").unwrap();
    assert!((info.pitch_cents - (-400.0)).abs() < 0.01);
}

// ---------------------------------------------------------------------------
// Variation control through engine
// ---------------------------------------------------------------------------

#[test]
fn variation_change_emits_event() {
    let mut engine = playing_engine_with_tempo(120.0);
    engine
        .set_track_variation("drums", 1, "Simple")
        .unwrap();

    let events = engine.drain_events();
    let var_event = events.iter().find_map(|e| {
        if let EngineEvent::VariationChanged {
            track_id,
            previous_variation_index,
            new_variation_index,
            variation_name,
        } = e
        {
            if track_id == "drums" {
                return Some((*previous_variation_index, *new_variation_index, variation_name.clone()));
            }
        }
        None
    });

    assert!(var_event.is_some());
    let (prev, new, name) = var_event.unwrap();
    assert_eq!(prev, 0);
    assert_eq!(new, 1);
    assert_eq!(name, "Simple");
}

#[test]
fn variation_change_reflected_in_track_info() {
    let mut engine = playing_engine_with_tempo(120.0);
    engine
        .set_track_variation("drums", 2, "Complex")
        .unwrap();

    let info = engine.get_track_info("drums").unwrap();
    assert_eq!(info.current_variation, 2);
}

// ---------------------------------------------------------------------------
// Combined tempo + pitch during playback
// ---------------------------------------------------------------------------

#[test]
fn simultaneous_tempo_and_pitch_changes() {
    let mut engine = playing_engine_with_tempo(120.0);

    // Change tempo and pitch together (realistic DJ scenario)
    engine.set_tempo(135.0).unwrap();
    engine.set_track_pitch_cents("drums", 100.0).unwrap();

    assert!((engine.current_bpm() - 135.0).abs() < 0.01);
    assert!((engine.track("drums").unwrap().controls().pitch_cents - 100.0).abs() < 0.01);

    let events = engine.drain_events();
    assert!(events
        .iter()
        .any(|e| matches!(e, EngineEvent::TempoChanged { .. })));
    assert!(events
        .iter()
        .any(|e| matches!(e, EngineEvent::TrackPitchChanged { .. })));
}
