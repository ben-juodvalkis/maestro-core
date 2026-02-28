//! Integration tests: Engine + beat tracking under simulated playback.
//!
//! These tests verify that beat/bar/loop events are emitted correctly
//! when the engine processes beat position updates during playback.

use maestro_core::beat_tracker::BeatTrackingConfig;
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

fn playing_engine() -> MaestroEngine {
    let mut e = MaestroEngine::new(48000);
    e.initialize().unwrap();
    e.create_track("drums", "url/drums", None, Some(120.0))
        .unwrap();
    e.load_track_audio("drums", VariationType::Default, make_pcm(2, 48000 * 10, 48000), "url/drums")
        .unwrap();
    e.arm_track("drums").unwrap();
    e.play(SafetyPreset::Permissive).unwrap();
    e.drain_events();
    e
}

// ---------------------------------------------------------------------------
// Beat events during simulated playback
// ---------------------------------------------------------------------------

#[test]
fn beat_updates_emit_beat_events() {
    let mut engine = playing_engine();

    // Simulate progressing through beats (4/4 time, 120 BPM)
    // At 120 BPM, each beat = 0.5 seconds. Beat positions are fractional.
    // BeatTracker starts with previous_beat=1, so we need to cross beat boundaries.
    for beat in [1.0, 2.0, 3.0, 4.0, 5.0] {
        engine.update_beat_position(beat);
    }

    let events = engine.drain_events();
    let beat_count = events
        .iter()
        .filter(|e| matches!(e, EngineEvent::BeatPrecise { .. }))
        .count();

    // Should have emitted beat events for each boundary crossing
    assert!(beat_count >= 3, "expected at least 3 beat events, got {beat_count}");
}

#[test]
fn bar_change_events_at_bar_boundaries() {
    let mut engine = playing_engine();

    // Default config: 4 beats per bar. Bars change at beats 4, 8, 12, etc.
    // Simulate one full bar of beats
    for beat in [1.0, 2.0, 3.0, 4.0, 5.0] {
        engine.update_beat_position(beat);
    }

    let events = engine.drain_events();
    let bar_changes = events
        .iter()
        .filter(|e| matches!(e, EngineEvent::BarChanged { .. }))
        .count();

    // Should have at least one bar change (crossing from bar 1 to bar 2)
    assert!(bar_changes >= 1, "expected at least 1 bar change, got {bar_changes}");
}

#[test]
fn loop_completed_event_at_loop_boundary() {
    let mut engine = playing_engine();

    // Configure a short loop: 4 beats per bar, 2 bars per loop = 8 beats total
    engine.set_beat_tracking_config(BeatTrackingConfig {
        beats_per_bar: 4,
        bars_per_loop: 2,
    });
    engine.drain_events();

    // Walk through all 8 beats of the loop and into the next
    for beat in 1..=9 {
        engine.update_beat_position(beat as f64);
    }

    let events = engine.drain_events();
    let loop_events = events
        .iter()
        .filter(|e| matches!(e, EngineEvent::LoopCompleted { .. }))
        .count();

    assert!(loop_events >= 1, "expected at least 1 loop completion event, got {loop_events}");
}

#[test]
fn position_updated_events_emitted() {
    let mut engine = playing_engine();

    engine.update_beat_position(2.0);
    engine.update_beat_position(3.0);
    engine.update_beat_position(4.0);

    let events = engine.drain_events();
    let position_updates = events
        .iter()
        .filter(|e| matches!(e, EngineEvent::PositionUpdated { .. }))
        .count();

    assert!(position_updates >= 1, "expected position update events, got {position_updates}");
}

#[test]
fn position_updated_includes_loop_progress() {
    let mut engine = playing_engine();

    // Configure: 4 beats/bar × 4 bars = 16 beats per loop
    engine.set_beat_tracking_config(BeatTrackingConfig {
        beats_per_bar: 4,
        bars_per_loop: 4,
    });
    engine.drain_events();

    // Jump to beat 8 (halfway through a 16-beat loop)
    engine.update_beat_position(8.0);

    let events = engine.drain_events();
    let position_event = events.iter().find_map(|e| {
        if let EngineEvent::PositionUpdated { loop_progress, .. } = e {
            Some(*loop_progress)
        } else {
            None
        }
    });

    assert!(position_event.is_some(), "expected a PositionUpdated event");
    let progress = position_event.unwrap();
    assert!(
        (progress - 0.5).abs() < 0.1,
        "expected ~0.5 loop progress at beat 8 of 16, got {progress}"
    );
}

// ---------------------------------------------------------------------------
// Beat tracking config changes
// ---------------------------------------------------------------------------

#[test]
fn beat_tracking_config_change_reflects_in_beat_info() {
    let mut engine = playing_engine();

    engine.set_beat_tracking_config(BeatTrackingConfig {
        beats_per_bar: 3,
        bars_per_loop: 8,
    });

    let config = engine.beat_tracker().config();
    assert_eq!(config.beats_per_bar, 3);
    assert_eq!(config.bars_per_loop, 8);
}

#[test]
fn beat_info_snapshot_reflects_tempo() {
    let mut engine = playing_engine();

    engine.set_tempo(140.0).unwrap();

    let info = engine.get_beat_info();
    assert!((info.bpm - 140.0).abs() < 0.01);
    assert_eq!(info.time_signature_numerator, 4);
    assert_eq!(info.time_signature_denominator, 4);
}

// ---------------------------------------------------------------------------
// No beat events when not playing
// ---------------------------------------------------------------------------

#[test]
fn no_beat_events_after_stop() {
    let mut engine = playing_engine();
    engine.stop().unwrap();
    engine.drain_events();

    // Update beat position after stop — beat tracker was stopped
    engine.update_beat_position(2.0);
    engine.update_beat_position(3.0);

    let events = engine.drain_events();
    let beat_events = events
        .iter()
        .filter(|e| matches!(e, EngineEvent::BeatPrecise { .. }))
        .count();

    // Beat tracker is stopped, so no beat events should fire
    assert_eq!(beat_events, 0, "no beat events expected after stop");
}
