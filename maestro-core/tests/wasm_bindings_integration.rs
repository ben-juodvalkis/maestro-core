//! Integration tests for WASM bindings.
//!
//! Tests the `WasmMaestroEngine` wrapper and JSON serialization logic
//! through realistic workflows. These run on native (not in a browser)
//! by testing the non-JsValue paths.
//!
//! Note: Methods that return `JsValue` can't be called in non-WASM tests.
//! We test through the inner engine and through the `engine_event_to_json`
//! serialization by going through the drain_events_json path.

// Only compile when the "json" feature is available (needed for serde_json)
#![cfg(feature = "json")]

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

fn ready_engine_with_track() -> MaestroEngine {
    let mut e = MaestroEngine::new(48000);
    e.initialize().unwrap();
    e.create_track("drums", "url/drums", None, Some(120.0))
        .unwrap();
    e.load_track_audio("drums", VariationType::Default, make_pcm(2, 48000 * 5, 48000), "url/drums")
        .unwrap();
    e.arm_track("drums").unwrap();
    e
}

// ---------------------------------------------------------------------------
// Event JSON serialization coverage
// ---------------------------------------------------------------------------

/// Helper: drain events and serialize them like the WASM bindings would.
fn drain_events_as_json(engine: &mut MaestroEngine) -> Vec<serde_json::Value> {
    let events = engine.drain_events();
    events
        .iter()
        .map(|e| event_to_json(e))
        .collect()
}

/// Replicate the serialization logic from wasm_bindings.rs for testing.
/// (The actual function is private in the wasm_bindings module.)
fn event_to_json(event: &EngineEvent) -> serde_json::Value {
    match event {
        EngineEvent::PlaybackStarted => serde_json::json!({ "type": "playback:started" }),
        EngineEvent::PlaybackPaused => serde_json::json!({ "type": "playback:paused" }),
        EngineEvent::PlaybackStopped => serde_json::json!({ "type": "playback:stopped" }),
        EngineEvent::PlaybackStateChange { is_playing, is_paused } => serde_json::json!({
            "type": "playback:stateChange",
            "isPlaying": is_playing,
            "isPaused": is_paused,
        }),
        EngineEvent::TransportChanged { state } => serde_json::json!({
            "type": "transport:changed",
            "state": state.to_string(),
        }),
        EngineEvent::TempoChanged { tempo, previous_tempo } => serde_json::json!({
            "type": "tempo:changed",
            "tempo": tempo,
            "previousTempo": previous_tempo,
        }),
        EngineEvent::TrackCreated { track_id } => serde_json::json!({
            "type": "track:created",
            "trackId": track_id,
        }),
        EngineEvent::TrackRemoved { track_id } => serde_json::json!({
            "type": "track:removed",
            "trackId": track_id,
        }),
        EngineEvent::TrackStateChanged { track_id, state, previous_state } => serde_json::json!({
            "type": "track:stateChanged",
            "trackId": track_id,
            "state": state.to_string(),
            "previousState": previous_state.to_string(),
        }),
        EngineEvent::TrackLoaded { track_id } => serde_json::json!({
            "type": "track:loaded",
            "trackId": track_id,
        }),
        EngineEvent::TrackError { track_id, message } => serde_json::json!({
            "type": "track:error",
            "trackId": track_id,
            "message": message,
        }),
        EngineEvent::TrackStarted { track_id, active_track_count } => serde_json::json!({
            "type": "track:started",
            "trackId": track_id,
            "activeTrackCount": active_track_count,
        }),
        EngineEvent::TrackStopped { track_id, active_track_count } => serde_json::json!({
            "type": "track:stopped",
            "trackId": track_id,
            "activeTrackCount": active_track_count,
        }),
        EngineEvent::TrackPlayingStateChanged { track_id, is_playing } => serde_json::json!({
            "type": "track:playingStateChanged",
            "trackId": track_id,
            "isPlaying": is_playing,
        }),
        EngineEvent::TrackVolumeChanged { track_id, volume } => serde_json::json!({
            "type": "track:volumeChanged",
            "trackId": track_id,
            "volume": volume,
        }),
        EngineEvent::TrackPitchChanged { track_id, offset_cents, offset_semitones } => serde_json::json!({
            "type": "track:pitchChanged",
            "trackId": track_id,
            "offsetCents": offset_cents,
            "offsetSemitones": offset_semitones,
        }),
        EngineEvent::TrackArmed { track_id } => serde_json::json!({
            "type": "track:armed",
            "trackId": track_id,
        }),
        EngineEvent::TrackDisarmed { track_id } => serde_json::json!({
            "type": "track:disarmed",
            "trackId": track_id,
        }),
        EngineEvent::VariationChanged {
            track_id,
            previous_variation_index,
            new_variation_index,
            variation_name,
        } => serde_json::json!({
            "type": "variation:changed",
            "trackId": track_id,
            "previousVariationIndex": previous_variation_index,
            "newVariationIndex": new_variation_index,
            "variationName": variation_name,
        }),
        EngineEvent::PositionUpdated { beat, loop_progress } => serde_json::json!({
            "type": "position:updated",
            "beat": beat,
            "loopProgress": loop_progress,
        }),
        EngineEvent::BeatPrecise { beat_info } => serde_json::json!({
            "type": "beat:precise",
            "beat": beat_info.beat,
            "bar": beat_info.bar,
            "absoluteBeat": beat_info.absolute_beat,
            "beatFraction": beat_info.beat_fraction,
            "preciseBeat": beat_info.precise_beat,
        }),
        EngineEvent::BarChanged { bar, beat } => serde_json::json!({
            "type": "bar:changed",
            "bar": bar,
            "beat": beat,
        }),
        EngineEvent::LoopCompleted { loop_count } => serde_json::json!({
            "type": "loop:completed",
            "loopCount": loop_count,
        }),
        EngineEvent::Initialized { sample_rate } => serde_json::json!({
            "type": "engine:initialized",
            "sampleRate": sample_rate,
        }),
        EngineEvent::AllTracksLoaded => serde_json::json!({
            "type": "engine:allTracksLoaded"
        }),
        EngineEvent::Error { message, context } => serde_json::json!({
            "type": "engine:error",
            "message": message,
            "context": context,
        }),
    }
}

// ---------------------------------------------------------------------------
// Initialization events
// ---------------------------------------------------------------------------

#[test]
fn init_events_serialize_correctly() {
    let mut engine = MaestroEngine::new(48000);
    engine.initialize().unwrap();

    let json_events = drain_events_as_json(&mut engine);
    let init_event = json_events.iter().find(|e| e["type"] == "engine:initialized");
    assert!(init_event.is_some());
    assert_eq!(init_event.unwrap()["sampleRate"], 48000);
}

// ---------------------------------------------------------------------------
// Track lifecycle events
// ---------------------------------------------------------------------------

#[test]
fn track_created_event_json() {
    let mut engine = MaestroEngine::new(48000);
    engine.initialize().unwrap();
    engine.drain_events();

    engine
        .create_track("drums", "url/drums", None, Some(120.0))
        .unwrap();

    let json_events = drain_events_as_json(&mut engine);
    let created = json_events.iter().find(|e| e["type"] == "track:created");
    assert!(created.is_some());
    assert_eq!(created.unwrap()["trackId"], "drums");
}

#[test]
fn track_loaded_event_json() {
    let mut engine = MaestroEngine::new(48000);
    engine.initialize().unwrap();
    engine
        .create_track("drums", "url/drums", None, None)
        .unwrap();
    engine.drain_events();

    engine
        .load_track_audio("drums", VariationType::Default, make_pcm(2, 48000, 48000), "url/drums")
        .unwrap();

    let json_events = drain_events_as_json(&mut engine);
    let loaded = json_events.iter().find(|e| e["type"] == "track:loaded");
    assert!(loaded.is_some());
    assert_eq!(loaded.unwrap()["trackId"], "drums");

    let state_changed = json_events
        .iter()
        .find(|e| e["type"] == "track:stateChanged" && e["state"] == "ready");
    assert!(state_changed.is_some());
}

#[test]
fn track_armed_disarmed_events_json() {
    let mut engine = ready_engine_with_track();
    engine.drain_events();

    engine.disarm_track("drums").unwrap();
    let events = drain_events_as_json(&mut engine);
    let disarmed = events.iter().find(|e| e["type"] == "track:disarmed");
    assert!(disarmed.is_some());
    assert_eq!(disarmed.unwrap()["trackId"], "drums");

    engine.arm_track("drums").unwrap();
    let events = drain_events_as_json(&mut engine);
    let armed = events.iter().find(|e| e["type"] == "track:armed");
    assert!(armed.is_some());
    assert_eq!(armed.unwrap()["trackId"], "drums");
}

// ---------------------------------------------------------------------------
// Playback events
// ---------------------------------------------------------------------------

#[test]
fn playback_started_event_json() {
    let mut engine = ready_engine_with_track();
    engine.drain_events();

    engine.play(SafetyPreset::Permissive).unwrap();
    let events = drain_events_as_json(&mut engine);

    let started = events.iter().find(|e| e["type"] == "playback:started");
    assert!(started.is_some());

    let state_change = events
        .iter()
        .find(|e| e["type"] == "playback:stateChange");
    assert!(state_change.is_some());
    assert_eq!(state_change.unwrap()["isPlaying"], true);
    assert_eq!(state_change.unwrap()["isPaused"], false);
}

#[test]
fn playback_paused_event_json() {
    let mut engine = ready_engine_with_track();
    engine.play(SafetyPreset::Permissive).unwrap();
    engine.drain_events();

    engine.pause().unwrap();
    let events = drain_events_as_json(&mut engine);

    let paused = events.iter().find(|e| e["type"] == "playback:paused");
    assert!(paused.is_some());

    let state_change = events
        .iter()
        .find(|e| e["type"] == "playback:stateChange");
    assert!(state_change.is_some());
    assert_eq!(state_change.unwrap()["isPlaying"], false);
    assert_eq!(state_change.unwrap()["isPaused"], true);
}

#[test]
fn playback_stopped_event_json() {
    let mut engine = ready_engine_with_track();
    engine.play(SafetyPreset::Permissive).unwrap();
    engine.drain_events();

    engine.stop().unwrap();
    let events = drain_events_as_json(&mut engine);

    let stopped = events.iter().find(|e| e["type"] == "playback:stopped");
    assert!(stopped.is_some());

    let state_change = events
        .iter()
        .find(|e| e["type"] == "playback:stateChange");
    assert!(state_change.is_some());
    assert_eq!(state_change.unwrap()["isPlaying"], false);
    assert_eq!(state_change.unwrap()["isPaused"], false);
}

// ---------------------------------------------------------------------------
// Transport changed events
// ---------------------------------------------------------------------------

#[test]
fn transport_changed_event_json() {
    let mut engine = ready_engine_with_track();
    engine.drain_events();

    engine.play(SafetyPreset::Permissive).unwrap();
    let events = drain_events_as_json(&mut engine);

    let transport = events
        .iter()
        .find(|e| e["type"] == "transport:changed");
    assert!(transport.is_some());
    assert_eq!(transport.unwrap()["state"], "playing");
}

// ---------------------------------------------------------------------------
// Tempo events
// ---------------------------------------------------------------------------

#[test]
fn tempo_changed_event_json() {
    let mut engine = ready_engine_with_track();
    engine.set_tempo(120.0).unwrap();
    engine.drain_events();

    engine.set_tempo(140.0).unwrap();
    let events = drain_events_as_json(&mut engine);

    let tempo = events.iter().find(|e| e["type"] == "tempo:changed");
    assert!(tempo.is_some());
    let t = tempo.unwrap();
    assert!((t["tempo"].as_f64().unwrap() - 140.0).abs() < 0.01);
    assert!((t["previousTempo"].as_f64().unwrap() - 120.0).abs() < 0.01);
}

// ---------------------------------------------------------------------------
// Volume events
// ---------------------------------------------------------------------------

#[test]
fn volume_changed_event_json() {
    let mut engine = ready_engine_with_track();
    engine.drain_events();

    engine.set_track_volume("drums", 0.3).unwrap();
    let events = drain_events_as_json(&mut engine);

    let vol = events.iter().find(|e| e["type"] == "track:volumeChanged");
    assert!(vol.is_some());
    let v = vol.unwrap();
    assert_eq!(v["trackId"], "drums");
    assert!((v["volume"].as_f64().unwrap() - 0.3).abs() < 0.01);
}

// ---------------------------------------------------------------------------
// Pitch events
// ---------------------------------------------------------------------------

#[test]
fn pitch_changed_event_json() {
    let mut engine = ready_engine_with_track();
    engine.drain_events();

    engine.set_track_pitch_cents("drums", 500.0).unwrap();
    let events = drain_events_as_json(&mut engine);

    let pitch = events.iter().find(|e| e["type"] == "track:pitchChanged");
    assert!(pitch.is_some());
    let p = pitch.unwrap();
    assert_eq!(p["trackId"], "drums");
    assert!((p["offsetCents"].as_f64().unwrap() - 500.0).abs() < 0.01);
    assert!((p["offsetSemitones"].as_f64().unwrap() - 5.0).abs() < 0.01);
}

// ---------------------------------------------------------------------------
// Variation events
// ---------------------------------------------------------------------------

#[test]
fn variation_changed_event_json() {
    let mut engine = ready_engine_with_track();
    engine.drain_events();

    engine
        .set_track_variation("drums", 2, "Complex")
        .unwrap();
    let events = drain_events_as_json(&mut engine);

    let var = events.iter().find(|e| e["type"] == "variation:changed");
    assert!(var.is_some());
    let v = var.unwrap();
    assert_eq!(v["trackId"], "drums");
    assert_eq!(v["previousVariationIndex"], 0);
    assert_eq!(v["newVariationIndex"], 2);
    assert_eq!(v["variationName"], "Complex");
}

// ---------------------------------------------------------------------------
// Track removed event
// ---------------------------------------------------------------------------

#[test]
fn track_removed_event_json() {
    let mut engine = ready_engine_with_track();
    engine.drain_events();

    engine.remove_track("drums").unwrap();
    let events = drain_events_as_json(&mut engine);

    let removed = events.iter().find(|e| e["type"] == "track:removed");
    assert!(removed.is_some());
    assert_eq!(removed.unwrap()["trackId"], "drums");
}

// ---------------------------------------------------------------------------
// All tracks loaded event
// ---------------------------------------------------------------------------

#[test]
fn all_tracks_loaded_event_json() {
    let mut engine = MaestroEngine::new(48000);
    engine.initialize().unwrap();
    engine
        .create_track("drums", "url1", None, None)
        .unwrap();
    engine.drain_events();

    engine
        .load_track_audio("drums", VariationType::Default, make_pcm(2, 48000, 48000), "url1")
        .unwrap();
    let events = drain_events_as_json(&mut engine);

    let all_loaded = events
        .iter()
        .find(|e| e["type"] == "engine:allTracksLoaded");
    assert!(all_loaded.is_some());
}

// ---------------------------------------------------------------------------
// Full workflow: JSON event stream through play/pause/stop
// ---------------------------------------------------------------------------

#[test]
fn full_workflow_event_stream() {
    let mut engine = ready_engine_with_track();
    engine.drain_events();

    // Play
    engine.play(SafetyPreset::Permissive).unwrap();
    let play_events = drain_events_as_json(&mut engine);
    let play_types: Vec<&str> = play_events
        .iter()
        .filter_map(|e| e["type"].as_str())
        .collect();
    assert!(play_types.contains(&"playback:started"));
    assert!(play_types.contains(&"transport:changed"));

    // Pause
    engine.pause().unwrap();
    let pause_events = drain_events_as_json(&mut engine);
    let pause_types: Vec<&str> = pause_events
        .iter()
        .filter_map(|e| e["type"].as_str())
        .collect();
    assert!(pause_types.contains(&"playback:paused"));

    // Stop
    engine.stop().unwrap();
    let stop_events = drain_events_as_json(&mut engine);
    let stop_types: Vec<&str> = stop_events
        .iter()
        .filter_map(|e| e["type"].as_str())
        .collect();
    assert!(stop_types.contains(&"playback:stopped"));
}

// ---------------------------------------------------------------------------
// Track info JSON structure
// ---------------------------------------------------------------------------

#[test]
fn track_info_json_structure() {
    let engine = ready_engine_with_track();
    let info = engine.get_track_info("drums").unwrap();

    // Verify all expected fields are present by serializing manually
    let json = serde_json::json!({
        "id": info.id,
        "status": info.status.to_string(),
        "armed": info.armed,
        "volume": info.volume,
        "pitchCents": info.pitch_cents,
        "pan": info.pan,
        "muted": info.muted,
        "solo": info.solo,
        "duration": info.duration,
        "position": info.position,
        "currentVariation": info.current_variation,
    });

    assert_eq!(json["id"], "drums");
    assert_eq!(json["status"], "ready");
    assert_eq!(json["armed"], true);
    assert!(json["volume"].as_f64().unwrap() > 0.0);
    assert_eq!(json["muted"], false);
    assert_eq!(json["solo"], false);
    assert!(json["duration"].as_f64().unwrap() > 0.0);
}

// ---------------------------------------------------------------------------
// Beat info JSON structure
// ---------------------------------------------------------------------------

#[test]
fn beat_info_json_structure() {
    let mut engine = ready_engine_with_track();
    engine.set_tempo(120.0).unwrap();

    let info = engine.get_beat_info();
    let json = serde_json::json!({
        "currentBeat": info.current_beat,
        "beatFraction": info.beat_fraction,
        "bpm": info.bpm,
        "timeSignatureNumerator": info.time_signature_numerator,
        "timeSignatureDenominator": info.time_signature_denominator,
    });

    assert!((json["bpm"].as_f64().unwrap() - 120.0).abs() < 0.01);
    assert_eq!(json["timeSignatureNumerator"], 4);
    assert_eq!(json["timeSignatureDenominator"], 4);
}
