//! # Event system for maestro-core
//!
//! Provides the event types and callback dispatch system that replaces the
//! TypeScript `UnifiedEventBus` + `AudioEngineEvents`.
//!
//! ## Design differences from TypeScript
//!
//! - **Enum-based events** instead of string-keyed event maps.  Each event
//!   variant carries its payload directly, giving exhaustive match coverage.
//! - **Collect-then-dispatch** pattern: the engine collects events into a
//!   `Vec<EngineEvent>` during an operation, then the platform layer drains
//!   and dispatches them (to React via Capacitor plugin, or to JS via
//!   wasm-bindgen callbacks).  No interior callback invocation.
//! - **No throttling/debouncing/batching** — those are platform-layer concerns
//!   (React's 10ms batch window, requestIdleCallback, etc.).
//! - **No priority system** — listeners execute in registration order.  The
//!   TypeScript priority system (`high`/`normal`/`low`) was only used for
//!   internal service ordering, which Rust handles via explicit call order.
//! - **Listener IDs** for O(1) removal instead of function-pointer equality.
//!
//! ## Usage
//!
//! The engine produces events:
//!
//! ```
//! use maestro_core::events::{EngineEvent, EventCollector};
//!
//! let mut collector = EventCollector::new();
//! collector.push(EngineEvent::PlaybackStarted);
//! collector.push(EngineEvent::TrackStarted {
//!     track_id: "drums".into(),
//!     active_track_count: 1,
//! });
//!
//! // Platform layer drains and dispatches
//! for event in collector.drain() {
//!     // send to Capacitor plugin / wasm-bindgen callback / etc.
//! }
//! ```
//!
//! The `EventBus` provides listener-based dispatch for contexts that need it:
//!
//! ```
//! use maestro_core::events::{EngineEvent, EventBus};
//!
//! let mut bus = EventBus::new();
//! let id = bus.add_listener(|event| {
//!     match event {
//!         EngineEvent::PlaybackStarted => println!("Playing!"),
//!         _ => {}
//!     }
//! });
//! bus.emit(&EngineEvent::PlaybackStarted);
//! bus.remove_listener(id);
//! ```

use crate::beat_tracker::BeatInfo;
use crate::track::TrackStatus;
use crate::transport::TransportState;

// ---------------------------------------------------------------------------
// Engine events
// ---------------------------------------------------------------------------

/// All events that the Maestro engine can emit.
///
/// Each variant corresponds to a TypeScript `AudioEngineEvents` key.
/// Payloads are embedded directly — no separate payload types needed for
/// simple cases.
#[derive(Debug, Clone, PartialEq)]
pub enum EngineEvent {
    // -- Playback events --------------------------------------------------

    /// Global playback started (at least one track playing).
    /// TS: `playback:started`
    PlaybackStarted,

    /// Global playback paused.
    /// TS: `playback:paused`
    PlaybackPaused,

    /// Global playback stopped.
    /// TS: `playback:stopped`
    PlaybackStopped,

    /// Composite playback state change.
    /// TS: `playback:stateChange`
    PlaybackStateChange {
        is_playing: bool,
        is_paused: bool,
    },

    // -- Transport events -------------------------------------------------

    /// Transport state changed.
    /// TS: emitted by TransportController state transitions.
    TransportChanged {
        state: TransportState,
    },

    // -- Tempo events -----------------------------------------------------

    /// Global tempo changed.
    /// TS: `tempo:changed`
    TempoChanged {
        tempo: f64,
        previous_tempo: f64,
    },

    // -- Track lifecycle events -------------------------------------------

    /// A new track was created and registered.
    /// TS: `track:created`
    TrackCreated {
        track_id: String,
    },

    /// A track was removed from the engine.
    /// TS: `track:removed`
    TrackRemoved {
        track_id: String,
    },

    /// Track's state machine transitioned.
    /// TS: `track:stateChanged`
    TrackStateChanged {
        track_id: String,
        state: TrackStatus,
        previous_state: TrackStatus,
    },

    /// Track finished loading audio data.
    /// TS: `track:loaded`
    TrackLoaded {
        track_id: String,
    },

    /// Track encountered an error.
    /// TS: `track:error`
    TrackError {
        track_id: String,
        message: String,
    },

    // -- Track playback events --------------------------------------------

    /// Track started playing (added to active set).
    /// TS: `track:started` (ADR081)
    TrackStarted {
        track_id: String,
        active_track_count: usize,
    },

    /// Track stopped playing (removed from active set).
    /// TS: `track:stopped` (ADR081)
    TrackStopped {
        track_id: String,
        active_track_count: usize,
    },

    /// Track's playing state changed.
    /// TS: `track:playingStateChanged`
    TrackPlayingStateChanged {
        track_id: String,
        is_playing: bool,
    },

    // -- Track control events ---------------------------------------------

    /// Track volume changed.
    /// TS: `track:volumeChanged`
    TrackVolumeChanged {
        track_id: String,
        volume: f64,
    },

    /// Track pitch changed.
    /// TS: `track:pitchChanged`
    TrackPitchChanged {
        track_id: String,
        offset_cents: f64,
        offset_semitones: f64,
    },

    /// Track armed state changed.
    TrackArmed {
        track_id: String,
    },

    /// Track disarmed.
    TrackDisarmed {
        track_id: String,
    },

    // -- Variation events -------------------------------------------------

    /// Track variation changed.
    /// TS: `variation:changed`
    VariationChanged {
        track_id: String,
        previous_variation_index: u32,
        new_variation_index: u32,
        variation_name: String,
    },

    // -- Beat/timing events -----------------------------------------------

    /// Beat position updated (high-frequency — typically per audio callback).
    /// TS: `position:updated`
    PositionUpdated {
        beat: f64,
        loop_progress: f64,
    },

    /// Precise beat info for UI display.
    /// TS: `beat:precise`
    BeatPrecise {
        beat_info: BeatInfo,
    },

    /// Bar boundary crossed.
    /// TS: `bar:changed`
    BarChanged {
        bar: u32,
        beat: u32,
    },

    /// Loop completed (returned to start).
    /// TS: `loop:completed`
    LoopCompleted {
        loop_count: u32,
    },

    // -- System events ----------------------------------------------------

    /// Engine initialized and ready.
    /// TS: `initialized`
    Initialized {
        sample_rate: u32,
    },

    /// All tracks have finished loading.
    /// TS: `allTracksLoaded`
    AllTracksLoaded,

    /// Engine encountered a non-fatal error.
    /// TS: `error`
    Error {
        message: String,
        context: String,
    },
}

impl std::fmt::Display for EngineEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EngineEvent::PlaybackStarted => write!(f, "playback:started"),
            EngineEvent::PlaybackPaused => write!(f, "playback:paused"),
            EngineEvent::PlaybackStopped => write!(f, "playback:stopped"),
            EngineEvent::PlaybackStateChange { .. } => write!(f, "playback:stateChange"),
            EngineEvent::TransportChanged { .. } => write!(f, "transport:changed"),
            EngineEvent::TempoChanged { .. } => write!(f, "tempo:changed"),
            EngineEvent::TrackCreated { .. } => write!(f, "track:created"),
            EngineEvent::TrackRemoved { .. } => write!(f, "track:removed"),
            EngineEvent::TrackStateChanged { .. } => write!(f, "track:stateChanged"),
            EngineEvent::TrackLoaded { .. } => write!(f, "track:loaded"),
            EngineEvent::TrackError { .. } => write!(f, "track:error"),
            EngineEvent::TrackStarted { .. } => write!(f, "track:started"),
            EngineEvent::TrackStopped { .. } => write!(f, "track:stopped"),
            EngineEvent::TrackPlayingStateChanged { .. } => {
                write!(f, "track:playingStateChanged")
            }
            EngineEvent::TrackVolumeChanged { .. } => write!(f, "track:volumeChanged"),
            EngineEvent::TrackPitchChanged { .. } => write!(f, "track:pitchChanged"),
            EngineEvent::TrackArmed { .. } => write!(f, "track:armed"),
            EngineEvent::TrackDisarmed { .. } => write!(f, "track:disarmed"),
            EngineEvent::VariationChanged { .. } => write!(f, "variation:changed"),
            EngineEvent::PositionUpdated { .. } => write!(f, "position:updated"),
            EngineEvent::BeatPrecise { .. } => write!(f, "beat:precise"),
            EngineEvent::BarChanged { .. } => write!(f, "bar:changed"),
            EngineEvent::LoopCompleted { .. } => write!(f, "loop:completed"),
            EngineEvent::Initialized { .. } => write!(f, "initialized"),
            EngineEvent::AllTracksLoaded => write!(f, "allTracksLoaded"),
            EngineEvent::Error { .. } => write!(f, "error"),
        }
    }
}

impl EngineEvent {
    /// Returns the event category for filtering.
    pub fn category(&self) -> EventCategory {
        match self {
            EngineEvent::PlaybackStarted
            | EngineEvent::PlaybackPaused
            | EngineEvent::PlaybackStopped
            | EngineEvent::PlaybackStateChange { .. } => EventCategory::Playback,

            EngineEvent::TransportChanged { .. } => EventCategory::Transport,

            EngineEvent::TempoChanged { .. } => EventCategory::Tempo,

            EngineEvent::TrackCreated { .. }
            | EngineEvent::TrackRemoved { .. }
            | EngineEvent::TrackStateChanged { .. }
            | EngineEvent::TrackLoaded { .. }
            | EngineEvent::TrackError { .. }
            | EngineEvent::TrackStarted { .. }
            | EngineEvent::TrackStopped { .. }
            | EngineEvent::TrackPlayingStateChanged { .. }
            | EngineEvent::TrackVolumeChanged { .. }
            | EngineEvent::TrackPitchChanged { .. }
            | EngineEvent::TrackArmed { .. }
            | EngineEvent::TrackDisarmed { .. } => EventCategory::Track,

            EngineEvent::VariationChanged { .. } => EventCategory::Variation,

            EngineEvent::PositionUpdated { .. }
            | EngineEvent::BeatPrecise { .. }
            | EngineEvent::BarChanged { .. }
            | EngineEvent::LoopCompleted { .. } => EventCategory::Timing,

            EngineEvent::Initialized { .. }
            | EngineEvent::AllTracksLoaded
            | EngineEvent::Error { .. } => EventCategory::System,
        }
    }
}

// ---------------------------------------------------------------------------
// Event categories (for filtering)
// ---------------------------------------------------------------------------

/// Broad categories for event filtering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EventCategory {
    Playback,
    Transport,
    Tempo,
    Track,
    Variation,
    Timing,
    System,
}

impl std::fmt::Display for EventCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EventCategory::Playback => write!(f, "playback"),
            EventCategory::Transport => write!(f, "transport"),
            EventCategory::Tempo => write!(f, "tempo"),
            EventCategory::Track => write!(f, "track"),
            EventCategory::Variation => write!(f, "variation"),
            EventCategory::Timing => write!(f, "timing"),
            EventCategory::System => write!(f, "system"),
        }
    }
}

// ---------------------------------------------------------------------------
// Event collector (collect-then-dispatch)
// ---------------------------------------------------------------------------

/// Accumulates events during an engine operation.
///
/// The platform layer drains the collector after each operation and dispatches
/// events to the appropriate targets (Capacitor plugin, wasm-bindgen callbacks,
/// React event bus, etc.).
///
/// This replaces the TypeScript pattern of calling `eventBus.emit()` inline
/// during service methods.  In Rust we collect events so the engine can remain
/// pure and testable — no callbacks during mutations.
#[derive(Debug, Default)]
pub struct EventCollector {
    events: Vec<EngineEvent>,
}

impl EventCollector {
    /// Create a new empty collector.
    pub fn new() -> Self {
        Self { events: Vec::new() }
    }

    /// Create a collector with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            events: Vec::with_capacity(capacity),
        }
    }

    /// Push an event into the collector.
    pub fn push(&mut self, event: EngineEvent) {
        self.events.push(event);
    }

    /// Push multiple events.
    pub fn extend(&mut self, events: impl IntoIterator<Item = EngineEvent>) {
        self.events.extend(events);
    }

    /// Drain all collected events, leaving the collector empty.
    pub fn drain(&mut self) -> Vec<EngineEvent> {
        std::mem::take(&mut self.events)
    }

    /// Number of pending events.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Whether the collector is empty.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Peek at collected events without draining.
    pub fn peek(&self) -> &[EngineEvent] {
        &self.events
    }

    /// Clear all pending events without returning them.
    pub fn clear(&mut self) {
        self.events.clear();
    }

    /// Check if a specific event type was collected.
    pub fn has_event(&self, predicate: impl Fn(&EngineEvent) -> bool) -> bool {
        self.events.iter().any(predicate)
    }
}

// ---------------------------------------------------------------------------
// Event bus (listener-based dispatch)
// ---------------------------------------------------------------------------

/// Unique identifier for a registered listener.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ListenerId(u64);

impl ListenerId {
    /// Get the raw numeric ID.
    pub fn as_u64(self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for ListenerId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "listener:{}", self.0)
    }
}

/// Listener-based event dispatch.
///
/// Replacement for the TypeScript `UnifiedEventBus`. Listeners receive a
/// reference to each emitted event.
///
/// In practice the Rust engine will mostly use [`EventCollector`] (collect
/// during operation, dispatch after).  The `EventBus` is useful for:
/// - Integration tests that want callback-style assertions.
/// - Platform layers that want to register Rust-side listeners (e.g.,
///   a metrics collector or a logger).
type Listener = Box<dyn Fn(&EngineEvent)>;

pub struct EventBus {
    listeners: Vec<(ListenerId, Listener)>,
    next_id: u64,
    /// Maximum listeners before logging a warning (memory leak detection).
    max_listeners: usize,
}

impl std::fmt::Debug for EventBus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EventBus")
            .field("listener_count", &self.listeners.len())
            .field("next_id", &self.next_id)
            .field("max_listeners", &self.max_listeners)
            .finish()
    }
}

/// Default max listeners (matches TypeScript's 100).
const DEFAULT_MAX_LISTENERS: usize = 100;

impl EventBus {
    /// Create a new event bus.
    pub fn new() -> Self {
        Self {
            listeners: Vec::new(),
            next_id: 0,
            max_listeners: DEFAULT_MAX_LISTENERS,
        }
    }

    /// Create a new event bus with a custom max listener warning threshold.
    pub fn with_max_listeners(max_listeners: usize) -> Self {
        Self {
            listeners: Vec::new(),
            next_id: 0,
            max_listeners,
        }
    }

    /// Register a listener.  Returns a [`ListenerId`] for later removal.
    pub fn add_listener(&mut self, listener: impl Fn(&EngineEvent) + 'static) -> ListenerId {
        let id = ListenerId(self.next_id);
        self.next_id += 1;
        self.listeners.push((id, Box::new(listener)));
        id
    }

    /// Remove a previously registered listener by ID.
    ///
    /// Returns `true` if a listener was found and removed.
    pub fn remove_listener(&mut self, id: ListenerId) -> bool {
        let len_before = self.listeners.len();
        self.listeners.retain(|(lid, _)| *lid != id);
        self.listeners.len() < len_before
    }

    /// Remove all listeners.
    pub fn remove_all_listeners(&mut self) {
        self.listeners.clear();
    }

    /// Emit an event to all registered listeners.
    pub fn emit(&self, event: &EngineEvent) {
        for (_, listener) in &self.listeners {
            listener(event);
        }
    }

    /// Emit all events from a collector, draining it.
    pub fn emit_all(&self, collector: &mut EventCollector) {
        for event in collector.drain() {
            self.emit(&event);
        }
    }

    /// Number of registered listeners.
    pub fn listener_count(&self) -> usize {
        self.listeners.len()
    }

    /// Whether the bus has exceeded the max listener threshold.
    ///
    /// The platform layer should check this and log a warning — potential
    /// memory leak from unremoved listeners.
    pub fn exceeds_max_listeners(&self) -> bool {
        self.listeners.len() > self.max_listeners
    }

    /// Get the max listener threshold.
    pub fn max_listeners(&self) -> usize {
        self.max_listeners
    }

    /// Set the max listener threshold.
    pub fn set_max_listeners(&mut self, max: usize) {
        self.max_listeners = max;
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Engine state (coordinator state machine)
// ---------------------------------------------------------------------------

/// State machine for the engine lifecycle.
///
/// Matches the TypeScript `CoordinatorState` exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineState {
    /// Constructed but not initialized.
    Uninitialized,
    /// Services are being created.
    Initializing,
    /// Services are created and ready (no tracks loaded).
    ServicesReady,
    /// At least one track is created (engines attached).
    EnginesReady,
    /// Fully operational — tracks loaded, ready for playback.
    Ready,
    /// Engine is suspended (e.g., app backgrounded).
    Suspended,
    /// Engine encountered a fatal error.
    Error,
    /// Engine is being destroyed.
    Destroying,
}

impl EngineState {
    /// Valid transitions from this state.
    pub fn can_transition_to(self, to: EngineState) -> bool {
        matches!(
            (self, to),
            // Initialization flow
            (EngineState::Uninitialized, EngineState::Initializing)
                | (EngineState::Initializing, EngineState::ServicesReady)
                | (EngineState::Initializing, EngineState::Error)
                | (EngineState::ServicesReady, EngineState::EnginesReady)
                | (EngineState::EnginesReady, EngineState::Ready)
                // Operational transitions
                | (EngineState::Ready, EngineState::Suspended)
                | (EngineState::Suspended, EngineState::Ready)
                // Error from any active state
                | (EngineState::ServicesReady, EngineState::Error)
                | (EngineState::EnginesReady, EngineState::Error)
                | (EngineState::Ready, EngineState::Error)
                // Error recovery
                | (EngineState::Error, EngineState::Uninitialized)
                // Destruction from any state
                | (_, EngineState::Destroying)
        )
    }
}

impl std::fmt::Display for EngineState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EngineState::Uninitialized => write!(f, "uninitialized"),
            EngineState::Initializing => write!(f, "initializing"),
            EngineState::ServicesReady => write!(f, "services-ready"),
            EngineState::EnginesReady => write!(f, "engines-ready"),
            EngineState::Ready => write!(f, "ready"),
            EngineState::Suspended => write!(f, "suspended"),
            EngineState::Error => write!(f, "error"),
            EngineState::Destroying => write!(f, "destroying"),
        }
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- EngineEvent tests ------------------------------------------------

    #[test]
    fn event_display_matches_typescript_names() {
        assert_eq!(EngineEvent::PlaybackStarted.to_string(), "playback:started");
        assert_eq!(EngineEvent::PlaybackPaused.to_string(), "playback:paused");
        assert_eq!(EngineEvent::PlaybackStopped.to_string(), "playback:stopped");
        assert_eq!(
            EngineEvent::PlaybackStateChange {
                is_playing: true,
                is_paused: false,
            }
            .to_string(),
            "playback:stateChange"
        );
        assert_eq!(
            EngineEvent::TransportChanged {
                state: TransportState::Playing,
            }
            .to_string(),
            "transport:changed"
        );
        assert_eq!(
            EngineEvent::TempoChanged {
                tempo: 120.0,
                previous_tempo: 100.0,
            }
            .to_string(),
            "tempo:changed"
        );
        assert_eq!(
            EngineEvent::TrackCreated {
                track_id: "x".into(),
            }
            .to_string(),
            "track:created"
        );
        assert_eq!(
            EngineEvent::TrackRemoved {
                track_id: "x".into(),
            }
            .to_string(),
            "track:removed"
        );
        assert_eq!(
            EngineEvent::TrackLoaded {
                track_id: "x".into(),
            }
            .to_string(),
            "track:loaded"
        );
        assert_eq!(
            EngineEvent::TrackError {
                track_id: "x".into(),
                message: "oops".into(),
            }
            .to_string(),
            "track:error"
        );
        assert_eq!(
            EngineEvent::TrackStarted {
                track_id: "x".into(),
                active_track_count: 1,
            }
            .to_string(),
            "track:started"
        );
        assert_eq!(
            EngineEvent::TrackStopped {
                track_id: "x".into(),
                active_track_count: 0,
            }
            .to_string(),
            "track:stopped"
        );
        assert_eq!(
            EngineEvent::TrackVolumeChanged {
                track_id: "x".into(),
                volume: 0.5,
            }
            .to_string(),
            "track:volumeChanged"
        );
        assert_eq!(
            EngineEvent::TrackPitchChanged {
                track_id: "x".into(),
                offset_cents: 100.0,
                offset_semitones: 1.0,
            }
            .to_string(),
            "track:pitchChanged"
        );
        assert_eq!(
            EngineEvent::TrackArmed {
                track_id: "x".into(),
            }
            .to_string(),
            "track:armed"
        );
        assert_eq!(
            EngineEvent::TrackDisarmed {
                track_id: "x".into(),
            }
            .to_string(),
            "track:disarmed"
        );
        assert_eq!(
            EngineEvent::VariationChanged {
                track_id: "x".into(),
                previous_variation_index: 0,
                new_variation_index: 1,
                variation_name: "Complex".into(),
            }
            .to_string(),
            "variation:changed"
        );
        assert_eq!(
            EngineEvent::PositionUpdated {
                beat: 1.5,
                loop_progress: 0.25,
            }
            .to_string(),
            "position:updated"
        );
        assert_eq!(
            EngineEvent::BarChanged { bar: 2, beat: 1 }.to_string(),
            "bar:changed"
        );
        assert_eq!(
            EngineEvent::LoopCompleted { loop_count: 3 }.to_string(),
            "loop:completed"
        );
        assert_eq!(
            EngineEvent::Initialized { sample_rate: 48000 }.to_string(),
            "initialized"
        );
        assert_eq!(EngineEvent::AllTracksLoaded.to_string(), "allTracksLoaded");
        assert_eq!(
            EngineEvent::Error {
                message: "bad".into(),
                context: "init".into(),
            }
            .to_string(),
            "error"
        );
    }

    #[test]
    fn event_category_classification() {
        assert_eq!(
            EngineEvent::PlaybackStarted.category(),
            EventCategory::Playback
        );
        assert_eq!(
            EngineEvent::TransportChanged {
                state: TransportState::Stopped
            }
            .category(),
            EventCategory::Transport
        );
        assert_eq!(
            EngineEvent::TempoChanged {
                tempo: 120.0,
                previous_tempo: 100.0
            }
            .category(),
            EventCategory::Tempo
        );
        assert_eq!(
            EngineEvent::TrackCreated {
                track_id: "a".into()
            }
            .category(),
            EventCategory::Track
        );
        assert_eq!(
            EngineEvent::TrackVolumeChanged {
                track_id: "a".into(),
                volume: 0.5,
            }
            .category(),
            EventCategory::Track
        );
        assert_eq!(
            EngineEvent::VariationChanged {
                track_id: "a".into(),
                previous_variation_index: 0,
                new_variation_index: 1,
                variation_name: "X".into(),
            }
            .category(),
            EventCategory::Variation
        );
        assert_eq!(
            EngineEvent::PositionUpdated {
                beat: 0.0,
                loop_progress: 0.0
            }
            .category(),
            EventCategory::Timing
        );
        assert_eq!(
            EngineEvent::BarChanged { bar: 1, beat: 1 }.category(),
            EventCategory::Timing
        );
        assert_eq!(
            EngineEvent::Initialized { sample_rate: 48000 }.category(),
            EventCategory::System
        );
        assert_eq!(
            EngineEvent::AllTracksLoaded.category(),
            EventCategory::System
        );
    }

    // -- EventCollector tests ---------------------------------------------

    #[test]
    fn collector_starts_empty() {
        let collector = EventCollector::new();
        assert!(collector.is_empty());
        assert_eq!(collector.len(), 0);
    }

    #[test]
    fn collector_push_and_drain() {
        let mut collector = EventCollector::new();
        collector.push(EngineEvent::PlaybackStarted);
        collector.push(EngineEvent::PlaybackStopped);
        assert_eq!(collector.len(), 2);
        assert!(!collector.is_empty());

        let events = collector.drain();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0], EngineEvent::PlaybackStarted);
        assert_eq!(events[1], EngineEvent::PlaybackStopped);

        // After drain, collector is empty
        assert!(collector.is_empty());
    }

    #[test]
    fn collector_extend() {
        let mut collector = EventCollector::new();
        collector.extend(vec![
            EngineEvent::PlaybackStarted,
            EngineEvent::PlaybackPaused,
            EngineEvent::PlaybackStopped,
        ]);
        assert_eq!(collector.len(), 3);
    }

    #[test]
    fn collector_peek_does_not_drain() {
        let mut collector = EventCollector::new();
        collector.push(EngineEvent::AllTracksLoaded);
        assert_eq!(collector.peek().len(), 1);
        assert_eq!(collector.len(), 1); // Still there
    }

    #[test]
    fn collector_clear() {
        let mut collector = EventCollector::new();
        collector.push(EngineEvent::PlaybackStarted);
        collector.push(EngineEvent::PlaybackStopped);
        collector.clear();
        assert!(collector.is_empty());
    }

    #[test]
    fn collector_has_event() {
        let mut collector = EventCollector::new();
        collector.push(EngineEvent::TrackStarted {
            track_id: "drums".into(),
            active_track_count: 1,
        });
        collector.push(EngineEvent::PlaybackStarted);

        assert!(collector.has_event(|e| matches!(e, EngineEvent::PlaybackStarted)));
        assert!(collector.has_event(|e| matches!(e, EngineEvent::TrackStarted { .. })));
        assert!(!collector.has_event(|e| matches!(e, EngineEvent::PlaybackStopped)));
    }

    #[test]
    fn collector_with_capacity() {
        let collector = EventCollector::with_capacity(16);
        assert!(collector.is_empty());
    }

    // -- EventBus tests ---------------------------------------------------

    #[test]
    fn bus_starts_empty() {
        let bus = EventBus::new();
        assert_eq!(bus.listener_count(), 0);
        assert!(!bus.exceeds_max_listeners());
    }

    #[test]
    fn bus_add_and_remove_listener() {
        let mut bus = EventBus::new();
        let id1 = bus.add_listener(|_| {});
        let id2 = bus.add_listener(|_| {});
        assert_eq!(bus.listener_count(), 2);

        assert!(bus.remove_listener(id1));
        assert_eq!(bus.listener_count(), 1);

        assert!(bus.remove_listener(id2));
        assert_eq!(bus.listener_count(), 0);
    }

    #[test]
    fn bus_remove_nonexistent_returns_false() {
        let mut bus = EventBus::new();
        assert!(!bus.remove_listener(ListenerId(999)));
    }

    #[test]
    fn bus_remove_all_listeners() {
        let mut bus = EventBus::new();
        bus.add_listener(|_| {});
        bus.add_listener(|_| {});
        bus.add_listener(|_| {});
        assert_eq!(bus.listener_count(), 3);

        bus.remove_all_listeners();
        assert_eq!(bus.listener_count(), 0);
    }

    #[test]
    fn bus_emit_calls_listeners() {
        use std::cell::Cell;

        let mut bus = EventBus::new();
        let called = std::rc::Rc::new(Cell::new(0u32));

        let called_clone = called.clone();
        bus.add_listener(move |_| {
            called_clone.set(called_clone.get() + 1);
        });

        let called_clone2 = called.clone();
        bus.add_listener(move |_| {
            called_clone2.set(called_clone2.get() + 1);
        });

        bus.emit(&EngineEvent::PlaybackStarted);
        assert_eq!(called.get(), 2);

        bus.emit(&EngineEvent::PlaybackStopped);
        assert_eq!(called.get(), 4);
    }

    #[test]
    fn bus_emit_passes_correct_event() {
        use std::cell::RefCell;

        let mut bus = EventBus::new();
        let received = std::rc::Rc::new(RefCell::new(Vec::new()));

        let received_clone = received.clone();
        bus.add_listener(move |event| {
            received_clone.borrow_mut().push(event.to_string());
        });

        bus.emit(&EngineEvent::TempoChanged {
            tempo: 140.0,
            previous_tempo: 120.0,
        });
        bus.emit(&EngineEvent::TrackCreated {
            track_id: "bass".into(),
        });

        let names = received.borrow();
        assert_eq!(names.len(), 2);
        assert_eq!(names[0], "tempo:changed");
        assert_eq!(names[1], "track:created");
    }

    #[test]
    fn bus_emit_all_from_collector() {
        use std::cell::Cell;

        let mut bus = EventBus::new();
        let count = std::rc::Rc::new(Cell::new(0u32));

        let count_clone = count.clone();
        bus.add_listener(move |_| {
            count_clone.set(count_clone.get() + 1);
        });

        let mut collector = EventCollector::new();
        collector.push(EngineEvent::PlaybackStarted);
        collector.push(EngineEvent::TrackStarted {
            track_id: "drums".into(),
            active_track_count: 1,
        });
        collector.push(EngineEvent::PlaybackStopped);

        bus.emit_all(&mut collector);
        assert_eq!(count.get(), 3);
        assert!(collector.is_empty()); // Drained
    }

    #[test]
    fn bus_exceeds_max_listeners_warning() {
        let mut bus = EventBus::with_max_listeners(3);
        bus.add_listener(|_| {});
        bus.add_listener(|_| {});
        bus.add_listener(|_| {});
        assert!(!bus.exceeds_max_listeners()); // Exactly at limit

        bus.add_listener(|_| {});
        assert!(bus.exceeds_max_listeners()); // Over limit
    }

    #[test]
    fn bus_listener_ids_are_unique() {
        let mut bus = EventBus::new();
        let id1 = bus.add_listener(|_| {});
        let id2 = bus.add_listener(|_| {});
        let id3 = bus.add_listener(|_| {});
        assert_ne!(id1, id2);
        assert_ne!(id2, id3);
        assert_ne!(id1, id3);
    }

    #[test]
    fn bus_removed_listener_not_called() {
        use std::cell::Cell;

        let mut bus = EventBus::new();
        let called = std::rc::Rc::new(Cell::new(false));

        let called_clone = called.clone();
        let id = bus.add_listener(move |_| {
            called_clone.set(true);
        });

        bus.remove_listener(id);
        bus.emit(&EngineEvent::PlaybackStarted);
        assert!(!called.get());
    }

    #[test]
    fn bus_default_max_listeners() {
        let bus = EventBus::new();
        assert_eq!(bus.max_listeners(), DEFAULT_MAX_LISTENERS);
    }

    #[test]
    fn bus_set_max_listeners() {
        let mut bus = EventBus::new();
        bus.set_max_listeners(50);
        assert_eq!(bus.max_listeners(), 50);
    }

    #[test]
    fn listener_id_display() {
        let id = ListenerId(42);
        assert_eq!(id.to_string(), "listener:42");
        assert_eq!(id.as_u64(), 42);
    }

    // -- EngineState tests ------------------------------------------------

    #[test]
    fn engine_state_initialization_flow() {
        assert!(EngineState::Uninitialized.can_transition_to(EngineState::Initializing));
        assert!(EngineState::Initializing.can_transition_to(EngineState::ServicesReady));
        assert!(EngineState::ServicesReady.can_transition_to(EngineState::EnginesReady));
        assert!(EngineState::EnginesReady.can_transition_to(EngineState::Ready));
    }

    #[test]
    fn engine_state_suspend_resume() {
        assert!(EngineState::Ready.can_transition_to(EngineState::Suspended));
        assert!(EngineState::Suspended.can_transition_to(EngineState::Ready));
    }

    #[test]
    fn engine_state_error_from_active_states() {
        assert!(EngineState::Initializing.can_transition_to(EngineState::Error));
        assert!(EngineState::ServicesReady.can_transition_to(EngineState::Error));
        assert!(EngineState::EnginesReady.can_transition_to(EngineState::Error));
        assert!(EngineState::Ready.can_transition_to(EngineState::Error));
    }

    #[test]
    fn engine_state_error_recovery() {
        assert!(EngineState::Error.can_transition_to(EngineState::Uninitialized));
    }

    #[test]
    fn engine_state_destroying_from_anywhere() {
        assert!(EngineState::Uninitialized.can_transition_to(EngineState::Destroying));
        assert!(EngineState::Initializing.can_transition_to(EngineState::Destroying));
        assert!(EngineState::ServicesReady.can_transition_to(EngineState::Destroying));
        assert!(EngineState::EnginesReady.can_transition_to(EngineState::Destroying));
        assert!(EngineState::Ready.can_transition_to(EngineState::Destroying));
        assert!(EngineState::Suspended.can_transition_to(EngineState::Destroying));
        assert!(EngineState::Error.can_transition_to(EngineState::Destroying));
    }

    #[test]
    fn engine_state_invalid_transitions() {
        // Can't skip from Uninitialized to Ready
        assert!(!EngineState::Uninitialized.can_transition_to(EngineState::Ready));
        // Can't go backwards in init flow
        assert!(!EngineState::ServicesReady.can_transition_to(EngineState::Initializing));
        // Can't go from Suspended to Error (must resume first)
        assert!(!EngineState::Suspended.can_transition_to(EngineState::Error));
    }

    #[test]
    fn engine_state_display() {
        assert_eq!(EngineState::Uninitialized.to_string(), "uninitialized");
        assert_eq!(EngineState::Initializing.to_string(), "initializing");
        assert_eq!(EngineState::ServicesReady.to_string(), "services-ready");
        assert_eq!(EngineState::EnginesReady.to_string(), "engines-ready");
        assert_eq!(EngineState::Ready.to_string(), "ready");
        assert_eq!(EngineState::Suspended.to_string(), "suspended");
        assert_eq!(EngineState::Error.to_string(), "error");
        assert_eq!(EngineState::Destroying.to_string(), "destroying");
    }

    // -- EventCategory tests ----------------------------------------------

    #[test]
    fn event_category_display() {
        assert_eq!(EventCategory::Playback.to_string(), "playback");
        assert_eq!(EventCategory::Transport.to_string(), "transport");
        assert_eq!(EventCategory::Tempo.to_string(), "tempo");
        assert_eq!(EventCategory::Track.to_string(), "track");
        assert_eq!(EventCategory::Variation.to_string(), "variation");
        assert_eq!(EventCategory::Timing.to_string(), "timing");
        assert_eq!(EventCategory::System.to_string(), "system");
    }

    #[test]
    fn event_clone_and_eq() {
        let e1 = EngineEvent::TrackStarted {
            track_id: "drums".into(),
            active_track_count: 2,
        };
        let e2 = e1.clone();
        assert_eq!(e1, e2);
    }
}
