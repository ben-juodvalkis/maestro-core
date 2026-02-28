//! Transport state machine and safety presets for the audio engine.
//!
//! The `TransportController` is the centralized controller for all transport
//! decisions — the single source of truth for transport state transitions.
//! It receives operation context before track operations and makes decisions
//! about auto-start/auto-stop based on safety presets.
//!
//! Ported from:
//! - `src/domains/coordination/services/transport/TransportController.ts`
//! - `src/domains/coordination/services/transport/types.ts`
//! - `src/hooks/engine/transportSafetyPresets.ts`
//!
//! Browser-specific concerns stripped: event bus wiring, dependency injection
//! of stateService/syncService callbacks.  In Rust, `on_track_started` and
//! `on_track_stopped` return a `TransportAction` enum describing what the
//! caller (engine) should do — the engine decides how to fulfill it.
//!
//! See ADR081 for architectural context.

use std::collections::HashSet;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum number of context changes to keep in history (for debugging).
const MAX_CONTEXT_HISTORY: usize = 50;

// ---------------------------------------------------------------------------
// Transport state
// ---------------------------------------------------------------------------

/// Global transport status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransportState {
    /// Transport is idle — no playback infrastructure running.
    Stopped,
    /// Transport is active — beat tracking, timeline, playback all running.
    Playing,
    /// Transport is paused — timeline preserved but beat tracking stopped.
    Paused,
}

impl std::fmt::Display for TransportState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TransportState::Stopped => write!(f, "stopped"),
            TransportState::Playing => write!(f, "playing"),
            TransportState::Paused => write!(f, "paused"),
        }
    }
}

// ---------------------------------------------------------------------------
// Safety presets (ADR081)
// ---------------------------------------------------------------------------

/// Flags controlling automatic transport start/stop behavior.
///
/// These are the critical rules that prevent UI interactions from
/// accidentally starting or stopping playback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SafetyFlags {
    /// Whether transport can auto-start when the first track begins playing.
    pub allow_auto_start: bool,
    /// Whether transport can auto-stop when the last track stops.
    pub allow_auto_stop: bool,
}

/// Predefined transport safety configurations for different interaction contexts.
///
/// Matches the TypeScript `TRANSPORT_SAFETY_PRESETS` exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafetyPreset {
    /// Slider operations (complexity, rhythmic-tonal controls).
    /// Never auto-starts, never auto-stops.
    SliderOperation,
    /// Direct track interactions (clicking track buttons).
    /// Can auto-start and auto-stop.
    TrackInteraction,
    /// Manual transport control (play/pause button).
    /// Never auto-starts, can auto-stop for safety.
    ManualControl,
    /// Programmatic operations (loading songs, switching variations).
    /// Never auto-starts, never auto-stops.
    ProgrammaticOperation,
    /// Testing/development — allows all transport operations.
    Permissive,
}

impl SafetyPreset {
    /// Get the safety flags for this preset.
    pub fn flags(self) -> SafetyFlags {
        match self {
            SafetyPreset::SliderOperation => SafetyFlags {
                allow_auto_start: false,
                allow_auto_stop: false,
            },
            SafetyPreset::TrackInteraction => SafetyFlags {
                allow_auto_start: true,
                allow_auto_stop: true,
            },
            SafetyPreset::ManualControl => SafetyFlags {
                allow_auto_start: false,
                allow_auto_stop: true,
            },
            SafetyPreset::ProgrammaticOperation => SafetyFlags {
                allow_auto_start: false,
                allow_auto_stop: false,
            },
            SafetyPreset::Permissive => SafetyFlags {
                allow_auto_start: true,
                allow_auto_stop: true,
            },
        }
    }
}

impl std::fmt::Display for SafetyPreset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SafetyPreset::SliderOperation => write!(f, "SLIDER_OPERATION"),
            SafetyPreset::TrackInteraction => write!(f, "TRACK_INTERACTION"),
            SafetyPreset::ManualControl => write!(f, "MANUAL_CONTROL"),
            SafetyPreset::ProgrammaticOperation => write!(f, "PROGRAMMATIC_OPERATION"),
            SafetyPreset::Permissive => write!(f, "PERMISSIVE"),
        }
    }
}

/// Identify which preset matches a given pair of flags, or `None` for custom.
///
/// Note: `SliderOperation` and `ProgrammaticOperation` share the same flags
/// `(false, false)`.  Similarly, `TrackInteraction` and `Permissive` share
/// `(true, true)`.  This function returns the first match in canonical order,
/// so it may not roundtrip for aliases.
pub fn identify_safety_preset(flags: SafetyFlags) -> Option<SafetyPreset> {
    match (flags.allow_auto_start, flags.allow_auto_stop) {
        (false, false) => Some(SafetyPreset::SliderOperation),
        (true, true) => Some(SafetyPreset::TrackInteraction),
        (false, true) => Some(SafetyPreset::ManualControl),
        (true, false) => None, // No standard preset has this combination
    }
}

// ---------------------------------------------------------------------------
// Operation context
// ---------------------------------------------------------------------------

/// Source of the transport operation — helps with debugging and logging.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationSource {
    /// Slider operations (rhythmic-tonal, complexity).
    LiveMixing,
    /// Direct track clicks.
    UserAction,
    /// Play/pause/stop buttons.
    TransportButton,
    /// Loading, switching variations.
    Programmatic,
    /// playAll, stopAll.
    BatchOperation,
}

impl std::fmt::Display for OperationSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OperationSource::LiveMixing => write!(f, "live-mixing"),
            OperationSource::UserAction => write!(f, "user-action"),
            OperationSource::TransportButton => write!(f, "transport-button"),
            OperationSource::Programmatic => write!(f, "programmatic"),
            OperationSource::BatchOperation => write!(f, "batch-operation"),
        }
    }
}

/// Context passed before track operations to inform transport decisions.
#[derive(Debug, Clone, PartialEq)]
pub struct OperationContext {
    /// Where the operation originated.
    pub source: OperationSource,
    /// Which safety preset governs transport behavior.
    pub safety_preset: SafetyPreset,
    /// Optional reason for debugging.
    pub reason: Option<String>,
    /// Optional track ID being operated on.
    pub track_id: Option<String>,
}

// ---------------------------------------------------------------------------
// Transport actions (returned to caller instead of calling deps directly)
// ---------------------------------------------------------------------------

/// Action the engine should take after a transport decision.
///
/// In TypeScript the TransportController called into stateService/syncService
/// directly.  In Rust we return an action so the engine can fulfill it — this
/// keeps transport pure and testable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransportAction {
    /// Initialize transport infrastructure: start beat tracking, init timeline,
    /// update global state to playing.
    Initialize,
    /// Resume from paused: restart beat tracking, update global state to playing.
    Resume,
    /// Pause transport: stop beat tracking, update global state to paused.
    Pause,
    /// Full shutdown: stop beat tracking, reset timeline, update global state
    /// to stopped.
    Shutdown,
    /// No action needed — transport state is already correct or auto-start/stop
    /// is disabled by the active safety preset.
    None,
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that can occur during transport operations.
#[derive(Debug, Clone, PartialEq)]
pub enum TransportError {
    /// Invalid state transition attempted.
    InvalidTransition {
        from: TransportState,
        action: &'static str,
    },
    /// Track ID not found in active set.
    TrackNotActive(String),
}

impl std::fmt::Display for TransportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TransportError::InvalidTransition { from, action } => {
                write!(f, "Cannot {action} from transport state '{from}'")
            }
            TransportError::TrackNotActive(id) => {
                write!(f, "Track '{id}' is not in the active set")
            }
        }
    }
}

impl std::error::Error for TransportError {}

// ---------------------------------------------------------------------------
// TransportController
// ---------------------------------------------------------------------------

/// Centralized controller for all transport decisions.
///
/// The controller tracks:
/// - Current transport state (`Stopped` / `Playing` / `Paused`).
/// - Active track set (which tracks are currently playing).
/// - Operation context (safety preset governing current operation).
///
/// When a track starts or stops the caller invokes [`on_track_started`] or
/// [`on_track_stopped`].  The controller returns a [`TransportAction`] the
/// engine should execute.
///
/// [`on_track_started`]: TransportController::on_track_started
/// [`on_track_stopped`]: TransportController::on_track_stopped
pub struct TransportController {
    /// Current transport state.
    state: TransportState,
    /// Currently active (playing) track IDs.
    active_tracks: HashSet<String>,
    /// Operation context set before track operations.
    context: Option<OperationContext>,
    /// Rolling context history for debugging (capped at `MAX_CONTEXT_HISTORY`).
    context_history: Vec<OperationContext>,
}

impl TransportController {
    /// Create a new controller in the `Stopped` state.
    pub fn new() -> Self {
        Self {
            state: TransportState::Stopped,
            active_tracks: HashSet::new(),
            context: None,
            context_history: Vec::new(),
        }
    }

    // -----------------------------------------------------------------------
    // Context management
    // -----------------------------------------------------------------------

    /// Set operation context BEFORE performing track operations.
    ///
    /// This determines how the controller responds to the next
    /// `on_track_started` / `on_track_stopped` call.
    pub fn set_operation_context(&mut self, context: OperationContext) {
        // Record in history
        if self.context_history.len() >= MAX_CONTEXT_HISTORY {
            self.context_history.remove(0);
        }
        self.context_history.push(context.clone());
        self.context = Some(context);
    }

    /// Clear operation context after the operation completes.
    /// Should be called in a cleanup/finally path.
    pub fn clear_operation_context(&mut self) {
        self.context = None;
    }

    /// Get current operation context (for debugging/testing).
    pub fn operation_context(&self) -> Option<&OperationContext> {
        self.context.as_ref()
    }

    /// Get context history (for debugging).
    pub fn context_history(&self) -> &[OperationContext] {
        &self.context_history
    }

    // -----------------------------------------------------------------------
    // Event handlers — return actions for the engine to execute
    // -----------------------------------------------------------------------

    /// Called when a track starts playing.
    ///
    /// Adds the track to the active set and decides whether to initialize
    /// or resume transport based on the current safety preset.
    ///
    /// Returns a `TransportAction` the engine should execute.
    pub fn on_track_started(&mut self, track_id: impl Into<String>) -> TransportAction {
        let id = track_id.into();
        self.active_tracks.insert(id);

        let active_count = self.active_tracks.len();
        let flags = self.effective_flags();

        // First track starting — may need to initialize transport
        if active_count == 1 {
            match self.state {
                TransportState::Stopped => {
                    if flags.allow_auto_start {
                        self.state = TransportState::Playing;
                        TransportAction::Initialize
                    } else {
                        TransportAction::None
                    }
                }
                TransportState::Paused => {
                    // Resume from paused (auto-start flag doesn't apply here)
                    self.state = TransportState::Playing;
                    TransportAction::Resume
                }
                TransportState::Playing => TransportAction::None,
            }
        } else {
            // Additional tracks — transport already running or not, no action
            TransportAction::None
        }
    }

    /// Called when a track stops playing.
    ///
    /// Removes the track from the active set and decides whether to
    /// shut down transport based on the current safety preset.
    ///
    /// Returns a `TransportAction` the engine should execute.
    pub fn on_track_stopped(&mut self, track_id: &str) -> TransportAction {
        self.active_tracks.remove(track_id);

        let active_count = self.active_tracks.len();
        let flags = self.effective_flags();

        // Last track stopped — may need to shutdown transport
        if active_count == 0 {
            if flags.allow_auto_stop {
                self.state = TransportState::Stopped;
                TransportAction::Shutdown
            } else {
                // Keep transport infrastructure running so new tracks can sync
                TransportAction::None
            }
        } else {
            TransportAction::None
        }
    }

    // -----------------------------------------------------------------------
    // Manual transport controls
    // -----------------------------------------------------------------------

    /// Manually pause transport.
    ///
    /// Returns `Pause` action if currently playing, `None` otherwise.
    pub fn pause(&mut self) -> TransportAction {
        if self.state == TransportState::Playing {
            self.state = TransportState::Paused;
            TransportAction::Pause
        } else {
            TransportAction::None
        }
    }

    /// Manually resume transport from paused.
    ///
    /// Returns `Resume` action if currently paused, `None` otherwise.
    pub fn resume(&mut self) -> TransportAction {
        if self.state == TransportState::Paused {
            self.state = TransportState::Playing;
            TransportAction::Resume
        } else {
            TransportAction::None
        }
    }

    /// Manually stop (shutdown) transport.
    ///
    /// Returns `Shutdown` action if not already stopped, `None` otherwise.
    /// Also clears the active track set.
    pub fn stop(&mut self) -> TransportAction {
        if self.state != TransportState::Stopped {
            self.state = TransportState::Stopped;
            self.active_tracks.clear();
            TransportAction::Shutdown
        } else {
            TransportAction::None
        }
    }

    // -----------------------------------------------------------------------
    // State queries
    // -----------------------------------------------------------------------

    /// Current transport state.
    pub fn state(&self) -> TransportState {
        self.state
    }

    /// Whether transport is active (playing or paused with timeline).
    pub fn is_active(&self) -> bool {
        self.state != TransportState::Stopped
    }

    /// Whether transport is currently playing.
    pub fn is_playing(&self) -> bool {
        self.state == TransportState::Playing
    }

    /// Whether transport is currently paused.
    pub fn is_paused(&self) -> bool {
        self.state == TransportState::Paused
    }

    /// Whether transport is stopped.
    pub fn is_stopped(&self) -> bool {
        self.state == TransportState::Stopped
    }

    /// Number of currently active (playing) tracks.
    pub fn active_track_count(&self) -> usize {
        self.active_tracks.len()
    }

    /// Whether a specific track is currently active.
    pub fn is_track_active(&self, track_id: &str) -> bool {
        self.active_tracks.contains(track_id)
    }

    /// Whether there are any active tracks.
    pub fn has_active_tracks(&self) -> bool {
        !self.active_tracks.is_empty()
    }

    /// Get all active track IDs (unordered).
    pub fn active_track_ids(&self) -> Vec<&str> {
        self.active_tracks.iter().map(|s| s.as_str()).collect()
    }

    // -----------------------------------------------------------------------
    // Testing/override
    // -----------------------------------------------------------------------

    /// Force transport state (for testing or manual override).
    /// Bypasses normal state machine transitions.
    pub fn force_state(&mut self, state: TransportState) {
        self.state = state;
    }

    /// Reset controller to initial state, clearing all tracks and context.
    pub fn reset(&mut self) {
        self.state = TransportState::Stopped;
        self.active_tracks.clear();
        self.context = None;
        self.context_history.clear();
    }

    // -----------------------------------------------------------------------
    // Internals
    // -----------------------------------------------------------------------

    /// Get effective safety flags — uses context preset or defaults to Permissive.
    fn effective_flags(&self) -> SafetyFlags {
        self.context
            .as_ref()
            .map(|ctx| ctx.safety_preset.flags())
            .unwrap_or_else(|| SafetyPreset::Permissive.flags())
    }
}

impl Default for TransportController {
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
    // TransportState display
    // -----------------------------------------------------------------------

    mod transport_state {
        use super::*;

        #[test]
        fn display() {
            assert_eq!(TransportState::Stopped.to_string(), "stopped");
            assert_eq!(TransportState::Playing.to_string(), "playing");
            assert_eq!(TransportState::Paused.to_string(), "paused");
        }

        #[test]
        fn equality() {
            assert_eq!(TransportState::Stopped, TransportState::Stopped);
            assert_ne!(TransportState::Stopped, TransportState::Playing);
        }
    }

    // -----------------------------------------------------------------------
    // Safety presets
    // -----------------------------------------------------------------------

    mod safety_presets {
        use super::*;

        #[test]
        fn slider_operation_flags() {
            let f = SafetyPreset::SliderOperation.flags();
            assert!(!f.allow_auto_start);
            assert!(!f.allow_auto_stop);
        }

        #[test]
        fn track_interaction_flags() {
            let f = SafetyPreset::TrackInteraction.flags();
            assert!(f.allow_auto_start);
            assert!(f.allow_auto_stop);
        }

        #[test]
        fn manual_control_flags() {
            let f = SafetyPreset::ManualControl.flags();
            assert!(!f.allow_auto_start);
            assert!(f.allow_auto_stop);
        }

        #[test]
        fn programmatic_operation_flags() {
            let f = SafetyPreset::ProgrammaticOperation.flags();
            assert!(!f.allow_auto_start);
            assert!(!f.allow_auto_stop);
        }

        #[test]
        fn permissive_flags() {
            let f = SafetyPreset::Permissive.flags();
            assert!(f.allow_auto_start);
            assert!(f.allow_auto_stop);
        }

        #[test]
        fn display() {
            assert_eq!(SafetyPreset::SliderOperation.to_string(), "SLIDER_OPERATION");
            assert_eq!(SafetyPreset::TrackInteraction.to_string(), "TRACK_INTERACTION");
            assert_eq!(SafetyPreset::ManualControl.to_string(), "MANUAL_CONTROL");
            assert_eq!(
                SafetyPreset::ProgrammaticOperation.to_string(),
                "PROGRAMMATIC_OPERATION"
            );
            assert_eq!(SafetyPreset::Permissive.to_string(), "PERMISSIVE");
        }

        #[test]
        fn identify_unique_presets() {
            // SliderOperation (false, false) — canonical for this flag combo
            assert_eq!(
                identify_safety_preset(SafetyPreset::SliderOperation.flags()),
                Some(SafetyPreset::SliderOperation)
            );
            // TrackInteraction (true, true) — canonical for this flag combo
            assert_eq!(
                identify_safety_preset(SafetyPreset::TrackInteraction.flags()),
                Some(SafetyPreset::TrackInteraction)
            );
            // ManualControl (false, true) — unique
            assert_eq!(
                identify_safety_preset(SafetyPreset::ManualControl.flags()),
                Some(SafetyPreset::ManualControl)
            );
        }

        #[test]
        fn identify_aliased_presets() {
            // ProgrammaticOperation has same flags as SliderOperation (false, false)
            // → returns SliderOperation (canonical first match)
            assert_eq!(
                identify_safety_preset(SafetyPreset::ProgrammaticOperation.flags()),
                Some(SafetyPreset::SliderOperation)
            );
            // Permissive has same flags as TrackInteraction (true, true)
            // → returns TrackInteraction (canonical first match)
            assert_eq!(
                identify_safety_preset(SafetyPreset::Permissive.flags()),
                Some(SafetyPreset::TrackInteraction)
            );
        }

        #[test]
        fn identify_custom_flags() {
            // (true, false) — no standard preset has this combination
            let custom = SafetyFlags {
                allow_auto_start: true,
                allow_auto_stop: false,
            };
            assert_eq!(identify_safety_preset(custom), None);
        }
    }

    // -----------------------------------------------------------------------
    // Operation context
    // -----------------------------------------------------------------------

    mod context {
        use super::*;

        fn make_context(preset: SafetyPreset) -> OperationContext {
            OperationContext {
                source: OperationSource::UserAction,
                safety_preset: preset,
                reason: None,
                track_id: None,
            }
        }

        #[test]
        fn set_and_get_context() {
            let mut tc = TransportController::new();
            assert!(tc.operation_context().is_none());

            let ctx = make_context(SafetyPreset::TrackInteraction);
            tc.set_operation_context(ctx.clone());
            assert_eq!(tc.operation_context(), Some(&ctx));
        }

        #[test]
        fn clear_context() {
            let mut tc = TransportController::new();
            tc.set_operation_context(make_context(SafetyPreset::Permissive));
            tc.clear_operation_context();
            assert!(tc.operation_context().is_none());
        }

        #[test]
        fn context_history() {
            let mut tc = TransportController::new();
            tc.set_operation_context(make_context(SafetyPreset::SliderOperation));
            tc.clear_operation_context();
            tc.set_operation_context(make_context(SafetyPreset::TrackInteraction));
            tc.clear_operation_context();

            assert_eq!(tc.context_history().len(), 2);
            assert_eq!(
                tc.context_history()[0].safety_preset,
                SafetyPreset::SliderOperation
            );
            assert_eq!(
                tc.context_history()[1].safety_preset,
                SafetyPreset::TrackInteraction
            );
        }

        #[test]
        fn context_history_capped() {
            let mut tc = TransportController::new();
            for i in 0..(MAX_CONTEXT_HISTORY + 10) {
                tc.set_operation_context(OperationContext {
                    source: OperationSource::Programmatic,
                    safety_preset: SafetyPreset::Permissive,
                    reason: Some(format!("op-{i}")),
                    track_id: None,
                });
            }
            assert_eq!(tc.context_history().len(), MAX_CONTEXT_HISTORY);
            // First entry should be the 11th operation (0-indexed: 10)
            assert_eq!(
                tc.context_history()[0].reason.as_deref(),
                Some("op-10")
            );
        }
    }

    // -----------------------------------------------------------------------
    // Controller — initial state
    // -----------------------------------------------------------------------

    mod initial_state {
        use super::*;

        #[test]
        fn starts_stopped() {
            let tc = TransportController::new();
            assert_eq!(tc.state(), TransportState::Stopped);
            assert!(tc.is_stopped());
            assert!(!tc.is_playing());
            assert!(!tc.is_paused());
            assert!(!tc.is_active());
        }

        #[test]
        fn no_active_tracks() {
            let tc = TransportController::new();
            assert_eq!(tc.active_track_count(), 0);
            assert!(!tc.has_active_tracks());
        }

        #[test]
        fn default_is_new() {
            let tc = TransportController::default();
            assert_eq!(tc.state(), TransportState::Stopped);
            assert_eq!(tc.active_track_count(), 0);
        }
    }

    // -----------------------------------------------------------------------
    // Controller — track started (auto-start decisions)
    // -----------------------------------------------------------------------

    mod track_started {
        use super::*;

        fn with_preset(preset: SafetyPreset) -> TransportController {
            let mut tc = TransportController::new();
            tc.set_operation_context(OperationContext {
                source: OperationSource::UserAction,
                safety_preset: preset,
                reason: None,
                track_id: None,
            });
            tc
        }

        #[test]
        fn first_track_auto_start_allowed() {
            let mut tc = with_preset(SafetyPreset::TrackInteraction);
            let action = tc.on_track_started("drums");
            assert_eq!(action, TransportAction::Initialize);
            assert_eq!(tc.state(), TransportState::Playing);
            assert_eq!(tc.active_track_count(), 1);
        }

        #[test]
        fn first_track_auto_start_denied() {
            let mut tc = with_preset(SafetyPreset::SliderOperation);
            let action = tc.on_track_started("drums");
            assert_eq!(action, TransportAction::None);
            assert_eq!(tc.state(), TransportState::Stopped);
            assert_eq!(tc.active_track_count(), 1);
        }

        #[test]
        fn first_track_from_paused_always_resumes() {
            let mut tc = with_preset(SafetyPreset::SliderOperation);
            tc.force_state(TransportState::Paused);
            let action = tc.on_track_started("drums");
            assert_eq!(action, TransportAction::Resume);
            assert_eq!(tc.state(), TransportState::Playing);
        }

        #[test]
        fn additional_tracks_no_action() {
            let mut tc = with_preset(SafetyPreset::TrackInteraction);
            tc.on_track_started("drums");
            let action = tc.on_track_started("bass");
            assert_eq!(action, TransportAction::None);
            assert_eq!(tc.active_track_count(), 2);
        }

        #[test]
        fn permissive_default_when_no_context() {
            let mut tc = TransportController::new();
            // No context set — defaults to Permissive (allow auto-start)
            let action = tc.on_track_started("drums");
            assert_eq!(action, TransportAction::Initialize);
            assert_eq!(tc.state(), TransportState::Playing);
        }

        #[test]
        fn track_becomes_active() {
            let mut tc = TransportController::new();
            assert!(!tc.is_track_active("drums"));
            tc.on_track_started("drums");
            assert!(tc.is_track_active("drums"));
        }

        #[test]
        fn first_track_while_already_playing_no_action() {
            let mut tc = with_preset(SafetyPreset::TrackInteraction);
            tc.force_state(TransportState::Playing);
            let action = tc.on_track_started("drums");
            assert_eq!(action, TransportAction::None);
        }
    }

    // -----------------------------------------------------------------------
    // Controller — track stopped (auto-stop decisions)
    // -----------------------------------------------------------------------

    mod track_stopped {
        use super::*;

        fn playing_with_tracks(
            preset: SafetyPreset,
            tracks: &[&str],
        ) -> TransportController {
            let mut tc = TransportController::new();
            tc.set_operation_context(OperationContext {
                source: OperationSource::UserAction,
                safety_preset: preset,
                reason: None,
                track_id: None,
            });
            for t in tracks {
                tc.on_track_started(*t);
            }
            tc
        }

        #[test]
        fn last_track_auto_stop_allowed() {
            let mut tc = playing_with_tracks(SafetyPreset::TrackInteraction, &["drums"]);
            let action = tc.on_track_stopped("drums");
            assert_eq!(action, TransportAction::Shutdown);
            assert_eq!(tc.state(), TransportState::Stopped);
            assert_eq!(tc.active_track_count(), 0);
        }

        #[test]
        fn last_track_auto_stop_denied() {
            let mut tc = playing_with_tracks(SafetyPreset::SliderOperation, &["drums"]);
            // SliderOperation doesn't auto-start, so force to Playing for this test
            tc.force_state(TransportState::Playing);
            let action = tc.on_track_stopped("drums");
            assert_eq!(action, TransportAction::None);
            // Transport stays active so new tracks can sync
            assert_eq!(tc.state(), TransportState::Playing);
        }

        #[test]
        fn non_last_track_no_action() {
            let mut tc = playing_with_tracks(SafetyPreset::TrackInteraction, &["drums", "bass"]);
            let action = tc.on_track_stopped("drums");
            assert_eq!(action, TransportAction::None);
            assert_eq!(tc.active_track_count(), 1);
            assert_eq!(tc.state(), TransportState::Playing);
        }

        #[test]
        fn track_removed_from_active_set() {
            let mut tc = playing_with_tracks(SafetyPreset::Permissive, &["drums", "bass"]);
            assert!(tc.is_track_active("drums"));
            tc.on_track_stopped("drums");
            assert!(!tc.is_track_active("drums"));
            assert!(tc.is_track_active("bass"));
        }

        #[test]
        fn stopping_unknown_track_is_noop() {
            let mut tc = playing_with_tracks(SafetyPreset::Permissive, &["drums"]);
            let action = tc.on_track_stopped("nonexistent");
            // drums is still active, so not last-track case
            assert_eq!(action, TransportAction::None);
            assert_eq!(tc.active_track_count(), 1);
        }

        #[test]
        fn manual_control_allows_auto_stop() {
            let mut tc = TransportController::new();
            tc.set_operation_context(OperationContext {
                source: OperationSource::TransportButton,
                safety_preset: SafetyPreset::ManualControl,
                reason: None,
                track_id: None,
            });
            // Manually set playing state and add track
            tc.force_state(TransportState::Playing);
            tc.on_track_started("drums");
            let action = tc.on_track_stopped("drums");
            assert_eq!(action, TransportAction::Shutdown);
            assert_eq!(tc.state(), TransportState::Stopped);
        }
    }

    // -----------------------------------------------------------------------
    // Controller — manual controls
    // -----------------------------------------------------------------------

    mod manual_controls {
        use super::*;

        #[test]
        fn pause_from_playing() {
            let mut tc = TransportController::new();
            tc.force_state(TransportState::Playing);
            let action = tc.pause();
            assert_eq!(action, TransportAction::Pause);
            assert_eq!(tc.state(), TransportState::Paused);
        }

        #[test]
        fn pause_from_stopped_noop() {
            let mut tc = TransportController::new();
            let action = tc.pause();
            assert_eq!(action, TransportAction::None);
            assert_eq!(tc.state(), TransportState::Stopped);
        }

        #[test]
        fn pause_from_paused_noop() {
            let mut tc = TransportController::new();
            tc.force_state(TransportState::Paused);
            let action = tc.pause();
            assert_eq!(action, TransportAction::None);
            assert_eq!(tc.state(), TransportState::Paused);
        }

        #[test]
        fn resume_from_paused() {
            let mut tc = TransportController::new();
            tc.force_state(TransportState::Paused);
            let action = tc.resume();
            assert_eq!(action, TransportAction::Resume);
            assert_eq!(tc.state(), TransportState::Playing);
        }

        #[test]
        fn resume_from_stopped_noop() {
            let mut tc = TransportController::new();
            let action = tc.resume();
            assert_eq!(action, TransportAction::None);
            assert_eq!(tc.state(), TransportState::Stopped);
        }

        #[test]
        fn resume_from_playing_noop() {
            let mut tc = TransportController::new();
            tc.force_state(TransportState::Playing);
            let action = tc.resume();
            assert_eq!(action, TransportAction::None);
            assert_eq!(tc.state(), TransportState::Playing);
        }

        #[test]
        fn stop_from_playing() {
            let mut tc = TransportController::new();
            tc.on_track_started("drums");
            assert_eq!(tc.state(), TransportState::Playing);
            let action = tc.stop();
            assert_eq!(action, TransportAction::Shutdown);
            assert_eq!(tc.state(), TransportState::Stopped);
            assert_eq!(tc.active_track_count(), 0);
        }

        #[test]
        fn stop_from_paused() {
            let mut tc = TransportController::new();
            tc.force_state(TransportState::Paused);
            tc.active_tracks.insert("drums".to_string());
            let action = tc.stop();
            assert_eq!(action, TransportAction::Shutdown);
            assert_eq!(tc.state(), TransportState::Stopped);
            assert_eq!(tc.active_track_count(), 0);
        }

        #[test]
        fn stop_from_stopped_noop() {
            let mut tc = TransportController::new();
            let action = tc.stop();
            assert_eq!(action, TransportAction::None);
        }
    }

    // -----------------------------------------------------------------------
    // Controller — reset and force state
    // -----------------------------------------------------------------------

    mod reset_and_force {
        use super::*;

        #[test]
        fn reset_clears_everything() {
            let mut tc = TransportController::new();
            tc.on_track_started("drums");
            tc.set_operation_context(OperationContext {
                source: OperationSource::UserAction,
                safety_preset: SafetyPreset::TrackInteraction,
                reason: Some("test".into()),
                track_id: Some("drums".into()),
            });

            tc.reset();

            assert_eq!(tc.state(), TransportState::Stopped);
            assert_eq!(tc.active_track_count(), 0);
            assert!(tc.operation_context().is_none());
            assert!(tc.context_history().is_empty());
        }

        #[test]
        fn force_state() {
            let mut tc = TransportController::new();
            tc.force_state(TransportState::Playing);
            assert_eq!(tc.state(), TransportState::Playing);
            tc.force_state(TransportState::Paused);
            assert_eq!(tc.state(), TransportState::Paused);
            tc.force_state(TransportState::Stopped);
            assert_eq!(tc.state(), TransportState::Stopped);
        }
    }

    // -----------------------------------------------------------------------
    // Controller — full workflow scenarios
    // -----------------------------------------------------------------------

    mod workflows {
        use super::*;

        #[test]
        fn typical_track_interaction_flow() {
            let mut tc = TransportController::new();

            // User clicks track → set context
            tc.set_operation_context(OperationContext {
                source: OperationSource::UserAction,
                safety_preset: SafetyPreset::TrackInteraction,
                reason: Some("User armed drums".into()),
                track_id: Some("drums".into()),
            });

            // First track starts → Initialize transport
            assert_eq!(tc.on_track_started("drums"), TransportAction::Initialize);
            assert_eq!(tc.state(), TransportState::Playing);
            tc.clear_operation_context();

            // User clicks second track
            tc.set_operation_context(OperationContext {
                source: OperationSource::UserAction,
                safety_preset: SafetyPreset::TrackInteraction,
                reason: Some("User armed bass".into()),
                track_id: Some("bass".into()),
            });
            assert_eq!(tc.on_track_started("bass"), TransportAction::None);
            tc.clear_operation_context();

            // User disarms first track
            tc.set_operation_context(OperationContext {
                source: OperationSource::UserAction,
                safety_preset: SafetyPreset::TrackInteraction,
                reason: Some("User disarmed drums".into()),
                track_id: Some("drums".into()),
            });
            assert_eq!(tc.on_track_stopped("drums"), TransportAction::None);
            assert_eq!(tc.state(), TransportState::Playing);
            tc.clear_operation_context();

            // User disarms last track → Shutdown
            tc.set_operation_context(OperationContext {
                source: OperationSource::UserAction,
                safety_preset: SafetyPreset::TrackInteraction,
                reason: Some("User disarmed bass".into()),
                track_id: Some("bass".into()),
            });
            assert_eq!(tc.on_track_stopped("bass"), TransportAction::Shutdown);
            assert_eq!(tc.state(), TransportState::Stopped);
            tc.clear_operation_context();
        }

        #[test]
        fn slider_operation_preserves_transport() {
            let mut tc = TransportController::new();

            // Start playing via direct interaction
            tc.set_operation_context(OperationContext {
                source: OperationSource::UserAction,
                safety_preset: SafetyPreset::TrackInteraction,
                reason: None,
                track_id: None,
            });
            tc.on_track_started("drums");
            tc.on_track_started("bass");
            tc.clear_operation_context();
            assert_eq!(tc.state(), TransportState::Playing);

            // Slider removes a track — uses SliderOperation preset
            tc.set_operation_context(OperationContext {
                source: OperationSource::LiveMixing,
                safety_preset: SafetyPreset::SliderOperation,
                reason: Some("Complexity slider moved".into()),
                track_id: Some("drums".into()),
            });
            tc.on_track_stopped("drums");
            // Still playing — only one track removed
            assert_eq!(tc.state(), TransportState::Playing);

            // Slider removes last track
            tc.on_track_stopped("bass");
            // STILL playing — SliderOperation doesn't auto-stop
            assert_eq!(tc.state(), TransportState::Playing);
            tc.clear_operation_context();
        }

        #[test]
        fn pause_resume_cycle() {
            let mut tc = TransportController::new();
            tc.on_track_started("drums");
            assert_eq!(tc.state(), TransportState::Playing);

            // Pause
            assert_eq!(tc.pause(), TransportAction::Pause);
            assert_eq!(tc.state(), TransportState::Paused);

            // Resume
            assert_eq!(tc.resume(), TransportAction::Resume);
            assert_eq!(tc.state(), TransportState::Playing);

            // Stop
            assert_eq!(tc.stop(), TransportAction::Shutdown);
            assert_eq!(tc.state(), TransportState::Stopped);
            assert_eq!(tc.active_track_count(), 0);
        }

        #[test]
        fn programmatic_operation_no_side_effects() {
            let mut tc = TransportController::new();

            tc.set_operation_context(OperationContext {
                source: OperationSource::Programmatic,
                safety_preset: SafetyPreset::ProgrammaticOperation,
                reason: Some("Loading song".into()),
                track_id: None,
            });

            // Tracks start during loading — no auto-start
            assert_eq!(tc.on_track_started("drums"), TransportAction::None);
            assert_eq!(tc.state(), TransportState::Stopped);

            // Tracks stop during loading — no auto-stop
            assert_eq!(tc.on_track_stopped("drums"), TransportAction::None);
            assert_eq!(tc.state(), TransportState::Stopped);

            tc.clear_operation_context();
        }
    }

    // -----------------------------------------------------------------------
    // Error display
    // -----------------------------------------------------------------------

    mod errors {
        use super::*;

        #[test]
        fn invalid_transition_display() {
            let err = TransportError::InvalidTransition {
                from: TransportState::Stopped,
                action: "pause",
            };
            assert_eq!(
                err.to_string(),
                "Cannot pause from transport state 'stopped'"
            );
        }

        #[test]
        fn track_not_active_display() {
            let err = TransportError::TrackNotActive("drums".into());
            assert_eq!(
                err.to_string(),
                "Track 'drums' is not in the active set"
            );
        }

        #[test]
        fn error_is_std_error() {
            let err: Box<dyn std::error::Error> = Box::new(TransportError::TrackNotActive("x".into()));
            assert!(!err.to_string().is_empty());
        }
    }

    // -----------------------------------------------------------------------
    // OperationSource display
    // -----------------------------------------------------------------------

    mod operation_source {
        use super::*;

        #[test]
        fn display() {
            assert_eq!(OperationSource::LiveMixing.to_string(), "live-mixing");
            assert_eq!(OperationSource::UserAction.to_string(), "user-action");
            assert_eq!(OperationSource::TransportButton.to_string(), "transport-button");
            assert_eq!(OperationSource::Programmatic.to_string(), "programmatic");
            assert_eq!(OperationSource::BatchOperation.to_string(), "batch-operation");
        }
    }
}
