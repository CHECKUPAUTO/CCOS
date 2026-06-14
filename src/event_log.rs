use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventLog {
    pub session_id: String,
    pub events: Vec<TraceEvent>,
    pub snapshot_index: HashMap<u64, usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceEvent {
    pub id: String,
    pub timestamp: u64,
    pub event_type: EventType,
    pub payload: EventPayload,
    pub sequence_number: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EventType {
    LlmCall,
    LlmResponse,
    Parsing,
    GraphMutation,
    FailureDetection,
    FailurePropagation,
    GuardCheck,
    CycleStart,
    CycleEnd,
    ReplayStart,
    ReplayEnd,
    Snapshot,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventPayload {
    LlmCallRequest {
        model: String,
        prompt_hash: String,
        input_tokens: usize,
    },
    LlmCallResponse {
        model: String,
        response_hash: String,
        output_tokens: usize,
        latency_ms: u64,
        guard_passed: bool,
        reliability_score: f64,
    },
    Parsing {
        file_path: String,
        file_hash: String,
        modules_found: usize,
        uses_found: usize,
        symbols_found: usize,
    },
    GraphMutation {
        node_id: String,
        operation: String,
        nodes_before: usize,
        nodes_after: usize,
        edges_before: usize,
        edges_after: usize,
    },
    FailureDetection {
        node_id: String,
        failure_type: String,
        severity: f64,
    },
    FailurePropagation {
        origin_node_id: String,
        affected_nodes: Vec<String>,
        depth: u32,
    },
    GuardCheck {
        input_hash: String,
        passed: bool,
        score: f64,
        warnings: Vec<String>,
    },
    CycleEvent {
        cycle_number: u64,
        action: String,
    },
    ReplayEvent {
        original_event_id: String,
        replayed_at: u64,
    },
    Snapshot {
        nodes_count: usize,
        edges_count: usize,
        total_events: usize,
    },
    Custom {
        key: String,
        value: String,
    },
}

impl EventLog {
    pub fn new(session_id: String) -> Self {
        Self {
            session_id,
            events: Vec::new(),
            snapshot_index: HashMap::new(),
        }
    }

    pub fn append(&mut self, event_type: EventType, payload: EventPayload) -> String {
        let id = Uuid::new_v4().to_string();
        let seq = self.events.len() as u64;
        let event = TraceEvent {
            id: id.clone(),
            timestamp: Self::now_millis(),
            event_type,
            payload,
            sequence_number: seq,
        };
        self.events.push(event);
        id
    }

    pub fn take_snapshot(&mut self) -> usize {
        let idx = self.events.len();
        self.snapshot_index.insert(self.events.len() as u64, idx);
        idx
    }

    pub fn replay_events(
        &self,
        from_sequence: u64,
        to_sequence: Option<u64>,
    ) -> Vec<&TraceEvent> {
        let end = to_sequence.unwrap_or(self.events.len() as u64);
        self.events
            .iter()
            .filter(|e| e.sequence_number >= from_sequence && e.sequence_number < end)
            .collect()
    }

    pub fn get_event_by_id(&self, id: &str) -> Option<&TraceEvent> {
        self.events.iter().find(|e| e.id == id)
    }

    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    pub fn events_by_type(&self, event_type: EventType) -> Vec<&TraceEvent> {
        self.events
            .iter()
            .filter(|e| e.event_type == event_type)
            .collect()
    }

    pub fn last_event(&self) -> Option<&TraceEvent> {
        self.events.last()
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    pub fn clear(&mut self) {
        self.events.clear();
        self.snapshot_index.clear();
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }

    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    fn now_millis() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }

    pub fn replay_deterministic(
        &self,
        replay_handler: &mut dyn ReplayHandler,
    ) -> Result<usize, String> {
        let mut processed = 0;
        for event in &self.events {
            replay_handler.handle_event(event)?;
            processed += 1;
        }
        Ok(processed)
    }
}

pub trait ReplayHandler {
    fn handle_event(&mut self, event: &TraceEvent) -> Result<(), String>;
}

#[derive(Debug, Clone)]
pub struct EventReplayer {
    pub statistics: ReplayStatistics,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReplayStatistics {
    pub total_events: usize,
    pub llm_calls: usize,
    pub parsing_events: usize,
    pub graph_mutations: usize,
    pub failures: usize,
    pub guard_checks: usize,
    pub cycles: usize,
}

impl ReplayHandler for EventReplayer {
    fn handle_event(&mut self, event: &TraceEvent) -> Result<(), String> {
        self.statistics.total_events += 1;
        match &event.payload {
            EventPayload::LlmCallRequest { .. } => self.statistics.llm_calls += 1,
            EventPayload::LlmCallResponse { .. } => self.statistics.llm_calls += 1,
            EventPayload::Parsing { .. } => self.statistics.parsing_events += 1,
            EventPayload::GraphMutation { .. } => self.statistics.graph_mutations += 1,
            EventPayload::FailureDetection { .. } => self.statistics.failures += 1,
            EventPayload::FailurePropagation { .. } => self.statistics.failures += 1,
            EventPayload::GuardCheck { .. } => self.statistics.guard_checks += 1,
            EventPayload::CycleEvent { .. } => self.statistics.cycles += 1,
            _ => {}
        }
        Ok(())
    }
}

impl EventReplayer {
    pub fn new() -> Self {
        Self {
            statistics: ReplayStatistics::default(),
        }
    }
}

impl Default for EventReplayer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_append_and_retrieve() {
        let mut log = EventLog::new("test-session".into());
        let id = log.append(
            EventType::CycleStart,
            EventPayload::CycleEvent {
                cycle_number: 1,
                action: "start".into(),
            },
        );
        assert_eq!(log.event_count(), 1);
        assert!(log.get_event_by_id(&id).is_some());
    }

    #[test]
    fn test_replay_events_range() {
        let mut log = EventLog::new("test".into());
        for i in 0..5 {
            log.append(
                EventType::CycleStart,
                EventPayload::CycleEvent {
                    cycle_number: i,
                    action: format!("cycle {}", i),
                },
            );
        }
        let replayed = log.replay_events(2, Some(4));
        assert_eq!(replayed.len(), 2);
    }

    #[test]
    fn test_events_by_type() {
        let mut log = EventLog::new("test".into());
        log.append(
            EventType::Parsing,
            EventPayload::Parsing {
                file_path: "test.rs".into(),
                file_hash: "abc".into(),
                modules_found: 1,
                uses_found: 2,
                symbols_found: 3,
            },
        );
        log.append(
            EventType::LlmCall,
            EventPayload::LlmCallRequest {
                model: "test".into(),
                prompt_hash: "def".into(),
                input_tokens: 100,
            },
        );
        let parsing_events = log.events_by_type(EventType::Parsing);
        assert_eq!(parsing_events.len(), 1);
    }

    #[test]
    fn test_deterministic_replay() {
        let mut log = EventLog::new("test".into());
        log.append(
            EventType::Parsing,
            EventPayload::Parsing {
                file_path: "a.rs".into(),
                file_hash: "h1".into(),
                modules_found: 1,
                uses_found: 0,
                symbols_found: 0,
            },
        );
        let mut replayer = EventReplayer::new();
        let result = log.replay_deterministic(&mut replayer);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 1);
        assert_eq!(replayer.statistics.parsing_events, 1);
    }

    #[test]
    fn test_serialization_roundtrip() {
        let mut log = EventLog::new("test".into());
        log.append(
            EventType::GuardCheck,
            EventPayload::GuardCheck {
                input_hash: "h".into(),
                passed: true,
                score: 0.9,
                warnings: vec![],
            },
        );
        let json = log.to_json();
        let restored: EventLog = EventLog::from_json(&json).unwrap();
        assert_eq!(restored.session_id, "test");
        assert_eq!(restored.event_count(), 1);
    }
}
