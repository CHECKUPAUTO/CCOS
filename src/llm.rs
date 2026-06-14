use crate::guard::{GuardConfig, GuardLayer};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone)]
pub struct LlmConfig {
    pub endpoint: String,
    pub model: String,
    pub timeout_secs: u64,
    pub max_retries: u32,
    pub guard_config: GuardConfig,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            endpoint: "http://localhost:11434".to_string(),
            model: "codellama".to_string(),
            timeout_secs: 30,
            max_retries: 2,
            guard_config: GuardConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmRequest {
    pub model: String,
    pub prompt: String,
    pub system: Option<String>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<RequestOptions>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_predict: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmResponse {
    pub model: String,
    pub created_at: String,
    pub response: String,
    pub done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<Vec<i64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub load_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_count: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_count: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_duration: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatedResponse {
    pub raw_response: String,
    pub sanitized_output: String,
    pub guard_passed: bool,
    pub reliability_score: f64,
    pub guard_warnings: Vec<String>,
    pub model: String,
    pub prompt_hash: String,
    pub response_hash: String,
    pub latency_ms: u64,
    pub is_fallback: bool,
}

#[derive(Debug, Clone)]
pub struct LlmClient {
    config: LlmConfig,
    client: Client,
    guard: GuardLayer,
}

impl LlmClient {
    pub fn new(config: LlmConfig) -> Self {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_secs))
            .build()
            .expect("Failed to build HTTP client");

        let guard = GuardLayer::new(config.guard_config.clone());

        Self {
            config,
            client,
            guard,
        }
    }

    pub async fn query(
        &self,
        prompt: &str,
        system: Option<&str>,
    ) -> ValidatedResponse {
        let prompt_hash = compute_hash(prompt);
        let start = std::time::Instant::now();

        let request = LlmRequest {
            model: self.config.model.clone(),
            prompt: prompt.to_string(),
            system: system.map(|s| s.to_string()),
            stream: false,
            options: Some(RequestOptions {
                temperature: Some(0.3),
                num_predict: Some(1024),
                top_k: Some(40),
                top_p: Some(0.9),
            }),
        };

        let endpoint = format!("{}/api/generate", self.config.endpoint);

        let mut last_error: Option<String> = None;

        for attempt in 0..=self.config.max_retries {
            if attempt > 0 {
                tokio::time::sleep(std::time::Duration::from_millis(200 * attempt as u64)).await;
            }

            match self.client.post(&endpoint).json(&request).send().await {
                Ok(resp) => {
                    match resp.json::<LlmResponse>().await {
                        Ok(llm_resp) => {
                            let latency = start.elapsed().as_millis() as u64;
                            let guard_result = self.guard.validate_and_sanitize(&llm_resp.response);

                            let response_hash = compute_hash(&guard_result.sanitized_output);

                            return ValidatedResponse {
                                raw_response: llm_resp.response,
                                sanitized_output: guard_result.sanitized_output,
                                guard_passed: guard_result.passed,
                                reliability_score: guard_result.reliability_score,
                                guard_warnings: guard_result.warnings,
                                model: self.config.model.clone(),
                                prompt_hash,
                                response_hash,
                                latency_ms: latency,
                                is_fallback: false,
                            };
                        }
                        Err(e) => {
                            last_error = Some(format!(
                                "Failed to parse response JSON: {}",
                                e
                            ));
                        }
                    }
                }
                Err(e) => {
                    last_error = Some(format!("HTTP request failed: {}", e));
                }
            }
        }

        // All retries exhausted — return fallback
        let latency = start.elapsed().as_millis() as u64;
        let guard_result = self.guard.validate_and_sanitize("");
        let fallback = GuardLayer::fallback_response();
        let response_hash = compute_hash(&fallback);

        ValidatedResponse {
            raw_response: last_error.unwrap_or_else(|| "unknown error".into()),
            sanitized_output: fallback.clone(),
            guard_passed: false,
            reliability_score: 0.0,
            guard_warnings: guard_result.warnings,
            model: self.config.model.clone(),
            prompt_hash,
            response_hash,
            latency_ms: latency,
            is_fallback: true,
        }
    }

    pub async fn query_deterministic(
        &self,
        prompt: &str,
        _system: Option<&str>,
    ) -> ValidatedResponse {
        // Deterministic fallback for replay scenarios (no network call)
        let prompt_hash = compute_hash(prompt);
        let deterministic_output = format!(
            r#"{{
                "status": "deterministic_replay",
                "analysis": {{
                    "summary": "Deterministic replay response for prompt hash: {}",
                    "dependencies": [],
                    "confidence": 1.0
                }}
            }}"#,
            prompt_hash
        );

        let response_hash = compute_hash(&deterministic_output);

        ValidatedResponse {
            raw_response: deterministic_output.clone(),
            sanitized_output: deterministic_output,
            guard_passed: true,
            reliability_score: 1.0,
            guard_warnings: Vec::new(),
            model: self.config.model.clone(),
            prompt_hash,
            response_hash,
            latency_ms: 0,
            is_fallback: false,
        }
    }

    pub fn guard_layer(&self) -> &GuardLayer {
        &self.guard
    }
}

fn compute_hash(input: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llm_request_serialization() {
        let req = LlmRequest {
            model: "test".into(),
            prompt: "hello".into(),
            system: Some("be helpful".into()),
            stream: false,
            options: Some(RequestOptions {
                temperature: Some(0.5),
                num_predict: Some(100),
                top_k: Some(40),
                top_p: Some(0.9),
            }),
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("hello"));
        assert!(json.contains("test"));
    }

    #[test]
    fn test_llm_response_deserialization() {
        let json = r#"{
            "model": "test",
            "created_at": "2024-01-01T00:00:00Z",
            "response": "Hello from LLM",
            "done": true
        }"#;
        let resp: LlmResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.model, "test");
        assert_eq!(resp.response, "Hello from LLM");
    }

    #[tokio::test]
    async fn test_deterministic_query() {
        let config = LlmConfig::default();
        let client = LlmClient::new(config);
        let result = client.query_deterministic("test prompt", None).await;
        assert!(result.guard_passed);
        assert_eq!(result.reliability_score, 1.0);
        assert!(!result.is_fallback);
    }

    #[test]
    fn test_hash_consistency() {
        let h1 = compute_hash("hello");
        let h2 = compute_hash("hello");
        assert_eq!(h1, h2);
    }
}
