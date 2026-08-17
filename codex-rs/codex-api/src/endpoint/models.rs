use crate::auth::SharedAuthProvider;
use crate::endpoint::session::EndpointSession;
use crate::error::ApiError;
use crate::provider::Provider;
use codex_client::HttpTransport;
use codex_client::RequestTelemetry;
use codex_protocol::openai_models::InputModality;
use codex_protocol::openai_models::ModelInfo;
use codex_protocol::openai_models::ModelsResponse;
use codex_protocol::openai_models::ReasoningEffort;
use codex_protocol::openai_models::ReasoningEffortPreset;
use http::HeaderMap;
use http::Method;
use http::header::ETAG;
use serde::Deserialize;
use std::collections::HashSet;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenAiCompatibleModel {
    pub id: String,
    pub display_name: Option<String>,
    pub description: Option<String>,
    pub default_reasoning_effort: Option<ReasoningEffort>,
    pub supported_reasoning_efforts: Option<Vec<ReasoningEffortPreset>>,
    pub context_window: Option<i64>,
    pub max_context_window: Option<i64>,
    pub input_modalities: Option<Vec<InputModality>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelsList {
    Codex(Vec<ModelInfo>),
    OpenAiCompatible(Vec<OpenAiCompatibleModel>),
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum WireModelsResponse {
    Codex(ModelsResponse),
    OpenAiCompatible(OpenAiCompatibleModelsResponse),
}

#[derive(Debug, Deserialize)]
struct OpenAiCompatibleModelsResponse {
    data: Vec<WireOpenAiCompatibleModel>,
}

#[derive(Debug, Deserialize)]
struct WireOpenAiCompatibleModel {
    id: String,
    #[serde(default, alias = "display_name")]
    name: Option<String>,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    default_reasoning_effort: Option<ReasoningEffort>,
    #[serde(default)]
    default_reasoning_level: Option<ReasoningEffort>,
    #[serde(default)]
    supported_reasoning_efforts: Option<Vec<WireReasoningEffort>>,
    #[serde(default)]
    supported_reasoning_levels: Option<Vec<WireReasoningEffort>>,
    #[serde(default)]
    reasoning_efforts: Option<Vec<WireReasoningEffort>>,
    #[serde(default)]
    context_window: Option<i64>,
    #[serde(default)]
    max_context_window: Option<i64>,
    #[serde(default)]
    input_modalities: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum WireReasoningEffort {
    Name(ReasoningEffort),
    Metadata {
        #[serde(alias = "reasoning_effort")]
        effort: ReasoningEffort,
        #[serde(default)]
        description: Option<String>,
    },
}

impl WireReasoningEffort {
    fn into_preset(self) -> ReasoningEffortPreset {
        match self {
            Self::Name(effort) => ReasoningEffortPreset {
                description: reasoning_effort_description(&effort),
                effort,
            },
            Self::Metadata {
                effort,
                description,
            } => ReasoningEffortPreset {
                description: description.unwrap_or_else(|| reasoning_effort_description(&effort)),
                effort,
            },
        }
    }
}

impl From<WireOpenAiCompatibleModel> for OpenAiCompatibleModel {
    fn from(model: WireOpenAiCompatibleModel) -> Self {
        let efforts = model
            .supported_reasoning_efforts
            .or(model.supported_reasoning_levels)
            .or(model.reasoning_efforts)
            .map(|efforts| {
                let mut seen = HashSet::new();
                efforts
                    .into_iter()
                    .map(WireReasoningEffort::into_preset)
                    .filter(|preset| seen.insert(preset.effort.clone()))
                    .collect()
            });
        let input_modalities = model.input_modalities.map(|modalities| {
            modalities
                .into_iter()
                .filter_map(|modality| match modality.to_ascii_lowercase().as_str() {
                    "text" => Some(InputModality::Text),
                    "image" | "vision" => Some(InputModality::Image),
                    "audio" => Some(InputModality::Audio),
                    _ => None,
                })
                .collect()
        });
        Self {
            id: model.id,
            display_name: model.name,
            description: model.description,
            default_reasoning_effort: model
                .default_reasoning_effort
                .or(model.default_reasoning_level),
            supported_reasoning_efforts: efforts,
            context_window: model.context_window,
            max_context_window: model.max_context_window,
            input_modalities,
        }
    }
}

fn reasoning_effort_description(effort: &ReasoningEffort) -> String {
    match effort {
        ReasoningEffort::None => "No reasoning".to_string(),
        ReasoningEffort::Minimal => "Minimal reasoning".to_string(),
        ReasoningEffort::Low => "Low reasoning".to_string(),
        ReasoningEffort::Medium => "Medium reasoning".to_string(),
        ReasoningEffort::High => "High reasoning".to_string(),
        ReasoningEffort::XHigh => "Extra high reasoning".to_string(),
        ReasoningEffort::Max => "Maximum reasoning".to_string(),
        ReasoningEffort::Ultra => "Ultra reasoning".to_string(),
        ReasoningEffort::Custom(value) => format!("{value} reasoning"),
    }
}

pub struct ModelsClient<T: HttpTransport> {
    session: EndpointSession<T>,
}

impl<T: HttpTransport> ModelsClient<T> {
    pub fn new(transport: T, provider: Provider, auth: SharedAuthProvider) -> Self {
        Self {
            session: EndpointSession::new(transport, provider, auth),
        }
    }

    pub fn with_telemetry(self, request: Option<Arc<dyn RequestTelemetry>>) -> Self {
        Self {
            session: self.session.with_request_telemetry(request),
        }
    }

    fn path() -> &'static str {
        "models"
    }

    fn append_client_version_query(req: &mut codex_client::Request, client_version: &str) {
        let separator = if req.url.contains('?') { '&' } else { '?' };
        req.url = format!("{}{}client_version={client_version}", req.url, separator);
    }

    pub fn request_url(provider: &Provider, client_version: &str) -> String {
        let mut request = provider.build_request(Method::GET, Self::path());
        Self::append_client_version_query(&mut request, client_version);
        request.url
    }

    pub async fn list_models(
        &self,
        request_url: String,
        extra_headers: HeaderMap,
    ) -> Result<(ModelsList, Option<String>), ApiError> {
        let resp = self
            .session
            .execute_with(
                Method::GET,
                Self::path(),
                extra_headers,
                /*body*/ None,
                move |req| {
                    req.url.clone_from(&request_url);
                },
            )
            .await?;

        let header_etag = resp
            .headers
            .get(ETAG)
            .and_then(|value| value.to_str().ok())
            .map(ToString::to_string);

        let models = serde_json::from_slice::<WireModelsResponse>(&resp.body).map_err(|e| {
            ApiError::Stream(format!(
                "failed to decode models response: {e}; body: {}",
                String::from_utf8_lossy(&resp.body)
            ))
        })?;
        let models = match models {
            WireModelsResponse::Codex(ModelsResponse { models }) => ModelsList::Codex(models),
            WireModelsResponse::OpenAiCompatible(response) => {
                ModelsList::OpenAiCompatible(response.data.into_iter().map(Into::into).collect())
            }
        };

        Ok((models, header_etag))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::AuthProvider;
    use crate::provider::RetryConfig;
    use codex_client::Request;
    use codex_client::Response;
    use codex_client::StreamResponse;
    use codex_client::TransportError;
    use http::HeaderMap;
    use http::Method;
    use http::StatusCode;
    use pretty_assertions::assert_eq;
    use serde_json::json;
    use std::sync::Arc;
    use std::sync::Mutex;
    use std::time::Duration;

    #[derive(Clone)]
    struct CapturingTransport {
        last_request: Arc<Mutex<Option<Request>>>,
        body: Arc<Vec<u8>>,
        etag: Option<String>,
    }

    impl Default for CapturingTransport {
        fn default() -> Self {
            Self {
                last_request: Arc::new(Mutex::new(None)),
                body: Arc::new(serde_json::to_vec(&ModelsResponse { models: Vec::new() }).unwrap()),
                etag: None,
            }
        }
    }

    impl HttpTransport for CapturingTransport {
        async fn execute(&self, req: Request) -> Result<Response, TransportError> {
            *self.last_request.lock().unwrap() = Some(req);
            let mut headers = HeaderMap::new();
            if let Some(etag) = &self.etag {
                headers.insert(ETAG, etag.parse().unwrap());
            }
            Ok(Response {
                status: StatusCode::OK,
                headers,
                body: self.body.as_ref().clone().into(),
            })
        }

        async fn stream(&self, _req: Request) -> Result<StreamResponse, TransportError> {
            Err(TransportError::Build("stream should not run".to_string()))
        }
    }

    #[derive(Clone, Default)]
    struct DummyAuth;

    impl AuthProvider for DummyAuth {
        fn add_auth_headers(&self, _headers: &mut HeaderMap) {}
    }

    fn provider(base_url: &str) -> Provider {
        Provider {
            name: "test".to_string(),
            base_url: base_url.to_string(),
            query_params: None,
            headers: HeaderMap::new(),
            retry: RetryConfig {
                max_attempts: 1,
                base_delay: Duration::from_millis(1),
                retry_429: false,
                retry_5xx: true,
                retry_transport: true,
            },
            stream_idle_timeout: Duration::from_secs(1),
        }
    }

    #[tokio::test]
    async fn appends_client_version_query() {
        let response = ModelsResponse { models: Vec::new() };

        let transport = CapturingTransport {
            last_request: Arc::new(Mutex::new(None)),
            body: Arc::new(serde_json::to_vec(&response).unwrap()),
            etag: None,
        };

        let provider = provider("https://example.com/api/codex");
        let request_url = ModelsClient::<CapturingTransport>::request_url(&provider, "0.99.0");
        let client = ModelsClient::new(transport.clone(), provider, Arc::new(DummyAuth));

        let (models, _) = client
            .list_models(request_url, HeaderMap::new())
            .await
            .expect("request should succeed");

        assert_eq!(models, ModelsList::Codex(Vec::new()));

        let url = transport
            .last_request
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .url
            .clone();
        assert_eq!(
            url,
            "https://example.com/api/codex/models?client_version=0.99.0"
        );
    }

    #[tokio::test]
    async fn parses_models_response() {
        let response = ModelsResponse {
            models: vec![
                serde_json::from_value(json!({
                    "slug": "gpt-test",
                    "display_name": "gpt-test",
                    "description": "desc",
                    "default_reasoning_level": "medium",
                    "supported_reasoning_levels": [{"effort": "low", "description": "low"}, {"effort": "medium", "description": "medium"}, {"effort": "high", "description": "high"}],
                    "shell_type": "shell_command",
                    "visibility": "list",
                    "minimal_client_version": [0, 99, 0],
                    "supported_in_api": true,
                    "priority": 1,
                    "upgrade": null,
                    "base_instructions": "base instructions",
                    "support_verbosity": false,
                    "default_verbosity": null,
                    "apply_patch_tool_type": null,
                    "truncation_policy": {"mode": "bytes", "limit": 10_000},
                    "supports_parallel_tool_calls": false,
                    "supports_image_detail_original": false,
                    "context_window": 272_000,
                    "experimental_supported_tools": [],
                }))
                .unwrap(),
            ],
        };

        let transport = CapturingTransport {
            last_request: Arc::new(Mutex::new(None)),
            body: Arc::new(serde_json::to_vec(&response).unwrap()),
            etag: None,
        };

        let provider = provider("https://example.com/api/codex");
        let request_url = ModelsClient::<CapturingTransport>::request_url(&provider, "0.99.0");
        let client = ModelsClient::new(transport, provider, Arc::new(DummyAuth));

        let (models, _) = client
            .list_models(request_url, HeaderMap::new())
            .await
            .expect("request should succeed");

        let ModelsList::Codex(models) = models else {
            panic!("expected Codex models response");
        };
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].slug, "gpt-test");
        assert!(models[0].supported_in_api);
        assert_eq!(models[0].priority, 1);
    }

    #[tokio::test]
    async fn parses_openai_compatible_models_response_with_optional_capabilities() {
        let transport = CapturingTransport {
            last_request: Arc::new(Mutex::new(None)),
            body: Arc::new(
                serde_json::to_vec(&json!({
                    "object": "list",
                    "data": [
                        {
                            "id": "provider-reasoning-model",
                            "display_name": "Provider Reasoning Model",
                            "description": "remote metadata",
                            "default_reasoning_effort": "max",
                            "supported_reasoning_efforts": [
                                "high",
                                {"effort": "max", "description": "Provider maximum"},
                                "future-effort"
                            ],
                            "context_window": 131072,
                            "max_context_window": 262144,
                            "input_modalities": ["text", "vision", "unsupported"]
                        },
                        {"id": "provider-basic-model"}
                    ]
                }))
                .unwrap(),
            ),
            etag: None,
        };
        let provider = provider("https://example.com/v1");
        let request_url = provider.build_request(Method::GET, "models").url;
        let client = ModelsClient::new(transport, provider, Arc::new(DummyAuth));

        let (models, _) = client
            .list_models(request_url, HeaderMap::new())
            .await
            .expect("request should succeed");

        let ModelsList::OpenAiCompatible(models) = models else {
            panic!("expected OpenAI-compatible models response");
        };
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].id, "provider-reasoning-model");
        assert_eq!(
            models[0].default_reasoning_effort,
            Some(ReasoningEffort::Max)
        );
        assert_eq!(
            models[0]
                .supported_reasoning_efforts
                .as_ref()
                .expect("reasoning efforts")
                .iter()
                .map(|preset| preset.effort.clone())
                .collect::<Vec<_>>(),
            vec![
                ReasoningEffort::High,
                ReasoningEffort::Max,
                ReasoningEffort::Custom("future-effort".to_string()),
            ]
        );
        assert_eq!(models[0].context_window, Some(131_072));
        assert_eq!(models[0].max_context_window, Some(262_144));
        assert_eq!(
            models[0].input_modalities,
            Some(vec![InputModality::Text, InputModality::Image])
        );
        assert_eq!(models[1].id, "provider-basic-model");
        assert_eq!(models[1].supported_reasoning_efforts, None);
    }

    #[tokio::test]
    async fn list_models_includes_etag() {
        let response = ModelsResponse { models: Vec::new() };

        let transport = CapturingTransport {
            last_request: Arc::new(Mutex::new(None)),
            body: Arc::new(serde_json::to_vec(&response).unwrap()),
            etag: Some("\"abc\"".to_string()),
        };

        let provider = provider("https://example.com/api/codex");
        let request_url = ModelsClient::<CapturingTransport>::request_url(&provider, "0.1.0");
        let client = ModelsClient::new(transport, provider, Arc::new(DummyAuth));

        let (models, etag) = client
            .list_models(request_url, HeaderMap::new())
            .await
            .expect("request should succeed");

        assert_eq!(models, ModelsList::Codex(Vec::new()));
        assert_eq!(etag, Some("\"abc\"".to_string()));
    }
}
