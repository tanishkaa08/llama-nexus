use crate::server::ServerId;
use chat_prompts::PromptTemplateType;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Default, Serialize, Deserialize)]
pub(crate) struct ServerInfo {
    #[serde(rename = "node_version", skip_serializing_if = "Option::is_none")]
    pub(crate) node: Option<String>,
    #[serde(rename = "servers", skip_serializing_if = "HashMap::is_empty")]
    pub(crate) servers: HashMap<ServerId, ApiServer>,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct ApiServer {
    #[serde(rename = "type")]
    pub(crate) ty: String,
    pub(crate) version: String,
    #[serde(rename = "plugin_version", skip_serializing_if = "Option::is_none")]
    pub(crate) plugin_version: Option<String>,
    pub(crate) port: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) chat_model: Option<ModelConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) embedding_model: Option<ModelConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) image_model: Option<ModelConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tts_model: Option<ModelConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) translate_model: Option<ModelConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) transcribe_model: Option<ModelConfig>,
    pub(crate) extras: HashMap<String, String>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub(crate) struct ModelConfig {
    // model name
    name: String,
    // type: chat or embedding
    #[serde(rename = "type")]
    ty: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ctx_size: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub batch_size: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ubatch_size: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_template: Option<PromptTemplateType>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_predict: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reverse_prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_gpu_layers: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub use_mmap: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repeat_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub split_mode: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub main_gpu: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tensor_split: Option<String>,
}

#[test]
fn test_deserialize_api_server() {
    let s = r#"{"type":"sd","version":"0.2.4","plugin_version":"Unknown","port":"12345","image_model":{"name":"sd-v1.5","type":"image","ctx_size":0,"batch_size":0,"ubatch_size":0},"extras":{}}"#;
    let server: ApiServer = serde_json::from_str(s).unwrap();
    assert_eq!(server.ty, "sd");
    assert_eq!(server.version, "0.2.4");
    assert_eq!(server.plugin_version, Some("Unknown".to_string()));
    assert_eq!(server.port, "12345");
    let image_model = server.image_model.unwrap();
    assert_eq!(image_model.name, "sd-v1.5");
    assert_eq!(image_model.ty, "image");
    assert_eq!(server.extras, HashMap::new());
}
