use chat_prompts::PromptTemplateType;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Default, Serialize, Deserialize)]
pub(crate) struct ServerInfo {
    #[serde(rename = "node_version", skip_serializing_if = "Option::is_none")]
    pub(crate) node: Option<String>,
    #[serde(rename = "servers", skip_serializing_if = "Vec::is_empty")]
    pub(crate) servers: Vec<ApiServer>,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct ApiServer {
    #[serde(rename = "type")]
    pub(crate) ty: String,
    pub(crate) version: String,
    #[serde(rename = "ggml_plugin_version")]
    pub(crate) plugin_version: String,
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
    pub ctx_size: u64,
    pub batch_size: u64,
    pub ubatch_size: u64,
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
