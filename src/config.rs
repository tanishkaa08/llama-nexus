use crate::error::ServerError;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct Config {
    pub(crate) server: ServerConfig,
    // pub(crate) chat: ChatConfig,
    // pub(crate) embedding: EmbeddingConfig,
    // pub(crate) tts: TtsConfig,
}
impl Config {
    pub(crate) fn load(path: impl AsRef<Path>) -> Result<Self, ServerError> {
        let config = config::Config::builder()
            .add_source(config::File::from(path.as_ref()))
            .build()
            .map_err(|e| ServerError::Operation(e.to_string()))?
            .try_deserialize()
            .map_err(|e| ServerError::Operation(e.to_string()))?;

        Ok(config)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ServerConfig {
    pub(crate) host: String,
    pub(crate) port: u16,
}
impl Default for ServerConfig {
    fn default() -> Self {
        ServerConfig {
            host: "0.0.0.0".to_string(),
            port: 8080,
        }
    }
}
