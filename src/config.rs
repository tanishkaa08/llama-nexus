use chat_prompts::MergeRagContextPolicy;
use clap::ValueEnum;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Config {
    pub server: ServerConfig,
    pub rag: RagConfig,
}
impl Config {
    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let config = config::Config::builder()
            .add_source(config::File::with_name(path))
            .build()?;
        Ok(config.try_deserialize::<Self>()?)
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
}

#[derive(Debug, Serialize, Clone)]
pub struct RagConfig {
    pub enable: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    pub rag_policy: MergeRagContextPolicy,
    pub vector_db: VectorDbConfig,
    pub kw_search: KwSearchConfig,
}

impl<'de> Deserialize<'de> for RagConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct RagConfigHelper {
            enable: bool,
            prompt: String,
            rag_policy: String,
            vector_db: VectorDbConfig,
            kw_search: KwSearchConfig,
        }

        let helper = RagConfigHelper::deserialize(deserializer)?;

        let prompt = if helper.prompt.is_empty() {
            None
        } else {
            Some(helper.prompt)
        };

        let rag_policy = MergeRagContextPolicy::from_str(&helper.rag_policy, true)
            .map_err(|e| serde::de::Error::custom(e.to_string()))?;

        Ok(RagConfig {
            enable: helper.enable,
            prompt,
            rag_policy,
            vector_db: helper.vector_db,
            kw_search: helper.kw_search,
        })
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct VectorDbConfig {
    pub url: String,
    pub collection_name: Vec<String>,
    pub limit: u64,
    pub score_threshold: f32,
}

#[derive(Debug, Default, Deserialize, Serialize, Clone)]
pub struct KwSearchConfig {
    pub enable: bool,
    pub url: String,
    pub index_name: String,
}
