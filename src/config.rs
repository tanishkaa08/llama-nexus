use crate::{
    dual_debug, dual_error, dual_info,
    error::{ServerError, ServerResult},
    mcp::{McpClient, MCP_CLIENTS, MCP_KEYWORD_SEARCH_CLIENT, MCP_TOOLS, MCP_VECTOR_SEARCH_CLIENT},
};
use chat_prompts::MergeRagContextPolicy;
use clap::ValueEnum;
use rmcp::{model::Tool as RmcpTool, service::ServiceExt, transport::SseClientTransport};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::sync::RwLock as TokioRwLock;

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Config {
    pub server: ServerConfig,
    pub rag: RagConfig,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub server_info_push_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub server_health_push_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mcp: Option<McpConfig>,
}
impl Config {
    pub async fn load(path: impl AsRef<std::path::Path>) -> ServerResult<Self> {
        let config = config::Config::builder()
            .add_source(config::File::with_name(path.as_ref().to_str().unwrap()))
            .build()
            .map_err(|e| {
                let err_msg = format!("Failed to build config: {e}");
                dual_error!("{}", &err_msg);
                ServerError::Operation(err_msg)
            })?;

        let mut config = config.try_deserialize::<Self>().map_err(|e| {
            let err_msg = format!("Failed to deserialize config: {e}");
            dual_error!("{}", &err_msg);
            ServerError::Operation(err_msg)
        })?;

        dual_debug!("config:\n{:#?}", config);

        if let Some(mcp_config) = config.mcp.as_mut() {
            if !mcp_config.server.tool_servers.is_empty() {
                dual_info!("Retrieve the mcp tools from mcp servers");

                for server_config in mcp_config.server.tool_servers.iter_mut() {
                    server_config.connect_mcp_server().await?;
                }

                if let Some(server_vector_search) = mcp_config.server.vector_search_server.as_mut()
                {
                    server_vector_search.connect_mcp_server().await?;
                }

                if let Some(server_keyword_search) =
                    mcp_config.server.keyword_search_server.as_mut()
                {
                    server_keyword_search.connect_mcp_server().await?;
                }
            }
        }

        Ok(config)
    }
}

// Add Default implementation for Config
impl Default for Config {
    fn default() -> Self {
        Self {
            server: ServerConfig {
                host: "127.0.0.1".to_string(),
                port: 8080,
            },
            rag: RagConfig {
                enable: false,
                prompt: None,
                policy: MergeRagContextPolicy::SystemMessage,
                context_window: 1,
            },
            server_info_push_url: None,
            server_health_push_url: None,
            mcp: None,
        }
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
    pub prompt: Option<String>,
    pub policy: MergeRagContextPolicy,
    pub context_window: u64,
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
            policy: String,
            context_window: u64,
        }

        let helper = RagConfigHelper::deserialize(deserializer)?;

        let prompt = if helper.prompt.is_empty() {
            None
        } else {
            Some(helper.prompt)
        };

        let policy = MergeRagContextPolicy::from_str(&helper.policy, true)
            .map_err(|e| serde::de::Error::custom(e.to_string()))?;

        Ok(RagConfig {
            enable: helper.enable,
            prompt,
            policy,
            context_window: helper.context_window,
        })
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct RagVectorSearchConfig {
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

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct McpConfig {
    #[serde(rename = "server")]
    pub server: McpServerConfig,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct McpServerConfig {
    #[serde(rename = "tool")]
    pub tool_servers: Vec<McpToolServerConfig>,
    #[serde(rename = "vector_search")]
    pub vector_search_server: Option<McpVectorSearchServerConfig>,
    #[serde(rename = "keyword_search")]
    pub keyword_search_server: Option<McpKeywordSearchServerConfig>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct McpVectorSearchServerConfig {
    pub name: String,
    pub transport: Transport,
    pub url: String,
    pub enable: bool,
    #[serde(skip_deserializing)]
    pub tools: Option<Vec<RmcpTool>>,
}
impl McpVectorSearchServerConfig {
    pub async fn connect_mcp_server(&mut self) -> ServerResult<()> {
        if self.enable {
            match self.transport {
                Transport::Sse => {
                    let url = self.url.trim_end_matches('/');
                    if !url.ends_with("/sse") {
                        let err_msg = format!(
                            "Invalid vector search mcp SSE URL: {}. The correct format should end with `/sse`",
                            self.url
                        );
                        dual_error!("{}", err_msg);
                        return Err(ServerError::Operation(err_msg.to_string()));
                    }
                    dual_debug!("Sync vector search mcp server: {}", url);

                    // create a sse transport
                    let transport = SseClientTransport::start(url).await.map_err(|e| {
                        let err_msg =
                            format!("Failed to create vector search mcp SSE transport: {e}");
                        dual_error!("{}", &err_msg);
                        ServerError::Operation(err_msg)
                    })?;

                    // create a mcp client
                    let mcp_client = ()
                        .into_dyn()
                        .serve(transport)
                        .await
                        .inspect_err(|e| {
                            tracing::error!("client error: {:?}", e);
                        })
                        .map_err(|e| {
                            let err_msg = format!("Failed to create vector search mcp client: {e}");
                            dual_error!("{}", &err_msg);
                            ServerError::Operation(err_msg)
                        })?;

                    // add mcp client to MCP_CLIENTS
                    match MCP_VECTOR_SEARCH_CLIENT.get() {
                        Some(client) => {
                            let mut locked_client = client.write().await;
                            *locked_client = McpClient::new(self.name.clone(), mcp_client);
                        }
                        None => {
                            MCP_VECTOR_SEARCH_CLIENT
                                .set(TokioRwLock::new(McpClient::new(
                                    self.name.clone(),
                                    mcp_client,
                                )))
                                .map_err(|_| {
                                    let err_msg = "Failed to set MCP_VECTOR_SEARCH_CLIENT";
                                    dual_error!("{}", err_msg);
                                    ServerError::Operation(err_msg.to_string())
                                })?;
                        }
                    }
                }
                _ => {
                    let err_msg = format!(
                        "Unsupported vector search mcp transport: {}",
                        self.transport
                    );
                    dual_error!("{}", err_msg);
                    return Err(ServerError::Operation(err_msg.to_string()));
                }
            }
        }

        Ok(())
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct McpKeywordSearchServerConfig {
    pub name: String,
    pub transport: Transport,
    pub url: String,
    pub enable: bool,
    #[serde(skip_deserializing)]
    pub tools: Option<Vec<RmcpTool>>,
}
impl McpKeywordSearchServerConfig {
    pub async fn connect_mcp_server(&mut self) -> ServerResult<()> {
        if self.enable {
            match self.transport {
                Transport::Sse => {
                    let url = self.url.trim_end_matches('/');
                    if !url.ends_with("/sse") {
                        let err_msg = format!(
                            "Invalid sse URL: {}. The correct format should end with `/sse`",
                            self.url
                        );
                        dual_error!("{}", err_msg);
                        return Err(ServerError::Operation(err_msg.to_string()));
                    }
                    dual_debug!("Sync keyword search mcp server: {}", url);

                    // create a sse transport
                    let transport = SseClientTransport::start(url).await.map_err(|e| {
                        let err_msg =
                            format!("Failed to create keyword search mcp SSE transport: {e}");
                        dual_error!("{}", &err_msg);
                        ServerError::Operation(err_msg)
                    })?;

                    // create a mcp client
                    let mcp_client = ()
                        .into_dyn()
                        .serve(transport)
                        .await
                        .inspect_err(|e| {
                            tracing::error!("client error: {:?}", e);
                        })
                        .map_err(|e| {
                            let err_msg =
                                format!("Failed to create keyword search mcp client: {e}");
                            dual_error!("{}", &err_msg);
                            ServerError::Operation(err_msg)
                        })?;

                    // add mcp client to MCP_CLIENTS
                    match MCP_KEYWORD_SEARCH_CLIENT.get() {
                        Some(client) => {
                            let mut locked_client = client.write().await;
                            *locked_client = McpClient::new(self.name.clone(), mcp_client);
                        }
                        None => {
                            MCP_KEYWORD_SEARCH_CLIENT
                                .set(TokioRwLock::new(McpClient::new(
                                    self.name.clone(),
                                    mcp_client,
                                )))
                                .map_err(|_| {
                                    let err_msg = "Failed to set MCP_KEYWORD_SEARCH_CLIENT";
                                    dual_error!("{}", err_msg);
                                    ServerError::Operation(err_msg.to_string())
                                })?;
                        }
                    }
                }
                _ => {
                    let err_msg = format!(
                        "Unsupported keyword search mcp transport: {}",
                        self.transport
                    );
                    dual_error!("{}", err_msg);
                    return Err(ServerError::Operation(err_msg.to_string()));
                }
            }
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct McpToolServerConfig {
    pub name: String,
    pub transport: Transport,
    pub url: String,
    pub enable: bool,
    #[serde(skip_deserializing)]
    pub tools: Option<Vec<RmcpTool>>,
}
impl McpToolServerConfig {
    pub async fn connect_mcp_server(&mut self) -> ServerResult<()> {
        if self.enable {
            match self.transport {
                Transport::Sse => {
                    let url = self.url.trim_end_matches('/');
                    if !url.ends_with("/sse") {
                        let err_msg = format!(
                            "Invalid sse URL: {}. The correct format should end with `/sse`",
                            self.url
                        );
                        dual_error!("{}", err_msg);
                        return Err(ServerError::Operation(err_msg.to_string()));
                    }
                    dual_debug!("Retrieve mcp tools from mcp server: {}", url);

                    // create a sse transport
                    let transport = SseClientTransport::start(url).await.map_err(|e| {
                        let err_msg = format!("Failed to create sse transport: {e}");
                        dual_error!("{}", &err_msg);
                        ServerError::Operation(err_msg)
                    })?;

                    // create a mcp client
                    let mcp_client = ()
                        .into_dyn()
                        .serve(transport)
                        .await
                        .inspect_err(|e| {
                            tracing::error!("client error: {:?}", e);
                        })
                        .map_err(|e| {
                            let err_msg = format!("Failed to create mcp client: {e}");
                            dual_error!("{}", &err_msg);
                            ServerError::Operation(err_msg)
                        })?;

                    // list tools
                    let tools = mcp_client
                        .peer()
                        .list_tools(Default::default())
                        .await
                        .map_err(|e| {
                            let err_msg = format!("Failed to list tools: {e}");
                            dual_error!("{}", &err_msg);
                            ServerError::Operation(err_msg)
                        })?;
                    dual_info!(
                        "Found {} tools from {} mcp server",
                        &tools.tools.len(),
                        self.name,
                    );

                    dual_debug!(
                        "Retrieved mcp tools: {}",
                        serde_json::to_string_pretty(&tools).unwrap()
                    );

                    // update tools
                    self.tools = Some(tools.tools.clone());

                    // print name of all tools
                    for (idx, tool) in tools.tools.iter().enumerate() {
                        dual_debug!(
                            "Tool {} - name: {}, description: {}",
                            idx,
                            tool.name,
                            tool.description.as_deref().unwrap_or("No description"),
                        );

                        match MCP_TOOLS.get() {
                            Some(mcp_tools) => {
                                let mut tools = mcp_tools.write().await;
                                tools.insert(tool.name.to_string(), self.name.clone());
                            }
                            None => {
                                let tools =
                                    HashMap::from([(tool.name.to_string(), self.name.clone())]);

                                MCP_TOOLS.set(TokioRwLock::new(tools)).map_err(|_| {
                                    let err_msg = "Failed to set MCP_TOOLS";
                                    dual_error!("{}", err_msg);
                                    ServerError::Operation(err_msg.to_string())
                                })?;
                            }
                        }
                    }

                    // add mcp client to MCP_CLIENTS
                    match MCP_CLIENTS.get() {
                        Some(clients) => {
                            let mut clients = clients.write().await;
                            clients.insert(
                                self.name.clone(),
                                TokioRwLock::new(McpClient::new(self.name.clone(), mcp_client)),
                            );
                        }
                        None => {
                            MCP_CLIENTS
                                .set(TokioRwLock::new(HashMap::from([(
                                    self.name.clone(),
                                    TokioRwLock::new(McpClient::new(self.name.clone(), mcp_client)),
                                )])))
                                .map_err(|_| {
                                    let err_msg = "Failed to set MCP_CLIENTS";
                                    dual_error!("{}", err_msg);
                                    ServerError::Operation(err_msg.to_string())
                                })?;
                        }
                    }
                }
                _ => {
                    let err_msg = format!("Unsupported transport: {}", self.transport);
                    dual_error!("{}", err_msg);
                    return Err(ServerError::Operation(err_msg.to_string()));
                }
            }
        }

        Ok(())
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub enum Transport {
    #[serde(rename = "sse")]
    Sse,
    #[serde(rename = "stdio")]
    Stdio,
    #[serde(rename = "streamable-http")]
    StreamableHttp,
}
impl std::fmt::Display for Transport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Transport::Sse => write!(f, "sse"),
            Transport::Stdio => write!(f, "stdio"),
            Transport::StreamableHttp => write!(f, "streamable-http"),
        }
    }
}
