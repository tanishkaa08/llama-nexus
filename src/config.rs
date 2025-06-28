use std::{collections::HashMap, env};

use chat_prompts::MergeRagContextPolicy;
use clap::ValueEnum;
use endpoints::chat::McpTransport;
use rmcp::{
    model::{ClientCapabilities, ClientInfo, Implementation, Tool as RmcpTool},
    service::ServiceExt,
    transport::{SseClientTransport, StreamableHttpClientTransport},
};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock as TokioRwLock;

use crate::{
    dual_debug, dual_error, dual_info,
    error::{ServerError, ServerResult},
    mcp::{MCP_SERVICES, MCP_TOOLS, McpService},
};

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Config {
    pub server: ServerConfig,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rag: Option<RagConfig>,
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

        if let Some(mcp_config) = config.mcp.as_mut()
            && !mcp_config.server.tool_servers.is_empty()
        {
            for server_config in mcp_config.server.tool_servers.iter_mut() {
                server_config.connect_mcp_server().await?;
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
            rag: None,
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
            policy: String,
            context_window: u64,
        }

        let helper = RagConfigHelper::deserialize(deserializer)?;

        let policy = MergeRagContextPolicy::from_str(&helper.policy, true)
            .map_err(|e| serde::de::Error::custom(e.to_string()))?;

        Ok(RagConfig {
            enable: helper.enable,
            prompt: None,
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
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct McpToolServerConfig {
    pub name: String,
    pub transport: McpTransport,
    pub url: String,
    pub enable: bool,
    #[serde(skip_deserializing)]
    pub tools: Option<Vec<RmcpTool>>,
}
impl McpToolServerConfig {
    /// Connect the mcp server if it is enabled
    pub async fn connect_mcp_server(&mut self) -> ServerResult<()> {
        if self.enable {
            match self.transport {
                McpTransport::Sse => {
                    let url = self.url.trim_end_matches('/');
                    if !url.ends_with("/sse") {
                        let err_msg = format!(
                            "Invalid mcp tools sse URL: {}. The correct format should end with `/sse`",
                            self.url
                        );
                        dual_error!("{}", err_msg);
                        return Err(ServerError::Operation(err_msg.to_string()));
                    }
                    dual_debug!("Sync mcp tools from mcp server: {}", url);

                    // create a sse transport
                    let transport = SseClientTransport::start(url).await.map_err(|e| {
                        let err_msg = format!("Failed to create sse transport: {e}");
                        dual_error!("{}", &err_msg);
                        ServerError::Operation(err_msg)
                    })?;

                    // create a mcp client
                    let service = ().into_dyn().serve(transport).await.map_err(|e| {
                        let err_msg = format!(
                            "Failed to connect to mcp server (name: {}, url: {}, transport: {}). {e}. Please check if the mcp server is running.",
                            self.name, self.url, self.transport
                        );
                        dual_error!("{}", &err_msg);
                        ServerError::Operation(err_msg)
                    })?;

                    // list tools
                    let tools = service.list_all_tools().await.map_err(|e| {
                        let err_msg = format!("Failed to list tools: {e}");
                        dual_error!("{}", &err_msg);
                        ServerError::Operation(err_msg)
                    })?;
                    dual_info!("Found {} tools from {} mcp server", tools.len(), self.name,);

                    dual_debug!(
                        "Retrieved mcp tools: {}",
                        serde_json::to_string_pretty(&tools).unwrap()
                    );

                    // update tools
                    self.tools = Some(tools.clone());

                    let mut client = McpService::new(self.name.clone(), service);
                    client.tools = tools.iter().map(|tool| tool.name.to_string()).collect();

                    // print name of all tools
                    for (idx, tool) in tools.iter().enumerate() {
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
                    match MCP_SERVICES.get() {
                        Some(clients) => {
                            let mut clients = clients.write().await;
                            clients.insert(self.name.clone(), TokioRwLock::new(client));
                        }
                        None => {
                            MCP_SERVICES
                                .set(TokioRwLock::new(HashMap::from([(
                                    self.name.clone(),
                                    TokioRwLock::new(client),
                                )])))
                                .map_err(|_| {
                                    let err_msg = "Failed to set MCP_CLIENTS";
                                    dual_error!("{}", err_msg);
                                    ServerError::Operation(err_msg.to_string())
                                })?;
                        }
                    }
                }
                McpTransport::StreamHttp => {
                    let url = self.url.trim_end_matches('/');
                    if !url.ends_with("/mcp") {
                        let err_msg = format!(
                            "Invalid mcp tools stream-http URL: {}. The correct format should end with `/mcp`",
                            self.url
                        );
                        dual_error!("{}", err_msg);
                        return Err(ServerError::Operation(err_msg.to_string()));
                    }
                    dual_debug!("Sync mcp tools from mcp server: {}", url);

                    // create a stream-http transport
                    let transport = StreamableHttpClientTransport::from_uri(url);

                    // create a mcp client
                    let client_info = ClientInfo {
                        protocol_version: Default::default(),
                        capabilities: ClientCapabilities::default(),
                        client_info: Implementation {
                            name: env!("CARGO_PKG_NAME").to_string(),
                            version: env!("CARGO_PKG_VERSION").to_string(),
                        },
                    };
                    let service = client_info.into_dyn().serve(transport).await.map_err(|e| {
                        let err_msg = format!(
                            "Failed to connect to mcp server (name: {}, url: {}, transport: {}). {e}. Please check if the mcp server is running.",
                            self.name, self.url, self.transport
                        );
                        dual_error!("{}", &err_msg);
                        ServerError::Operation(err_msg)
                    })?;

                    // list tools
                    let tools = service.list_all_tools().await.map_err(|e| {
                        let err_msg = format!("Failed to list tools: {e}");
                        dual_error!("{}", &err_msg);
                        ServerError::Operation(err_msg)
                    })?;
                    dual_info!("Found {} tools from {} mcp server", tools.len(), self.name,);

                    dual_debug!(
                        "Retrieved mcp tools: {}",
                        serde_json::to_string_pretty(&tools).unwrap()
                    );

                    // update tools
                    self.tools = Some(tools.clone());

                    let mut client = McpService::new(self.name.clone(), service);
                    client.tools = tools.iter().map(|tool| tool.name.to_string()).collect();

                    // print name of all tools
                    for (idx, tool) in tools.iter().enumerate() {
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
                    match MCP_SERVICES.get() {
                        Some(clients) => {
                            let mut clients = clients.write().await;
                            clients.insert(self.name.clone(), TokioRwLock::new(client));
                        }
                        None => {
                            MCP_SERVICES
                                .set(TokioRwLock::new(HashMap::from([(
                                    self.name.clone(),
                                    TokioRwLock::new(client),
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
    #[serde(rename = "stream-http")]
    StreamHttp,
}
impl std::fmt::Display for Transport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Transport::Sse => write!(f, "sse"),
            Transport::Stdio => write!(f, "stdio"),
            Transport::StreamHttp => write!(f, "streamable-http"),
        }
    }
}
