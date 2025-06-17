use once_cell::sync::OnceCell;
use rmcp::{
    service::{DynService, RunningService},
    RoleClient,
};
use std::collections::HashMap;
use tokio::sync::RwLock as TokioRwLock;

// MCP tools and clients by user (or by request)
pub static USER_TO_MCP_TOOLS: OnceCell<TokioRwLock<McpToolMap>> = OnceCell::new();
// MCP clients by user (or by request)
pub static USER_TO_MCP_CLIENTS: OnceCell<TokioRwLock<McpClientMap>> = OnceCell::new();

// Global MCP tools and clients
pub static MCP_TOOLS: OnceCell<TokioRwLock<HashMap<McpToolName, McpClientName>>> = OnceCell::new();
// Global MCP clients
pub static MCP_CLIENTS: OnceCell<TokioRwLock<HashMap<McpClientName, TokioRwLock<McpClient>>>> =
    OnceCell::new();

// Global MCP clients for vector search
pub static MCP_VECTOR_SEARCH_CLIENT: OnceCell<TokioRwLock<McpClient>> = OnceCell::new();
// Global MCP clients for keyword search
pub static MCP_KEYWORD_SEARCH_CLIENT: OnceCell<TokioRwLock<McpClient>> = OnceCell::new();

pub type McpToolMap = HashMap<UserId, TokioRwLock<HashMap<McpToolName, McpClientName>>>;
pub type McpClientMap =
    HashMap<UserId, TokioRwLock<HashMap<McpClientName, TokioRwLock<McpClient>>>>;
pub type RawMcpClient = RunningService<RoleClient, Box<dyn DynService<RoleClient>>>;
pub type McpClientName = String;
pub type McpToolName = String;
pub type UserId = String;

#[allow(dead_code)]
pub struct McpClient {
    pub name: McpClientName,
    pub raw: RawMcpClient,
}
impl McpClient {
    pub fn new(name: McpClientName, raw: RawMcpClient) -> Self {
        Self { name, raw }
    }
}
