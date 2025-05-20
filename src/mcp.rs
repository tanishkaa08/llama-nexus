use once_cell::sync::OnceCell;
use rmcp::{
    service::{DynService, RunningService},
    RoleClient,
};
use std::collections::HashMap;
use tokio::sync::RwLock as TokioRwLock;

pub static MCP_TOOLS: OnceCell<TokioRwLock<HashMap<String, McpClientName>>> = OnceCell::new();
pub static MCP_CLIENTS: OnceCell<TokioRwLock<HashMap<McpClientName, TokioRwLock<McpClient>>>> =
    OnceCell::new();
pub static MCP_VECTOR_SEARCH_CLIENT: OnceCell<TokioRwLock<McpClient>> = OnceCell::new();
pub static MCP_KEYWORD_SEARCH_CLIENT: OnceCell<TokioRwLock<McpClient>> = OnceCell::new();

pub type RawMcpClient = RunningService<RoleClient, Box<dyn DynService<RoleClient>>>;
pub type McpClientName = String;

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
