use once_cell::sync::OnceCell;
use rmcp::{
    service::{DynService, RunningService},
    RoleClient,
};
use std::collections::HashMap;
use tokio::sync::RwLock as TokioRwLock;

// Global MCP tools and clients
pub static MCP_TOOLS: OnceCell<TokioRwLock<HashMap<McpToolName, McpClientName>>> = OnceCell::new();
// Global MCP clients
pub static MCP_CLIENTS: OnceCell<TokioRwLock<HashMap<McpClientName, TokioRwLock<McpClient>>>> =
    OnceCell::new();

pub type RawMcpClient = RunningService<RoleClient, Box<dyn DynService<RoleClient>>>;
pub type McpClientName = String;
pub type McpToolName = String;

#[allow(dead_code)]
pub struct McpClient {
    pub name: McpClientName,
    pub raw: RawMcpClient,
    pub tools: Vec<McpToolName>,
}
impl McpClient {
    pub fn new(name: McpClientName, raw: RawMcpClient) -> Self {
        Self {
            name,
            raw,
            tools: Vec::new(),
        }
    }

    pub fn has_tool(&self, tool_name: impl AsRef<str>) -> bool {
        if self.tools.is_empty() {
            false
        } else {
            self.tools.contains(&tool_name.as_ref().to_string())
        }
    }
}
