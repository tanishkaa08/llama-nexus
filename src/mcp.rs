use std::collections::HashMap;

use once_cell::sync::OnceCell;
use rmcp::{
    RoleClient,
    service::{DynService, RunningService},
};
use tokio::sync::RwLock as TokioRwLock;

// Global MCP tools and clients
pub static MCP_TOOLS: OnceCell<TokioRwLock<HashMap<McpToolName, ServiceName>>> = OnceCell::new();
// Global MCP clients
pub static MCP_SERVICES: OnceCell<TokioRwLock<HashMap<ServiceName, TokioRwLock<McpService>>>> =
    OnceCell::new();

pub type RawMcpService = RunningService<RoleClient, Box<dyn DynService<RoleClient>>>;
pub type ServiceName = String;
pub type McpToolName = String;

#[allow(dead_code)]
pub struct McpService {
    pub name: ServiceName,
    pub raw: RawMcpService,
    pub tools: Vec<McpToolName>,
}
impl McpService {
    pub fn new(name: ServiceName, raw: RawMcpService) -> Self {
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
