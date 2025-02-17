mod config;
mod error;
mod handlers;
mod server;

use crate::server::{Server, ServerGroup, ServerKind};
use axum::routing::{get, post, Router};
use clap::Parser;
use config::Config;
use error::{ServerError, ServerResult};
use std::{collections::HashMap, net::SocketAddr, path::PathBuf, sync::Arc};
use tokio::sync::RwLock;
use tower_http::{
    cors::{Any, CorsLayer},
    trace::TraceLayer,
};
use tracing::{error, info, Level};

#[derive(Debug, Parser)]
#[command(name = "LlamaEdge-Q", version = env!("CARGO_PKG_VERSION"), author = env!("CARGO_PKG_AUTHORS"), about = "LlamaEdge Proxy Server")]
struct Cli {
    /// Path to the configuration file (*.toml)
    #[arg(short, long, default_value = "config.toml")]
    config: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> ServerResult<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_target(false)
        .with_level(true)
        .with_file(true)
        .with_line_number(true)
        .with_thread_ids(true)
        .with_max_level(Level::INFO)
        .init();

    // parse the command line arguments
    let cli = Cli::parse();

    // log the version of the server
    info!(target: "stdout", "Version: {}", env!("CARGO_PKG_VERSION"));

    // Load the config
    let config = Config::load(&cli.config.unwrap()).map_err(|e| {
        let err_msg = format!("Failed to load config: {}", e);

        error!(target: "stdout", "{}", err_msg);

        ServerError::Operation(err_msg)
    })?;

    // Set up CORS
    let cors = CorsLayer::new()
        .allow_methods(Any)
        .allow_headers(Any)
        .allow_origin(Any);

    let state = Arc::new(AppState::new());

    // Set up the router
    let app = Router::new()
        .route("/v1/chat/completions", post(handlers::chat_handler))
        .route("/v1/embeddings", post(handlers::embeddings_handler))
        .route(
            "/admin/servers/register",
            post(handlers::register_downstream_server_handler),
        )
        .route(
            "/admin/servers/unregister",
            post(handlers::remove_downstream_server_handler),
        )
        .route(
            "/admin/servers",
            get(handlers::list_downstream_servers_handler),
        )
        .layer(TraceLayer::new_for_http())
        .layer(cors)
        .with_state(state);

    let addr: SocketAddr = format!("{}:{}", &config.server.host, &config.server.port)
        .parse()
        .expect("Invalid host/port configuration");

    // Create the listener
    let listener = tokio::net::TcpListener::bind(&addr).await.map_err(|e| {
        let err_msg = format!("Failed to bind to address: {}", e);

        error!(target: "stdout", "{}", err_msg);

        ServerError::Operation(err_msg)
    })?;
    info!(target: "stdout", "Listening on {}", addr);

    // Start the server
    match axum::serve(listener, app.into_make_service()).await {
        Ok(_) => Ok(()),
        Err(e) => {
            let err_msg = format!("Server failed to start: {}", e);

            error!(target: "stdout", "{}", err_msg);

            return Err(ServerError::Operation(err_msg));
        }
    }
}

/// Application state
pub(crate) struct AppState {
    chat_servers: Arc<RwLock<ServerGroup>>,
    embeddings_servers: Arc<RwLock<ServerGroup>>,
    // audio_services: Arc<RwLock<ServerGroup>>,
    // image_services: Arc<RwLock<ServerGroup>>,
    // rag_services: Arc<RwLock<ServerGroup>>,
}
impl AppState {
    pub(crate) fn new() -> Self {
        Self {
            chat_servers: Arc::new(RwLock::new(ServerGroup::new(ServerKind::CHAT))),
            embeddings_servers: Arc::new(RwLock::new(ServerGroup::new(ServerKind::EMBEDDINGS))),
            // audio_services: Arc::new(RwLock::new(ServerGroup::new(ServerKind::TRANSCRIPT))),
            // image_services: Arc::new(RwLock::new(ServerGroup::new(ServerKind::IMAGE))),
            // rag_services: Arc::new(RwLock::new(ServerGroup::new(ServerKind::RAG))),
        }
    }

    pub(crate) async fn register_downstream_server(&self, server: Server) -> ServerResult<()> {
        match server.kind {
            ServerKind::CHAT => {
                let mut chat_servers = self.chat_servers.write().await;
                chat_servers.register(server).await
            }
            ServerKind::EMBEDDINGS => {
                let mut embeddings_servers = self.embeddings_servers.write().await;
                embeddings_servers.register(server).await
            }
            _ => Err(ServerError::InvalidServerKind(format!(
                "Invalid server kind: {}",
                server.kind
            ))),
        }
    }

    pub(crate) async fn unregister_downstream_server(&self, server: Server) -> ServerResult<()> {
        match server.kind {
            ServerKind::CHAT => {
                let mut chat_servers = self.chat_servers.write().await;
                chat_servers.unregister(server).await
            }
            ServerKind::EMBEDDINGS => {
                let mut embeddings_servers = self.embeddings_servers.write().await;
                embeddings_servers.unregister(server).await
            }
            _ => Err(ServerError::InvalidServerKind(format!(
                "Invalid server kind: {}",
                server.kind
            ))),
        }
    }

    pub(crate) async fn list_downstream_servers(
        &self,
    ) -> ServerResult<HashMap<String, Vec<String>>> {
        let mut servers = HashMap::new();

        // list all the chat servers
        let chat_servers = self
            .chat_servers
            .read()
            .await
            .list_servers()
            .await
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<String>>();
        servers.insert(ServerKind::CHAT.to_string(), chat_servers);

        // list all the embeddings servers
        let embeddings_servers = self
            .embeddings_servers
            .read()
            .await
            .list_servers()
            .await
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<String>>();
        servers.insert(ServerKind::EMBEDDINGS.to_string(), embeddings_servers);

        Ok(servers)
    }
}
