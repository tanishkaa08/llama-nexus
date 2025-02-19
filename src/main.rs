mod config;
mod error;
mod handlers;
mod server;

use crate::server::{Server, ServerGroup, ServerKind};
use axum::{
    body::Body,
    http::{HeaderValue, Request},
    routing::{get, post, Router},
};
use clap::Parser;
use config::Config;
use error::{ServerError, ServerResult};
use std::{collections::HashMap, net::SocketAddr, path::PathBuf, sync::Arc};
use tokio::signal;
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;
use tower_http::{
    cors::{Any, CorsLayer},
    trace::TraceLayer,
};
use tracing::{error, info, Level};
use uuid::Uuid;

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
    let config = Config::load(cli.config.unwrap()).map_err(|e| {
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
            "/v1/audio/transcriptions",
            post(handlers::audio_transcriptions_handler),
        )
        .route(
            "/v1/audio/translations",
            post(handlers::audio_translations_handler),
        )
        .route("/v1/audio/speech", post(handlers::audio_tts_handler))
        .route("/v1/images/generations", post(handlers::image_handler))
        .route("/v1/images/edits", post(handlers::image_handler))
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
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .layer(axum::middleware::from_fn(
            |mut req: Request<Body>, next: axum::middleware::Next| async move {
                // 生成请求 ID
                let request_id = Uuid::new_v4().to_string();

                // 将请求 ID 添加到请求头
                req.headers_mut()
                    .insert("x-request-id", HeaderValue::from_str(&request_id).unwrap());

                // 添加取消令牌
                let cancel_token = CancellationToken::new();
                req.extensions_mut().insert(cancel_token);

                // 记录请求开始的日志
                info!(target: "stdout", "Request started - ID: {}", request_id);

                let response = next.run(req).await;

                // 记录请求结束的日志
                info!(target: "stdout", "Request completed - ID: {}", request_id);

                response
            },
        ))
        .with_state(state.clone());

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

    // Set up graceful shutdown
    let server =
        axum::serve(listener, app.into_make_service()).with_graceful_shutdown(shutdown_signal());

    // Start the server
    match server.await {
        Ok(_) => {
            info!(target: "stdout", "Server shutdown completed");
            Ok(())
        }
        Err(e) => {
            let err_msg = format!("Server failed: {}", e);
            error!(target: "stdout", "{}", err_msg);
            Err(ServerError::Operation(err_msg))
        }
    }
}

async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            info!(target: "stdout", "Received Ctrl+C, starting graceful shutdown");
        },
        _ = terminate => {
            info!(target: "stdout", "Received SIGTERM, starting graceful shutdown");
        },
    }
}

/// Application state
pub(crate) struct AppState {
    chat_servers: Arc<RwLock<ServerGroup>>,
    embeddings_servers: Arc<RwLock<ServerGroup>>,
    whisper_servers: Arc<RwLock<ServerGroup>>,
    tts_servers: Arc<RwLock<ServerGroup>>,
    image_servers: Arc<RwLock<ServerGroup>>,
}
impl AppState {
    pub(crate) fn new() -> Self {
        Self {
            chat_servers: Arc::new(RwLock::new(ServerGroup::new(ServerKind::Chat))),
            embeddings_servers: Arc::new(RwLock::new(ServerGroup::new(ServerKind::Embeddings))),
            whisper_servers: Arc::new(RwLock::new(ServerGroup::new(ServerKind::Whisper))),
            tts_servers: Arc::new(RwLock::new(ServerGroup::new(ServerKind::Tts))),
            image_servers: Arc::new(RwLock::new(ServerGroup::new(ServerKind::Image))),
        }
    }

    pub(crate) async fn register_downstream_server(&self, server: Server) -> ServerResult<()> {
        match server.kind {
            ServerKind::Chat => {
                let mut chat_servers = self.chat_servers.write().await;
                chat_servers.register(server).await
            }
            ServerKind::Embeddings => {
                let mut embeddings_servers = self.embeddings_servers.write().await;
                embeddings_servers.register(server).await
            }
            ServerKind::Whisper => {
                let mut whisper_servers = self.whisper_servers.write().await;
                whisper_servers.register(server).await
            }
            ServerKind::Tts => {
                let mut tts_servers = self.tts_servers.write().await;
                tts_servers.register(server).await
            }
            ServerKind::Image => {
                let mut image_servers = self.image_servers.write().await;
                image_servers.register(server).await
            }
        }
    }

    pub(crate) async fn unregister_downstream_server(&self, server: Server) -> ServerResult<()> {
        match server.kind {
            ServerKind::Chat => {
                let mut chat_servers = self.chat_servers.write().await;
                chat_servers.unregister(server).await
            }
            ServerKind::Embeddings => {
                let mut embeddings_servers = self.embeddings_servers.write().await;
                embeddings_servers.unregister(server).await
            }
            ServerKind::Whisper => {
                let mut whisper_servers = self.whisper_servers.write().await;
                whisper_servers.unregister(server).await
            }
            ServerKind::Tts => {
                let mut tts_servers = self.tts_servers.write().await;
                tts_servers.unregister(server).await
            }
            ServerKind::Image => {
                let mut image_servers = self.image_servers.write().await;
                image_servers.unregister(server).await
            }
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
        servers.insert(ServerKind::Chat.to_string(), chat_servers);

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
        servers.insert(ServerKind::Embeddings.to_string(), embeddings_servers);

        // list all the whisper servers
        let whisper_servers = self
            .whisper_servers
            .read()
            .await
            .list_servers()
            .await
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<String>>();
        servers.insert(ServerKind::Whisper.to_string(), whisper_servers);

        // list all the tts servers
        let tts_servers = self
            .tts_servers
            .read()
            .await
            .list_servers()
            .await
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<String>>();
        servers.insert(ServerKind::Tts.to_string(), tts_servers);

        // list all the image servers
        let image_servers = self
            .image_servers
            .read()
            .await
            .list_servers()
            .await
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<String>>();
        servers.insert(ServerKind::Image.to_string(), image_servers);

        Ok(servers)
    }
}
