mod config;
mod error;
mod handlers;
mod rag;
mod server;

use crate::server::{Server, ServerGroup, ServerKind};
use axum::{
    body::Body,
    http::{self, HeaderValue, Request},
    routing::{get, post, Router},
};
use clap::Parser;
use config::Config;
use error::{ServerError, ServerResult};
use once_cell::sync::OnceCell;
use std::{
    collections::HashMap,
    net::{IpAddr, SocketAddr},
    str::FromStr,
    sync::Arc,
};
use tokio::signal;
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;
use tower_http::{
    cors::{Any, CorsLayer},
    trace::TraceLayer,
};
use tracing::{error, info, Level};
use uuid::Uuid;

// global system prompt
pub(crate) static GLOBAL_RAG_PROMPT: OnceCell<String> = OnceCell::new();

// Global context window used for setting the max number of user messages for the retrieval
pub(crate) static CONTEXT_WINDOW: OnceCell<u64> = OnceCell::new();

#[derive(Debug, Parser)]
struct Cli {
    /// Path to the config file
    #[arg(long, default_value = "config.toml", value_parser = clap::value_parser!(String))]
    config: String,
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
        .with_max_level(get_log_level_from_env())
        .init();

    // parse the command line arguments
    let cli = Cli::parse();

    // log the version of the server
    info!(target: "stdout", "Version: {}", env!("CARGO_PKG_VERSION"));

    // Set up CORS
    let cors = CorsLayer::new()
        .allow_methods([http::Method::GET, http::Method::POST])
        .allow_headers(Any)
        .allow_origin(Any);

    // load the config
    let config = Config::load(&cli.config).map_err(|e| {
        let err_msg = format!("Failed to load config: {}", e);

        error!(target: "stdout", "{}", err_msg);

        ServerError::FailedToLoadConfig(err_msg)
    })?;

    if config.rag.enable {
        info!(target: "stdout", "RAG is enabled");
    }

    // socket address
    let addr = SocketAddr::from((
        config.server.host.parse::<IpAddr>().unwrap(),
        config.server.port,
    ));

    let state = Arc::new(AppState::new(config));

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
        .route("/v1/chunks", post(handlers::chunks_handler))
        .route(
            "/admin/servers/register",
            post(handlers::admin::register_downstream_server_handler),
        )
        .route(
            "/admin/servers/unregister",
            post(handlers::admin::remove_downstream_server_handler),
        )
        .route(
            "/admin/servers",
            get(handlers::admin::list_downstream_servers_handler),
        )
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .layer(axum::middleware::from_fn(
            |mut req: Request<Body>, next: axum::middleware::Next| async move {
                // Generate request ID
                let request_id = Uuid::new_v4().to_string();

                // Add request ID to headers
                req.headers_mut()
                    .insert("x-request-id", HeaderValue::from_str(&request_id).unwrap());

                // Add cancellation token
                let cancel_token = CancellationToken::new();
                req.extensions_mut().insert(cancel_token);

                // Log request start
                info!(target: "stdout", "Request started - ID: {}", request_id);

                let response = next.run(req).await;

                // Log request completion
                info!(target: "stdout", "Request completed - ID: {}", request_id);

                response
            },
        ))
        .with_state(state.clone());

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
    servers: Arc<RwLock<HashMap<ServerKind, ServerGroup>>>,
    config: Arc<RwLock<Config>>,
}
impl AppState {
    pub(crate) fn new(config: Config) -> Self {
        Self {
            servers: Arc::new(RwLock::new(HashMap::new())),
            config: Arc::new(RwLock::new(config)),
        }
    }

    pub(crate) async fn register_downstream_server(&self, server: Server) -> ServerResult<()> {
        let server_kind = &server.kind;

        if server_kind.contains(ServerKind::chat) {
            self.servers
                .write()
                .await
                .entry(ServerKind::chat)
                .or_insert(ServerGroup::new(ServerKind::chat))
                .register(&server)
                .await?;
        }
        if server_kind.contains(ServerKind::embeddings) {
            self.servers
                .write()
                .await
                .entry(ServerKind::embeddings)
                .or_insert(ServerGroup::new(ServerKind::embeddings))
                .register(&server)
                .await?;
        }
        if server.kind.contains(ServerKind::image) {
            self.servers
                .write()
                .await
                .entry(ServerKind::image)
                .or_insert(ServerGroup::new(ServerKind::image))
                .register(&server)
                .await?;
        }
        if server.kind.contains(ServerKind::tts) {
            self.servers
                .write()
                .await
                .entry(ServerKind::tts)
                .or_insert(ServerGroup::new(ServerKind::tts))
                .register(&server)
                .await?;
        }
        if server_kind.contains(ServerKind::translate) {
            self.servers
                .write()
                .await
                .entry(ServerKind::translate)
                .or_insert(ServerGroup::new(ServerKind::translate))
                .register(&server)
                .await?;
        }
        if server_kind.contains(ServerKind::transcribe) {
            self.servers
                .write()
                .await
                .entry(ServerKind::transcribe)
                .or_insert(ServerGroup::new(ServerKind::transcribe))
                .register(&server)
                .await?;
        }

        Ok(())
    }

    pub(crate) async fn unregister_downstream_server(
        &self,
        server_id: impl AsRef<str>,
    ) -> ServerResult<()> {
        let mut servers = self.servers.write().await;
        let mut found = false;

        let server_kind_s = server_id.as_ref().split("-server-").next().unwrap();
        for kind in server_kind_s.split("-") {
            let kind = ServerKind::from_str(kind).unwrap();
            if let Some(servers) = servers.get_mut(&kind) {
                servers.unregister(server_id.as_ref()).await?;
                info!(
                    target: "stdout",
                    message = format!("Unregistered {} server: {}", &kind, server_id.as_ref())
                );

                if !found {
                    found = true;
                }
            }
        }

        if !found {
            return Err(ServerError::Operation(format!(
                "Server {} not found",
                server_id.as_ref()
            )));
        }

        Ok(())
    }

    pub(crate) async fn list_downstream_servers(
        &self,
    ) -> ServerResult<HashMap<ServerKind, Vec<Server>>> {
        let servers = self.servers.read().await;

        let mut server_groups = HashMap::new();
        for (kind, group) in servers.iter() {
            if !group.is_empty().await {
                let servers = group.servers.read().await;

                server_groups.insert(*kind, servers.clone());
            }
        }

        Ok(server_groups)
    }
}

fn get_log_level_from_env() -> Level {
    match std::env::var("LLAMA_LOG").ok().as_deref() {
        Some("trace") => Level::TRACE,
        Some("debug") => Level::DEBUG,
        Some("info") => Level::INFO,
        Some("warn") => Level::WARN,
        Some("error") => Level::ERROR,
        _ => Level::INFO,
    }
}
