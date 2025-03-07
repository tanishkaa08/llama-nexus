mod config;
mod error;
mod handlers;
mod info;
mod rag;
mod server;

use crate::{
    info::ServerInfo,
    server::{Server, ServerGroup, ServerId, ServerKind},
};
use axum::{
    body::Body,
    http::{self, HeaderValue, Request},
    routing::{get, post, Router},
};
use clap::{Parser, Subcommand};
use config::Config;
use error::{ServerError, ServerResult};
use futures_util::stream::{self, StreamExt};
use once_cell::sync::OnceCell;
use std::{
    collections::{HashMap, HashSet},
    net::{IpAddr, SocketAddr},
    path::PathBuf,
    str::FromStr,
    sync::Arc,
};
use tokio::signal;
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;
use tower_http::{
    cors::{Any, CorsLayer},
    services::ServeDir,
    trace::TraceLayer,
};
use tracing::{debug, error, info, warn, Level};
use uuid::Uuid;

// global system prompt
pub(crate) static GLOBAL_RAG_PROMPT: OnceCell<String> = OnceCell::new();

// Global context window used for setting the max number of user messages for the retrieval
pub(crate) static CONTEXT_WINDOW: OnceCell<u64> = OnceCell::new();

// Global health check interval for downstream servers in seconds
pub(crate) static HEALTH_CHECK_INTERVAL: OnceCell<u64> = OnceCell::new();

#[derive(Debug, Parser)]
struct Cli {
    /// Enable RAG
    #[arg(long, default_value = "false")]
    rag: bool,
    /// Enable health check for downstream servers
    #[arg(long, default_value = "false")]
    check_health: bool,
    /// Health check interval for downstream servers in seconds
    #[arg(long, default_value = "60")]
    check_health_interval: u64,
    /// Root path for the Web UI files
    #[arg(long, default_value = "chatbot-ui")]
    web_ui: PathBuf,
    /// Subcommands
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Configuration mode - use configuration file
    Config {
        /// Path to the config file
        #[arg(long, default_value = "config.toml", value_parser = clap::value_parser!(String))]
        file: String,
    },
    /// Gaia mode - use Gaia settings
    Gaia {
        /// Gaia domain
        #[arg(long, value_parser = clap::value_parser!(String), required = true)]
        domain: String,
        /// Gaia device ID
        #[arg(long, value_parser = clap::value_parser!(String), required = true)]
        device_id: String,
        /// Vector database URL
        #[arg(long, default_value = "http://localhost:6333")]
        vdb_url: String,
        /// Vector database collection names
        #[arg(long, default_value = "default", value_delimiter = ',')]
        vdb_collection_name: Vec<String>,
        /// Vector database result limit
        #[arg(long, default_value = "1")]
        vdb_limit: u64,
        /// Vector database score threshold
        #[arg(long, default_value = "0.5")]
        vdb_score_threshold: f32,
        /// Host address to bind to
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        /// Port to listen on
        #[arg(long, default_value = "9068")]
        port: u16,
    },
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

    // Load the config based on the command
    let config = match &cli.command {
        Command::Config { file } => {
            // Load config from file
            match Config::load(file) {
                Ok(mut config) => {
                    if cli.rag {
                        config.rag.enable = true;
                        info!(target: "stdout", "RAG is enabled");
                    }
                    config
                }
                Err(e) => {
                    let err_msg = format!("Failed to load config: {}", e);
                    error!(target: "stdout", "{}", err_msg);
                    return Err(ServerError::FailedToLoadConfig(err_msg));
                }
            }
        }
        Command::Gaia {
            domain,
            device_id,
            host,
            port,
            vdb_url,
            vdb_collection_name,
            vdb_limit,
            vdb_score_threshold,
        } => {
            // Use default config for gaia command
            info!(target: "stdout", "Using default configuration for gaia command");
            let mut config = Config::default();

            // set the server info push url
            let server_info_url =
                format!("https://hub.domain.{}/device-info/{}", domain, device_id);
            config.server_info_push_url = Some(server_info_url);

            // set the server health push url
            let server_health_url =
                format!("https://hub.domain.{}/device-health/{}", domain, device_id);
            config.server_health_push_url = Some(server_health_url);

            if cli.rag {
                config.rag.enable = true;
                info!(target: "stdout", "RAG is enabled");
            }

            // Set the VDB configuration
            info!(target: "stdout", "VDB URL: {}", vdb_url);
            config.rag.vector_db.url = vdb_url.clone();
            info!(target: "stdout", "VDB Collections: {:?}", vdb_collection_name);
            config.rag.vector_db.collection_name = vdb_collection_name.clone();
            info!(target: "stdout", "VDB Limit: {}", vdb_limit);
            config.rag.vector_db.limit = *vdb_limit;
            info!(target: "stdout", "VDB Score Threshold: {}", vdb_score_threshold);
            config.rag.vector_db.score_threshold = *vdb_score_threshold;

            // Set the host and port
            config.server.host = host.clone();
            config.server.port = *port;
            info!(target: "stdout", "Server will listen on {}:{}", host, port);

            config
        }
    };

    // set the health check interval
    HEALTH_CHECK_INTERVAL
        .set(cli.check_health_interval)
        .map_err(|e| {
            let err_msg = format!("Failed to set health check interval: {}", e);
            error!(target: "stdout", "{}", err_msg);
            ServerError::Operation(err_msg)
        })?;

    // socket address
    let addr = SocketAddr::from((
        config.server.host.parse::<IpAddr>().unwrap(),
        config.server.port,
    ));

    let mut server_info = ServerInfo::default();
    {
        // get the environment variable `NODE_VERSION`
        // Note that this is for satisfying the requirement of `gaianet-node` project.
        let node = std::env::var("NODE_VERSION").ok();
        if node.is_some() {
            // log node version
            info!(target: "stdout", "gaianet_node_version: {}", node.as_ref().unwrap());
        }

        server_info.node = node;
    }

    let state = Arc::new(AppState::new(config, server_info));

    // Start the health check task if enabled
    if cli.check_health {
        info!(target: "stdout", "Health check is enabled");
        Arc::clone(&state).start_health_check_task().await;
    }

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
        .route("/v1/create/rag", post(handlers::create_rag_handler))
        .route("/v1/chunks", post(handlers::chunks_handler))
        .route("/v1/models", get(handlers::models_handler))
        .route("/v1/info", get(handlers::info_handler))
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
        .nest_service(
            "/",
            ServeDir::new(&cli.web_ui).not_found_service(
                ServeDir::new(&cli.web_ui).append_index_html_on_directories(true),
            ),
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
    server_group: Arc<RwLock<HashMap<ServerKind, ServerGroup>>>,
    config: Arc<RwLock<Config>>,
    server_info: Arc<RwLock<ServerInfo>>,
    models: Arc<RwLock<HashMap<ServerId, Vec<endpoints::models::Model>>>>,
}
impl AppState {
    pub(crate) fn new(config: Config, server_info: ServerInfo) -> Self {
        Self {
            server_group: Arc::new(RwLock::new(HashMap::new())),
            config: Arc::new(RwLock::new(config)),
            server_info: Arc::new(RwLock::new(server_info)),
            models: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub(crate) async fn register_downstream_server(&self, server: Server) -> ServerResult<()> {
        if server.kind.contains(ServerKind::chat) {
            self.server_group
                .write()
                .await
                .entry(ServerKind::chat)
                .or_insert(ServerGroup::new(ServerKind::chat))
                .register(server.clone())
                .await?;
        }
        if server.kind.contains(ServerKind::embeddings) {
            self.server_group
                .write()
                .await
                .entry(ServerKind::embeddings)
                .or_insert(ServerGroup::new(ServerKind::embeddings))
                .register(server.clone())
                .await?;
        }
        if server.kind.contains(ServerKind::image) {
            self.server_group
                .write()
                .await
                .entry(ServerKind::image)
                .or_insert(ServerGroup::new(ServerKind::image))
                .register(server.clone())
                .await?;
        }
        if server.kind.contains(ServerKind::tts) {
            self.server_group
                .write()
                .await
                .entry(ServerKind::tts)
                .or_insert(ServerGroup::new(ServerKind::tts))
                .register(server.clone())
                .await?;
        }
        if server.kind.contains(ServerKind::translate) {
            self.server_group
                .write()
                .await
                .entry(ServerKind::translate)
                .or_insert(ServerGroup::new(ServerKind::translate))
                .register(server.clone())
                .await?;
        }
        if server.kind.contains(ServerKind::transcribe) {
            self.server_group
                .write()
                .await
                .entry(ServerKind::transcribe)
                .or_insert(ServerGroup::new(ServerKind::transcribe))
                .register(server.clone())
                .await?;
        }

        // Push server info to external service if configured
        if let Some(push_url) = &self.config.read().await.server_info_push_url {
            let server_info = self.server_info.read().await.clone();

            // Retry up to 3 times to push the server info to the external service
            let mut retry_count = 0;
            let max_retries = 3;
            let mut last_error = None;

            info!(target: "stdout", "Push server info");

            while retry_count < max_retries {
                match reqwest::Client::new()
                    .post(push_url)
                    .json(&server_info)
                    .send()
                    .await
                {
                    Ok(_) => break,
                    Err(e) => {
                        retry_count += 1;
                        last_error = Some(e);
                        if retry_count < max_retries {
                            // Wait a moment before retrying
                            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                        }
                    }
                }
            }

            if let Some(e) = last_error {
                if retry_count >= max_retries {
                    error!(
                        target: "stdout",
                        message = format!("Failed to push server info after {} attempts: {}", max_retries, e)
                    );
                }
            }
        }

        Ok(())
    }

    pub(crate) async fn unregister_downstream_server(
        &self,
        server_id: impl AsRef<str>,
    ) -> ServerResult<()> {
        let mut found = false;

        // unregister the server from the servers
        {
            // parse server kind from server id
            let kinds = server_id
                .as_ref()
                .split("-server-")
                .next()
                .unwrap()
                .split("-")
                .collect::<Vec<&str>>();

            let group_map = self.server_group.read().await;

            for kind in kinds {
                let kind = ServerKind::from_str(kind).unwrap();
                if let Some(group) = group_map.get(&kind) {
                    group.unregister(server_id.as_ref()).await?;
                    info!(
                        target: "stdout",
                        message = format!("Unregistered {} server: {}", &kind, server_id.as_ref())
                    );

                    if !found {
                        found = true;
                    }
                }
            }
        }

        if found {
            // remove the server info from the server_info
            let mut server_info = self.server_info.write().await;
            server_info.servers.remove(server_id.as_ref());

            // remove the server from the models
            let mut models = self.models.write().await;
            models.remove(server_id.as_ref());
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
        let servers = self.server_group.read().await;

        let mut server_groups = HashMap::new();
        for (kind, group) in servers.iter() {
            if !group.is_empty().await {
                let servers = group.servers.read().await;

                // Create a new Vec with cloned Server instances using async stream
                let server_vec = stream::iter(servers.iter())
                    .then(|server_lock| async move {
                        let server = server_lock.read().await;
                        server.clone()
                    })
                    .collect::<Vec<_>>()
                    .await;

                server_groups.insert(*kind, server_vec);
            }
        }

        Ok(server_groups)
    }

    pub(crate) async fn check_server_health(&self) -> ServerResult<()> {
        if !self.server_group.read().await.is_empty() {
            let mut unhealthy_servers = Vec::new();

            // Check health status of downstream servers
            // 1. Get all registered downstream servers
            // 2. Check health status of downstream servers
            //   2.1 If a downstream server has multiple types, only perform one health check
            //   2.2 If there are multiple downstream servers of the same type, health checks are needed for all
            //   2.3 If two or more downstream servers have different types but the same URL, only perform one health check
            // 3. Remove unhealthy downstream servers
            {
                let group_map = self.server_group.read().await;

                // check health of unique servers
                let mut unique_server_ids = HashSet::new();
                for (kind, group) in group_map.iter() {
                    if !group.is_empty().await {
                        let servers = group.servers.read().await;
                        for server_lock in servers.iter() {
                            let mut server = server_lock.write().await;

                            if !unique_server_ids.contains(&server.id)
                                && unique_server_ids.contains(&server.url)
                            {
                                info!(target: "stdout", "Checking health of {}", &server.id);

                                unique_server_ids.insert(server.id.clone());
                                unique_server_ids.insert(server.url.clone());

                                let is_healthy = server.check_health().await;
                                if !is_healthy {
                                    warn!(
                                        target: "stdout",
                                        message = format!("{} server {} is unhealthy", kind, &server.id)
                                    );
                                    unhealthy_servers.push(server.id.clone());
                                }
                            }
                        }
                    }
                }
            }

            // Unregister unhealthy servers
            if !unhealthy_servers.is_empty() {
                for server_id in unhealthy_servers {
                    self.unregister_downstream_server(&server_id).await?;
                }
            }

            // Push the healthy servers to the external service if configured
            if let Some(push_url) = &self.config.read().await.server_health_push_url {
                // collect the healthy servers by kind
                let mut healthy_servers: HashMap<ServerKind, Vec<String>> = HashMap::new();
                {
                    let group_map = self.server_group.read().await;
                    for (kind, group) in group_map.iter() {
                        if group.is_empty().await {
                            warn!(target: "stdout", "No {} servers available after health check", kind);
                        }

                        healthy_servers.insert(
                            *kind,
                            group.healthy_servers.read().await.iter().cloned().collect(),
                        );
                    }
                }

                // Send the healthy servers to the external service
                reqwest::Client::new()
                    .post(push_url)
                    .json(&healthy_servers)
                    .send()
                    .await
                    .map_err(|e| {
                        let err_msg = format!("Failed to send health check result: {}", e);

                        error!(target: "stdout", "{}", err_msg);

                        ServerError::Operation(err_msg)
                    })?;
            }
        } else {
            warn!(target: "stdout", "No servers registered, skipping health check");
        }

        Ok(())
    }

    pub(crate) async fn start_health_check_task(self: Arc<Self>) {
        let check_interval = HEALTH_CHECK_INTERVAL.get().unwrap_or(&60);
        let check_interval = tokio::time::Duration::from_secs(*check_interval);

        tokio::spawn(async move {
            loop {
                debug!(target: "stdout", "Starting health check");

                if let Err(e) = self.check_server_health().await {
                    error!(
                        target: "stdout",
                        message = format!("Health check error: {}", e)
                    );
                }

                tokio::time::sleep(check_interval).await;
            }
        });
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
