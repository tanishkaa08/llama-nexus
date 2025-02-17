use crate::error::{ServerError, ServerResult};
use async_trait::async_trait;
use axum::http::Uri;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::RwLock;
use tracing::{error, warn};

/// Represents a LlamaEdge API server
#[derive(Debug, Serialize)]
pub struct Server {
    pub url: String,
    pub kind: ServerKind,
    #[serde(skip)]
    connections: AtomicUsize,
}
impl<'de> Deserialize<'de> for Server {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        // Create a helper struct to deserialize into
        #[derive(Deserialize)]
        struct ServerHelper {
            url: String,
            kind: ServerKind,
        }

        // Deserialize into the helper struct
        let helper = ServerHelper::deserialize(deserializer)?;

        // Create the actual Server instance
        Ok(Server {
            url: helper.url,
            kind: helper.kind,
            connections: AtomicUsize::new(0),
        })
    }
}

#[test]
fn test_deserialize_server() {
    let serialized = r#"{"url": "http://localhost:8000", "kind": "chat"}"#;
    let server: Server = serde_json::from_str(serialized).unwrap();
    assert_eq!(server.url, "http://localhost:8000");
    assert_eq!(server.kind, ServerKind::CHAT);
}
#[test]
fn test_serialize_server() {
    let server = Server {
        url: "http://localhost:8000".to_string(),
        kind: ServerKind::CHAT,
        connections: AtomicUsize::new(0),
    };
    let serialized = serde_json::to_string(&server).unwrap();
    assert_eq!(
        serialized,
        r#"{"url":"http://localhost:8000","kind":"chat"}"#
    );
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ServerKind {
    #[serde(rename = "chat")]
    CHAT,
    #[serde(rename = "embeddings")]
    EMBEDDINGS,
    #[serde(rename = "image")]
    IMAGE,
    #[serde(rename = "translate")]
    TRANSLATE,
    #[serde(rename = "transcript")]
    TRANSCRIPT,
    #[serde(rename = "tts")]
    TTS,
}
impl std::fmt::Display for ServerKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ServerKind::CHAT => write!(f, "chat"),
            ServerKind::EMBEDDINGS => write!(f, "embeddings"),
            ServerKind::IMAGE => write!(f, "image"),
            ServerKind::TRANSLATE => write!(f, "translate"),
            ServerKind::TRANSCRIPT => write!(f, "transcript"),
            ServerKind::TTS => write!(f, "tts"),
        }
    }
}
impl std::str::FromStr for ServerKind {
    type Err = ServerError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.to_lowercase();

        match s.as_str() {
            "chat" => Ok(Self::CHAT),
            "embeddings" => Ok(Self::EMBEDDINGS),
            "image" => Ok(Self::IMAGE),
            "translate" => Ok(Self::TRANSLATE),
            "transcript" => Ok(Self::TRANSCRIPT),
            "tts" => Ok(Self::TTS),
            _ => Err(ServerError::InvalidServerKind(s)),
        }
    }
}

#[test]
fn test_serialize_server_kind() {
    let kind = ServerKind::CHAT;
    let serialized = serde_json::to_string(&kind).unwrap();
    assert_eq!(serialized, "\"chat\"");
}

#[test]
fn test_deserialize_server_kind() {
    let serialized = "\"chat\"";
    let kind: ServerKind = serde_json::from_str(serialized).unwrap();
    assert_eq!(kind, ServerKind::CHAT);
}

#[derive(Debug)]
pub(crate) struct ServerGroup {
    pub(crate) servers: RwLock<Vec<Server>>,
    pub(crate) ty: ServerKind,
}
impl ServerGroup {
    pub(crate) fn new(ty: ServerKind) -> Self {
        Self {
            servers: RwLock::new(Vec::new()),
            ty,
        }
    }

    pub(crate) async fn register(&mut self, server: Server) -> ServerResult<()> {
        self.push(server).await
    }

    pub(crate) async fn push(&mut self, server: Server) -> ServerResult<()> {
        // check if the server is already registered
        if self
            .servers
            .read()
            .await
            .iter()
            .any(|s| s.url == server.url)
        {
            let err_msg = format!("Server already registered: {}", server.url);

            warn!(target: "stdout", "{}", &err_msg);

            return Err(ServerError::Operation(err_msg));
        }

        self.servers.write().await.push(server);

        Ok(())
    }

    pub(crate) async fn unregister(&mut self, server: Server) -> ServerResult<()> {
        // check if the server is registered
        if !self
            .servers
            .read()
            .await
            .iter()
            .any(|s| s.url == server.url)
        {
            let err_msg = format!("Server not found: {}", server.url);

            error!(target: "stdout", "{}", &err_msg);

            return Err(ServerError::NotFoundServer(err_msg));
        }

        self.servers.write().await.retain(|s| s.url != server.url);
        Ok(())
    }

    pub(crate) async fn list_servers(&self) -> Vec<String> {
        self.servers
            .read()
            .await
            .iter()
            .map(|s| s.url.clone())
            .collect()
    }
}
#[async_trait]
impl RoutingPolicy for ServerGroup {
    async fn next(&self) -> Result<Uri, ServerError> {
        if self.servers.read().await.is_empty() {
            return Err(ServerError::NotFoundServer(self.ty.to_string()));
        }

        let servers = self.servers.read().await;
        let server = if servers.len() == 1 {
            servers.first().unwrap()
        } else {
            servers
                .iter()
                .min_by(|s1, s2| {
                    s1.connections
                        .load(Ordering::Relaxed)
                        .cmp(&s2.connections.load(Ordering::Relaxed))
                })
                .unwrap()
        };

        server.connections.fetch_add(1, Ordering::Relaxed);
        Ok(server.url.parse().unwrap())
    }
}

#[async_trait]
pub(crate) trait RoutingPolicy: Sync + Send {
    async fn next(&self) -> Result<Uri, ServerError>;
}
