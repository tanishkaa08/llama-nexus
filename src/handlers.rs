use crate::{
    error::{ServerError, ServerResult},
    server::{RoutingPolicy, Server},
    AppState,
};
use axum::{
    body::Body,
    extract::State,
    http::{HeaderMap, Response, StatusCode},
    Json,
};
use endpoints::{chat::ChatCompletionRequest, embeddings::EmbeddingRequest};
use std::sync::Arc;
use tracing::{error, info};

pub(crate) async fn chat_handler(
    State(state): State<Arc<AppState>>,
    _headers: HeaderMap,
    Json(request): Json<ChatCompletionRequest>,
) -> ServerResult<axum::response::Response> {
    info!(target: "stdout", "handling chat request");

    let chat_servers = state.chat_servers.read().await;
    let chat_server_base_url = match chat_servers.next().await {
        Ok(url) => url,
        Err(e) => {
            let err_msg = e.to_string();

            error!(target: "stdout", "{}", &err_msg);

            return Err(ServerError::Operation(err_msg));
        }
    };
    let chat_service_url = format!("{}v1/chat/completions", chat_server_base_url);
    info!(target: "stdout", "dispatch the chat request to {}", chat_service_url);

    let response = reqwest::Client::new()
        .post(chat_service_url)
        .json(&request)
        .send()
        .await
        .map_err(|e| {
            let err_msg = format!(
                "failed to forward the request to the downstream server: {}",
                e
            );

            error!(target: "stdout", "{}", &err_msg);

            ServerError::Operation(err_msg)
        })?;

    let status = response.status();

    let bytes = response
        .bytes()
        .await
        .map_err(|e| ServerError::Operation(e.to_string()))?;

    match request.stream {
        Some(true) => Ok(Response::builder()
            .status(status)
            .header("Content-Type", "text/event-stream")
            .body(Body::from(bytes))
            .unwrap()),
        Some(false) | None => Ok(Response::builder()
            .status(status)
            .header("Content-Type", "application/json")
            .body(Body::from(bytes))
            .unwrap()),
    }
}

pub(crate) async fn embeddings_handler(
    State(state): State<Arc<AppState>>,
    _headers: HeaderMap,
    Json(request): Json<EmbeddingRequest>,
) -> ServerResult<axum::response::Response> {
    info!(target: "stdout", "handling embeddings request");

    let embeddings_servers = state.embeddings_servers.read().await;
    let embeddings_server_base_url = match embeddings_servers.next().await {
        Ok(url) => url,
        Err(e) => {
            let err_msg = e.to_string();
            error!(target: "stdout", "{}", &err_msg);
            return Err(ServerError::Operation(err_msg));
        }
    };
    let embeddings_service_url = format!("{}v1/embeddings", embeddings_server_base_url);
    info!(target: "stdout", "dispatch the embeddings request to {}", embeddings_service_url);

    let response = reqwest::Client::new()
        .post(embeddings_service_url)
        .json(&request)
        .send()
        .await
        .map_err(|e| ServerError::Operation(e.to_string()))?;

    let status = response.status();

    let bytes = response
        .bytes()
        .await
        .map_err(|e| ServerError::Operation(e.to_string()))?;

    Ok(Response::builder()
        .status(status)
        .header("Content-Type", "application/json")
        .body(Body::from(bytes))
        .unwrap())
}

pub(crate) async fn register_downstream_server_handler(
    State(state): State<Arc<AppState>>,
    Json(server): Json<Server>,
) -> ServerResult<axum::response::Response> {
    let server_url = server.url.clone();
    let server_kind = server.kind;

    state.register_downstream_server(server).await?;

    // create a response with status code 200. Content-Type is JSON
    let json_body = serde_json::json!({
        "message": "URL registered successfully",
        "url": server_url,
        "kind": server_kind
    });

    let response = Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "application/json")
        .body(Body::from(json_body.to_string()))
        .unwrap();

    Ok(response)
}

pub(crate) async fn remove_downstream_server_handler(
    State(state): State<Arc<AppState>>,
    Json(server): Json<Server>,
) -> ServerResult<axum::response::Response> {
    let server_url = server.url.clone();

    state.unregister_downstream_server(server).await?;

    // create a response with status code 200. Content-Type is JSON
    let json_body = serde_json::json!({
        "message": "URL unregistered successfully",
        "url": server_url
    });

    let response = Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "application/json")
        .body(Body::from(json_body.to_string()))
        .unwrap();

    Ok(response)
}

pub(crate) async fn list_downstream_servers_handler(
    State(state): State<Arc<AppState>>,
) -> ServerResult<axum::response::Response> {
    let servers = state.list_downstream_servers().await?;

    let json_body = serde_json::json!({
        "servers": servers
    });

    let response = Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "application/json")
        .body(Body::from(json_body.to_string()))
        .map_err(|e| ServerError::Operation(e.to_string()))?;

    Ok(response)
}
