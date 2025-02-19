use crate::{
    error::{ServerError, ServerResult},
    server::{RoutingPolicy, Server},
    AppState,
};
use axum::{
    body::Body,
    extract::{Extension, State},
    http::{Response, StatusCode},
    Json,
};
use endpoints::chat::ChatCompletionRequest;
use std::sync::Arc;
use tokio::select;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};

pub(crate) async fn chat_handler(
    State(state): State<Arc<AppState>>,
    Extension(cancel_token): Extension<CancellationToken>,
    Json(request): Json<ChatCompletionRequest>,
) -> ServerResult<axum::response::Response> {
    debug!(target: "stdout", "Received a new chat request");

    let chat_servers = state.chat_servers.read().await;
    let chat_server_base_url = match chat_servers.next().await {
        Ok(url) => url,
        Err(e) => {
            let err_msg = format!("Failed to get the chat server: {}", e);
            error!(target: "stdout", "{}", &err_msg);
            return Err(ServerError::Operation(err_msg));
        }
    };

    let chat_service_url = format!("{}v1/chat/completions", chat_server_base_url);
    info!(target: "stdout", "Forward the chat request to {}", chat_service_url);

    let stream = request.stream;

    // Create a request client that can be cancelled
    let client = reqwest::Client::new();
    let request = client.post(chat_service_url).json(&request);

    // Use select! to handle request cancellation
    let response = select! {
        response = request.send() => {
            response.map_err(|e| {
                let err_msg = format!(
                    "Failed to forward the request to the downstream server: {}",
                    e
                );
                error!(target: "stdout", "{}", &err_msg);
                ServerError::Operation(err_msg)
            })?
        }
        _ = cancel_token.cancelled() => {
            let warn_msg = "Request was cancelled by client";
            warn!(target: "stdout", "{}", &warn_msg);
            return Err(ServerError::Operation(warn_msg.to_string()));
        }
    };

    let status = response.status();

    // Handle response body reading with cancellation
    let bytes = select! {
        bytes = response.bytes() => {
            bytes.map_err(|e| {
                let err_msg = format!("Failed to get the full response as bytes: {}", e);
                error!(target: "stdout", "{}", &err_msg);
                ServerError::Operation(err_msg)
            })?
        }
        _ = cancel_token.cancelled() => {
            let warn_msg = "Request was cancelled while reading response";
            warn!(target: "stdout", "{}", &warn_msg);
            return Err(ServerError::Operation(warn_msg.to_string()));
        }
    };

    match stream {
        Some(true) => {
            match Response::builder()
                .status(status)
                .header("Content-Type", "text/event-stream")
                .body(Body::from(bytes))
            {
                Ok(response) => {
                    info!(target: "stdout", "Handled the chat request");
                    Ok(response)
                }
                Err(e) => {
                    let err_msg = format!("Failed to create the response: {}", e);

                    error!(target: "stdout", "{}", &err_msg);

                    Err(ServerError::Operation(err_msg))
                }
            }
        }
        Some(false) | None => {
            match Response::builder()
                .status(status)
                .header("Content-Type", "application/json")
                .body(Body::from(bytes))
            {
                Ok(response) => {
                    info!(target: "stdout", "Handled the chat request");
                    Ok(response)
                }
                Err(e) => {
                    let err_msg = format!("Failed to create the response: {}", e);

                    error!(target: "stdout", "{}", &err_msg);

                    Err(ServerError::Operation(err_msg))
                }
            }
        }
    }
}

pub(crate) async fn embeddings_handler(
    State(state): State<Arc<AppState>>,
    Extension(cancel_token): Extension<CancellationToken>,
    req: axum::extract::Request<Body>,
) -> ServerResult<axum::response::Response> {
    debug!(target: "stdout", "Received a new embeddings request");

    let embeddings_servers = state.embeddings_servers.read().await;
    let embeddings_server_base_url = match embeddings_servers.next().await {
        Ok(url) => url,
        Err(e) => {
            let err_msg = format!("Failed to get the embeddings server: {}", e);
            error!(target: "stdout", "{}", &err_msg);
            return Err(ServerError::Operation(err_msg));
        }
    };
    let embeddings_service_url = format!("{}v1/embeddings", embeddings_server_base_url);
    info!(target: "stdout", "Forward the embeddings request to {}", embeddings_service_url);

    // parse the content-type header
    let content_type = &req
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| {
            let err_msg = "Missing Content-Type header".to_string();

            error!(target: "stdout", "{}", &err_msg);

            ServerError::Operation(err_msg)
        })?;
    let content_type = content_type.to_string();
    debug!(target: "stdout", "content-type: {}", &content_type);

    // convert the request body into bytes
    let body = req.into_body();
    let body_bytes = axum::body::to_bytes(body, usize::MAX).await.map_err(|e| {
        let err_msg = format!("Failed to convert the request body into bytes: {}", e);

        error!(target: "stdout", "{}", &err_msg);

        ServerError::Operation(err_msg)
    })?;

    // Create request client
    let client = reqwest::Client::new();
    let request = client
        .post(embeddings_service_url)
        .header("Content-Type", content_type)
        .body(body_bytes);

    // Use select! to handle request cancellation
    let response = select! {
        response = request.send() => {
            response.map_err(|e| {
                let err_msg = format!(
                    "Failed to forward the request to the downstream server: {}",
                    e
                );
                error!(target: "stdout", "{}", &err_msg);
                ServerError::Operation(err_msg)
            })?
        }
        _ = cancel_token.cancelled() => {
            let warn_msg = "Request was cancelled by client";
            warn!(target: "stdout", "{}", warn_msg);
            return Err(ServerError::Operation(warn_msg.to_string()));
        }
    };

    let status = response.status();

    // Handle response body reading with cancellation
    let bytes = select! {
        bytes = response.bytes() => {
            bytes.map_err(|e| {
                let err_msg = format!("Failed to get the full response as bytes: {}", e);
                error!(target: "stdout", "{}", &err_msg);
                ServerError::Operation(err_msg)
            })?
        }
        _ = cancel_token.cancelled() => {
            info!(target: "stdout", "Request was cancelled while reading response");
            return Err(ServerError::Operation("Request cancelled by client".to_string()));
        }
    };

    match Response::builder()
        .status(status)
        .header("Content-Type", "application/json")
        .body(Body::from(bytes))
    {
        Ok(response) => {
            info!(target: "stdout", "Handled the embeddings request");
            Ok(response)
        }
        Err(e) => {
            let err_msg = format!("Failed to create the response: {}", e);

            error!(target: "stdout", "{}", &err_msg);

            Err(ServerError::Operation(err_msg))
        }
    }
}

pub(crate) async fn audio_transcriptions_handler(
    State(state): State<Arc<AppState>>,
    Extension(cancel_token): Extension<CancellationToken>,
    req: axum::extract::Request<Body>,
) -> ServerResult<axum::response::Response> {
    debug!(target: "stdout", "Received a new audio transcription request");

    let whisper_servers = state.whisper_servers.read().await;
    let whisper_server_base_url = match whisper_servers.next().await {
        Ok(url) => url,
        Err(e) => {
            let err_msg = format!("Failed to get the whisper server: {}", e);

            error!(target: "stdout", "{}", &err_msg);

            return Err(ServerError::Operation(err_msg));
        }
    };
    let transcription_service_url = format!("{}v1/audio/transcriptions", whisper_server_base_url);
    info!(
        target: "stdout",
        "Forward the audio transcription request to {}",
        transcription_service_url
    );

    // parse the content-type header
    let content_type = &req
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| {
            let err_msg = "Missing Content-Type header".to_string();

            error!(target: "stdout", "{}", &err_msg);

            ServerError::Operation(err_msg)
        })?;
    let content_type = content_type.to_string();
    debug!(target: "stdout", "content-type: {}", &content_type);

    // convert the request body into bytes
    let body = req.into_body();
    let body_bytes = axum::body::to_bytes(body, usize::MAX).await.map_err(|e| {
        let err_msg = format!("Failed to convert the request body into bytes: {}", e);

        error!(target: "stdout", "{}", &err_msg);

        ServerError::Operation(err_msg)
    })?;

    // Create request client
    let client = reqwest::Client::new();
    let request = client
        .post(transcription_service_url)
        .header("Content-Type", content_type)
        .body(body_bytes);

    // Use select! to handle request cancellation
    let response = select! {
        response = request.send() => {
            response.map_err(|e| {
                let err_msg = format!(
                    "Failed to forward the request to the downstream server: {}",
                    e
                );
                error!(target: "stdout", "{}", &err_msg);
                ServerError::Operation(err_msg)
            })?
        }
        _ = cancel_token.cancelled() => {
            let warn_msg = "Request was cancelled by client";
            warn!(target: "stdout", "{}", &warn_msg);
            return Err(ServerError::Operation(warn_msg.to_string()));
        }
    };

    let status = response.status();

    // Handle response body reading with cancellation
    let bytes = select! {
        bytes = response.bytes() => {
            bytes.map_err(|e| {
                let err_msg = format!("Failed to get the full response as bytes: {}", e);
                error!(target: "stdout", "{}", &err_msg);
                ServerError::Operation(err_msg)
            })?
        }
        _ = cancel_token.cancelled() => {
            let warn_msg = "Request was cancelled while reading response";
            warn!(target: "stdout", "{}", &warn_msg);
            return Err(ServerError::Operation(warn_msg.to_string()));
        }
    };

    // create the response
    match Response::builder()
        .status(status)
        .header("Content-Type", "application/json")
        .body(Body::from(bytes))
    {
        Ok(response) => {
            info!(target: "stdout", "Handled the audio transcription request");
            Ok(response)
        }
        Err(e) => {
            let err_msg = format!("Failed to create the response: {}", e);

            error!(target: "stdout", "{}", &err_msg);

            Err(ServerError::Operation(err_msg))
        }
    }
}

pub(crate) async fn audio_translations_handler(
    State(state): State<Arc<AppState>>,
    Extension(cancel_token): Extension<CancellationToken>,
    req: axum::extract::Request<Body>,
) -> ServerResult<axum::response::Response> {
    debug!(target: "stdout", "Received a new audio translation request");

    let whisper_servers = state.whisper_servers.read().await;
    let whisper_server_base_url = match whisper_servers.next().await {
        Ok(url) => url,
        Err(e) => {
            let err_msg = format!("Failed to get the whisper server: {}", e);

            error!(target: "stdout", "{}", &err_msg);

            return Err(ServerError::Operation(err_msg));
        }
    };
    let translation_service_url = format!("{}v1/audio/translations", whisper_server_base_url);
    info!(
        target: "stdout",
        "Forward the audio translation request to {}",
        translation_service_url
    );

    // parse the content-type header
    let content_type = &req
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| {
            let err_msg = "Missing Content-Type header".to_string();

            error!(target: "stdout", "{}", &err_msg);

            ServerError::Operation(err_msg)
        })?;
    let content_type = content_type.to_string();
    debug!(target: "stdout", "content-type: {}", &content_type);

    // convert the request body into bytes
    let body = req.into_body();
    let body_bytes = axum::body::to_bytes(body, usize::MAX).await.map_err(|e| {
        let err_msg = format!("Failed to convert the request body into bytes: {}", e);

        error!(target: "stdout", "{}", &err_msg);

        ServerError::Operation(err_msg)
    })?;

    // Create request client
    let client = reqwest::Client::new();
    let request = client
        .post(translation_service_url)
        .header("Content-Type", content_type)
        .body(body_bytes);

    // Use select! to handle request cancellation
    let response = select! {
        response = request.send() => {
            response.map_err(|e| {
                let err_msg = format!(
                    "Failed to forward the request to the downstream server: {}",
                    e
                );
                error!(target: "stdout", "{}", &err_msg);
                ServerError::Operation(err_msg)
            })?
        }
        _ = cancel_token.cancelled() => {
            let warn_msg = "Request was cancelled by client";
            warn!(target: "stdout", "{}", &warn_msg);
            return Err(ServerError::Operation(warn_msg.to_string()));
        }
    };

    let status = response.status();

    // Handle response body reading with cancellation
    let bytes = select! {
        bytes = response.bytes() => {
            bytes.map_err(|e| {
                let err_msg = format!("Failed to get the full response as bytes: {}", e);
                error!(target: "stdout", "{}", &err_msg);
                ServerError::Operation(err_msg)
            })?
        }
        _ = cancel_token.cancelled() => {
            let warn_msg = "Request was cancelled while reading response";
            warn!(target: "stdout", "{}", &warn_msg);
            return Err(ServerError::Operation(warn_msg.to_string()));
        }
    };

    // create the response
    match Response::builder()
        .status(status)
        .header("Content-Type", "application/json")
        .body(Body::from(bytes))
    {
        Ok(response) => {
            info!(target: "stdout", "Handled the audio translation request");
            Ok(response)
        }
        Err(e) => {
            let err_msg = format!("Failed to create the response: {}", e);

            error!(target: "stdout", "{}", &err_msg);

            Err(ServerError::Operation(err_msg))
        }
    }
}

pub(crate) async fn audio_tts_handler(
    State(state): State<Arc<AppState>>,
    Extension(cancel_token): Extension<CancellationToken>,
    req: axum::extract::Request<Body>,
) -> ServerResult<axum::response::Response> {
    debug!(target: "stdout", "Received a new audio speech request");

    let tts_servers = state.tts_servers.read().await;
    let tts_server_base_url = match tts_servers.next().await {
        Ok(url) => url,
        Err(e) => {
            let err_msg = format!("Failed to get the tts server: {}", e);

            error!(target: "stdout", "{}", &err_msg);

            return Err(ServerError::Operation(err_msg));
        }
    };
    let tts_service_url = format!("{}v1/audio/speech", tts_server_base_url);
    info!(
        target: "stdout",
        "Forward the audio speech request to {}",
        tts_service_url
    );

    let body = req.into_body();
    let body_bytes = axum::body::to_bytes(body, usize::MAX).await.map_err(|e| {
        let err_msg = format!("Failed to convert the request body into bytes: {}", e);

        error!(target: "stdout", "{}", &err_msg);

        ServerError::Operation(err_msg)
    })?;

    // Create request client
    let client = reqwest::Client::new();
    let request = client
        .post(tts_service_url)
        .header("Content-Type", "application/json")
        .body(body_bytes);

    // Use select! to handle request cancellation
    let response = select! {
        response = request.send() => {
            response.map_err(|e| {
                let err_msg = format!(
                    "Failed to forward the request to the downstream server: {}",
                    e
                );
                error!(target: "stdout", "{}", &err_msg);
                ServerError::Operation(err_msg)
            })?
        }
        _ = cancel_token.cancelled() => {
            let warn_msg = "Request was cancelled by client";
            warn!(target: "stdout", "{}", &warn_msg);
            return Err(ServerError::Operation(warn_msg.to_string()));
        }
    };

    let status = response.status();

    // Handle response body reading with cancellation
    let bytes = select! {
        bytes = response.bytes() => {
            bytes.map_err(|e| {
                let err_msg = format!("Failed to get the full response as bytes: {}", e);
                error!(target: "stdout", "{}", &err_msg);
                ServerError::Operation(err_msg)
            })?
        }
        _ = cancel_token.cancelled() => {
            let warn_msg = "Request was cancelled while reading response";
            warn!(target: "stdout", "{}", &warn_msg);
            return Err(ServerError::Operation(warn_msg.to_string()));
        }
    };

    match Response::builder()
        .status(status)
        .header("Content-Type", "application/json")
        .body(Body::from(bytes))
    {
        Ok(response) => {
            info!(target: "stdout", "Handled the audio speech request");
            Ok(response)
        }
        Err(e) => {
            let err_msg = format!("Failed to create the response: {}", e);

            error!(target: "stdout", "{}", &err_msg);

            Err(ServerError::Operation(err_msg))
        }
    }
}

pub(crate) async fn image_handler(
    State(state): State<Arc<AppState>>,
    Extension(cancel_token): Extension<CancellationToken>,
    req: axum::extract::Request<Body>,
) -> ServerResult<axum::response::Response> {
    debug!(target: "stdout", "Received a new image request");

    let image_servers = state.image_servers.read().await;
    let image_server_base_url = match image_servers.next().await {
        Ok(url) => url,
        Err(e) => {
            let err_msg = format!("Failed to get the image server: {}", e);

            error!(target: "stdout", "{}", &err_msg);

            return Err(ServerError::Operation(err_msg));
        }
    };
    let image_service_url = format!("{}v1/images/generations", image_server_base_url);
    info!(
        target: "stdout",
        "Forward the image request to {}",
        image_service_url
    );

    // parse the content-type header
    let content_type = &req
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| {
            let err_msg = "Missing Content-Type header".to_string();

            error!(target: "stdout", "{}", &err_msg);

            ServerError::Operation(err_msg)
        })?;
    let content_type = content_type.to_string();
    debug!(target: "stdout", "content-type: {}", &content_type);

    // convert the request body into bytes
    let body = req.into_body();
    let body_bytes = axum::body::to_bytes(body, usize::MAX).await.map_err(|e| {
        let err_msg = format!("Failed to convert the request body into bytes: {}", e);

        error!(target: "stdout", "{}", &err_msg);

        ServerError::Operation(err_msg)
    })?;

    // Create request client
    let client = reqwest::Client::new();
    let request = client
        .post(image_service_url)
        .header("Content-Type", content_type)
        .body(body_bytes);

    // Use select! to handle request cancellation
    let response = select! {
        response = request.send() => {
            response.map_err(|e| {
                let err_msg = format!(
                    "Failed to forward the request to the downstream server: {}",
                    e
                );
                error!(target: "stdout", "{}", &err_msg);
                ServerError::Operation(err_msg)
            })?
        }
        _ = cancel_token.cancelled() => {
            let warn_msg = "Request was cancelled by client";
            warn!(target: "stdout", "{}", &warn_msg);
            return Err(ServerError::Operation(warn_msg.to_string()));
        }
    };

    let status = response.status();

    // Handle response body reading with cancellation
    let bytes = select! {
        bytes = response.bytes() => {
            bytes.map_err(|e| {
                let err_msg = format!("Failed to get the full response as bytes: {}", e);
                error!(target: "stdout", "{}", &err_msg);
                ServerError::Operation(err_msg)
            })?
        }
        _ = cancel_token.cancelled() => {
            let warn_msg = "Request was cancelled while reading response";
            warn!(target: "stdout", "{}", &warn_msg);
            return Err(ServerError::Operation(warn_msg.to_string()));
        }
    };

    // create the response
    match Response::builder()
        .status(status)
        .header("Content-Type", "application/json")
        .body(Body::from(bytes))
    {
        Ok(response) => {
            info!(target: "stdout", "Handled the image request");
            Ok(response)
        }
        Err(e) => {
            let err_msg = format!("Failed to create the response: {}", e);

            error!(target: "stdout", "{}", &err_msg);

            Err(ServerError::Operation(err_msg))
        }
    }
}

pub(crate) async fn register_downstream_server_handler(
    State(state): State<Arc<AppState>>,
    Json(server): Json<Server>,
) -> ServerResult<axum::response::Response> {
    let server_url = server.url.clone();
    let server_kind = server.kind;

    state.register_downstream_server(server).await?;
    info!(target: "stdout", "Registered {} server: {}", server_kind, server_url);

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
    let server_kind = server.kind;
    let server_url = server.url.clone();

    state.unregister_downstream_server(server).await?;
    info!(target: "stdout", "Unregistered {} server: {}", server_kind, server_url);

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
    info!(target: "stdout", "Found {} downstream servers", servers.len());

    let json_body = serde_json::json!({
        "servers": servers
    });

    let response = Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "application/json")
        .body(Body::from(json_body.to_string()))
        .map_err(|e| {
            let err_msg = format!("Failed to create the response: {}", e);

            error!(target: "stdout", "{}", &err_msg);

            ServerError::Operation(err_msg)
        })?;

    Ok(response)
}
