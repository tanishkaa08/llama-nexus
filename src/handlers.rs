use crate::{
    error::{ServerError, ServerResult},
    info::ApiServer,
    rag,
    server::{RoutingPolicy, Server, ServerIdToRemove, ServerKind},
    AppState,
};
use axum::{
    body::Body,
    extract::{Extension, State},
    http::{HeaderMap, Response, StatusCode},
    Json,
};
use endpoints::{
    chat::ChatCompletionRequest,
    embeddings::{EmbeddingRequest, EmbeddingsResponse},
    models::ListModelsResponse,
};
use std::{sync::Arc, time::SystemTime};
use tokio::select;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};

pub(crate) async fn chat_handler(
    State(state): State<Arc<AppState>>,
    Extension(cancel_token): Extension<CancellationToken>,
    headers: HeaderMap,
    Json(request): Json<ChatCompletionRequest>,
) -> ServerResult<axum::response::Response> {
    let enable_rag = state.config.read().await.rag.enable;
    match enable_rag {
        true => {
            rag::chat(
                State(state),
                Extension(cancel_token),
                headers,
                Json(request),
            )
            .await
        }
        false => {
            chat(
                State(state),
                Extension(cancel_token),
                headers,
                Json(request),
            )
            .await
        }
    }
}

pub(crate) async fn chat(
    State(state): State<Arc<AppState>>,
    Extension(cancel_token): Extension<CancellationToken>,
    headers: HeaderMap,
    Json(request): Json<ChatCompletionRequest>,
) -> ServerResult<axum::response::Response> {
    let request_id = headers
        .get("x-request-id")
        .and_then(|h| h.to_str().ok())
        .unwrap_or("unknown")
        .to_string();

    info!(
        target: "stdout",
        request_id = %request_id,
        message = "Received a new chat request"
    );

    // get the chat server
    let chat_server_base_url = {
        let servers = state.group_map.read().await;
        let chat_servers = match servers.get(&ServerKind::chat) {
            Some(servers) => servers,
            None => {
                let err_msg = "No chat server available";
                error!(
                    target: "stdout",
                    request_id = %request_id,
                    message = %err_msg,
                );
                return Err(ServerError::Operation(err_msg.to_string()));
            }
        };

        match chat_servers.next().await {
            Ok(url) => url,
            Err(e) => {
                let err_msg = format!("Failed to get the chat server: {}", e);
                error!(
                    target: "stdout",
                    request_id = %request_id,
                    message = %err_msg,
                );
                return Err(ServerError::Operation(err_msg));
            }
        }
    };

    let chat_service_url = format!("{}v1/chat/completions", chat_server_base_url);
    info!(
        target: "stdout",
        request_id = %request_id,
        message = format!("Forward the chat request to {}", chat_service_url),
    );

    let stream = request.stream;

    // Create a request client that can be cancelled
    let request_builder = reqwest::Client::new()
        .post(chat_service_url)
        .header("content-type", "application/json")
        .json(&request);

    // Use select! to handle request cancellation
    let ds_response = select! {
        response = request_builder.send() => {
            response.map_err(|e| {
                let err_msg = format!(
                    "Failed to forward the request to the downstream server: {}",
                    e
                );
                error!(
                    target: "stdout",
                    request_id = %request_id,
                    message = %err_msg,
                );
                ServerError::Operation(err_msg)
            })?
        }
        _ = cancel_token.cancelled() => {
            let warn_msg = "Request was cancelled by client";
            warn!(
                target: "stdout",
                request_id = %request_id,
                message = %warn_msg,
            );
            return Err(ServerError::Operation(warn_msg.to_string()));
        }
    };

    let status = ds_response.status();

    // Handle response body reading with cancellation
    let bytes = select! {
        bytes = ds_response.bytes() => {
            bytes.map_err(|e| {
                let err_msg = format!("Failed to get the full response as bytes: {}", e);
                error!(
                    target: "stdout",
                    request_id = %request_id,
                    message = %err_msg,
                );
                ServerError::Operation(err_msg)
            })?
        }
        _ = cancel_token.cancelled() => {
            let warn_msg = "Request was cancelled while reading response";
            warn!(
                target: "stdout",
                request_id = %request_id,
                message = %warn_msg,
            );
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
                    info!(
                        target: "stdout",
                        request_id = %request_id,
                        message = "Chat request completed successfully",
                    );
                    Ok(response)
                }
                Err(e) => {
                    let err_msg = format!("Failed to create the response: {}", e);
                    error!(
                        target: "stdout",
                        request_id = %request_id,
                        message = %err_msg,
                    );
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
                    info!(
                        target: "stdout",
                        request_id = %request_id,
                        message = "Chat request completed successfully",
                    );
                    Ok(response)
                }
                Err(e) => {
                    let err_msg = format!("Failed to create the response: {}", e);
                    error!(
                        target: "stdout",
                        request_id = %request_id,
                        message = %err_msg,
                    );
                    Err(ServerError::Operation(err_msg))
                }
            }
        }
    }
}

pub(crate) async fn embeddings_handler(
    State(state): State<Arc<AppState>>,
    Extension(cancel_token): Extension<CancellationToken>,
    headers: HeaderMap,
    Json(request): Json<EmbeddingRequest>,
) -> ServerResult<axum::response::Response> {
    // Get request ID from headers
    let request_id = headers
        .get("x-request-id")
        .and_then(|h| h.to_str().ok())
        .unwrap_or("unknown")
        .to_string();

    info!(
        target: "stdout",
        request_id = %request_id,
        message = "Received a new embeddings request"
    );

    // get the embeddings server
    let servers = state.group_map.read().await;
    let embeddings_servers = match servers.get(&ServerKind::embeddings) {
        Some(servers) => servers,
        None => {
            let err_msg = "No embeddings server available";
            error!(
                target: "stdout",
                request_id = %request_id,
                message = %err_msg,
            );
            return Err(ServerError::Operation(err_msg.to_string()));
        }
    };

    let embeddings_server_base_url = match embeddings_servers.next().await {
        Ok(url) => url,
        Err(e) => {
            let err_msg = format!("Failed to get the embeddings server: {}", e);
            error!(
                target: "stdout",
                request_id = %request_id,
                message = %err_msg,
            );
            return Err(ServerError::Operation(err_msg));
        }
    };
    let embeddings_service_url = format!("{}v1/embeddings", embeddings_server_base_url);
    info!(
        target: "stdout",
        request_id = %request_id,
        message = format!("Forward the embeddings request to {}", embeddings_service_url),
    );

    // parse the content-type header
    let content_type = headers
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| {
            let err_msg = "Missing Content-Type header".to_string();
            error!(
                target: "stdout",
                request_id = %request_id,
                message = %err_msg,
            );
            ServerError::Operation(err_msg)
        })?;
    let content_type = content_type.to_string();
    debug!(
        target: "stdout",
        request_id = %request_id,
        message = format!("Request content type: {}", content_type)
    );

    // Create request client
    let client = reqwest::Client::new();
    let request = client
        .post(embeddings_service_url)
        .header("Content-Type", content_type)
        .json(&request);

    // Use select! to handle request cancellation
    let response = select! {
        response = request.send() => {
            response.map_err(|e| {
                let err_msg = format!(
                    "Failed to forward the request to the downstream server: {}",
                    e
                );
                error!(
                    target: "stdout",
                    request_id = %request_id,
                    message = %err_msg,
                );
                ServerError::Operation(err_msg)
            })?
        }
        _ = cancel_token.cancelled() => {
            let warn_msg = "Request was cancelled by client";
            warn!(
                target: "stdout",
                request_id = %request_id,
                message = %warn_msg,
            );
            return Err(ServerError::Operation(warn_msg.to_string()));
        }
    };

    let status = response.status();

    // Handle response body reading with cancellation
    let bytes = select! {
        bytes = response.bytes() => {
            bytes.map_err(|e| {
                let err_msg = format!("Failed to get the full response as bytes: {}", e);
                error!(
                    target: "stdout",
                    request_id = %request_id,
                    message = %err_msg,
                );
                ServerError::Operation(err_msg)
            })?
        }
        _ = cancel_token.cancelled() => {
            let warn_msg = "Request was cancelled while reading response";
            warn!(
                target: "stdout",
                request_id = %request_id,
                message = %warn_msg,
            );
            return Err(ServerError::Operation(warn_msg.to_string()));
        }
    };

    match Response::builder()
        .status(status)
        .header("Content-Type", "application/json")
        .body(Body::from(bytes))
    {
        Ok(response) => {
            info!(
                target: "stdout",
                request_id = %request_id,
                message = "Embeddings request completed successfully",
            );
            Ok(response)
        }
        Err(e) => {
            let err_msg = format!("Failed to create the response: {}", e);
            error!(
                target: "stdout",
                request_id = %request_id,
                message = %err_msg,
            );
            Err(ServerError::Operation(err_msg))
        }
    }
}

pub(crate) async fn audio_transcriptions_handler(
    State(state): State<Arc<AppState>>,
    Extension(cancel_token): Extension<CancellationToken>,
    req: axum::extract::Request<Body>,
) -> ServerResult<axum::response::Response> {
    // Get request ID from headers
    let request_id = req
        .headers()
        .get("x-request-id")
        .and_then(|h| h.to_str().ok())
        .unwrap_or("unknown")
        .to_string();

    info!(
        target: "stdout",
        request_id = %request_id,
        message = "Received a new audio transcription request"
    );

    // get the transcribe server
    let transcribe_server_base_url = {
        let servers = state.group_map.read().await;
        let transcribe_servers = match servers.get(&ServerKind::transcribe) {
            Some(servers) => servers,
            None => {
                let err_msg = "No transcribe server available";
                error!(
                    target: "stdout",
                    request_id = %request_id,
                    message = %err_msg,
                );
                return Err(ServerError::Operation(err_msg.to_string()));
            }
        };

        match transcribe_servers.next().await {
            Ok(url) => url,
            Err(e) => {
                let err_msg = format!("Failed to get the transcribe server: {}", e);
                error!(
                    target: "stdout",
                    request_id = %request_id,
                    message = %err_msg,
                );
                return Err(ServerError::Operation(err_msg));
            }
        }
    };

    let transcription_service_url =
        format!("{}v1/audio/transcriptions", transcribe_server_base_url);
    info!(
        target: "stdout",
        request_id = %request_id,
        message = format!("Forward the audio transcription request to {}", transcription_service_url),
    );

    // Create request client
    let mut request_builder = reqwest::Client::new().post(transcription_service_url);
    for (name, value) in req.headers().iter() {
        request_builder = request_builder.header(name, value);
    }

    // convert the request body into bytes
    let body = req.into_body();
    let body_bytes = axum::body::to_bytes(body, usize::MAX).await.map_err(|e| {
        let err_msg = format!("Failed to convert the request body into bytes: {}", e);
        error!(
            target: "stdout",
            request_id = %request_id,
            message = %err_msg,
        );
        ServerError::Operation(err_msg)
    })?;

    request_builder = request_builder.body(body_bytes);

    // Use select! to handle request cancellation
    let response = select! {
        response = request_builder.send() => {
            response.map_err(|e| {
                let err_msg = format!(
                    "Failed to forward the request to the downstream server: {}",
                    e
                );
                error!(
                    target: "stdout",
                    request_id = %request_id,
                    message = %err_msg,
                );
                ServerError::Operation(err_msg)
            })?
        }
        _ = cancel_token.cancelled() => {
            let warn_msg = "Request was cancelled by client";
            warn!(
                target: "stdout",
                request_id = %request_id,
                message = %warn_msg,
            );
            return Err(ServerError::Operation(warn_msg.to_string()));
        }
    };

    let status = response.status();

    // Handle response body reading with cancellation
    let bytes = select! {
        bytes = response.bytes() => {
            bytes.map_err(|e| {
                let err_msg = format!("Failed to get the full response as bytes: {}", e);
                error!(
                    target: "stdout",
                    request_id = %request_id,
                    message = %err_msg,
                );
                ServerError::Operation(err_msg)
            })?
        }
        _ = cancel_token.cancelled() => {
            let warn_msg = "Request was cancelled while reading response";
            warn!(
                target: "stdout",
                request_id = %request_id,
                message = %warn_msg,
            );
            return Err(ServerError::Operation(warn_msg.to_string()));
        }
    };

    match Response::builder()
        .status(status)
        .header("Content-Type", "application/json")
        .body(Body::from(bytes))
    {
        Ok(response) => {
            info!(
                target: "stdout",
                request_id = %request_id,
                message = "Audio transcription request completed successfully",
            );
            Ok(response)
        }
        Err(e) => {
            let err_msg = format!("Failed to create the response: {}", e);
            error!(
                target: "stdout",
                request_id = %request_id,
                message = %err_msg,
            );
            Err(ServerError::Operation(err_msg))
        }
    }
}

pub(crate) async fn audio_translations_handler(
    State(state): State<Arc<AppState>>,
    Extension(cancel_token): Extension<CancellationToken>,
    req: axum::extract::Request<Body>,
) -> ServerResult<axum::response::Response> {
    // Get request ID from headers
    let request_id = req
        .headers()
        .get("x-request-id")
        .and_then(|h| h.to_str().ok())
        .unwrap_or("unknown")
        .to_string();

    info!(
        target: "stdout",
        request_id = %request_id,
        message = "Received a new audio translation request"
    );

    // get the transcribe server
    let translate_server_base_url = {
        let servers = state.group_map.read().await;
        let translate_servers = match servers.get(&ServerKind::translate) {
            Some(servers) => servers,
            None => {
                let err_msg = "No translate server available";
                error!(
                    target: "stdout",
                    request_id = %request_id,
                    message = %err_msg,
                );
                return Err(ServerError::Operation(err_msg.to_string()));
            }
        };

        match translate_servers.next().await {
            Ok(url) => url,
            Err(e) => {
                let err_msg = format!("Failed to get the translate server: {}", e);
                error!(
                    target: "stdout",
                    request_id = %request_id,
                    message = %err_msg,
                );
                return Err(ServerError::Operation(err_msg));
            }
        }
    };

    let translation_service_url = format!("{}v1/audio/translations", translate_server_base_url);
    info!(
        target: "stdout",
        request_id = %request_id,
        message = format!("Forward the audio translation request to {}", translation_service_url),
    );

    // Create request client
    let mut request_builder = reqwest::Client::new().post(translation_service_url);
    for (name, value) in req.headers().iter() {
        request_builder = request_builder.header(name, value);
    }

    // convert the request body into bytes
    let body = req.into_body();
    let body_bytes = axum::body::to_bytes(body, usize::MAX).await.map_err(|e| {
        let err_msg = format!("Failed to convert the request body into bytes: {}", e);
        error!(
            target: "stdout",
            request_id = %request_id,
            message = %err_msg,
        );
        ServerError::Operation(err_msg)
    })?;

    request_builder = request_builder.body(body_bytes);

    // Use select! to handle request cancellation
    let response = select! {
        response = request_builder.send() => {
            response.map_err(|e| {
                let err_msg = format!(
                    "Failed to forward the request to the downstream server: {}",
                    e
                );
                error!(
                    target: "stdout",
                    request_id = %request_id,
                    message = %err_msg,
                );
                ServerError::Operation(err_msg)
            })?
        }
        _ = cancel_token.cancelled() => {
            let warn_msg = "Request was cancelled by client";
            warn!(
                target: "stdout",
                request_id = %request_id,
                message = %warn_msg,
            );
            return Err(ServerError::Operation(warn_msg.to_string()));
        }
    };

    let status = response.status();

    // Handle response body reading with cancellation
    let bytes = select! {
        bytes = response.bytes() => {
            bytes.map_err(|e| {
                let err_msg = format!("Failed to get the full response as bytes: {}", e);
                error!(
                    target: "stdout",
                    request_id = %request_id,
                    message = %err_msg,
                );
                ServerError::Operation(err_msg)
            })?
        }
        _ = cancel_token.cancelled() => {
            let warn_msg = "Request was cancelled while reading response";
            warn!(
                target: "stdout",
                request_id = %request_id,
                message = %warn_msg,
            );
            return Err(ServerError::Operation(warn_msg.to_string()));
        }
    };

    match Response::builder()
        .status(status)
        .header("Content-Type", "application/json")
        .body(Body::from(bytes))
    {
        Ok(response) => {
            info!(
                target: "stdout",
                request_id = %request_id,
                message = "Audio translation request completed successfully",
            );
            Ok(response)
        }
        Err(e) => {
            let err_msg = format!("Failed to create the response: {}", e);
            error!(
                target: "stdout",
                request_id = %request_id,
                message = %err_msg,
            );
            Err(ServerError::Operation(err_msg))
        }
    }
}

pub(crate) async fn audio_tts_handler(
    State(state): State<Arc<AppState>>,
    Extension(cancel_token): Extension<CancellationToken>,
    req: axum::extract::Request<Body>,
) -> ServerResult<axum::response::Response> {
    // Get request ID from headers
    let request_id = req
        .headers()
        .get("x-request-id")
        .and_then(|h| h.to_str().ok())
        .unwrap_or("unknown")
        .to_string();

    info!(
        target: "stdout",
        request_id = %request_id,
        message = "Received a new audio speech request"
    );

    // get the tts server
    let tts_server_base_url = {
        let servers = state.group_map.read().await;
        let tts_servers = match servers.get(&ServerKind::tts) {
            Some(servers) => servers,
            None => {
                let err_msg = "No tts server available";
                error!(
                    target: "stdout",
                    request_id = %request_id,
                    message = %err_msg,
                );
                return Err(ServerError::Operation(err_msg.to_string()));
            }
        };

        match tts_servers.next().await {
            Ok(url) => url,
            Err(e) => {
                let err_msg = format!("Failed to get the tts server: {}", e);
                error!(
                    target: "stdout",
                    request_id = %request_id,
                    message = %err_msg,
                );
                return Err(ServerError::Operation(err_msg));
            }
        }
    };

    let tts_service_url = format!("{}v1/audio/speech", tts_server_base_url);
    info!(
        target: "stdout",
        request_id = %request_id,
        message = format!("Forward the audio speech request to {}", tts_service_url),
    );

    // Create request client
    let mut request_builder = reqwest::Client::new().post(tts_service_url);
    for (name, value) in req.headers().iter() {
        request_builder = request_builder.header(name, value);
    }

    let body = req.into_body();
    let body_bytes = axum::body::to_bytes(body, usize::MAX).await.map_err(|e| {
        let err_msg = format!("Failed to convert the request body into bytes: {}", e);
        error!(
            target: "stdout",
            request_id = %request_id,
            message = %err_msg,
        );
        ServerError::Operation(err_msg)
    })?;

    request_builder = request_builder.body(body_bytes);

    // Use select! to handle request cancellation
    let ds_response = select! {
        response = request_builder.send() => {
            response.map_err(|e| {
                let err_msg = format!(
                    "Failed to forward the request to the downstream server: {}",
                    e
                );
                error!(
                    target: "stdout",
                    request_id = %request_id,
                    message = %err_msg,
                );
                ServerError::Operation(err_msg)
            })?
        }
        _ = cancel_token.cancelled() => {
            let warn_msg = "Request was cancelled by client";
            warn!(
                target: "stdout",
                request_id = %request_id,
                message = %warn_msg,
            );
            return Err(ServerError::Operation(warn_msg.to_string()));
        }
    };

    // create a response builder with the status and headers of the downstream response
    let mut response_builder = Response::builder().status(ds_response.status());
    for (name, value) in ds_response.headers().iter() {
        response_builder = response_builder.header(name, value);
    }

    // Handle response body reading with cancellation
    let bytes = select! {
        bytes = ds_response.bytes() => {
            bytes.map_err(|e| {
                let err_msg = format!("Failed to get the full response as bytes: {}", e);
                error!(
                    target: "stdout",
                    request_id = %request_id,
                    message = %err_msg,
                );
                ServerError::Operation(err_msg)
            })?
        }
        _ = cancel_token.cancelled() => {
            let warn_msg = "Request was cancelled while reading response";
            warn!(
                target: "stdout",
                request_id = %request_id,
                message = %warn_msg,
            );
            return Err(ServerError::Operation(warn_msg.to_string()));
        }
    };

    match response_builder.body(Body::from(bytes)) {
        Ok(response) => {
            info!(
                target: "stdout",
                request_id = %request_id,
                message = "Audio speech request completed successfully",
            );
            Ok(response)
        }
        Err(e) => {
            let err_msg = format!("Failed to create the response: {}", e);
            error!(
                target: "stdout",
                request_id = %request_id,
                message = %err_msg,
            );
            Err(ServerError::Operation(err_msg))
        }
    }
}

pub(crate) async fn image_handler(
    State(state): State<Arc<AppState>>,
    Extension(cancel_token): Extension<CancellationToken>,
    req: axum::extract::Request<Body>,
) -> ServerResult<axum::response::Response> {
    // Get request ID from headers
    let request_id = req
        .headers()
        .get("x-request-id")
        .and_then(|h| h.to_str().ok())
        .unwrap_or("unknown")
        .to_string();

    info!(
        target: "stdout",
        request_id = %request_id,
        message = "Received a new image request"
    );

    // get the image server
    let image_server_base_url = {
        let servers = state.group_map.read().await;
        let image_servers = match servers.get(&ServerKind::image) {
            Some(servers) => servers,
            None => {
                let err_msg = "No image server available";
                error!(
                    target: "stdout",
                    request_id = %request_id,
                    message = %err_msg,
                );
                return Err(ServerError::Operation(err_msg.to_string()));
            }
        };

        match image_servers.next().await {
            Ok(url) => url,
            Err(e) => {
                let err_msg = format!("Failed to get the image server: {}", e);
                error!(
                    target: "stdout",
                    request_id = %request_id,
                    message = %err_msg,
                );
                return Err(ServerError::Operation(err_msg));
            }
        }
    };

    let image_service_url = format!("{}v1/images/generations", image_server_base_url);
    info!(
        target: "stdout",
        request_id = %request_id,
        message = format!("Forward the image request to {}", image_service_url),
    );

    // Create request client
    let mut request_builder = reqwest::Client::new().post(image_service_url);
    for (name, value) in req.headers().iter() {
        request_builder = request_builder.header(name, value);
    }

    // convert the request body into bytes
    let body = req.into_body();
    let body_bytes = axum::body::to_bytes(body, usize::MAX).await.map_err(|e| {
        let err_msg = format!("Failed to convert the request body into bytes: {}", e);
        error!(
            target: "stdout",
            request_id = %request_id,
            message = %err_msg,
        );
        ServerError::Operation(err_msg)
    })?;

    request_builder = request_builder.body(body_bytes);

    // Use select! to handle request cancellation
    let ds_response = select! {
        response = request_builder.send() => {
            response.map_err(|e| {
                let err_msg = format!(
                    "Failed to forward the request to the downstream server: {}",
                    e
                );
                error!(
                    target: "stdout",
                    request_id = %request_id,
                    message = %err_msg,
                );
                ServerError::Operation(err_msg)
            })?
        }
        _ = cancel_token.cancelled() => {
            let warn_msg = "Request was cancelled by client";
            warn!(
                target: "stdout",
                request_id = %request_id,
                message = %warn_msg,
            );
            return Err(ServerError::Operation(warn_msg.to_string()));
        }
    };

    // create a response builder with the status and headers of the downstream response
    let mut response_builder = Response::builder().status(ds_response.status());
    for (name, value) in ds_response.headers().iter() {
        response_builder = response_builder.header(name, value);
    }

    // Handle response body reading with cancellation
    let bytes = select! {
        bytes = ds_response.bytes() => {
            bytes.map_err(|e| {
                let err_msg = format!("Failed to get the full response as bytes: {}", e);
                error!(
                    target: "stdout",
                    request_id = %request_id,
                    message = %err_msg,
                );
                ServerError::Operation(err_msg)
            })?
        }
        _ = cancel_token.cancelled() => {
            let warn_msg = "Request was cancelled while reading response";
            warn!(
                target: "stdout",
                request_id = %request_id,
                message = %warn_msg,
            );
            return Err(ServerError::Operation(warn_msg.to_string()));
        }
    };

    match response_builder.body(Body::from(bytes)) {
        Ok(response) => {
            info!(
                target: "stdout",
                request_id = %request_id,
                message = "Image request completed successfully",
            );
            Ok(response)
        }
        Err(e) => {
            let err_msg = format!("Failed to create the response: {}", e);
            error!(
                target: "stdout",
                request_id = %request_id,
                message = %err_msg,
            );
            Err(ServerError::Operation(err_msg))
        }
    }
}

pub(crate) async fn chunks_handler(
    State(_state): State<Arc<AppState>>,
    Extension(_cancel_token): Extension<CancellationToken>,
    headers: HeaderMap,
    mut multipart: axum::extract::Multipart,
) -> ServerResult<axum::response::Response> {
    let request_id = headers
        .get("x-request-id")
        .and_then(|h| h.to_str().ok())
        .unwrap_or("unknown")
        .to_string();

    info!(
        target: "stdout",
        request_id = %request_id,
        message = "Received a new chunks request"
    );

    // Process multipart form data
    let mut contents = String::new();
    let mut extension = String::new();
    let mut chunk_capacity: usize = 0;
    while let Some(mut field) = multipart.next_field().await.map_err(|e| {
        let err_msg = format!("Failed to get next field: {}", e);
        error!(target: "stdout", request_id = %request_id, message = %err_msg);
        ServerError::Operation(err_msg)
    })? {
        match field.name() {
            Some("file") => {
                // Get content type if available
                if let Some(content_type) = field.content_type() {
                    // check if the content type is a multipart/form-data
                    match content_type {
                        "text/plain" => {
                            extension = "txt".to_string();
                        }
                        "text/markdown" => {
                            extension = "md".to_string();
                        }
                        _ => {
                            let err_msg = "The file should be a plain text or markdown file";
                            error!(
                                target: "stdout",
                                request_id = %request_id,
                                message = %err_msg,
                            );
                            return Err(ServerError::Operation(err_msg.to_string()));
                        }
                    }
                }

                // get the file contents
                while let Some(chunk) = field.chunk().await.map_err(|e| {
                    let err_msg = format!("Failed to get the next chunk: {}", e);
                    error!(target: "stdout", request_id = %request_id, message = %err_msg);
                    ServerError::Operation(err_msg)
                })? {
                    let chunk_data = String::from_utf8(chunk.to_vec()).map_err(|e| {
                        let err_msg =
                            format!("Failed to convert the chunk data to a string: {}", e);
                        error!(target: "stdout", request_id = %request_id, message = %err_msg);
                        ServerError::Operation(err_msg)
                    })?;

                    contents.push_str(&chunk_data);
                }
            }
            Some("chunk_capacity") => {
                // Get content type if available
                if let Some(content_type) = field.content_type() {
                    info!(
                        target: "stdout",
                        request_id = %request_id,
                        message = format!("Content type: {}", content_type)
                    );
                }

                // Get the field data as a string
                let capacity = field.text().await.map_err(|e| {
                    let err_msg = format!("`chunk_capacity` field should be a text field. {}", e);
                    error!(
                        target: "stdout",
                        request_id = %request_id,
                        message = %err_msg,
                    );
                    ServerError::Operation(err_msg)
                })?;

                chunk_capacity = capacity.parse().map_err(|e| {
                    let err_msg = format!("Failed to convert the chunk capacity to a usize: {}", e);
                    error!(
                        target: "stdout",
                        request_id = %request_id,
                        message = %err_msg,
                    );
                    ServerError::Operation(err_msg)
                })?;

                debug!(
                    target: "stdout",
                    request_id = %request_id,
                    message = format!("Got chunk capacity: {}", chunk_capacity),
                );
            }
            Some(field_name) => {
                let warn_msg = format!("Unknown field: {}", field_name);
                warn!(
                    target: "stdout",
                    request_id = %request_id,
                    message = %warn_msg,
                );
            }
            None => {
                let warn_msg = "No field name found";
                error!(
                    target: "stdout",
                    request_id = %request_id,
                    message = %warn_msg,
                );
                return Err(ServerError::Operation(warn_msg.to_string()));
            }
        }
    }

    let chunks = rag::chunk_text(&contents, extension, chunk_capacity, &request_id)?;

    let json_body = serde_json::json!({
        "chunks": chunks,
    });
    let data = serde_json::to_string(&json_body).map_err(|e| {
        let err_msg = format!("Failed to serialize chunks response: {}", e);
        error!(target: "stdout", request_id = %request_id, message = %err_msg);
        ServerError::Operation(err_msg)
    })?;

    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "application/json")
        .body(Body::from(data))
        .map_err(|e| {
            let err_msg = format!("Failed to create response: {}", e);
            error!(target: "stdout", request_id = %request_id, message = %err_msg);
            ServerError::Operation(err_msg)
        })
}

pub(crate) async fn create_rag_handler(
    State(state): State<Arc<AppState>>,
    Extension(cancel_token): Extension<CancellationToken>,
    headers: HeaderMap,
    mut multipart: axum::extract::Multipart,
) -> ServerResult<axum::response::Response> {
    let request_id = headers
        .get("x-request-id")
        .and_then(|h| h.to_str().ok())
        .unwrap_or("unknown")
        .to_string();

    info!(
        target: "stdout",
        request_id = %request_id,
        message = "Received a new create RAG request"
    );

    // process the multipart form data
    let mut contents = String::new();
    let mut extension = String::new();
    let mut vdb_server_url = String::new();
    let mut vdb_collection_name = String::new();
    let mut vdb_api_key = String::new();
    let mut chunk_capacity = 100;
    while let Some(mut field) = multipart.next_field().await.map_err(|e| {
        let err_msg = format!("Failed to get next field: {}", e);
        error!(target: "stdout", request_id = %request_id, message = %err_msg);
        ServerError::Operation(err_msg)
    })? {
        match field.name() {
            Some("file") => {
                // Get content type if available
                if let Some(content_type) = field.content_type() {
                    // check if the content type is a multipart/form-data
                    match content_type {
                        "text/plain" => {
                            extension = "txt".to_string();
                        }
                        "text/markdown" => {
                            extension = "md".to_string();
                        }
                        _ => {
                            let err_msg = "The file should be a plain text or markdown file";
                            error!(
                                target: "stdout",
                                request_id = %request_id,
                                message = %err_msg,
                            );
                            return Err(ServerError::Operation(err_msg.to_string()));
                        }
                    }
                }

                // get the file contents
                while let Some(chunk) = field.chunk().await.map_err(|e| {
                    let err_msg = format!("Failed to get the next chunk: {}", e);
                    error!(target: "stdout", request_id = %request_id, message = %err_msg);
                    ServerError::Operation(err_msg)
                })? {
                    let chunk_data = String::from_utf8(chunk.to_vec()).map_err(|e| {
                        let err_msg =
                            format!("Failed to convert the chunk data to a string: {}", e);
                        error!(target: "stdout", request_id = %request_id, message = %err_msg);
                        ServerError::Operation(err_msg)
                    })?;

                    contents.push_str(&chunk_data);
                }
            }
            Some("chunk_capacity") => {
                // Get content type if available
                if let Some(content_type) = field.content_type() {
                    info!(
                        target: "stdout",
                        request_id = %request_id,
                        message = format!("Content type: {}", content_type)
                    );
                }

                // Get the field data as a string
                let capacity = field.text().await.map_err(|e| {
                    let err_msg = format!("`chunk_capacity` field should be a text field. {}", e);
                    error!(
                        target: "stdout",
                        request_id = %request_id,
                        message = %err_msg,
                    );
                    ServerError::Operation(err_msg)
                })?;

                chunk_capacity = capacity.parse().map_err(|e| {
                    let err_msg = format!("Failed to convert the chunk capacity to a usize: {}", e);
                    error!(
                        target: "stdout",
                        request_id = %request_id,
                        message = %err_msg,
                    );
                    ServerError::Operation(err_msg)
                })?;

                debug!(
                    target: "stdout",
                    request_id = %request_id,
                    message = format!("Got chunk capacity: {}", chunk_capacity),
                );
            }
            Some("vdb_server_url") => {
                // Get content type if available
                if let Some(content_type) = field.content_type() {
                    info!(
                        target: "stdout",
                        request_id = %request_id,
                        message = format!("Content type: {}", content_type)
                    );
                }

                // Get the field data as a string
                vdb_server_url = field.text().await.map_err(|e| {
                    let err_msg = format!("`vdb_server_url` field should be a text field. {}", e);
                    error!(
                        target: "stdout",
                        request_id = %request_id,
                        message = %err_msg,
                    );
                    ServerError::Operation(err_msg)
                })?;

                debug!(
                    target: "stdout",
                    request_id = %request_id,
                    message = format!("Got VectorDB server URL: {}", vdb_server_url),
                );
            }
            Some("vdb_collection_name") => {
                // Get content type if available
                if let Some(content_type) = field.content_type() {
                    info!(
                        target: "stdout",
                        request_id = %request_id,
                        message = format!("Content type: {}", content_type)
                    );
                }

                // Get the field data as a string
                vdb_collection_name = field.text().await.map_err(|e| {
                    let err_msg =
                        format!("`vdb_collection_name` field should be a text field. {}", e);
                    error!(
                        target: "stdout",
                        request_id = %request_id,
                        message = %err_msg,
                    );
                    ServerError::Operation(err_msg)
                })?;

                debug!(
                    target: "stdout",
                    request_id = %request_id,
                    message = format!("Got VectorDB collection name: {}", vdb_collection_name),
                );
            }
            Some("vdb_api_key") => {
                // Get content type if available
                if let Some(content_type) = field.content_type() {
                    info!(
                        target: "stdout",
                        request_id = %request_id,
                        message = format!("Content type: {}", content_type)
                    );
                }

                // Get the field data as a string
                vdb_api_key = field.text().await.map_err(|e| {
                    let err_msg = format!("`vdb_api_key` field should be a text field. {}", e);
                    error!(
                        target: "stdout",
                        request_id = %request_id,
                        message = %err_msg,
                    );
                    ServerError::Operation(err_msg)
                })?;
            }
            Some(field_name) => {
                let warn_msg = format!("Unknown field: {}", field_name);
                warn!(
                    target: "stdout",
                    request_id = %request_id,
                    message = %warn_msg,
                );
            }
            None => {
                let warn_msg = "No field name found";
                error!(
                    target: "stdout",
                    request_id = %request_id,
                    message = %warn_msg,
                );
                return Err(ServerError::Operation(warn_msg.to_string()));
            }
        }
    }

    // segment the contents into chunks
    info!(target: "stdout", request_id = %request_id, message = "Segment the contents into chunks.");
    let chunks = rag::chunk_text(&contents, extension, chunk_capacity, &request_id)?;

    // compute the embeddings for each chunk
    info!(target: "stdout", request_id = %request_id, message = "Compute embeddings for document chunks");
    let embedding_response = {
        let embedding_request = EmbeddingRequest {
            model: None,
            input: chunks.as_slice().into(),
            encoding_format: None,
            user: None,
            vdb_server_url: None,
            vdb_collection_name: None,
            vdb_api_key: None,
        };

        // get the embeddings server
        let servers = state.group_map.read().await;
        let embeddings_servers = match servers.get(&ServerKind::embeddings) {
            Some(servers) => servers,
            None => {
                let err_msg = "No embeddings server available";
                error!(
                    target: "stdout",
                    request_id = %request_id,
                    message = %err_msg,
                );
                return Err(ServerError::Operation(err_msg.to_string()));
            }
        };

        let embeddings_server_base_url = match embeddings_servers.next().await {
            Ok(url) => url,
            Err(e) => {
                let err_msg = format!("Failed to get the embeddings server: {}", e);
                error!(
                    target: "stdout",
                    request_id = %request_id,
                    message = %err_msg,
                );
                return Err(ServerError::Operation(err_msg));
            }
        };
        let embeddings_service_url = format!("{}v1/embeddings", embeddings_server_base_url);
        info!(
            target: "stdout",
            request_id = %request_id,
            message = format!("Forward the embeddings request to {}", embeddings_service_url),
        );

        // parse the content-type header
        let content_type = headers
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .ok_or_else(|| {
                let err_msg = "Missing Content-Type header".to_string();
                error!(
                    target: "stdout",
                    request_id = %request_id,
                    message = %err_msg,
                );
                ServerError::Operation(err_msg)
            })?;
        let content_type = content_type.to_string();
        debug!(
            target: "stdout",
            request_id = %request_id,
            message = format!("Request content type: {}", content_type)
        );

        // Create request client
        let client = reqwest::Client::new();
        let request = client
            .post(embeddings_service_url)
            .header("Content-Type", content_type)
            .json(&embedding_request);

        // Use select! to handle request cancellation
        let response = select! {
            response = request.send() => {
                response.map_err(|e| {
                    let err_msg = format!(
                        "Failed to forward the request to the downstream embedding server: {}",
                        e
                    );
                    error!(
                        target: "stdout",
                        request_id = %request_id,
                        message = %err_msg,
                    );
                    ServerError::Operation(err_msg)
                })?
            }
            _ = cancel_token.cancelled() => {
                let warn_msg = "Request was cancelled by client";
                warn!(
                    target: "stdout",
                    request_id = %request_id,
                    message = %warn_msg,
                );
                return Err(ServerError::Operation(warn_msg.to_string()));
            }
        };

        response.json::<EmbeddingsResponse>().await.map_err(|e| {
            let err_msg = format!("Failed to parse the embedding response: {}", e);
            error!(
                target: "stdout",
                request_id = %request_id,
                message = %err_msg,
            );
            ServerError::Operation(err_msg)
        })?
    };
    let embeddings = embedding_response.data;

    info!(
        target: "stdout",
        request_id = %request_id,
        message = format!("Got {} embeddings", embeddings.len()),
    );

    // create a Qdrant client
    let mut qdrant_client = qdrant::Qdrant::new_with_url(vdb_server_url);
    if !vdb_api_key.is_empty() {
        qdrant_client.set_api_key(vdb_api_key);
    }

    // create a collection in VectorDB
    let dim = embeddings[0].embedding.len();
    rag::qdrant_create_collection(&qdrant_client, &vdb_collection_name, dim, &request_id).await?;

    // persist the embeddings to the collection
    rag::qdrant_persist_embeddings(
        &qdrant_client,
        &vdb_collection_name,
        embeddings.as_slice(),
        chunks.as_slice(),
        &request_id,
    )
    .await?;

    // create a response with status code 200. Content-Type is JSON
    let json_body = serde_json::json!({
        "message": format!("Collection `{}` created successfully.", vdb_collection_name),
    });

    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "application/json")
        .body(Body::from(json_body.to_string()))
        .map_err(|e| {
            let err_msg = format!("Failed to create response: {}", e);
            error!(
                target: "stdout",
                request_id = %request_id,
                message = %err_msg,
            );
            ServerError::Operation(err_msg)
        })
}

pub(crate) async fn models_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> ServerResult<axum::response::Response> {
    let request_id = headers
        .get("x-request-id")
        .and_then(|h| h.to_str().ok())
        .unwrap_or("unknown")
        .to_string();

    let models = state.models.read().await;
    let list_response = ListModelsResponse {
        object: String::from("list"),
        data: models.values().flatten().cloned().collect(),
    };

    let json_body = serde_json::to_string(&list_response).map_err(|e| {
        let err_msg = format!("Failed to serialize the models: {}", e);
        error!(
            target: "stdout",
            request_id = %request_id,
            message = %err_msg,
        );
        ServerError::Operation(err_msg)
    })?;

    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "application/json")
        .body(Body::from(json_body))
        .map_err(|e| {
            let err_msg = format!("Failed to create response: {}", e);
            error!(
                target: "stdout",
                request_id = %request_id,
                message = %err_msg,
            );
            ServerError::Operation(err_msg)
        })
}

pub(crate) async fn info_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> ServerResult<axum::response::Response> {
    let request_id = headers
        .get("x-request-id")
        .and_then(|h| h.to_str().ok())
        .unwrap_or("unknown")
        .to_string();

    let server_info = state.server_info.read().await;
    let json_body = serde_json::json!(&*server_info);

    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "application/json")
        .body(Body::from(json_body.to_string()))
        .map_err(|e| {
            let err_msg = format!("Failed to create response: {}", e);
            error!(
                target: "stdout",
                request_id = %request_id,
                message = %err_msg,
            );
            ServerError::Operation(err_msg)
        })
}

pub(crate) mod admin {
    use super::*;

    pub(crate) async fn register_downstream_server_handler(
        State(state): State<Arc<AppState>>,
        headers: HeaderMap,
        Json(mut server): Json<Server>,
    ) -> ServerResult<axum::response::Response> {
        // Get request ID from headers
        let request_id = headers
            .get("x-request-id")
            .and_then(|h| h.to_str().ok())
            .unwrap_or("unknown")
            .to_string();

        let server_url = server.url.clone();
        let server_kind = server.kind;
        let server_id = server.id.clone();

        // verify the server
        if server_kind.contains(ServerKind::chat)
            || server_kind.contains(ServerKind::embeddings)
            || server_kind.contains(ServerKind::image)
            || server_kind.contains(ServerKind::transcribe)
            || server_kind.contains(ServerKind::translate)
            || server_kind.contains(ServerKind::tts)
        {
            verify_server(
                State(state.clone()),
                &request_id,
                &server_id,
                &server_url,
                &server_kind,
            )
            .await?;
        }

        // update health status of the server
        server.health_status.is_healthy = true;
        server.health_status.last_check = SystemTime::now();

        // register the server
        state.register_downstream_server(server).await?;
        info!(
            target: "stdout",
            request_id = %request_id,
            message = format!("Registered successfully. Assigned Server Id: {}", server_id),
        );

        // create a response with status code 200. Content-Type is JSON
        let json_body = serde_json::json!({
            "id": server_id,
            "url": server_url,
            "kind": server_kind
        });

        let response = Response::builder()
            .status(StatusCode::OK)
            .header("Content-Type", "application/json")
            .body(Body::from(json_body.to_string()))
            .map_err(|e| {
                let err_msg = format!("Failed to create response: {}", e);
                error!(
                    target: "stdout",
                    request_id = %request_id,
                    message = %err_msg,
                );
                ServerError::Operation(err_msg)
            })?;

        Ok(response)
    }

    // verify the server and get the server info and model list
    async fn verify_server(
        State(state): State<Arc<AppState>>,
        request_id: impl AsRef<str>,
        server_id: impl AsRef<str>,
        server_url: impl AsRef<str>,
        server_kind: &ServerKind,
    ) -> ServerResult<()> {
        let request_id = request_id.as_ref();
        let server_url = server_url.as_ref();

        let client = reqwest::Client::new();

        let server_info_url = format!("{}/v1/info", server_url);
        let response = client.get(&server_info_url).send().await.map_err(|e| {
            let err_msg = format!("Failed to verify the downstream server: {}", e);
            error!(
                target: "stdout",
                request_id = %request_id,
                message = %err_msg,
            );
            ServerError::Operation(err_msg)
        })?;

        if !response.status().is_success() {
            let err_msg = format!(
                "Failed to verify the downstream server: {}",
                response.status()
            );
            error!(
                target: "stdout",
                request_id = %request_id,
                message = %err_msg,
            );
            return Err(ServerError::Operation(err_msg));
        }

        let api_server = response.json::<ApiServer>().await.map_err(|e| {
            let err_msg = format!("Failed to parse the server info: {}", e);
            error!(
                target: "stdout",
                request_id = %request_id,
                message = %err_msg,
            );
            ServerError::Operation(err_msg)
        })?;

        // verify the server kind
        {
            if server_kind.contains(ServerKind::chat) && api_server.chat_model.is_none() {
                let err_msg = "You are trying to register a chat server. However, the server does not support `chat`. Please check the server kind.";
                error!(
                    target: "stdout",
                    request_id = %request_id,
                    message = %err_msg,
                );
                return Err(ServerError::Operation(err_msg.to_string()));
            }
            if server_kind.contains(ServerKind::embeddings) && api_server.embedding_model.is_none()
            {
                let err_msg = "You are trying to register an embedding server. However, the server does not support `embeddings`. Please check the server kind.";
                error!(
                    target: "stdout",
                    request_id = %request_id,
                    message = %err_msg,
                );
                return Err(ServerError::Operation(err_msg.to_string()));
            }
            if server_kind.contains(ServerKind::image) && api_server.image_model.is_none() {
                let err_msg = "You are trying to register an image server. However, the server does not support `image`. Please check the server kind.";
                error!(
                    target: "stdout",
                    request_id = %request_id,
                    message = %err_msg,
                );
                return Err(ServerError::Operation(err_msg.to_string()));
            }
            if server_kind.contains(ServerKind::tts) && api_server.tts_model.is_none() {
                let err_msg = "You are trying to register a TTS server. However, the server does not support `tts`. Please check the server kind.";
                error!(
                    target: "stdout",
                    request_id = %request_id,
                    message = %err_msg,
                );
                return Err(ServerError::Operation(err_msg.to_string()));
            }
            if server_kind.contains(ServerKind::translate) && api_server.translate_model.is_none() {
                let err_msg = "You are trying to register a translation server. However, the server does not support `translate`. Please check the server kind.";
                error!(
                    target: "stdout",
                    request_id = %request_id,
                    message = %err_msg,
                );
                return Err(ServerError::Operation(err_msg.to_string()));
            }
            if server_kind.contains(ServerKind::transcribe) && api_server.transcribe_model.is_none()
            {
                let err_msg = "You are trying to register a transcription server. However, the server does not support `transcribe`. Please check the server kind.";
                error!(
                    target: "stdout",
                    request_id = %request_id,
                    message = %err_msg,
                );
                return Err(ServerError::Operation(err_msg.to_string()));
            }
        }

        // update the server info
        let server_info = &mut state.server_info.write().await;
        server_info
            .servers
            .insert(server_id.as_ref().to_string(), api_server);

        // get the models from the downstream server
        let list_models_url = format!("{}/v1/models", server_url);
        let list_models_response = client.get(&list_models_url).send().await.map_err(|e| {
            let err_msg = format!("Failed to get the models from the downstream server: {}", e);
            error!(
                target: "stdout",
                request_id = %request_id,
                message = %err_msg,
            );
            ServerError::Operation(err_msg)
        })?;

        let list_models_response = list_models_response
            .json::<ListModelsResponse>()
            .await
            .map_err(|e| {
                let err_msg = format!("Failed to parse the models: {}", e);
                error!(
                    target: "stdout",
                    request_id = %request_id,
                    message = %err_msg,
                );
                ServerError::Operation(err_msg)
            })?;

        // update the models
        let mut models = state.models.write().await;
        models.insert(server_id.as_ref().to_string(), list_models_response.data);

        Ok(())
    }

    pub(crate) async fn remove_downstream_server_handler(
        State(state): State<Arc<AppState>>,
        headers: HeaderMap,
        Json(server_id): Json<ServerIdToRemove>,
    ) -> ServerResult<axum::response::Response> {
        // Get request ID from headers
        let request_id = headers
            .get("x-request-id")
            .and_then(|h| h.to_str().ok())
            .unwrap_or("unknown")
            .to_string();

        state
            .unregister_downstream_server(&server_id.server_id)
            .await?;

        // create a response with status code 200. Content-Type is JSON
        let json_body = serde_json::json!({
            "message": "Server unregistered successfully.",
            "id": server_id.server_id,
        });

        let response = Response::builder()
            .status(StatusCode::OK)
            .header("Content-Type", "application/json")
            .body(Body::from(json_body.to_string()))
            .map_err(|e| {
                let err_msg = format!("Failed to create response: {}", e);
                error!(
                    target: "stdout",
                    request_id = %request_id,
                    message = %err_msg,
                );
                ServerError::Operation(err_msg)
            })?;

        Ok(response)
    }

    pub(crate) async fn list_downstream_servers_handler(
        State(state): State<Arc<AppState>>,
        headers: HeaderMap,
    ) -> ServerResult<axum::response::Response> {
        // Get request ID from headers
        let request_id = headers
            .get("x-request-id")
            .and_then(|h| h.to_str().ok())
            .unwrap_or("unknown")
            .to_string();

        let servers = state.list_downstream_servers().await?;

        // compute the total number of servers
        let total_servers = servers.values().fold(0, |acc, servers| acc + servers.len());
        info!(
            target: "stdout",
            request_id = %request_id,
            message = format!("Found {} downstream servers", total_servers),
        );

        let json_body = serde_json::to_string(&servers).unwrap();

        let response = Response::builder()
            .status(StatusCode::OK)
            .header("Content-Type", "application/json")
            .body(Body::from(json_body))
            .map_err(|e| {
                let err_msg = format!("Failed to create response: {}", e);
                error!(
                    target: "stdout",
                    request_id = %request_id,
                    message = %err_msg,
                );
                ServerError::Operation(err_msg)
            })?;

        Ok(response)
    }
}
