use crate::{
    AppState, dual_debug, dual_error, dual_info, dual_warn,
    error::{ServerError, ServerResult},
    info::ApiServer,
    mcp::{MCP_SERVICES, MCP_TOOLS},
    rag,
    server::{RoutingPolicy, Server, ServerIdToRemove, ServerKind},
};
use axum::{
    Json,
    body::Body,
    extract::{Extension, State},
    http::{HeaderMap, Response, StatusCode},
};
use endpoints::{
    chat::{
        ChatCompletionAssistantMessage, ChatCompletionChunk, ChatCompletionObject,
        ChatCompletionRequest, ChatCompletionRequestMessage, ChatCompletionToolMessage, Tool,
        ToolCall, ToolChoice, ToolFunction,
    },
    embeddings::EmbeddingRequest,
    models::ListModelsResponse,
};
use futures_util::StreamExt;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE};
use rmcp::model::{CallToolRequestParam, RawContent};
use std::{sync::Arc, time::SystemTime};
use tokio::select;
use tokio_util::sync::CancellationToken;

pub(crate) async fn chat_handler(
    State(state): State<Arc<AppState>>,
    Extension(cancel_token): Extension<CancellationToken>,
    headers: HeaderMap,
    Json(mut request): Json<ChatCompletionRequest>,
) -> ServerResult<axum::response::Response> {
    let request_id = headers
        .get("x-request-id")
        .and_then(|h| h.to_str().ok())
        .unwrap_or("unknown")
        .to_string();

    // check if the user id is provided
    if request.user.is_none() {
        request.user = Some(gen_chat_id());
    };
    dual_info!(
        "Received a new chat request from user: {} - request_id: {}",
        request.user.as_ref().unwrap(),
        request_id
    );

    // update the request with MCP tools
    dual_info!("Updating the request with MCP tools");
    if let Some(mcp_config) = state.config.read().await.mcp.as_ref()
        && !mcp_config.server.tool_servers.is_empty()
    {
        let mut more_tools = Vec::new();
        for server_config in mcp_config.server.tool_servers.iter() {
            if server_config.enable {
                server_config
                    .tools
                    .as_ref()
                    .unwrap()
                    .iter()
                    .for_each(|mcp_tool| {
                        let tool = Tool::new(ToolFunction {
                            name: mcp_tool.name.to_string(),
                            description: mcp_tool.description.as_ref().map(|s| s.to_string()),
                            parameters: Some((*mcp_tool.input_schema).clone()),
                        });

                        more_tools.push(tool.clone());
                    });
            }
        }

        if !more_tools.is_empty() {
            if let Some(tools) = &mut request.tools {
                tools.extend(more_tools);
            } else {
                request.tools = Some(more_tools);
            }

            // set the tool choice to auto
            if let Some(ToolChoice::None) | None = request.tool_choice {
                request.tool_choice = Some(ToolChoice::Auto);
            }
        }
    }

    // ! DO NOT REMOVE THIS BLOCK
    {
        // let enable_rag = state.config.read().await.rag.as_ref().unwrap().enable;
        // match enable_rag {
        //     true => {
        //         rag::chat(
        //             State(state),
        //             Extension(cancel_token),
        //             headers,
        //             Json(request),
        //             &request_id,
        //         )
        //         .await
        //     }
        //     false => {
        //         chat(
        //             State(state),
        //             Extension(cancel_token),
        //             headers,
        //             Json(request),
        //             &request_id,
        //         )
        //         .await
        //     }
        // }
    }

    let rag = state.config.read().await.rag.clone();
    match rag {
        Some(rag_config) if rag_config.enable => {
            rag::chat(
                State(state),
                Extension(cancel_token),
                headers,
                Json(request),
                &request_id,
            )
            .await
        }
        _ => {
            chat(
                State(state),
                Extension(cancel_token),
                headers,
                Json(request),
                &request_id,
            )
            .await
        }
    }
}

pub(crate) async fn chat(
    State(state): State<Arc<AppState>>,
    Extension(cancel_token): Extension<CancellationToken>,
    headers: HeaderMap,
    Json(mut request): Json<ChatCompletionRequest>,
    request_id: impl AsRef<str>,
) -> ServerResult<axum::response::Response> {
    let request_id = request_id.as_ref();

    // get the chat server
    let target_server_info = {
        let servers = state.server_group.read().await;
        let chat_servers = match servers.get(&ServerKind::chat) {
            Some(servers) => servers,
            None => {
                let err_msg = "No chat server available";
                dual_error!("{} - request_id: {}", err_msg, request_id);
                return Err(ServerError::Operation(err_msg.to_string()));
            }
        };

        match chat_servers.next().await {
            Ok(target_server_info) => target_server_info,
            Err(e) => {
                let err_msg = format!("Failed to get the chat server: {e}");
                dual_error!("{} - request_id: {}", err_msg, request_id);
                return Err(ServerError::Operation(err_msg));
            }
        }
    };

    let chat_service_url = format!(
        "{}/v1/chat/completions",
        target_server_info.url.trim_end_matches('/')
    );
    dual_info!(
        "Forward the chat request to {} - request_id: {}",
        chat_service_url,
        request_id
    );

    // Create a request client that can be cancelled
    let ds_request = if headers.contains_key("authorization") {
        let authorization = headers
            .get("authorization")
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();

        reqwest::Client::new()
            .post(&chat_service_url)
            .header(CONTENT_TYPE, "application/json")
            .header(AUTHORIZATION, authorization)
            .json(&request)
    } else {
        reqwest::Client::new()
            .post(&chat_service_url)
            .header(CONTENT_TYPE, "application/json")
            .json(&request)
    };

    // Use select! to handle request cancellation
    let ds_response = select! {
        response = ds_request.send() => {
            response.map_err(|e| {
                let err_msg = format!(
                    "Failed to forward the request to the downstream server: {e}"
                );
                dual_error!("{} - request_id: {}", err_msg, request_id);
                ServerError::Operation(err_msg)
            })?
        }
        _ = cancel_token.cancelled() => {
            let warn_msg = "Request was cancelled by client";
            dual_warn!("{} - request_id: {}", warn_msg, request_id);
            return Err(ServerError::Operation(warn_msg.to_string()));
        }
    };

    let status = ds_response.status();
    dual_debug!("status: {} - request_id: {}", status, request_id);
    let headers = ds_response.headers().clone();

    match request.stream {
        Some(true) => {
            // check if the response has a header with the key "requires-tool-call"
            let mut requires_tool_call = false;
            if let Some(value) = headers.get("requires-tool-call") {
                // convert the value to a boolean
                requires_tool_call = value.to_str().unwrap().parse().unwrap();
            }

            dual_debug!(
                "requires_tool_call: {} - request_id: {}",
                requires_tool_call,
                request_id
            );

            match requires_tool_call {
                true => {
                    // Handle response body reading with cancellation
                    let mut ds_stream = ds_response.bytes_stream();

                    let mut tool_calls: Vec<ToolCall> = Vec::new();
                    while let Some(item) = ds_stream.next().await {
                        match item {
                            Ok(bytes) => {
                                match String::from_utf8(bytes.to_vec()) {
                                    Ok(s) => {
                                        let x = s
                                            .trim_start_matches("data:")
                                            .trim()
                                            .split("data:")
                                            .collect::<Vec<_>>();
                                        let s = x[0];

                                        dual_debug!("s: {}", s);

                                        // convert the bytes to ChatCompletionChunk
                                        if let Ok(chunk) =
                                            serde_json::from_str::<ChatCompletionChunk>(s)
                                        {
                                            dual_debug!(
                                                "chunk: {:?} - request_id: {}",
                                                &chunk,
                                                request_id
                                            );

                                            if !chunk.choices.is_empty() {
                                                for tool in chunk.choices[0].delta.tool_calls.iter()
                                                {
                                                    let tool_call = tool.clone().into();

                                                    dual_debug!("tool_call: {:?}", &tool_call);

                                                    tool_calls.push(tool_call);
                                                }

                                                break;
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        let err_msg = format!(
                                            "Failed to convert bytes from downstream server into string: {e}"
                                        );
                                        dual_error!("{} - request_id: {}", err_msg, request_id);
                                        return Err(ServerError::Operation(err_msg));
                                    }
                                }
                            }
                            Err(e) => {
                                let err_msg =
                                    format!("Failed to get the full response as bytes: {e}");
                                dual_error!("{} - request_id: {}", err_msg, request_id);
                                return Err(ServerError::Operation(err_msg));
                            }
                        }
                    }

                    match call_mcp_server(
                        tool_calls.as_slice(),
                        &mut request,
                        &headers,
                        &chat_service_url,
                        &request_id,
                        cancel_token,
                    )
                    .await
                    {
                        Ok(response) => Ok(response),
                        Err(ServerError::McpNotFoundClient) => {
                            let err_msg =
                                format!("Not found MCP server - request_id: {request_id}");
                            dual_error!("{}", err_msg);
                            Err(ServerError::Operation(err_msg))
                        }
                        Err(e) => {
                            let err_msg = format!(
                                "Failed to call MCP server: {e} - request_id: {request_id}"
                            );
                            dual_error!("{}", err_msg);
                            Err(ServerError::Operation(err_msg))
                        }
                    }
                }
                false => {
                    // Handle response body reading with cancellation
                    let bytes = select! {
                        bytes = ds_response.bytes() => {
                            bytes.map_err(|e| {
                                let err_msg = format!("Failed to get the full response as bytes: {e}");
                                dual_error!("{} - request_id: {}", err_msg, request_id);
                                ServerError::Operation(err_msg)
                            })?
                        }
                        _ = cancel_token.cancelled() => {
                            let warn_msg = "Request was cancelled while reading response";
                            dual_warn!("{} - request_id: {}", warn_msg, request_id);
                            return Err(ServerError::Operation(warn_msg.to_string()));
                        }
                    };

                    let mut response_builder = Response::builder().status(status);
                    // Copy all headers from downstream response
                    for (name, value) in headers.iter() {
                        match name.as_str() {
                            "access-control-allow-origin" => {
                                response_builder = response_builder.header(name, value);
                            }
                            "access-control-allow-headers" => {
                                response_builder = response_builder.header(name, value);
                            }
                            "access-control-allow-methods" => {
                                response_builder = response_builder.header(name, value);
                            }
                            "content-type" => {
                                response_builder = response_builder.header(name, value);
                            }
                            "cache-control" => {
                                response_builder = response_builder.header(name, value);
                            }
                            "connection" => {
                                response_builder = response_builder.header(name, value);
                            }
                            "user" => {
                                response_builder = response_builder.header(name, value);
                            }
                            "date" => {
                                response_builder = response_builder.header(name, value);
                            }
                            _ => {
                                dual_debug!(
                                    "ignore header: {} - {}",
                                    name,
                                    value.to_str().unwrap()
                                );
                            }
                        }
                    }

                    match response_builder.body(Body::from(bytes)) {
                        Ok(response) => {
                            dual_info!(
                                "Chat request completed successfully - request_id: {}",
                                request_id
                            );
                            Ok(response)
                        }
                        Err(e) => {
                            let err_msg = format!("Failed to create the response: {e}");
                            dual_error!("{} - request_id: {}", err_msg, request_id);
                            Err(ServerError::Operation(err_msg))
                        }
                    }
                }
            }
        }
        Some(false) | None => {
            // Handle response body reading with cancellation
            let bytes = select! {
                bytes = ds_response.bytes() => {
                    bytes.map_err(|e| {
                        let err_msg = format!("Failed to get the full response as bytes: {e}");
                        dual_error!("{} - request_id: {}", err_msg, request_id);
                        ServerError::Operation(err_msg)
                    })?
                }
                _ = cancel_token.cancelled() => {
                    let warn_msg = "Request was cancelled while reading response";
                    dual_warn!("{} - request_id: {}", warn_msg, request_id);
                    return Err(ServerError::Operation(warn_msg.to_string()));
                }
            };

            // check if the response has a header with the key "requires-tool-call"
            if let Some(value) = headers.get("requires-tool-call") {
                // convert the value to a boolean
                let requires_tool_call: bool = value.to_str().unwrap().parse().unwrap();

                dual_debug!(
                    "requires_tool_call: {} - request_id: {}",
                    requires_tool_call,
                    request_id
                );

                if requires_tool_call {
                    let chat_completion: ChatCompletionObject = match serde_json::from_slice(&bytes)
                    {
                        Ok(message) => message,
                        Err(e) => {
                            dual_error!("Failed to parse the response: {}", e);
                            return Err(ServerError::Operation(e.to_string()));
                        }
                    };

                    let assistant_message = &chat_completion.choices[0].message;

                    match call_mcp_server(
                        assistant_message.tool_calls.as_slice(),
                        &mut request,
                        &headers,
                        &chat_service_url,
                        &request_id,
                        cancel_token,
                    )
                    .await
                    {
                        Ok(response) => return Ok(response),
                        Err(ServerError::McpNotFoundClient) => {
                            dual_warn!("Not found MCP server - request_id: {}", request_id);
                        }
                        Err(e) => {
                            let err_msg = format!(
                                "Failed to call MCP server: {e} - request_id: {request_id}"
                            );
                            dual_error!("{}", err_msg);
                            return Err(ServerError::Operation(err_msg));
                        }
                    }
                }
            }

            let mut response_builder = Response::builder().status(status);
            // Copy all headers from downstream response
            for (name, value) in headers.iter() {
                dual_debug!("{}: {}", name, value.to_str().unwrap());
                response_builder = response_builder.header(name, value);
            }

            match response_builder.body(Body::from(bytes)) {
                Ok(response) => {
                    dual_info!(
                        "Chat request completed successfully - request_id: {}",
                        request_id
                    );
                    Ok(response)
                }
                Err(e) => {
                    let err_msg = format!("Failed to create the response: {e}");
                    dual_error!("{} - request_id: {}", err_msg, request_id);
                    Err(ServerError::Operation(err_msg))
                }
            }
        }
    }
}

async fn call_mcp_server(
    tool_calls: &[ToolCall],
    request: &mut ChatCompletionRequest,
    headers: &HeaderMap,
    chat_service_url: impl AsRef<str>,
    request_id: impl AsRef<str>,
    cancel_token: CancellationToken,
) -> ServerResult<axum::response::Response> {
    let request_id = request_id.as_ref();
    let chat_service_url = chat_service_url.as_ref();

    let tool_call = &tool_calls[0];
    let tool_name = tool_call.function.name.as_str();
    let tool_args = &tool_call.function.arguments;

    dual_debug!(
        "tool name: {}, tool args: {} - request_id: {}",
        tool_name,
        tool_args,
        request_id
    );

    // convert the func_args to a json object
    let arguments =
        serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(tool_args).ok();

    // find mcp client by tool name
    if let Some(mcp_tools) = MCP_TOOLS.get() {
        let tools = mcp_tools.read().await;
        dual_debug!("mcp_tools: {:?}", mcp_tools);

        // look up the tool name in MCP_TOOLS
        if let Some(mcp_client_name) = tools.get(tool_name) {
            if let Some(services) = MCP_SERVICES.get() {
                let service_map = services.read().await;
                // get the mcp client
                let service = match service_map.get(mcp_client_name) {
                    Some(mcp_client) => mcp_client,
                    None => {
                        let err_msg = format!("Tool not found: {tool_name}");
                        dual_error!("{} - request_id: {}", err_msg, request_id);
                        return Err(ServerError::Operation(err_msg.to_string()));
                    }
                };

                // call a tool
                let request_param = CallToolRequestParam {
                    name: tool_name.to_string().into(),
                    arguments,
                };
                let res = service
                    .read()
                    .await
                    .raw
                    .call_tool(request_param)
                    .await
                    .map_err(|e| {
                        dual_error!("Failed to call the tool: {}", e);
                        ServerError::Operation(e.to_string())
                    })?;
                dual_debug!("{}", serde_json::to_string_pretty(&res).unwrap());

                match res.is_error {
                    Some(false) => {
                        match res.content.is_empty() {
                            true => Err(ServerError::McpEmptyContent),
                            false => {
                                let content = &res.content[0];
                                match &content.raw {
                                    RawContent::Text(text) => {
                                        dual_debug!("tool result: {}", text.text);

                                        // create an assistant message
                                        let tool_completion_message =
                                            ChatCompletionRequestMessage::Tool(
                                                ChatCompletionToolMessage::new(&text.text, None),
                                            );

                                        // append assistant message with tool call to request messages
                                        let assistant_completion_message =
                                            ChatCompletionRequestMessage::Assistant(
                                                ChatCompletionAssistantMessage::new(
                                                    None,
                                                    None,
                                                    Some(tool_calls.to_vec()),
                                                ),
                                            );
                                        request.messages.push(assistant_completion_message);
                                        // append tool message with tool result to request messages
                                        request.messages.push(tool_completion_message);

                                        // disable tool choice
                                        if request.tool_choice.is_some() {
                                            request.tool_choice = Some(ToolChoice::None);
                                        }

                                        dual_info!(
                                            "request messages:\n{}",
                                            serde_json::to_string_pretty(&request.messages)
                                                .unwrap()
                                        );

                                        // Create a request client that can be cancelled
                                        let ds_request = if headers.contains_key("authorization") {
                                            let authorization = headers
                                                .get("authorization")
                                                .unwrap()
                                                .to_str()
                                                .unwrap()
                                                .to_string();

                                            reqwest::Client::new()
                                                .post(chat_service_url)
                                                .header(CONTENT_TYPE, "application/json")
                                                .header(AUTHORIZATION, authorization)
                                                .json(&request)
                                        } else {
                                            reqwest::Client::new()
                                                .post(chat_service_url)
                                                .header(CONTENT_TYPE, "application/json")
                                                .json(&request)
                                        };

                                        // Use select! to handle request cancellation
                                        let ds_response = select! {
                                            response = ds_request.send() => {
                                                response.map_err(|e| {
                                                    let err_msg = format!(
                                                        "Failed to forward the request to the downstream server: {e}"
                                                    );
                                                    dual_error!("{} - request_id: {}", err_msg, request_id);
                                                    ServerError::Operation(err_msg)
                                                })?
                                            }
                                            _ = cancel_token.cancelled() => {
                                                let warn_msg = "Request was cancelled by client";
                                                dual_warn!("{} - request_id: {}", warn_msg, request_id);
                                                return Err(ServerError::Operation(warn_msg.to_string()));
                                            }
                                        };

                                        let status = ds_response.status();
                                        let headers = ds_response.headers().clone();

                                        // Handle response body reading with cancellation
                                        let bytes = select! {
                                            bytes = ds_response.bytes() => {
                                                bytes.map_err(|e| {
                                                    let err_msg = format!("Failed to get the full response as bytes: {e}");
                                                    dual_error!("{} - request_id: {}", err_msg, request_id);
                                                    ServerError::Operation(err_msg)
                                                })?
                                            }
                                            _ = cancel_token.cancelled() => {
                                                let warn_msg = "Request was cancelled while reading response";
                                                dual_warn!("{} - request_id: {}", warn_msg, request_id);
                                                return Err(ServerError::Operation(warn_msg.to_string()));
                                            }
                                        };

                                        let mut response_builder =
                                            Response::builder().status(status);

                                        // Copy all headers from downstream response
                                        match request.stream {
                                            Some(true) => {
                                                for (name, value) in headers.iter() {
                                                    match name.as_str() {
                                                        "access-control-allow-origin" => {
                                                            response_builder = response_builder
                                                                .header(name, value);
                                                        }
                                                        "access-control-allow-headers" => {
                                                            response_builder = response_builder
                                                                .header(name, value);
                                                        }
                                                        "access-control-allow-methods" => {
                                                            response_builder = response_builder
                                                                .header(name, value);
                                                        }
                                                        "content-type" => {
                                                            response_builder = response_builder
                                                                .header(name, value);
                                                        }
                                                        "cache-control" => {
                                                            response_builder = response_builder
                                                                .header(name, value);
                                                        }
                                                        "connection" => {
                                                            response_builder = response_builder
                                                                .header(name, value);
                                                        }
                                                        "user" => {
                                                            response_builder = response_builder
                                                                .header(name, value);
                                                        }
                                                        "date" => {
                                                            response_builder = response_builder
                                                                .header(name, value);
                                                        }
                                                        _ => {
                                                            dual_debug!(
                                                                "ignore header: {} - {}",
                                                                name,
                                                                value.to_str().unwrap()
                                                            );
                                                        }
                                                    }
                                                }
                                            }
                                            Some(false) | None => {
                                                for (name, value) in headers.iter() {
                                                    dual_debug!(
                                                        "{}: {}",
                                                        name,
                                                        value.to_str().unwrap()
                                                    );
                                                    response_builder =
                                                        response_builder.header(name, value);
                                                }
                                            }
                                        }

                                        match response_builder.body(Body::from(bytes)) {
                                            Ok(response) => {
                                                dual_info!(
                                                    "Chat request completed successfully - request_id: {}",
                                                    request_id
                                                );
                                                Ok(response)
                                            }
                                            Err(e) => {
                                                let err_msg =
                                                    format!("Failed to create the response: {e}");
                                                dual_error!(
                                                    "{} - request_id: {}",
                                                    err_msg,
                                                    request_id
                                                );
                                                Err(ServerError::Operation(err_msg))
                                            }
                                        }
                                    }
                                    _ => {
                                        let err_msg =
                                            "Only text content is supported for tool call results";
                                        dual_error!("{} - request_id: {}", err_msg, request_id);
                                        Err(ServerError::Operation(err_msg.to_string()))
                                    }
                                }
                            }
                        }
                    }
                    _ => {
                        let err_msg = format!("Failed to call the tool: {tool_name}");
                        dual_error!("{} - request_id: {}", err_msg, request_id);
                        Err(ServerError::Operation(err_msg))
                    }
                }
            } else {
                let err_msg = "Empty MCP CLIENTS";
                dual_error!("{} - request_id: {}", err_msg, request_id);
                Err(ServerError::Operation(err_msg.to_string()))
            }
        } else {
            let err_msg = format!("Failed to find the MCP client with tool name: {tool_name}");
            dual_error!("{} - request_id: {}", err_msg, request_id);
            Err(ServerError::McpNotFoundClient)
        }
    } else {
        let err_msg = "Empty MCP TOOLS";
        dual_error!("{} - request_id: {}", err_msg, request_id);
        Err(ServerError::Operation(err_msg.to_string()))
    }
}

// Generate a unique chat id for the chat completion request
fn gen_chat_id() -> String {
    format!("chatcmpl-{}", uuid::Uuid::new_v4())
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

    dual_info!(
        "Received a new embeddings request - request_id: {}",
        request_id
    );

    // get the embeddings server
    let servers = state.server_group.read().await;
    let embeddings_servers = match servers.get(&ServerKind::embeddings) {
        Some(servers) => servers,
        None => {
            let err_msg = "No embeddings server available";
            dual_error!("{} - request_id: {}", err_msg, request_id);
            return Err(ServerError::Operation(err_msg.to_string()));
        }
    };

    let target_server_info = match embeddings_servers.next().await {
        Ok(target_server_info) => target_server_info,
        Err(e) => {
            let err_msg = format!("Failed to get the embeddings server: {e}");
            dual_error!("{} - request_id: {}", err_msg, request_id);
            return Err(ServerError::Operation(err_msg));
        }
    };
    let embeddings_service_url = format!(
        "{}/v1/embeddings",
        target_server_info.url.trim_end_matches('/')
    );
    dual_info!(
        "Forward the embeddings request to {} - request_id: {}",
        embeddings_service_url,
        request_id
    );

    // parse the content-type header
    let content_type = headers
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| {
            let err_msg = "Missing Content-Type header".to_string();
            dual_error!("{} - request_id: {}", err_msg, request_id);
            ServerError::Operation(err_msg)
        })?;
    let content_type = content_type.to_string();
    dual_debug!(
        "Request content type: {} - request_id: {}",
        content_type,
        request_id
    );

    // Create request client
    let ds_request = if headers.contains_key("authorization") {
        let authorization = headers
            .get("authorization")
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();

        reqwest::Client::new()
            .post(embeddings_service_url)
            .header("Content-Type", content_type)
            .header("Authorization", authorization)
            .json(&request)
    } else {
        reqwest::Client::new()
            .post(embeddings_service_url)
            .header("Content-Type", content_type)
            .json(&request)
    };

    // Use select! to handle request cancellation
    let ds_response = select! {
        response = ds_request.send() => {
            response.map_err(|e| {
                let err_msg = format!(
                    "Failed to forward the request to the downstream server: {e}",
                );
                dual_error!("{err_msg} - request_id: {request_id}");
                ServerError::Operation(err_msg)
            })?
        }
        _ = cancel_token.cancelled() => {
            let warn_msg = "Request was cancelled by client";
            dual_warn!("{} - request_id: {}", warn_msg, request_id);
            return Err(ServerError::Operation(warn_msg.to_string()));
        }
    };

    let status = ds_response.status();

    // Handle response body reading with cancellation
    let bytes = select! {
        bytes = ds_response.bytes() => {
            bytes.map_err(|e| {
                let err_msg = format!("Failed to get the full response as bytes: {e}");
                dual_error!("{err_msg} - request_id: {request_id}");
                ServerError::Operation(err_msg)
            })?
        }
        _ = cancel_token.cancelled() => {
            let warn_msg = "Request was cancelled while reading response";
            dual_warn!("{} - request_id: {}", warn_msg, request_id);
            return Err(ServerError::Operation(warn_msg.to_string()));
        }
    };

    match Response::builder()
        .status(status)
        .header("Content-Type", "application/json")
        .body(Body::from(bytes))
    {
        Ok(response) => {
            dual_info!(
                "Embeddings request completed successfully - request_id: {}",
                request_id
            );
            Ok(response)
        }
        Err(e) => {
            let err_msg = format!("Failed to create the response: {e}");
            dual_error!("{err_msg} - request_id: {request_id}");
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

    dual_info!(
        "Received a new audio transcription request - request_id: {}",
        request_id
    );

    // get the transcribe server
    let target_server_info = {
        let servers = state.server_group.read().await;
        let transcribe_servers = match servers.get(&ServerKind::transcribe) {
            Some(servers) => servers,
            None => {
                let err_msg = "No transcribe server available";
                dual_error!("{} - request_id: {}", err_msg, request_id);
                return Err(ServerError::Operation(err_msg.to_string()));
            }
        };

        match transcribe_servers.next().await {
            Ok(target_server_info) => target_server_info,
            Err(e) => {
                let err_msg = format!("Failed to get the transcribe server: {e}");
                dual_error!("{} - request_id: {}", err_msg, request_id);
                return Err(ServerError::Operation(err_msg));
            }
        }
    };

    let transcription_service_url = format!(
        "{}/v1/audio/transcriptions",
        target_server_info.url.trim_end_matches('/')
    );
    dual_info!(
        "Forward the audio transcription request to {} - request_id: {}",
        transcription_service_url,
        request_id
    );

    // Create request client
    let mut ds_request = reqwest::Client::new().post(transcription_service_url);
    for (name, value) in req.headers().iter() {
        ds_request = ds_request.header(name, value);
    }

    // convert the request body into bytes
    let body = req.into_body();
    let body_bytes = axum::body::to_bytes(body, usize::MAX).await.map_err(|e| {
        let err_msg = format!("Failed to convert the request body into bytes: {e}");
        dual_error!("{err_msg} - request_id: {request_id}");
        ServerError::Operation(err_msg)
    })?;

    ds_request = ds_request.body(body_bytes);

    // Use select! to handle request cancellation
    let ds_response = select! {
        response = ds_request.send() => {
            response.map_err(|e| {
                let err_msg = format!(
                    "Failed to forward the request to the downstream server: {e}"
                );
                dual_error!("{err_msg} - request_id: {request_id}");
                ServerError::Operation(err_msg)
            })?
        }
        _ = cancel_token.cancelled() => {
            let warn_msg = "Request was cancelled by client";
            dual_warn!("{} - request_id: {}", warn_msg, request_id);
            return Err(ServerError::Operation(warn_msg.to_string()));
        }
    };

    let status = ds_response.status();

    // Handle response body reading with cancellation
    let bytes = select! {
        bytes = ds_response.bytes() => {
            bytes.map_err(|e| {
                let err_msg = format!("Failed to get the full response as bytes: {e}");
                dual_error!("{err_msg} - request_id: {request_id}");
                ServerError::Operation(err_msg)
            })?
        }
        _ = cancel_token.cancelled() => {
            let warn_msg = "Request was cancelled while reading response";
            dual_warn!("{} - request_id: {}", warn_msg, request_id);
            return Err(ServerError::Operation(warn_msg.to_string()));
        }
    };

    match Response::builder()
        .status(status)
        .header("Content-Type", "application/json")
        .body(Body::from(bytes))
    {
        Ok(response) => {
            dual_info!(
                "Audio transcription request completed successfully - request_id: {}",
                request_id
            );
            Ok(response)
        }
        Err(e) => {
            let err_msg = format!("Failed to create the response: {e}");
            dual_error!("{err_msg} - request_id: {request_id}");
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

    dual_info!(
        "Received a new audio translation request - request_id: {}",
        request_id
    );

    // get the transcribe server
    let target_server_info = {
        let servers = state.server_group.read().await;
        let translate_servers = match servers.get(&ServerKind::translate) {
            Some(servers) => servers,
            None => {
                let err_msg = "No translate server available";
                dual_error!("{} - request_id: {}", err_msg, request_id);
                return Err(ServerError::Operation(err_msg.to_string()));
            }
        };

        match translate_servers.next().await {
            Ok(target_server_info) => target_server_info,
            Err(e) => {
                let err_msg = format!("Failed to get the translate server: {e}");
                dual_error!("{} - request_id: {}", err_msg, request_id);
                return Err(ServerError::Operation(err_msg));
            }
        }
    };

    let translation_service_url = format!(
        "{}/v1/audio/translations",
        target_server_info.url.trim_end_matches('/')
    );
    dual_info!(
        "Forward the audio translation request to {} - request_id: {}",
        translation_service_url,
        request_id
    );

    // Create request client
    let mut ds_request = reqwest::Client::new().post(translation_service_url);
    for (name, value) in req.headers().iter() {
        ds_request = ds_request.header(name, value);
    }

    // convert the request body into bytes
    let body = req.into_body();
    let body_bytes = axum::body::to_bytes(body, usize::MAX).await.map_err(|e| {
        let err_msg = format!("Failed to convert the request body into bytes: {e}");
        dual_error!("{err_msg} - request_id: {request_id}");
        ServerError::Operation(err_msg)
    })?;

    ds_request = ds_request.body(body_bytes);

    // Use select! to handle request cancellation
    let ds_response = select! {
        response = ds_request.send() => {
            response.map_err(|e| {
                let err_msg = format!(
                    "Failed to forward the request to the downstream server: {e}"
                );
                dual_error!("{err_msg} - request_id: {request_id}");
                ServerError::Operation(err_msg)
            })?
        }
        _ = cancel_token.cancelled() => {
            let warn_msg = "Request was cancelled by client";
            dual_warn!("{} - request_id: {}", warn_msg, request_id);
            return Err(ServerError::Operation(warn_msg.to_string()));
        }
    };

    let status = ds_response.status();

    // Handle response body reading with cancellation
    let bytes = select! {
        bytes = ds_response.bytes() => {
            bytes.map_err(|e| {
                let err_msg = format!("Failed to get the full response as bytes: {e}");
                dual_error!("{err_msg} - request_id: {request_id}");
                ServerError::Operation(err_msg)
            })?
        }
        _ = cancel_token.cancelled() => {
            let warn_msg = "Request was cancelled while reading response";
            dual_warn!("{} - request_id: {}", warn_msg, request_id);
            return Err(ServerError::Operation(warn_msg.to_string()));
        }
    };

    match Response::builder()
        .status(status)
        .header("Content-Type", "application/json")
        .body(Body::from(bytes))
    {
        Ok(response) => {
            dual_info!(
                "Audio translation request completed successfully - request_id: {}",
                request_id
            );
            Ok(response)
        }
        Err(e) => {
            let err_msg = format!("Failed to create the response: {e}");
            dual_error!("{err_msg} - request_id: {request_id}");
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

    dual_info!(
        "Received a new audio speech request - request_id: {}",
        request_id
    );

    // get the tts server
    let target_server_info = {
        let servers = state.server_group.read().await;
        let tts_servers = match servers.get(&ServerKind::tts) {
            Some(servers) => servers,
            None => {
                let err_msg = "No tts server available";
                dual_error!("{} - request_id: {}", err_msg, request_id);
                return Err(ServerError::Operation(err_msg.to_string()));
            }
        };

        match tts_servers.next().await {
            Ok(target_server_info) => target_server_info,
            Err(e) => {
                let err_msg = format!("Failed to get the tts server: {e}");
                dual_error!("{} - request_id: {}", err_msg, request_id);
                return Err(ServerError::Operation(err_msg));
            }
        }
    };

    let tts_service_url = format!(
        "{}/v1/audio/speech",
        target_server_info.url.trim_end_matches('/')
    );
    dual_info!(
        "Forward the audio speech request to {} - request_id: {}",
        tts_service_url,
        request_id
    );

    // Create request client
    let mut ds_request = reqwest::Client::new().post(tts_service_url);
    for (name, value) in req.headers().iter() {
        ds_request = ds_request.header(name, value);
    }

    let body = req.into_body();
    let body_bytes = axum::body::to_bytes(body, usize::MAX).await.map_err(|e| {
        let err_msg = format!("Failed to convert the request body into bytes: {e}");
        dual_error!("{err_msg} - request_id: {request_id}");
        ServerError::Operation(err_msg)
    })?;

    ds_request = ds_request.body(body_bytes);

    // Use select! to handle request cancellation
    let ds_response = select! {
        response = ds_request.send() => {
            response.map_err(|e| {
                let err_msg = format!(
                    "Failed to forward the request to the downstream server: {e}"
                );
                dual_error!("{err_msg} - request_id: {request_id}");
                ServerError::Operation(err_msg)
            })?
        }
        _ = cancel_token.cancelled() => {
            let warn_msg = "Request was cancelled by client";
            dual_warn!("{} - request_id: {}", warn_msg, request_id);
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
                let err_msg = format!("Failed to get the full response as bytes: {e}");
                dual_error!("{err_msg} - request_id: {request_id}");
                ServerError::Operation(err_msg)
            })?
        }
        _ = cancel_token.cancelled() => {
            let warn_msg = "Request was cancelled while reading response";
            dual_warn!("{} - request_id: {}", warn_msg, request_id);
            return Err(ServerError::Operation(warn_msg.to_string()));
        }
    };

    match response_builder.body(Body::from(bytes)) {
        Ok(response) => {
            dual_info!(
                "Audio speech request completed successfully - request_id: {}",
                request_id
            );
            Ok(response)
        }
        Err(e) => {
            let err_msg = format!("Failed to create the response: {e}");
            dual_error!("{err_msg} - request_id: {request_id}");
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

    dual_info!("Received a new image request - request_id: {}", request_id);

    // get the image server
    let target_server_info = {
        let servers = state.server_group.read().await;
        let image_servers = match servers.get(&ServerKind::image) {
            Some(servers) => servers,
            None => {
                let err_msg = "No image server available";
                dual_error!("{} - request_id: {}", err_msg, request_id);
                return Err(ServerError::Operation(err_msg.to_string()));
            }
        };

        match image_servers.next().await {
            Ok(target_server_info) => target_server_info,
            Err(e) => {
                let err_msg = format!("Failed to get the image server: {e}");
                dual_error!("{} - request_id: {}", err_msg, request_id);
                return Err(ServerError::Operation(err_msg));
            }
        }
    };

    let image_service_url = format!(
        "{}/v1/images/generations",
        target_server_info.url.trim_end_matches('/')
    );
    dual_info!(
        "Forward the image request to {} - request_id: {}",
        image_service_url,
        request_id
    );

    // Create request client
    let mut ds_request = reqwest::Client::new().post(image_service_url);
    for (name, value) in req.headers().iter() {
        ds_request = ds_request.header(name, value);
    }

    // convert the request body into bytes
    let body = req.into_body();
    let body_bytes = axum::body::to_bytes(body, usize::MAX).await.map_err(|e| {
        let err_msg = format!("Failed to convert the request body into bytes: {e}");
        dual_error!("{err_msg} - request_id: {request_id}");
        ServerError::Operation(err_msg)
    })?;

    ds_request = ds_request.body(body_bytes);

    // Use select! to handle request cancellation
    let ds_response = select! {
        response = ds_request.send() => {
            response.map_err(|e| {
                let err_msg = format!(
                    "Failed to forward the request to the downstream server: {e}"
                );
                dual_error!("{err_msg} - request_id: {request_id}");
                ServerError::Operation(err_msg)
            })?
        }
        _ = cancel_token.cancelled() => {
            let warn_msg = "Request was cancelled by client";
            dual_warn!("{} - request_id: {}", warn_msg, request_id);
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
                let err_msg = format!("Failed to get the full response as bytes: {e}");
                dual_error!("{err_msg} - request_id: {request_id}");
                ServerError::Operation(err_msg)
            })?
        }
        _ = cancel_token.cancelled() => {
            let warn_msg = "Request was cancelled while reading response";
            dual_warn!("{} - request_id: {}", warn_msg, request_id);
            return Err(ServerError::Operation(warn_msg.to_string()));
        }
    };

    match response_builder.body(Body::from(bytes)) {
        Ok(response) => {
            dual_info!(
                "Image request completed successfully - request_id: {}",
                request_id
            );
            Ok(response)
        }
        Err(e) => {
            let err_msg = format!("Failed to create the response: {e}");
            dual_error!("{err_msg} - request_id: {request_id}");
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

    dual_info!("Received a new chunks request - request_id: {}", request_id);

    // Process multipart form data
    let mut contents = String::new();
    let mut extension = String::new();
    let mut chunk_capacity: usize = 0;
    while let Some(mut field) = multipart.next_field().await.map_err(|e| {
        let err_msg = format!("Failed to get next field: {e}");
        dual_error!("{err_msg} - request_id: {request_id}");
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
                            dual_error!("{} - request_id: {}", err_msg, request_id);
                            return Err(ServerError::Operation(err_msg.to_string()));
                        }
                    }
                }

                // get the file contents
                while let Some(chunk) = field.chunk().await.map_err(|e| {
                    let err_msg = format!("Failed to get the next chunk: {e}");
                    dual_error!("{err_msg} - request_id: {request_id}");
                    ServerError::Operation(err_msg)
                })? {
                    let chunk_data = String::from_utf8(chunk.to_vec()).map_err(|e| {
                        let err_msg = format!("Failed to convert the chunk data to a string: {e}");
                        dual_error!("{err_msg} - request_id: {request_id}");
                        ServerError::Operation(err_msg)
                    })?;

                    contents.push_str(&chunk_data);
                }
            }
            Some("chunk_capacity") => {
                // Get content type if available
                if let Some(content_type) = field.content_type() {
                    dual_info!(
                        "Content type: {} - request_id: {}",
                        content_type,
                        request_id
                    );
                }

                // Get the field data as a string
                let capacity = field.text().await.map_err(|e| {
                    let err_msg = format!("`chunk_capacity` field should be a text field. {e}");
                    dual_error!("{err_msg} - request_id: {request_id}");
                    ServerError::Operation(err_msg)
                })?;

                chunk_capacity = capacity.parse().map_err(|e| {
                    let err_msg = format!("Failed to convert the chunk capacity to a usize: {e}");
                    dual_error!("{err_msg} - request_id: {request_id}");
                    ServerError::Operation(err_msg)
                })?;

                dual_debug!(
                    "Got chunk capacity: {} - request_id: {}",
                    chunk_capacity,
                    request_id
                );
            }
            Some(field_name) => {
                let warn_msg = format!("Unknown field: {field_name}");
                dual_warn!("{warn_msg} - request_id: {request_id}");
            }
            None => {
                let warn_msg = "No field name found";
                dual_error!("{} - request_id: {}", warn_msg, request_id);
                return Err(ServerError::Operation(warn_msg.to_string()));
            }
        }
    }

    // segment the contents into chunks
    dual_info!(
        "Segment the contents into chunks - request_id: {}",
        request_id
    );
    let chunks = rag::chunk_text(&contents, extension, chunk_capacity, &request_id)?;

    let json_body = serde_json::json!({
        "chunks": chunks,
    });
    let data = serde_json::to_string(&json_body).map_err(|e| {
        let err_msg = format!("Failed to serialize chunks response: {e}");
        dual_error!("{err_msg} - request_id: {request_id}");
        ServerError::Operation(err_msg)
    })?;

    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "application/json")
        .body(Body::from(data))
        .map_err(|e| {
            let err_msg = format!("Failed to create response: {e}");
            dual_error!("{err_msg} - request_id: {request_id}");
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
        let err_msg = format!("Failed to serialize the models: {e}");
        dual_error!("{err_msg} - request_id: {request_id}");
        ServerError::Operation(err_msg)
    })?;

    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "application/json")
        .body(Body::from(json_body))
        .map_err(|e| {
            let err_msg = format!("Failed to create response: {e}");
            dual_error!("{err_msg} - request_id: {request_id}");
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

    let mut chat_models = vec![];
    let mut embedding_models = vec![];
    let mut image_models = vec![];
    let mut tts_models = vec![];
    let mut translate_models = vec![];
    let mut transcribe_models = vec![];
    let server_info = state.server_info.read().await;
    for server in server_info.servers.values() {
        if let Some(ref model) = server.chat_model {
            chat_models.push(model.clone());
        }
        if let Some(ref model) = server.embedding_model {
            embedding_models.push(model.clone());
        }
        if let Some(ref model) = server.image_model {
            image_models.push(model.clone());
        }
        if let Some(ref model) = server.tts_model {
            tts_models.push(model.clone());
        }
        if let Some(ref model) = server.translate_model {
            translate_models.push(model.clone());
        }
        if let Some(ref model) = server.transcribe_model {
            transcribe_models.push(model.clone());
        }
    }

    let json_body = serde_json::json!({
        "models": {
            "chat": chat_models,
            "embedding": embedding_models,
            "image": image_models,
            "tts": tts_models,
            "translate": translate_models,
            "transcribe": transcribe_models,
        },
    });

    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "application/json")
        .body(Body::from(json_body.to_string()))
        .map_err(|e| {
            let err_msg = format!("Failed to create response: {e}");
            dual_error!("{err_msg} - request_id: {request_id}");
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
            dual_warn!(
                "Ignore the server verification for: {server_id} - request_id: {request_id}"
            );
            // _verify_server(
            //     State(state.clone()),
            //     &headers,
            //     &request_id,
            //     &server_id,
            //     &server_url,
            //     &server_kind,
            // )
            // .await?;
        }

        // update the model list
        update_model_list(
            State(state.clone()),
            &headers,
            &request_id,
            &server_id,
            &server_url,
        )
        .await?;

        // update health status of the server
        server.health_status.is_healthy = true;
        server.health_status.last_check = SystemTime::now();

        // register the server
        state.register_downstream_server(server).await?;
        dual_info!(
            "Registered successfully. Assigned Server Id: {} - request_id: {}",
            server_id,
            request_id
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
                let err_msg = format!("Failed to create response: {e}");
                dual_error!("{err_msg} - request_id: {request_id}");
                ServerError::Operation(err_msg)
            })?;

        Ok(response)
    }

    // verify the server and get the server info and model list
    async fn _verify_server(
        State(state): State<Arc<AppState>>,
        headers: &HeaderMap,
        request_id: impl AsRef<str>,
        server_id: impl AsRef<str>,
        server_url: impl AsRef<str>,
        server_kind: &ServerKind,
    ) -> ServerResult<()> {
        let request_id = request_id.as_ref();
        let server_url = server_url.as_ref();
        let server_id = server_id.as_ref();

        let server_info_url = format!("{server_url}/v1/info");

        let client = reqwest::Client::new();
        let response = if headers.contains_key("authorization") {
            let authorization = headers
                .get("authorization")
                .unwrap()
                .to_str()
                .unwrap()
                .to_string();

            client
                .get(&server_info_url)
                .header(CONTENT_TYPE, "application/json")
                .header(AUTHORIZATION, authorization)
                .send()
                .await
                .map_err(|e| {
                    let err_msg =
                        format!("Failed to verify the {server_kind} downstream server: {e}",);
                    dual_error!("{err_msg} - request_id: {request_id}");
                    ServerError::Operation(err_msg)
                })?
        } else {
            client.get(&server_info_url).send().await.map_err(|e| {
                let err_msg = format!("Failed to verify the {server_kind} downstream server: {e}",);
                dual_error!("{err_msg} - request_id: {request_id}");
                ServerError::Operation(err_msg)
            })?
        };
        if !response.status().is_success() {
            let err_msg = format!(
                "Failed to verify the {} downstream server: {}",
                server_kind,
                response.status()
            );
            dual_error!("{} - request_id: {}", err_msg, request_id);
            return Err(ServerError::Operation(err_msg));
        }

        let mut api_server = response.json::<ApiServer>().await.map_err(|e| {
            let err_msg = format!("Failed to parse the server info: {e}");
            dual_error!("{err_msg} - request_id: {request_id}");
            ServerError::Operation(err_msg)
        })?;
        api_server.server_id = Some(server_id.to_string());

        dual_debug!("server kind: {}", server_kind.to_string());
        dual_debug!("api server: {:?}", api_server);

        // verify the server kind
        {
            if server_kind.contains(ServerKind::chat) && api_server.chat_model.is_none() {
                let err_msg = "You are trying to register a chat server. However, the server does not support `chat`. Please check the server kind.";
                dual_error!("{} - request_id: {}", err_msg, request_id);
                return Err(ServerError::Operation(err_msg.to_string()));
            }
            if server_kind.contains(ServerKind::embeddings) && api_server.embedding_model.is_none()
            {
                let err_msg = "You are trying to register an embedding server. However, the server does not support `embeddings`. Please check the server kind.";
                dual_error!("{} - request_id: {}", err_msg, request_id);
                return Err(ServerError::Operation(err_msg.to_string()));
            }
            if server_kind.contains(ServerKind::image) && api_server.image_model.is_none() {
                let err_msg = "You are trying to register an image server. However, the server does not support `image`. Please check the server kind.";
                dual_error!("{} - request_id: {}", err_msg, request_id);
                return Err(ServerError::Operation(err_msg.to_string()));
            }
            if server_kind.contains(ServerKind::tts) && api_server.tts_model.is_none() {
                let err_msg = "You are trying to register a TTS server. However, the server does not support `tts`. Please check the server kind.";
                dual_error!("{} - request_id: {}", err_msg, request_id);
                return Err(ServerError::Operation(err_msg.to_string()));
            }
            if server_kind.contains(ServerKind::translate) && api_server.translate_model.is_none() {
                let err_msg = "You are trying to register a translation server. However, the server does not support `translate`. Please check the server kind.";
                dual_error!("{} - request_id: {}", err_msg, request_id);
                return Err(ServerError::Operation(err_msg.to_string()));
            }
            if server_kind.contains(ServerKind::transcribe) && api_server.transcribe_model.is_none()
            {
                let err_msg = "You are trying to register a transcription server. However, the server does not support `transcribe`. Please check the server kind.";
                dual_error!("{} - request_id: {}", err_msg, request_id);
                return Err(ServerError::Operation(err_msg.to_string()));
            }
        }

        // update the server info
        let server_info = &mut state.server_info.write().await;
        server_info
            .servers
            .insert(server_id.to_string(), api_server);

        Ok(())
    }

    // update the model list
    pub(crate) async fn update_model_list(
        State(state): State<Arc<AppState>>,
        headers: &HeaderMap,
        request_id: impl AsRef<str>,
        server_id: impl AsRef<str>,
        server_url: impl AsRef<str>,
    ) -> ServerResult<()> {
        let request_id = request_id.as_ref();
        let server_url = server_url.as_ref();
        let server_id = server_id.as_ref();

        // get the models from the downstream server
        let list_models_url = format!("{server_url}/v1/models");
        let response = if headers.contains_key("authorization") {
            let authorization = headers
                .get("authorization")
                .unwrap()
                .to_str()
                .unwrap()
                .to_string();
            reqwest::Client::new()
                .get(&list_models_url)
                .header(CONTENT_TYPE, "application/json")
                .header(AUTHORIZATION, authorization)
                .send()
                .await
                .map_err(|e| {
                    let err_msg =
                        format!("Failed to get the models from the downstream server: {e}");
                    dual_error!("{err_msg} - request_id: {request_id}");
                    ServerError::Operation(err_msg)
                })?
        } else {
            reqwest::Client::new()
                .get(&list_models_url)
                .send()
                .await
                .map_err(|e| {
                    let err_msg =
                        format!("Failed to get the models from the downstream server: {e}");
                    dual_error!("{err_msg} - request_id: {request_id}");
                    ServerError::Operation(err_msg)
                })?
        };
        let status = response.status();
        if !status.is_success() {
            let err_msg = format!(
                "Failed to get model info from {server_url} downstream server. Status: {status}",
            );
            dual_error!("{} - request_id: {}", err_msg, request_id);
            return Err(ServerError::Operation(err_msg));
        }

        let list_models_response = response.json::<ListModelsResponse>().await.map_err(|e| {
            let err_msg = format!("Failed to parse the models: {e}");
            dual_error!("{err_msg} - request_id: {request_id}");
            ServerError::Operation(err_msg)
        })?;

        // update the models
        let mut models = state.models.write().await;
        models.insert(server_id.to_string(), list_models_response.data);

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
                let err_msg = format!("Failed to create response: {e}");
                dual_error!("{err_msg} - request_id: {request_id}");
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
        dual_info!(
            "Found {} downstream servers - request_id: {}",
            total_servers,
            request_id
        );

        let json_body = serde_json::to_string(&servers).unwrap();

        let response = Response::builder()
            .status(StatusCode::OK)
            .header("Content-Type", "application/json")
            .body(Body::from(json_body))
            .map_err(|e| {
                let err_msg = format!("Failed to create response: {e}");
                dual_error!("{err_msg} - request_id: {request_id}");
                ServerError::Operation(err_msg)
            })?;

        Ok(response)
    }
}
