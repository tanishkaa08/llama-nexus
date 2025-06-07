use crate::{
    config::McpToolServerConfig,
    dual_debug, dual_error, dual_info, dual_warn,
    error::{ServerError, ServerResult},
    info::ApiServer,
    mcp::{MCP_KEYWORD_SEARCH_CLIENT, USER_TO_MCP_CLIENTS, USER_TO_MCP_TOOLS},
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
    chat::{
        ChatCompletionAssistantMessage, ChatCompletionChunk, ChatCompletionObject,
        ChatCompletionRequest, ChatCompletionRequestMessage, ChatCompletionToolMessage, Tool,
        ToolCall, ToolChoice, ToolFunction,
    },
    embeddings::{EmbeddingRequest, EmbeddingsResponse},
    models::ListModelsResponse,
};
use futures_util::StreamExt;
use gaia_kwsearch_mcp_common::{CreateIndexResponse, KwDocumentInput};
use reqwest::header::CONTENT_TYPE;
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
    // check if the user id is provided
    if request.user.is_none() {
        request.user = Some(gen_chat_id());
    };
    dual_info!("user: {}", request.user.as_ref().unwrap());

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
    Json(mut request): Json<ChatCompletionRequest>,
) -> ServerResult<axum::response::Response> {
    let request_id = headers
        .get("x-request-id")
        .and_then(|h| h.to_str().ok())
        .unwrap_or("unknown")
        .to_string();

    dual_info!("Received a new chat request - request_id: {}", request_id);

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

    // load tools from the `mcp_tools` field in the request
    if let Some(config_mcp_servers) = &request.mcp_tools {
        if !config_mcp_servers.is_empty() {
            let mut more_tools = Vec::new();
            for config in config_mcp_servers {
                dual_debug!("mcp server info: {:?}", config);

                let mut server_config = McpToolServerConfig {
                    name: config.server_label.clone(),
                    transport: config.transport,
                    url: config.server_url.clone(),
                    enable: true,
                    tools: None,
                };

                // connect the mcp server and get the tools
                server_config
                    .connect_mcp_server(request.user.as_ref().unwrap())
                    .await?;

                // get the allowed tools
                let allowed_tools = config.allowed_tools.as_deref().unwrap_or(&[]);

                // get the tools from the mcp server
                if let Some(tools) = server_config.tools.as_deref() {
                    tools.iter().for_each(|rmcp_tool| {
                        let tool_name = rmcp_tool.name.to_string();
                        let tool = Tool::new(ToolFunction {
                            name: tool_name.clone(),
                            description: rmcp_tool.description.as_ref().map(|s| s.to_string()),
                            parameters: Some((*rmcp_tool.input_schema).clone()),
                        });

                        if allowed_tools.is_empty() || allowed_tools.contains(&tool_name) {
                            dual_debug!("tool to be added: {:?}", &tool);
                            more_tools.push(tool.clone());
                        }
                    });
                }
            }

            // update the request with MCP tools
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
    }

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
    let request_builder = reqwest::Client::new()
        .post(&chat_service_url)
        .header(CONTENT_TYPE, "application/json")
        .json(&request);

    // Use select! to handle request cancellation
    let ds_response = select! {
        response = request_builder.send() => {
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

                                        // let s = s.trim_start_matches("data:").trim();

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
    chat_service_url: impl AsRef<str>,
    request_id: impl AsRef<str>,
    cancel_token: CancellationToken,
) -> ServerResult<axum::response::Response> {
    let request_id = request_id.as_ref();
    let chat_service_url = chat_service_url.as_ref();

    // get the user id from the request
    let user_id = match request.user.as_ref() {
        Some(user_id) => user_id,
        None => {
            let err_msg = "User ID is not found in the request";
            dual_error!("{} - request_id: {}", err_msg, request_id);
            return Err(ServerError::Operation(err_msg.to_string()));
        }
    };

    // let tool_calls = assistant_message.tool_calls.clone();
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

    // get the mcp client by user id and tool name, then call the tool
    let response = match USER_TO_MCP_TOOLS.get() {
        Some(user_to_mcp_tools) => {
            let user_to_mcp_tools = user_to_mcp_tools.read().await;
            match user_to_mcp_tools.get(user_id) {
                Some(mcp_tools) => {
                    let mcp_tools = mcp_tools.read().await;
                    match mcp_tools.get(tool_name) {
                        Some(mcp_client_name) => {
                            match USER_TO_MCP_CLIENTS.get() {
                                Some(user_to_mcp_clients) => {
                                    let user_to_mcp_clients = user_to_mcp_clients.read().await;
                                    match user_to_mcp_clients.get(user_id) {
                                        Some(mcp_clients) => {
                                            let mcp_clients = mcp_clients.read().await;
                                            match mcp_clients.get(mcp_client_name) {
                                                Some(mcp_client) => {
                                                    // call a tool
                                                    let request_param = CallToolRequestParam {
                                                        name: tool_name.to_string().into(),
                                                        arguments,
                                                    };
                                                    let res = mcp_client
                                                        .read()
                                                        .await
                                                        .raw
                                                        .peer()
                                                        .call_tool(request_param)
                                                        .await
                                                        .map_err(|e| {
                                                            dual_error!(
                                                                "Failed to call the tool: {}",
                                                                e
                                                            );
                                                            ServerError::Operation(e.to_string())
                                                        })?;
                                                    dual_debug!(
                                                        "{} - request_id: {}",
                                                        serde_json::to_string_pretty(&res).unwrap(),
                                                        request_id
                                                    );

                                                    match res.is_error {
                                                        Some(false) => {
                                                            match res.content.is_empty() {
                                                                true => {
                                                                    dual_error!(
                                    "MCP server returned empty content - request_id: {}",
                                    request_id
                                );
                                                                    Err(ServerError::McpEmptyContent)
                                                                }
                                                                false => {
                                                                    let content = &res.content[0];
                                                                    match &content.raw {
                                                                        RawContent::Text(text) => {
                                                                            dual_debug!(
                                            "tool result: {} - request_id: {}",
                                            text.text,
                                            request_id
                                        );

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

                                                                            // Create a request client that can be cancelled
                                                                            let request_builder = reqwest::Client::new()
                                            .post(chat_service_url)
                                            .header(CONTENT_TYPE, "application/json")
                                            .json(&request);

                                                                            // Use select! to handle request cancellation
                                                                            let ds_response = select! {
                                                                                response = request_builder.send() => {
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

                                                                            let status =
                                                                                ds_response
                                                                                    .status();
                                                                            let headers =
                                                                                ds_response
                                                                                    .headers()
                                                                                    .clone();

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

                                                                            let mut
                                                                            response_builder =
                                                                                Response::builder()
                                                                                    .status(status);

                                                                            // Copy all headers from downstream response
                                                                            match request.stream {
                                                                                Some(true) => {
                                                                                    for (
                                                                                        name,
                                                                                        value,
                                                                                    ) in headers
                                                                                        .iter()
                                                                                    {
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
                                                                                Some(false)
                                                                                | None => {
                                                                                    for (
                                                                                        name,
                                                                                        value,
                                                                                    ) in headers
                                                                                        .iter()
                                                                                    {
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

                                                                            match response_builder
                                                                                .body(Body::from(
                                                                                    bytes,
                                                                                )) {
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
                                                            dual_error!(
                                                                "{} - request_id: {}",
                                                                err_msg,
                                                                request_id
                                                            );
                                                            Err(ServerError::Operation(err_msg))
                                                        }
                                                    }
                                                }
                                                None => {
                                                    let err_msg = format!(
                                                        "Not found the MCP client name `{mcp_client_name}` in USER_TO_MCP_CLIENTS"
                                                    );
                                                    dual_error!(
                                                        "{} - request_id: {}",
                                                        err_msg,
                                                        request_id
                                                    );
                                                    return Err(ServerError::Operation(
                                                        err_msg.to_string(),
                                                    ));
                                                }
                                            }
                                        }
                                        None => {
                                            let err_msg =
                                                "Not found the user ID in USER_TO_MCP_CLIENTS";
                                            dual_error!("{} - request_id: {}", err_msg, request_id);
                                            return Err(ServerError::Operation(
                                                err_msg.to_string(),
                                            ));
                                        }
                                    }
                                }
                                None => {
                                    let err_msg = "USER_TO_MCP_CLIENTS is empty or not initialized";
                                    dual_error!("{} - request_id: {}", err_msg, request_id);
                                    return Err(ServerError::Operation(err_msg.to_string()));
                                }
                            }
                        }
                        None => {
                            dual_error!(
                                "Failed to find the MCP client with tool name: {} - request_id: {}",
                                tool_name,
                                request_id,
                            );
                            return Err(ServerError::McpNotFoundClient);
                        }
                    }
                }
                None => {
                    let err_msg = "Not found the user ID in USER_TO_MCP_TOOLS";
                    dual_error!("{} - request_id: {}", err_msg, request_id);
                    return Err(ServerError::Operation(err_msg.to_string()));
                }
            }
        }
        None => {
            let err_msg = "USER_TO_MCP_TOOLS is empty or not initialized";
            dual_error!("{} - request_id: {}", err_msg, request_id);
            return Err(ServerError::Operation(err_msg.to_string()));
        }
    };

    // erase mcp tools from USER_TO_MCP_TOOLS by user id
    if let Some(user_to_mcp_tools) = USER_TO_MCP_TOOLS.get() {
        let mut user_to_mcp_tools = user_to_mcp_tools.write().await;
        user_to_mcp_tools.remove(user_id);

        dual_debug!(
            "Erase mcp tools from USER_TO_MCP_TOOLS by user id: {} - request_id: {}",
            user_id,
            request_id
        );
    }

    // erase mcp clients from USER_TO_MCP_CLIENTS by user id
    if let Some(user_to_mcp_clients) = USER_TO_MCP_CLIENTS.get() {
        let mut user_to_mcp_clients = user_to_mcp_clients.write().await;
        user_to_mcp_clients.remove(user_id);

        dual_debug!(
            "Erase mcp clients from USER_TO_MCP_CLIENTS by user id: {} - request_id: {}",
            user_id,
            request_id
        );
    }

    response
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

    let status = response.status();

    // Handle response body reading with cancellation
    let bytes = select! {
        bytes = response.bytes() => {
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
    let mut request_builder = reqwest::Client::new().post(transcription_service_url);
    for (name, value) in req.headers().iter() {
        request_builder = request_builder.header(name, value);
    }

    // convert the request body into bytes
    let body = req.into_body();
    let body_bytes = axum::body::to_bytes(body, usize::MAX).await.map_err(|e| {
        let err_msg = format!("Failed to convert the request body into bytes: {e}");
        dual_error!("{err_msg} - request_id: {request_id}");
        ServerError::Operation(err_msg)
    })?;

    request_builder = request_builder.body(body_bytes);

    // Use select! to handle request cancellation
    let response = select! {
        response = request_builder.send() => {
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

    let status = response.status();

    // Handle response body reading with cancellation
    let bytes = select! {
        bytes = response.bytes() => {
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
    let mut request_builder = reqwest::Client::new().post(translation_service_url);
    for (name, value) in req.headers().iter() {
        request_builder = request_builder.header(name, value);
    }

    // convert the request body into bytes
    let body = req.into_body();
    let body_bytes = axum::body::to_bytes(body, usize::MAX).await.map_err(|e| {
        let err_msg = format!("Failed to convert the request body into bytes: {e}");
        dual_error!("{err_msg} - request_id: {request_id}");
        ServerError::Operation(err_msg)
    })?;

    request_builder = request_builder.body(body_bytes);

    // Use select! to handle request cancellation
    let response = select! {
        response = request_builder.send() => {
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

    let status = response.status();

    // Handle response body reading with cancellation
    let bytes = select! {
        bytes = response.bytes() => {
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
    let mut request_builder = reqwest::Client::new().post(tts_service_url);
    for (name, value) in req.headers().iter() {
        request_builder = request_builder.header(name, value);
    }

    let body = req.into_body();
    let body_bytes = axum::body::to_bytes(body, usize::MAX).await.map_err(|e| {
        let err_msg = format!("Failed to convert the request body into bytes: {e}");
        dual_error!("{err_msg} - request_id: {request_id}");
        ServerError::Operation(err_msg)
    })?;

    request_builder = request_builder.body(body_bytes);

    // Use select! to handle request cancellation
    let ds_response = select! {
        response = request_builder.send() => {
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
    let mut request_builder = reqwest::Client::new().post(image_service_url);
    for (name, value) in req.headers().iter() {
        request_builder = request_builder.header(name, value);
    }

    // convert the request body into bytes
    let body = req.into_body();
    let body_bytes = axum::body::to_bytes(body, usize::MAX).await.map_err(|e| {
        let err_msg = format!("Failed to convert the request body into bytes: {e}");
        dual_error!("{err_msg} - request_id: {request_id}");
        ServerError::Operation(err_msg)
    })?;

    request_builder = request_builder.body(body_bytes);

    // Use select! to handle request cancellation
    let ds_response = select! {
        response = request_builder.send() => {
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

    dual_info!(
        "Received a new create RAG request - request_id: {}",
        request_id
    );

    // process the multipart form data
    let mut contents = String::new();
    let mut extension = String::new();
    let mut vdb_server_url = String::new();
    let mut vdb_collection_name = String::new();
    let mut vdb_api_key = String::new();
    let mut kw_search_url = String::new();
    let mut kw_search_index_name = String::new();
    let mut chunk_capacity = 100;
    while let Some(mut field) = multipart.next_field().await.map_err(|e| {
        let err_msg = format!("Failed to get next field: {e}");
        dual_error!("{} - request_id: {}", err_msg, request_id);
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
                    dual_error!("{} - request_id: {}", err_msg, request_id);
                    ServerError::Operation(err_msg)
                })? {
                    let chunk_data = String::from_utf8(chunk.to_vec()).map_err(|e| {
                        let err_msg = format!("Failed to convert the chunk data to a string: {e}");
                        dual_error!("{} - request_id: {}", err_msg, request_id);
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
                    dual_error!("{} - request_id: {}", err_msg, request_id);
                    ServerError::Operation(err_msg)
                })?;

                chunk_capacity = capacity.parse().map_err(|e| {
                    let err_msg = format!("Failed to convert the chunk capacity to a usize: {e}");
                    dual_error!("{} - request_id: {}", err_msg, request_id);
                    ServerError::Operation(err_msg)
                })?;

                dual_debug!(
                    "Got chunk capacity: {} - request_id: {}",
                    chunk_capacity,
                    request_id
                );
            }
            Some("vdb_server_url") => {
                // Get content type if available
                if let Some(content_type) = field.content_type() {
                    dual_info!(
                        "Content type: {} - request_id: {}",
                        content_type,
                        request_id
                    );
                }

                // Get the field data as a string
                vdb_server_url = field.text().await.map_err(|e| {
                    let err_msg = format!("`vdb_server_url` field should be a text field. {e}");
                    dual_error!("{} - request_id: {}", err_msg, request_id);
                    ServerError::Operation(err_msg)
                })?;

                dual_debug!(
                    "Got VectorDB server URL: {} - request_id: {}",
                    vdb_server_url,
                    request_id
                );
            }
            Some("vdb_collection_name") => {
                // Get content type if available
                if let Some(content_type) = field.content_type() {
                    dual_info!(
                        "Content type: {} - request_id: {}",
                        content_type,
                        request_id
                    );
                }

                // Get the field data as a string
                vdb_collection_name = field.text().await.map_err(|e| {
                    let err_msg =
                        format!("`vdb_collection_name` field should be a text field. {e}");
                    dual_error!("{} - request_id: {}", err_msg, request_id);
                    ServerError::Operation(err_msg)
                })?;

                dual_debug!(
                    "Got VectorDB collection name: {} - request_id: {}",
                    vdb_collection_name,
                    request_id
                );
            }
            Some("vdb_api_key") => {
                // Get content type if available
                if let Some(content_type) = field.content_type() {
                    dual_info!(
                        "Content type: {} - request_id: {}",
                        content_type,
                        request_id
                    );
                }

                // Get the field data as a string
                vdb_api_key = field.text().await.map_err(|e| {
                    let err_msg = format!("`vdb_api_key` field should be a text field. {e}");
                    dual_error!("{} - request_id: {}", err_msg, request_id);
                    ServerError::Operation(err_msg)
                })?;
            }
            Some("kw_search_url") => {
                // Get content type if available
                if let Some(content_type) = field.content_type() {
                    dual_info!(
                        "Content type: {} - request_id: {}",
                        content_type,
                        request_id
                    );
                }

                // Get the field data as a string
                kw_search_url = field.text().await.map_err(|e| {
                    let err_msg = format!("`kw_search_url` field should be a text field. {e}");
                    dual_error!("{} - request_id: {}", err_msg, request_id);
                    ServerError::Operation(err_msg)
                })?;
            }
            Some("kw_search_index_name") => {
                // Get content type if available
                if let Some(content_type) = field.content_type() {
                    dual_info!(
                        "Content type: {} - request_id: {}",
                        content_type,
                        request_id
                    );
                }

                // Get the field data as a string
                kw_search_index_name = field.text().await.map_err(|e| {
                    let err_msg =
                        format!("`kw_search_index_name` field should be a text field. {e}");
                    dual_error!("{} - request_id: {}", err_msg, request_id);
                    ServerError::Operation(err_msg)
                })?;
            }
            Some(field_name) => {
                let warn_msg = format!("Unknown field: {field_name}");
                dual_warn!("{} - request_id: {}", warn_msg, request_id);
            }
            None => {
                let warn_msg = "No field name found";
                dual_warn!("{} - request_id: {}", warn_msg, request_id);
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

    // * create index for the chunks for keyword search
    let mut index_response: Option<CreateIndexResponse> = None;
    match (kw_search_url.is_empty(), kw_search_index_name.is_empty()) {
        (false, false) => {
            match MCP_KEYWORD_SEARCH_CLIENT.get() {
                Some(mcp_client) => {
                    let mcp_name = mcp_client.read().await.name.clone();

                    match mcp_name.as_str() {
                        "gaia-keyword-search" => {
                            let documents: Vec<KwDocumentInput> = chunks
                                .iter()
                                .map(|c| KwDocumentInput {
                                    content: c.to_string(),
                                    title: None,
                                })
                                .collect();

                            // request param
                            let request_param = CallToolRequestParam {
                                name: "create_index".into(),
                                arguments: Some(serde_json::Map::from_iter([
                                    (
                                        "name".to_string(),
                                        serde_json::Value::from(kw_search_index_name),
                                    ),
                                    (
                                        "documents".to_string(),
                                        serde_json::Value::Array(
                                            documents
                                                .into_iter()
                                                .map(|d| serde_json::to_value(d).unwrap())
                                                .collect(),
                                        ),
                                    ),
                                ])),
                            };
                            dual_debug!(
                                "request_param: {:#?} - request_id: {}",
                                &request_param,
                                request_id
                            );

                            // call the create_index tool
                            let tool_result = mcp_client
                                .read()
                                .await
                                .raw
                                .peer()
                                .call_tool(request_param)
                                .await
                                .map_err(|e| {
                                    let err_msg = format!("Failed to call the tool: {e}");
                                    dual_error!("{} - request_id: {}", err_msg, request_id);
                                    ServerError::Operation(err_msg)
                                })?;

                            let response = CreateIndexResponse::from(tool_result);

                            dual_info!("Index created successfully - request_id: {}", request_id);

                            index_response = Some(response);
                        }
                        _ => {
                            let err_msg =
                                format!("Unsupported keyword search mcp server: {mcp_name}");
                            dual_error!("{} - request_id: {}", err_msg, request_id);
                            return Err(ServerError::Operation(err_msg));
                        }
                    }
                }
                None => {
                    let warn_msg = "No keyword search mcp client available";
                    dual_warn!("{} - request_id: {}", warn_msg, request_id);
                }
            }
        }
        (false, true) => {
            let warn_msg = "No keyword search index name provided";
            dual_warn!("{} - request_id: {}", warn_msg, request_id);
        }
        (true, false) => {
            let warn_msg = "No keyword search URL provided";
            dual_warn!("{} - request_id: {}", warn_msg, request_id);
        }
        (true, true) => {
            let warn_msg = "No keyword search URL and index name provided";
            dual_warn!("{} - request_id: {}", warn_msg, request_id);
        }
    }

    // compute the embeddings for each chunk
    dual_info!(
        "Compute embeddings for document chunks - request_id: {}",
        request_id
    );
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
        let servers = state.server_group.read().await;
        let embeddings_servers = match servers.get(&ServerKind::embeddings) {
            Some(servers) => servers,
            None => {
                let err_msg = "No embeddings server available";
                dual_error!("{} - request_id: {}", err_msg, request_id);
                return Err(ServerError::Operation(err_msg.to_string()));
            }
        };

        let embeddings_server_base_url = match embeddings_servers.next().await {
            Ok(url) => url,
            Err(e) => {
                let err_msg = format!("Failed to get the embeddings server: {e}");
                dual_error!("{} - request_id: {}", err_msg, request_id);
                return Err(ServerError::Operation(err_msg));
            }
        };
        let embeddings_service_url = format!(
            "{}/v1/embeddings",
            embeddings_server_base_url.url.trim_end_matches('/')
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
                        "Failed to forward the request to the downstream embedding server: {e}"
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

        response.json::<EmbeddingsResponse>().await.map_err(|e| {
            let err_msg = format!("Failed to parse the embedding response: {e}");
            dual_error!("{} - request_id: {}", err_msg, request_id);
            ServerError::Operation(err_msg)
        })?
    };
    let embeddings = embedding_response.data;

    dual_info!(
        "Got {} embeddings - request_id: {}",
        embeddings.len(),
        request_id
    );

    // * create a collection in VectorDB
    let dim = embeddings[0].embedding.len();
    rag::qdrant_create_collection(
        &vdb_server_url,
        &vdb_api_key,
        &vdb_collection_name,
        dim,
        &request_id,
    )
    .await?;

    // persist the embeddings to the collection
    rag::qdrant_persist_embeddings(
        &vdb_server_url,
        &vdb_api_key,
        &vdb_collection_name,
        embeddings.as_slice(),
        chunks.as_slice(),
        &request_id,
    )
    .await?;

    // create a response with status code 200. Content-Type is JSON
    let json_body = match index_response {
        Some(index_response) => serde_json::json!({
            "vdb": format!("Collection `{}` created successfully.", vdb_collection_name),
            "index": index_response,
        }),
        None => serde_json::json!({
            "vdb": format!("Collection `{}` created successfully.", vdb_collection_name),
        }),
    };

    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "application/json")
        .body(Body::from(json_body.to_string()))
        .map_err(|e| {
            let err_msg = format!("Failed to create response: {e}");
            dual_error!("{} - request_id: {}", err_msg, request_id);
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
    async fn verify_server(
        State(state): State<Arc<AppState>>,
        request_id: impl AsRef<str>,
        server_id: impl AsRef<str>,
        server_url: impl AsRef<str>,
        server_kind: &ServerKind,
    ) -> ServerResult<()> {
        let request_id = request_id.as_ref();
        let server_url = server_url.as_ref();
        let server_id = server_id.as_ref();

        let client = reqwest::Client::new();

        let server_info_url = format!("{server_url}/v1/info");
        let response = client.get(&server_info_url).send().await.map_err(|e| {
            let err_msg = format!("Failed to verify the {server_kind} downstream server: {e}",);
            dual_error!("{err_msg} - request_id: {request_id}");
            ServerError::Operation(err_msg)
        })?;

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

        // get the models from the downstream server
        let list_models_url = format!("{server_url}/v1/models");
        let list_models_response = client.get(&list_models_url).send().await.map_err(|e| {
            let err_msg = format!("Failed to get the models from the downstream server: {e}");
            dual_error!("{err_msg} - request_id: {request_id}");
            ServerError::Operation(err_msg)
        })?;

        let list_models_response = list_models_response
            .json::<ListModelsResponse>()
            .await
            .map_err(|e| {
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
