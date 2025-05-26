use crate::{
    dual_debug, dual_error, dual_info, dual_warn,
    error::{ServerError, ServerResult},
    mcp::{MCP_KEYWORD_SEARCH_CLIENT, MCP_VECTOR_SEARCH_CLIENT},
    AppState,
};
use axum::{
    extract::{Extension, State},
    http::HeaderMap,
    Json,
};
use chat_prompts::{error as ChatPromptsError, MergeRagContext, MergeRagContextPolicy};
use endpoints::{
    chat::{ChatCompletionRequest, ChatCompletionRequestMessage, ChatCompletionUserMessageContent},
    embeddings::{EmbeddingObject, EmbeddingRequest, EmbeddingsResponse, InputText},
    rag::vector_search::{DataFrom, RagScoredPoint, RetrieveObject},
};
use gaia_elastic_mcp_common::SearchResponse;
use gaia_kwsearch_common::{KwSearchHit, SearchDocumentsResponse};
use gaia_qdrant_common::{
    CreateCollectionResponse, Point, ScoredPoint, SearchPointsResponse, UpsertPointsResponse,
};
use rmcp::model::CallToolRequestParam;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::{hash_map::DefaultHasher, HashMap, HashSet},
    fmt,
    hash::{Hash, Hasher},
    sync::Arc,
};
use text_splitter::{MarkdownSplitter, TextSplitter};
use tokio_util::sync::CancellationToken;

const DEFAULT_FILTER_LIMIT: u64 = 10;
const DEFAULT_FILTER_SCORE_THRESHOLD: f32 = 0.5;
const DEFAULT_FILTER_WEIGHTED_ALPHA: f64 = 0.5;

pub async fn chat(
    State(state): State<Arc<AppState>>,
    Extension(cancel_token): Extension<CancellationToken>,
    headers: HeaderMap,
    Json(mut chat_request): Json<ChatCompletionRequest>,
) -> ServerResult<axum::response::Response> {
    let request_id = headers
        .get("x-request-id")
        .and_then(|h| h.to_str().ok())
        .unwrap_or("unknown")
        .to_string();

    dual_info!("Received a new chat request - request_id: {}", request_id);

    // * filter parameters
    let filter_limit = match chat_request.limit {
        Some(limit) => limit,
        None => DEFAULT_FILTER_LIMIT,
    };
    dual_debug!(
        "filter_limit: {} - request_id: {}",
        filter_limit,
        request_id
    );
    let filter_score_threshold = match chat_request.score_threshold {
        Some(score_threshold) => score_threshold,
        None => DEFAULT_FILTER_SCORE_THRESHOLD,
    };
    dual_debug!(
        "filter_score_threshold: {} - request_id: {}",
        filter_score_threshold,
        request_id
    );
    let weighted_alpha = match chat_request.weighted_alpha {
        Some(weighted_alpha) => weighted_alpha,
        None => DEFAULT_FILTER_WEIGHTED_ALPHA,
    };
    dual_debug!(
        "weighted_alpha: {} - request_id: {}",
        weighted_alpha,
        request_id
    );

    // * perform keyword search
    let mut kw_hits = Vec::new();
    match chat_request.messages.last() {
        Some(ChatCompletionRequestMessage::User(user_message)) => {
            match user_message.content() {
                ChatCompletionUserMessageContent::Text(text) => {
                    match MCP_KEYWORD_SEARCH_CLIENT.get() {
                        Some(mcp_client) => {
                            let mcp_name = mcp_client.read().await.name.clone();

                            match mcp_name.as_str() {
                                "gaia-keyword-search" => {
                                    let kw_search_url = match chat_request.kw_search_url.as_ref() {
                                        Some(url) if !url.is_empty() => {
                                            let url = url.trim_end_matches('/');

                                            dual_info!(
                                                    "URL to the kw-search mcp-server: {} - request_id: {}",
                                                    url,
                                                    request_id
                                                );

                                            url.to_string()
                                        }
                                        _ => {
                                            let err_msg = "Not found `kw_search_url` field in the request. `kw_search_url` field is required for kw-search-server. ";

                                            dual_error!("{} - request_id: {}", err_msg, request_id);

                                            return Err(ServerError::BadRequest(
                                                err_msg.to_string(),
                                            ));
                                        }
                                    };

                                    let kw_search_index = match chat_request
                                        .kw_search_index
                                        .as_ref()
                                    {
                                        Some(index) if !index.is_empty() => index.to_string(),
                                        _ => {
                                            let err_msg = "Not found `kw_search_index` field in the request. `kw_search_index` field is required for kw-search-server. ";

                                            dual_error!("{} - request_id: {}", err_msg, request_id);

                                            return Err(ServerError::BadRequest(
                                                err_msg.to_string(),
                                            ));
                                        }
                                    };

                                    // request param
                                    let request_param = CallToolRequestParam {
                                        name: "search_documents".into(),
                                        arguments: Some(serde_json::Map::from_iter([
                                            ("base_url".to_string(), Value::from(kw_search_url)),
                                            (
                                                "index_name".to_string(),
                                                Value::from(kw_search_index),
                                            ),
                                            ("query".to_string(), Value::from(text.to_string())),
                                            ("limit".to_string(), Value::from(filter_limit)),
                                        ])),
                                    };
                                    dual_debug!(
                                        "request_param: {:#?} - request_id: {}",
                                        &request_param,
                                        request_id
                                    );

                                    // call the search_documents tool
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

                                    let search_response =
                                        SearchDocumentsResponse::from(tool_result);
                                    kw_hits = search_response.hits;

                                    dual_info!(
                                        "Got {} kw-search hits - request_id: {}",
                                        kw_hits.len(),
                                        request_id
                                    );
                                }
                                "gaia-elastic-search" => {
                                    let es_search_url = match chat_request.es_search_url.as_ref() {
                                        Some(url) if !url.is_empty() => {
                                            let url = url.trim_end_matches('/');

                                            dual_info!(
                                                    "URL to the es-search mcp-server: {} - request_id: {}",
                                                    url,
                                                    request_id
                                                );

                                            url.to_string()
                                        }
                                        _ => {
                                            let err_msg = "Not found `es_search_url` field in the request. `es_search_url` field is required for Elasticsearch server. ";

                                            dual_error!("{} - request_id: {}", err_msg, request_id);

                                            return Err(ServerError::BadRequest(
                                                err_msg.to_string(),
                                            ));
                                        }
                                    };

                                    let es_search_index = match chat_request
                                        .es_search_index
                                        .as_ref()
                                    {
                                        Some(index) if !index.is_empty() => {
                                            let index = index.clone();

                                            dual_info!(
                                                "Index name: {} - request_id: {}",
                                                &index,
                                                request_id
                                            );

                                            index
                                        }
                                        _ => {
                                            let err_msg = "Not found `es_search_index` field in the request. `es_search_index` field is required for Elasticsearch server. ";

                                            dual_error!("{} - request_id: {}", err_msg, request_id);

                                            return Err(ServerError::BadRequest(
                                                err_msg.to_string(),
                                            ));
                                        }
                                    };

                                    // parse api key
                                    let api_key = match &chat_request.es_api_key {
                                        Some(api_key) => api_key.clone(),
                                        None => {
                                            let err_msg = "No elastic search api key provided";
                                            dual_error!("{} - request_id: {}", err_msg, request_id);
                                            return Err(ServerError::BadRequest(
                                                err_msg.to_string(),
                                            ));
                                        }
                                    };

                                    // parse fields to search
                                    let es_search_fields: Vec<Value> = match chat_request
                                        .es_search_fields
                                        .as_ref()
                                    {
                                        Some(fields) if !fields.is_empty() => {
                                            let fields = fields
                                                .iter()
                                                .map(|f| serde_json::Value::String(f.clone()))
                                                .collect();

                                            dual_info!(
                                                "Fields to search: {:?} - request_id: {}",
                                                &fields,
                                                request_id
                                            );

                                            fields
                                        }
                                        _ => {
                                            let err_msg = "Not found `es_search_fields` field in the request. `es_search_fields` field is required for Elasticsearch server. ";

                                            dual_error!("{} - request_id: {}", err_msg, request_id);

                                            return Err(ServerError::BadRequest(
                                                err_msg.to_string(),
                                            ));
                                        }
                                    };

                                    // request param
                                    let request_param = CallToolRequestParam {
                                        name: "search".into(),
                                        arguments: Some(serde_json::Map::from_iter([
                                            ("base_url".to_string(), Value::from(es_search_url)),
                                            ("index".to_string(), Value::from(es_search_index)),
                                            ("query".to_string(), Value::from(text.to_string())),
                                            ("fields".to_string(), Value::Array(es_search_fields)),
                                            ("size".to_string(), Value::from(filter_limit)),
                                            (
                                                "api_key".to_string(),
                                                serde_json::Value::String(api_key),
                                            ),
                                        ])),
                                    };

                                    // call tool
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

                                    // parse tool result
                                    let search_response = SearchResponse::from(tool_result);

                                    {
                                        if !search_response.hits.hits.is_empty() {
                                            for hit in search_response.hits.hits.iter() {
                                                let score = hit.score;
                                                let title = hit
                                                    .source
                                                    .get("title")
                                                    .unwrap()
                                                    .as_str()
                                                    .unwrap()
                                                    .to_string();
                                                let content = hit
                                                    .source
                                                    .get("content")
                                                    .unwrap()
                                                    .as_str()
                                                    .unwrap()
                                                    .to_string();

                                                let kw_hit = KwSearchHit {
                                                    title,
                                                    content,
                                                    score,
                                                };

                                                kw_hits.push(kw_hit);
                                            }
                                        }
                                    }

                                    dual_info!(
                                        "Got {} kw-search hits - request_id: {}",
                                        kw_hits.len(),
                                        request_id
                                    );
                                }
                                _ => {
                                    let err_msg = format!(
                                        "Unsupported keyword search mcp server: {mcp_name}"
                                    );
                                    dual_error!("{} - request_id: {}", err_msg, request_id);
                                    return Err(ServerError::Operation(err_msg));
                                }
                            }
                        }
                        None => {
                            let warn_msg = "No keyword search mcp server connected";
                            dual_warn!("{} - request_id: {}", warn_msg, request_id);
                        }
                    }
                }
                _ => {
                    let err_msg = "The last message in the request is not a text-only user message";
                    dual_error!("{} - request_id: {}", err_msg, request_id);
                    return Err(ServerError::BadRequest(err_msg.to_string()));
                }
            }
        }
        _ => {
            let err_msg = "The last message in the request is not a user message";
            dual_error!("{} - request_id: {}", err_msg, request_id);
            return Err(ServerError::BadRequest(err_msg.to_string()));
        }
    }

    // qdrant config
    let qdrant_config_vec = match get_qdrant_configs(
        &chat_request,
        filter_limit,
        filter_score_threshold,
        &request_id,
    )
    .await
    {
        Ok(qdrant_config_vec) => qdrant_config_vec,
        Err(e) => {
            let err_msg = format!("Failed to get the VectorDB config: {e}");
            dual_error!(
                "Failed to get the VectorDB config: {} - request_id: {}",
                e,
                request_id
            );
            return Err(ServerError::Operation(err_msg));
        }
    };

    // * perform vector search
    let mut retrieve_object_vec = retrieve_context_with_multiple_qdrant_configs(
        State(state.clone()),
        Extension(cancel_token.clone()),
        headers.clone(),
        &request_id,
        &chat_request,
        &qdrant_config_vec,
    )
    .await?;

    // log retrieve object
    dual_debug!(
        "request_id: {} - retrieve_object_vec:\n{}",
        request_id,
        serde_json::to_string_pretty(&retrieve_object_vec).unwrap()
    );

    // * rerank
    retrieve_object_vec = {
        // create a hash map from kw_hits: key is the hash value of the content of the hit, value is the hit
        let mut kw_hits_map = HashMap::new();
        let mut kw_scores = HashMap::new();
        if !kw_hits.is_empty() {
            for hit in kw_hits {
                let hash_value = calculate_hash(&hit.content);
                kw_scores.insert(hash_value, hit.score);
                kw_hits_map.insert(hash_value, hit);
            }

            dual_info!(
                "kw_hits_map: {:#?} - request_id: {}",
                &kw_hits_map,
                request_id
            );

            // normalize the kw_scores
            let kw_scores = normalize(&kw_scores);

            dual_info!("kw_scores: {:#?} - request_id: {}", &kw_scores, request_id);
        }

        // create a hash map from retrieve_object_vec: key is the hash value of the source of the point, value is the point
        let mut em_hits_map = HashMap::new();
        let mut em_scores = HashMap::new();
        if !retrieve_object_vec.is_empty() {
            let points = retrieve_object_vec[0].points.as_ref().unwrap().clone();
            if !points.is_empty() {
                for point in points {
                    let hash_value = calculate_hash(&point.source);
                    em_scores.insert(hash_value, point.score);
                    em_hits_map.insert(hash_value, point);
                }

                dual_info!(
                    "em_hits_map: {:#?} - request_id: {}",
                    &em_hits_map,
                    request_id
                );

                // normalize the em_scores
                let em_scores = normalize(&em_scores);

                dual_info!("em_scores: {:#?} - request_id: {}", &em_scores, request_id);
            }
        }

        // fuse the two hash maps
        let fused_scores = weighted_fusion(kw_scores, em_scores, weighted_alpha);

        dual_info!(
            "final_scores: {:#?} - request_id: {}",
            &fused_scores,
            request_id
        );

        // Sort by score from high to low
        let mut final_ranking: Vec<(u64, f64)> = fused_scores.into_iter().collect();
        final_ranking.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Print final ranking
        dual_info!(
            "final_ranking: {:#?} - request_id: {}",
            &final_ranking,
            request_id
        );

        let mut retrieved = Vec::new();
        for (hash_value, score) in final_ranking.iter() {
            if kw_hits_map.contains_key(hash_value) {
                retrieved.push(RagScoredPoint {
                    source: kw_hits_map[hash_value].content.clone(),
                    score: *score,
                    from: DataFrom::KeywordSearch,
                });
            } else if em_hits_map.contains_key(hash_value) {
                retrieved.push(RagScoredPoint {
                    source: em_hits_map[hash_value].source.clone(),
                    score: *score,
                    from: DataFrom::VectorSearch,
                });
            }
        }

        // filter the retrieved points by the score threshold
        let mut retrieved = Vec::new();
        for (hash_value, score) in final_ranking.iter() {
            if *score >= filter_score_threshold as f64 {
                if kw_hits_map.contains_key(hash_value) {
                    retrieved.push(RagScoredPoint {
                        source: kw_hits_map[hash_value].content.clone(),
                        score: *score,
                        from: DataFrom::KeywordSearch,
                    });
                } else if em_hits_map.contains_key(hash_value) {
                    retrieved.push(RagScoredPoint {
                        source: em_hits_map[hash_value].source.clone(),
                        score: *score,
                        from: DataFrom::VectorSearch,
                    });
                }
            }
        }

        // truncate the retrieved points to the limit
        if retrieved.len() > filter_limit as usize {
            retrieved.truncate(filter_limit as usize);
            dual_info!(
                "Keep {} retrieved points - request_id: {}",
                filter_limit,
                request_id
            );
        } else {
            dual_info!(
                "Keep {} retrieved points - request_id: {}",
                retrieved.len(),
                request_id
            );
        }

        dual_info!("retrieved: {:#?} - request_id: {}", &retrieved, request_id);

        let retrieve_object = RetrieveObject {
            limit: retrieved.len(),
            score_threshold: filter_score_threshold,
            points: Some(retrieved),
        };

        vec![retrieve_object]
    };

    // * generate context
    let mut context = String::new();
    if !retrieve_object_vec.is_empty() {
        for (idx, retrieve_object) in retrieve_object_vec.iter().enumerate() {
            match retrieve_object.points.as_ref() {
                Some(scored_points) => {
                    match scored_points.is_empty() {
                        false => {
                            for (idx, point) in scored_points.iter().enumerate() {
                                // log
                                dual_debug!(
                                    "request_id: {} - Point-{}, score: {}, source: {}",
                                    request_id,
                                    idx,
                                    point.score,
                                    &point.source
                                );

                                context.push_str(&point.source);
                                context.push_str("\n\n");
                            }
                        }
                        true => {
                            // log
                            dual_warn!("No point retrieved from the collection `{}` (score < threshold {}) - request_id: {}", qdrant_config_vec[idx].collection_name, qdrant_config_vec[idx].score_threshold, request_id);
                        }
                    }
                }
                None => {
                    // log
                    dual_warn!("No point retrieved from the collection `{}` (score < threshold {}) - request_id: {}", qdrant_config_vec[idx].collection_name, qdrant_config_vec[idx].score_threshold, request_id);
                }
            }
        }
        dual_debug!("request_id: {} - context:\n{}", request_id, context);
    }

    // * merge context into chat request
    if !context.is_empty() {
        if chat_request.messages.is_empty() {
            let err_msg = "Found empty chat messages";

            // log
            dual_error!("{} - request_id: {}", err_msg, request_id);

            return Err(ServerError::BadRequest(err_msg.to_string()));
        }

        // get the prompt template from the chat server
        let prompt_template = {
            let server_info = state.server_info.read().await;
            let chat_server = server_info
                .servers
                .iter()
                .find(|(_server_id, server)| server.chat_model.is_some());
            match chat_server {
                Some((_server_id, chat_server)) => {
                    let chat_model = chat_server.chat_model.as_ref().unwrap();
                    chat_model.prompt_template.unwrap()
                }
                None => {
                    let err_msg = "No chat server available";
                    dual_error!("{} - request_id: {}", err_msg, request_id);
                    return Err(ServerError::Operation(err_msg.to_string()));
                }
            }
        };

        // get the rag policy
        let (rag_policy, rag_prompt) = {
            let config = state.config.read().await;
            (config.rag.policy, config.rag.prompt.clone())
        };

        if let Err(e) = RagPromptBuilder::build(
            &mut chat_request.messages,
            &[context],
            prompt_template.has_system_prompt(),
            rag_policy,
            rag_prompt,
        ) {
            let err_msg = e.to_string();

            // log
            dual_error!("{} - request_id: {}", err_msg, request_id);

            return Err(ServerError::Operation(err_msg));
        }
    }

    // * perform chat completion
    crate::handlers::chat(
        State(state.clone()),
        Extension(cancel_token.clone()),
        headers,
        Json(chat_request),
    )
    .await
}

async fn get_qdrant_configs(
    chat_request: &ChatCompletionRequest,
    limit: u64,
    score_threshold: f32,
    request_id: impl AsRef<str>,
) -> Result<Vec<QdrantConfig>, ServerError> {
    let request_id = request_id.as_ref();

    match (
        chat_request.vdb_server_url.as_deref(),
        chat_request.vdb_collection_name.as_deref(),
    ) {
        (Some(url), Some(collection_name)) => {
            dual_info!(
                "Use the VectorDB settings from the request - request_id: {}",
                request_id
            );

            let collection_name_str = collection_name.join(",");

            dual_info!(
                "qdrant url: {}, collection name: {} - request_id: {}",
                url,
                collection_name_str,
                request_id
            );

            let mut qdrant_config_vec = vec![];
            for col_name in collection_name.iter() {
                qdrant_config_vec.push(QdrantConfig {
                    url: url.to_string(),
                    collection_name: col_name.to_string(),
                    limit,
                    score_threshold,
                });
            }

            Ok(qdrant_config_vec)
        }
        _ => {
            let err_msg = "The settings for vector search in the request are not correct. The `vdb_server_url` and `vdb_collection_name` fields in the request should be provided.";

            dual_error!("{} - request_id: {}", err_msg, request_id);

            Err(ServerError::Operation(err_msg.into()))
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct QdrantConfig {
    pub(crate) url: String,
    pub(crate) collection_name: String,
    pub(crate) limit: u64,
    pub(crate) score_threshold: f32,
}
impl fmt::Display for QdrantConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "url: {}, collection_name: {}, limit: {}, score_threshold: {}",
            self.url, self.collection_name, self.limit, self.score_threshold
        )
    }
}

async fn retrieve_context_with_multiple_qdrant_configs(
    State(state): State<Arc<AppState>>,
    Extension(cancel_token): Extension<CancellationToken>,
    headers: HeaderMap,
    request_id: impl AsRef<str>,
    chat_request: &ChatCompletionRequest,
    qdrant_config_vec: &[QdrantConfig],
) -> Result<Vec<RetrieveObject>, ServerError> {
    let mut retrieve_object_vec: Vec<RetrieveObject> = Vec::new();
    let mut set: HashSet<String> = HashSet::new();
    for qdrant_config in qdrant_config_vec {
        let mut retrieve_object = retrieve_context_with_single_qdrant_config(
            State(state.clone()),
            Extension(cancel_token.clone()),
            headers.clone(),
            request_id.as_ref(),
            chat_request,
            qdrant_config,
        )
        .await?;

        if let Some(points) = retrieve_object.points.as_mut() {
            if !points.is_empty() {
                // find the duplicate points
                let mut idx_removed = vec![];
                for (idx, point) in points.iter().enumerate() {
                    if set.contains(&point.source) {
                        idx_removed.push(idx);
                    } else {
                        set.insert(point.source.clone());
                    }
                }

                // remove the duplicate points
                if !idx_removed.is_empty() {
                    let num = idx_removed.len();

                    for idx in idx_removed.iter().rev() {
                        points.remove(*idx);
                    }

                    dual_info!(
                        "removed duplicated {} point(s) retrieved from the collection `{}` - request_id: {}",
                        num,
                        qdrant_config.collection_name,
                        request_id.as_ref()
                    );
                }

                if !points.is_empty() {
                    retrieve_object_vec.push(retrieve_object);
                }
            }
        }
    }

    Ok(retrieve_object_vec)
}

async fn retrieve_context_with_single_qdrant_config(
    State(state): State<Arc<AppState>>,
    Extension(cancel_token): Extension<CancellationToken>,
    headers: HeaderMap,
    request_id: impl AsRef<str>,
    chat_request: &ChatCompletionRequest,
    qdrant_config: &QdrantConfig,
) -> Result<RetrieveObject, ServerError> {
    let request_id = request_id.as_ref();

    dual_info!(
        "Computing embeddings for user query - request_id: {}",
        request_id
    );

    // get the context window from config
    let config_ctx_window = state.config.read().await.rag.context_window;

    // get context_window: chat_request.context_window prioritized CONTEXT_WINDOW
    let context_window = chat_request
        .context_window
        .or(Some(config_ctx_window))
        .unwrap_or(1);
    dual_info!(
        "Context window: {} - request_id: {}",
        context_window,
        request_id
    );

    // compute embeddings for user query
    let embedding_response = match chat_request.messages.is_empty() {
        true => {
            let err_msg = "Found empty chat messages";

            // log
            dual_error!("{} - request_id: {}", err_msg, request_id);

            return Err(ServerError::BadRequest(err_msg.to_string()));
        }
        false => {
            // get the last `n` user messages in the context window.
            // `n` is determined by the `context_window` in the chat request.
            let mut last_n_user_messages = Vec::new();
            for (idx, message) in chat_request.messages.iter().rev().enumerate() {
                if let ChatCompletionRequestMessage::User(user_message) = message {
                    if let ChatCompletionUserMessageContent::Text(text) = user_message.content() {
                        if !text.ends_with("<server-health>") {
                            last_n_user_messages.push(text.clone());
                        } else if idx == 0 {
                            let content = text.trim_end_matches("<server-health>").to_string();
                            last_n_user_messages.push(content);
                            break;
                        }
                    }
                }

                if last_n_user_messages.len() == context_window as usize {
                    break;
                }
            }

            // join the user messages in the context window into a single string
            let query_text = if !last_n_user_messages.is_empty() {
                dual_info!(
                    "Found the latest {} user message(s) - request_id: {}",
                    last_n_user_messages.len(),
                    request_id
                );

                last_n_user_messages.reverse();
                last_n_user_messages.join("\n")
            } else {
                let error_msg = "No user messages found.";

                // log
                dual_error!("{} - request_id: {}", error_msg, request_id);

                return Err(ServerError::BadRequest(error_msg.to_string()));
            };

            // log
            dual_info!(
                "Query text for the context retrieval: {} - request_id: {}",
                query_text,
                request_id
            );

            // create a embedding request
            let embedding_request = EmbeddingRequest {
                model: None,
                input: InputText::String(query_text),
                encoding_format: None,
                user: chat_request.user.clone(),
                vdb_server_url: None,
                vdb_collection_name: None,
                vdb_api_key: None,
            };

            // compute embeddings for query
            let response = crate::handlers::embeddings_handler(
                State(state.clone()),
                Extension(cancel_token.clone()),
                headers.clone(),
                Json(embedding_request),
            )
            .await?;

            // parse the response
            let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .map_err(|e| {
                    let err_msg = format!("Failed to parse embeddings response: {e}");

                    // log
                    dual_error!("{} - request_id: {}", err_msg, request_id);

                    ServerError::Operation(err_msg)
                })?;

            // parse the response
            serde_json::from_slice::<EmbeddingsResponse>(&bytes).map_err(|e| {
                let err_msg = format!("Failed to parse embeddings response: {e}");

                // log
                dual_error!("{} - request_id: {}", err_msg, request_id);

                ServerError::Operation(err_msg)
            })?
        }
    };

    let query_embedding: Vec<f32> = match embedding_response.data.first() {
        Some(embedding) => embedding.embedding.iter().map(|x| *x as f32).collect(),
        None => {
            let err_msg = "No embeddings returned";

            // log
            dual_error!("{} - request_id: {}", err_msg, request_id);

            return Err(ServerError::Operation(err_msg.to_string()));
        }
    };

    // get vdb_api_key if it is provided in the request, otherwise get it from the environment variable `VDB_API_KEY`
    let vdb_api_key = chat_request
        .vdb_api_key
        .clone()
        .or_else(|| std::env::var("VDB_API_KEY").ok());

    // perform the context retrieval
    let mut retrieve_object: RetrieveObject = match retrieve_context(
        query_embedding.as_slice(),
        &qdrant_config.url,
        &qdrant_config.collection_name,
        vdb_api_key,
        qdrant_config.limit as usize,
        Some(qdrant_config.score_threshold),
        request_id,
    )
    .await
    {
        Ok(search_result) => search_result,
        Err(e) => {
            let err_msg = format!("No point retrieved. {e}");

            // log
            dual_error!("{} - request_id: {}", err_msg, request_id);

            return Err(ServerError::Operation(err_msg));
        }
    };
    if retrieve_object.points.is_none() {
        retrieve_object.points = Some(Vec::new());
    }

    dual_info!(
        "Retrieved {} point(s) from the collection `{}` - request_id: {}",
        retrieve_object.points.as_ref().unwrap().len(),
        qdrant_config.collection_name,
        request_id
    );

    Ok(retrieve_object)
}

async fn retrieve_context(
    query_embedding: &[f32],
    vdb_server_url: impl AsRef<str>,
    vdb_collection_name: impl AsRef<str>,
    vdb_api_key: Option<String>,
    limit: usize,
    score_threshold: Option<f32>,
    request_id: impl AsRef<str>,
) -> Result<RetrieveObject, ServerError> {
    let request_id = request_id.as_ref();

    dual_info!(
        "Retrieve context from {}/collections/{}, max number of result to return: {}, score threshold: {} - request_id: {}",
        vdb_server_url.as_ref(),
        vdb_collection_name.as_ref(),
        limit,
        score_threshold.unwrap_or_default(),
        request_id
    );

    // search points from gaia-qdrant-mcp-server
    let scored_points = match MCP_VECTOR_SEARCH_CLIENT.get() {
        Some(mcp_client) => {
            // request param
            let request_param = match vdb_api_key {
                Some(api_key) => CallToolRequestParam {
                    name: "search_points".into(),
                    arguments: Some(serde_json::Map::from_iter([
                        (
                            "base_url".to_string(),
                            Value::from(vdb_server_url.as_ref().to_string()),
                        ),
                        ("api_key".to_string(), Value::from(api_key)),
                        (
                            "name".to_string(),
                            Value::from(vdb_collection_name.as_ref().to_string()),
                        ),
                        ("vector".to_string(), Value::from(query_embedding.to_vec())),
                        ("limit".to_string(), Value::from(limit as u64)),
                        (
                            "score_threshold".to_string(),
                            Value::from(score_threshold.unwrap_or(0.0)),
                        ),
                    ])),
                },
                None => CallToolRequestParam {
                    name: "search_points".into(),
                    arguments: Some(serde_json::Map::from_iter([
                        (
                            "base_url".to_string(),
                            Value::from(vdb_server_url.as_ref().to_string()),
                        ),
                        (
                            "name".to_string(),
                            Value::from(vdb_collection_name.as_ref().to_string()),
                        ),
                        ("vector".to_string(), Value::from(query_embedding.to_vec())),
                        ("limit".to_string(), Value::from(limit as u64)),
                        (
                            "score_threshold".to_string(),
                            Value::from(score_threshold.unwrap_or(0.0)),
                        ),
                    ])),
                },
            };

            // call the search_points tool
            let tool_result = mcp_client
                .read()
                .await
                .raw
                .peer()
                .call_tool(request_param)
                .await
                .map_err(|e| {
                    let err_msg = format!("Failed to call the search_points tool: {e}");
                    dual_error!("{} - request_id: {}", err_msg, request_id);
                    ServerError::Operation(err_msg)
                })?;

            // parse the response
            let response = SearchPointsResponse::from(tool_result);

            dual_info!(
                "Got {} points from the gaia-qdrant-mcp-server in {} seconds - request_id: {}",
                response.result.len(),
                response.time,
                request_id
            );

            response.result
        }
        None => {
            let err_msg = "No vector search mcp client available";
            dual_error!("{} - request_id: {}", err_msg, request_id);
            return Err(ServerError::Operation(err_msg.to_string()));
        }
    };

    dual_info!(
        "Try to remove duplicated points - request_id: {}",
        request_id
    );

    // remove duplicates, which have the same source
    let mut seen = HashSet::new();
    let unique_scored_points: Vec<ScoredPoint> = scored_points
        .into_iter()
        .filter(|point| seen.insert(point.payload.get("source").unwrap().to_string()))
        .collect();

    dual_debug!(
        "Found {} unique scored points - request_id: {}",
        unique_scored_points.len(),
        request_id
    );

    let ro = match unique_scored_points.is_empty() {
        true => RetrieveObject {
            points: None,
            limit,
            score_threshold: score_threshold.unwrap_or(0.0),
        },
        false => {
            let mut points: Vec<RagScoredPoint> = vec![];
            for point in unique_scored_points.iter() {
                if point.payload.is_empty() {
                    continue;
                }

                dual_debug!("point: {:?}", point);

                if let Some(source) = point.payload.get("source").and_then(Value::as_str) {
                    points.push(RagScoredPoint {
                        source: source.to_string(),
                        score: point.score,
                        from: DataFrom::VectorSearch,
                    })
                }

                // For debugging purpose, log the optional search field if it exists
                if let Some(search) = point.payload.get("search").and_then(Value::as_str) {
                    dual_info!("search: {} - request_id: {}", search, request_id);
                }
            }

            RetrieveObject {
                points: Some(points),
                limit,
                score_threshold: score_threshold.unwrap_or(0.0),
            }
        }
    };

    Ok(ro)
}

#[derive(Debug, Default)]
struct RagPromptBuilder;
impl MergeRagContext for RagPromptBuilder {
    fn build(
        messages: &mut Vec<endpoints::chat::ChatCompletionRequestMessage>,
        context: &[String],
        has_system_prompt: bool,
        policy: MergeRagContextPolicy,
        rag_prompt: Option<String>,
    ) -> ChatPromptsError::Result<()> {
        if messages.is_empty() {
            dual_error!("Found empty messages in the chat request.");

            return Err(ChatPromptsError::PromptError::NoMessages);
        }

        if context.is_empty() {
            let err_msg = "No context provided.";

            // log
            dual_error!("{}", &err_msg);

            return Err(ChatPromptsError::PromptError::Operation(err_msg.into()));
        }

        // check rag policy
        let mut policy = policy;
        if policy == MergeRagContextPolicy::SystemMessage && !has_system_prompt {
            // log
            dual_info!("The chat model does not support system message. Switch the currect rag policy to `last-user-message`");

            policy = MergeRagContextPolicy::LastUserMessage;
        }
        dual_info!("rag_policy: {}", &policy);

        let context = context[0].trim_end();
        dual_info!("context:\n{}", context);

        match policy {
            MergeRagContextPolicy::SystemMessage => {
                match &messages[0] {
                    ChatCompletionRequestMessage::System(message) => {
                        let system_message = {
                            match rag_prompt {
                                Some(global_rag_prompt) => {
                                    let processed_rag_prompt =
                                        global_rag_prompt.replace("\\n", "\n");
                                    // compose new system message content
                                    let content = format!(
                                        "{system_message}\n{rag_prompt}\n{context}",
                                        system_message = message.content().trim(),
                                        rag_prompt = &processed_rag_prompt,
                                        context = context
                                    );

                                    // create system message
                                    ChatCompletionRequestMessage::new_system_message(
                                        content,
                                        message.name().cloned(),
                                    )
                                }
                                None => {
                                    // compose new system message content
                                    let content = format!(
                                        "{system_message}\n{context}",
                                        system_message = message.content().trim(),
                                        context = context
                                    );

                                    // create system message
                                    ChatCompletionRequestMessage::new_system_message(
                                        content,
                                        message.name().cloned(),
                                    )
                                }
                            }
                        };

                        // replace the original system message
                        messages[0] = system_message;
                    }
                    _ => {
                        let system_message = match rag_prompt {
                            Some(global_rag_prompt) => {
                                let processed_rag_prompt = global_rag_prompt.replace("\\n", "\n");
                                // compose new system message content
                                let content = format!(
                                    "{rag_prompt}\n{context}",
                                    rag_prompt = &processed_rag_prompt,
                                    context = context
                                );

                                // create system message
                                ChatCompletionRequestMessage::new_system_message(content, None)
                            }
                            None => {
                                // create system message
                                ChatCompletionRequestMessage::new_system_message(
                                    context.to_string(),
                                    None,
                                )
                            }
                        };

                        // insert system message
                        messages.insert(0, system_message);
                    }
                }
            }
            MergeRagContextPolicy::LastUserMessage => {
                dual_info!("Merge RAG context into last user message.");

                let len = messages.len();
                match &messages.last() {
                    Some(ChatCompletionRequestMessage::User(message)) => {
                        if let ChatCompletionUserMessageContent::Text(content) = message.content() {
                            let content = match rag_prompt {
                                Some(global_rag_prompt) => {
                                    let processed_rag_prompt =
                                        global_rag_prompt.replace("\\n", "\n");

                                    // compose new user message content
                                    format!(
                                            "{rag_prompt}\n{context}\n\nAnswer the question based on the pieces of context above. The question is:\n{user_message}",
                                            rag_prompt = &processed_rag_prompt,
                                            context = context,
                                            user_message = content.trim(),
                                        )
                                }
                                None => {
                                    // compose new user message content
                                    format!(
                                            "{context}\n\nAnswer the question based on the pieces of context above. The question is:\n{user_message}",
                                            context = context,
                                            user_message = content.trim(),
                                        )
                                }
                            };

                            let content = ChatCompletionUserMessageContent::Text(content);

                            // create user message
                            let user_message = ChatCompletionRequestMessage::new_user_message(
                                content,
                                message.name().cloned(),
                            );
                            // replace the original user message
                            messages[len - 1] = user_message;
                        }
                    }
                    _ => {
                        let err_msg =
                            "The last message in the chat request should be a user message.";

                        // log
                        dual_error!("{}", &err_msg);

                        return Err(ChatPromptsError::PromptError::BadMessages(err_msg.into()));
                    }
                }
            }
        }

        Ok(())
    }
}

// Segment the given text into chunks
pub(crate) fn chunk_text(
    text: impl AsRef<str>,
    ty: impl AsRef<str>,
    chunk_capacity: usize,
    request_id: impl AsRef<str>,
) -> Result<Vec<String>, ServerError> {
    let request_id = request_id.as_ref();

    if ty.as_ref().to_lowercase().as_str() != "txt" && ty.as_ref().to_lowercase().as_str() != "md" {
        let err_msg = "Failed to upload the target file. Only files with 'txt' and 'md' extensions are supported.";

        dual_error!("{} - request_id: {}", err_msg, request_id);

        return Err(ServerError::Operation(err_msg.into()));
    }

    match ty.as_ref().to_lowercase().as_str() {
        "txt" => {
            dual_info!("Chunk the plain text contents - request_id: {}", request_id);

            // create a text splitter
            let splitter = TextSplitter::new(chunk_capacity);

            let chunks = splitter
                .chunks(text.as_ref())
                .map(|s| s.to_string())
                .collect::<Vec<_>>();

            dual_info!("{} chunks - request_id: {}", chunks.len(), request_id);

            Ok(chunks)
        }
        "md" => {
            dual_info!("Chunk the markdown contents - request_id: {}", request_id);

            // create a markdown splitter
            let splitter = MarkdownSplitter::new(chunk_capacity);

            let chunks = splitter
                .chunks(text.as_ref())
                .map(|s| s.to_string())
                .collect::<Vec<_>>();

            dual_info!(
                "Number of chunks: {} - request_id: {}",
                chunks.len(),
                request_id
            );

            Ok(chunks)
        }
        _ => {
            let err_msg =
                "Failed to upload the target file. Only text and markdown files are supported.";

            dual_error!("{}", err_msg);

            Err(ServerError::Operation(err_msg.into()))
        }
    }
}

pub(crate) async fn qdrant_create_collection(
    vdb_server_url: impl AsRef<str>,
    vdb_api_key: impl AsRef<str>,
    collection_name: impl AsRef<str>,
    dim: usize,
    request_id: impl AsRef<str>,
) -> Result<(), ServerError> {
    let request_id = request_id.as_ref();

    dual_info!(
        "Create a collection `{}` of {} dimensions - request_id: {}",
        collection_name.as_ref(),
        dim,
        request_id
    );

    match MCP_VECTOR_SEARCH_CLIENT.get() {
        Some(mcp_client) => {
            // request param
            let request_param = match vdb_api_key.as_ref().is_empty() {
                true => CallToolRequestParam {
                    name: "create_collection".into(),
                    arguments: Some(serde_json::Map::from_iter([
                        (
                            "base_url".to_string(),
                            serde_json::Value::from(vdb_server_url.as_ref().to_string()),
                        ),
                        (
                            "name".to_string(),
                            serde_json::Value::String(collection_name.as_ref().to_string()),
                        ),
                        ("size".to_string(), serde_json::Value::from(dim as u64)),
                    ])),
                },
                false => CallToolRequestParam {
                    name: "create_collection".into(),
                    arguments: Some(serde_json::Map::from_iter([
                        (
                            "base_url".to_string(),
                            serde_json::Value::from(vdb_server_url.as_ref().to_string()),
                        ),
                        (
                            "api_key".to_string(),
                            serde_json::Value::String(vdb_api_key.as_ref().to_string()),
                        ),
                        (
                            "name".to_string(),
                            serde_json::Value::String(collection_name.as_ref().to_string()),
                        ),
                        ("size".to_string(), serde_json::Value::from(dim as u64)),
                    ])),
                },
            };

            // call the create_collection tool
            let tool_result = mcp_client
                .read()
                .await
                .raw
                .peer()
                .call_tool(request_param)
                .await
                .map_err(|e| {
                    let err_msg = format!("Failed to call the create_collection tool: {e}");
                    dual_error!("{} - request_id: {}", err_msg, request_id);
                    ServerError::Operation(err_msg)
                })?;

            // parse the response
            let response = CreateCollectionResponse::from(tool_result);

            if response.result {
                dual_info!(
                    "Collection `{}` created successfully - request_id: {}",
                    collection_name.as_ref(),
                    request_id
                );
                Ok(())
            } else {
                let err_msg = format!("Failed to create collection `{}`", collection_name.as_ref());
                dual_error!("{} - request_id: {}", err_msg, request_id);
                Err(ServerError::Operation(err_msg))
            }
        }
        None => {
            let err_msg = "No vector search mcp client found";
            dual_error!("{} - request_id: {}", err_msg, request_id);
            Err(ServerError::Operation(err_msg.to_string()))
        }
    }
}

pub(crate) async fn qdrant_persist_embeddings(
    vdb_server_url: impl AsRef<str>,
    vdb_api_key: impl AsRef<str>,
    collection_name: impl AsRef<str>,
    embeddings: &[EmbeddingObject],
    chunks: &[String],
    request_id: impl AsRef<str>,
) -> Result<(), ServerError> {
    let request_id = request_id.as_ref();

    dual_info!(
        "Persist embeddings to the Qdrant instance - request_id: {}",
        request_id
    );

    let mut points = Vec::<Point>::new();
    for embedding in embeddings {
        // convert the embedding to a vector
        let vector: Vec<_> = embedding.embedding.iter().map(|x| *x as f32).collect();

        // create a payload
        let payload = serde_json::json!({"source": chunks[embedding.index as usize]})
            .as_object()
            .map(|m| m.to_owned());

        // create a point
        let point = Point {
            id: embedding.index,
            vector,
            payload: payload.unwrap_or_default(),
        };

        points.push(point);
    }

    dual_info!(
        "{} points to be upserted - request_id: {}",
        points.len(),
        request_id
    );

    match MCP_VECTOR_SEARCH_CLIENT.get() {
        Some(mcp_client) => {
            // request param
            let request_param = match vdb_api_key.as_ref().is_empty() {
                true => CallToolRequestParam {
                    name: "upsert_points".into(),
                    arguments: Some(serde_json::Map::from_iter([
                        (
                            "base_url".to_string(),
                            serde_json::Value::from(vdb_server_url.as_ref().to_string()),
                        ),
                        (
                            "name".to_string(),
                            serde_json::Value::from(collection_name.as_ref().to_string()),
                        ),
                        (
                            "points".to_string(),
                            serde_json::Value::Array(
                                points
                                    .into_iter()
                                    .map(|p| serde_json::to_value(p).unwrap())
                                    .collect(),
                            ),
                        ),
                    ])),
                },
                false => CallToolRequestParam {
                    name: "upsert_points".into(),
                    arguments: Some(serde_json::Map::from_iter([
                        (
                            "base_url".to_string(),
                            serde_json::Value::from(vdb_server_url.as_ref().to_string()),
                        ),
                        (
                            "api_key".to_string(),
                            serde_json::Value::String(vdb_api_key.as_ref().to_string()),
                        ),
                        (
                            "name".to_string(),
                            serde_json::Value::from(collection_name.as_ref().to_string()),
                        ),
                        (
                            "points".to_string(),
                            serde_json::Value::Array(
                                points
                                    .into_iter()
                                    .map(|p| serde_json::to_value(p).unwrap())
                                    .collect(),
                            ),
                        ),
                    ])),
                },
            };

            // call the upsert_points tool
            let tool_result = mcp_client
                .read()
                .await
                .raw
                .peer()
                .call_tool(request_param)
                .await
                .map_err(|e| {
                    let err_msg = format!("Failed to call the upsert_points tool: {e}");
                    dual_error!("{} - request_id: {}", err_msg, request_id);
                    ServerError::Operation(err_msg)
                })?;
            let response = UpsertPointsResponse::from(tool_result);

            dual_info!(
                "Upsert points - Status: {} - Time: {} - request_id: {}",
                response.status,
                response.time,
                request_id
            );

            Ok(())
        }
        None => {
            let err_msg = "No vector search mcp client found";
            dual_error!("{} - request_id: {}", err_msg, request_id);
            Err(ServerError::Operation(err_msg.to_string()))
        }
    }
}

fn calculate_hash(s: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    s.hash(&mut hasher);
    hasher.finish()
}

/// Normalize scores with min-max normalization
fn normalize(scores: &HashMap<u64, f64>) -> HashMap<u64, f64> {
    if scores.is_empty() {
        return scores.clone();
    }

    let min_score = scores.values().cloned().fold(f64::INFINITY, f64::min);
    let max_score = scores.values().cloned().fold(f64::NEG_INFINITY, f64::max);

    dual_debug!(
        "Normalize scores: min_score: {}, max_score: {}",
        min_score,
        max_score
    );

    scores
        .iter()
        .map(|(&doc_id, &score)| {
            let normalized_score = if max_score - min_score > 0.0 {
                (score - min_score) / (max_score - min_score)
            } else {
                0.0
            };

            dual_debug!(
                "Normalize score: doc_id: {}, score: {}, normalized_score: {}",
                doc_id,
                score,
                normalized_score
            );

            (doc_id, normalized_score)
        })
        .collect()
}

/// Fuse keyword search and vector search scores with min-max normalization and weighted fusion
fn weighted_fusion(
    kw_search_scores: HashMap<u64, f64>,
    vector_search_scores: HashMap<u64, f64>,
    alpha: f64,
) -> HashMap<u64, f64> {
    match (kw_search_scores.is_empty(), vector_search_scores.is_empty()) {
        (false, false) => {
            dual_info!("Fusing keyword search and vector search scores");

            // Normalize keyword search scores
            let kw_normalized = normalize(&kw_search_scores);
            // Normalize vector search scores
            let vector_normalized = normalize(&vector_search_scores);

            // filter out duplicates
            let all_doc_ids: HashSet<u64> = kw_search_scores
                .keys()
                .chain(vector_search_scores.keys())
                .cloned()
                .collect();

            // Calculate fusion scores
            all_doc_ids
                .into_iter()
                .map(|doc_id| {
                    let k_score = *kw_normalized.get(&doc_id).unwrap_or(&0.0);
                    let v_score = *vector_normalized.get(&doc_id).unwrap_or(&0.0);
                    let fused_score = alpha * k_score + (1.0 - alpha) * v_score;
                    (doc_id, fused_score)
                })
                .collect()
        }
        (false, true) => {
            dual_info!("Only keyword search scores are available in the fusion");

            // Normalize keyword search scores
            normalize(&kw_search_scores)
        }
        (true, false) => {
            dual_info!("Only vector search scores are available in the fusion");

            // Normalize vector search scores
            normalize(&vector_search_scores)
        }
        (true, true) => {
            // Return empty HashMap
            HashMap::new()
        }
    }
}
