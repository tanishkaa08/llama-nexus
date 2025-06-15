# Configuring MCP Servers in Chat Completion Requests

Llama-nexus supports the specification of MCP servers in chat completion requests. This document demonstrates how to configure one or more MCP servers through the chat completion request parameters.

## Parameter Specification

Llama-nexus enables users to specify MCP servers in chat completion requests through the following parameters:

- `mcp_tools`: An array of MCP server configurations. Each configuration includes the following fields:
  - `type`: The type of the tool. Currently, only `mcp` is supported.
  - `server_label`: The name of the MCP server.
  - `server_url`: The URL of the MCP server. The URL suffix must match the transport protocol: use `/mcp` for `stream-http` transport and `/sse` for `sse` transport. See the [example](#example) below.
  - `transport`: The type of the transport for the MCP server. Possible values are `stream-http` or `sse`.

- `kw_search_mcp_tool`: An MCP server configuration for the keyword search mcp server.
  - `type`: The type of the tool. Currently, only `mcp` is supported.
  - `server_label`: The name of the keyword search MCP server.
  - `server_url`: The URL of the keyword search MCP server. The URL suffix must match the transport protocol: use `/mcp` for `stream-http` transport and `/sse` for `sse` transport. See the [example](#example) below.
  - `transport`: The type of the transport for the keyword search MCP server. Possible values are `stream-http` or `sse`.

> [!IMPORTANT]
> The parameters `mcp_tools` and `kw_search_mcp_tool` are still under development. The current implementation is not stable and may change in the future.

## Example

### Example 1: Configure two MCP servers

In the following example, the user configures two MCP servers, `gaia_calculator` and `gaia_weather`, through the `mcp_tools` parameter in the chat completion request. The `gaia_calculator` uses the `stream-http` transport, while `gaia_weather` uses the `sse` transport.

```bash
curl --location 'http://localhost:3389/v1/chat/completions' \
--header 'Content-Type: application/json' \
--data '{
    "model": "Qwen3-4B",
    "messages": [
        {
            "role": "user",
            "content": "What is the summation of 23 and 32?"
        }
    ],
    "mcp_tools": [
        {
            "type": "mcp",
            "server_label": "gaia_calculator",
            "server_url": "http://127.0.0.1:8001/mcp",
            "transport": "stream-http"
        },
        {
            "type": "mcp",
            "server_label": "gaia_weather",
            "server_url": "http://127.0.0.1:8002/sse",
            "transport": "sse"
        }
    ],
    "stream": false
}'
```

### Example 2: Configure a keyword search MCP server

Llama-nexus currently supports three types of keyword search MCP servers:

- `gaia-elastic-mcp-server`: The keyword search MCP server for Elasticsearch.
- `gaia-tidb-mcp-server`: The keyword search MCP server for TiDB.
- `gaia-kwsearch-mcp-server`: The keyword search MCP server for [kw-search-server](https://github.com/LlamaEdge/kw-search-server/tree/main).

#### Set `gaia-elastic-mcp-server` in chat completion request

The following example CURL command sets the `gaia-elastic-mcp-server` in the chat completion request:

```bash
curl --location 'http://localhost:3389/v1/chat/completions' \
--header 'Content-Type: application/json' \
--data '{
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer questions as concisely as possible."
        },
        {
            "role": "user",

            "content": "What risks does financial management involve?"
        }
    ],
    "kw_search_mcp_tool": {
        "type": "mcp",
        "server_label": "gaia-elastic-mcp-server",
        "server_url": "http://127.0.0.1:8006/mcp",
        "transport": "stream-http"
    },
    "vdb_collection_name": ["paris-03"],
    "limit": 5,
    "score_threshold": 0.5,
    "model": "Qwen3-4B",
    "stream": false
}'
```

#### Set `gaia-tidb-mcp-server` in chat completion request

The following example CURL command sets the `gaia-tidb-mcp-server` in the chat completion request:

```bash
curl --location 'http://localhost:3389/v1/chat/completions' \
--header 'Content-Type: application/json' \
--data '{
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer questions as concisely as possible."
        },
        {
            "role": "user",

            "content": "bluetooth earphone"
        }
    ],
    "kw_search_mcp_tool": {
        "type": "mcp",
        "server_label": "gaia-tidb-mcp-server",
        "server_url": "http://127.0.0.1:8007/mcp",
        "transport": "stream-http"
    },
    "vdb_collection_name": [
        "paris-03"
    ],
    "limit": 5,
    "score_threshold": 0.5,
    "model": "Qwen3-4B",
    "stream": false
}'
```

#### Set `gaia-kwsearch-mcp-server` in chat completion request

The following example CURL command sets the `gaia-kwsearch-mcp-server` in the chat completion request:

```bash
curl --location 'http://localhost:3389/v1/chat/completions' \
--header 'Content-Type: application/json' \
--data '{
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful AI assistant."
        },
        {
            "role": "user",

            "content": "What risks does financial management involve?"
        }
    ],
    "kw_search_mcp_tool": {
        "type": "mcp",
        "server_label": "gaia-kwsearch-mcp-server",
        "server_url": "http://127.0.0.1:8005/mcp",
        "transport": "stream-http"
    },
    "vdb_collection_name": [
        "xibei-01"
    ],
    "limit": 5,
    "score_threshold": 0.5,
    "model": "Qwen3-4B",
    "stream": false
}'
```
