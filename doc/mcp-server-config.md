# Configuring MCP Servers in Llama-nexus Chat Completion Requests

Llama-nexus supports the specification of MCP servers in chat completion requests. This document demonstrates how to configure one or more MCP servers through the chat completion request parameters.

## Parameter Specification

Llama-nexus enables users to specify MCP servers in chat completion requests through the following parameters:

- `mcp_tools`: An array of MCP server configurations. Each configuration includes the following fields:
  - `type`: The type of the tool. Currently, only `mcp` is supported.
  - `server_label`: The name of the MCP server.
  - `server_url`: The URL of the MCP server. Note that if `transport` is set to `stream-http`, the `server_url` must end with `/mcp`; if `transport` is `sse`, the `server_url` must end with `/sse`. See the [example](#example) below.
  - `transport`: The type of the transport for the MCP server. Possible values are `stream-http` or `sse`.

## Example

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
