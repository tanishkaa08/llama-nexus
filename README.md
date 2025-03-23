# LlamaEdge-Nexus

## Usage

- Download LlamaEdge-Nexus binary

The LlamaEdge-Nexus binaries can be found at the [release page](https://github.com/llamaedge/llamaedge-nexus/releases). To download the binary, you can use the following command:

```bash
# Download the binary for Linux x86_64
curl -L https://github.com/llamaedge/llamaedge-nexus/releases/download/v0.1.0/llama-nexus-linux-x86_64 -o llama-nexus

# Download the binary for Linux ARM64
curl -L https://github.com/llamaedge/llamaedge-nexus/releases/download/v0.1.0/llama-nexus-linux-arm64 -o llama-nexus

# Download the binary for macOS x86_64
curl -L https://github.com/llamaedge/llamaedge-nexus/releases/download/v0.1.0/llama-nexus-macos-x86_64 -o llama-nexus

# Download the binary for macOS ARM64
curl -L https://github.com/llamaedge/llamaedge-nexus/releases/download/v0.1.0/llama-nexus-macos-arm64 -o llama-nexus
```

- Download LlamaEdge API Servers

LlamaEdge provides four types of API servers:

- `llama-api-server` provides chat and embedding APIs
- `whisper-api-server` provides audio transcription and translation APIs
- `sd-api-server` provides image generation and editing APIs
- `tts-api-server` provides text-to-speech APIs

For example, to download the `llama-api-server`, use the following command:

```bash
curl -L https://github.com/llamaedge/llamaedge-api-server/releases/download/v0.1.0/llama-api-server-linux-x86_64 -o llama-api-server
```

- Start LlamaEdge-Nexus

Run the following command to start LlamaEdge-Nexus:

```bash
./llama-nexus
```

- Register LlamaEdge API Servers to LlamaEdge-Nexus

Run the following commands to start LlamaEdge API Servers and register them to LlamaEdge-Nexus:

```bash
# Start LlamaEdge API Servers
./llama-api-server
./whisper-api-server
./sd-api-server
./tts-api-server

# Register LlamaEdge API Servers to LlamaEdge-Nexus
./llama-nexus register llama-api-server
```

## Command Line Usage

LlamaEdge-Nexus provides various command line options to configure the service behavior. You can specify the config file path, enable RAG functionality, set up health checks, configure the Web UI, and manage logging. Here are the available command line options by running `llama-nexus --help`:

```bash
LlamaEdge Nexus - A gateway service for LLM backends

Usage: llama-nexus [OPTIONS]

Options:
      --config <CONFIG>
          Path to the config file [default: config.toml]
      --rag
          Enable RAG
      --check-health
          Enable health check for downstream servers
      --check-health-interval <CHECK_HEALTH_INTERVAL>
          Health check interval for downstream servers in seconds [default: 60]
      --web-ui <WEB_UI>
          Root path for the Web UI files [default: chatbot-ui]
      --log-destination <LOG_DESTINATION>
          Log destination: "stdout", "file", or "both" [default: stdout]
      --log-file <LOG_FILE>
          Log file path (required when log_destination is "file" or "both")
  -h, --help
          Print help
  -V, --version
          Print version
```
