#!/bin/bash

# Exit on error
set -e

# Function to detect OS and architecture
detect_os_arch() {
    OS=$(uname -s)
    ARCH=$(uname -m)

    if [ "$OS" = "Darwin" ]; then
        if [ "$ARCH" = "arm64" ]; then
            echo "apple-darwin-aarch64"
        else
            echo "apple-darwin-x86_64"
        fi
    else
        if [ "$ARCH" = "aarch64" ]; then
            echo "unknown-linux-gnu-aarch64"
        else
            echo "unknown-linux-gnu-x86_64"
        fi
    fi
}

# Function to download and extract file
download_and_extract() {
    local url=$1
    local output_file=$2
    local extract_file=$3

    echo "Downloading $output_file..."
    curl -LO "$url"
    echo "Extracting $extract_file..."
    tar -xvzf "$output_file" "$extract_file"
}

# Set versions
NEXUS_VERSION="0.1.0"
API_SERVER_VERSION="0.19.0"
GAIA_MCP_VERSION="0.1.2"
WASMEDGE_VERSION="0.14.1"
GGML_PLUGIN="b5361"

# Detect OS and architecture
PLATFORM=$(detect_os_arch)


# Set colors
RED=$'\e[0;31m'
GREEN=$'\e[0;32m'
YELLOW=$'\e[0;33m'
NC=$'\e[0m'

info() {
    printf "${GREEN}$1${NC}\n\n"
}

error() {
    printf "${RED}$1${NC}\n\n"
}

warning() {
    printf "${YELLOW}$1${NC}\n\n"
}

# 0. check if 3389 is already in use. If so, print error and exit.
if lsof -i :3389 > /dev/null; then
    error "\nPort 3389 is already in use. Please guarantee that 3389 is not in use and try again."
    exit 1
fi


# 1. Download llama-nexus
printf "Step 1: Downloading llama-nexus...\n"
curl -LO "https://github.com/LlamaEdge/llama-nexus/releases/download/${NEXUS_VERSION}/llama-nexus-${PLATFORM}.tar.gz"
tar -xvzf llama-nexus-${PLATFORM}.tar.gz llama-nexus config.toml
rm llama-nexus-${PLATFORM}.tar.gz
info "\n        👍 Done!"

# 2. Install WasmEdge Runtime
printf "Step 2: Installing WasmEdge Runtime...\n"
curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install_v2.sh | bash -s -- -v $WASMEDGE_VERSION --ggmlbn=$GGML_PLUGIN
source $HOME/.wasmedge/env
info "\n        👍 Done!"

# 3. Download llama-api-server and models
printf "Step 3: Downloading llama-api-server...\n"
curl -LO "https://github.com/LlamaEdge/LlamaEdge/releases/download/${API_SERVER_VERSION}/llama-api-server.wasm"
info "\n        👍 Done!"

# 4. Download Qwen3-4B-Q5_K_M.gguf
printf "Step 4: Downloading Qwen3-4B-Q5_K_M.gguf...\n"
if [ ! -f "Qwen3-4B-Q5_K_M.gguf" ]; then
    curl -LO "https://huggingface.co/second-state/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q5_K_M.gguf"
    info "\n        👍 Done!"
else
    echo "Qwen3-4B-Q5_K_M.gguf already exists"
fi

# 5. Download nomic-embed-text-v1.5-f16.gguf
printf "Step 5: Downloading nomic-embed-text-v1.5-f16.gguf...\n"
if [ ! -f "nomic-embed-text-v1.5-f16.gguf" ]; then
    curl -LO "https://huggingface.co/second-state/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5-f16.gguf"
    info "\n        👍 Done!"
else
    echo "nomic-embed-text-v1.5-f16.gguf already exists"
fi

# 6. Download MCP servers
printf "Step 6: Downloading MCP servers...\n"
curl -LO "https://github.com/apepkuss/mcp-examples/releases/download/${GAIA_MCP_VERSION}/gaia-mcp-servers-${PLATFORM}.tar.gz"
tar -xvzf gaia-mcp-servers-${PLATFORM}.tar.gz gaia-qdrant-mcp-server-sse gaia-kwsearch-mcp-server-sse
rm gaia-mcp-servers-${PLATFORM}.tar.gz
info "\n        👍 Done!"

# 7. Start services
printf "Step 7: Starting services...\n"

# Start chat server
printf "  * Starting chat server...\n"
nohup wasmedge --dir .:. --nn-preload default:GGML:AUTO:Qwen3-4B-Q5_K_M.gguf \
    llama-api-server.wasm \
    --model-name Qwen3-4B \
    --prompt-template qwen3-no-think \
    --ctx-size 8192 --port 9068 > chat-server.log 2>&1 &
sleep 5
info "\n    👍 Done!"

# Start embedding server
printf "  * Starting embedding server...\n"
nohup wasmedge --dir .:. --nn-preload default:GGML:AUTO:nomic-embed-text-v1.5-f16.gguf \
    llama-api-server.wasm \
    --prompt-template embedding \
    --ctx-size 768 \
    --model-name nomic-embed-text-v1.5 --port 9069 > embedding-server.log 2>&1 &
sleep 5
info "\n    👍 Done!"

# Start MCP servers
printf "  * Starting MCP servers...\n"
nohup ./gaia-qdrant-mcp-server-sse > gaia-qdrant-mcp-server-sse.log 2>&1 &
sleep 2
nohup ./gaia-kwsearch-mcp-server-sse > gaia-kwsearch-mcp-server-sse.log 2>&1 &
sleep 2
info "\n    👍 Done!"

# Start llama-nexus
printf "  * Starting llama-nexus...\n"
nohup ./llama-nexus > llama-nexus.log 2>&1 &
sleep 5
info "\n    👍 Done!"

# 8. Register servers
printf "Step 8: Registering servers...\n"

printf "  * Registering chat server...\n"
curl --location 'http://localhost:3389/admin/servers/register' \
    --header 'Content-Type: application/json' \
    --data '{
        "url": "http://localhost:9068",
        "kind": "chat"
    }'
info "\n    👍 Done!"

printf "  * Registering embedding server...\n"
curl --location 'http://localhost:3389/admin/servers/register' \
    --header 'Content-Type: application/json' \
    --data '{
        "url": "http://localhost:9069",
        "kind": "embeddings"
    }'
info "\n    👍 Done!"

info "\nAll services have been started and registered successfully!"
info "llama-nexus is running on port 3389"
info "Chat server is running on port 9068"
info "Embedding server is running on port 9069"
info "gaia-qdrant MCP server is running on port 8003"
info "gaia-keyword-search MCP server is running on port 8005"