# LlamaEdge-Nexus

<!-- LlamaEdge-Nexus is a Rust-based RAG (Retrieval-Augmented Generation) service that integrates with LLM (Large Language Models) and vector databases to provide enhanced conversation and information retrieval capabilities.

## Features

- Provides RAG (Retrieval-Augmented Generation) functionality
- Supports integration with vector databases (such as Qdrant)
- Supports multiple RAG context merging strategies
- Provides Web UI interface
- Flexible configuration options

## Installation

### Prerequisites

- Rust and Cargo (version 1.75.0 or higher recommended)
- Vector database (such as Qdrant) for storing and retrieving vector embeddings

### Building from Source

```bash
git clone https://github.com/yourusername/LlamaEdge-Nexus.git
cd LlamaEdge-Nexus
cargo build --release
```

After building, the executable will be located in the `target/release/` directory. -->

## Usage

LlamaEdge-Nexus supports two running modes: Configuration File mode and Gaia mode.

### Common Command Line Options

Regardless of which mode you use, the following options are available:

```bash
--rag                      Enable RAG functionality (default: false)
--check-health             Enable health check for downstream servers (default: false)
--check-health-interval    Health check interval for downstream servers in seconds (default: 60)
--web-ui                   Root path for the Web UI files (default: "chatbot-ui")
--log-destination          Log destination: "stdout", "file", or "both" (default: "stdout")
--log-file                 Log file path (required when log_destination is "file" or "both")
```

### Configuration File Mode

Use a configuration file to set server and RAG-related options:

```bash
./llamaedge-nexus --rag --check-health config --file config.toml
```

### Gaia Mode

Use command line arguments to directly configure the application:

```bash
./llamaedge-nexus --rag --check-health gaia \
  --domain <gaia_domain> \
  --device-id <device_id> \
  --vdb-url <vector_db_url> \
  --vdb-collection-name <collection_name> \
  --vdb-limit <limit> \
  --vdb-score-threshold <threshold> \
  --rag-prompt <optional_custom_prompt>
```

#### Gaia Mode Parameters

- `--domain`: (Required) Gaia domain
- `--device-id`: (Required) Gaia device ID
- `--vdb-url`: Vector database URL (default: "http://localhost:6333")
- `--vdb-collection-name`: Vector database collection name (default: "default")
- `--vdb-limit`: Vector database result limit (default: 1)
- `--vdb-score-threshold`: Vector database score threshold (default: 0.5)
- `--rag-prompt`: Custom RAG prompt (optional)

## Configuration File

The configuration file (`config.toml`) contains the following sections:

### Server Configuration

```toml
[server]
host = "0.0.0.0"    # The host to listen on
port = 9068         # The port to listen on
```

### RAG Configuration

```toml
[rag]
prompt = ""                    # Custom RAG prompt (optional)
rag_policy = "system-message"  # Strategy for merging RAG context into chat messages
                               # Possible values: "system-message", "last-user-message"
context_window = 1             # Maximum number of user messages used in the retrieval

[rag.vector_db]                # Vector database configuration
url = "http://localhost:6333"  # The URL of the vector database
collection_name = ["default"]  # The name of the collection to use
limit = 10                     # The maximum number of results to return
score_threshold = 0.5          # The minimum score threshold for a result to be returned
```

<!-- ## Example Usage

### 1. Start the Server with a Configuration File

```bash
./llamaedge-nexus config --file config.toml
```

### 2. Start the Server with Gaia Mode

```bash
./llamaedge-nexus gaia \
  --domain example.com \
  --device-id device123 \
  --vdb-url http://localhost:6333 \
  --vdb-collection-name my_collection \
  --vdb-limit 5 \
  --vdb-score-threshold 0.7
```

### 3. Enable RAG

```bash
# enable RAG with a configuration file
./llamaedge-nexus --rag config --file config.toml

# enable RAG with Gaia mode
./llamaedge-nexus --rag gaia \
  --domain example.com \
  --device-id device123 \
  --vdb-url http://localhost:6333 \
  --vdb-collection-name my_collection \
  --vdb-limit 5 \
  --vdb-score-threshold 0.7
```

### 4. Enable Health Check

```bash
./llamaedge-nexus --check-health --check-health-interval 30 config --file config.toml
```

## API Endpoints

After starting the server, you can interact with it through HTTP APIs. The API endpoints include but are not limited to:

- Chat completion API
- Embedding API
- RAG retrieval API

For detailed information about the APIs, please refer to the `handlers.rs` and `rag.rs` files in the code.

## Notes

- Ensure the vector database is set up before use
- By default, the server listens on 0.0.0.0:9068
- When using RAG functionality, ensure that appropriate vector database settings are configured -->
