# 🚀 MCP Server for Document Processing

This project implements a Model Context Protocol (MCP) server that processes Markdown and text files, chunks and tokenizes the content using embedding models, and makes this processed content available through MCP tools.

## 🏗️ Architecture

The system consists of two main components:

1. **📝 Processing Pipeline**: Reads Markdown and text files, chunks them, generates embeddings, and stores them in a vector database.
2. **🔌 MCP Server**: Exposes the processed content through MCP tools, allowing Roo Code to search and retrieve relevant information.

## ✅ Prerequisites

- Docker and Docker Compose
- OpenAI API key (required for embeddings unless using a custom embedding function)
- Anthropic API key (optional, for Claude response generation)

## 🛠️ Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/donphi/mcp-server.git
   cd mcp-server
   ```

2. Create a `.env` file with your configuration:
   ```
   # Copy the example file
   cp .env.example .env
   
   # Edit the file with your settings
   nano .env
   ```

3. Place your Markdown (.md) and text (.txt) files in the `data/` directory.

## ⚙️ Configuration

You can configure the MCP server using environment variables in the `.env` file:

```
# API Keys
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here  # Optional

# Pipeline Configuration
CHUNK_SIZE=1000                # Size of text chunks
CHUNK_OVERLAP=200              # Overlap between chunks (in tokens)
BATCH_SIZE=10                  # Batch size for embedding generation
EMBEDDING_MODEL=text-embedding-ada-002  # OpenAI embedding model to use
SUPPORTED_EXTENSIONS=.md,.txt  # Comma-separated list of supported file extensions

# Server Configuration
CLAUDE_MODEL=claude-3-7-sonnet-20240307  # Claude model to use
MAX_RESULTS=10                 # Maximum number of results to return
USE_ANTHROPIC=true             # Whether to use Anthropic API for responses

# Custom Embedding Model (optional)
# CUSTOM_EMBEDDING_MODULE=/path/to/your/module.py
# CUSTOM_EMBEDDING_FUNCTION=your_embedding_function

# Paths
DATA_DIR=/data                 # Directory containing input files
OUTPUT_DIR=/output             # Directory for output files
DB_PATH=/db                    # Directory for vector database
CONFIG_PATH=/config/server_config.json  # Path to server configuration file
```

## 🚀 Usage

### 🔄 Processing the Files

To process your files and generate embeddings:

```bash
docker-compose build pipeline
docker-compose run pipeline
```

This will:
- Read all supported files in the `data/` directory
- Process and chunk the content
- Generate embeddings
- Store the embeddings in the vector database

### 🏃‍♂️ Running the MCP Server

To start the MCP server:

```bash
docker-compose build server
docker-compose up -d server
```

### 🔗 Connecting to Roo Code

Configure Roo Code to connect to the MCP server by adding the following to your Roo Code configuration:

```json
{
  "mcpServers": {
    "mcp-server": {
      "command": "docker",
      "args": [
        "exec",
        "-i",
        "mcp_server_1",
        "python",
        "server.py"
      ]
    }
  }
}
```

## 🧰 MCP Tools

The MCP server exposes the following tools:

- **📚 read-md-files**: Process and retrieve files
- **🔍 search-content**: Search across processed content
- **📋 get-context**: Retrieve contextual information
- **🏗️ project-structure**: Provide project structure information
- **💡 suggest-implementation**: Generate implementation suggestions

## 📄 Supported File Types

By default, the following file types are supported:
- Markdown files (.md)
- Text files (.txt)

You can configure additional file extensions by setting the `SUPPORTED_EXTENSIONS` environment variable in your `.env` file.

## 🔄 Operational Modes

The MCP server can operate in two modes:

1. **🤖 Full Processing Mode**: When the Anthropic API key is provided and `USE_ANTHROPIC` is set to `true`, the server will use Claude to generate responses based on the retrieved context.

2. **📋 Context Retrieval Mode**: When the Anthropic API key is not provided or `USE_ANTHROPIC` is set to `false`, the server will only retrieve and return the relevant context, allowing the client (e.g., Roo Code) to process it using its own LLM.

## 🔧 Custom Embedding Models

You can use your own embedding model instead of OpenAI's by implementing a custom embedding function:

1. Create a Python module with your embedding function:

```python
# custom_embeddings.py

def my_embedding_function(texts):
    """
    Generate embeddings for a list of texts.
    
    Args:
        texts: List of texts to embed
        
    Returns:
        List of embeddings (each embedding is a list of floats)
    """
    # Your embedding logic here
    # ...
    return embeddings  # List of embeddings
```

2. Set the environment variables in your `.env` file:

```
CUSTOM_EMBEDDING_MODULE=/path/to/custom_embeddings.py
CUSTOM_EMBEDDING_FUNCTION=my_embedding_function
```

## 📁 Project Structure

```
mcp-server/
├── Dockerfile.pipeline
├── Dockerfile.server
├── docker-compose.yml
├── requirements.pipeline.txt
├── requirements.server.txt
├── README.md
├── .env.example
├── src/
│   ├── pipeline.py
│   ├── server.py
│   └── utils/
│       ├── __init__.py
│       ├── chunking.py
│       ├── embedding.py
│       └── vector_db.py
├── config/
│   ├── pipeline_config.json
│   └── server_config.json
├── data/
├── output/
└── db/
```

## 📄 License

MIT

---

Created with ❤️ by [donphi](https://github.com/donphi)