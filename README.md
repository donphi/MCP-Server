# 🚀 MCP Server for Document Processing

## 🔗 About Model Context Protocol (MCP)

The [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) is a new standard created by Anthropic to enable AI assistants to access external tools and data sources. This protocol allows AI models to extend their capabilities beyond their training data by connecting to specialized services like this MCP server.

By implementing the MCP standard, this server enables AI assistants to query and retrieve information from your custom document collection, effectively extending their knowledge base with your specific content.

## 🧠 Extend LLM Knowledge with Up-to-Date Information

This Model Context Protocol (MCP) server lets you overcome one of the biggest limitations of large language models: knowledge cutoffs. By creating your own MCP server, you can feed AI assistants up-to-date information about:

- **Latest Framework Documentation**: Use content not in LLM training data (React 19, Angular 17, Vue 3.4+, etc.)
- **Private Codebases**: Help AI assistants understand your proprietary code patterns and structures
- **Technical Specifications**: Import documentation on new APIs, protocols, or tools

**Recommended Data Sources:**
- [FireCrawl.dev](https://www.firecrawl.dev/): A powerful tool for scraping documentation websites
- Official GitHub repositories: Download READMEs and documentation
- Technical blogs and tutorials: Save key articles as Markdown files

## 🏗️ Architecture

The system consists of two main components:

1. **📝 Processing Pipeline**: Reads Markdown and text files, chunks them, generates embeddings, and stores them in a vector database.
2. **🔌 MCP Server**: Exposes the processed content through MCP tools, allowing AI assistants to search and retrieve relevant information.

## 💡 Example Use Cases

### Upgrading AI Knowledge with Latest Framework Documentation
```
# Scrape latest React 19 docs using FireCrawl.dev
# Place the saved markdown files in the data/ directory
# Run the pipeline to process the documentation
# Now ask your AI assistant about React 19 features!
```

### Using Private Codebase Documentation
```
# Export your API documentation as markdown
# Place the markdown files in the data/ directory
# Run the pipeline to process
# Now your AI assistant can help debug issues with your specific APIs!
```

## ✅ Prerequisites

- **Docker**: Docker Desktop for [Windows](https://docs.docker.com/desktop/install/windows-install/) or [Mac](https://docs.docker.com/desktop/install/mac-install/), or [Docker Engine](https://docs.docker.com/engine/install/) for Linux
- **OpenAI API key** (Optional): Can use free local embeddings instead
- **AI assistant that supports MCP**: Such as Roo or other compatible assistants

## 🛠️ Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/donphi/mcp-server.git
   cd mcp-server
   ```

2. Create a `.env` file with your configuration:
   ```bash
   # Copy the example file
   cp .env.example .env
   
   # Edit the file with your settings
   nano .env
   ```
   On Windows, you can use Notepad to edit the .env file.

3. Place your Markdown (.md) and text (.txt) files in the `data/` directory.

## ⚙️ Configuration

You can configure the MCP server using environment variables in the `.env` file:

```
# API Keys
OPENAI_API_KEY=your_openai_api_key_here  # Optional - can use free local embeddings instead
ANTHROPIC_API_KEY=your_anthropic_api_key_here  # Optional

# Pipeline Configuration
CHUNK_SIZE=1000                # Size of text chunks
CHUNK_OVERLAP=200              # Overlap between chunks (in tokens)
BATCH_SIZE=10                  # Batch size for embedding generation
EMBEDDING_MODEL=text-embedding-3-small  # Model to use (see options below)
SUPPORTED_EXTENSIONS=.md,.txt  # Comma-separated list of supported file extensions

# Server Configuration
CLAUDE_MODEL=claude-3-7-sonnet-20240307  # Claude model to use
MAX_RESULTS=10                 # Maximum number of results to return
USE_ANTHROPIC=true             # Whether to use Anthropic API for responses
```

## 📊 Embedding Models

The system supports multiple embedding models for converting text to vector representations:

### Free Models (no API key required)
These models run locally within the Docker container and don't require any API keys:

- **sentence-transformers/all-MiniLM-L6-v2**: A compact model designed for sentence and short paragraph encoding, providing efficient embeddings suitable for rapid retrieval tasks.

- **BAAI/bge-m3**: A versatile model supporting multiple retrieval functionalities, over 100 languages, and inputs up to 8192 tokens, making it ideal for comprehensive retrieval tasks.

- **Snowflake/snowflake-arctic-embed-m**: Optimized for high-quality retrieval performance, this model balances accuracy and inference speed effectively.

### Paid Models (require OpenAI API key)
- **text-embedding-3-small**: Optimized for speed and cost-effectiveness with good quality
- **text-embedding-3-large**: Highest quality embeddings (more expensive)

When you run the pipeline, you'll be prompted to choose which model to use. If you don't have an OpenAI API key, the system will automatically use one of the free local models.

## 🚀 Usage

### 🔄 Processing the Files

To process your files and generate embeddings:

```bash
docker-compose build pipeline
docker-compose run pipeline
```

On Windows, you can run these commands in Command Prompt or PowerShell after installing Docker Desktop.

This will:
- Prompt you to choose an embedding model
- Install necessary packages if needed
- Read all supported files in the `data/` directory
- Process and chunk the content
- Generate embeddings
- Store the embeddings in the vector database (creates a `chroma.sqlite3` file in the `db/` directory)

### 🔗 Connecting to an MCP-Compatible AI Assistant

The MCP server needs to be configured with your AI assistant. We provide scripts to generate the configuration:

#### For macOS/Linux:

1. Make the setup script executable and run it:
   ```bash
   chmod +x setup-mcpServer-json.sh
   ./setup-mcpServer-json.sh
   ```

2. This will create a `mcp-config.json` file with the correct configuration.

3. Add the configuration to your AI assistant.

#### For Windows:

1. Double-click on the `setup-mcpServer-json.bat` file or run it from Command Prompt:
   ```cmd
   setup-mcpServer-json.bat
   ```

2. This will create a `mcp-config.json` file with the correct configuration.

3. Add the configuration to your AI assistant.

#### Example: Configuring with Roo

If you're using Roo as your AI assistant:

1. Run the appropriate setup script for your platform to generate the configuration file
2. In Roo, click the "MCP Server" button/tab in the sidebar
3. Enable the "Enable MCP Servers" toggle
4. Click "Edit MCP Settings"
5. Copy and paste the entire contents of the mcp-config.json file
6. Save the settings

## 🧩 Using the MCP Server

Once configured, you can use the MCP server with an AI assistant that supports MCP. With compatible assistants like Roo, you can use it in two ways:

1. **Automatic mode** (with `autoQuery: true`): Ask questions normally, and the AI will automatically check your vector database for relevant information.

   Example: "What are the key features of React 19?"

2. **Explicit tool usage**: Directly ask the AI to use a specific tool.

   Example: "Use the search_content tool to find information about React 19 Compiler."

## 🧰 MCP Tools

The MCP server exposes the following tools:

- **📚 read_md_files**: Process and retrieve files. Parameters: `file_path` (optional path to a specific file or directory)
- **🔍 search_content**: Search across processed content. Parameters: `query` (required search query)
- **📋 get_context**: Retrieve contextual information. Parameters: `query` (required context query), `window_size` (optional number of context items to retrieve)
- **🏗️ project_structure**: Provide project structure information. No parameters.
- **💡 suggest_implementation**: Generate implementation suggestions. Parameters: `description` (required description of what to implement)

## 📄 Supported File Types

By default, the following file types are supported:
- Markdown files (.md)
- Text files (.txt)

You can configure additional file extensions by setting the `SUPPORTED_EXTENSIONS` environment variable in your `.env` file.

## 🔄 Operational Modes

The MCP server can operate in two modes:

1. **🤖 Full Processing Mode**: When the Anthropic API key is provided and `USE_ANTHROPIC` is set to `true`, the server will use Claude to generate responses based on the retrieved context.

2. **📋 Context Retrieval Mode**: When the Anthropic API key is not provided or `USE_ANTHROPIC` is set to `false`, the server will only retrieve and return the relevant context, allowing the client (e.g., AI assistant) to process it using its own LLM.

## 📁 Project Structure

```
mcp-server/
├── Dockerfile.pipeline
├── Dockerfile.server
├── docker-compose.yml
├── requirements.pipeline.txt
├── requirements.server.txt
├── README.md
├── .env
├── .env.example
├── run-mcp-server.sh       # For macOS/Linux
├── run-mcp-server.bat      # For Windows
├── setup-mcpServer-json.sh # Setup script for macOS/Linux
├── setup-mcpServer-json.bat # Setup script for Windows
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
    └── chroma.sqlite3  # Created after running the pipeline
```

## ❓ Troubleshooting

- **Docker not found**: Ensure Docker is installed and running. Check with `docker --version`.
- **API key issues**: Not to worry! You can use the free local embedding models without any API keys.
- **Missing sentence-transformers package**: If you select a free model, the system will automatically install the required package.
- **Chroma database not found**: Make sure you've run the pipeline to process your documents first.
- **Connection issues**: Verify the path in your MCP configuration points to the correct location of the run script.
- **Windows path issues**: If you encounter path problems on Windows, ensure paths use double backslashes (\\\\) in the JSON configuration.
- **Embedding model mismatch**: The server automatically detects which model was used to create the database and uses the same model for retrieval.

## 📄 License

MIT

---

Created with ❤️ by [donphi](https://github.com/donphi)