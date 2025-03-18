#!/bin/bash

# Change to the directory where the script is located
cd "$(dirname "$0")"

echo "Checking MCP server prerequisites..."

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "ERROR: .env file not found!"
    echo "Please run setup-mcpServer-json.sh first or create a .env file from .env.example"
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed or not in PATH!"
    echo "Please install Docker and try again."
    exit 1
fi

# Check if Docker image exists and build it if needed
if ! docker images | grep -q "mcp-server-server"; then
    echo "Docker image 'mcp-server-server' not found."
    echo "Building Docker image (this may take a few minutes)..."
    docker build -t mcp-server-server -f Dockerfile.server .
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to build Docker image!"
        exit 1
    fi
    echo "Docker image built successfully."
fi

# Load environment variables from .env file
source .env

# Set default values for environment variables if not set
: "${EMBEDDING_MODEL:=sentence-transformers/all-MiniLM-L6-v2}"
: "${CLAUDE_MODEL:=claude-3-7-sonnet-20240307}"
: "${MAX_RESULTS:=10}"
: "${USE_ANTHROPIC:=true}"

echo "Starting MCP server..."
echo "Using embedding model: ${EMBEDDING_MODEL}"

# Run the Docker container
docker run -i --rm \
  --name mcp-server \
  -v "$(pwd)/db:/db" \
  -v "$(pwd)/config:/config" \
  -e OPENAI_API_KEY=${OPENAI_API_KEY} \
  -e ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY} \
  -e DB_PATH=/db \
  -e CONFIG_PATH=/config/server_config.json \
  -e EMBEDDING_MODEL=${EMBEDDING_MODEL} \
  -e CLAUDE_MODEL=${CLAUDE_MODEL} \
  -e MAX_RESULTS=${MAX_RESULTS} \
  -e USE_ANTHROPIC=${USE_ANTHROPIC} \
  -e TRANSPORT=stdio \
  mcp-server-server