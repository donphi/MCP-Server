#!/bin/bash
cd "$(dirname "$0")"
source .env
docker run -i --rm \
  --name mcp-server \
  -v "$(pwd)/db:/db" \
  -v "$(pwd)/config:/config" \
  -e OPENAI_API_KEY=${OPENAI_API_KEY} \
  -e ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY} \
  -e DB_PATH=/db \
  -e CONFIG_PATH=/config/server_config.json \
  -e EMBEDDING_MODEL=${EMBEDDING_MODEL:-sentence-transformers/all-MiniLM-L6-v2} \
  -e CLAUDE_MODEL=${CLAUDE_MODEL} \
  -e MAX_RESULTS=${MAX_RESULTS} \
  -e USE_ANTHROPIC=${USE_ANTHROPIC} \
  -e TRANSPORT=stdio \
  mcp-server-server