#!/bin/bash

# Get the absolute path of the run-mcp-server.sh script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MCP_SCRIPT_PATH="$SCRIPT_DIR/run-mcp-server.sh"

# Ensure the run script is executable
chmod +x "$MCP_SCRIPT_PATH"

# Create the MCP configuration file
cat > "$SCRIPT_DIR/mcp-config.json" << EOL
{
  "mcpServers": {
    "mcp-server": {
      "enabled": true,
      "transport": "stdio",
      "command": "$MCP_SCRIPT_PATH",
      "description": "Local MCP server for document retrieval and search",
      "alwaysAllow": [
        "read_md_files",
        "search_content",
        "get_context",
        "project_structure",
        "suggest_implementation"
      ],
      "auth": {
        "type": "none"
      },
      "autoQuery": true
    }
  }
}
EOL

echo "MCP configuration file created at: $SCRIPT_DIR/mcp-config.json"
echo "Add this file to Roo settings by following the instructions in the README"