#!/bin/bash

# Get the absolute path of the run-mcp-server.sh script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MCP_SCRIPT_PATH="$SCRIPT_DIR/run-mcp-server.sh"

# Check if .env file exists and create it if not
if [ ! -f "$SCRIPT_DIR/.env" ]; then
  echo "Creating .env file from .env.example..."
  cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
  echo "Please edit .env file with your API keys if needed"
fi

# Ensure the run script is executable
chmod +x "$MCP_SCRIPT_PATH"

echo "Creating MCP configuration file..."
echo "Command path: $MCP_SCRIPT_PATH"

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

echo ""
echo "=============================================================="
echo "MCP configuration file created at: $SCRIPT_DIR/mcp-config.json"
echo ""
echo "SETUP INSTRUCTIONS:"
echo "1. Copy the mcp-config.json file to your Roo settings location:"
echo "   - macOS: ~/Library/Application Support/Code/User/globalStorage/rooveterinaryinc.roo-cline/settings/"
echo "   - Linux: ~/.config/Code/User/globalStorage/rooveterinaryinc.roo-cline/settings/"
echo ""
echo "2. Verify your .env file has been created (copy of .env.example)"
echo ""
echo "3. To test the server, run:"
echo "   ./run-mcp-server.sh"
echo "=============================================================="

# Display JSON preview
echo ""
echo "JSON content preview:"
cat "$SCRIPT_DIR/mcp-config.json"
echo ""