@echo off
setlocal enabledelayedexpansion

rem Get the directory where the batch file is located
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

rem Path to the run-mcp-server.bat script
set "MCP_SCRIPT_PATH=%SCRIPT_DIR%\run-mcp-server.bat"

rem Create the MCP configuration file
echo {> "%SCRIPT_DIR%\mcp-config.json"
echo   "mcpServers": {>> "%SCRIPT_DIR%\mcp-config.json"
echo     "mcp-server": {>> "%SCRIPT_DIR%\mcp-config.json"
echo       "enabled": true,>> "%SCRIPT_DIR%\mcp-config.json"
echo       "transport": "stdio",>> "%SCRIPT_DIR%\mcp-config.json"
echo       "command": "%MCP_SCRIPT_PATH:\=\\%",>> "%SCRIPT_DIR%\mcp-config.json"
echo       "description": "Local MCP server for document retrieval and search",>> "%SCRIPT_DIR%\mcp-config.json"
echo       "alwaysAllow": [>> "%SCRIPT_DIR%\mcp-config.json"
echo         "read_md_files",>> "%SCRIPT_DIR%\mcp-config.json"
echo         "search_content",>> "%SCRIPT_DIR%\mcp-config.json"
echo         "get_context",>> "%SCRIPT_DIR%\mcp-config.json"
echo         "project_structure",>> "%SCRIPT_DIR%\mcp-config.json"
echo         "suggest_implementation">> "%SCRIPT_DIR%\mcp-config.json"
echo       ],>> "%SCRIPT_DIR%\mcp-config.json"
echo       "auth": {>> "%SCRIPT_DIR%\mcp-config.json"
echo         "type": "none">> "%SCRIPT_DIR%\mcp-config.json"
echo       },>> "%SCRIPT_DIR%\mcp-config.json"
echo       "autoQuery": true>> "%SCRIPT_DIR%\mcp-config.json"
echo     }>> "%SCRIPT_DIR%\mcp-config.json"
echo   }>> "%SCRIPT_DIR%\mcp-config.json"
echo }>> "%SCRIPT_DIR%\mcp-config.json"

echo MCP configuration file created at: %SCRIPT_DIR%\mcp-config.json
echo Add this file to your AI assistant's MCP settings by following the instructions in the README