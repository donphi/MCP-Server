@echo off
setlocal enabledelayedexpansion

rem Get the directory where the batch file is located
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

rem Path to the run-mcp-server.bat script (with forward slashes for JSON)
set "MCP_SCRIPT_PATH=%SCRIPT_DIR:\=/%/run-mcp-server.bat"

rem Create a clean JSON file without using echo (avoids special character issues)
> "%SCRIPT_DIR%\windows-mcp-config.json" (
  echo {
  echo   "mcpServers": {
  echo     "mcp-server": {
  echo       "enabled": true,
  echo       "transport": "stdio",
  echo       "command": "%MCP_SCRIPT_PATH%",
  echo       "description": "Local MCP server for document retrieval and search",
  echo       "alwaysAllow": [
  echo         "read_md_files",
  echo         "search_content",
  echo         "get_context",
  echo         "project_structure",
  echo         "suggest_implementation"
  echo       ],
  echo       "auth": {
  echo         "type": "none"
  echo       },
  echo       "autoQuery": true
  echo     }
  echo   }
  echo }
)

echo.
echo MCP configuration file created at: %SCRIPT_DIR%\windows-mcp-config.json
echo.
echo JSON content preview:
type "%SCRIPT_DIR%\windows-mcp-config.json"
echo.
echo.
echo ********************************************************************
echo * IMPORTANT INSTRUCTIONS:                                          *
echo * 1. Verify the command path looks correct above                   *
echo * 2. Copy this file to VSCode MCP settings location                *
echo * 3. Make sure an .env file exists (copy from .env.example)        *
echo * 4. Make sure the Docker image 'mcp-server-server' is built       *
echo ********************************************************************
echo.
pause