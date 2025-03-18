@echo off
setlocal enabledelayedexpansion

rem Get the directory where the batch file is located
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

rem Path to the run-mcp-server.bat script
set "MCP_SCRIPT_PATH=%SCRIPT_DIR%\run-mcp-server.bat"

rem Create proper JSON path with forward slashes
set "JSON_PATH=%MCP_SCRIPT_PATH:\=/%"

rem Display what will be written
echo Creating MCP configuration file...
echo Command path: %JSON_PATH%

rem Create the MCP configuration file
(
echo {
echo   "mcpServers": {
echo     "mcp-server": {
echo       "enabled": true,
echo       "transport": "stdio",
echo       "command": "%JSON_PATH%",
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
) > "%SCRIPT_DIR%\mcp-config.json"

echo.
echo MCP configuration file created at: %SCRIPT_DIR%\mcp-config.json
echo.
echo JSON content preview:
type "%SCRIPT_DIR%\mcp-config.json"
echo.
echo Please copy this file to your VSCode MCP settings location
echo.
pause