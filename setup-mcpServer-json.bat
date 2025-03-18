@echo off
setlocal enabledelayedexpansion

rem Get the directory where the batch file is located
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

rem Path to the run-mcp-server.bat script
set "MCP_SCRIPT_PATH=%SCRIPT_DIR%\run-mcp-server.bat"

echo Checking MCP server prerequisites...

rem Check if .env file exists and create it if not
if not exist "%SCRIPT_DIR%\.env" (
  echo Creating .env file from .env.example...
  copy "%SCRIPT_DIR%\.env.example" "%SCRIPT_DIR%\.env"
  echo Please edit .env file with your API keys if needed
)

rem Convert backslashes to forward slashes for JSON
set "JSON_PATH=%MCP_SCRIPT_PATH:\=/%"

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
echo ==============================================================
echo MCP configuration file created at: %SCRIPT_DIR%\mcp-config.json
echo.
echo SETUP INSTRUCTIONS:
echo 1. Copy the mcp-config.json file to your Roo settings location:
echo    - Windows: %%APPDATA%%\Code\User\globalStorage\rooveterinaryinc.roo-cline\settings\
echo    - macOS: ~/Library/Application Support/Code/User/globalStorage/rooveterinaryinc.roo-cline/settings/
echo    - Linux: ~/.config/Code/User/globalStorage/rooveterinaryinc.roo-cline/settings/
echo.
echo 2. Verify your .env file has been created (copy of .env.example)
echo.
echo 3. To test the server, run:
echo    run-mcp-server.bat
echo ==============================================================
echo.
echo JSON content preview:
type "%SCRIPT_DIR%\mcp-config.json"
echo.
echo.
pause