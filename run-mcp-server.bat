@echo off
setlocal enabledelayedexpansion

rem Change to the directory where the batch file is located
cd /d "%~dp0"

rem Check if .env file exists
if not exist ".env" (
  echo ERROR: .env file not found!
  echo Please run setup-mcpServer-json.bat first or create a .env file from .env.example
  pause
  exit /b 1
)

rem Check if Docker is installed
docker --version > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
  echo ERROR: Docker is not installed or not running!
  echo Please install Docker Desktop for Windows and try again.
  pause
  exit /b 1
)

rem Check if Docker image exists and build it if needed
docker images | findstr "mcp-server-server" > nul
if %ERRORLEVEL% NEQ 0 (
  echo Docker image 'mcp-server-server' not found.
  echo Building Docker image now...
  docker build -t mcp-server-server -f Dockerfile.server .
  if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to build Docker image!
    pause
    exit /b 1
  )
  echo Docker image built successfully.
)

rem Load environment variables from .env file
for /F "tokens=*" %%A in (.env) do (
  set line=%%A
  if not "!line:~0,1!"=="#" (
    if not "!line!"=="" (
      set "!line!"
    )
  )
)

rem Set default values for environment variables if not set
if not defined EMBEDDING_MODEL set EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
if not defined CLAUDE_MODEL set CLAUDE_MODEL=claude-3-7-sonnet-20240307
if not defined MAX_RESULTS set MAX_RESULTS=10
if not defined USE_ANTHROPIC set USE_ANTHROPIC=true

echo Starting MCP server...
echo Using embedding model: !EMBEDDING_MODEL!

rem Run the Docker container
docker run -i --rm ^
  --name mcp-server ^
  -v "%cd%\db:/db" ^
  -v "%cd%\config:/config" ^
  -e OPENAI_API_KEY=!OPENAI_API_KEY! ^
  -e ANTHROPIC_API_KEY=!ANTHROPIC_API_KEY! ^
  -e DB_PATH=/db ^
  -e CONFIG_PATH=/config/server_config.json ^
  -e EMBEDDING_MODEL=!EMBEDDING_MODEL! ^
  -e CLAUDE_MODEL=!CLAUDE_MODEL! ^
  -e MAX_RESULTS=!MAX_RESULTS! ^
  -e USE_ANTHROPIC=!USE_ANTHROPIC! ^
  -e TRANSPORT=stdio ^
  mcp-server-server