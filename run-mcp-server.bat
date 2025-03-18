@echo off
setlocal

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
docker-compose images server | findstr "server" > nul
if %ERRORLEVEL% NEQ 0 (
  echo Docker image 'server' not found.
  echo Building Docker image now...
  docker-compose build server
  if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to build Docker image!
    pause
    exit /b 1
  )
  echo Docker image built successfully.
)

rem Set TRANSPORT environment variable for MCP communication
set TRANSPORT=stdio

echo Starting MCP server...
docker-compose run --rm server