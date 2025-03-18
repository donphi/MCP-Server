@echo off
setlocal

REM Change to the directory where the batch file is located
cd /d "%~dp0"

REM Check if .env file exists
if not exist ".env" (
  echo ERROR: .env file not found!
  echo Please run setup-mcpServer-json.bat first or create a .env file from .env.example
  pause
  exit /b 1
)

REM Check if Docker is installed
docker --version > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
  echo ERROR: Docker is not installed or not running!
  echo Please install Docker Desktop for Windows and try again.
  pause
  exit /b 1
)

REM Check if Docker Compose is installed
docker-compose --version > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
  echo ERROR: Docker Compose is not installed!
  echo Please install Docker Desktop for Windows (which includes Docker Compose) and try again.
  pause
  exit /b 1
)

REM Build or rebuild the server image for consistency
echo Building MCP server image...
docker-compose build server

REM Set the TRANSPORT environment variable
set TRANSPORT=stdio

REM Start the MCP server using docker-compose
echo Starting MCP server...
docker-compose run --rm server