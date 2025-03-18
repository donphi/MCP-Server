@echo off
setlocal enabledelayedexpansion

rem Change to the directory where the batch file is located
cd /d "%~dp0"

rem Check if Docker is installed
docker --version > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
  echo ERROR: Docker is not installed or not running!
  echo Please install Docker Desktop for Windows and try again.
  pause
  exit /b 1
)

rem Check if Docker Compose is installed
docker-compose --version > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
  echo ERROR: Docker Compose is not installed!
  echo Please install Docker Desktop for Windows (which includes Docker Compose) and try again.
  pause
  exit /b 1
)

rem Build the server image if it doesn't exist
echo Building the server image to ensure it's up to date...
docker-compose build server

echo Starting MCP server...
echo Using docker-compose to run the server...

rem Run the server
docker-compose run --rm -e TRANSPORT=stdio server