@echo off
setlocal enabledelayedexpansion

rem Change to the directory where the batch file is located
cd /d "%~dp0"

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