@echo on
echo Starting minimal batch file...

REM Try different approaches to run the server
echo Approach 1: Direct run with explicit env var
docker-compose run --rm -e TRANSPORT=stdio server
echo Exit code: %ERRORLEVEL%

echo.
echo Approach 2: Set var then run
set TRANSPORT=stdio
docker-compose run --rm server
echo Exit code: %ERRORLEVEL%

echo.
echo Approach 3: Using env_file from docker-compose.yml
docker-compose run --rm server
echo Exit code: %ERRORLEVEL%

echo.
echo Approach 4: Using pure docker command
docker run -i --rm --name mcp-server -e TRANSPORT=stdio mcp-server-server
echo Exit code: %ERRORLEVEL%

echo.
echo All approaches completed.
pause