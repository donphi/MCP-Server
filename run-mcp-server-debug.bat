@echo on
REM Enable debugging output
echo Starting debug batch file...

REM Change directory
cd /d "%~dp0"
echo Current directory: %CD%

REM Check Docker
echo Checking Docker...
docker --version
echo Docker check completed with exit code: %ERRORLEVEL%

REM Check Docker Compose
echo Checking Docker Compose...
docker-compose --version
echo Docker Compose check completed with exit code: %ERRORLEVEL%

REM Build server
echo Attempting to build server...
docker-compose build server
echo Build completed with exit code: %ERRORLEVEL%

REM Run server with minimal parameters
echo Attempting to run server...
echo Setting TRANSPORT variable...
set TRANSPORT=stdio
echo TRANSPORT=%TRANSPORT%

echo Running docker-compose command...
docker-compose run --rm server
echo Server command completed with exit code: %ERRORLEVEL%

echo Batch file completed.
pause