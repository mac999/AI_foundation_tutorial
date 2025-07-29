@echo off
REM Docker build and run script for YOLO FastAPI Server

echo Building Docker image...
docker build -t yolo-fastapi-server .

REM If build fails, try Ubuntu version
if %ERRORLEVEL% neq 0 (
    echo.
    echo Main build failed, trying Ubuntu version...
    docker build -f Dockerfile.ubuntu -t yolo-fastapi-server .
)

echo.
echo Running Docker container...
docker run -d -p 8000:8000 --name yolo-server yolo-fastapi-server

echo.
echo Container started successfully!
echo API is available at: http://localhost:8000
echo API Documentation: http://localhost:8000/docs
echo.
echo To stop the container: docker stop yolo-server
echo To remove the container: docker rm yolo-server
echo To view logs: docker logs yolo-server
echo docker run -it --rm yolo-fastapi-server bash