#!/bin/bash

# Stop and remove the existing container
echo "Stopping and removing existing container..."
docker stop mcp-server-1
docker rm mcp-server-1

# Rebuild the Docker image
echo "Rebuilding Docker image..."
docker-compose build server

# Start the container
echo "Starting the container..."
docker-compose up -d server

# Wait for the container to start
echo "Waiting for container to start..."
sleep 5

# Check if the container is running
if docker ps | grep -q mcp-server-1; then
    echo "Container is running. MCP Server should be ready."
    echo "You can now use Roo Code with the MCP server."
    
    # Display logs to check for any issues
    echo "Container logs:"
    docker logs mcp-server-1
else
    echo "Container failed to start. Check the logs with: docker logs mcpcopy-server-1"
fi