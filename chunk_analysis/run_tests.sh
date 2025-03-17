#!/bin/bash
# Run chunking tests in a Docker container
# This script automates the process of testing both standard and enhanced chunking
# without modifying the original application

# Default settings
CLEAN_UP=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --clean)
      CLEAN_UP=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--clean]"
      echo "  --clean    Remove Docker container and volumes after completion"
      exit 1
      ;;
  esac
done

# Go to project root directory
cd "$(dirname "$0")/.."
ROOT_DIR=$(pwd)

# Check if .env file exists, if not copy from .env.example
if [ ! -f ".env" ]; then
  echo "No .env file found, creating one from .env.example"
  cp .env.example .env
  echo "Created .env file. You may want to review and adjust settings in .env before continuing."
  read -p "Press Enter to continue or Ctrl+C to cancel..."
fi

# Print current settings
echo "Current chunking settings:"
grep "CHUNK_SIZE" .env || echo "CHUNK_SIZE not found in .env file"
grep "CHUNK_OVERLAP" .env || echo "CHUNK_OVERLAP not found in .env file"

# Confirm with user
read -p "Do you want to proceed with these settings? (y/n): " confirm
if [[ $confirm != [yY] && $confirm != [yY][eE][sS] ]]; then
  echo "Edit your .env file and run this script again"
  exit 0
fi

echo "Creating test chunks directory if it doesn't exist"
mkdir -p "$ROOT_DIR/chunk_analysis/test_chunks"

echo "Building Docker container for chunk analysis"
cd "$ROOT_DIR/chunk_analysis"
docker-compose build

echo "Running test chunking analysis in interactive mode"
# Run docker-compose without the -i flag since it's not supported in this version
# The stdin_open and tty settings in docker-compose.yml should handle interactivity
docker-compose up

# Output the results location
echo ""
echo "Test completed. Results are available in:"
echo "  $ROOT_DIR/chunk_analysis/test_chunks/standard/ - Standard chunking results"
echo "  $ROOT_DIR/chunk_analysis/test_chunks/enhanced/ - Enhanced chunking results"
echo "  $ROOT_DIR/chunk_analysis/test_chunks/comparison/ - Comparison reports"

# Clean up if requested
if [ "$CLEAN_UP" = true ]; then
  echo "Cleaning up Docker resources"
  docker-compose down --volumes --remove-orphans
  echo "Cleanup complete"
fi

echo "Test script completed successfully"