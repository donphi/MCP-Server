# Chunk Analysis Tools

This directory contains tools for analyzing and comparing document chunking strategies without modifying the main application code.

## Overview

These tools allow you to:
1. Test the standard chunking algorithm from the main pipeline
2. Test the enhanced chunking with lemmatization and POS filtering
3. Compare the results side-by-side
4. All while using environment variables from your `.env` file

## Files

- `Dockerfile` - Builds a container with both chunking methods
- `docker-compose.yml` - Sets up the testing environment
- `test_chunking.py` - Main script that runs both chunking methods
- `inspect_chunks.py` - Script used by the main test to inspect chunks
- `run_tests.sh` - Helper script to run the tests easily

## How to Run Tests

Simply run:
```bash
./run_tests.sh
```

This will:
1. Check for a `.env` file (creates one from `.env.example` if needed)
2. Build a Docker container with all necessary components
3. Mount your data directory and config files
4. Run tests using both standard and enhanced chunking
5. Save results to the `test_chunks` directory

### Cleanup Option

To clean up Docker resources after testing:
```bash
./run_tests.sh --clean
```

## Environment Variables

All settings are read from your `.env` file in the root directory, including:

- `CHUNK_SIZE` - Maximum size of each chunk (default: 1000)
- `CHUNK_OVERLAP` - Overlap between chunks (default: 200)

You can modify these values in your `.env` file before running the tests.

## Results

Results are stored in:

- `test_chunks/standard/` - Results from standard chunking
- `test_chunks/enhanced/` - Results from enhanced chunking
- `test_chunks/comparison/` - Comparison reports and statistics

Each document directory contains:
- The original document content
- Each chunk as a separate file
- Markers for oversized chunks
- A summary file with metrics

The comparison directory contains:
- File-by-file comparisons
- An overall summary report
- Recommendations based on the results

## How It Works

1. The testing container isolates all the testing from your main application
2. It copies necessary components from your main app into the container
3. It uses the exact same environment variables from your `.env` file
4. It runs tests on both chunking methods side-by-side
5. Results are stored in a volume mounted back to your host machine

This ensures your main application remains unchanged while allowing you to experiment with different chunking strategies.