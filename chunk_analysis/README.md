# 🧩 Chunk Analysis Tools

## 🔍 Overview

This directory contains tools for analyzing and comparing document chunking strategies without modifying the main application code.

These tools allow you to:
1. Test the standard chunking algorithm from the main pipeline
2. Test the enhanced chunking with lemmatization and POS filtering
3. Compare the results side-by-side
4. All while using environment variables from your `.env` file

## 📋 Components

- `🐳 Dockerfile` - Builds a container with both chunking methods
- `🔄 docker-compose.yml` - Sets up the testing environment
- `🧪 test_chunking.py` - Main script that runs both chunking methods
- `🔎 inspect_chunks.py` - Script used by the main test to inspect chunks
- `▶️ run_tests.sh` - Helper script to run the tests easily

## 🚀 How to Run Tests

Simply run:
```bash
# First build the Docker container
docker-compose build

# Then run the tests
./run_tests.sh
```

This will:
1. Check for a `.env` file (creates one from `.env.example` if needed)
2. Build a Docker container with all necessary components
3. Mount your data directory and config files
4. Run tests using both standard and enhanced chunking
5. Save results to the `test_chunks` directory

### 🧹 Cleanup Option

To clean up Docker resources after testing:
```bash
./run_tests.sh --clean
```

## ⚙️ Environment Variables

All settings are read from your `.env` file in the root directory, including:

- `CHUNK_SIZE` - Maximum size of each chunk (default: 800)
- `CHUNK_OVERLAP` - Overlap between chunks (default: 120)

You can modify these values in your `.env` file before running the tests.

## 📊 Results

Results are stored in:

- 📁 `test_chunks/standard/` - Results from standard chunking
- 📁 `test_chunks/enhanced/` - Results from enhanced chunking
- 📁 `test_chunks/comparison/` - Comparison reports and statistics

Each document directory contains:
- The original document content
- Each chunk as a separate file
- Markers for oversized chunks
- A summary file with metrics

The comparison directory contains:
- File-by-file comparisons
- An overall summary report
- Recommendations based on the results

## 🔄 Chunking Methods Compared

### 📝 Standard Chunking
- Preserves document structure and formatting
- Maintains code blocks, tables, and section headings
- Creates semantically coherent chunks
- Used by default in the main application

### 🔬 Enhanced Chunking
- Uses lemmatization (reducing words to base forms)
- Applies part-of-speech filtering
- Can produce more semantically dense chunks
- Experimental approach for comparison

## ⚙️ How It Works

1. The testing container isolates all the testing from your main application
2. It copies necessary components from your main app into the container
3. It uses the exact same environment variables from your `.env` file
4. It runs tests on both chunking methods side-by-side
5. Results are stored in a volume mounted back to your host machine

This ensures your main application remains unchanged while allowing you to experiment with different chunking strategies.

## 📋 Interpreting Results

When examining the chunks and comparison reports, look for:

- **Chunk Size Distribution**: Are chunks consistently within size limits?
- **Content Preservation**: Is important information preserved in both methods?
- **Semantic Coherence**: Do chunks make sense as standalone units?
- **Format Preservation**: Are code blocks, tables, and formatting maintained?

For most technical documentation, standard chunking typically provides better results by preserving formatting and structure, while enhanced chunking might be beneficial for narrative or academic text.

---

Created with ❤️ for document processing experimentation