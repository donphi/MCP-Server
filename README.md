# MCP-Server

A versatile server for processing and embedding documents using LangChain, OpenAI, and Sentence Transformers.

## Quick Start

### Run with Docker Compose

```bash
# Build and start the pipeline and server
docker-compose up --build
```

### Processing Documents

1. Place your documents in the `data/` directory
2. Run the pipeline:
```bash
docker-compose run pipeline
```

### Querying Documents

```bash
# Ask a question (interactive mode)
docker-compose run server
```

## Configuration

### Environment Variables

Configuration is provided through environment variables in the `.env` file, which override values in config files.

### Document Processing Pipeline

The pipeline processes documents from the `data/` directory, chunks them, and embeds them into a vector database:

1. **Document Loading**: Files are loaded based on their extension (.md, .txt, .pdf, .docx)
2. **Document Type Detection**: Content is analyzed to determine its structure (scientific, financial, technical, narrative, general)
3. **Chunking**: Content is divided into chunks using spaCy (preferred) or fallback methods
4. **Embedding**: Chunks are embedded using the specified embedding model
5. **Storage**: Embeddings are stored in a vector database for retrieval

## Chunking Strategies

The pipeline uses these document-specific chunking strategies:

- **Scientific Papers**: Split by sections, preserve references
- **Financial Documents**: Preserve tables and numerical sections
- **Technical Documentation**: Preserve code blocks and examples
- **Narrative Text**: Use semantic boundaries via spaCy NLP
- **General**: Balanced approach using section headers and semantic breaks

SpaCy is used as the preferred chunking method for all document types when available.

## Troubleshooting

### Common Issues

**Inconsistent Chunking**

If you notice inconsistent chunking between files, it may be due to:
- The document type detection system remembering previous decisions
- Missing spaCy dependencies
- Config file vs environment variable conflicts

**Solutions**:
- The pipeline now automatically resets document type memory between runs
- Ensure spaCy is installed: `pip install spacy && python -m spacy download en_core_web_md`
- Verify .env and config files are consistent

**PDF Processing**

PDFs may not chunk properly if:
- The PDF contains scanned images rather than text
- The PDF has complex formatting
- Required dependencies are missing

**Solutions**:
- The pipeline now has improved PDF handling with better diagnostics
- For scanned PDFs, consider pre-processing with OCR
- Install PyPDF: `pip install pypdf`

## Advanced Configuration

For advanced use cases, the pipeline and server can be customized:

- **Custom Embedding Functions**: Create custom embedding logic
- **Document Type Classification**: Modify document type detection
- **Chunking Behavior**: Adjust chunking parameters for specific needs
- **Chunk Analysis**: Compare standard and enhanced chunking methods using the testing tools in `/chunk_analysis`:
  ```bash
  # First build the Docker container
  cd chunk_analysis
  docker-compose build
  
  # Then run the tests
  ./run_tests.sh
  ```

## Enhancements

Recent improvements:
- Consistent use of spaCy for all document types when available
- Enforced chunk size limits to maintain retrieval quality
- Improved PDF handling for multi-page documents
- Better logging and diagnostics
- Configuration consistency between .env and config files