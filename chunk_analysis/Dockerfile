FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file from main app
COPY requirements.pipeline.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.pipeline.txt

# Install spaCy and download language model for enhanced chunking
RUN pip install --no-cache-dir spacy
RUN python -m spacy download en_core_web_sm

# Install markdown support for processing .md files
RUN pip install --no-cache-dir markdown

# Create directory structure for imports to work
RUN mkdir -p /app/src/utils

# Copy source code
COPY src/ /app/src/
COPY config/ /app/config/
COPY enhanced_chunking.py /app/
COPY inspect_chunks.py /app/

# Copy testing scripts
COPY chunk_analysis/inspect_chunks.py /app/chunk_analysis/
COPY chunk_analysis/test_chunking.py /app/chunk_analysis/
COPY chunk_analysis/semi_interactive_chunking.py /app/chunk_analysis/

# Fix import issues by creating __init__.py files
RUN touch /app/__init__.py
RUN touch /app/src/__init__.py
RUN touch /app/src/utils/__init__.py

# Create a special utils folder at the top level to help with imports
RUN mkdir -p /app/utils
RUN touch /app/utils/__init__.py
# Copy utils module to both places to support different import styles
RUN cp -r /app/src/utils/* /app/utils/

# Set environment variables
ENV PYTHONPATH=/app:/app/src
ENV CONFIG_PATH=/app/config/pipeline_config.json

# Create directories to ensure they exist
RUN mkdir -p /test_chunks/standard /test_chunks/enhanced /test_chunks/comparison

# Copy .env file (environment variables will be set from .env or docker-compose)
COPY .env* /app/

# Copy entrypoint script
COPY chunk_analysis/docker_entrypoint.sh /app/docker_entrypoint.sh
RUN chmod +x /app/docker_entrypoint.sh

# Final diagnostic command to verify directory structure
RUN echo "Directory structure:" && \
    find /app -type d | sort && \
    echo "Files:" && \
    find /app -type f -name "*.py" | sort

ENTRYPOINT ["/app/docker_entrypoint.sh"]