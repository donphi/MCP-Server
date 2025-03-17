#!/bin/bash
# Setup for enhanced chunking - installs required components

echo "Setting up enhanced chunking environment..."

# Make sure spaCy is installed
pip install spacy>=3.6.0

# Download the English language model needed for lemmatization and POS tagging
python -m spacy download en_core_web_sm

echo "Setup complete! You can now run ./run_chunk_analysis.sh to analyze documents."