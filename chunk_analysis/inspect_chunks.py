#!/usr/bin/env python3
"""
Chunk Inspection Script

This script processes documents using the same components as the main pipeline,
but saves chunks to disk before embedding for inspection purposes.
It doesn't modify any existing code and runs separately from the main pipeline.
"""
import os
import json
import sys
import argparse
from datetime import datetime
from typing import Dict, List, Any

# Import from the existing pipeline
from src.pipeline import PipelineConfig, TextProcessor
from src.utils.chunking import ChunkingEngine

def inspect_chunks(data_dir, output_dir='test_chunks/standard', config_path=None):
    """
    Process files using the same pipeline components but save chunks for inspection
    
    Args:
        data_dir: Directory containing input files
        output_dir: Directory to save chunk inspection results
        config_path: Path to the pipeline configuration file
    """
    # Use the default config path if not provided
    if not config_path:
        config_path = os.environ.get("CONFIG_PATH", "/config/pipeline_config.json")
        
    print(f"Loading configuration from {config_path}")
    config = PipelineConfig(config_path)
    
    # Override directories
    config.data_dir = data_dir
    
    # Create processors
    print(f"Initializing processors with chunk size: {config.chunk_size}, overlap: {config.chunk_overlap}")
    text_processor = TextProcessor(config)
    chunking_engine = ChunkingEngine(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap
    )
    
    # Reset document type memory to avoid carrying over from previous runs
    print("Resetting document type memory")
    chunking_engine.reset_document_type_memory()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")
    
    # Find all files to process
    files = []
    for root, _, filenames in os.walk(data_dir):
        for filename in filenames:
            file_path = os.path.join(root, filename)
            _, ext = os.path.splitext(file_path)
            if ext.lower() in ['.pdf', '.txt', '.md', '.docx', '.doc']:
                files.append(file_path)
                
    print(f"Found {len(files)} files to process")
    
    # Process each file and save chunks
    for file_path in files:
        try:
            filename = os.path.basename(file_path)
            print(f"\nProcessing {filename}...")
            
            # Process using the same methods as the main pipeline
            document = text_processor.process_file(file_path)
            
            # Get document type
            detected_type = chunking_engine.detect_document_type(document['content'])
            print(f"Detected document type: {detected_type}")
            
            # Create a directory for this file's chunks
            file_dir = os.path.join(output_dir, os.path.splitext(filename)[0])
            os.makedirs(file_dir, exist_ok=True)
            
            # Save the original document
            with open(os.path.join(file_dir, "original.json"), 'w') as f:
                json.dump(document, f, indent=2)
                
            # Process chunks
            print(f"Creating chunks for {filename}...")
            chunks = chunking_engine.chunk_document(document)
            
            # Save each chunk as a separate file
            oversized_count = 0
            for i, chunk in enumerate(chunks):
                chunk_path = os.path.join(file_dir, f"chunk_{i+1:04d}.txt")
                with open(chunk_path, 'w') as f:
                    f.write(f"CHUNK #{i+1}\n")
                    f.write(f"Length: {len(chunk['content'])} characters\n")
                    f.write(f"Metadata: {json.dumps(chunk['metadata'], indent=2)}\n\n")
                    f.write(f"CONTENT:\n{chunk['content']}\n")
                
                # If chunk is oversized, flag it
                if len(chunk['content']) > config.chunk_size:
                    oversized_count += 1
                    with open(os.path.join(file_dir, f"chunk_{i+1:04d}.OVERSIZED"), 'w') as f:
                        f.write(f"WARNING: Chunk size {len(chunk['content'])} exceeds limit {config.chunk_size}\n")
            
            # Save a summary file
            with open(os.path.join(file_dir, "summary.txt"), 'w') as f:
                f.write(f"File: {filename}\n")
                f.write(f"Document type: {detected_type}\n")
                f.write(f"Total chunks: {len(chunks)}\n")
                oversized = [i for i, c in enumerate(chunks) if len(c['content']) > config.chunk_size]
                f.write(f"Oversized chunks: {len(oversized)}/{len(chunks)}\n")
                if oversized:
                    f.write(f"Oversized chunk numbers: {', '.join(str(i+1) for i in oversized)}\n")
                
            print(f"Saved {len(chunks)} chunks for {filename} ({oversized_count} oversized)")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            
    print(f"\nChunk inspection complete. Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inspect document chunks')
    parser.add_argument('--data-dir', default="/data", help='Directory containing input files')
    parser.add_argument('--output-dir', default="test_chunks/standard", help='Directory to save chunk inspection results')
    parser.add_argument('--config-path', help='Path to the pipeline configuration file')
    
    args = parser.parse_args()
    
    inspect_chunks(args.data_dir, args.output_dir, args.config_path)