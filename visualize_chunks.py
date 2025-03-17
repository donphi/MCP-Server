#!/usr/bin/env python3
"""
Temporary script to visualize document chunking with spaCy.
This script takes a file path as input (including PDFs), chunks it using the enhanced chunking system,
and outputs the chunks to see how the document is divided.

Usage:
    python visualize_chunks.py /path/to/your/file.pdf [document_type]

    document_type (optional): "scientific", "financial", "technical", "narrative", or "general"
                              If not provided, it will be auto-detected.
"""

import os
import sys
import json
from typing import Dict, Any, List
import argparse

# Import langchain document loaders for different file types
from langchain_community.document_loaders import (
    UnstructuredMarkdownLoader, 
    TextLoader, 
    PyPDFLoader, 
    Docx2txtLoader,
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader
)

# Import the chunking engine and spaCy model name
from src.utils.chunking import ChunkingEngine, DOCUMENT_TYPES, SPACY_MODEL

def load_document(file_path: str) -> Dict[str, Any]:
    """
    Load a document using the appropriate loader based on file extension.
    
    Args:
        file_path: Path to the document
        
    Returns:
        Document as a dictionary with content and metadata
    """
    print(f"Loading document: {file_path}")
    
    filename = os.path.basename(file_path)
    _, file_ext = os.path.splitext(file_path)
    file_ext = file_ext.lower()
    
    # Select the appropriate loader based on file extension
    if file_ext == '.md':
        loader = UnstructuredMarkdownLoader(file_path)
    elif file_ext == '.txt':
        loader = TextLoader(file_path)
    elif file_ext == '.pdf':
        try:
            print("Trying PyPDFLoader...")
            loader = PyPDFLoader(file_path)
        except Exception as e:
            print(f"PyPDFLoader failed: {e}")
            print("Falling back to UnstructuredPDFLoader...")
            try:
                loader = UnstructuredPDFLoader(file_path, mode="elements")
            except Exception as e:
                print(f"UnstructuredPDFLoader with 'elements' mode failed: {e}")
                print("Trying UnstructuredPDFLoader in 'single' mode...")
                loader = UnstructuredPDFLoader(file_path, mode="single")
    elif file_ext == '.docx':
        loader = Docx2txtLoader(file_path)
    elif file_ext == '.doc':
        loader = UnstructuredWordDocumentLoader(file_path)
    else:
        # Default to text loader for unknown extensions
        print(f"Warning: Unknown file extension '{file_ext}'. Trying as text file.")
        loader = TextLoader(file_path)
    
    try:
        # Load the document
        loaded_docs = loader.load()
        
        if not loaded_docs:
            print("Warning: No content loaded from document.")
            return {'content': '', 'metadata': {'source': file_path, 'title': filename, 'file_type': file_ext[1:]}}
        
        # Get the first document (most loaders return a list)
        doc = loaded_docs[0]
        
        # Extract content and metadata
        content = doc.page_content
        metadata = {
            'source': file_path,
            'title': filename,
            'file_type': file_ext[1:] if file_ext else 'txt',
        }
        
        # Add any additional metadata from the loader
        if hasattr(doc, 'metadata'):
            for key, value in doc.metadata.items():
                if key not in ['source']:  # Avoid overwriting our keys
                    metadata[key] = value
        
        return {'content': content, 'metadata': metadata}
    
    except Exception as e:
        print(f"Error loading document: {e}")
        raise

def print_chunk_info(chunks, max_preview_length=200):
    """
    Print information about the chunks in a readable format.
    
    Args:
        chunks: List of document chunks
        max_preview_length: Maximum length of content preview
    """
    print(f"\nTotal chunks created: {len(chunks)}")
    print("=" * 80)
    
    for i, chunk in enumerate(chunks):
        content = chunk['content']
        metadata = chunk['metadata']
        
        # Get a preview of the content, truncated if too long
        if len(content) > max_preview_length:
            content_preview = content[:max_preview_length] + "..."
        else:
            content_preview = content
            
        # Replace newlines with spaces for better display
        content_preview = content_preview.replace('\n', ' ')
        
        print(f"CHUNK #{i+1}")
        print(f"Length: {len(content)} characters")
        print(f"Metadata: {json.dumps(metadata, indent=2)}")
        print(f"Content Preview: {content_preview}")
        print("=" * 80)

def save_chunks_to_file(chunks, output_path):
    """
    Save all chunks to a file for inspection.
    
    Args:
        chunks: List of document chunks
        output_path: Path to save the output
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, chunk in enumerate(chunks):
            f.write(f"{'='*80}\n")
            f.write(f"CHUNK #{i+1}\n")
            f.write(f"{'='*80}\n\n")
            f.write(chunk['content'])
            f.write("\n\n")
    
    print(f"Saved detailed chunks to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Visualize document chunking')
    parser.add_argument('file_path', help='Path to the document to chunk')
    parser.add_argument('--type', choices=list(DOCUMENT_TYPES.keys()), 
                        help='Force a specific document type instead of auto-detection')
    parser.add_argument('--chunk-size', type=int, default=1000,
                        help='Size of chunks (default: 1000)')
    parser.add_argument('--chunk-overlap', type=int, default=200,
                        help='Overlap between chunks (default: 200)')
    parser.add_argument('--output', default='chunks_output.txt',
                        help='Path to save detailed chunks (default: chunks_output.txt)')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.file_path):
        print(f"Error: File '{args.file_path}' not found")
        return 1
    
    # Verify that spaCy model is available
    try:
        import spacy
        print(f"Verifying spaCy model: {SPACY_MODEL}")
        try:
            nlp = spacy.load(SPACY_MODEL)
            print(f"Successfully loaded spaCy model: {SPACY_MODEL}")
        except OSError:
            print(f"SpaCy model '{SPACY_MODEL}' not found. Attempting to download...")
            import subprocess
            subprocess.run([sys.executable, "-m", "spacy", "download", SPACY_MODEL], check=True)
            print(f"Downloaded spaCy model: {SPACY_MODEL}")
            nlp = spacy.load(SPACY_MODEL)
    except Exception as e:
        print(f"Warning: spaCy setup failed: {e}")
        print("Will use fallback chunking methods.")
    
    # Create chunking engine with specified parameters
    print(f"Creating chunking engine with size={args.chunk_size}, overlap={args.chunk_overlap}")
    chunking_engine = ChunkingEngine(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    # Load the document using appropriate loader
    try:
        document = load_document(args.file_path)
        content = document['content']
        
        print(f"Successfully loaded document. Content length: {len(content)} characters")
        
        if len(content) < 100:  # Just a sanity check
            print("Warning: Document content is very short. Preview:")
            print(content)
    except Exception as e:
        print(f"Failed to load document: {e}")
        return 1
    
    # Auto-detect document type
    detected_type = chunking_engine.detect_document_type(content)
    print(f"Auto-detected document type: {detected_type}")
    
    # Override with user-specified type if provided
    if args.type:
        document['metadata']['doc_type'] = args.type
        print(f"Using user-specified document type: {args.type}")
    else:
        document['metadata']['doc_type'] = detected_type
    
    # Process the document based on its type
    print(f"Processing document as: {document['metadata']['doc_type']}")
    if document['metadata']['doc_type'] == "scientific":
        chunks = chunking_engine.create_scientific_chunks(content)
    elif document['metadata']['doc_type'] == "financial":
        chunks = chunking_engine.create_financial_chunks(content)
    elif document['metadata']['doc_type'] == "technical":
        chunks = chunking_engine.create_technical_chunks(content)
    elif document['metadata']['doc_type'] == "narrative":
        chunks = chunking_engine.create_narrative_chunks(content)
    else:  # general
        chunks = chunking_engine.create_general_chunks(content)
    
    # Add metadata to chunks
    result = []
    for i, chunk in enumerate(chunks):
        chunk_metadata = document['metadata'].copy()
        chunk_metadata['chunk_id'] = i
        
        result.append({
            'content': chunk,
            'metadata': chunk_metadata
        })
    
    # Print information about the chunks
    print_chunk_info(result)
    
    # Save detailed chunks to a file
    save_chunks_to_file(result, args.output)
    
    print(f"\nTo test with different document type:")
    print(f"python {sys.argv[0]} {args.file_path} --type scientific")
    print(f"python {sys.argv[0]} {args.file_path} --type financial")
    print(f"python {sys.argv[0]} {args.file_path} --type technical")
    print(f"python {sys.argv[0]} {args.file_path} --type narrative")
    print(f"python {sys.argv[0]} {args.file_path} --type general")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())