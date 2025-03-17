#!/usr/bin/env python3
"""
Enhanced Chunking Script

This script implements an enhanced version of the document chunking process
with lemmatization and selective POS filtering to improve chunk quality.
It's a duplicate of the main pipeline with these additional NLP features.
"""
import os
import json
import sys
import argparse
import re
from typing import Dict, List, Any, Optional
from datetime import datetime

# Import from the existing pipeline
from src.pipeline import PipelineConfig, TextProcessor, FileReader
from src.utils.chunking import ChunkingEngine, DOCUMENT_TYPES, SPACY_AVAILABLE, SPACY_MODEL_LOADED
from src.utils.embedding import EmbeddingGenerator
from src.utils.vector_db import VectorDatabaseWriter

# Try to import spaCy - we'll need it for our enhancements
try:
    import spacy
    ENHANCED_SPACY_AVAILABLE = True
    print(f"Enhanced spaCy processing is available")
except ImportError:
    ENHANCED_SPACY_AVAILABLE = False
    print(f"Enhanced spaCy processing is NOT available")

class EnhancedChunkingEngine(ChunkingEngine):
    """
    Enhanced version of the ChunkingEngine with lemmatization and POS filtering.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the enhanced chunking engine.
        
        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Overlap between chunks
        """
        # Initialize the parent class
        super().__init__(chunk_size, chunk_overlap)
        
        # Initialize spaCy with POS tagging and lemmatization pipeline
        self.nlp = None
        if ENHANCED_SPACY_AVAILABLE:
            try:
                # Load spaCy model with only the components we need
                print(f"Loading spaCy model for enhanced processing: {SPACY_MODEL_LOADED or SPACY_MODEL or 'en_core_web_sm'}")
                # Keep the parser for sentence boundaries or use sentencizer
                self.nlp = spacy.load(SPACY_MODEL_LOADED or SPACY_MODEL or "en_core_web_sm",
                                     disable=["ner"])  # Only disable ner to keep sentence boundaries
                
                # Add sentencizer if the parser is not available
                if not self.nlp.has_pipe("parser"):
                    self.nlp.add_pipe("sentencizer")
                
                print(f"Successfully loaded spaCy model for lemmatization and POS filtering")
                
                # Set up important POS tags (content-bearing parts of speech)
                # NOUN: nouns, PROPN: proper nouns, VERB: verbs, ADJ: adjectives, ADV: adverbs
                self.content_pos = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}
                
            except Exception as e:
                print(f"Error loading spaCy for enhanced processing: {e}")
                self.nlp = None
    
    def lemmatize_text(self, text: str) -> str:
        """
        Apply lemmatization to text while preserving structure.
        
        Args:
            text: Text to lemmatize
            
        Returns:
            Lemmatized text
        """
        if not self.nlp:
            return text  # Return original if spaCy not available
            
        try:
            # Process the text with spaCy
            doc = self.nlp(text)
            
            # Replace each token with its lemma if it's a content word
            lemmatized_tokens = []
            for token in doc:
                # Use lemma for content words, original for others
                if token.pos_ in self.content_pos:
                    lemmatized_tokens.append(token.lemma_)
                else:
                    lemmatized_tokens.append(token.text)
            
            # Reconstruct the text
            lemmatized_text = " ".join(lemmatized_tokens)
            
            # Clean up excessive whitespace
            lemmatized_text = re.sub(r'\s+', ' ', lemmatized_text)
            lemmatized_text = re.sub(r'\s([.,;:!?])', r'\1', lemmatized_text)
            
            return lemmatized_text
        except Exception as e:
            print(f"Error during lemmatization: {e}")
            return text  # Return original if error
    
    def find_linguistic_boundaries(self, text: str, target_size: int) -> List[int]:
        """
        Find better chunk boundaries based on linguistic features.
        
        Args:
            text: Text to analyze
            target_size: Target chunk size
            
        Returns:
            List of indices where text should be split
        """
        if not self.nlp or len(text) <= target_size:
            # Fall back to character-based chunking if spaCy not available
            # or text is already small enough
            return []
        
        try:
            # Process with spaCy
            doc = self.nlp(text)
            
            # Check if we have sentence boundaries
            boundaries = []
            try:
                # Try to get sentence boundaries safely
                boundaries = [sent.end_char for sent in doc.sents]
            except ValueError as e:
                # This happens when sentence boundaries aren't set
                print(f"Warning: Couldn't get sentence boundaries: {e}")
                # Will continue with token-based boundaries
            
            # If we don't have sentence boundaries, or not enough, add token-based boundaries
            if not boundaries or (len(text) > target_size and
                                 (not boundaries or boundaries[-1] < target_size)):
                # Fall back to token-based boundaries
                content_boundaries = []
                
                # Add boundaries after punctuation
                for i, token in enumerate(doc):
                    if token.is_punct and token.text in ['.', '!', '?', ';', ':', '\n', '\r\n']:
                        content_boundaries.append(token.idx + len(token.text))
                    
                    # Also add boundaries after content words followed by punctuation
                    # or at transition points between content and function words
                    if token.pos_ in self.content_pos and i < len(doc) - 1:
                        if doc[i+1].is_punct or doc[i+1].pos_ not in self.content_pos:
                            content_boundaries.append(token.idx + len(token.text))
                
                # Combine all boundaries
                boundaries.extend(content_boundaries)
                if boundaries:
                    boundaries = sorted(set(boundaries))  # Remove duplicates
                else:
                    # If still no good boundaries, create artificial ones based on target size
                    chunk_size = target_size
                    boundaries = list(range(chunk_size, len(text), chunk_size))
            
            # Filter boundaries to ones that help create chunks of the right size
            filtered_boundaries = []
            last_boundary = 0
            for b in boundaries:
                # Only use this boundary if it's not too close to the last one
                # and not too far past the target size
                if (b - last_boundary >= target_size * 0.5 and 
                    b - last_boundary <= target_size * 1.5):
                    filtered_boundaries.append(b)
                    last_boundary = b
            
            return filtered_boundaries
            
        except Exception as e:
            print(f"Error finding linguistic boundaries: {e}")
            return []  # Return empty if error
    
    def create_enhanced_chunks(self, text: str, doc_type: str) -> List[str]:
        """
        Create chunks using enhanced linguistic-aware splitting.
        
        Args:
            text: Document text
            doc_type: Document type
            
        Returns:
            List of text chunks
        """
        # Apply lemmatization first
        lemmatized_text = self.lemmatize_text(text)
        
        # Find linguistic boundaries
        boundaries = self.find_linguistic_boundaries(lemmatized_text, self.chunk_size)
        
        # If we have good boundaries, use them
        if boundaries:
            chunks = []
            start = 0
            for end in boundaries:
                chunk = lemmatized_text[start:end]
                chunks.append(chunk)
                # Apply overlap - start a bit before the end of the previous chunk
                start = max(0, end - self.chunk_overlap)
            
            # Don't forget the last chunk
            if start < len(lemmatized_text):
                chunks.append(lemmatized_text[start:])
            
            return chunks
        
        # If linguistic boundaries didn't work, fall back to the original chunking method
        print(f"Falling back to standard chunking for document type: {doc_type}")
        if doc_type == "scientific":
            return self.create_scientific_chunks(lemmatized_text)
        elif doc_type == "financial":
            return self.create_financial_chunks(lemmatized_text)
        elif doc_type == "technical":
            return self.create_technical_chunks(lemmatized_text)
        elif doc_type == "narrative":
            return self.create_narrative_chunks(lemmatized_text)
        else:  # general
            return self.create_general_chunks(lemmatized_text)
    
    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Enhanced chunking with lemmatization and POS filtering.
        
        Args:
            document: Document to chunk
            
        Returns:
            List of chunks with metadata
        """
        # Get document type
        doc_type = self.prompt_for_document_type(document)
        
        # Store document type in metadata
        document['metadata']['doc_type'] = doc_type
        
        print(f"Enhanced chunking for document type: {doc_type}")
        
        # First, try the enhanced chunking with linguistic boundaries
        chunks = []
        if self.nlp:
            try:
                chunks = self.create_enhanced_chunks(document['content'], doc_type)
                print(f"Created {len(chunks)} chunks using enhanced linguistic chunking")
            except Exception as e:
                print(f"Error in enhanced chunking: {e}")
                chunks = []
        
        # If enhanced chunking failed, produced no chunks, or is not available, fall back to standard chunking
        if not chunks:
            print(f"Enhanced chunking failed or produced no chunks. Falling back to standard chunking.")
            return super().chunk_document(document)
        
        # Enforce maximum chunk size - even with linguistic boundaries, we want to ensure
        # no chunks exceed the maximum size
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > self.chunk_size:
                print(f"Created a chunk of size {len(chunk)}, which is longer than the specified {self.chunk_size}")
                # Re-split this chunk using the default splitter
                # Apply more aggressive chunking for oversized chunks
                # Use lower chunk size target to ensure we don't exceed the limit
                aggressive_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size - 100,  # Use smaller target to ensure we stay under limit
                    chunk_overlap=self.chunk_overlap // 2,  # Use smaller overlap
                    separators=["\n\n", "\n", ". ", " ", ""]  # More aggressive splitting
                )
                
                sub_chunks = aggressive_splitter.split_text(chunk)
                
                # Double-check the sub-chunks to ensure they're within limits
                # Sometimes even the aggressive splitter might still create oversized chunks
                for sub_chunk in sub_chunks:
                    if len(sub_chunk) > self.chunk_size:
                        # Forcibly split by character count as a last resort
                        print(f"Forcing character-based split for chunk of size {len(sub_chunk)}")
                        forced_chunks = []
                        for i in range(0, len(sub_chunk), self.chunk_size - 100):
                            end = min(i + self.chunk_size - 100, len(sub_chunk))
                            forced_chunk = sub_chunk[i:end]
                            # Final safety check
                            assert len(forced_chunk) <= self.chunk_size, f"Forced chunk still oversized: {len(forced_chunk)}"
                            forced_chunks.append(forced_chunk)
                        final_chunks.extend(forced_chunks)
                    else:
                        final_chunks.append(sub_chunk)
            else:
                final_chunks.append(chunk)
        
        # Add metadata to chunks
        result = []
        for i, chunk in enumerate(final_chunks):
            chunk_metadata = document['metadata'].copy()
            chunk_metadata['chunk_id'] = i
            chunk_metadata['enhanced'] = True
            
            result.append({
                'content': chunk,
                'metadata': chunk_metadata
            })
        
        print(f"  Created {len(result)} chunks with enhanced processing")
        return result


def process_with_enhanced_chunking(data_dir, output_dir='enhanced_chunks_output', config_path=None):
    """
    Process files using enhanced chunking with lemmatization and POS filtering.
    
    Args:
        data_dir: Directory containing input files
        output_dir: Directory to save chunk results
        config_path: Path to the pipeline configuration file
    """
    # Use the default config path if not provided
    if not config_path:
        config_path = os.environ.get("CONFIG_PATH", "config/pipeline_config.json")
        
    print(f"Loading configuration from {config_path}")
    config = PipelineConfig(config_path)
    
    # Override directories
    config.data_dir = data_dir
    
    # Create processors
    print(f"Initializing processors with chunk size: {config.chunk_size}, overlap: {config.chunk_overlap}")
    text_processor = TextProcessor(config)
    
    # Create our enhanced chunking engine instead of the standard one
    enhanced_chunking_engine = EnhancedChunkingEngine(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap
    )
    
    # For comparison, also create a standard chunking engine
    standard_chunking_engine = ChunkingEngine(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap
    )
    
    # Reset document type memory for both engines
    print("Resetting document type memory")
    enhanced_chunking_engine.reset_document_type_memory()
    standard_chunking_engine.reset_document_type_memory()
    
    # Create output directories
    enhanced_output_dir = os.path.join(output_dir, "enhanced")
    standard_output_dir = os.path.join(output_dir, "standard")
    os.makedirs(enhanced_output_dir, exist_ok=True)
    os.makedirs(standard_output_dir, exist_ok=True)
    print(f"Created output directories: {enhanced_output_dir} and {standard_output_dir}")
    
    # Find all files to process
    files = []
    for root, _, filenames in os.walk(data_dir):
        for filename in filenames:
            file_path = os.path.join(root, filename)
            _, ext = os.path.splitext(file_path)
            if ext.lower() in ['.pdf', '.txt', '.md', '.docx', '.doc']:
                files.append(file_path)
                
    print(f"Found {len(files)} files to process")
    
    # Process each file with both chunking methods
    for file_path in files:
        try:
            filename = os.path.basename(file_path)
            print(f"\nProcessing {filename}...")
            
            # Process using the same methods as the main pipeline
            document = text_processor.process_file(file_path)
            
            # Get document type
            detected_type = enhanced_chunking_engine.detect_document_type(document['content'])
            print(f"Detected document type: {detected_type}")
            
            # Create directories for this file's chunks
            enhanced_file_dir = os.path.join(enhanced_output_dir, os.path.splitext(filename)[0])
            standard_file_dir = os.path.join(standard_output_dir, os.path.splitext(filename)[0])
            os.makedirs(enhanced_file_dir, exist_ok=True)
            os.makedirs(standard_file_dir, exist_ok=True)
            
            # Save the original document in both directories
            with open(os.path.join(enhanced_file_dir, "original.json"), 'w') as f:
                json.dump(document, f, indent=2)
            with open(os.path.join(standard_file_dir, "original.json"), 'w') as f:
                json.dump(document, f, indent=2)
            
            # Process with enhanced chunking
            print(f"Creating enhanced chunks for {filename}...")
            enhanced_chunks = enhanced_chunking_engine.chunk_document(document)
            
            # For comparison, also process with standard chunking
            print(f"Creating standard chunks for {filename}...")
            standard_chunks = standard_chunking_engine.chunk_document(document)
            
            # Save enhanced chunks
            enhanced_oversized = 0
            for i, chunk in enumerate(enhanced_chunks):
                chunk_path = os.path.join(enhanced_file_dir, f"chunk_{i+1:04d}.txt")
                with open(chunk_path, 'w') as f:
                    f.write(f"ENHANCED CHUNK #{i+1}\n")
                    f.write(f"Length: {len(chunk['content'])} characters\n")
                    f.write(f"Metadata: {json.dumps(chunk['metadata'], indent=2)}\n\n")
                    f.write(f"CONTENT:\n{chunk['content']}\n")
                
                # If chunk is oversized, flag it
                if len(chunk['content']) > config.chunk_size:
                    enhanced_oversized += 1
                    with open(os.path.join(enhanced_file_dir, f"chunk_{i+1:04d}.OVERSIZED"), 'w') as f:
                        f.write(f"WARNING: Chunk size {len(chunk['content'])} exceeds limit {config.chunk_size}\n")
            
            # Save standard chunks
            standard_oversized = 0
            for i, chunk in enumerate(standard_chunks):
                chunk_path = os.path.join(standard_file_dir, f"chunk_{i+1:04d}.txt")
                with open(chunk_path, 'w') as f:
                    f.write(f"STANDARD CHUNK #{i+1}\n")
                    f.write(f"Length: {len(chunk['content'])} characters\n")
                    f.write(f"Metadata: {json.dumps(chunk['metadata'], indent=2)}\n\n")
                    f.write(f"CONTENT:\n{chunk['content']}\n")
                
                # If chunk is oversized, flag it
                if len(chunk['content']) > config.chunk_size:
                    standard_oversized += 1
                    with open(os.path.join(standard_file_dir, f"chunk_{i+1:04d}.OVERSIZED"), 'w') as f:
                        f.write(f"WARNING: Chunk size {len(chunk['content'])} exceeds limit {config.chunk_size}\n")
            
            # Save enhanced summary
            with open(os.path.join(enhanced_file_dir, "summary.txt"), 'w') as f:
                f.write(f"File: {filename}\n")
                f.write(f"Document type: {detected_type}\n")
                f.write(f"Total enhanced chunks: {len(enhanced_chunks)}\n")
                f.write(f"Oversized chunks: {enhanced_oversized}/{len(enhanced_chunks)}\n")
                
            # Save standard summary
            with open(os.path.join(standard_file_dir, "summary.txt"), 'w') as f:
                f.write(f"File: {filename}\n")
                f.write(f"Document type: {detected_type}\n")
                f.write(f"Total standard chunks: {len(standard_chunks)}\n")
                f.write(f"Oversized chunks: {standard_oversized}/{len(standard_chunks)}\n")
            
            # Save comparison summary
            with open(os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_comparison.txt"), 'w') as f:
                f.write(f"File: {filename}\n")
                f.write(f"Document type: {detected_type}\n\n")
                f.write(f"Enhanced chunking: {len(enhanced_chunks)} chunks, {enhanced_oversized} oversized\n")
                f.write(f"Standard chunking: {len(standard_chunks)} chunks, {standard_oversized} oversized\n\n")
                f.write(f"Difference: {len(enhanced_chunks) - len(standard_chunks)} chunks\n")
            
            print(f"Enhanced: {len(enhanced_chunks)} chunks ({enhanced_oversized} oversized)")
            print(f"Standard: {len(standard_chunks)} chunks ({standard_oversized} oversized)")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            
    print(f"\nProcessing complete. Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process documents with enhanced chunking')
    parser.add_argument('--data-dir', default="/data", help='Directory containing input files')
    parser.add_argument('--output-dir', default="enhanced_chunks_output", help='Directory to save results')
    parser.add_argument('--config-path', help='Path to the pipeline configuration file')
    
    args = parser.parse_args()
    
    process_with_enhanced_chunking(args.data_dir, args.output_dir, args.config_path)