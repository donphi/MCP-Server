"""
Chunking utilities for the MCP Server.
"""
import re
import os
from typing import Dict, Any, List, Optional, Tuple, Union
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

# Get spaCy model name from environment variable or use default
# Note: We provide two backup options to ensure the code works
SPACY_MODEL = os.environ.get("SPACY_MODEL", "en_core_web_md")
FALLBACK_MODEL = "en_core_web_sm"

# Set up a global flag for spaCy availability
SPACY_AVAILABLE = False
SPACY_MODEL_LOADED = None  # Track which model is actually loaded

# Try to import spaCy, but don't fail if it's not available
try:
    import spacy
    from langchain.text_splitter import SpacyTextSplitter
    
    # Try to load the primary model or fallback
    try:
        print(f"Attempting to load spaCy model: {SPACY_MODEL}")
        nlp = spacy.load(SPACY_MODEL)
        SPACY_AVAILABLE = True
        SPACY_MODEL_LOADED = SPACY_MODEL
        print(f"Successfully loaded spaCy model: {SPACY_MODEL}")
    except Exception as e:
        print(f"Failed to load primary model {SPACY_MODEL}: {e}")
        try:
            # Try with fallback model
            print(f"Attempting to load fallback spaCy model: {FALLBACK_MODEL}")
            nlp = spacy.load(FALLBACK_MODEL)
            SPACY_AVAILABLE = True
            SPACY_MODEL_LOADED = FALLBACK_MODEL
            print(f"Successfully loaded fallback spaCy model: {FALLBACK_MODEL}")
        except Exception as e:
            print(f"Failed to load fallback model {FALLBACK_MODEL}: {e}")
            print("WARNING: spaCy models not found. Using fallback chunking strategy.")
except ImportError as e:
    print(f"spaCy not available ({e}). Using fallback chunking strategy.")

# Print status for debugging
if SPACY_AVAILABLE:
    print(f"spaCy is available and using model: {SPACY_MODEL_LOADED}")
else:
    print("spaCy is NOT available. Will use basic text splitting only.")

# Document type descriptions for prompting
DOCUMENT_TYPES = {
    "scientific": "Scientific/academic papers with formal sections, citations, and technical terminology",
    "financial": "Financial documents with tables, numerical data, and business terminology",
    "technical": "Technical documentation, manuals, or specifications",
    "narrative": "Text-heavy narrative documents like reports or articles",
    "general": "General purpose documents with mixed content (default)",
}

class ChunkingEngine:
    """
    Enhanced engine for chunking documents using various strategies.
    Falls back to basic strategies when spaCy is not available.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the chunking engine.
        
        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Default text splitter (always available)
        self.default_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n## ", "\n### ", "\n#### ", "\n", ". ", " ", ""]
        )
        
        # Markdown-specific splitter for documents with clear headers
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "header1"),
                ("##", "header2"),
                ("###", "header3"),
                ("####", "header4"),
            ]
        )
        
        # spaCy-based splitter (only if available)
        self.spacy_splitter = None
        if SPACY_AVAILABLE and SPACY_MODEL_LOADED:
            try:
                from langchain.text_splitter import SpacyTextSplitter
                print(f"Initializing SpacyTextSplitter with pipeline: {SPACY_MODEL_LOADED}")
                self.spacy_splitter = SpacyTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    pipeline=SPACY_MODEL_LOADED  # Use the model that was actually loaded
                )
                print("Successfully initialized SpacyTextSplitter")
            except Exception as e:
                print(f"Failed to initialize SpacyTextSplitter: {e}")
        
        # Store user document type selections to avoid repeated prompts
        self.document_type_memory = {}
    
    def detect_document_type(self, content: str) -> str:
        """
        Attempt to automatically detect the document type based on content analysis.
        
        Args:
            content: Document content
            
        Returns:
            Detected document type
        """
        # Use regex patterns to detect document type even without spaCy
        
        # Check for scientific/academic patterns
        scientific_patterns = [
            r'\b(?:fig\.|figure|table|eq\.|equation)\s+\d+\b',
            r'\b(?:et\s+al\.|\(\d{4}\)|\[\d+\])\b',
            r'\bAbstract\b.*\bIntroduction\b.*\bMethodology\b',
            r'\bReferences\b.*\d+\.\s+[A-Z][a-z]+,\s+[A-Z]\.',
        ]
        scientific_score = sum(bool(re.search(pattern, content, re.I)) for pattern in scientific_patterns)
        
        # Check for financial patterns
        financial_patterns = [
            r'\$\s*\d+(?:,\d{3})*(?:\.\d+)?',
            r'\b(?:USD|EUR|GBP|JPY)\b',
            r'\b(?:revenue|profit|margin|dividend|fiscal|quarterly)\b',
            r'(?:\d+\s*%)|(?:percent(?:age)?)',
            r'\bQ[1-4]\b|\bFY\d{2,4}\b',
        ]
        financial_score = sum(bool(re.search(pattern, content, re.I)) for pattern in financial_patterns)
        
        # Check for technical documentation patterns
        technical_patterns = [
            r'\b(?:function|class|method|api|interface|parameter)\b',
            r'\b(?:installation|configuration|setup|deployment)\b',
            r'```[a-z]*\n[\s\S]*?```',  # Code blocks
            r'\b(?:version|v\d+\.\d+\.\d+)\b',
        ]
        technical_score = sum(bool(re.search(pattern, content, re.I)) for pattern in technical_patterns)
        
        # Detect narrative text based on sentence structure - simplified without spaCy
        narrative_score = 0
        # Count sentences with more than 20 words as an indicator of narrative text
        sentences = re.split(r'[.!?]+', content)
        long_sentences = sum(1 for s in sentences if len(s.split()) > 20)
        if long_sentences > len(sentences) * 0.3:  # If 30% of sentences are long
            narrative_score = 1
        
        # Determine document type based on highest score
        scores = {
            "scientific": scientific_score,
            "financial": financial_score,
            "technical": technical_score,
            "narrative": narrative_score,
            "general": 1  # Base score for general
        }
        
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def get_document_key(self, document: Dict[str, Any]) -> str:
        """
        Generate a key for document type memory based on file extension and size range.
        This helps identify similar documents.
        
        Args:
            document: Document to process
            
        Returns:
            A string key representing the document's category
        """
        file_path = document['metadata'].get('source', '')
        file_name = os.path.basename(file_path)
        _, ext = os.path.splitext(file_path)
        content_size = len(document['content'])
        
        # Group documents by extension and size category
        # Size categories: small (<100KB), medium (100KB-1MB), large (>1MB)
        if content_size < 100_000:
            size_category = "small"
        elif content_size < 1_000_000:
            size_category = "medium"
        else:
            size_category = "large"
        
        # Create a key that represents similar documents
        return f"{ext.lower()}_{size_category}"
    
    def prompt_for_document_type(self, document: Dict[str, Any]) -> str:
        """
        Ask the user to select the document type for better chunking,
        but only if we haven't already processed a similar document.
        
        Args:
            document: Document to be processed
            
        Returns:
            Selected document type
        """
        # Get file metadata
        file_path = document['metadata'].get('source', '')
        file_name = os.path.basename(file_path)
        file_type = document['metadata'].get('file_type', '')
        
        # Try to detect the document type
        detected_type = self.detect_document_type(document['content'][:50000])
        
        # Get document key for memory lookup
        doc_key = self.get_document_key(document)
        
        # Check if we've already processed a similar document
        if doc_key in self.document_type_memory:
            remembered_type = self.document_type_memory[doc_key]
            print(f"\nFile: {file_name} (Type: {file_type})")
            print(f"Detected document type: {detected_type}")
            print(f"Using previously selected type: {remembered_type} (no prompt needed)")
            return remembered_type
        
        # If this is the first document of its kind, prompt the user
        print("\nDocument Type Detection")
        print(f"File: {file_name} (Type: {file_type})")
        print(f"Detected document type: {detected_type} - {DOCUMENT_TYPES[detected_type]}")
        print("\nAvailable document types:")
        
        for i, (doc_type, description) in enumerate(DOCUMENT_TYPES.items(), 1):
            print(f"  {i}. {doc_type}: {description}")
            
        while True:
            try:
                type_map = {i: doc_type for i, doc_type in enumerate(DOCUMENT_TYPES.keys(), 1)}
                # Default to the detected type
                default_index = list(DOCUMENT_TYPES.keys()).index(detected_type) + 1
                
                choice = input(f"\nSelect document type (1-{len(DOCUMENT_TYPES)}) [default: {default_index}]: ")
                if choice.strip() == "":
                    selected_type = detected_type
                else:
                    choice = int(choice)
                    if 1 <= choice <= len(DOCUMENT_TYPES):
                        selected_type = type_map[choice]
                    else:
                        print(f"Please enter a number between 1 and {len(DOCUMENT_TYPES)}")
                        continue
                
                # Remember this selection for similar documents
                self.document_type_memory[doc_key] = selected_type
                return selected_type
            except ValueError:
                print("Please enter a valid number")
    
    def create_scientific_chunks(self, text: str) -> List[str]:
        """
        Create chunks optimized for scientific content.
        
        Args:
            text: Document text
            
        Returns:
            List of text chunks
        """
        # First try to split on section headers
        if "## " in text or "# " in text:
            # Process with Markdown header splitter first
            splits = self.markdown_splitter.split_text(text)
            # Further split long sections
            chunks = []
            for split in splits:
                if len(split.page_content) > self.chunk_size:
                    if self.spacy_splitter and SPACY_AVAILABLE:
                        sub_chunks = self.spacy_splitter.split_text(split.page_content)
                    else:
                        sub_chunks = self.default_splitter.split_text(split.page_content)
                    for chunk in sub_chunks:
                        # Add header context to each chunk
                        header_context = ""
                        if hasattr(split, 'metadata'):
                            for i in range(1, 5):
                                key = f"header{i}"
                                if key in split.metadata and split.metadata[key]:
                                    header_context += f"{'#' * i} {split.metadata[key]}\n"
                        chunks.append(header_context + chunk if header_context else chunk)
                else:
                    chunks.append(split.page_content)
        elif self.spacy_splitter and SPACY_AVAILABLE:
            # Use spaCy for smarter linguistic splitting
            chunks = self.spacy_splitter.split_text(text)
        else:
            # Fallback to default splitting
            chunks = self.default_splitter.split_text(text)
            
        # Preserve references section as a single chunk if possible
        ref_pattern = r'\n(?:References|Bibliography|Works Cited|Literature Cited)\n'
        for i, chunk in enumerate(chunks):
            if re.search(ref_pattern, chunk, re.I):
                # Check if this is a reference section start
                ref_match = re.search(ref_pattern, chunk, re.I)
                if ref_match:
                    ref_start_pos = ref_match.start()
                    # If reference section is not near the beginning, it's likely the actual references section
                    if ref_start_pos > len(chunk) * 0.7:
                        # Extract complete references across multiple chunks if needed
                        ref_section = chunk[ref_start_pos:]
                        j = i + 1
                        while j < len(chunks) and len(ref_section) < self.chunk_size * 1.5:
                            ref_section += chunks[j]
                            chunks.pop(j)
                        # Replace the current chunk with content before references
                        chunks[i] = chunk[:ref_start_pos]
                        # Add references as a separate chunk
                        chunks.append(ref_section)
                        break
        
        return chunks
    
    def create_financial_chunks(self, text: str) -> List[str]:
        """
        Create chunks optimized for financial content.
        
        Args:
            text: Document text
            
        Returns:
            List of text chunks
        """
        # Identify table blocks
        table_pattern = r'(?:\n\|[^\n]+\|[^\n]+\n\|[-:| ]+\|[-:| ]+\n(?:\|[^\n]+\|[^\n]+\n)+)'
        tables = re.finditer(table_pattern, text)
        table_spans = [(m.start(), m.end()) for m in tables]
        
        # Split around tables
        chunks = []
        last_end = 0
        
        for start, end in table_spans:
            # Add text before table
            if start > last_end:
                pre_text = text[last_end:start]
                if len(pre_text) > self.chunk_size:
                    if self.spacy_splitter and SPACY_AVAILABLE:
                        pre_chunks = self.spacy_splitter.split_text(pre_text)
                    else:
                        pre_chunks = self.default_splitter.split_text(pre_text)
                    chunks.extend(pre_chunks)
                else:
                    chunks.append(pre_text)
            
            # Add table as separate chunk
            table_text = text[start:end]
            chunks.append(table_text)
            last_end = end
        
        # Add remaining text
        if last_end < len(text):
            remaining_text = text[last_end:]
            if len(remaining_text) > self.chunk_size:
                if self.spacy_splitter and SPACY_AVAILABLE:
                    remaining_chunks = self.spacy_splitter.split_text(remaining_text)
                else:
                    remaining_chunks = self.default_splitter.split_text(remaining_text)
                chunks.extend(remaining_chunks)
            else:
                chunks.append(remaining_text)
        
        # If no tables were found, use standard splitting
        if not table_spans:
            if self.spacy_splitter and SPACY_AVAILABLE:
                chunks = self.spacy_splitter.split_text(text)
            else:
                chunks = self.default_splitter.split_text(text)
        
        return chunks
    
    def create_technical_chunks(self, text: str) -> List[str]:
        """
        Create chunks optimized for technical documentation.
        
        Args:
            text: Document text
            
        Returns:
            List of text chunks
        """
        # Identify code blocks
        code_pattern = r'```[a-z]*\n[\s\S]*?```'
        code_blocks = re.finditer(code_pattern, text)
        code_spans = [(m.start(), m.end()) for m in code_blocks]
        
        # Split around code blocks
        chunks = []
        last_end = 0
        
        for start, end in code_spans:
            # Add text before code block
            if start > last_end:
                pre_text = text[last_end:start]
                if len(pre_text) > self.chunk_size:
                    if self.spacy_splitter and SPACY_AVAILABLE:
                        pre_chunks = self.spacy_splitter.split_text(pre_text)
                    else:
                        pre_chunks = self.default_splitter.split_text(pre_text)
                    chunks.extend(pre_chunks)
                else:
                    chunks.append(pre_text)
            
            # Add code block as separate chunk
            code_text = text[start:end]
            chunks.append(code_text)
            last_end = end
        
        # Add remaining text
        if last_end < len(text):
            remaining_text = text[last_end:]
            if len(remaining_text) > self.chunk_size:
                if self.spacy_splitter and SPACY_AVAILABLE:
                    remaining_chunks = self.spacy_splitter.split_text(remaining_text)
                else:
                    remaining_chunks = self.default_splitter.split_text(remaining_text)
                chunks.extend(remaining_chunks)
            else:
                chunks.append(remaining_text)
        
        # If no code blocks were found, try to split by headers
        if not code_spans:
            if "## " in text or "# " in text:
                # Process with Markdown header splitter first
                splits = self.markdown_splitter.split_text(text)
                chunks = [split.page_content for split in splits]
            elif self.spacy_splitter and SPACY_AVAILABLE:
                chunks = self.spacy_splitter.split_text(text)
            else:
                chunks = self.default_splitter.split_text(text)
        
        return chunks
    
    def create_narrative_chunks(self, text: str) -> List[str]:
        """
        Create chunks optimized for narrative text content.
        
        Args:
            text: Document text
            
        Returns:
            List of text chunks
        """
        # For narrative text, spaCy's sentence boundary detection is ideal
        if self.spacy_splitter and SPACY_AVAILABLE:
            return self.spacy_splitter.split_text(text)
        else:
            # Fallback to paragraph-based splitting
            paragraph_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            return paragraph_splitter.split_text(text)
    
    def create_general_chunks(self, text: str) -> List[str]:
        """
        Create chunks using a general-purpose strategy.
        
        Args:
            text: Document text
            
        Returns:
            List of text chunks
        """
        # Check for markdown structure first
        if "## " in text or "# " in text:
            # Process with Markdown header splitter first
            splits = self.markdown_splitter.split_text(text)
            chunks = [split.page_content for split in splits]
            
            # Further split long chunks
            result = []
            for chunk in chunks:
                if len(chunk) > self.chunk_size:
                    if self.spacy_splitter and SPACY_AVAILABLE:
                        result.extend(self.spacy_splitter.split_text(chunk))
                    else:
                        result.extend(self.default_splitter.split_text(chunk))
                else:
                    result.append(chunk)
            return result
        
        # Try spaCy if available
        if self.spacy_splitter and SPACY_AVAILABLE:
            return self.spacy_splitter.split_text(text)
        
        # Fallback to default splitting
        return self.default_splitter.split_text(text)
    
    def reset_document_type_memory(self):
        """Reset the document type memory to force re-detection"""
        self.document_type_memory = {}
        print("Document type memory has been reset")
        
    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk a document into smaller pieces using intelligent strategies.
        
        Args:
            document: Document to chunk
            
        Returns:
            List of chunks with metadata
        """
        # Prompt for document type to choose the best chunking strategy,
        # but reuse previous selections for similar documents
        doc_type = self.prompt_for_document_type(document)
        
        # Store document type in metadata
        document['metadata']['doc_type'] = doc_type
        
        # Always try spaCy first regardless of document type if available
        chunks = None
        if self.spacy_splitter and SPACY_AVAILABLE:
            try:
                print(f"Attempting to use spaCy for {document['metadata'].get('source', 'unknown document')}")
                chunks = self.spacy_splitter.split_text(document['content'])
                print(f"Successfully used spaCy to create {len(chunks)} chunks")
            except Exception as e:
                print(f"spaCy chunking failed: {e}. Falling back to document-type specific strategy.")
                chunks = None
        
        # If spaCy failed or wasn't available, use document-type specific strategy
        if chunks is None:
            if doc_type == "scientific":
                chunks = self.create_scientific_chunks(document['content'])
            elif doc_type == "financial":
                chunks = self.create_financial_chunks(document['content'])
            elif doc_type == "technical":
                chunks = self.create_technical_chunks(document['content'])
            elif doc_type == "narrative":
                chunks = self.create_narrative_chunks(document['content'])
            else:  # general
                chunks = self.create_general_chunks(document['content'])
        
        # Enforce maximum chunk size for ALL chunks
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > self.chunk_size:
                print(f"Created a chunk of size {len(chunk)}, which is longer than the specified {self.chunk_size}")
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
                            if len(forced_chunk) > self.chunk_size:
                                print(f"Warning: Even after forced splitting, chunk size is {len(forced_chunk)}")
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
            
            result.append({
                'content': chunk,
                'metadata': chunk_metadata
            })
        
        print(f"  Created {len(result)} chunks")
        return result