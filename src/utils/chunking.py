"""
Chunking utilities for the MCP Server.
"""
from typing import Dict, Any, List
from langchain.text_splitter import RecursiveCharacterTextSplitter

class ChunkingEngine:
    """
    Engine for chunking documents using various strategies.
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
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n## ", "\n### ", "\n#### ", "\n", " ", ""]
        )
    
    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk a document into smaller pieces.
        
        Args:
            document: Document to chunk
            
        Returns:
            List of chunks with metadata
        """
        chunks = self.text_splitter.split_text(document['content'])
        
        result = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = document['metadata'].copy()
            chunk_metadata['chunk_id'] = i
            
            result.append({
                'content': chunk,
                'metadata': chunk_metadata
            })
        
        return result