"""
Vector database utilities for the MCP Server.
"""
from typing import List, Dict, Any
import chromadb

class VectorDatabaseWriter:
    """
    Writer for storing embeddings in a vector database.
    """
    
    def __init__(self, db_path: str, collection_name: str = "documents"):
        """
        Initialize the vector database writer.
        
        Args:
            db_path: Path to the vector database
            collection_name: Name of the collection to use
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(collection_name)
    
    def write_chunks(self, chunks: List[Dict[str, Any]]):
        """
        Write chunks to the vector database.
        
        Args:
            chunks: List of chunks to write
        """
        ids = []
        embeddings = []
        metadatas = []
        documents = []
        
        for chunk in chunks:
            chunk_id = f"{chunk['metadata']['source']}_{chunk['metadata']['chunk_id']}"
            ids.append(chunk_id)
            embeddings.append(chunk['embedding'])
            metadatas.append(chunk['metadata'])
            documents.append(chunk['content'])
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )

class VectorDatabaseReader:
    """
    Reader for retrieving embeddings from a vector database.
    """
    
    def __init__(self, db_path: str, collection_name: str = "documents"):
        """
        Initialize the vector database reader.
        
        Args:
            db_path: Path to the vector database
            collection_name: Name of the collection to use
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(collection_name)
    
    def search(self, query_embedding: List[float], n_results: int = 10):
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding
            n_results: Number of results to return
            
        Returns:
            Search results
        """
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )