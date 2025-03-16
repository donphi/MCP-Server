"""
Embedding utilities for the MCP Server.
"""
import os
from typing import List, Dict, Any, Optional, Callable
from openai import OpenAI

class EmbeddingGenerator:
    """
    Generator for creating embeddings from text.
    """
    
    def __init__(self, 
                 api_key: str = None, 
                 model: str = "text-embedding-3-small", 
                 batch_size: int = 10,
                 custom_embedding_function: Optional[Callable[[List[str]], List[List[float]]]] = None):
        """
        Initialize the embedding generator.
        
        Args:
            api_key: OpenAI API key (not needed if using custom_embedding_function)
            model: Embedding model to use (not needed if using custom_embedding_function)
            batch_size: Batch size for embedding generation
            custom_embedding_function: Optional custom function to generate embeddings
        """
        self.custom_embedding_function = custom_embedding_function
        self.batch_size = batch_size
        
        # Only set up OpenAI if we're not using a custom embedding function
        if custom_embedding_function is None:
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OpenAI API key is required when not using a custom embedding function")
            
            self.client = OpenAI(api_key=self.api_key)
            self.model = model
            
            # Print information about the selected model
            if model == "text-embedding-3-small":
                print("Using text-embedding-3-small: Optimized for speed and cost-effectiveness with good quality.")
            elif model == "text-embedding-3-large":
                print("Using text-embedding-3-large: Optimized for highest quality embeddings but slower and more expensive.")
            else:
                print(f"Using {model} embedding model.")
    
    def generate_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for a list of chunks.
        
        Args:
            chunks: List of chunks to embed
            
        Returns:
            List of chunks with embeddings
        """
        batch_size = self.batch_size
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            texts = [chunk['content'] for chunk in batch]
            
            if self.custom_embedding_function:
                # Use custom embedding function
                embeddings = self.custom_embedding_function(texts)
                for j, embedding in enumerate(embeddings):
                    batch[j]['embedding'] = embedding
            else:
                # Use OpenAI with the new API format
                response = self.client.embeddings.create(
                    input=texts,
                    model=self.model
                )
                
                for j, embedding_data in enumerate(response.data):
                    batch[j]['embedding'] = embedding_data.embedding
        
        return chunks