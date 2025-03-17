"""
Embedding utilities for the MCP Server.
"""
import os
import sys
from typing import List, Dict, Any, Optional, Callable
from openai import OpenAI

# Add Hugging Face support
try:
    from sentence_transformers import SentenceTransformer
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

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
            api_key: OpenAI API key (not needed if using custom_embedding_function or huggingface model)
            model: Embedding model to use (not needed if using custom_embedding_function)
            batch_size: Batch size for embedding generation
            custom_embedding_function: Optional custom function to generate embeddings
        """
        self.custom_embedding_function = custom_embedding_function
        self.batch_size = batch_size
        self.model_name = model
        self.model_type = self._get_model_type(model)
        
        # Initialize Hugging Face model if needed
        self.hf_model = None
        if self.model_type == "huggingface":
            if not HUGGINGFACE_AVAILABLE:
                print("ERROR: sentence-transformers package not installed. Run: pip install sentence-transformers")
                sys.exit(1)
            try:
                print(f"Loading Hugging Face model {model}...")
                # For snowflake model, we need to specify the specific variant
                if "snowflake-arctic-embed" in model and not model.endswith("-m"):
                    model = model + "-m"  # Use the medium variant by default
                self.hf_model = SentenceTransformer(model)
                print(f"Successfully loaded {model}")
            except Exception as e:
                print(f"Error loading Hugging Face model: {e}")
                sys.exit(1)
        
        # Only set up OpenAI if we're using an OpenAI model
        elif self.model_type == "openai":
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OpenAI API key is required when using OpenAI embedding models")
            
            self.client = OpenAI(api_key=self.api_key)
            
            # Print information about the selected model
            if model == "text-embedding-3-small":
                print("Using text-embedding-3-small: Optimized for speed and cost-effectiveness with good quality.")
            elif model == "text-embedding-3-large":
                print("Using text-embedding-3-large: Optimized for highest quality embeddings but slower and more expensive.")
            else:
                print(f"Using {model} embedding model.")
    
    def _get_model_type(self, model: str) -> str:
        """
        Determine the type of model being used.
        
        Args:
            model: Model name
            
        Returns:
            Model type ("openai", "huggingface", or "custom")
        """
        if model.startswith("text-embedding-") or model == "ada":
            return "openai"
        elif any(x in model for x in ["sentence-transformers/", "BAAI/", "Snowflake/", "/"]):
            return "huggingface"
        else:
            # If custom embedding function is provided, it's a custom model
            if self.custom_embedding_function:
                return "custom"
            # Otherwise, assume it's OpenAI
            return "openai"
    
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
                    batch[j]['metadata']['embedding_model'] = "custom"
            
            elif self.model_type == "huggingface":
                # Use Hugging Face model
                embeddings = self.hf_model.encode(texts)
                for j, embedding in enumerate(embeddings):
                    batch[j]['embedding'] = embedding.tolist()  # Convert numpy array to list
                    batch[j]['metadata']['embedding_model'] = self.model_name
            
            else:
                # Use OpenAI with the new API format
                response = self.client.embeddings.create(
                    input=texts,
                    model=self.model_name
                )
                
                for j, embedding_data in enumerate(response.data):
                    batch[j]['embedding'] = embedding_data.embedding
                    batch[j]['metadata']['embedding_model'] = self.model_name
        
        return chunks