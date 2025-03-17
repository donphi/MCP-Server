"""
MCP Server for providing access to processed documents.
"""
import os
import json
import importlib.util
import sys
from typing import Dict, Any, List, Optional, Callable
import dotenv
from mcp.server.fastmcp import FastMCP
from openai import OpenAI
import anthropic
import numpy as np

# Add Hugging Face support
try:
    from sentence_transformers import SentenceTransformer
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

from utils.vector_db import VectorDatabaseReader

# Load environment variables
dotenv.load_dotenv()

class ServerConfig:
    """
    Configuration for the MCP server.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the server configuration.
        
        Args:
            config_path: Path to the configuration file
        """
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Load configuration from environment variables or config file
        self.db_path = os.environ.get('DB_PATH', config.get('db_path', '/db'))
        self.embedding_model = os.environ.get('EMBEDDING_MODEL', config.get('embedding_model', 'text-embedding-3-small'))
        self.claude_model = os.environ.get('CLAUDE_MODEL', config.get('claude_model', 'claude-3-7-sonnet-20240307'))
        self.max_results = int(os.environ.get('MAX_RESULTS', config.get('max_results', 10)))
        self.transport = os.environ.get('TRANSPORT', config.get('transport', 'stdio'))
        self.use_anthropic = os.environ.get('USE_ANTHROPIC', config.get('use_anthropic', True)) in ['true', 'True', True, 1, '1']
        
        # Custom embedding model
        self.custom_embedding_module = os.environ.get('CUSTOM_EMBEDDING_MODULE', config.get('custom_embedding_module', None))
        self.custom_embedding_function = os.environ.get('CUSTOM_EMBEDDING_FUNCTION', config.get('custom_embedding_function', None))
        
        # API keys
        self.openai_api_key = os.environ.get('OPENAI_API_KEY')
        self.anthropic_api_key = os.environ.get('ANTHROPIC_API_KEY')
        
        # Only require OpenAI API key if we're using an OpenAI embedding model
        if self.embedding_model.startswith("text-embedding-") and not self.openai_api_key:
            print("WARNING: OpenAI API key is required for OpenAI embeddings. Will try to use Hugging Face model instead.")
            self.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        
        # Only require Anthropic API key if we're using it
        if self.use_anthropic and not self.anthropic_api_key:
            print("WARNING: Anthropic API key not found. The server will rely on the client for LLM processing.")
            self.use_anthropic = False
        
        # Default embedding models
        self.embedding_model_options = {
            # OpenAI paid models
            "text-embedding-3-small": "Optimized for speed and cost-effectiveness with good quality (PAID).",
            "text-embedding-3-large": "Optimized for highest quality embeddings but slower and more expensive (PAID).",
            
            # Free Hugging Face models
            "sentence-transformers/all-MiniLM-L6-v2": "A compact model for efficient embeddings and rapid retrieval (FREE).",
            "BAAI/bge-m3": "Versatile model supporting 100+ languages and inputs up to 8192 tokens (FREE).",
            "Snowflake/snowflake-arctic-embed-m": "Optimized for high-quality retrieval balancing accuracy and speed (FREE)."
        }

def load_custom_embedding_function(module_path: str, function_name: str) -> Optional[Callable]:
    """
    Load a custom embedding function from a Python module.
    
    Args:
        module_path: Path to the Python module
        function_name: Name of the function in the module
        
    Returns:
        The custom embedding function, or None if it couldn't be loaded
    """
    try:
        spec = importlib.util.spec_from_file_location("custom_embedding_module", module_path)
        if spec is None or spec.loader is None:
            print(f"Could not load module from {module_path}")
            return None
            
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        if not hasattr(module, function_name):
            print(f"Function {function_name} not found in module {module_path}")
            return None
            
        return getattr(module, function_name)
    except Exception as e:
        print(f"Error loading custom embedding function: {e}")
        return None

# Initialize FastMCP server
mcp = FastMCP("mcp-server")

# Load configuration
config_path = os.environ.get("CONFIG_PATH", "/config/server_config.json")
config = ServerConfig(config_path)

# Initialize vector database reader
vector_db = VectorDatabaseReader(config.db_path)

# Initialize OpenAI client
openai_client = None
if config.openai_api_key and config.embedding_model.startswith("text-embedding-"):
    openai_client = OpenAI(api_key=config.openai_api_key)

# Initialize Anthropic client if API key is available
claude_client = None
if config.use_anthropic and config.anthropic_api_key:
    claude_client = anthropic.Anthropic(api_key=config.anthropic_api_key)

# Load custom embedding function if specified
custom_embedding_function = None
if config.custom_embedding_module and config.custom_embedding_function:
    custom_embedding_function = load_custom_embedding_function(
        config.custom_embedding_module,
        config.custom_embedding_function
    )
    if custom_embedding_function:
        print(f"Using custom embedding function {config.custom_embedding_function} from {config.custom_embedding_module}")
    else:
        print(f"Failed to load custom embedding function. Falling back to OpenAI or Hugging Face.")

# Initialize Hugging Face model if needed
hf_model = None
if config.embedding_model.startswith("sentence-transformers/") or \
   config.embedding_model.startswith("BAAI/") or \
   config.embedding_model.startswith("Snowflake/") or \
   "/" in config.embedding_model:
    if not HUGGINGFACE_AVAILABLE:
        print("ERROR: sentence-transformers package not installed. Run: pip install sentence-transformers")
        sys.exit(1)
    try:
        print(f"Loading Hugging Face model {config.embedding_model}...")
        # For snowflake model, we need to specify the specific variant
        model_name = config.embedding_model
        if "snowflake-arctic-embed" in model_name and not model_name.endswith("-m"):
            model_name = model_name + "-m"  # Use the medium variant by default
            
        hf_model = SentenceTransformer(model_name)
        print(f"Successfully loaded {model_name}")
    except Exception as e:
        print(f"Error loading Hugging Face model: {e}")
        sys.exit(1)

# Get the model that was used for indexing
def get_stored_embedding_model():
    """
    Determine which embedding model was used to create the database.
    
    Returns:
        The embedding model name, or None if it couldn't be determined
    """
    try:
        # Check if the database exists
        if not os.path.exists(os.path.join(config.db_path, "chroma.sqlite3")):
            print("WARNING: Vector database not found. You need to run the pipeline first.")
            return None
            
        # Query a small sample to determine the model
        dummy_embedding = [0.0] * 768  # most models use 768 dimensions
        try:
            results = vector_db.search(
                query_embedding=dummy_embedding,
                n_results=1
            )
        except Exception as e:
            # If the dummy embedding fails (wrong dimensions), try another size
            try:
                dummy_embedding = [0.0] * 1024  # some models use 1024 dimensions
                results = vector_db.search(
                    query_embedding=dummy_embedding,
                    n_results=1
                )
            except Exception as e:
                print(f"WARNING: Could not determine model dimensions: {e}")
                return None
        
        if 'metadatas' in results and results['metadatas'] and results['metadatas'][0]:
            for metadata in results['metadatas'][0]:
                if 'embedding_model' in metadata:
                    return metadata['embedding_model']
    except Exception as e:
        print(f"Error getting stored embedding model: {e}")
    
    return None

# Try to detect the embedding model used in the database
stored_model = get_stored_embedding_model()
if stored_model:
    if stored_model != config.embedding_model:
        print(f"WARNING: Database was created with '{stored_model}' but current setting is '{config.embedding_model}'")
        print(f"Switching to '{stored_model}' for compatibility")
        config.embedding_model = stored_model
        
        # Re-initialize clients if needed
        if config.embedding_model.startswith("text-embedding-") and not openai_client and config.openai_api_key:
            openai_client = OpenAI(api_key=config.openai_api_key)
        elif (config.embedding_model.startswith("sentence-transformers/") or 
             config.embedding_model.startswith("BAAI/") or 
             config.embedding_model.startswith("Snowflake/") or
             "/" in config.embedding_model) and not hf_model:
            if not HUGGINGFACE_AVAILABLE:
                print("ERROR: sentence-transformers package not installed. Run: pip install sentence-transformers")
                sys.exit(1)
            try:
                print(f"Loading Hugging Face model {config.embedding_model}...")
                # For snowflake model, we need to specify the specific variant
                model_name = config.embedding_model
                if "snowflake-arctic-embed" in model_name and not model_name.endswith("-m"):
                    model_name = model_name + "-m"  # Use the medium variant by default
                
                hf_model = SentenceTransformer(model_name)
                print(f"Successfully loaded {model_name}")
            except Exception as e:
                print(f"Error loading Hugging Face model: {e}")
                sys.exit(1)

async def generate_embedding(text: str) -> List[float]:
    """
    Generate an embedding for a text.
    
    Args:
        text: Text to embed
        
    Returns:
        Embedding vector
    """
    if custom_embedding_function:
        # Use custom embedding function
        embeddings = custom_embedding_function([text])
        return embeddings[0]
    elif hf_model and (config.embedding_model.startswith("sentence-transformers/") or 
                      config.embedding_model.startswith("BAAI/") or 
                      config.embedding_model.startswith("Snowflake/") or
                      "/" in config.embedding_model):
        # Use Hugging Face model
        embedding = hf_model.encode([text])[0]
        return embedding.tolist()
    elif openai_client:
        # Use OpenAI with the new API format
        response = openai_client.embeddings.create(
            input=text,
            model=config.embedding_model
        )
        return response.data[0].embedding
    else:
        raise ValueError("No embedding method available. Please provide an OpenAI API key or install sentence-transformers.")

async def process_query(query: str) -> str:
    """
    Process a query using the general query processing loop.
    
    Args:
        query: The user's query
        
    Returns:
        A comprehensive response
    """
    try:
        # Generate embedding for query
        query_embedding = await generate_embedding(query)
        
        # Search for relevant documents
        results = vector_db.search(
            query_embedding=query_embedding,
            n_results=config.max_results
        )
        
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        
        # Format context
        context = ""
        for i, (doc, meta) in enumerate(zip(documents, metadatas)):
            source = meta.get('source', 'Unknown')
            title = meta.get('title', 'Untitled')
            file_type = meta.get('file_type', 'unknown')
            context += f"Document {i+1} (from {source}, title: {title}, type: {file_type}):\n{doc}\n\n"
        
        # If Anthropic client is available, use it to generate a response
        if claude_client:
            response = claude_client.messages.create(
                model=config.claude_model,
                max_tokens=1024,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
                ]
            )
            return response.content[0].text
        else:
            # If no Anthropic client, just return the context for the client to process
            return f"Context for query '{query}':\n\n{context}"
    except Exception as e:
        return f"Error processing query: {str(e)}"

@mcp.tool()
async def read_md_files(file_path: str = None) -> str:
    """
    Read and process files.
    
    Args:
        file_path: Optional path to a specific file or directory
        
    Returns:
        Information about the processed files
    """
    query = f"Provide information about the files"
    if file_path:
        query += f" in {file_path}"
    return await process_query(query)

@mcp.tool()
async def search_content(query: str) -> str:
    """
    Search across processed content.
    
    Args:
        query: The search query
        
    Returns:
        Search results
    """
    return await process_query(f"Search for: {query}")

@mcp.tool()
async def get_context(query: str, window_size: int = 3) -> str:
    """
    Retrieve contextual information.
    
    Args:
        query: The context query
        window_size: Number of context items to retrieve
        
    Returns:
        Contextual information
    """
    return await process_query(f"Provide context about: {query}")

@mcp.tool()
async def project_structure() -> str:
    """
    Provide information about the project structure.
    
    Returns:
        Project structure information
    """
    return await process_query("Describe the structure of the project")

@mcp.tool()
async def suggest_implementation(description: str) -> str:
    """
    Generate implementation suggestions.
    
    Args:
        description: Description of what to implement
        
    Returns:
        Implementation suggestions
    """
    return await process_query(f"Suggest an implementation for: {description}")

if __name__ == "__main__":
    print(f"Starting MCP Server with configuration:")
    print(f"  Transport: {config.transport}")
    print(f"  DB path: {config.db_path}")
    print(f"  Embedding model: {config.embedding_model}")
    print(f"  Max results: {config.max_results}")
    print(f"  Using Anthropic API: {config.use_anthropic}")
    
    # Print embedding model details
    model_type = "Unknown"
    if hf_model:
        model_type = "Hugging Face (FREE)"
    elif openai_client:
        model_type = "OpenAI (PAID)"
    elif custom_embedding_function:
        model_type = "Custom"
    
    print(f"  Embedding method: {model_type}")
    
    if stored_model:
        print(f"  Database created with: {stored_model}")
        if model_type != "Unknown":
            print(f"  Using compatible model: {config.embedding_model}")
    
    # Print supported file types from the database
    try:
        file_types = set()
        results = vector_db.search(
            query_embedding=[0.0] * 768,
            n_results=100
        )
        for metadata_list in results.get('metadatas', []):
            for metadata in metadata_list:
                if 'file_type' in metadata:
                    file_types.add(metadata['file_type'])
        
        if file_types:
            print(f"  Documents in database: {', '.join(file_types)}")
    except Exception:
        # Ignore errors when trying to detect file types
        pass
    
    # Set binary mode for stdin/stdout to avoid encoding issues
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stdin, 'reconfigure'):
        sys.stdin.reconfigure(encoding='utf-8')
    
    # Initialize and run the server
    mcp.run(transport=config.transport)