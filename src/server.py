"""
MCP Server for providing access to processed Markdown files.
"""
import os
import json
import importlib.util
from typing import Dict, Any, List, Optional, Callable
import dotenv
from mcp.server.fastmcp import FastMCP
from openai import OpenAI
import anthropic
import sys

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
        
        # Only require OpenAI API key if we're not using a custom embedding function
        if not self.custom_embedding_module and not self.openai_api_key:
            raise ValueError("OpenAI API key is required for embeddings when not using a custom embedding function")
        
        # Only require Anthropic API key if we're using it
        if self.use_anthropic and not self.anthropic_api_key:
            print("WARNING: Anthropic API key not found. The server will rely on the client for LLM processing.")
            self.use_anthropic = False

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
if config.openai_api_key:
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
        print(f"Failed to load custom embedding function. Falling back to OpenAI.")

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
    elif openai_client:
        # Use OpenAI with the new API format
        response = openai_client.embeddings.create(
            input=text,
            model=config.embedding_model
        )
        return response.data[0].embedding
    else:
        raise ValueError("No embedding method available. Please provide an OpenAI API key or a custom embedding function.")

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
            context += f"Document {i+1} (from {source}, title: {title}):\n{doc}\n\n"
        
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
    Read and process Markdown files.
    
    Args:
        file_path: Optional path to a specific file or directory
        
    Returns:
        Information about the processed files
    """
    query = f"Provide information about the Markdown files"
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
    print(f"  Custom embedding: {'Yes' if custom_embedding_function else 'No'}")
    
    # Set binary mode for stdin/stdout to avoid encoding issues
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stdin, 'reconfigure'):
        sys.stdin.reconfigure(encoding='utf-8')
    
    # Initialize and run the server
    mcp.run(transport=config.transport)