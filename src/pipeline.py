"""
Main pipeline script for processing Markdown and text files.
"""
import os
import json
import hashlib
import importlib.util
import sys
from typing import List, Dict, Any, Optional, Callable
import dotenv

# Import directly from langchain_community
from langchain_community.document_loaders import UnstructuredMarkdownLoader, TextLoader

from utils.chunking import ChunkingEngine
from utils.embedding import EmbeddingGenerator
from utils.vector_db import VectorDatabaseWriter

# Load environment variables
dotenv.load_dotenv()

class PipelineConfig:
    """
    Configuration for the pipeline.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the pipeline configuration.
        
        Args:
            config_path: Path to the configuration file
        """
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Load configuration from environment variables or config file
        self.data_dir = os.environ.get('DATA_DIR', config.get('data_dir', '/data'))
        self.output_dir = os.environ.get('OUTPUT_DIR', config.get('output_dir', '/output'))
        self.chunk_size = int(os.environ.get('CHUNK_SIZE', config.get('chunk_size', 1000)))
        self.chunk_overlap = int(os.environ.get('CHUNK_OVERLAP', config.get('chunk_overlap', 200)))
        self.openai_api_key = os.environ.get('OPENAI_API_KEY')
        self.embedding_model = os.environ.get('EMBEDDING_MODEL', config.get('embedding_model', 'text-embedding-3-small'))
        self.batch_size = int(os.environ.get('BATCH_SIZE', config.get('batch_size', 10)))
        self.db_path = os.environ.get('DB_PATH', config.get('db_path', '/db'))
        self.priority_files = config.get('priority_files', [])
        self.supported_extensions = os.environ.get('SUPPORTED_EXTENSIONS', config.get('supported_extensions', '.md,.txt')).split(',')
        
        # Custom embedding model
        self.custom_embedding_module = os.environ.get('CUSTOM_EMBEDDING_MODULE', config.get('custom_embedding_module', None))
        self.custom_embedding_function = os.environ.get('CUSTOM_EMBEDDING_FUNCTION', config.get('custom_embedding_function', None))
        
        # Files to ignore
        self.ignore_files = config.get('ignore_files', ['README.md'])
        
        # Embedding model options
        self.embedding_model_options = config.get('embedding_model_options', {
            'text-embedding-3-small': 'Optimized for speed and cost-effectiveness with good quality.',
            'text-embedding-3-large': 'Optimized for highest quality embeddings but slower and more expensive.'
        })
    
    def prompt_for_embedding_model(self):
        """
        Prompt the user to select an embedding model.
        """
        print("\nAvailable embedding models:")
        models = list(self.embedding_model_options.keys())
        for i, model in enumerate(models):
            print(f"  {i+1}. {model}: {self.embedding_model_options[model]}")
        
        while True:
            try:
                choice = input(f"\nSelect embedding model (1-{len(models)}) [default: 1]: ")
                if choice.strip() == "":
                    choice = "1"
                choice = int(choice)
                if 1 <= choice <= len(models):
                    self.embedding_model = models[choice-1]
                    print(f"\nSelected model: {self.embedding_model}")
                    break
                else:
                    print(f"Please enter a number between 1 and {len(models)}")
            except ValueError:
                print("Please enter a valid number")
    
    def validate(self, interactive=True):
        """
        Validate the configuration.
        
        Args:
            interactive: Whether to prompt the user for input if validation fails
        
        Returns:
            True if validation passed, False otherwise
        """
        # Check if OpenAI API key is set and not the default placeholder
        if not self.openai_api_key or self.openai_api_key == "your_openai_api_key_here" or self.openai_api_key.startswith("your_"):
            print("\nERROR: OpenAI API key is not set or is using the default placeholder value.")
            print("Please set a valid OPENAI_API_KEY in your .env file or export it in your shell.")
            
            if interactive:
                api_key = input("\nEnter your OpenAI API key: ")
                if api_key.strip() and not api_key.startswith("your_") and api_key != "your_openai_api_key_here":
                    self.openai_api_key = api_key
                    os.environ['OPENAI_API_KEY'] = api_key
                    print("API key set for this session.")
                else:
                    print("\nInvalid API key. Please update your .env file with a valid OpenAI API key and run again.")
                    return False
            else:
                return False
        
        # Check if data directory exists
        if not os.path.exists(self.data_dir):
            print(f"\nERROR: Data directory '{self.data_dir}' does not exist.")
            return False
        
        # Check if data directory contains any files
        has_files = False
        for root, _, filenames in os.walk(self.data_dir):
            for filename in filenames:
                file_path = os.path.join(root, filename)
                if self._is_supported_file(file_path) and not self._should_ignore(file_path):
                    has_files = True
                    break
            if has_files:
                break
        
        if not has_files:
            print(f"\nERROR: No supported files found in data directory '{self.data_dir}'.")
            print(f"Supported extensions: {self.supported_extensions}")
            print(f"Ignored files: {self.ignore_files}")
            return False
        
        return True
    
    def _is_supported_file(self, file_path: str) -> bool:
        """
        Check if a file is supported.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if the file is supported, False otherwise
        """
        _, ext = os.path.splitext(file_path)
        return ext in self.supported_extensions
    
    def _should_ignore(self, file_path: str) -> bool:
        """
        Check if a file should be ignored.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if the file should be ignored, False otherwise
        """
        filename = os.path.basename(file_path)
        return filename in self.ignore_files

class FileReader:
    """
    Reader for Markdown and text files.
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the file reader.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.processed_files = set()
        self._load_processed_files()
    
    def _load_processed_files(self):
        """
        Load the list of processed files.
        """
        processed_file = os.path.join(self.config.output_dir, 'processed_files.json')
        if os.path.exists(processed_file):
            with open(processed_file, 'r') as f:
                self.processed_files = set(json.load(f))
    
    def _save_processed_files(self):
        """
        Save the list of processed files.
        """
        processed_file = os.path.join(self.config.output_dir, 'processed_files.json')
        with open(processed_file, 'w') as f:
            json.dump(list(self.processed_files), f)
    
    def get_files(self) -> List[Dict[str, Any]]:
        """
        Get the list of files to process.
        
        Returns:
            List of files to process
        """
        files = []
        
        # First, process priority files
        for priority_file in self.config.priority_files:
            file_path = os.path.join(self.config.data_dir, priority_file)
            if os.path.exists(file_path) and self._is_supported_file(file_path) and not self._should_ignore(file_path):
                file_hash = self._get_file_hash(file_path)
                if file_hash not in self.processed_files:
                    files.append({
                        'path': file_path,
                        'hash': file_hash,
                        'relative_path': priority_file,
                        'priority': True
                    })
        
        # Then process other files
        for root, _, filenames in os.walk(self.config.data_dir):
            for filename in filenames:
                file_path = os.path.join(root, filename)
                if self._is_supported_file(file_path) and not self._should_ignore(file_path):
                    relative_path = os.path.relpath(file_path, self.config.data_dir)
                    
                    # Skip if already in priority files
                    if relative_path in self.config.priority_files:
                        continue
                    
                    file_hash = self._get_file_hash(file_path)
                    if file_hash not in self.processed_files:
                        files.append({
                            'path': file_path,
                            'hash': file_hash,
                            'relative_path': relative_path,
                            'priority': False
                        })
        
        return files
    
    def _is_supported_file(self, file_path: str) -> bool:
        """
        Check if a file is supported.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if the file is supported, False otherwise
        """
        _, ext = os.path.splitext(file_path)
        return ext in self.config.supported_extensions
    
    def _should_ignore(self, file_path: str) -> bool:
        """
        Check if a file should be ignored.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if the file should be ignored, False otherwise
        """
        filename = os.path.basename(file_path)
        return filename in self.config.ignore_files
    
    def _get_file_hash(self, file_path: str) -> str:
        """
        Get the hash of a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Hash of the file
        """
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def mark_processed(self, file_hash: str):
        """
        Mark a file as processed.
        
        Args:
            file_hash: Hash of the file
        """
        self.processed_files.add(file_hash)
        self._save_processed_files()

class TextProcessor:
    """
    Processor for text documents.
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the text processor.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
    
    def process_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Processed document
        """
        _, ext = os.path.splitext(file_path)
        
        if ext == '.md':
            loader = UnstructuredMarkdownLoader(file_path)
        else:  # Default to text loader for other supported extensions
            loader = TextLoader(file_path)
            
        document = loader.load()[0]
        
        # Extract metadata
        metadata = {
            'source': file_path,
            'title': self._extract_title(document.page_content, file_path),
            'file_type': ext[1:]  # Remove the dot
        }
        
        return {
            'content': document.page_content,
            'metadata': metadata
        }
    
    def _extract_title(self, content: str, file_path: str) -> str:
        """
        Extract the title from a document.
        
        Args:
            content: Document content
            file_path: Path to the file
            
        Returns:
            Document title
        """
        # Try to extract title from content
        lines = content.split('\n')
        for line in lines:
            if line.startswith('# '):
                return line[2:].strip()
        
        # If no title found, use the filename
        return os.path.basename(file_path)

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

class Pipeline:
    """
    Main pipeline for processing Markdown and text files.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the pipeline.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config = PipelineConfig(config_path)
        
        # Prompt for embedding model
        self.config.prompt_for_embedding_model()
        
        # Validate configuration before proceeding
        if not self.config.validate(interactive=True):
            print("\nValidation failed. Exiting.")
            sys.exit(1)
        
        self.file_reader = FileReader(self.config)
        self.text_processor = TextProcessor(self.config)
        self.chunking_engine = ChunkingEngine(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        
        # Load custom embedding function if specified
        custom_embedding_function = None
        if self.config.custom_embedding_module and self.config.custom_embedding_function:
            custom_embedding_function = load_custom_embedding_function(
                self.config.custom_embedding_module,
                self.config.custom_embedding_function
            )
            if custom_embedding_function:
                print(f"Using custom embedding function {self.config.custom_embedding_function} from {self.config.custom_embedding_module}")
            else:
                print(f"Failed to load custom embedding function. Falling back to OpenAI.")
        
        self.embedding_generator = EmbeddingGenerator(
            api_key=self.config.openai_api_key,
            model=self.config.embedding_model,
            batch_size=self.config.batch_size,
            custom_embedding_function=custom_embedding_function
        )
        
        self.db_writer = VectorDatabaseWriter(
            db_path=self.config.db_path
        )
    
    def run(self):
        """
        Run the pipeline.
        """
        print(f"\nStarting pipeline with configuration:")
        print(f"  Data directory: {self.config.data_dir}")
        print(f"  Output directory: {self.config.output_dir}")
        print(f"  Chunk size: {self.config.chunk_size}")
        print(f"  Chunk overlap: {self.config.chunk_overlap}")
        print(f"  Embedding model: {self.config.embedding_model}")
        
        # Display embedding model options
        print(f"  Available embedding models:")
        for model, description in self.config.embedding_model_options.items():
            if model == self.config.embedding_model:
                print(f"    * {model} (SELECTED): {description}")
            else:
                print(f"    * {model}: {description}")
        
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  DB path: {self.config.db_path}")
        print(f"  Supported file extensions: {self.config.supported_extensions}")
        print(f"  Custom embedding: {'Yes' if self.config.custom_embedding_module else 'No'}")
        print(f"  Ignored files: {self.config.ignore_files}")
        
        files = self.file_reader.get_files()
        
        print(f"Found {len(files)} files to process")
        
        for file in files:
            try:
                print(f"Processing {file['relative_path']}...")
                
                # Process file
                document = self.text_processor.process_file(file['path'])
                
                # Chunk document
                chunks = self.chunking_engine.chunk_document(document)
                print(f"  Created {len(chunks)} chunks")
                
                # Generate embeddings
                chunks_with_embeddings = self.embedding_generator.generate_embeddings(chunks)
                
                # Write to vector database
                self.db_writer.write_chunks(chunks_with_embeddings)
                
                # Mark file as processed
                self.file_reader.mark_processed(file['hash'])
                
                print(f"Successfully processed {file['relative_path']}")
            except Exception as e:
                print(f"Error processing {file['relative_path']}: {e}")
        
        print("Pipeline completed successfully")

if __name__ == "__main__":
    config_path = os.environ.get("CONFIG_PATH", "/config/pipeline_config.json")
    pipeline = Pipeline(config_path)
    pipeline.run()