"""
Main pipeline script for processing Markdown,text, PDF and Word files.
"""
import os
import json
import hashlib
import importlib.util
import sys
import subprocess
import importlib
from typing import List, Dict, Any, Optional, Callable
import dotenv

# Import directly from langchain_community
from langchain_community.document_loaders import (
    UnstructuredMarkdownLoader, 
    TextLoader, 
    PyPDFLoader, 
    Docx2txtLoader,
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader
)

# Add Hugging Face support
try:
    from sentence_transformers import SentenceTransformer
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

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
        
        # Load configuration with clear priority: environment variables override config file
        self.data_dir = os.environ.get('DATA_DIR', config.get('data_dir', '/data'))
        self.output_dir = os.environ.get('OUTPUT_DIR', config.get('output_dir', '/output'))
        self.chunk_size = int(os.environ.get('CHUNK_SIZE', config.get('chunk_size', 1000)))
        self.chunk_overlap = int(os.environ.get('CHUNK_OVERLAP', config.get('chunk_overlap', 200)))
        self.openai_api_key = os.environ.get('OPENAI_API_KEY')
        
        # Print chunk size for debugging
        print(f"Using chunk size: {self.chunk_size} (from {'environment' if 'CHUNK_SIZE' in os.environ else 'config file'})")
        print(f"Using chunk overlap: {self.chunk_overlap} (from {'environment' if 'CHUNK_OVERLAP' in os.environ else 'config file'})")
        
        # Make embedding model selection explicit
        env_embedding_model = os.environ.get('EMBEDDING_MODEL')
        config_embedding_model = config.get('embedding_model', 'text-embedding-3-small')
        
        if env_embedding_model:
            self.embedding_model = env_embedding_model
            print(f"Using embedding model from environment: {self.embedding_model}")
        else:
            self.embedding_model = config_embedding_model
            print(f"Using embedding model from config file: {self.embedding_model}")
            
        self.batch_size = int(os.environ.get('BATCH_SIZE', config.get('batch_size', 10)))
        self.db_path = os.environ.get('DB_PATH', config.get('db_path', '/db'))
        self.priority_files = config.get('priority_files', [])
        
        # Set the default extensions (always include PDF and Word documents)
        default_extensions = ['.md', '.txt', '.pdf', '.docx', '.doc']
        
        # Get extensions from config file
        config_extensions = config.get('supported_extensions', default_extensions)
        
        # Check if env variable is set
        env_extensions = os.environ.get('SUPPORTED_EXTENSIONS')
        
        if env_extensions:
            # Environment variable overrides config
            # Convert comma-separated string to list
            self.supported_extensions = [ext.strip() for ext in env_extensions.split(',')]
            # Ensure each extension has a leading dot
            self.supported_extensions = [ext if ext.startswith('.') else f'.{ext}' for ext in self.supported_extensions]
        else:
            # Use config value
            if isinstance(config_extensions, list):
                self.supported_extensions = config_extensions
            else:
                # Handle comma-separated string
                self.supported_extensions = [ext.strip() for ext in config_extensions.split(',')]
            
            # Ensure each extension has a leading dot
            self.supported_extensions = [ext if ext.startswith('.') else f'.{ext}' for ext in self.supported_extensions]
        
        # Print debug info about extensions
        print(f"DEBUG: Using file extensions: {self.supported_extensions}")
        print(f"DEBUG: Config extensions: {config_extensions}")
        print(f"DEBUG: Env extensions: {env_extensions}")
        
        # Custom embedding model
        self.custom_embedding_module = os.environ.get('CUSTOM_EMBEDDING_MODULE', config.get('custom_embedding_module', None))
        self.custom_embedding_function = os.environ.get('CUSTOM_EMBEDDING_FUNCTION', config.get('custom_embedding_function', None))
        
        # Files to ignore
        self.ignore_files = config.get('ignore_files', ['README.md'])
        
        # Embedding model options
        self.embedding_model_options = {
            # OpenAI paid models
            'text-embedding-3-small': 'Optimized for speed and cost-effectiveness with good quality (PAID - requires OpenAI API key).',
            'text-embedding-3-large': 'Highest quality embeddings but slower and more expensive (PAID - requires OpenAI API key).',
            
            # Free Hugging Face models
            'sentence-transformers/all-MiniLM-L6-v2': 'A compact model for efficient embeddings and rapid retrieval (FREE - no API key required).',
            'BAAI/bge-m3': 'Versatile model supporting 100+ languages and inputs up to 8192 tokens (FREE - no API key required).',
            'Snowflake/snowflake-arctic-embed-m': 'Optimized for high-quality retrieval balancing accuracy and speed (FREE - no API key required).'
        }
    
    def install_sentence_transformers(self):
        """
        Install the sentence-transformers package and properly import it.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print("\nInstalling sentence-transformers package...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "sentence-transformers"])
            print("Successfully installed sentence-transformers.")
            
            # Force reload of the module
            if 'sentence_transformers' in sys.modules:
                del sys.modules['sentence_transformers']
            
            # Try to import again
            try:
                import sentence_transformers
                from sentence_transformers import SentenceTransformer
                globals()['HUGGINGFACE_AVAILABLE'] = True
                self.sentence_transformers_module = sentence_transformers
                return True
            except ImportError as e:
                print(f"ERROR: Still couldn't import sentence_transformers after installation: {e}")
                return False
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Failed to install sentence-transformers: {e}")
            return False
    
    def prompt_for_embedding_model(self):
        """
        Prompt the user to select an embedding model.
        """
        # If OpenAI API key is not set, only allow Hugging Face models
        openai_available = self.openai_api_key and not self.openai_api_key.startswith("your_")
        
        print("\nAvailable embedding models:")
        
        # Split into paid and free sections
        print("\nFREE Models (no API key required):")
        free_models = [
            'sentence-transformers/all-MiniLM-L6-v2',
            'BAAI/bge-m3',
            'Snowflake/snowflake-arctic-embed-m'
        ]
        
        # If sentence-transformers is not installed, we need to inform the user
        if not HUGGINGFACE_AVAILABLE:
            print("  NOTE: sentence-transformers package not installed.")
            print("  Installation will be offered if you select a free model.\n")
        
        for i, model in enumerate(free_models):
            print(f"  {i+1}. {model}")
            print(f"     {self.embedding_model_options.get(model, 'No description available')}")
        
        # Only show paid models if the OpenAI API key is available
        paid_models = []
        if openai_available:
            print("\nPAID Models (require OpenAI API key):")
            paid_models = [
                'text-embedding-3-small',
                'text-embedding-3-large'
            ]
            for i, model in enumerate(paid_models):
                print(f"  {len(free_models) + i + 1}. {model}")
                print(f"     {self.embedding_model_options.get(model, 'No description available')}")
        
        # Combine all available models
        all_models = free_models + paid_models
        
        # Set default model - prefer free models
        default_model = 'sentence-transformers/all-MiniLM-L6-v2' if HUGGINGFACE_AVAILABLE else \
                       'text-embedding-3-small' if openai_available else 'sentence-transformers/all-MiniLM-L6-v2'
        
        # Find the index of the default model
        try:
            default_index = all_models.index(default_model) + 1
        except ValueError:
            default_index = 1 if all_models else 0
        
        if all_models:
            while True:
                try:
                    choice = input(f"\nSelect embedding model (1-{len(all_models)}) [default: {default_index}]: ")
                    if choice.strip() == "":
                        choice = str(default_index)
                    choice = int(choice)
                    if 1 <= choice <= len(all_models):
                        selected_model = all_models[choice-1]
                        self.embedding_model = selected_model
                        print(f"\nSelected model: {self.embedding_model}")
                        
                        # Handle Hugging Face model selection
                        if (selected_model.startswith('sentence-transformers/') or
                            selected_model.startswith('BAAI/') or
                            selected_model.startswith('Snowflake/') or 
                            selected_model.find('/') != -1):
                            
                            # Ensure sentence-transformers is installed
                            if not HUGGINGFACE_AVAILABLE:
                                install = input("This model requires the sentence-transformers package. Install it now? (y/n): ")
                                if install.lower().startswith('y'):
                                    if self.install_sentence_transformers():
                                        print("Successfully installed and imported sentence-transformers.")
                                        globals()['HUGGINGFACE_AVAILABLE'] = True
                                        break
                                    else:
                                        print("ERROR: Failed to install or import sentence-transformers.")
                                        print("Please select a different model or install the package manually.")
                                        continue
                                else:
                                    print("WARNING: You selected a model that requires sentence-transformers, but chose not to install it.")
                                    print("Please select a different model or install the package manually.")
                                    continue
                            break
                        else:
                            break
                    else:
                        print(f"Please enter a number between 1 and {len(all_models)}")
                except ValueError:
                    print("Please enter a valid number")
        else:
            print("ERROR: No embedding models available. Please check your configuration.")
            sys.exit(1)
    
    def validate(self, interactive=True):
        """
        Validate the configuration.
        
        Args:
            interactive: Whether to prompt the user for input if validation fails
        
        Returns:
            True if validation passed, False otherwise
        """
        # For OpenAI models, check if API key is set
        if self.embedding_model.startswith('text-embedding-'):
            if not self.openai_api_key or self.openai_api_key == "your_openai_api_key_here" or self.openai_api_key.startswith("your_"):
                print("\nWARNING: OpenAI API key is not set or is using the default placeholder value.")
                print("You can either:")
                print("  1. Set a valid OPENAI_API_KEY in your .env file")
                print("  2. Choose a free Hugging Face model instead")
                
                if interactive:
                    print("\nOptions:")
                    print("  1. Enter an OpenAI API key")
                    print("  2. Switch to a free Hugging Face model")
                    
                    while True:
                        try:
                            choice = input("\nSelect an option (1-2): ")
                            if choice.strip() == "1":
                                api_key = input("\nEnter your OpenAI API key: ")
                                if api_key.strip() and not api_key.startswith("your_") and api_key != "your_openai_api_key_here":
                                    self.openai_api_key = api_key
                                    os.environ['OPENAI_API_KEY'] = api_key
                                    print("API key set for this session.")
                                    break
                                else:
                                    print("\nInvalid API key. Please try again.")
                            elif choice.strip() == "2":
                                if HUGGINGFACE_AVAILABLE:
                                    self.embedding_model = 'sentence-transformers/all-MiniLM-L6-v2'
                                    print(f"\nSwitched to free model: {self.embedding_model}")
                                    break
                                else:
                                    print("\nError: sentence-transformers package not installed.")
                                    print("Installing sentence-transformers package...")
                                    if self.install_sentence_transformers():
                                        self.embedding_model = 'sentence-transformers/all-MiniLM-L6-v2'
                                        print(f"\nSwitched to free model: {self.embedding_model}")
                                        break
                                    else:
                                        print("Cannot proceed without either an OpenAI API key or the sentence-transformers package.")
                                        return False
                            else:
                                print("Please enter either 1 or 2")
                        except ValueError:
                            print("Please enter a valid number")
                else:
                    # In non-interactive mode, switch to Hugging Face if available
                    if HUGGINGFACE_AVAILABLE:
                        self.embedding_model = 'sentence-transformers/all-MiniLM-L6-v2'
                        print(f"\nAutomatically switched to free model: {self.embedding_model}")
                    else:
                        print("\nError: No valid embedding option available. Please set a valid OpenAI API key or install sentence-transformers.")
                        return False
        
        # For Hugging Face models, check if the package is installed
        if (self.embedding_model.startswith('sentence-transformers/') or 
            self.embedding_model.startswith('BAAI/') or
            self.embedding_model.startswith('Snowflake/') or
            '/' in self.embedding_model) and not HUGGINGFACE_AVAILABLE:
            print("\nWARNING: sentence-transformers package not installed but required for the selected model.")
            if interactive:
                install = input("Would you like to install it now? (y/n): ")
                if install.lower().startswith('y'):
                    return self.install_sentence_transformers()
                else:
                    print("Cannot proceed with the selected model without the sentence-transformers package.")
                    return False
            else:
                print("Install it with: pip install sentence-transformers")
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
        # Ensure case-insensitive matching and handle extensions with or without leading dot
        ext_lower = ext.lower()
        for supported_ext in self.supported_extensions:
            if ext_lower == supported_ext.lower() or ext_lower == ('.' + supported_ext.lower().lstrip('.')):
                return True
        return False
    
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
    
    def debug_file_detection(self):
        """
        Debug file detection issues by listing all files and checking if they're supported.
        """
        print("\nDEBUG: Checking file detection in directory:", self.config.data_dir)
        for root, dirs, filenames in os.walk(self.config.data_dir):
            print(f"\nChecking {len(filenames)} files in {root}:")
            for filename in filenames:
                file_path = os.path.join(root, filename)
                _, ext = os.path.splitext(file_path)
                should_ignore = self._should_ignore(file_path)
                is_supported = self._is_supported_file(file_path)
                print(f"  - {filename} (ext: {ext}): {'✓' if is_supported else '✗'} supported, {'✗' if should_ignore else '✓'} included")
                if not is_supported:
                    print(f"     Checking against extensions: {self.config.supported_extensions}")
    
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
        # Ensure case-insensitive matching and handle extensions with or without leading dot
        ext_lower = ext.lower()
        for supported_ext in self.config.supported_extensions:
            if ext_lower == supported_ext.lower() or ext_lower == ('.' + supported_ext.lower().lstrip('.')):
                return True
        return False
    
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
        
        # Setup document loaders for different file types
        self.loader_map = {
            '.md': lambda path: UnstructuredMarkdownLoader(path),
            '.txt': lambda path: TextLoader(path),
            '.pdf': lambda path: self._get_pdf_loader(path),
            '.docx': lambda path: self._get_docx_loader(path),
            '.doc': lambda path: self._get_doc_loader(path),
        }
        
        # Print supported document types
        print("Initialized document loaders for file types:", list(self.loader_map.keys()))
    
    def _get_pdf_loader(self, file_path: str):
        """
        Get the appropriate PDF loader based on the file and available libraries.
        
        Args:
            file_path: Path to the PDF file
        
        Returns:
            An appropriate PDF loader
        """
        print(f"Processing PDF file: {file_path}")
        
        # Check if file exists and is readable
        try:
            with open(file_path, 'rb') as f:
                header = f.read(5)
                if header != b'%PDF-':
                    print(f"WARNING: File {file_path} does not appear to be a valid PDF (header: {header})")
        except Exception as e:
            print(f"WARNING: Could not check PDF header for {file_path}: {e}")
        
        # Always try PyPDFLoader first - it's better at page-based chunking
        try:
            print(f"Attempting to use PyPDFLoader for {file_path}")
            loader = PyPDFLoader(file_path)
            
            # Test the loader by attempting to load page 0
            try:
                test_page = loader.load_page(page_num=0)
                print(f"Successfully loaded first page with PyPDFLoader: {len(test_page.page_content)} chars")
                
                # Count number of pages to verify we're actually handling a multi-page document
                try:
                    import pypdf
                    with open(file_path, 'rb') as f:
                        pdf = pypdf.PdfReader(f)
                        page_count = len(pdf.pages)
                    print(f"PDF has {page_count} pages - each page will be a separate document")
                except Exception as page_count_err:
                    print(f"Could not determine page count: {page_count_err}")
                
                return loader
            except Exception as e:
                print(f"PyPDFLoader could load the file but failed to load page 0: {e}")
                raise
        except Exception as e1:
            print(f"PyPDFLoader failed: {e1}")
            
            # Try UnstructuredPDFLoader with elements mode
            try:
                print(f"Attempting to use UnstructuredPDFLoader (mode=elements) for {file_path}")
                loader = UnstructuredPDFLoader(file_path, mode="elements")
                # Test if it can actually load
                test_docs = loader.load()
                print(f"UnstructuredPDFLoader (elements) successfully loaded {len(test_docs)} documents")
                return loader
            except Exception as e2:
                print(f"UnstructuredPDFLoader (elements) failed: {e2}")
                
                # Last resort - use UnstructuredPDFLoader in "single" mode
                print(f"Attempting to use UnstructuredPDFLoader (mode=single) for {file_path}")
                try:
                    return UnstructuredPDFLoader(file_path, mode="single")
                except Exception as e3:
                    print(f"All PDF loaders failed for {file_path}. Using TextLoader as absolute last resort")
                    print(f"1. PyPDFLoader: {e1}")
                    print(f"2. UnstructuredPDFLoader (elements): {e2}")
                    print(f"3. UnstructuredPDFLoader (single): {e3}")
                    # As absolute last resort, try TextLoader
                    return TextLoader(file_path)
    
    def _get_docx_loader(self, file_path: str):
        """
        Get the appropriate DOCX loader based on the file and available libraries.
        
        Args:
            file_path: Path to the DOCX file
        
        Returns:
            An appropriate DOCX loader
        """
        # Check if file exists and has correct extension
        try:
            with open(file_path, 'rb') as f:
                header = f.read(4)
                # Check for DOCX signature (PK zip header)
                if header != b'PK\x03\x04':
                    print(f"WARNING: File {file_path} may not be a valid DOCX (header: {header})")
        except Exception as e:
            print(f"WARNING: Could not check DOCX header for {file_path}: {e}")
        
        # Check if docx2txt is available
        docx2txt_available = False
        try:
            import docx2txt
            docx2txt_available = True
        except ImportError:
            print("INFO: docx2txt not available, will use UnstructuredWordDocumentLoader instead")
            
        # Choose the appropriate loader based on available packages
        if docx2txt_available:
            # Try Docx2txtLoader first if docx2txt is available
            try:
                print(f"Attempting to use Docx2txtLoader for {file_path}")
                return Docx2txtLoader(file_path)
            except Exception as e1:
                print(f"Docx2txtLoader failed: {e1}")
                
                # Fall back to UnstructuredWordDocumentLoader
                try:
                    print(f"Attempting to use UnstructuredWordDocumentLoader for {file_path}")
                    return UnstructuredWordDocumentLoader(file_path)
                except Exception as e2:
                    print(f"UnstructuredWordDocumentLoader failed: {e2}")
                    print(f"All DOCX loaders failed for {file_path}:")
                    print(f"1. Docx2txtLoader: {e1}")
                    print(f"2. UnstructuredWordDocumentLoader: {e2}")
                    # Return the first loader anyway to let process_file handle the exception
                    return Docx2txtLoader(file_path)
        else:
            # Use UnstructuredWordDocumentLoader directly if docx2txt is not available
            try:
                print(f"Attempting to use UnstructuredWordDocumentLoader for {file_path}")
                return UnstructuredWordDocumentLoader(file_path)
            except Exception as e:
                print(f"UnstructuredWordDocumentLoader failed: {e}")
                # Return it anyway and let process_file handle the exception
                return UnstructuredWordDocumentLoader(file_path)
    
    def _get_doc_loader(self, file_path: str):
        """
        Get the appropriate DOC loader based on the file and available libraries.
        
        Args:
            file_path: Path to the DOC file
        
        Returns:
            An appropriate DOC loader
        """
        # DOC files are older format and can be trickier
        print(f"Attempting to load legacy DOC file: {file_path}")
        
        # Check if LibreOffice is available for conversion
        lo_available = False
        try:
            import subprocess
            result = subprocess.run(['which', 'libreoffice'], capture_output=True, text=True)
            if result.returncode == 0:
                lo_available = True
                print(f"LibreOffice is available at: {result.stdout.strip()}")
            else:
                print("LibreOffice not found - DOC file handling may be limited")
        except Exception as e:
            print(f"Error checking for LibreOffice: {e}")
        
        # Try UnstructuredWordDocumentLoader
        try:
            print(f"Attempting to use UnstructuredWordDocumentLoader for {file_path}")
            return UnstructuredWordDocumentLoader(file_path)
        except Exception as e:
            print(f"UnstructuredWordDocumentLoader failed for DOC file: {e}")
            
            # If LibreOffice is available, we could try to convert DOC to DOCX
            # But for now we'll just return the loader and let process_file handle the exception
            return UnstructuredWordDocumentLoader(file_path)
    
    def process_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Processed document
        """
        _, ext = os.path.splitext(file_path)
        ext_lower = ext.lower()
        filename = os.path.basename(file_path)
        
        print(f"Processing {filename} (type: {ext_lower})...")
        
        # Get the appropriate loader for this file type
        loader_factory = self.loader_map.get(ext_lower)
        
        if loader_factory:
            loader = loader_factory(file_path)
        else:  # Fall back to text loader for unsupported extensions
            loader = TextLoader(file_path)
            
        try:
            print(f"Attempting to load file {file_path} with {loader.__class__.__name__}")
            
            # Special handling for PDF files
            if ext_lower == '.pdf':
                print("PDF detected - using page-by-page processing")
                try:
                    # First, try to count the pages
                    try:
                        import pypdf
                        with open(file_path, 'rb') as f:
                            pdf = pypdf.PdfReader(f)
                            page_count = len(pdf.pages)
                        print(f"PDF has {page_count} pages")
                    except Exception as e:
                        print(f"Could not determine PDF page count: {e}")
                        page_count = "unknown"
                    
                    # Try to load all pages
                    if isinstance(loader, PyPDFLoader):
                        all_pages = loader.load_and_split()
                        print(f"Successfully loaded {len(all_pages)} pages with PyPDFLoader")
                    else:
                        # Fall back to regular loading for other PDF loaders
                        all_docs = loader.load()
                        all_pages = all_docs
                        print(f"Using alternative PDF loader, got {len(all_pages)} documents")
                    
                    if not all_pages:
                        raise ValueError("PDF loader returned no pages")
                    
                    # Combine all pages with clear page markers
                    combined_content = ""
                    for i, page in enumerate(all_pages):
                        combined_content += f"\n--- PAGE {i+1} ---\n"
                        combined_content += page.page_content
                        combined_content += "\n\n"
                    
                    # Extract metadata from first page
                    metadata = {
                        'source': file_path,
                        'title': self._extract_title(all_pages[0].page_content, file_path),
                        'file_type': ext[1:],
                        'page_count': len(all_pages),
                        'embedding_model': self.config.embedding_model
                    }
                    
                    print(f"Successfully loaded {filename}, content length: {len(combined_content)} chars")
                    
                    document = {
                        'content': combined_content,
                        'metadata': metadata
                    }
                    
                    print("\nDocument Type Detection")
                    print(f"File: {filename} (Type: {ext[1:]})")
                    
                    return document
                    
                except Exception as pdf_err:
                    print(f"Error in PDF page-by-page processing: {pdf_err}. Falling back to standard loading.")
            
            # Special handling for Word documents
            if ext_lower in ['.docx', '.doc']:
                print(f"Word document detected ({ext_lower}) - using section-based processing")
                try:
                    # Get all document segments
                    documents = loader.load()
                    
                    if not documents or len(documents) == 0:
                        raise ValueError(f"Word document loader returned no content for {file_path}")
                    
                    # For Word documents, we need to identify logical sections
                    # First combine all content
                    full_content = ""
                    for doc in documents:
                        full_content += doc.page_content + "\n"
                    
                    # Try to detect sections by headers
                    sections = []
                    current_section = ""
                    current_title = "Document Start"
                    
                    # Simple header detection for Word docs
                    lines = full_content.split('\n')
                    for line in lines:
                        # Look for potential headers (short lines, possibly with numbering)
                        stripped = line.strip()
                        if stripped and len(stripped) < 100:  # Potential header
                            if (stripped.endswith(':') or                # Ends with colon
                                (len(stripped.split()) <= 10 and         # Short phrase
                                 any(c.isupper() for c in stripped[0:2]) # Starts with uppercase
                                )):
                                # This might be a header - start a new section
                                if current_section:
                                    sections.append((current_title, current_section))
                                current_title = stripped
                                current_section = line + "\n"
                                continue
                        
                        # Add to current section
                        current_section += line + "\n"
                    
                    # Add the last section
                    if current_section:
                        sections.append((current_title, current_section))
                    
                    # If we couldn't find sections, create artificial ones
                    if len(sections) <= 1:
                        # Create sections of approximately 1000 characters
                        sections = []
                        section_size = 1000
                        for i in range(0, len(full_content), section_size):
                            section_content = full_content[i:i+section_size]
                            section_title = f"Section {i//section_size + 1}"
                            sections.append((section_title, section_content))
                    
                    # Combine all sections with clear section markers
                    combined_content = ""
                    for i, (title, content) in enumerate(sections):
                        combined_content += f"\n--- SECTION {i+1}: {title} ---\n"
                        combined_content += content
                        combined_content += "\n\n"
                    
                    # Extract metadata
                    metadata = {
                        'source': file_path,
                        'title': self._extract_title(full_content, file_path),
                        'file_type': ext[1:],
                        'section_count': len(sections),
                        'embedding_model': self.config.embedding_model
                    }
                    
                    print(f"Successfully loaded {filename}, content length: {len(combined_content)} chars")
                    
                    document = {
                        'content': combined_content,
                        'metadata': metadata
                    }
                    
                    print("\nDocument Type Detection")
                    print(f"File: {filename} (Type: {ext[1:]})")
                    
                    return document
                    
                except Exception as word_err:
                    print(f"Error in Word document section-based processing: {word_err}. Falling back to standard loading.")
            
            # Standard processing for other files or if special handling failed
            documents = loader.load()
            
            if not documents:
                raise ValueError(f"Loader returned empty documents list for {file_path}")
                
            document = documents[0]
            
            # Extract metadata
            metadata = {
                'source': file_path,
                'title': self._extract_title(document.page_content, file_path),
                'file_type': ext[1:],  # Remove the dot
                'embedding_model': self.config.embedding_model  # Store the model used for embedding
            }
            
            print(f"Successfully loaded {filename}, content length: {len(document.page_content)} chars")
            
            # Create document dictionary
            document_dict = {
                'content': document.page_content,
                'metadata': metadata
            }
            
            # Print document type detection info for consistency with output
            print("\nDocument Type Detection")
            print(f"File: {filename} (Type: {ext[1:]})")
            
            return document_dict
        except Exception as e:
            print(f"ERROR: Failed to load file {file_path}: {e}")
            print(f"File extension: {ext}, Loader: {loader.__class__.__name__}")
            
            # For PDF and Word errors, give more detailed diagnostics
            if ext.lower() in ['.pdf', '.docx', '.doc']:
                # Check if file exists and is readable
                try:
                    with open(file_path, 'rb') as f:
                        file_size = len(f.read())
                    print(f"File exists and is readable. Size: {file_size} bytes")
                    
                    # Check if needed dependencies are available for PDFs
                    if ext.lower() == '.pdf':
                        try:
                            import pypdf
                            print("pypdf is available")
                        except ImportError:
                            print("ERROR: pypdf is not available")
                        
                        try:
                            import subprocess
                            result = subprocess.run(['which', 'pdftotext'], capture_output=True, text=True)
                            if result.returncode == 0:
                                print(f"pdftotext is available: {result.stdout.strip()}")
                            else:
                                print("pdftotext not found in PATH")
                        except Exception as e2:
                            print(f"Error checking for pdftotext: {e2}")
                    
                    # Check if needed dependencies are available for Word docs
                    if ext.lower() in ['.docx', '.doc']:
                        try:
                            import docx2txt
                            print("docx2txt is available")
                        except ImportError:
                            print("ERROR: docx2txt is not available")
                        
                        try:
                            import subprocess
                            result = subprocess.run(['which', 'libreoffice'], capture_output=True, text=True)
                            if result.returncode == 0:
                                print(f"libreoffice is available: {result.stdout.strip()}")
                            else:
                                print("libreoffice not found in PATH")
                        except Exception as e2:
                            print(f"Error checking for libreoffice: {e2}")
                            
                except Exception as e2:
                    print(f"Error reading file: {e2}")
            
            # Still return an empty document rather than failing completely,
            # but now we have better diagnostics in the logs
            return {
                'content': f"Error loading file {file_path}: {str(e)}",
                'metadata': {
                    'source': file_path,
                    'title': os.path.basename(file_path),
                    'file_type': ext[1:],
                    'embedding_model': self.config.embedding_model,
                    'error': str(e)
                }
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
        if content:
            lines = content.split('\n')
            for line in lines:
                if line.startswith('# '):
                    return line[2:].strip()
        
        # If no title found, use the filename without extension
        basename = os.path.basename(file_path)
        filename, _ = os.path.splitext(basename)
        return filename

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
                print(f"Failed to load custom embedding function. Falling back to OpenAI or Hugging Face.")
        
        # Verify one last time that sentence-transformers is available if needed
        if (self.config.embedding_model.startswith('sentence-transformers/') or 
            self.config.embedding_model.startswith('BAAI/') or
            self.config.embedding_model.startswith('Snowflake/') or
            '/' in self.config.embedding_model):
            # Double-check that sentence-transformers is really installed
            try:
                from sentence_transformers import SentenceTransformer
                print("Sentence Transformers package is available.")
            except ImportError:
                print("Final check: sentence-transformers not found. Installing...")
                if not self.config.install_sentence_transformers():
                    print("ERROR: Could not install sentence-transformers. Pipeline will likely fail.")
                    sys.exit(1)
        
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
        
        # Reset document type memory to ensure we don't carry over classifications
        print("\nResetting document type memory for fresh detection...")
        self.chunking_engine.reset_document_type_memory()
        
        # Debug file detection to see which files are being found and recognized
        self.file_reader.debug_file_detection()
        
        # Check for PDF files specifically
        pdf_files = []
        for root, _, filenames in os.walk(self.config.data_dir):
            for filename in filenames:
                if filename.lower().endswith('.pdf'):
                    file_path = os.path.join(root, filename)
                    print(f"DEBUG: Found PDF file: {file_path}")
                    pdf_files.append(file_path)
        
        if pdf_files:
            print(f"DEBUG: Found {len(pdf_files)} PDF files in data directory")
        else:
            print("DEBUG: No PDF files found in data directory")
        
        files = self.file_reader.get_files()
        
        # Debug file list
        for file_entry in files:
            print(f"DEBUG: Will process: {file_entry['relative_path']} (hash: {file_entry['hash'][:8]}...)")
        
        # Check if any PDFs were detected but not in the processing list
        pdf_to_process = []
        for file_entry in files:
            if file_entry['relative_path'].lower().endswith('.pdf'):
                pdf_to_process.append(file_entry['relative_path'])
        
        missing_pdfs = set([os.path.basename(p) for p in pdf_files]) - set([os.path.basename(p) for p in pdf_to_process])
        if missing_pdfs:
            print(f"DEBUG: WARNING - Some PDF files were detected but not included for processing: {missing_pdfs}")
            print("DEBUG: This might be because they are in the ignored files list or have already been processed.")
            
            # Try to force process any missing PDFs
            for pdf_path in pdf_files:
                if os.path.basename(pdf_path) in missing_pdfs:
                    print(f"DEBUG: Forcing processing of PDF: {pdf_path}")
                    file_hash = self.file_reader._get_file_hash(pdf_path)
                    if file_hash in self.file_reader.processed_files:
                        print(f"DEBUG: File {pdf_path} was already processed (hash: {file_hash[:8]}...)")
                        print(f"DEBUG: Removing from processed files list to reprocess")
                        self.file_reader.processed_files.remove(file_hash)
                    
                    # Add it to the list of files to process
                    relative_path = os.path.relpath(pdf_path, self.config.data_dir)
                    files.append({
                        'path': pdf_path,
                        'hash': file_hash,
                        'relative_path': relative_path,
                        'priority': True  # Mark as priority to process first
                    })
        
        print(f"\nFound {len(files)} files to process")
        
        # Track statistics by file type
        file_type_counts = {}
        file_type_success = {}
        file_type_errors = {}
        
        for file in files:
            # Get file extension
            _, ext = os.path.splitext(file['path'])
            ext = ext.lower()
            
            # Update file type counts
            if ext not in file_type_counts:
                file_type_counts[ext] = 0
                file_type_success[ext] = 0
                file_type_errors[ext] = 0
            file_type_counts[ext] += 1
            
            try:
                print(f"\nProcessing {file['relative_path']} (type: {ext})...")
                
                # Process file
                document = self.text_processor.process_file(file['path'])
                
                # Check if there was an error
                if 'error' in document.get('metadata', {}):
                    print(f"WARNING: Document processed with errors: {document['metadata']['error']}")
                    file_type_errors[ext] += 1
                    # Continue processing anyway to capture at least some content
                else:
                    file_type_success[ext] += 1
                
                # Document type detection and display
                from utils.chunking import DOCUMENT_TYPES
                detected_type = self.chunking_engine.detect_document_type(document['content'])
                print(f"Detected document type: {detected_type} - {DOCUMENT_TYPES[detected_type]}")
                
                print("\nAvailable document types:")
                for i, (doc_type, description) in enumerate(DOCUMENT_TYPES.items(), 1):
                    print(f"  {i}. {doc_type}: {description}")
                
                # Show prompt for document type selection (same as visualize_chunks.py)
                doc_key = self.chunking_engine.get_document_key(document)
                if doc_key in self.chunking_engine.document_type_memory:
                    remembered_type = self.chunking_engine.document_type_memory[doc_key]
                    print(f"Using previously selected type: {remembered_type} (no prompt needed)")
                else:
                    # We don't actually prompt but show this for UI consistency
                    print(f"Select document type (1-{len(DOCUMENT_TYPES)}) [default: {list(DOCUMENT_TYPES.keys()).index(detected_type)+1}]: {list(DOCUMENT_TYPES.keys()).index(detected_type)+1}")
                
                # Show available SpaCy and chunking info
                if hasattr(self.chunking_engine, 'spacy_splitter') and self.chunking_engine.spacy_splitter:
                    print(f"SpaCy is available for this document")
                else:
                    print(f"SpaCy is NOT available - using fallback chunking")
                
                # Chunk document
                document['metadata']['detected_type'] = detected_type
                chunks = self.chunking_engine.chunk_document(document)
                
                # Generate embeddings
                chunks_with_embeddings = self.embedding_generator.generate_embeddings(chunks)
                
                # Write to vector database
                self.db_writer.write_chunks(chunks_with_embeddings)
                
                # Mark file as processed
                self.file_reader.mark_processed(file['hash'])
                
                print(f"Successfully processed {file['relative_path']}")
            except Exception as e:
                print(f"Error processing {file['relative_path']}: {e}")
                file_type_errors[ext] += 1
        
        # Print summary statistics by file type
        print("\n===== Processing Summary =====")
        print(f"Total files processed: {len(files)}")
        
        print("\nBy file type:")
        for ext in sorted(file_type_counts.keys()):
            total = file_type_counts[ext]
            success = file_type_success[ext]
            errors = file_type_errors[ext]
            success_rate = (success / total) * 100 if total > 0 else 0
            
            print(f"  {ext}: {success}/{total} successful ({success_rate:.1f}%), {errors} errors")
        
        print("\nPipeline completed successfully")

if __name__ == "__main__":
    config_path = os.environ.get("CONFIG_PATH", "/config/pipeline_config.json")
    pipeline = Pipeline(config_path)
    pipeline.run()