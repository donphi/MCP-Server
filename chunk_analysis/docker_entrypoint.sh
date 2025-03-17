#!/bin/bash
# Docker entrypoint script for chunk analysis
# This script sets up the environment before running the tests

echo "Starting chunk analysis in Docker container"

# Show environment information
echo "Environment variables:"
echo "PYTHONPATH: $PYTHONPATH"
echo "CHUNK_SIZE: $CHUNK_SIZE"
echo "CHUNK_OVERLAP: $CHUNK_OVERLAP"
echo "CONFIG_PATH: $CONFIG_PATH"

# Show directory structure
echo "Directory structure:"
ls -la /app
echo "Source directory:"
ls -la /app/src
echo "Utils directory:"
ls -la /app/src/utils
echo "Config directory:"
ls -la /app/config
echo "Chunk analysis directory:"
ls -la /app/chunk_analysis

# Create symlinks to help with imports
echo "Creating symlinks for imports..."
mkdir -p /app/utils
ln -sf /app/src/utils/chunking.py /app/utils/chunking.py
ln -sf /app/src/utils/embedding.py /app/utils/embedding.py
ln -sf /app/src/utils/vector_db.py /app/utils/vector_db.py
touch /app/utils/__init__.py

# Create direct import wrappers instead of using importlib.util
echo "Creating direct import wrappers..."
mkdir -p /app/wrappers

# Create a wrapper for pipeline
cat > /app/wrappers/pipeline.py << 'EOL'
# Direct import wrapper for pipeline.py
import sys
import os

# Add the original file's directory to sys.path
sys.path.insert(0, '/app/src')

# Import the original module
from src.pipeline import PipelineConfig, TextProcessor, FileReader

# Make these classes available at the top level
__all__ = ['PipelineConfig', 'TextProcessor', 'FileReader']
EOL

# Create a wrapper for chunking
cat > /app/wrappers/chunking.py << 'EOL'
# Direct import wrapper for chunking.py
import sys
import os

# Add the original file's directory to sys.path
sys.path.insert(0, '/app/src/utils')

# Import the original module
from src.utils.chunking import ChunkingEngine, DOCUMENT_TYPES, SPACY_AVAILABLE, SPACY_MODEL_LOADED

# Make these classes available at the top level
__all__ = ['ChunkingEngine', 'DOCUMENT_TYPES', 'SPACY_AVAILABLE', 'SPACY_MODEL_LOADED']
EOL

# Create a wrapper for enhanced_chunking
cat > /app/wrappers/enhanced_chunking.py << 'EOL'
# Direct import wrapper for enhanced_chunking.py
import sys
import os

# Add the original file's directory to sys.path
sys.path.insert(0, '/app')

# Try to import the original module
try:
    from enhanced_chunking import EnhancedChunkingEngine
    ENHANCED_AVAILABLE = True
except ImportError as e:
    print(f"Error importing EnhancedChunkingEngine: {e}")
    # Create a dummy class
    class EnhancedChunkingEngine:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("EnhancedChunkingEngine could not be imported")
    ENHANCED_AVAILABLE = False

# Make these classes available at the top level
__all__ = ['EnhancedChunkingEngine', 'ENHANCED_AVAILABLE']
EOL

# Check for semi-interactive chunking module
echo "Checking for semi-interactive chunking files..."
if [ -f "/app/chunk_analysis/semi_interactive_chunking.py" ]; then
    echo "Found semi_interactive_chunking.py, creating wrapper"
    # Create a wrapper for semi_interactive_chunking
    cat > /app/wrappers/semi_interactive_chunking.py << 'EOL'
# Direct import wrapper for semi_interactive_chunking.py
import sys
import os

# Add the original file's directory to sys.path
sys.path.insert(0, '/app/chunk_analysis')

# Try to import the original module
try:
    from chunk_analysis.semi_interactive_chunking import SemiInteractiveChunkingEngine, SemiInteractiveEnhancedChunkingEngine, DocTypeMemory
    print("Successfully imported SemiInteractiveChunkingEngine")
except ImportError as e:
    print(f"Error importing SemiInteractiveChunkingEngine: {e}")
    # Create a dummy class
    from src.utils.chunking import ChunkingEngine
    
    class SemiInteractiveChunkingEngine(ChunkingEngine):
        pass
        
    SemiInteractiveEnhancedChunkingEngine = None
    DocTypeMemory = None

# Make these classes available at the top level
__all__ = ['SemiInteractiveChunkingEngine', 'SemiInteractiveEnhancedChunkingEngine', 'DocTypeMemory']
EOL
else
    echo "WARNING: semi_interactive_chunking.py not found in chunk_analysis directory"
    ls -la /app/chunk_analysis
fi

# Create __init__.py for wrappers
cat > /app/wrappers/__init__.py << 'EOL'
# Wrappers package
EOL

# Update PYTHONPATH to include wrappers
export PYTHONPATH=$PYTHONPATH:/app/wrappers

# Create simplified test script that uses the wrappers
cat > /app/run_test.py << 'EOL'
#!/usr/bin/env python3
"""
Simplified test script that uses wrapper modules
"""
import os
import sys
import json
import select
from datetime import datetime

# Import from wrappers instead of using dynamic loading
from wrappers.pipeline import PipelineConfig, TextProcessor
from wrappers.chunking import ChunkingEngine, DOCUMENT_TYPES
from wrappers.enhanced_chunking import EnhancedChunkingEngine, ENHANCED_AVAILABLE

# Verify we're in interactive mode
def is_interactive():
    try:
        if not sys.stdin.isatty():
            print("STDIN is not a TTY - interactive features limited")
            return False
        
        # Check if we can read from stdin with a short timeout
        ready, _, _ = select.select([sys.stdin], [], [], 0.1)
        print(f"Interactive mode test: {'Ready to receive input' if ready else 'Input may have delay'}")
        return True
    except Exception as e:
        print(f"Interactive mode check error: {e}")
        return False

interactive_mode = is_interactive()
print(f"Running in {'interactive' if interactive_mode else 'non-interactive'} mode")

# Try to import semi-interactive chunking
try:
    from wrappers.semi_interactive_chunking import SemiInteractiveChunkingEngine, SemiInteractiveEnhancedChunkingEngine
    SEMI_INTERACTIVE_AVAILABLE = True
    print("Successfully imported semi-interactive chunking modules")
except ImportError:
    SEMI_INTERACTIVE_AVAILABLE = False
    print("Semi-interactive chunking not available, will use standard chunking")

print("Successfully imported all modules using wrappers")

# Rest of the test script
print("Starting chunk analysis...")
print(f"CHUNK_SIZE: {os.environ.get('CHUNK_SIZE', 'Not set')}")
print(f"CHUNK_OVERLAP: {os.environ.get('CHUNK_OVERLAP', 'Not set')}")

# Just a simple test - load config and create engines
try:
    config = PipelineConfig(os.environ.get('CONFIG_PATH', '/app/config/pipeline_config.json'))
    
    # Try to create engines
    if 'SEMI_INTERACTIVE_AVAILABLE' in locals() and SEMI_INTERACTIVE_AVAILABLE:
        chunking_engine = SemiInteractiveChunkingEngine(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        print("Using semi-interactive chunking engine")
    else:
        chunking_engine = ChunkingEngine(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        print("Using standard chunking engine")
    
    if ENHANCED_AVAILABLE:
        if 'SEMI_INTERACTIVE_AVAILABLE' in locals() and SEMI_INTERACTIVE_AVAILABLE:
            enhanced_engine = SemiInteractiveEnhancedChunkingEngine(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap
            )
            print("Using semi-interactive enhanced chunking engine")
        else:
            enhanced_engine = EnhancedChunkingEngine(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap
            )
            print("Using standard enhanced chunking engine")
        print("Created both standard and enhanced chunking engines")
    else:
        print("Enhanced chunking not available")
    
    print("Successfully loaded config and created engines")
    
    # Now run the actual test script with input redirection to maintain interactivity
    print("Running main test script...")
    os.system("python /app/chunk_analysis/test_chunking.py < /dev/tty")
    
except Exception as e:
    print(f"Error in simplified test: {e}")
    import traceback
    traceback.print_exc()
EOL

# Make the test script executable
chmod +x /app/run_test.py

# Run the simplified test script first as a check
echo "Running simplified test to verify imports..."
python /app/run_test.py