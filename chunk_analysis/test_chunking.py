#!/usr/bin/env python3
"""
Chunk Analysis Testing Script

This script orchestrates testing of both standard and enhanced chunking methods
within a Docker container, without modifying the original application code.
It uses semi-interactive chunking engines that remember document type selections.
"""
import os
import sys
import json
import argparse
import importlib
import importlib.util  # Explicitly import importlib.util
import traceback
from datetime import datetime
from typing import Dict, List, Any, Tuple

# Add app directories to path
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/src')

# Import modules directly by file path using importlib
print("Importing modules directly using importlib...")

# Function to import modules from file paths
def import_from_file(file_path, module_name):
    print(f"Loading module {module_name} from {file_path}")
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if not spec or not spec.loader:
        raise ImportError(f"Could not load spec for {file_path}")
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Import standard libraries first
try:
    print("Current directory:", os.getcwd())
    print("Files in /app:", os.listdir("/app") if os.path.exists("/app") else "Directory not found")
    print("Files in /app/src:", os.listdir("/app/src") if os.path.exists("/app/src") else "Directory not found")
    print("Files in /app/src/utils:", os.listdir("/app/src/utils") if os.path.exists("/app/src/utils") else "Directory not found")
    
    # Import the semi-interactive chunking module
    semi_interactive_path = "/app/chunk_analysis/semi_interactive_chunking.py"
    if os.path.exists(semi_interactive_path):
        semi_interactive_module = import_from_file(semi_interactive_path, "semi_interactive_chunking")
        print("Successfully loaded semi_interactive_chunking module")
    else:
        print(f"ERROR: Could not find {semi_interactive_path}")
        sys.exit(1)
    
    # Import the standard pipeline and chunking modules
    pipeline_module = import_from_file("/app/src/pipeline.py", "pipeline")
    chunking_module = import_from_file("/app/src/utils/chunking.py", "chunking")
    
    # Get the classes we need
    PipelineConfig = pipeline_module.PipelineConfig
    TextProcessor = pipeline_module.TextProcessor
    ChunkingEngine = chunking_module.ChunkingEngine
    
    # Get the semi-interactive versions
    SemiInteractiveChunkingEngine = semi_interactive_module.SemiInteractiveChunkingEngine
    
    print("Successfully imported standard modules")
except Exception as e:
    print(f"Error importing standard modules: {e}")
    traceback.print_exc()
    sys.exit(1)

# Import enhanced chunking
try:
    # Import using our direct file path method
    enhanced_chunking = import_from_file("/app/enhanced_chunking.py", "enhanced_chunking")
    EnhancedChunkingEngine = enhanced_chunking.EnhancedChunkingEngine
    
    # Try to get the semi-interactive enhanced version
    if hasattr(semi_interactive_module, 'SemiInteractiveEnhancedChunkingEngine'):
        SemiInteractiveEnhancedChunkingEngine = semi_interactive_module.SemiInteractiveEnhancedChunkingEngine
        print("Successfully imported SemiInteractiveEnhancedChunkingEngine")
    else:
        print("Warning: SemiInteractiveEnhancedChunkingEngine not found, will use regular EnhancedChunkingEngine")
        SemiInteractiveEnhancedChunkingEngine = EnhancedChunkingEngine
    
    print("Successfully imported EnhancedChunkingEngine")
    ENHANCED_AVAILABLE = True
except Exception as e:
    print(f"Warning: Could not import EnhancedChunkingEngine: {e}")
    traceback.print_exc()
    print("Will only run standard chunking")
    ENHANCED_AVAILABLE = False

def setup_output_dirs():
    """Create output directories if they don't exist"""
    os.makedirs("/test_chunks/standard", exist_ok=True)
    os.makedirs("/test_chunks/enhanced", exist_ok=True)
    os.makedirs("/test_chunks/comparison", exist_ok=True)
    print(f"Created output directories")

def get_config():
    """Load configuration from environment/config file"""
    config_path = os.environ.get("CONFIG_PATH", "/app/config/pipeline_config.json")
    print(f"Loading configuration from {config_path}")
    
    # Load config and check if environment variables are set
    config = PipelineConfig(config_path)
    
    # Print configuration details
    print(f"Using CHUNK_SIZE: {config.chunk_size} (from {'environment' if 'CHUNK_SIZE' in os.environ else 'config file'})")
    print(f"Using CHUNK_OVERLAP: {config.chunk_overlap} (from {'environment' if 'CHUNK_OVERLAP' in os.environ else 'config file'})")
    
    return config

def find_files(data_dir):
    """Find all supported files in data directory"""
    files = []
    
    for root, _, filenames in os.walk(data_dir):
        for filename in filenames:
            file_path = os.path.join(root, filename)
            _, ext = os.path.splitext(file_path)
            if ext.lower() in ['.pdf', '.txt', '.md', '.docx', '.doc']:
                files.append(file_path)
                
    print(f"Found {len(files)} files to process")
    return files

def process_with_standard_chunking(config, files):
    """Process files with standard chunking"""
    print("\n=== Processing with Standard Chunking ===")
    results = {}
    
    text_processor = TextProcessor(config)
    print("Initialized document loaders for file types:", list(text_processor.loader_map.keys()))
    
    # Use our semi-interactive chunking engine that will remember selections
    chunking_engine = SemiInteractiveChunkingEngine(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap
    )
    
    # Reset document type memory
    chunking_engine.reset_document_type_memory()
    print("Document type memory has been reset")
    
    for file_path in files:
        try:
            filename = os.path.basename(file_path)
            print(f"\nProcessing {filename} with standard chunking...")
            
            # Create output directory for this file
            file_dir = os.path.join("/test_chunks/standard", os.path.splitext(filename)[0])
            os.makedirs(file_dir, exist_ok=True)
            
            # Process the file
            document = text_processor.process_file(file_path)
            
            # Save original document
            with open(os.path.join(file_dir, "original.json"), 'w') as f:
                json.dump(document, f, indent=2)
            
            # Detect document type
            detected_type = chunking_engine.detect_document_type(document['content'])
            print(f"Detected document type: {detected_type}")
            
            # Create chunks
            print(f"Creating standard chunks...")
            chunks = chunking_engine.chunk_document(document)
            
            # Save chunks
            save_chunks(chunks, file_dir, config.chunk_size)
            
            # Store results for comparison
            results[filename] = {
                'chunks': len(chunks),
                'oversized': sum(1 for c in chunks if len(c['content']) > config.chunk_size),
                'type': detected_type
            }
            
        except Exception as e:
            print(f"Error processing {file_path} with standard chunking: {e}")
            traceback.print_exc()
    
    return results

def process_with_enhanced_chunking(config, files):
    """Process files with enhanced chunking"""
    if not ENHANCED_AVAILABLE:
        print("\n=== Enhanced chunking not available, skipping ===")
        return {}
        
    print("\n=== Processing with Enhanced Chunking ===")
    results = {}
    
    text_processor = TextProcessor(config)
    print("Initialized document loaders for file types:", list(text_processor.loader_map.keys()))
    
    # Use our semi-interactive enhanced chunking engine that will remember selections
    enhanced_engine = SemiInteractiveEnhancedChunkingEngine(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap
    )
    
    # Reset document type memory
    enhanced_engine.reset_document_type_memory()
    print("Document type memory has been reset")
    
    for file_path in files:
        try:
            filename = os.path.basename(file_path)
            print(f"\nProcessing {filename} with enhanced chunking...")
            
            # Create output directory for this file
            file_dir = os.path.join("/test_chunks/enhanced", os.path.splitext(filename)[0])
            os.makedirs(file_dir, exist_ok=True)
            
            # Process the file
            document = text_processor.process_file(file_path)
            
            # Save original document
            with open(os.path.join(file_dir, "original.json"), 'w') as f:
                json.dump(document, f, indent=2)
            
            # Detect document type
            detected_type = enhanced_engine.detect_document_type(document['content'])
            print(f"Detected document type: {detected_type}")
            
            # Create chunks
            print(f"Creating enhanced chunks...")
            chunks = enhanced_engine.chunk_document(document)
            
            # Save chunks
            save_chunks(chunks, file_dir, config.chunk_size)
            
            # Store results for comparison
            results[filename] = {
                'chunks': len(chunks),
                'oversized': sum(1 for c in chunks if len(c['content']) > config.chunk_size),
                'type': detected_type
            }
            
        except Exception as e:
            print(f"Error processing {file_path} with enhanced chunking: {e}")
            traceback.print_exc()
    
    return results

def save_chunks(chunks, output_dir, max_size):
    """Save chunks to files"""
    oversized_count = 0
    
    for i, chunk in enumerate(chunks):
        chunk_path = os.path.join(output_dir, f"chunk_{i+1:04d}.txt")
        with open(chunk_path, 'w') as f:
            f.write(f"CHUNK #{i+1}\n")
            f.write(f"Length: {len(chunk['content'])} characters\n")
            f.write(f"Metadata: {json.dumps(chunk['metadata'], indent=2)}\n\n")
            f.write(f"CONTENT:\n{chunk['content']}\n")
        
        # If chunk is oversized, flag it
        if len(chunk['content']) > max_size:
            oversized_count += 1
            with open(os.path.join(output_dir, f"chunk_{i+1:04d}.OVERSIZED"), 'w') as f:
                f.write(f"WARNING: Chunk size {len(chunk['content'])} exceeds limit {max_size}\n")
    
    # Save summary
    with open(os.path.join(output_dir, "summary.txt"), 'w') as f:
        f.write(f"Total chunks: {len(chunks)}\n")
        f.write(f"Oversized chunks: {oversized_count}/{len(chunks)}\n")
        if oversized_count > 0:
            oversized = [i for i, c in enumerate(chunks) if len(c['content']) > max_size]
            f.write(f"Oversized chunk numbers: {', '.join(str(i+1) for i in oversized)}\n")
    
    print(f"Saved {len(chunks)} chunks ({oversized_count} oversized)")

def create_comparison_report(standard_results, enhanced_results):
    """Create a comparison report between standard and enhanced chunking"""
    if not ENHANCED_AVAILABLE or not enhanced_results:
        print("\n=== Skipping comparison - enhanced chunking not available ===")
        return
        
    print("\n=== Creating Comparison Report ===")
    
    # Create comparison directory if it doesn't exist
    os.makedirs("/test_chunks/comparison", exist_ok=True)
    
    # Generate comparison for each file
    for filename in standard_results:
        if filename in enhanced_results:
            std = standard_results[filename]
            enh = enhanced_results[filename]
            
            # Write comparison file
            output_path = os.path.join("/test_chunks/comparison", f"{os.path.splitext(filename)[0]}_comparison.txt")
            with open(output_path, 'w') as f:
                f.write(f"File: {filename}\n")
                f.write(f"Document type: {std['type']}\n\n")
                f.write(f"Standard chunking: {std['chunks']} chunks, {std['oversized']} oversized\n")
                f.write(f"Enhanced chunking: {enh['chunks']} chunks, {enh['oversized']} oversized\n\n")
                
                chunk_diff = enh['chunks'] - std['chunks']
                oversized_diff = enh['oversized'] - std['oversized']
                
                f.write(f"Difference in chunks: {chunk_diff:+d}\n")
                f.write(f"Difference in oversized chunks: {oversized_diff:+d}\n\n")
                
                if enh['oversized'] < std['oversized']:
                    f.write(f"Enhanced chunking reduced oversized chunks by {std['oversized'] - enh['oversized']}\n")
                elif enh['oversized'] > std['oversized']:
                    f.write(f"Enhanced chunking increased oversized chunks by {enh['oversized'] - std['oversized']}\n")
                else:
                    f.write(f"Both methods produced the same number of oversized chunks\n")
    
    # Create overall summary
    std_total_chunks = sum(r['chunks'] for r in standard_results.values())
    std_total_oversized = sum(r['oversized'] for r in standard_results.values())
    
    if not enhanced_results:
        # Just show standard results if enhanced not available
        with open("/test_chunks/comparison/overall_summary.txt", 'w') as f:
            f.write("=== Chunking Summary ===\n\n")
            f.write(f"Total files processed: {len(standard_results)}\n\n")
            f.write(f"Standard chunking:\n")
            f.write(f"  Total chunks: {std_total_chunks}\n")
            f.write(f"  Oversized chunks: {std_total_oversized}")
            if std_total_chunks > 0:
                f.write(f" ({std_total_oversized/std_total_chunks*100:.1f}%)\n\n")
            else:
                f.write("\n\n")
        return
        
    enh_total_chunks = sum(r['chunks'] for r in enhanced_results.values())
    enh_total_oversized = sum(r['oversized'] for r in enhanced_results.values())
    
    with open("/test_chunks/comparison/overall_summary.txt", 'w') as f:
        f.write("=== Chunking Comparison Summary ===\n\n")
        f.write(f"Total files processed: {len(standard_results)}\n\n")
        f.write(f"Standard chunking:\n")
        f.write(f"  Total chunks: {std_total_chunks}\n")
        if std_total_chunks > 0:
            f.write(f"  Oversized chunks: {std_total_oversized} ({std_total_oversized/std_total_chunks*100:.1f}%)\n\n")
        else:
            f.write(f"  Oversized chunks: {std_total_oversized} (0.0%)\n\n")
            
        f.write(f"Enhanced chunking:\n")
        f.write(f"  Total chunks: {enh_total_chunks}\n")
        if enh_total_chunks > 0:
            f.write(f"  Oversized chunks: {enh_total_oversized} ({enh_total_oversized/enh_total_chunks*100:.1f}%)\n\n")
        else:
            f.write(f"  Oversized chunks: {enh_total_oversized} (0.0%)\n\n")
        
        if std_total_oversized > 0 and enh_total_oversized < std_total_oversized:
            f.write(f"Enhanced chunking reduced oversized chunks by {std_total_oversized - enh_total_oversized} ")
            f.write(f"({(std_total_oversized - enh_total_oversized)/std_total_oversized*100:.1f}%)\n")
        elif std_total_oversized > 0 and enh_total_oversized > std_total_oversized:
            f.write(f"Enhanced chunking increased oversized chunks by {enh_total_oversized - std_total_oversized} ")
            f.write(f"({(enh_total_oversized - std_total_oversized)/std_total_oversized*100:.1f}%)\n")
        else:
            f.write(f"Both methods produced the same number of oversized chunks\n")
            
        f.write("\nOverall recommendation: ")
        if enh_total_oversized < std_total_oversized:
            f.write("Enhanced chunking appears to be beneficial for reducing oversized chunks.\n")
        else:
            f.write("Standard chunking may be preferable as it produces similar or better results.\n")
    
    print(f"Comparison reports created in /test_chunks/comparison/")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Test different chunking methods')
    parser.add_argument('--data-dir', default="/data", help='Directory containing input files')
    args = parser.parse_args()
    
    print(f"Starting chunk analysis tests on {args.data_dir}")
    print(f"Current environment variables:")
    print(f"  CHUNK_SIZE: {os.environ.get('CHUNK_SIZE', 'Not set')}")
    print(f"  CHUNK_OVERLAP: {os.environ.get('CHUNK_OVERLAP', 'Not set')}")
    print(f"  PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
    
    # Setup directories
    setup_output_dirs()
    
    # Load configuration
    try:
        config = get_config()
    except Exception as e:
        print(f"Error loading configuration: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Find files to process
    files = find_files(args.data_dir)
    
    if not files:
        print("No files found to process. Exiting.")
        return
    
    # Process with standard chunking
    try:
        standard_results = process_with_standard_chunking(config, files)
    except Exception as e:
        print(f"Error in standard chunking: {e}")
        traceback.print_exc()
        standard_results = {}
    
    # Process with enhanced chunking
    try:
        enhanced_results = process_with_enhanced_chunking(config, files)
    except Exception as e:
        print(f"Error in enhanced chunking: {e}")
        traceback.print_exc()
        enhanced_results = {}
    
    # Create comparison report
    try:
        create_comparison_report(standard_results, enhanced_results)
    except Exception as e:
        print(f"Error creating comparison report: {e}")
        traceback.print_exc()
    
    print("\nChunk analysis complete!")
    print("Results saved to:")
    print("  Standard chunks: /test_chunks/standard/")
    print("  Enhanced chunks: /test_chunks/enhanced/")
    print("  Comparison: /test_chunks/comparison/")

if __name__ == "__main__":
    main()