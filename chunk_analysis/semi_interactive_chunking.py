#!/usr/bin/env python3
"""
Semi-interactive chunking wrappers

This module provides wrapper classes for chunking engines that remember document type selections
but can still run in non-interactive environments like Docker by using default values when needed.
"""
import os
import sys
import json
import select
from typing import Dict, Any

# Add the src directory to the path
sys.path.insert(0, '/app/src')
sys.path.insert(0, '/app')

# Import the standard chunking engine
from src.utils.chunking import ChunkingEngine, DOCUMENT_TYPES

# Try to import the enhanced chunking engine
try:
    from enhanced_chunking import EnhancedChunkingEngine
    ENHANCED_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import EnhancedChunkingEngine: {e}")
    ENHANCED_AVAILABLE = False

# Path to store document type selections
SELECTIONS_FILE = '/test_chunks/document_type_selections.json'

class DocTypeMemory:
    """Helper class to store and retrieve document type selections"""
    
    @staticmethod
    def load_selections():
        """Load saved document type selections"""
        if os.path.exists(SELECTIONS_FILE):
            try:
                with open(SELECTIONS_FILE, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading document selections: {e}")
        return {}
    
    @staticmethod
    def save_selection(doc_key, doc_type):
        """Save a document type selection"""
        selections = DocTypeMemory.load_selections()
        selections[doc_key] = doc_type
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(SELECTIONS_FILE), exist_ok=True)
        
        try:
            with open(SELECTIONS_FILE, 'w') as f:
                json.dump(selections, f, indent=2)
        except Exception as e:
            print(f"Error saving document selections: {e}")

class SemiInteractiveChunkingEngine(ChunkingEngine):
    """
    A version of ChunkingEngine that uses document type memory and default selections
    when running in non-interactive mode, but still maintains the behavior of the
    original system.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Load the persistent document type selections
        self.persistent_memory = DocTypeMemory.load_selections()
        print(f"Loaded {len(self.persistent_memory)} saved document type selections")
    
    def prompt_for_document_type(self, document: Dict[str, Any]) -> str:
        """
        Override to use document type memory and default selections when in non-interactive mode.
        
        Args:
            document: Document to detect type for
            
        Returns:
            Selected document type
        """
        # Get the content to analyze
        content = document.get('content', '')
        
        # Get a unique key for this document
        doc_key = self.get_document_key(document)
        
        # First check if we've already processed this document type
        if doc_key in self.document_type_memory:
            remembered_type = self.document_type_memory[doc_key]
            print(f"Using previously selected type: {remembered_type} (from memory)")
            return remembered_type
        
        # Then check if we have it in our persistent storage
        if doc_key in self.persistent_memory:
            remembered_type = self.persistent_memory[doc_key]
            print(f"Using previously selected type: {remembered_type} (from storage)")
            self.document_type_memory[doc_key] = remembered_type
            return remembered_type
        
        # Detect document type automatically
        detected_type = self.detect_document_type(content)
        
        # In an interactive environment, prompt the user for input
        # But keep track of file extensions to only prompt once per extension type
        file_ext = document.get('metadata', {}).get('source', '').split('.')[-1].lower() if document.get('metadata', {}).get('source', '') else ''
        
        ext_key = f"ext_{file_ext}" if file_ext else "unknown_ext"
        
        # Check if we've already prompted for this file extension
        if ext_key in self.document_type_memory:
            # Use the previously selected type for this extension
            selected_type = self.document_type_memory[ext_key]
            print(f"Using previously selected type for .{file_ext} files: {selected_type}")
            
            # Store for this specific document too
            self.document_type_memory[doc_key] = selected_type
            DocTypeMemory.save_selection(doc_key, selected_type)
            return selected_type
        
        # This is the first time we're seeing this extension type, prompt user
        print(f"\nDocument Type Detection")
        if file_ext:
            print(f"File: {os.path.basename(document.get('metadata', {}).get('source', ''))} (Type: {file_ext})")
        print(f"Detected document type: {detected_type}")
        print(f"\nSelect document type (1-{len(DOCUMENT_TYPES)}) [default: {list(DOCUMENT_TYPES.keys()).index(detected_type)+1}]: ", end="", flush=True)
        
        # Try to get user input, but use default in Docker (non-interactive)
        # This is designed to be an interactive process, so try to get user input
        # with a reasonable timeout
        try:
            # First check if we already have a choice for this document type in our memory
            ext_key = f"ext_{file_ext}" if file_ext else "unknown_ext"
            if ext_key in self.document_type_memory:
                remembered_type = self.document_type_memory[ext_key]
                print(f"Using previously selected type for {file_ext} files: {remembered_type}")
                choice = str(list(DOCUMENT_TYPES.keys()).index(remembered_type)+1)
            else:
                # Show the prompt and wait for input with a timeout
                print(f"\nSelect document type (1-{len(DOCUMENT_TYPES)}) [default: {list(DOCUMENT_TYPES.keys()).index(detected_type)+1}]: ", end="", flush=True)
                
                # Set a timeout for input - if we don't get input within the timeout, use the default
                try:
                    # Use select to implement timeout (works in both terminal and container)
                    ready, _, _ = select.select([sys.stdin], [], [], 10)  # 10-second timeout
                    if ready:
                        choice = input().strip()
                    else:
                        # If no input received within timeout, use detected type
                        choice = str(list(DOCUMENT_TYPES.keys()).index(detected_type)+1)
                        print(f"{choice} (auto-selected after 10-second timeout)")
                except Exception as e:
                    # If select doesn't work, try a direct input
                    try:
                        choice = input().strip()
                    except Exception as inner_e:
                        # Last resort fallback
                        choice = str(list(DOCUMENT_TYPES.keys()).index(detected_type)+1)
                        print(f"{choice} (auto-selected due to input error: {inner_e})")
        except Exception as e:
            # Ultimate fallback for any error
            choice = str(list(DOCUMENT_TYPES.keys()).index(detected_type)+1)
            print(f"{choice} (auto-selected due to: {e})")
        
        if not choice:
            choice = str(list(DOCUMENT_TYPES.keys()).index(detected_type)+1)
        
        # Convert choice to integer
        try:
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(DOCUMENT_TYPES):
                selected_type = list(DOCUMENT_TYPES.keys())[choice_idx]
            else:
                print(f"Invalid choice, using detected type: {detected_type}")
                selected_type = detected_type
        except ValueError:
            print(f"Invalid input, using detected type: {detected_type}")
            selected_type = detected_type
        
        # Store the selection in memory for both the document and the extension type
        self.document_type_memory[doc_key] = selected_type
        self.document_type_memory[ext_key] = selected_type
        DocTypeMemory.save_selection(doc_key, selected_type)
        
        return selected_type

# Only create the enhanced version if it's available
if ENHANCED_AVAILABLE:
    class SemiInteractiveEnhancedChunkingEngine(EnhancedChunkingEngine):
        """
        A version of EnhancedChunkingEngine that uses document type memory and default selections
        when running in non-interactive mode.
        """
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Load the persistent document type selections
            self.persistent_memory = DocTypeMemory.load_selections()
            print(f"Loaded {len(self.persistent_memory)} saved document type selections (enhanced)")
        
        def prompt_for_document_type(self, document: Dict[str, Any]) -> str:
            """
            Override to use document type memory and default selections when in non-interactive mode.
            
            Args:
                document: Document to detect type for
                
            Returns:
                Selected document type
            """
            # Get the content to analyze
            content = document.get('content', '')
            
            # Get a unique key for this document
            doc_key = self.get_document_key(document)
            
            # First check if we've already processed this document type
            if doc_key in self.document_type_memory:
                remembered_type = self.document_type_memory[doc_key]
                print(f"Using previously selected type: {remembered_type} (from memory)")
                return remembered_type
            
            # Then check if we have it in our persistent storage
            if doc_key in self.persistent_memory:
                remembered_type = self.persistent_memory[doc_key]
                print(f"Using previously selected type: {remembered_type} (from storage)")
                self.document_type_memory[doc_key] = remembered_type
                return remembered_type
            
            # Detect document type automatically
            detected_type = self.detect_document_type(content)
            
            # In an interactive environment, prompt the user for input
            # But keep track of file extensions to only prompt once per extension type
            file_ext = document.get('metadata', {}).get('source', '').split('.')[-1].lower() if document.get('metadata', {}).get('source', '') else ''
            
            ext_key = f"ext_{file_ext}" if file_ext else "unknown_ext"
            
            # Check if we've already prompted for this file extension
            if ext_key in self.document_type_memory:
                # Use the previously selected type for this extension
                selected_type = self.document_type_memory[ext_key]
                print(f"Using previously selected type for .{file_ext} files: {selected_type}")
                
                # Store for this specific document too
                self.document_type_memory[doc_key] = selected_type
                DocTypeMemory.save_selection(doc_key, selected_type)
                return selected_type
            
            # This is the first time we're seeing this extension type, prompt user
            print(f"\nDocument Type Detection")
            if file_ext:
                print(f"File: {os.path.basename(document.get('metadata', {}).get('source', ''))} (Type: {file_ext})")
            print(f"Detected document type: {detected_type}")
            print(f"\nSelect document type (1-{len(DOCUMENT_TYPES)}) [default: {list(DOCUMENT_TYPES.keys()).index(detected_type)+1}]: ", end="", flush=True)
            
            # Try to get user input, but use default in Docker (non-interactive)
            # This is designed to be an interactive process, so try to get user input
            # with a reasonable timeout
            try:
                # First check if we already have a choice for this document type in our memory
                ext_key = f"ext_{file_ext}" if file_ext else "unknown_ext"
                if ext_key in self.document_type_memory:
                    remembered_type = self.document_type_memory[ext_key]
                    print(f"Using previously selected type for {file_ext} files: {remembered_type}")
                    choice = str(list(DOCUMENT_TYPES.keys()).index(remembered_type)+1)
                else:
                    # Show the prompt and wait for input with a timeout
                    print(f"\nSelect document type (1-{len(DOCUMENT_TYPES)}) [default: {list(DOCUMENT_TYPES.keys()).index(detected_type)+1}]: ", end="", flush=True)
                    
                    # Set a timeout for input - if we don't get input within the timeout, use the default
                    try:
                        # Use select to implement timeout (works in both terminal and container)
                        ready, _, _ = select.select([sys.stdin], [], [], 10)  # 10-second timeout
                        if ready:
                            choice = input().strip()
                        else:
                            # If no input received within timeout, use detected type
                            choice = str(list(DOCUMENT_TYPES.keys()).index(detected_type)+1)
                            print(f"{choice} (auto-selected after 10-second timeout)")
                    except Exception as e:
                        # If select doesn't work, try a direct input
                        try:
                            choice = input().strip()
                        except Exception as inner_e:
                            # Last resort fallback
                            choice = str(list(DOCUMENT_TYPES.keys()).index(detected_type)+1)
                            print(f"{choice} (auto-selected due to input error: {inner_e})")
            except Exception as e:
                # Ultimate fallback for any error
                choice = str(list(DOCUMENT_TYPES.keys()).index(detected_type)+1)
                print(f"{choice} (auto-selected due to: {e})")
            
            if not choice:
                choice = str(list(DOCUMENT_TYPES.keys()).index(detected_type)+1)
            
            # Convert choice to integer
            try:
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(DOCUMENT_TYPES):
                    selected_type = list(DOCUMENT_TYPES.keys())[choice_idx]
                else:
                    print(f"Invalid choice, using detected type: {detected_type}")
                    selected_type = detected_type
            except ValueError:
                print(f"Invalid input, using detected type: {detected_type}")
                selected_type = detected_type
            
            # Store the selection in memory for both the document and the extension type
            self.document_type_memory[doc_key] = selected_type
            self.document_type_memory[ext_key] = selected_type
            DocTypeMemory.save_selection(doc_key, selected_type)
            
            return selected_type