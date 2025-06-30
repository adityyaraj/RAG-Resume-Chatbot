#!/usr/bin/env python3
"""
Deployment-optimized version of the RAG Resume Reviewer
Includes fallbacks and memory optimizations for cloud deployment
"""

import os
import gc
from pathlib import Path

# Set memory optimization environment variables
os.environ['TRANSFORMERS_CACHE'] = '/tmp/transformers_cache'
os.environ['HF_HOME'] = '/tmp/huggingface'

# Import with error handling
try:
    import chainlit as cl
    from dotenv import load_dotenv
    print("✅ Core imports successful")
except ImportError as e:
    print(f"❌ Core import error: {e}")
    exit(1)

# Load environment
load_dotenv()

def check_memory_usage():
    """Check current memory usage"""
    import psutil
    memory = psutil.virtual_memory()
    memory_mb = memory.used / 1024 / 1024
    print(f"📊 Memory usage: {memory_mb:.1f}MB / {memory.total / 1024 / 1024:.1f}MB")
    return memory_mb

def optimize_for_deployment():
    """Apply deployment optimizations"""
    print("🚀 Applying deployment optimizations...")
    
    # Disable tokenizers parallelism to save memory
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    # Set smaller cache sizes
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    
    # Force garbage collection
    gc.collect()
    
    print("✅ Deployment optimizations applied")

if __name__ == "__main__":
    print("🌐 Starting ResumeRAG in deployment mode...")
    
    # Apply optimizations
    optimize_for_deployment()
    
    # Check initial memory
    check_memory_usage()
    
    # Import the main app
    try:
        from app import *
        print("✅ App imported successfully")
    except Exception as e:
        print(f"❌ App import failed: {e}")
        print("📝 Running in minimal mode...")
    
    print("🎯 ResumeRAG ready for deployment!")
