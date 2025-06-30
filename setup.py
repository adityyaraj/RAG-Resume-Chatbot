#!/usr/bin/env python3
"""
Setup script for RAG Resume Reviewer
"""

import subprocess
import sys
import os

def run_command(command):
    """Run a command and return the result"""
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr

def main():
    print("🚀 Setting up RAG Resume Reviewer...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    print("✅ Python version check passed")
    
    # Install requirements
    print("📦 Installing requirements...")
    success, output = run_command("pip install -r requirements.txt")
    if not success:
        print(f"❌ Failed to install requirements: {output}")
        sys.exit(1)
    
    print("✅ Requirements installed successfully")
    
    # Check if .env exists
    if not os.path.exists(".env"):
        print("⚠️  .env file not found")
        print("📝 Creating .env from template...")
        
        if os.path.exists(".env.example"):
            import shutil
            shutil.copy(".env.example", ".env")
            print("✅ .env file created from template")
            print("🔧 Please edit .env file and add your GOOGLE_API_KEY")
        else:
            print("❌ .env.example not found")
    else:
        print("✅ .env file exists")
    
    # Test imports
    print("🧪 Testing imports...")
    try:
        import chainlit
        import langchain
        import faiss
        from sentence_transformers import SentenceTransformer
        print("✅ All imports successful")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Try running: pip install -r requirements.txt")
    
    print("\n🎉 Setup complete!")
    print("\n📋 Next steps:")
    print("1. Edit .env file and add your GOOGLE_API_KEY")
    print("2. Run: chainlit run app.py")
    print("3. Open http://localhost:8000 in your browser")
    print("\n📚 For help, see README.md")

if __name__ == "__main__":
    main()
