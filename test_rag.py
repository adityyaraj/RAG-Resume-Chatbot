#!/usr/bin/env python3
"""
Test script for RAG functionality in the resume chatbot
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_rag_functionality():
    """Test the RAG components"""
    print("🔍 Testing RAG functionality...")
    
    try:
        # Test imports
        print("1. Testing imports...")
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_community.vectorstores import FAISS
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain.schema import Document as LangchainDocument
        print("   ✅ All imports successful")
        
        # Test embeddings model
        print("2. Testing embeddings model...")
        embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        print("   ✅ Embeddings model loaded")
        
        # Test text splitting
        print("3. Testing text splitting...")
        sample_text = """
        John Doe
        Software Engineer
        
        Experience:
        - 5 years of Python development
        - Expert in machine learning
        - Led a team of 10 developers
        
        Skills:
        - Python, JavaScript, SQL
        - TensorFlow, PyTorch
        - AWS, Docker, Kubernetes
        
        Education:
        - Bachelor's in Computer Science
        - Master's in Data Science
        """
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = text_splitter.split_text(sample_text)
        print(f"   ✅ Created {len(chunks)} text chunks")
        
        # Test document creation
        print("4. Testing document creation...")
        documents = []
        for i, chunk in enumerate(chunks):
            doc = LangchainDocument(
                page_content=chunk,
                metadata={
                    "source": "test_resume.txt",
                    "chunk_id": i,
                    "total_chunks": len(chunks)
                }
            )
            documents.append(doc)
        print(f"   ✅ Created {len(documents)} documents")
        
        # Test vector store creation
        print("5. Testing vector store creation...")
        vectorstore = FAISS.from_documents(documents, embeddings_model)
        print("   ✅ Vector store created successfully")
        
        # Test similarity search
        print("6. Testing similarity search...")
        query = "What programming languages does the candidate know?"
        relevant_docs = vectorstore.similarity_search(query, k=2)
        print(f"   ✅ Found {len(relevant_docs)} relevant documents for query")
        
        for i, doc in enumerate(relevant_docs):
            print(f"      Document {i+1}: {doc.page_content[:100]}...")
        
        print("\n🎉 All RAG functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_rag_functionality()
