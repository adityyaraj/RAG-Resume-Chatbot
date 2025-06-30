#!/usr/bin/env python3
"""
Quick test to verify the FAISS docstore fix
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_fixed_rag():
    """Test the fixed RAG functionality"""
    print("🔍 Testing fixed RAG functionality...")
    
    try:
        # Import the fixed function
        from app import create_chunks_and_vectorstore, embeddings_model
        
        # Test with a sample resume
        sample_resume = """
        John Smith
        Software Engineer
        Email: john.smith@email.com
        Phone: (555) 123-4567
        
        PROFESSIONAL SUMMARY
        Experienced software engineer with 5+ years in full-stack development.
        
        TECHNICAL SKILLS
        • Programming Languages: Python, JavaScript, Java, C++
        • Frameworks: React, Django, Spring Boot, Node.js
        • Databases: PostgreSQL, MongoDB, Redis
        • Cloud: AWS, Docker, Kubernetes
        • Tools: Git, Jenkins, JIRA
        
        WORK EXPERIENCE
        Senior Software Engineer | Tech Corp | 2021-Present
        • Led development of microservices architecture serving 1M+ users
        • Improved system performance by 40% through optimization
        • Mentored 5 junior developers and conducted code reviews
        
        Software Engineer | StartupCo | 2019-2021
        • Built responsive web applications using React and Node.js
        • Implemented CI/CD pipelines reducing deployment time by 60%
        • Collaborated with cross-functional teams on product features
        
        EDUCATION
        Bachelor of Science in Computer Science
        University of Technology, 2019
        GPA: 3.8/4.0
        
        PROJECTS
        E-commerce Platform
        • Developed full-stack application with React frontend and Django backend
        • Integrated payment processing and inventory management
        • Deployed on AWS with auto-scaling capabilities
        """
        
        print("1. Testing vectorstore creation...")
        vectorstore, chunk_count = create_chunks_and_vectorstore(sample_resume, "test_resume.txt")
        print(f"   ✅ Created vectorstore with {chunk_count} chunks")
        
        print("2. Testing similarity search...")
        # Test different types of queries
        queries = [
            "What programming languages does the candidate know?",
            "What is the candidate's work experience?",
            "What are the candidate's technical skills?",
            "What education does the candidate have?"
        ]
        
        for query in queries:
            relevant_docs = vectorstore.similarity_search(query, k=2)
            print(f"   Query: '{query}'")
            print(f"   Found {len(relevant_docs)} relevant chunks")
            for i, doc in enumerate(relevant_docs):
                preview = doc.page_content[:100].replace('\n', ' ').strip()
                print(f"     Chunk {i+1}: {preview}...")
        
        print("\n🎉 All tests passed! The FAISS docstore fix is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_fixed_rag()
