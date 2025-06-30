# 🤖 RAG-Enhanced Resume Reviewer Assistant

A professional AI-powered resume analysis tool built with **Chainlit** and **RAG (Retrieval-Augmented Generation)** technology. This application provides intelligent, context-aware resume feedback by chunking resume content and using vector similarity search for precise analysis.

## ✨ Features

### 🔍 **RAG-Powered Analysis**
- **Text Chunking**: Breaks resumes into 500-character chunks with overlap
- **Vector Search**: Uses FAISS and HuggingFace embeddings for semantic similarity
- **Context-Aware**: Retrieves only relevant sections for each query
- **Intelligent Retrieval**: Different search strategies for different analysis types

### 📄 **Multi-Format Support**
- **PDF Files**: Standard text extraction + OCR for scanned documents
- **Word Documents**: .docx and .doc file support
- **Text Files**: Plain text resume support
- **Manual Input**: Copy-paste resume text directly

### 🎯 **Professional Analysis Types**
- **📋 Complete Resume Review**: Comprehensive analysis of all sections
- **🎯 Skills Analysis**: Focused evaluation of technical and soft skills
- **💼 Experience Review**: Work history optimization and quantification
- **📈 ATS Optimization**: Applicant Tracking System compatibility check
- **✨ Improvement Tips**: Specific, actionable enhancement suggestions

### 🚀 **Advanced Capabilities**
- **OCR Support**: Extract text from scanned/image-based PDFs
- **Smart Chunking**: Preserves context while enabling precise retrieval
- **Action Buttons**: Quick access to common analysis types
- **Error Handling**: Robust file processing with detailed debugging
- **Real-time Feedback**: Progress updates during file processing

## 🛠️ Technology Stack

- **Frontend**: Chainlit (Interactive Chat Interface)
- **LLM**: Google Gemini 2.5 Flash (via LangChain)
- **RAG**: LangChain + FAISS Vector Store
- **Embeddings**: HuggingFace Sentence Transformers
- **Document Processing**: PyPDF2, python-docx, PyMuPDF
- **OCR**: Tesseract + Pillow (optional)

## 📦 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/rag-resume-reviewer.git
cd rag-resume-reviewer
```

### 2. Create Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Setup
Create a `.env` file in the project root:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 5. Optional: Install OCR Support
For scanned PDF support:
```bash
pip install PyMuPDF pillow pytesseract
```

## 🚀 Usage

### Start the Application
```bash
chainlit run app.py
```

### Using the Application
1. **Upload Resume**: Drag & drop or use upload button
2. **Wait for Processing**: File extraction and RAG setup
3. **Get Analysis**: Use action buttons or ask specific questions
4. **Interactive Chat**: Ask follow-up questions for detailed feedback

### Example Queries
- "Give me a complete resume review"
- "How can I improve my skills section?"
- "Is my resume ATS-friendly?"
- "What achievements should I quantify better?"
- "Rate my resume from 1-10"

## 🏗️ Project Structure

```
rag-resume-reviewer/
├── app.py                 # Main Chainlit application
├── requirements.txt       # Python dependencies
├── .env                  # Environment variables (create this)
├── README.md             # Project documentation
├── RAG_README.md         # Technical RAG implementation details
├── test_rag.py          # RAG functionality tests
├── test_fixed_rag.py    # Fixed RAG tests
└── resume_vector_db/    # FAISS vector store (auto-created)
    ├── index.faiss
    └── index.pkl
```

## 🔧 Configuration

### RAG Settings
```python
# Chunking parameters (app.py)
chunk_size=500           # Characters per chunk
chunk_overlap=50         # Overlap between chunks
k=4                     # Number of chunks to retrieve

# Embedding model
model_name="sentence-transformers/all-mpnet-base-v2"
```

### LLM Settings
```python
# Gemini configuration
model="gemini-2.5-flash"
temperature=0.4
```

## 🧪 Testing

### Test RAG Functionality
```bash
python test_fixed_rag.py
```

### Manual Testing
```bash
python -c "from app import create_chunks_and_vectorstore; print('RAG working!')"
```

## 📊 How RAG Works

1. **Document Upload** → Text extraction from PDF/DOCX/TXT
2. **Text Chunking** → Split into semantic chunks with overlap
3. **Vectorization** → Create embeddings using HuggingFace transformers
4. **Storage** → Store in FAISS vector database
5. **Query Processing** → Similarity search for relevant chunks
6. **LLM Analysis** → Generate response based on relevant context only

## 🔍 RAG Benefits

- **Precision**: Only relevant resume sections analyzed
- **Efficiency**: Smaller context windows for faster processing
- **Accuracy**: Reduced hallucinations with focused context
- **Scalability**: Works with resumes of any length
- **Intelligence**: Context-aware analysis per question type

## 🐛 Troubleshooting

### Common Issues

**File Upload Fails**
- Check file format (PDF, DOCX, TXT)
- Ensure file isn't password-protected
- Try smaller file size
- Use manual text input as alternative

**OCR Not Working**
- Install: `pip install PyMuPDF pillow pytesseract`
- Install Tesseract OCR on your system

**Import Errors**
- Update packages: `pip install -U langchain-huggingface`
- Check Python version (3.8+)

**Memory Issues**
- Reduce chunk_size in configuration
- Use smaller embedding models

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Chainlit** for the interactive chat interface
- **LangChain** for RAG implementation framework
- **FAISS** for efficient vector similarity search
- **HuggingFace** for transformer models and embeddings
- **Google Gemini** for advanced language model capabilities

## 📞 Support

For issues and questions:
- Open a GitHub Issue
- Check the troubleshooting section
- Review the RAG_README.md for technical details

---

**Built with ❤️ using RAG technology for intelligent resume analysis**
