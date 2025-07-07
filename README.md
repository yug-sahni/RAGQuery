# 📄 Free Document Q&A System

A powerful, completely free document question-answering system that lets you upload documents and ask questions in natural language. Built with advanced AI techniques including RAG (Retrieval-Augmented Generation) and hybrid search.

## 🚀 Features

- **📁 Multi-format Support**: PDF, DOCX, and TXT files
- **🔍 Smart Search**: Combines semantic search with keyword matching
- **📅 Date-aware Queries**: Specialized handling for date-based questions
- **🆓 Completely Free**: Uses only free AI services (Ollama + HuggingFace)
- **🌐 Web Interface**: User-friendly Streamlit web app
- **📊 Table Extraction**: Handles both text and tabular data
- **💾 Persistent Storage**: Save and load your document database
- **🔄 Chat History**: Track previous questions and answers
- **📚 Source Citations**: See exactly where answers come from

## 🛠️ Technologies Used

- **AI/ML**: Sentence Transformers, FAISS, HuggingFace Transformers, Ollama
- **Document Processing**: PyPDF2, python-docx, tabula-py
- **Web Framework**: Streamlit
- **Data Processing**: NumPy, Pandas
- **Vector Search**: FAISS (Facebook AI Similarity Search)

## 📋 Prerequisites

- Python 3.8 or higher
- 4GB+ RAM (for AI models)
- Optional: Ollama for better AI responses

## 🔧 Installation

1. **Clone the repository**
```bash git clone <your-repo-url>
cd document-qa-system
```

2. **Install dependencies**
```bash pip install -r requirements.txt
```

3. **Optional: Install Ollama** (for better AI responses)
- Visit [ollama.ai](https://ollama.ai) and download for your OS
- Run: `ollama serve`
- Download a model: `ollama pull llama3.2:3b`

## 🚀 How to Run

### Windows/Mac/Linux
```bash streamlit run app.py
```

The web interface will open at `http://localhost:8501`

### Alternative (if Streamlit conflicts occur)
```bash streamlit run app.py --server.fileWatcherType none
```

## 📖 Usage Example

1. **Initialize System**: Click "Initialize System" in the sidebar
2. **Upload Documents**: Add your PDF, DOCX, or TXT files
3. **Process Documents**: Click "🚀 Process Documents"
4. **Ask Questions**: 
   - Date queries: "What was done on 6-Sept?"
   - Technical queries: "What drilling procedures were used?"
   - Summary queries: "Summarize all activities"

### Example Queries
✅ "What activities were performed on September 6th?"
✅ "What drilling procedures were mentioned?"
✅ "Summarize the key findings"
✅ "What was the WBM density?"
✅ "List all technical specifications"

## 📁 File Structure

document-qa-system/
├── app.py # Main Streamlit web application
├── document_parser.py # PDF/DOCX/TXT parsing with table extraction
├── text_processor.py # Text chunking and embedding generation
├── vector_database.py # FAISS vector storage and search
├── llm_handler.py # Free LLM integration (Ollama/HuggingFace)
├── qa_system.py # Main Q&A orchestration logic
├── hybrid_search.py # Date-aware hybrid search engine
├── requirements.txt # Python dependencies
├── setup.py # Installation and setup script
└── data/ # Directory for saved databases

### File Descriptions

- **`app.py`**: Web interface with file upload, processing, and Q&A
- **`document_parser.py`**: Handles multiple file formats and table extraction
- **`text_processor.py`**: Smart text chunking with date normalization
- **`vector_database.py`**: Vector storage using FAISS for fast search
- **`llm_handler.py`**: Manages Ollama and HuggingFace language models
- **`qa_system.py`**: Orchestrates search and answer generation
- **`hybrid_search.py`**: Specialized search for date-based queries

## 🔧 Configuration Options

- **LLM Choice**: Ollama (recommended) or HuggingFace (fallback)
- **Search Method**: Auto, Semantic Only, or Hybrid Only
- **Response Length**: Short (200), Medium (400), or Long (800 tokens)
- **Chunk Retrieval**: 1-10 relevant chunks per query

## 🐛 Troubleshooting

**Common Issues:**

1. **Ollama not running**: System automatically falls back to HuggingFace
2. **Memory issues**: Reduce batch size or use smaller models
3. **Date queries not working**: Ensure documents contain clear date formats
4. **Truncated responses**: Increase response length in Advanced Options

**Performance Tips:**
- Use Ollama for better quality responses
- Process documents in smaller batches for large files
- Save database to avoid reprocessing documents

## 🆓 Why This is Completely Free

- **Ollama**: Runs locally, no API costs
- **HuggingFace**: Free transformer models
- **Sentence Transformers**: Open-source embeddings
- **FAISS**: Free vector search by Facebook AI
- **Streamlit**: Free web framework

## 📄 License

This project is open source and available under the [MIT License](LICENSE).



