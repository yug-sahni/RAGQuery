import os
import torch

# Fix PyTorch-Streamlit compatibility issue
torch.classes.__path__ = []

# Disable Streamlit file watcher to prevent torch conflicts
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'

import streamlit as st
import tempfile
from typing import List, Dict, Optional
from document_parser import DocumentParser
from qa_system import QASystem
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", message=".*max_length.*")
warnings.filterwarnings("ignore", message=".*truncation.*")

def main():
    st.set_page_config(
        page_title="Free Document Q&A System",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("📄 Free Document Q&A System")
    st.markdown("Upload documents and ask questions using **completely free AI services**!")
    
    # Initialize session state
    if 'qa_system' not in st.session_state:
        st.session_state.qa_system = None
        st.session_state.documents_processed = False
        st.session_state.processing_status = []
        st.session_state.chat_history = []
        st.session_state.current_question = ""
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        
        use_ollama = st.checkbox("Use Ollama (Recommended)", value=True)
        st.info("💡 Ollama provides better quality responses")
        
        if st.button("Initialize System"):
            with st.spinner("Initializing AI system..."):
                try:
                    st.session_state.qa_system = QASystem(use_ollama=use_ollama)
                    system_info = st.session_state.qa_system.get_system_info()
                    st.success(f"✅ System initialized with {system_info['llm_type']}!")
                    
                    with st.expander("System Details"):
                        st.json(system_info)
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("💡 If Ollama fails, the system will use HuggingFace automatically")
        
        # System status
        if st.session_state.qa_system:
            st.success("✅ System Ready")
            try:
                system_info = st.session_state.qa_system.get_system_info()
                st.write(f"**LLM:** {system_info['llm_type']}")
                if st.session_state.documents_processed:
                    st.write(f"**Documents:** {system_info['vector_db_stats']['unique_documents']}")
                    st.write(f"**Chunks:** {system_info['vector_db_stats']['total_chunks']}")
            except Exception as e:
                st.write(f"**Status:** System initialized")
        
        st.markdown("---")
        st.markdown("### 📖 Usage Tips")
        st.markdown("""
        - **Date queries**: "What was done on 6-Sept?"
        - **Technical queries**: "Drilling procedures used"
        - **Summary queries**: "Summarize all activities"
        - **Specific details**: "WBM density values"
        """)
    
    # Document upload section
    st.header("📁 Upload Documents")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            "Choose files (PDF, DOCX, TXT)",
            accept_multiple_files=True,
            type=['pdf', 'docx', 'txt'],
            help="Upload multiple documents for comprehensive Q&A"
        )
    
    with col2:
        st.info("📊 **Supported Data Types:**\n- Regular text\n- Tables (PDF/DOCX)\n- Mixed content\n- Technical reports\n- Date-based logs")
    
    if uploaded_files and st.session_state.qa_system:
        if st.button("🚀 Process Documents", type="primary"):
            st.session_state.processing_status = []
            
            with st.spinner("Processing documents..."):
                parser = DocumentParser()
                progress_bar = st.progress(0)
                status_placeholder = st.empty()
                
                total_chunks = 0
                
                for i, uploaded_file in enumerate(uploaded_files):
                    # Update progress
                    progress = i / len(uploaded_files)
                    progress_bar.progress(progress)
                    status_placeholder.write(f"Processing: {uploaded_file.name}")
                    
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    try:
                        # Parse document
                        document = parser.parse_document(tmp_file_path)
                        
                        # Process into chunks and embeddings
                        processed_chunks = st.session_state.qa_system.text_processor.process_document(document)
                        
                        # Add to both vector database and hybrid search
                        st.session_state.qa_system.add_documents(processed_chunks)
                        
                        total_chunks += len(processed_chunks)
                        status = f"✅ {uploaded_file.name} - {len(processed_chunks)} chunks"
                        st.session_state.processing_status.append(status)
                        
                    except Exception as e:
                        status = f"❌ {uploaded_file.name} - Error: {str(e)}"
                        st.session_state.processing_status.append(status)
                        
                    finally:
                        # Clean up temporary file
                        os.unlink(tmp_file_path)
                
                st.session_state.documents_processed = True
                progress_bar.progress(1.0)
                status_placeholder.empty()
            
            # Show processing results
            st.subheader("📋 Processing Results")
            for status in st.session_state.processing_status:
                if "✅" in status:
                    st.success(status)
                else:
                    st.error(status)
            
            # Show database stats
            if st.session_state.documents_processed:
                try:
                    stats = st.session_state.qa_system.vector_db.get_stats()
                    st.balloons()
                    st.success(f"🎉 Successfully processed {stats['unique_documents']} documents into {stats['total_chunks']} searchable chunks!")
                except Exception as e:
                    st.success("🎉 Documents processed successfully!")
    
    # Q&A section
    if st.session_state.documents_processed:
        st.header("❓ Ask Questions")
        
        # Create two columns for better layout
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Question input - use current_question from session state
            question = st.text_input(
                "Enter your question:",
                value=st.session_state.current_question,
                placeholder="e.g., What was done on 6-Sept? or What drilling procedures were used?",
                key="question_input"
            )
            
            # Update current_question when user types
            if question != st.session_state.current_question:
                st.session_state.current_question = question
        
        with col2:
            # Quick question buttons with proper callback functions
            st.write("**Quick Questions:**")
            
            def set_recent_question():
                st.session_state.current_question = "What activities were performed recently?"
            
            def set_technical_question():
                st.session_state.current_question = "What technical procedures were mentioned?"
            
            def set_date_question():
                st.session_state.current_question = "What was done on 6-Sept?"
            
            st.button("📅 Recent Activities", key="recent", on_click=set_recent_question)
            st.button("🔧 Technical Details", key="technical", on_click=set_technical_question)
            st.button("📆 Date Query", key="date_query", on_click=set_date_question)
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            col1, col2 = st.columns(2)
            
            with col1:
                top_k = st.slider("Number of relevant chunks to retrieve", 1, 10, 3)
                search_method = st.selectbox(
                    "Search Method",
                    ["Auto (Recommended)", "Semantic Only", "Hybrid Only"],
                    help="Auto uses hybrid search for date queries, semantic for others"
                )
            
            with col2:
                response_length = st.selectbox(
                    "Response Length",
                    ["Short (200 tokens)", "Medium (400 tokens)", "Long (800 tokens)"],
                    index=1
                )
                show_sources = st.checkbox("Show detailed sources", value=True)
        
        # Map response length to token counts
        token_map = {
            "Short (200 tokens)": 200,
            "Medium (400 tokens)": 400,
            "Long (800 tokens)": 800
        }
        max_tokens = token_map[response_length]
        
        # Use the current question from session state
        current_question = st.session_state.current_question
        
        if current_question and st.button("🔍 Get Answer", type="primary"):
            with st.spinner("Searching for answer..."):
                try:
                    # Choose the appropriate method based on user selection
                    if search_method == "Hybrid Only":
                        result = st.session_state.qa_system.answer_question_enhanced(current_question, top_k=top_k)
                    elif search_method == "Semantic Only":
                        result = st.session_state.qa_system.answer_question(current_question, top_k=top_k)
                    else:  # Auto
                        result = st.session_state.qa_system.answer_question_enhanced(current_question, top_k=top_k)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        'question': current_question,
                        'answer': result['answer'],
                        'sources': result['sources'],
                        'search_method': result.get('search_method', 'semantic')
                    })
                    
                    # Display answer
                    st.subheader("💡 Answer:")
                    st.write(result['answer'])
                    
                    # Display metadata
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.caption(f"🤖 Generated using: {result['llm_used']}")
                    with col2:
                        st.caption(f"🔍 Search method: {result.get('search_method', 'semantic')}")
                    with col3:
                        st.caption(f"📊 Sources found: {len(result['sources'])}")
                    
                    # Display sources
                    if show_sources and result['sources']:
                        st.subheader("📚 Sources:")
                        for i, source in enumerate(result['sources']):
                            relevance_color = "🟢" if source['relevance_score'] > 0.7 else "🟡" if source['relevance_score'] > 0.5 else "🔴"
                            
                            with st.expander(f"{relevance_color} Source {i+1}: {source['filename']} (Relevance: {source['relevance_score']:.3f})"):
                                st.write(result['context_used'][i])
                                st.caption(f"Chunk index: {source['chunk_index']}")
                
                except Exception as e:
                    st.error(f"❌ Error processing question: {str(e)}")
                    st.info("💡 Try rephrasing your question or check if documents are properly processed")
        
        # Chat history
        if st.session_state.chat_history:
            st.subheader("💬 Chat History")
            with st.expander(f"View {len(st.session_state.chat_history)} previous questions"):
                for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):  # Show last 5
                    st.write(f"**Q{len(st.session_state.chat_history)-i}:** {chat['question']}")
                    st.write(f"**A:** {chat['answer'][:200]}{'...' if len(chat['answer']) > 200 else ''}")
                    st.caption(f"Search: {chat['search_method']} | Sources: {len(chat['sources'])}")
                    st.markdown("---")
            
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()
        
        # Database management
        st.subheader("💾 Database Management")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Save Database"):
                try:
                    os.makedirs("data", exist_ok=True)
                    st.session_state.qa_system.vector_db.save("data/vector_db.pkl")
                    st.success("✅ Database saved to data/vector_db.pkl")
                except Exception as e:
                    st.error(f"❌ Error saving database: {str(e)}")
        
        with col2:
            if st.button("📂 Load Database"):
                try:
                    st.session_state.qa_system.vector_db.load("data/vector_db.pkl")
                    st.success("✅ Database loaded from data/vector_db.pkl")
                    st.session_state.documents_processed = True
                    st.rerun()
                except Exception as e:
                    st.error("❌ No saved database found or error loading!")
        
        with col3:
            if st.button("🔄 Reset System"):
                # Reset session state
                for key in ['qa_system', 'documents_processed', 'processing_status', 'chat_history', 'current_question']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.success("✅ System reset! Please reinitialize.")
                st.rerun()
    
    # Footer with instructions
    st.markdown("---")
    st.markdown("""
    ### 🚀 Getting Started
    1. **Initialize System** - Click the button in the sidebar
    2. **Upload Documents** - Add your PDF, DOCX, or TXT files
    3. **Process Documents** - Click to index your documents
    4. **Ask Questions** - Use natural language to query your documents
    
    ### 💡 Tips for Better Results
    - **Date queries**: Use formats like "6-Sept", "September 6", or "what happened on [date]"
    - **Technical queries**: Use specific terms from your documents
    - **Summary queries**: Ask for overviews or summaries of topics
    - **Specific details**: Query for exact values, procedures, or measurements
    
    ### 🔧 Troubleshooting
    - If responses seem cut off, try increasing response length in Advanced Options
    - For date-related queries, ensure your documents contain clear date formats
    - Use "Hybrid Only" search method for better date-based query results
    - If initialization fails, the system will automatically fall back to HuggingFace
    
    ### 🆓 Free AI Services Used
    - **Ollama** (Primary): Local LLM for high-quality responses
    - **HuggingFace** (Fallback): Cloud-based transformer models
    - **Sentence Transformers**: Free embedding models for semantic search
    - **FAISS**: Efficient vector similarity search
    """)

if __name__ == "__main__":
    main()
