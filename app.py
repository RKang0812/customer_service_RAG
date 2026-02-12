"""
Customer Service RAG System - Streamlit Application

"""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from vector_store import create_vector_store
from llm_client import create_llm_client
from chat_service import create_chat_service
from knowledge_service import create_knowledge_service
from config import UI_TEXT, SUPPORTED_LANGUAGES
from logger_config import setup_logger

logger = setup_logger("streamlit_app")

# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="Customer Service RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# Session State Initialization
# ============================================================================

def init_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "language" not in st.session_state:
        st.session_state.language = "en"
    
    if "services_initialized" not in st.session_state:
        st.session_state.services_initialized = False
    
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    
    if "llm_client" not in st.session_state:
        st.session_state.llm_client = None
    
    if "chat_service" not in st.session_state:
        st.session_state.chat_service = None
    
    if "knowledge_service" not in st.session_state:
        st.session_state.knowledge_service = None

def get_text(key: str) -> str:
    """Get UI text in current language"""
    lang = st.session_state.language
    return UI_TEXT.get(lang, UI_TEXT["en"]).get(key, key)

# ============================================================================
# Service Initialization
# ============================================================================

@st.cache_resource
def initialize_services():
    """Initialize all services (cached) """
    try:
        with st.spinner("Initializing services..."):
            vector_store = create_vector_store()
            llm_client = create_llm_client()
            chat_service = create_chat_service(vector_store, llm_client)
            knowledge_service = create_knowledge_service(vector_store, llm_client)
            
            logger.info("Services initialized successfully")
            return vector_store, llm_client, chat_service, knowledge_service
    
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        st.error(f"Failed to initialize services: {e}")
        st.stop()

# ============================================================================
# Chat Interface
# ============================================================================

def chat_tab():
    """Main chat interface"""
    st.title(get_text("app_title"))
    st.caption(get_text("app_subtitle"))
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources if available
            if message["role"] == "assistant" and "sources" in message:
                if message["sources"]:
                    with st.expander(f"📚 {get_text('sources')} ({len(message['sources'])})"):
                        for source in message["sources"]:
                            st.text(f"• {source}")
                
                # Show confidence
                if "confidence" in message:
                    confidence = message["confidence"]
                    st.caption(f"{get_text('confidence')}: {confidence:.0%}")
    
    # Chat input
    if prompt := st.chat_input(get_text("user_input_placeholder")):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner(get_text("processing")):
                try:
                    # Get conversation history
                    history = [
                        {"role": msg["role"], "content": msg["content"]}
                        for msg in st.session_state.messages[-6:]  # Last 6 messages
                    ]
                    
                    # Process query
                    response = st.session_state.chat_service.process_query(
                        prompt,
                        history=history[:-1]  # Exclude current message
                    )
                    
                    # Display answer 
                    st.markdown(response["answer"])
                    
                    # Display sources
                    if response.get("sources"):
                        with st.expander(f"📚 {get_text('sources')} ({len(response['sources'])})"):
                            for source in response["sources"]:
                                st.text(f"• {source}")
                    
                    # Display confidence
                    confidence = response.get("confidence", 0)
                    st.caption(f"{get_text('confidence')}: {confidence:.0%}")
                    
                    # Add assistant message
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response["answer"],
                        "sources": response.get("sources", []),
                        "confidence": confidence
                    })
                
                except Exception as e:
                    logger.error(f"Error processing query: {e}")
                    error_msg = f"{get_text('error')}: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })

# ============================================================================
# Knowledge Base Management 
# ============================================================================

def knowledge_tab():
    """Knowledge base management interface """
    st.title(f"📚 {get_text('knowledge_tab')}")
    
    # Get collection stats
    try:
        stats = st.session_state.knowledge_service.get_collection_stats()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Documents", stats.get("total_documents", 0))
        with col2:
            st.metric("Total Chunks ", stats.get("total_chunks", 0))
        with col3:
            status = stats.get("status", "unknown")
            st.metric("Status", status)
        
        st.divider()
        
        # Document list
        st.subheader(get_text("doc_list"))
        
        documents = st.session_state.knowledge_service.list_documents()
        
        if documents:
            for doc in documents:
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.text(f"📄 {doc['source']}")
                    st.caption(f"Chunks: {doc['num_chunks']} | Category: {doc.get('category', 'general')}")
                
                with col2:
                    if st.button(get_text("delete_doc"), key=f"del_{doc['source']}"):
                        with st.spinner(get_text("processing")):
                            result = st.session_state.knowledge_service.delete_by_source(doc['source'])
                            
                            if result["success"]:
                                st.success(f"{get_text('success')}: {result['message']}")
                                st.rerun()
                            else:
                                st.error(f"{get_text('error')}: {result['message']}")
        else:
            st.info(get_text("no_docs"))
        
        st.divider()
        
        # Upload new document
        st.subheader(get_text("upload_doc"))
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["txt", "pdf", "docx"],
            help="Upload .txt, .pdf, or .docx files"
        )
        
        if uploaded_file:
            # Save uploaded file
            upload_path = Path("data/uploads") / uploaded_file.name
            upload_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(upload_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            if st.button("Upload and Index"):
                with st.spinner(get_text("processing")):
                    result = st.session_state.knowledge_service.upload_document(
                        str(upload_path)
                    )
                    
                    if result["success"]:
                        st.success(f"{get_text('success')}: {result['message']}")
                        st.info(f"Created {result['num_chunks']} chunks in {result['duration']:.2f}s")
                        st.rerun()
                    else:
                        st.error(f"{get_text('error')}: {result['message']}")
    
    except Exception as e:
        logger.error(f"Error in knowledge tab: {e}")
        st.error(f"{get_text('error')}: {str(e)}")

# ============================================================================
# Settings
# ============================================================================

def settings_tab():
    """Settings interface"""
    st.title(f"⚙️ {get_text('settings_tab')}")
    
    # Language selection
    st.subheader("Language")
    
    language_options = {
        "en": "English",
        "zh": "中文"
    }
    
    selected_lang = st.selectbox(
        "Select Language / 选择语言",
        options=list(language_options.keys()),
        format_func=lambda x: language_options[x],
        index=list(language_options.keys()).index(st.session_state.language)
    )
    
    if selected_lang != st.session_state.language:
        st.session_state.language = selected_lang
        st.rerun()
    
    st.divider()
    
    # System information
    st.subheader("System Information")
    
    try:
        stats = st.session_state.chat_service.get_stats()
        
        st.json(stats)
    
    except Exception as e:
        st.error(f"{get_text('error')}: {str(e)}")
    
    st.divider()
    
    # Clear chat history 
    st.subheader("Clear History")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.success("Chat history cleared")
        st.rerun()

# ============================================================================
# Main Application
# ============================================================================

def main():
    """Main application entry point"""
    init_session_state()
    
    # Initialize services
    if not st.session_state.services_initialized:
        try:
            (
                st.session_state.vector_store,
                st.session_state.llm_client,
                st.session_state.chat_service,
                st.session_state.knowledge_service
            ) = initialize_services()
            
            st.session_state.services_initialized = True
        
        except Exception as e:
            st.error(f"Failed to initialize: {e}")
            st.info("Make sure Qdrant is running: docker-compose up -d")
            st.stop()
    
    # Sidebar
    with st.sidebar:
        st.image("https://image2url.com/r2/default/images/1770897387207-b329c78e-978a-48ac-a1df-f3afb6a84abf.png", width=60)
        
        st.markdown("---")
        
        # Tab selection
        selected_tab = st.radio(
            "Navigation",
            [
                get_text("chat_tab"),
                get_text("knowledge_tab"),
                get_text("settings_tab")
            ],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Quick info
        st.caption("💡 Quick Tips")
        st.caption("• View sources for each answer")
        st.caption("• Manage documents in Knowledge Base")
    
    # Main content area
    if selected_tab == get_text("chat_tab"):
        chat_tab()
    elif selected_tab == get_text("knowledge_tab"):
        knowledge_tab()
    elif selected_tab == get_text("settings_tab"):
        settings_tab()

if __name__ == "__main__":
    main()
