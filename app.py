"""
Streamlit UI for Customer Support Chatbot

This is a customer-facing interface for the RAG-powered chatbot.
Customers can ask questions and get AI-powered answers with source citations.

Author: Sushanth Reddy
Date: October 2025
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.router.orchestrator import LangGraphOrchestrator
from src.config import model_config


# Page configuration
st.set_page_config(
    page_title="Customer Support",
    page_icon="👟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Dark theme */
    .stApp {
        background-color: #1a1a1a;
    }
    
    .main-header {
        font-size: 3rem;
        color: #4FC3F7;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #B0BEC5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #263238;
        border-left: 4px solid #4FC3F7;
        color: #E0F7FA;
    }
    .assistant-message {
        background-color: #1E3A2E;
        border-left: 4px solid #66BB6A;
        color: #E8F5E9;
    }
    .source-box {
        background-color: #2A2A2A;
        border-left: 3px solid #FF9800;
        padding: 1rem;
        margin-top: 1rem;
        border-radius: 0.3rem;
        font-size: 0.95rem;
        color: #FFE082;
    }
    .source-item {
        background-color: #333333;
        padding: 0.6rem;
        margin: 0.5rem 0;
        border-radius: 0.3rem;
        border-left: 2px solid #FF9800;
        color: #FFE082;
        font-weight: 500;
    }
    .metadata-text {
        font-size: 0.85rem;
        color: #90A4AE;
        margin-top: 0.5rem;
        font-weight: 400;
    }
    .suggestion-chip {
        display: inline-block;
        padding: 0.5rem 1rem;
        margin: 0.3rem;
        background-color: #2E3B4E;
        color: #81D4FA;
        border-radius: 20px;
        cursor: pointer;
        border: 1px solid #4FC3F7;
        transition: background-color 0.3s;
    }
    .suggestion-chip:hover {
        background-color: #37474F;
    }
    .stats-card {
        background-color: #263238;
        color: #B0BEC5;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 3px solid #4FC3F7;
    }
    /* Style text inputs and buttons */
    .stTextInput > div > div > input {
        background-color: #2A2A2A;
        color: #E0E0E0;
        border: 1px solid #424242;
    }
    .stButton > button {
        background-color: #1E88E5;
        color: white;
    }
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #212121;
    }
    section[data-testid="stSidebar"] * {
        color: #B0BEC5 !important;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'initialized' not in st.session_state:
    st.session_state.initialized = False


def initialize_system():
    """Initialize the RAG system."""
    if not st.session_state.initialized:
        with st.spinner("🔄 Initializing AI Assistant..."):
            try:
                st.session_state.orchestrator = LangGraphOrchestrator(
                    use_real_retrieval=True,
                    use_gemma_generation=True,
                    auto_load_docs=True
                )
                st.session_state.initialized = True
                return True
            except Exception as e:
                st.error(f"Failed to initialize system: {e}")
                return False
    return True


def format_response(result):
    """Format the response for display."""
    response_text = result.get('response', 'No response available')
    routing = result.get('routing_decision', 'Unknown')
    confidence = result.get('confidence', 'N/A')
    
    # Extract metadata if available
    metadata = {
        'routing': routing,
        'confidence': confidence,
        'num_results': result.get('retrieval_metadata', {}).get('num_results', 0),
        'latency': result.get('retrieval_metadata', {}).get('latency_ms', 0)
    }
    
    return response_text, metadata


def display_message(role, content, metadata=None):
    """Display a chat message with improved formatting."""
    import re
    
    css_class = "user-message" if role == "user" else "assistant-message"
    icon = "👤" if role == "user" else "🤖"
    
    # Parse content to separate answer from sources
    if role == "assistant" and "📚 Sources" in content:
        parts = content.split("📚 Sources")
        answer_part = parts[0].replace("📝 Generated Answer:", "").strip()
        answer_part = re.sub(r'={40,}', '', answer_part).strip()
        
        # Extract sources
        sources_part = parts[1] if len(parts) > 1 else ""
        source_lines = re.findall(r'\[(\d+)\]\s*([^\(]+)\s*\(Relevance:\s*([0-9.]+)\)', sources_part)
        
        # Display answer
        st.markdown(f"""
        <div class="chat-message {css_class}">
            <div><strong>{icon} {role.title()}</strong></div>
            <div style="margin-top: 1rem; font-size: 1.05rem; line-height: 1.6;">{answer_part}</div>
        """, unsafe_allow_html=True)
        
        # Display sources in a clean box
        if source_lines:
            st.markdown("""
            <div class="source-box">
                <strong>📚 Sources:</strong>
            """, unsafe_allow_html=True)
            
            for rank, title, score in source_lines:
                st.markdown(f"""
                <div class="source-item">
                    [{rank}] {title.strip()} • Relevance: {float(score):.2f}
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Display metadata
        if metadata and metadata.get('routing') == 'retrieval':
            st.markdown(f"""
            <div class="metadata-text">
                ⚡ Response time: {metadata.get('latency', 0):.0f}ms 
                | 🎯 Confidence: {metadata.get('confidence', 'N/A')}
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        # Simple message display for user or non-source messages
        st.markdown(f"""
        <div class="chat-message {css_class}">
            <div><strong>{icon} {role.title()}</strong></div>
            <div style="margin-top: 0.5rem;">{content}</div>
        </div>
        """, unsafe_allow_html=True)


def main():
    """Main application."""
    
    # Header
    st.markdown('<h1 class="main-header">👟 Customer Support</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ask me anything about products, policies, or running advice!</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("""
        This AI-powered chatbot helps you with:
        - Product information
        - Return & shipping policies
        - Running shoe recommendations
        - Technical specifications
        - General inquiries
        """)
        
        st.divider()
        
        # System information
        st.header("⚙️ System Info")
        if st.session_state.initialized:
            st.success("✅ System Ready")
            
            # Display current model
            model_type = model_config.MODEL_TYPE
            model_icon = "💻" if model_type == "gemma" else "☁️"
            st.info(f"{model_icon} Using: {model_type.title()}")
            
            if model_type == "gemini":
                st.caption("🚀 Fast API-based generation")
            else:
                st.caption("🔒 Private local generation")
        else:
            st.warning("⏳ Not initialized")
        
        st.divider()
        
        # Statistics
        if st.session_state.chat_history:
            st.header("📊 Session Stats")
            st.metric("Messages", len(st.session_state.chat_history))
            
            # Count queries by type
            user_messages = [m for m in st.session_state.chat_history if m['role'] == 'user']
            st.metric("Questions Asked", len(user_messages))
        
        st.divider()
        
        # Clear chat button
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
        
        st.divider()
        st.caption("Powered by Brooks RAG System")
        st.caption(f"Model: {model_config.MODEL_TYPE.title()}")
    
    # Initialize system
    if not initialize_system():
        st.error("❌ Failed to initialize the system. Please check the configuration.")
        return
    
    # Suggested questions
    if not st.session_state.chat_history:
        st.subheader("💡 Try asking:")
        
        suggestions = [
            "What is your return policy?",
            "How do I choose the right running shoe?",
            "Tell me about your shoes",
            "What is GTS in running shoes?",
            "Do you ship internationally?",
            "What shoes are good for overpronation?"
        ]
        
        cols = st.columns(3)
        for idx, suggestion in enumerate(suggestions):
            with cols[idx % 3]:
                if st.button(suggestion, key=f"suggestion_{idx}", use_container_width=True):
                    # Add to chat and process
                    st.session_state.chat_history.append({
                        'role': 'user',
                        'content': suggestion,
                        'timestamp': datetime.now()
                    })
                    st.rerun()
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            display_message(
                message['role'],
                message['content'],
                message.get('metadata')
            )
    
    # Process pending user message
    if st.session_state.chat_history and st.session_state.chat_history[-1]['role'] == 'user':
        user_query = st.session_state.chat_history[-1]['content']
        
        with st.spinner("🤔 Thinking..."):
            try:
                # Get response from orchestrator
                result = st.session_state.orchestrator.route_query(user_query)
                
                # Format response
                response_text, metadata = format_response(result)
                
                # Add assistant response to history
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': response_text,
                    'metadata': metadata,
                    'timestamp': datetime.now()
                })
                
                st.rerun()
                
            except Exception as e:
                st.error(f"Error processing query: {e}")
                # Remove the user message if processing failed
                st.session_state.chat_history.pop()
    
    # Chat input
    st.divider()
    
    # Create a form for the chat input
    with st.form(key='chat_form', clear_on_submit=True):
        cols = st.columns([6, 1])
        with cols[0]:
            user_input = st.text_input(
                "Your question:",
                placeholder="Ask me anything about our products...",
                label_visibility="collapsed"
            )
        with cols[1]:
            submit_button = st.form_submit_button("Send 📤", use_container_width=True)
        
        if submit_button and user_input:
            # Add user message to history
            st.session_state.chat_history.append({
                'role': 'user',
                'content': user_input,
                'timestamp': datetime.now()
            })
            st.rerun()
    
    # Footer
    st.divider()
    st.caption("💡 Tip: Ask specific questions for more accurate answers. This chatbot uses AI and may occasionally provide incomplete information.")


if __name__ == "__main__":
    main()
