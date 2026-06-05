import streamlit as st
from ui.components import show_chat_message, show_index_status, clear_chat_button, clear_index_button

st.set_page_config(page_title="Research Assistant", layout="wide")

st.markdown("""
    <style>
        section[data-testid="stSidebar"] { min-width: 280px; max-width: 320px; }
    </style>
""", unsafe_allow_html=True)

st.title("📄 RAG Research Assistant")

# Session state defaults
if "messages" not in st.session_state:
    st.session_state.messages = []
if "index_built" not in st.session_state:
    st.session_state.index_built = False

# Sidebar
with st.sidebar:
    uploaded_files = st.file_uploader(
        "Upload PDFs", type="pdf", accept_multiple_files=True
    )
    if st.button("Build Index") and uploaded_files:
        with st.spinner("Building index..."):
            pass  # Person A: replace with build_index(uploaded_files)
            st.session_state.index_built = True
            st.session_state.uploaded_names = [f.name for f in uploaded_files]

    show_index_status(st.session_state.index_built)
    clear_chat_button()
    clear_index_button()

# Welcome screen
if not st.session_state.messages:
    st.markdown("""
    ### 👋 Welcome!
    1. Upload one or more PDFs in the sidebar
    2. Click **Build Index**
    3. Ask any question about your documents
    """)

# Chat history
for msg in st.session_state.messages:
    show_chat_message(msg["role"], msg["content"], msg.get("sources"))

# Mock ask — swap for real ask(query) on Day 3
def mock_ask(query: str) -> dict:
    names = st.session_state.get("uploaded_names", ["document.pdf"])
    return {
        "answer": f"Mock answer for: '{query}'. This will be replaced by the real RAG pipeline.",
        "sources": [{"doc": names[0], "page": 1}]
    }

# Chat input
if query := st.chat_input("Ask a question about your documents..."):
    show_chat_message("user", query)
    st.session_state.messages.append({"role": "user", "content": query, "sources": None})

    if not st.session_state.index_built:
        st.warning("Please upload PDFs and click Build Index first.")
    else:
        with st.spinner("Thinking..."):
            response = mock_ask(query)  # Day 3: swap to response = ask(query)
        show_chat_message("assistant", response["answer"], response["sources"])
        st.session_state.messages.append({
            "role": "assistant",
            "content": response["answer"],
            "sources": response["sources"]
        })