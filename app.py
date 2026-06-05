import streamlit as st
from ui.components import show_chat_message, show_index_status, clear_chat_button, clear_index_button
from rag.ingestion import load_and_chunk
from rag.embeddings import build_index
from rag.retriever import build_qa_chain, ask

st.set_page_config(page_title="RAG Research Assistant", layout="wide", page_icon="📄")

st.markdown("""
    <style>
        section[data-testid="stSidebar"] { min-width: 280px; max-width: 320px; }
        .stChatMessage { padding: 0.5rem 0; }
        .block-container { padding-top: 2rem; }
    </style>
""", unsafe_allow_html=True)

st.title("📄 RAG Research Assistant")

# ── Session state defaults ───────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "index_built" not in st.session_state:
    st.session_state.index_built = False

if "uploaded_names" not in st.session_state:
    st.session_state.uploaded_names = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None
# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Knowledge Base")

    uploaded_files = st.file_uploader(
        "Upload PDFs", type="pdf", accept_multiple_files=True
    )

    if st.button("⚡ Build Index", use_container_width=True):
        if not uploaded_files:
            st.error("Please upload at least one PDF first.")
        else:
            with st.spinner("Building index... this may take a moment."):
                chunks = load_and_chunk(uploaded_files)
                db = build_index(chunks)
                st.session_state.retriever = build_qa_chain(db)
                st.session_state.index_built = True
                st.session_state.uploaded_names = [f.name for f in uploaded_files]
    show_index_status(st.session_state.index_built)

    # Show indexed file names
    if st.session_state.uploaded_names:
        st.markdown("**Indexed files:**")
        for name in st.session_state.uploaded_names:
            st.markdown(f"- 📄 {name}")

    st.divider()
    clear_chat_button()
    clear_index_button()

# ── Main area ────────────────────────────────────────────────────────────────

# Welcome screen — only when no messages yet
if not st.session_state.messages:
    st.markdown("""
    ### 👋 Welcome!
    **Get started in 3 steps:**
    1. Upload one or more PDFs in the sidebar
    2. Click **⚡ Build Index**
    3. Ask any question about your documents
    ---
    """)

# Chat history
for msg in st.session_state.messages:
    show_chat_message(msg["role"], msg["content"], msg.get("sources"))

# Chat input
if query := st.chat_input("Ask a question about your documents..."):

    # Guard: empty query
    if not query.strip():
        st.warning("Please type a question first.")
        st.stop()

    # Guard: index not built
    if not st.session_state.index_built:
        st.warning("⚠️ Please upload PDFs and click **Build Index** first.")
        st.stop()

    # Show user message
    show_chat_message("user", query)
    st.session_state.messages.append({"role": "user", "content": query, "sources": None})

    # Get answer
    with st.spinner("Thinking..."):
        # ✅ Day 3: swap this line:
        # from rag.retriever import ask
        # response = ask(query)
        response = ask(query, st.session_state.retriever)

    # Guard: empty answer
    if not response.get("answer"):
        st.error("No answer returned. Try rephrasing your question.")
        st.stop()

    # Show assistant message
    show_chat_message("assistant", response["answer"], response.get("sources"))
    st.session_state.messages.append({
        "role": "assistant",
        "content": response["answer"],
        "sources": response.get("sources", [])
    })