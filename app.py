import streamlit as st
from ui.components import show_chat_message, show_index_status, clear_chat_button, clear_index_button

st.set_page_config(page_title="Research Assistant", layout="wide")
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
            pass  # TODO: build_index(uploaded_files)
            st.session_state.index_built = True

    show_index_status(st.session_state.index_built)
    clear_chat_button()
    clear_index_button()

# Chat history
for msg in st.session_state.messages:
    show_chat_message(msg["role"], msg["content"], msg.get("sources"))

# Chat input
if query := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": None, "sources": None})
    show_chat_message("user", query)

    if not st.session_state.index_built:
        st.warning("Please upload PDFs and build the index first.")
    else:
        # TODO: response = ask(query)
        # mock for now:
        response = {"answer": "Mock answer — backend not connected yet.", "sources": []}
        st.session_state.messages.append({"role": "assistant", "content": response["answer"], "sources": response["sources"]})
        show_chat_message("assistant", response["answer"], response["sources"])