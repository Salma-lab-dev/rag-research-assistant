import streamlit as st

def show_chat_message(role: str, content: str, sources: list = None):
    with st.chat_message(role):
        st.write(content)
        if sources:
            with st.expander(" Sources"):
                for s in sources:
                    st.markdown(f"**{s['source']}** — page {s['page']}")
                    if s.get("snippet"):
                        st.caption(s["snippet"])

def show_index_status(built: bool):
    """Show whether the index is ready or not."""
    if built:
        st.sidebar.success("Index ready")
    else:
        st.sidebar.info("Upload PDFs and click Build Index")

def clear_chat_button():
    """Button to reset the conversation."""
    if st.sidebar.button(" Clear chat"):
        st.session_state.messages = []
        st.rerun()

def clear_index_button():
    """Button to wipe the knowledge base."""
    if st.sidebar.button(" Clear knowledge base"):
        st.session_state.index_built = False
        st.session_state.messages = []
        st.rerun()