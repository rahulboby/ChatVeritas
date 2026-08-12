"""ChatVeritas home page and Streamlit multipage entry point."""

import streamlit as st

st.set_page_config(page_title="ChatVeritas", page_icon="💬", layout="wide")

st.title("ChatVeritas")
st.subheader("A document-grounded RAG assistant")
st.markdown(
    """
Use the sidebar to choose a page:

- **Offline Chat** runs either the local Qwen model with an optional LoRA adapter
  or an LM Studio server on your machine.
- **Cloud Chat** uses Groq for generation while keeping the same retrieval flow.
- **Architecture** renders the current project design directly from its Markdown
  document.

Before asking questions, place `.txt` documents in `data/raw/` and run
`python scripts/ingest.py` to build the vector store.
"""
)
