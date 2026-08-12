"""Render the maintained architecture Markdown as a Streamlit page."""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import streamlit_mermaid as stm

st.set_page_config(page_title="ChatVeritas Architecture", page_icon="📐", layout="wide")


def render_architecture(markdown: str) -> None:
    """Render Markdown dynamically while retaining Mermaid diagrams."""
    parts = re.split(r"```mermaid\s*\n(.*?)```", markdown, flags=re.DOTALL)
    for index, part in enumerate(parts):
        if not part.strip():
            continue
        if index % 2:
            stm.st_mermaid(part.strip())
        else:
            st.markdown(part)


architecture_path = PROJECT_ROOT / "ChatVeritas_ARCHITECTURE.md"
st.title("ChatVeritas Architecture")
try:
    render_architecture(architecture_path.read_text(encoding="utf-8"))
except OSError as exc:
    st.error(f"Unable to load {architecture_path.name}: {exc}")
