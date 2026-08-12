import sys
import pickle
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.constants import apply_thread_limits

apply_thread_limits()

import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from configs import settings


def clean_text(text):
    """Remove excessive separators and extra blank lines."""
    text = re.sub(r"^[=\-_*]{3,}\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def main():
    raw_dir = settings.RAW_DATA_DIR
    vectorstore_dir = settings.VECTOR_STORE_DIR
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw document directory not found: {raw_dir}")
    vectorstore_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load embedding model ----
    print("Loading embedding model...")
    embedder = SentenceTransformer(
        settings.EMBEDDING_MODEL_ID, device=settings.EMBEDDING_DEVICE
    )

    # ---- 2. Set up the text splitter (character‑based) ----
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.RETRIEVAL_CHUNK_SIZE,
        chunk_overlap=settings.RETRIEVAL_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],           # prefer sentence/paragraph breaks
        length_function=len,
    )

    # ---- 3. Process all .txt files ----
    all_chunks = []
    chunk_id = 0
    txt_files = sorted(raw_dir.glob("*.txt"))
    print(f"Found {len(txt_files)} files")

    for file in txt_files:
        print(f"Reading {file.name}")
        text = file.read_text(encoding="utf-8")
        text = clean_text(text)

        # Split into chunks
        chunks = splitter.split_text(text)

        for chunk in chunks:
            if not chunk.strip():
                continue
            all_chunks.append({
                "chunk_id": chunk_id,
                "source": file.name,
                "chunk": chunk,
            })
            chunk_id += 1

    print(f"Created {len(all_chunks)} chunks")
    if not all_chunks:
        raise ValueError(
            f"No usable text chunks were found in {raw_dir}. Add non-empty .txt files first."
        )

    # ---- 4. Generate embeddings ----
    print("Generating embeddings...")
    texts = [item["chunk"] for item in all_chunks]
    embeddings = embedder.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    embeddings = embeddings.astype(np.float32)

    # ---- 5. Build FAISS index ----
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    faiss.write_index(index, str(vectorstore_dir / "index.faiss"))

    # ---- 6. Save chunks ----
    with open(vectorstore_dir / "chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"Saved {len(all_chunks)} chunks to {vectorstore_dir}")


if __name__ == "__main__":
    main()
