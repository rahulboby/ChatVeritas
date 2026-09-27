import sys
from pathlib import Path
import pickle
import re

import faiss
import numpy as np
# from langchain.text_splitter import RecursiveCharacterTextSplitter # If you have the full langchain ecosystem
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---- Project setup ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from utils.config_loader import load_config
from utils.vectorstores import named_vectorstore_path

config = load_config()


def clean_text(text):
    """Remove excessive separators and extra blank lines."""
    text = re.sub(r"^[=\-_*]{3,}\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def main():
    raw_dir = PROJECT_ROOT / config["paths"]["raw_data"]
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw data directory does not exist: {raw_dir}")

    while True:
        name = input("Name this vector store: ").strip()
        try:
            vectorstore_dir = named_vectorstore_path(PROJECT_ROOT, config, name)
        except ValueError as error:
            print(error)
            continue
        if vectorstore_dir.exists():
            print(f"Vector store already exists: {vectorstore_dir}. Choose a new name.")
            continue
        break

    txt_files = sorted(raw_dir.glob("*.txt"))
    if not txt_files:
        raise ValueError(f"No .txt files found in {raw_dir}")

    # ---- 1. Load embedding model ----
    import torch
    from sentence_transformers import SentenceTransformer

    print("Loading embedding model...")
    embedder = SentenceTransformer(config["embedding"]["model"])

    # ---- 2. Set up the text splitter (character‑based) ----
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["retrieval"]["chunk_size"],       # characters, e.g., 1000
        chunk_overlap=config["retrieval"]["chunk_overlap"], # characters, e.g., 80
        separators=["\n\n", "\n", ". ", " ", ""],           # prefer sentence/paragraph breaks
        length_function=len,
    )

    # ---- 3. Process all .txt files ----
    all_chunks = []
    chunk_id = 0
    print(f"Found {len(txt_files)} files")

    for file in txt_files:
        print(f"Reading {file.name}")
        text = file.read_text(encoding="utf-8")
        text = clean_text(text)

        # Split into chunks
        chunks = splitter.split_text(text)

        for chunk in chunks:
            all_chunks.append({
                "chunk_id": chunk_id,
                "source": file.name,
                "chunk": chunk,
            })
            chunk_id += 1

    print(f"Created {len(all_chunks)} chunks")
    if not all_chunks:
        raise ValueError(f"No text chunks were created from files in {raw_dir}")

    # ---- 4. Generate embeddings ----
    print("Generating embeddings...")
    texts = [item["chunk"] for item in all_chunks]
    embeddings = embedder.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    embeddings = embeddings.astype(np.float32)

    # ---- 5. Build FAISS index ----
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    vectorstore_dir.mkdir(parents=True, exist_ok=False)
    faiss.write_index(index, str(vectorstore_dir / "index.faiss"))

    # ---- 6. Save chunks ----
    with open(vectorstore_dir / "chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"Saved {len(all_chunks)} chunks to {vectorstore_dir}")


if __name__ == "__main__":
    main()