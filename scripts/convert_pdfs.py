"""Purpose: Convert PDF folders to text and build temporary retrieval systems from PDF/TXT inputs.
Dependencies: built-in: hashlib, io, pathlib, pickle, re, sys, tempfile; installed: pypdf, faiss, numpy, langchain-text-splitters, sentence-transformers.
Custom: utils.config_loader, utils.model_store, utils.retriever.
"""

import io
import hashlib
import pickle
import re
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


def extract_pdf_text(pdf_source):
    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise RuntimeError(
            "PDF conversion requires pypdf. Install it with: pip install pypdf"
        ) from error

    reader = PdfReader(pdf_source)
    return "\n\n".join(page.extract_text() or "" for page in reader.pages).strip()


def convert_pdf_folder(input_folder, output_folder=None):
    source_directory = Path(input_folder).expanduser()
    if not source_directory.is_dir():
        raise NotADirectoryError(f"PDF input folder does not exist: {source_directory}")

    pdf_files = sorted(
        path
        for path in source_directory.iterdir()
        if path.is_file() and path.suffix.casefold() == ".pdf"
    )
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {source_directory}")

    destination = (
        Path(output_folder).expanduser()
        if output_folder is not None
        else PROJECT_ROOT / "data" / "raw"
    )
    destination.mkdir(parents=True, exist_ok=True)

    converted_files = []
    for pdf_path in pdf_files:
        text_path = destination / f"{pdf_path.stem}.txt"
        suffix_number = 2
        while text_path.exists():
            text_path = destination / f"{pdf_path.stem}_{suffix_number}.txt"
            suffix_number += 1
        try:
            text = extract_pdf_text(pdf_path)
        except Exception as error:
            raise ValueError(f"Could not convert {pdf_path.name}: {error}") from error
        text_path.write_text(text, encoding="utf-8")
        converted_files.append(text_path)

    return converted_files


def _clean_text(text):
    text = re.sub(r"^[=\-_*]{3,}\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def uploaded_files_signature(uploaded_files):
    signature = hashlib.sha256()
    for uploaded_file in uploaded_files:
        signature.update(Path(uploaded_file.name).name.encode("utf-8"))
        signature.update(b"\0")
        signature.update(uploaded_file.getvalue())
        signature.update(b"\0")
    return signature.hexdigest()


def build_session_retriever(uploaded_files, config):
    import faiss
    import numpy as np
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from sentence_transformers import SentenceTransformer

    from utils.model_store import ensure_model_available
    from utils.retriever import Retriever

    with tempfile.TemporaryDirectory(prefix="chatveritas-upload-") as temporary_root:
        temporary_root = Path(temporary_root)
        text_directory = temporary_root / "text"
        text_directory.mkdir()
        uploaded_texts = []

        for file_number, uploaded_file in enumerate(uploaded_files, start=1):
            original_name = Path(uploaded_file.name).name
            suffix = Path(original_name).suffix.casefold()
            if suffix not in {".pdf", ".txt"}:
                raise ValueError(f"Unsupported file type: {original_name}")

            payload = uploaded_file.getvalue()
            if not payload:
                raise ValueError(f"Uploaded file is empty: {original_name}")

            if suffix == ".pdf":
                text = extract_pdf_text(io.BytesIO(payload))
            else:
                try:
                    text = payload.decode("utf-8-sig")
                except UnicodeDecodeError as error:
                    raise ValueError(
                        f"Text file is not valid UTF-8: {original_name}"
                    ) from error

            safe_stem = re.sub(r"[^\w.-]+", "_", Path(original_name).stem).strip("._")
            if not safe_stem:
                safe_stem = "document"
            staged_path = text_directory / f"{file_number:04d}_{safe_stem}.txt"
            staged_path.write_text(text, encoding="utf-8")
            uploaded_texts.append((original_name, staged_path))

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config["retrieval"]["chunk_size"],
            chunk_overlap=config["retrieval"]["chunk_overlap"],
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

        chunks = []
        for source_name, text_path in uploaded_texts:
            text = _clean_text(text_path.read_text(encoding="utf-8"))
            for chunk in splitter.split_text(text):
                chunks.append(
                    {
                        "chunk_id": len(chunks),
                        "source": source_name,
                        "chunk": chunk,
                    }
                )

        if not chunks:
            raise ValueError("No readable text was found in the uploaded documents.")

        embedder = SentenceTransformer(
            ensure_model_available(config["embedding"]["model"]),
            device=config["embedding"].get("device", "cpu"),
        )
        embeddings = embedder.encode(
            [item["chunk"] for item in chunks],
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)
        del embedder

        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        vectorstore_directory = temporary_root / "vectorstore"
        vectorstore_directory.mkdir()
        index_path = vectorstore_directory / "index.faiss"
        chunks_path = vectorstore_directory / "chunks.pkl"
        faiss.write_index(index, str(index_path))
        with chunks_path.open("wb") as chunk_file:
            pickle.dump(chunks, chunk_file)

        return Retriever(
            index_path=index_path,
            chunks_path=chunks_path,
            embedding_model=config["embedding"]["model"],
            top_k=config["retrieval"]["top_k"],
            faiss_candidates=config["retrieval"]["faiss_candidates"],
            embedding_device=config["embedding"].get("device", "cpu"),
            reranker_model=config["reranker"]["model"],
            reranker_device=config["reranker"].get("device", "cpu"),
        )


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/convert_pdfs.py <folder_path>")
        return 2

    try:
        converted_files = convert_pdf_folder(sys.argv[1])
    except (OSError, RuntimeError, ValueError) as error:
        print(f"PDF conversion failed: {error}")
        return 1

    print(f"Converted {len(converted_files)} PDF file(s):")
    for text_file in converted_files:
        print(f"  {text_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())