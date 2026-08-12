import pickle
from pathlib import Path

from configs import settings


"""Print all chunks from the configured `chunks.pkl` file in a readable format."""


def main():
    chunks_path: Path = settings.CHUNKS_PATH
    if not chunks_path.is_file():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)

    print("=" * 100)
    print(f"Total Chunks: {len(chunks)}")
    print("=" * 100)

    for chunk in chunks:
        print(f"Chunk ID : {chunk.get('chunk_id', 'N/A')}")
        print(f"Source   : {chunk.get('source', 'Unknown')}")
        print(f"Length   : {len(chunk.get('chunk',''))} characters")
        print("-" * 100)
        print(chunk.get("chunk", ""))
        print("=" * 100)
        print()


if __name__ == "__main__":
    main()
