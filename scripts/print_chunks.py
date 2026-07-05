import sys
import pickle
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from utils.config_loader import load_config

"""
    Prints all the chunks frorm the chunks.pkl file in a readable format.
"""

def main():

    config = load_config()

    chunks_path = (
        PROJECT_ROOT /
        config["paths"]["vectorstore"] /
        "chunks.pkl"
    )

    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)

    print("=" * 100)
    print(f"Total Chunks: {len(chunks)}")
    print("=" * 100)

    for chunk in chunks:

        print(f"Chunk ID : {chunk['chunk_id']}")
        print(f"Source   : {chunk['source']}")
        print(f"Length   : {len(chunk['chunk'])} characters")
        print("-" * 100)
        print(chunk["chunk"])
        print("=" * 100)
        print()


if __name__ == "__main__":
    main()