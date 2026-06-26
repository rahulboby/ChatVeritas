# For testing reranker crash in terminal (If app working, delete this file)
from sentence_transformers import CrossEncoder
import time

print("Starting")

start = time.perf_counter()

model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

print("Loaded")

print(time.perf_counter() - start)