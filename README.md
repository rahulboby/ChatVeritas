# End-to-End Pipeline

This project implements a **Retrieval-Augmented Generation (RAG)** chatbot using **Qwen 2.5**, **Sentence Transformers**, **FAISS**, and **Streamlit**. The system allows a user to place custom `.txt` documents into a folder, index them into a vector database, and query them through a conversational interface.

The project combines RAG with optional QLoRA fine-tuning. RAG keeps document knowledge in FAISS, while QLoRA trains a small PEFT adapter and leaves the base Qwen weights unchanged.

---

# Pipeline

## 1. Data Collection

### Input

The knowledge base consists of plain text files placed inside:

```text
data/raw/
```

Example:

```text
data/raw/
├── report.txt
├── paper.txt
└── notes.txt
```

### Purpose

These files represent the external knowledge that the chatbot should answer questions about.

### Dependencies Used

None (standard Python file handling).

---

## 2. Document Ingestion (`scripts/ingest.py`)

Run:

```bash
python scripts/ingest.py
```

This script performs all preprocessing required before the chatbot can answer questions.

The overall workflow is:

```text
TXT Files
    ↓
Read Documents
    ↓
Chunk Documents
    ↓
Generate Embeddings
    ↓
Build FAISS Index
    ↓
Save Vector Database
```

### Dependencies

| Module | Purpose |
|---------|---------|
| `pathlib` | Locate `.txt` files |
| `SentenceTransformer` | Generate semantic embeddings |
| `numpy` | Store embeddings as float arrays |
| `faiss` | Create vector database |
| `pickle` | Store document chunks |

**Sentence transformers** modify the BERT architecture with siamese networks. It is designed to form a vector space of sentences instead of words. BERT is good at word meaning recognition, but can't really do good with whole sentences. Eg. "What is the weather today?" and "Is it raining outside?" will have similar vector representations.
- Siamese neural network is architecture designed to compare two or more inputs for their similarity

They map sentences, paragraphs, or short texts into a dense vector space (typically 384 or 768 dimensions).

**FAISS** (Facebook AI Similarity Search) is an open-source library developed by Meta specifically designed for efficient similarity search and clustering of dense vectors. This basically does it more efficiently than brute force. 

The Sentence tranformer converts the sentences (or chunks) into vectors. FAISS stores these vectores and efficiently finding matches for queries. 


## 3. Reading Documents

The ingestion script scans:

```text
data/raw/
```

using:

```python
Path.glob("*.txt")
```

Each file is opened and loaded into memory as a Python string.

Example:

```text
report.txt

↓

"The DataVeritas framework uses Streamlit..."
```

At this point, the project simply has raw text.

---

## 4. Document Chunking

Large documents cannot be embedded or retrieved efficiently as a single block.

The ingestion script therefore splits every document into smaller chunks.

Current implementation:

- Chunk size: **500 characters**
- Chunk overlap: **50 characters**

Example:

```text
Entire Report

↓

Chunk 1

Chunk 2

Chunk 3

...
```

The overlap ensures that information near chunk boundaries is preserved across adjacent chunks.

### Why chunking?

Without chunking:

- embeddings become too broad
- retrieval quality decreases
- prompt size grows unnecessarily

### Dependencies

Custom Python function.

---

## 5. Embedding Generation

Each chunk is converted into a numerical vector using:

```python
SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)
```

### Role of the Embedding Model

The embedding model **does not generate text**.

Instead, it converts text into a dense numerical representation that captures semantic meaning.

Example:

```text
DataVeritas uses Streamlit

↓

[0.124,
-0.551,
0.847,
...
384 values]
```

The resulting vector has **384 dimensions**.

```sentence-transformer/all-MiniLM-L6-v2``` is a specific, highly optimized pre-trained deep learning model designed to map sentences and paragraphs into a dense vector space. It is one of the most popular baseline models in the sentence-transformers library.

What makes it work?
- MiniLM Architecture: It is a distilled (compressed) version of a larger transformer model (like BERT or RoBERTa). It retains roughly 99% of the performance of larger models while being vastly smaller and faster.

- The "L6" and "v2": It features 6 layers (making it very lightweight and fast to run on CPU or GPU) and is on its second major training iteration.

- Dimensions: It outputs a vector of exactly 384 dimensions for every piece of text you feed it.

Key Characteristics:
- Max Sequence Length: It can handle inputs up to 256 tokens (roughly 150–200 words). Anything longer is truncated.

- Trained for General Purpose: The all- prefix means it was trained on over one billion sentence pairs from diverse sources (Reddit, Wikipedia, StackExchange, etc.). It acts as an excellent all-rounder for matching similar concepts, even if they use entirely different vocabularies.

Semantically similar sentences produce vectors that are close together in vector space.

### Dependencies

| Package | Role |
|----------|------|
| `sentence-transformers` | Embedding generation |
| `torch` | Executes the neural network |

---

## 6. Building the Vector Database

Once embeddings are generated, they are inserted into a FAISS index.

Current implementation:

```python
faiss.IndexFlatL2()
```

### What is FAISS?

FAISS (Facebook AI Similarity Search) is a library for efficient similarity search over high-dimensional vectors.

Instead of searching text directly, it searches embedding vectors.

Current similarity metric:

- Euclidean Distance (L2)

```faiss.IndexFlatL2()``` is an indexing structure provided by Meta's FAISS library. It serves as the storage and search engine for the vectors generated by your model.

How it works:
- "Flat": The word "Flat" means the index stores the vectors exactly as they are, in their raw form, without compressing them or restructuring the vector space into clusters or graphs.

- "L2": This specifies the distance metric used to determine similarity. L2 distance is Euclidean distance. Geometrically, it calculates the straight-line distance between two coordinates in a 384-dimensional space.

The mathematical formula for the L2 distance between two vectors $u$ and $v$ in an $n$-dimensional space is:

$$d(u, v) = \sqrt{\sum_{i=1}^{n} (u_i - v_i)^2}$$

Key Characteristics:
- Brute-Force Search (K-Nearest Neighbors): When you pass a query vector to this index, it calculates the L2 distance between your query and every single vector stored in the index.

- 100% Recall (Perfect Accuracy): Because it checks everything, it is guaranteed to find the absolute mathematically closest vectors. It is the gold standard for accuracy.

- Memory & Speed Trade-off: Because it keeps raw vectors in RAM and does an exhaustive search, it becomes slow and memory-intensive if you try to scale it to millions of documents.

### Dependencies

| Package | Role |
|----------|------|
| `faiss-cpu` | Vector indexing and nearest-neighbor search |

---

## 7. Saving the Knowledge Base

The ingestion pipeline stores two files:

### `index.faiss`

Contains:

- embedding vectors
- FAISS search index

Does **not** contain readable text.

---

### `chunks.pkl`

Contains:

```python
[
    {
        "source": "report.txt",
        "chunk": "..."
    },
    ...
]
```

Each embedding stored inside `index.faiss` corresponds to one entry inside `chunks.pkl`.

The index position maintains this relationship.

---

# Online Pipeline (Inference)

The remaining stages execute every time the chatbot runs.

---

## 8. Launching the Chatbot

Run:

```bash
streamlit run app.py
```

Streamlit starts the web application and loads all required components.

### Dependencies

| Package | Role |
|----------|------|
| `streamlit` | Web interface |
| `torch` | Neural network execution |
| `transformers` | Load Qwen model |
| `sentence-transformers` | Embed user queries |
| `faiss` | Retrieve relevant chunks |

---

## 9. Loading the Language Model

The chatbot loads:

```python
AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct"
)
```

along with

```python
AutoTokenizer
```

### Purpose

Qwen is responsible only for:

- reasoning
- text generation
- language understanding

It **does not store project knowledge**.

Knowledge remains external inside the vector database.

### Dependencies

| Package | Role |
|----------|------|
| `transformers` | Load tokenizer and model |
| `accelerate` | Automatic GPU placement |
| `torch` | Executes inference |

---

## 10. Loading the Retriever

The retriever loads:

```text
index.faiss
```

and

```text
chunks.pkl
```

along with the same embedding model used during ingestion.

Using the **same embedding model** is critical because:

- document vectors
- query vectors

must exist in the same embedding space.

---

## 11. User Query

Example:

```text
What frontend does DataVeritas use?
```

The query first goes through the embedding model.

It becomes:

```text
Question

↓

384-dimensional embedding vector
```

No LLM is involved yet.

---

## 12. Semantic Retrieval

The query embedding is compared against every stored embedding inside FAISS.

The retriever performs:

```python
index.search(
    query_embedding,
    top_k
)
```

Current configuration:

```json
{
  "retrieval": {
    "top_k": 5
  }
}
```

The five nearest document chunks are returned.

Unlike keyword search, retrieval is based on **semantic similarity**, not exact word matching.

---

## 13. Prompt Construction

The retrieved chunks are concatenated into a context block.

Example:

```text
Context:

DataVeritas uses Streamlit...

The dashboard...

Entity Resolution...

Question:

What frontend does DataVeritas use?
```

This augmented prompt is passed to Qwen.

This process is called **Prompt Augmentation**, and is the defining characteristic of Retrieval-Augmented Generation.

---

## 14. Tokenization

The complete prompt is tokenized using:

```python
AutoTokenizer
```

Example:

```text
Prompt

↓

Tokens

↓

Token IDs
```

Qwen operates on token IDs rather than raw text.

---

## 15. Response Generation

The tokenized prompt is passed to:

```python
model.generate(...)
```

Qwen performs autoregressive decoding:

```text
Prompt

↓

Predict Next Token

↓

Predict Next Token

↓

Predict Next Token

↓

End of Sequence
```

The generated tokens are decoded back into natural language.

### Dependencies

| Package | Role |
|----------|------|
| `transformers` | Generation pipeline |
| `torch` | Tensor operations and inference |

---

## 16. Streamlit Interface

The generated response is displayed inside the chat interface.

The application also displays:

- retrieved document chunks
- source documents (if enabled)
- similarity scores (future enhancement)

using Streamlit components such as:

- `st.chat_message()`
- `st.chat_input()`
- `st.expander()`
- `st.session_state`

---

# Summary of Major Components

| Component | Purpose |
|------------|---------|
| `.txt` Documents | External knowledge source |
| `ingest.py` | Preprocess and index documents |
| Chunking | Split large documents into searchable units |
| Sentence Transformer | Convert text into semantic embeddings |
| Embeddings | Numerical representation of semantic meaning |
| FAISS | Vector database for similarity search |
| Retriever | Finds the most relevant document chunks |
| Prompt Builder | Combines retrieved context with the user's question |
| Qwen 2.5 | Generates the final response |
| Streamlit | Provides the conversational user interface |

---

# Complete Execution Sequence

1. User places `.txt` files inside `data/raw/`.
2. `ingest.py` reads every document.
3. Documents are divided into overlapping chunks.
4. Each chunk is converted into a 384-dimensional embedding using `SentenceTransformer`.
5. All embeddings are indexed using FAISS (`IndexFlatL2`).
6. The FAISS index (`index.faiss`) and chunk metadata (`chunks.pkl`) are saved locally.
7. The user launches the chatbot using Streamlit.
8. Qwen, the tokenizer, the embedding model, and the FAISS index are loaded into memory.
9. The user asks a question.
10. The question is embedded using the same Sentence Transformer model.
11. FAISS performs semantic similarity search to retrieve the most relevant document chunks.
12. The retrieved chunks are combined with the user's question to construct an augmented prompt.
13. The prompt is tokenized and passed to Qwen.
14. Qwen generates a response conditioned on the retrieved context.
15. Streamlit displays the generated answer along with the retrieved context (and optionally metadata such as sources and similarity scores).
