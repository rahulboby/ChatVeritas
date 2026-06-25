# LLM Fine-Tuning on Custom Data

A learning-focused project to understand how Large Language Models (LLMs) can be specialized using LoRA fine-tuning on custom text data.

The project is designed to be reusable across domains. By simply replacing the text files in `data/raw` and optionally changing the base model in `config.yaml`, the same pipeline can be used to fine-tune a model on any subject.

# To train with new data:

1. Replace all the txt files in data\raw with the data you wish
2. Install requirements - ```pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121```
2. do ```python scripts/ingest.py``` - This prepares the dataset and stores it in data\processed\train.jsonl
3. run ```python scripts/chat.py``` - Runs chatbot in terminal
---

# Project Roadmap

## Version 1 (Current)

### Fine-Tuning Only

This version focuses exclusively on supervised fine-tuning using LoRA.

No RAG, vector databases, embeddings, or retrieval pipelines are used.

### Workflow

```text
TXT Files
    ↓
Dataset Preparation
    ↓
train.jsonl
    ↓
LoRA Fine-Tuning
    ↓
Adapter
    ↓
Chat Interface
```

### Technologies Used

* Qwen 2.5 1.5B Instruct
* LoRA (Low-Rank Adaptation)
* QLoRA (4-bit quantization)
* Hugging Face Transformers
* PEFT
* TRL
* Streamlit

### Why LoRA?

A full fine-tune would require updating all model parameters, which is computationally expensive and requires significantly more GPU memory.

Instead, this project uses LoRA:

* Most model parameters remain frozen.
* Only small adapter layers are trained.
* Training is significantly faster.
* GPU memory requirements are much lower.
* Adapter files are small and portable.

The resulting adapter can later be attached to the original base model to recreate the fine-tuned model.

---

# Project Structure

```text
qwen_fine_tune_v1/

├── adapters/
│   └── qwen-lora/
│
├── app/
│   └── app.py
│
├── config/
│   └── config.yaml
│
├── data/
│   ├── raw/
│   └── processed/
│
├── scripts/
│   ├── prepare_dataset.py
│   ├── train.py
│   └── chat.py
│
├── utils/
│   └── config_loader.py
│
└── README.md
```

---

# Dataset Preparation

Place one or more text files inside:

```text
data/raw/
```

Example:

```text
data/raw/

note1.txt
note2.txt
note3.txt
```

All text files are automatically processed and combined into a single training dataset.

The dataset preparation script:

```bash
python scripts/prepare_dataset.py
```

will:

1. Read all `.txt` files.
2. Split them into chunks.
3. Convert them into instruction-tuning samples.
4. Save the final dataset to:

```text
data/processed/train.jsonl
```

---

# Training

Run:

```bash
python scripts/train.py
```

The training script:

* Loads the base model specified in `config.yaml`
* Loads the processed dataset
* Applies LoRA adapters
* Fine-tunes the model
* Saves the trained adapter

Output:

```text
adapters/qwen-lora/
```

---

# Chat Interfaces

## Terminal Chat

```bash
python scripts/chat.py
```

Allows chatting with the fine-tuned model directly from the terminal.

---

## Streamlit Chatbot

```bash
streamlit run app.py
```

Provides a browser-based chat interface with conversation history stored using Streamlit session state.

---

# Reusing the Project

One of the primary goals of this project is reusability.

## Changing the Knowledge Domain

To fine-tune on a completely different subject:

### Step 1

Remove or move the existing files inside:

```text
data/raw/
```

### Step 2

Add new text files.

Example:

```text
data/raw/

resnet.txt
cnn.txt
pytorch.txt
```

### Step 3

Regenerate the dataset:

```bash
python scripts/prepare_dataset.py
```

### Step 4

Train again:

```bash
python scripts/train.py
```

The new adapter will now contain information derived from the new dataset.

No code changes are required.

---

# Changing the Base Model

The base model is controlled through:

```text
config/config.yaml
```

Example:

```yaml
model:
  name: "Qwen/Qwen2.5-1.5B-Instruct"
```

To use a different model:

```yaml
model:
  name: "Qwen/Qwen2.5-3B-Instruct"
```

or

```yaml
model:
  name: "microsoft/Phi-3-mini-4k-instruct"
```

Then:

1. Delete the old adapter.
2. Run training again.

Important:

LoRA adapters are model-specific.

An adapter trained for one model architecture cannot be reused with a different model.

---

# Version 2 (Planned)

### Retrieval-Augmented Generation (RAG)

No fine-tuning.

```text
User Question
      ↓
Embedding Model
      ↓
Vector Database
      ↓
Retrieve Top Chunks
      ↓
Append Context
      ↓
Base Model
      ↓
Answer
```

Topics to learn:

* Embeddings
* Chunking
* FAISS
* Similarity Search
* Retrieval Pipelines

---

# Version 3 (Planned)

### Fine-Tuning + RAG

Combines the strengths of both approaches.

```text
User Question
      ↓
Retriever
      ↓
Relevant Context
      ↓
Fine-Tuned Model
      ↓
Answer
```

Fine-Tuning provides:

* Domain-specific behavior
* Writing style
* Response formatting

RAG provides:

* Updatable knowledge
* Better factual recall
* Access to larger document collections

This architecture is much closer to modern production LLM systems.

---

# Learning Objectives

By completing all three versions of the project, the following concepts will be covered:

* Instruction Tuning
* Tokenization
* LoRA
* QLoRA
* Adapter-Based Fine-Tuning
* Model Inference
* Streamlit Deployment
* Embeddings
* Vector Databases
* FAISS
* Retrieval-Augmented Generation (RAG)
* Hybrid Fine-Tuned + RAG Systems
