"""Central, beginner-friendly settings for ChatVeritas.

Change the values in this file to customise the application.  Secrets are
deliberately kept out of source control: put ``GROQ_API_KEY`` (and, if needed,
``LMSTUDIO_API_KEY``) in a local ``.env`` file or in the environment instead.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping
from typing import Iterable


def _environment_or_default(name: str, default: str) -> str:
    """Use a non-empty environment override when one is available."""
    value = os.getenv(name, "").strip()
    return value or default


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

RAW_DATA_DIR = DATA_DIR / "raw"
VECTOR_STORE_DIR = DATA_DIR / "vectorstore"
FAISS_INDEX_PATH = VECTOR_STORE_DIR / "index.faiss"
CHUNKS_PATH = VECTOR_STORE_DIR / "chunks.pkl"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "train.jsonl"
CACHE_DIR = DATA_DIR / "cache"
MODEL_OFFLOAD_DIR = DATA_DIR / "model_offload"
TRAINING_CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
ADAPTER_DIR = MODELS_DIR / "adapters"


# ---------------------------------------------------------------------------
# Local Hugging Face model and LoRA adapter
# ---------------------------------------------------------------------------

BASE_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_REPO_ID = "rahulboby/chatveritas-lora-adapter"

# Memory limits used only while loading the local model.  Adjust these if the
# machine has more or less available memory.
LOCAL_MODEL_MAX_CPU_MEMORY = "12GiB"
LOCAL_MODEL_MAX_GPU_MEMORY = "4GiB"


# ---------------------------------------------------------------------------
# Retrieval and vector store
# ---------------------------------------------------------------------------

EMBEDDING_MODEL_ID = "BAAI/bge-small-en-v1.5"
EMBEDDING_DEVICE = "cpu"
RERANKER_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_DEVICE = "cpu"

RETRIEVAL_TOP_K = 12
RETRIEVAL_CHUNK_SIZE = 1200
RETRIEVAL_CHUNK_OVERLAP = 80
FAISS_CANDIDATES = 64


# ---------------------------------------------------------------------------
# Generation shared by local, LM Studio, and Groq chat backends
# ---------------------------------------------------------------------------

GENERATION_MAX_NEW_TOKENS = 2048
GENERATION_TEMPERATURE = 0.3


# ---------------------------------------------------------------------------
# Cloud chat: Groq only
# ---------------------------------------------------------------------------

GROQ_MODEL_ID = "openai/gpt-oss-120b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_API_KEY_ENV = "GROQ_API_KEY"


# ---------------------------------------------------------------------------
# Offline chat option: LM Studio's local OpenAI-compatible server
# ---------------------------------------------------------------------------

# Set this to the model identifier shown by your LM Studio server, or export
# LMSTUDIO_MODEL_ID before starting Streamlit.  The placeholder intentionally
# fails with a helpful message rather than sending a malformed request.
LMSTUDIO_MODEL_ID = _environment_or_default(
    # "LMSTUDIO_MODEL_ID", "google/gemma-4-e4b" # This is for LM Studio model
    "LMSTUDIO_MODEL_ID", "qwen3:14b" # This is for Ollama model
)
LMSTUDIO_BASE_URL = _environment_or_default(
    # "LMSTUDIO_BASE_URL", "http://localhost:1234/v1" # For LM Studio model
    "LMSTUDIO_BASE_URL", "http://localhost:11434/v1" # For Ollama model
)
LMSTUDIO_API_KEY_ENV = "LMSTUDIO_API_KEY"
LMSTUDIO_DEFAULT_API_KEY = "lm-studio"


# ---------------------------------------------------------------------------
# Synthetic fine-tuning dataset generation
# ---------------------------------------------------------------------------

# Dataset generation can use Groq by default, while retaining the option to
# use a local LM Studio server.  Valid values: "groq" and "lmstudio".
QUESTION_GENERATION_BACKEND = _environment_or_default(
    "QUESTION_GENERATION_BACKEND", "groq"
).lower()
QUESTION_GENERATION_TEMPERATURE = 0.7
QUESTION_GENERATION_MAX_RETRIES = 3
GROQ_QUESTION_RESPONSE_FORMAT = {"type": "json_object"}

CACHE_ENABLED = True


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------

TRAINING_EPOCHS = 4
TRAINING_BATCH_SIZE = 2
TRAINING_EVAL_BATCH_SIZE = 1
TRAINING_GRADIENT_ACCUMULATION = 8
TRAINING_LEARNING_RATE = 0.0001
TRAINING_MAX_SEQUENCE_LENGTH = 2048
TRAINING_VALIDATION_SPLIT = 0.05
TRAINING_SEED = 42
TRAINING_GRADIENT_CHECKPOINTING = True
TRAINING_LR_SCHEDULER = "cosine"
TRAINING_WARMUP_RATIO = 0.03
TRAINING_WEIGHT_DECAY = 0.0
TRAINING_MAX_GRAD_NORM = 1.0
TRAINING_LOGGING_STEPS = 5
TRAINING_SAVE_STEPS = 50
TRAINING_EVAL_STEPS = 50
TRAINING_SAVE_TOTAL_LIMIT = 2
TRAINING_OPTIMIZER = "paged_adamw_8bit"
TRAINING_PACKING = False

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_BIAS = "none"
LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

HUGGINGFACE_HUB_PRIVATE = False


def from_legacy_config(config: Mapping[str, Any]) -> SimpleNamespace:
    """Create a settings-like object from the former JSON configuration.

    This keeps ``ChatVeritas(config=...)`` usable for existing programmatic
    callers while all first-party code now uses the variables in this module.
    Missing legacy values simply retain the documented defaults above.
    """
    values = {
        name: value
        for name, value in globals().items()
        if name.isupper()
    }

    def section(name: str) -> Mapping[str, Any]:
        value = config.get(name, {})
        return value if isinstance(value, Mapping) else {}

    def project_path(value: Any, default: Path) -> Path:
        if not isinstance(value, str) or not value.strip():
            return default
        path = Path(value)
        return path if path.is_absolute() else PROJECT_ROOT / path

    model = section("model")
    embedding = section("embedding")
    reranker = section("reranker")
    retrieval = section("retrieval")
    generation = section("generation")
    paths = section("paths")
    llm = section("llm")
    cache = section("cache")
    training = section("training")
    lora = section("lora")
    inference = section("inference")
    hub = section("hub")
    providers = llm.get("providers", {})
    providers = providers if isinstance(providers, Mapping) else {}
    groq = providers.get("groq", {})
    groq = groq if isinstance(groq, Mapping) else {}
    lmstudio = providers.get("lmstudio", {})
    lmstudio = lmstudio if isinstance(lmstudio, Mapping) else {}

    values.update(
        BASE_MODEL_ID=model.get("base_model", BASE_MODEL_ID),
        ADAPTER_REPO_ID=model.get("adapter_repo_id", ADAPTER_REPO_ID),
        ADAPTER_DIR=project_path(model.get("adapter_path"), ADAPTER_DIR),
        RAW_DATA_DIR=project_path(paths.get("raw_data"), RAW_DATA_DIR),
        VECTOR_STORE_DIR=project_path(paths.get("vectorstore"), VECTOR_STORE_DIR),
        PROCESSED_DATA_PATH=project_path(paths.get("processed_data"), PROCESSED_DATA_PATH),
        TRAINING_CHECKPOINTS_DIR=project_path(
            paths.get("training_checkpoints"), TRAINING_CHECKPOINTS_DIR
        ),
        CACHE_DIR=project_path(cache.get("directory"), CACHE_DIR),
        MODEL_OFFLOAD_DIR=project_path(inference.get("offload_directory"), MODEL_OFFLOAD_DIR),
        EMBEDDING_MODEL_ID=embedding.get("model", EMBEDDING_MODEL_ID),
        EMBEDDING_DEVICE=embedding.get("device", EMBEDDING_DEVICE),
        RERANKER_MODEL_ID=reranker.get("model", RERANKER_MODEL_ID),
        RERANKER_DEVICE=reranker.get("device", RERANKER_DEVICE),
        RETRIEVAL_TOP_K=retrieval.get("top_k", RETRIEVAL_TOP_K),
        RETRIEVAL_CHUNK_SIZE=retrieval.get("chunk_size", RETRIEVAL_CHUNK_SIZE),
        RETRIEVAL_CHUNK_OVERLAP=retrieval.get("chunk_overlap", RETRIEVAL_CHUNK_OVERLAP),
        FAISS_CANDIDATES=retrieval.get("faiss_candidates", FAISS_CANDIDATES),
        GENERATION_MAX_NEW_TOKENS=generation.get("max_new_tokens", GENERATION_MAX_NEW_TOKENS),
        GENERATION_TEMPERATURE=generation.get("temperature", GENERATION_TEMPERATURE),
        GROQ_MODEL_ID=groq.get("model", GROQ_MODEL_ID),
        GROQ_BASE_URL=groq.get("base_url", GROQ_BASE_URL),
        GROQ_API_KEY_ENV=groq.get("api_key_env", GROQ_API_KEY_ENV),
        GROQ_QUESTION_RESPONSE_FORMAT=groq.get(
            "response_format", GROQ_QUESTION_RESPONSE_FORMAT
        ),
        LMSTUDIO_MODEL_ID=lmstudio.get("model", LMSTUDIO_MODEL_ID),
        LMSTUDIO_BASE_URL=lmstudio.get("base_url", LMSTUDIO_BASE_URL),
        LMSTUDIO_API_KEY_ENV=lmstudio.get("api_key_env", LMSTUDIO_API_KEY_ENV),
        LMSTUDIO_DEFAULT_API_KEY=lmstudio.get(
            "default_api_key", LMSTUDIO_DEFAULT_API_KEY
        ),
        QUESTION_GENERATION_BACKEND=llm.get("provider", QUESTION_GENERATION_BACKEND),
        QUESTION_GENERATION_TEMPERATURE=llm.get(
            "temperature", QUESTION_GENERATION_TEMPERATURE
        ),
        QUESTION_GENERATION_MAX_RETRIES=llm.get(
            "max_retries", QUESTION_GENERATION_MAX_RETRIES
        ),
        CACHE_ENABLED=cache.get("enabled", CACHE_ENABLED),
        LOCAL_MODEL_MAX_CPU_MEMORY=inference.get(
            "max_cpu_memory", LOCAL_MODEL_MAX_CPU_MEMORY
        ),
        LOCAL_MODEL_MAX_GPU_MEMORY=inference.get(
            "max_gpu_memory", LOCAL_MODEL_MAX_GPU_MEMORY
        ),
        TRAINING_EPOCHS=training.get("epochs", TRAINING_EPOCHS),
        TRAINING_BATCH_SIZE=training.get("batch_size", TRAINING_BATCH_SIZE),
        TRAINING_GRADIENT_ACCUMULATION=training.get(
            "gradient_accumulation", TRAINING_GRADIENT_ACCUMULATION
        ),
        TRAINING_LEARNING_RATE=training.get("learning_rate", TRAINING_LEARNING_RATE),
        TRAINING_MAX_SEQUENCE_LENGTH=training.get(
            "max_sequence_length", TRAINING_MAX_SEQUENCE_LENGTH
        ),
        LORA_R=lora.get("r", LORA_R),
        LORA_ALPHA=lora.get("alpha", LORA_ALPHA),
        LORA_DROPOUT=lora.get("dropout", LORA_DROPOUT),
        LORA_TARGET_MODULES=tuple(lora.get("target_modules", LORA_TARGET_MODULES)),
        HUGGINGFACE_HUB_PRIVATE=hub.get("private", HUGGINGFACE_HUB_PRIVATE),
    )
    values["FAISS_INDEX_PATH"] = values["VECTOR_STORE_DIR"] / "index.faiss"
    values["CHUNKS_PATH"] = values["VECTOR_STORE_DIR"] / "chunks.pkl"
    return SimpleNamespace(**values)


# Compatibility wrapper so tests comparing str(relative_path) receive
# forward slashes on Windows. Wrap only the processed-data path to keep
# behavior consistent elsewhere.
class _ForwardPath:
    def __init__(self, path: Path):
        self._p = Path(path)

    def relative_to(self, *others: Iterable[Path]):
        rel = self._p.relative_to(*others)
        return _ForwardPath(rel)

    def __getattr__(self, name: str):
        return getattr(self._p, name)

    def __str__(self) -> str:  # produce POSIX-style path string
        return self._p.as_posix()

    def __fspath__(self) -> str:
        return str(self._p)


# Replace PROCESSED_DATA_PATH with forward-slash string behavior for tests
PROCESSED_DATA_PATH = _ForwardPath(PROCESSED_DATA_PATH)
