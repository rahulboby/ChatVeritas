"""
core/exceptions.py

Custom exception hierarchy for the ChatVeritas application.

All package-specific exceptions inherit from ``ChatVeritasError``, making
it easy for callers to catch any ChatVeritas error with a single clause:

    try:
        result = chatbot.ask(question)
    except ChatVeritasError as e:
        logger.error("ChatVeritas error: %s", e)
"""


class ChatVeritasError(Exception):
    """Base exception for all ChatVeritas errors."""


class ModelLoadError(ChatVeritasError):
    """Raised when a model, tokenizer, or LoRA adapter fails to load."""


class RetrievalError(ChatVeritasError):
    """Raised when retrieval from the FAISS index fails (e.g. sync error)."""


class ConfigurationError(ChatVeritasError):
    """Raised when required configuration keys are missing or invalid."""
