"""High-level public interface for the ChatVeritas RAG application."""

from __future__ import annotations

from typing import Any, Mapping

from configs import settings as default_settings
from core.exceptions import ConfigurationError
from core.llm import normalise_backend
from core.logger import get_logger

logger = get_logger(__name__)
_SUPPORTED_MODES = frozenset({"offline", "deploy"})


class ChatVeritas:
    """Answer document-grounded questions through one stable API.

    ``mode='offline'`` accepts ``backend='local'`` (Qwen with optional LoRA)
    or ``backend='lmstudio'``.  ``mode='deploy'`` always uses Groq.
    """

    def __init__(
        self,
        mode: str = "offline",
        use_lora: bool = True,
        backend: str = "local",
        config: Mapping[str, Any] | None = None,
        settings_module: Any = default_settings,
    ) -> None:
        mode = str(mode).strip().lower()
        if mode == "cloud":
            mode = "deploy"
        if mode not in _SUPPORTED_MODES:
            raise ConfigurationError(
                f"Unsupported mode '{mode}'. Must be one of: {sorted(_SUPPORTED_MODES)}"
            )

        self._mode = mode
        self._backend = normalise_backend(backend) if mode == "offline" else "groq"
        self._settings = (
            default_settings.from_legacy_config(config)
            if config is not None
            else settings_module
        )
        logger.info("Initialising ChatVeritas | mode=%s | backend=%s", mode, self._backend)
        self._pipeline = self._build_pipeline(use_lora)

    def ask(self, question: str) -> dict[str, Any]:
        """Return an answer, supporting chunks, and combined RAG metrics."""
        if not isinstance(question, str):
            raise TypeError("question must be a string.")
        question = question.strip()
        if not question:
            return {"response": "", "chunks": [], "metrics": {}}
        return self._pipeline.run(question)

    def _build_pipeline(self, use_lora: bool):
        if self._mode == "offline":
            from core.pipelines import OfflinePipeline

            return OfflinePipeline(
                backend=self._backend,
                use_lora=use_lora,
                settings_module=self._settings,
            )
        from core.pipelines import DeployPipeline

        return DeployPipeline(settings_module=self._settings)
