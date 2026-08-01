"""
interfaces/chatveritas.py

High-level ChatVeritas interface class.

This is the **single object** that all entry points instantiate:

    - ``app_offline.py``   → ``ChatVeritas(mode="offline", use_lora=True)``
    - ``app_deploy.py``    → ``ChatVeritas(mode="deploy")``
    - ``scripts/chat.py``  → ``ChatVeritas(mode="offline", use_lora=use_lora)``

All retrieval, prompt engineering, and model inference are completely
hidden behind the :meth:`ask` method.  Entry points only need to:

1. Instantiate ``ChatVeritas``.
2. Call ``result = chatbot.ask(question)``.
3. Display ``result["response"]``, ``result["chunks"]``,
   and ``result["metrics"]``.
"""

from core.config import load_config
from core.exceptions import ConfigurationError
from core.logger import get_logger

logger = get_logger(__name__)

_SUPPORTED_MODES = frozenset({"offline", "deploy"})


class ChatVeritas:
    """
    High-level interface for the ChatVeritas RAG chatbot.

    Parameters
    ----------
    mode : str
        Inference backend.  ``"offline"`` uses the local Qwen 2.5 (3B)
        model with optional LoRA.  ``"deploy"`` uses the Groq API.
    use_lora : bool
        Whether to apply the LoRA fine-tuning adapter.  Only relevant
        when ``mode="offline"``.
    config : dict | None
        Pre-loaded configuration dictionary.  If ``None``, the
        configuration is loaded automatically from
        ``config/config.json``.

    Examples
    --------
    Offline (local model):

    .. code-block:: python

        from interfaces.chatveritas import ChatVeritas

        bot = ChatVeritas(mode="offline", use_lora=True)
        result = bot.ask("What is DataVeritas?")
        print(result["response"])

    Deploy (Groq API):

    .. code-block:: python

        from interfaces.chatveritas import ChatVeritas

        bot = ChatVeritas(mode="deploy")
        result = bot.ask("What is DataVeritas?")
        print(result["response"])
    """

    def __init__(
        self,
        mode: str = "offline",
        use_lora: bool = True,
        config: dict | None = None,
    ) -> None:
        if mode not in _SUPPORTED_MODES:
            raise ConfigurationError(
                f"Unsupported mode '{mode}'. Must be one of: {sorted(_SUPPORTED_MODES)}"
            )

        self._config = config if config is not None else load_config()
        self._mode = mode

        logger.info("Initialising ChatVeritas (mode=%s).", mode)
        self._pipeline = self._build_pipeline(mode, use_lora)
        logger.info("ChatVeritas ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ask(self, question: str) -> dict:
        """
        Answer a question using the full RAG pipeline.

        Parameters
        ----------
        question : str
            The user's question.  Leading and trailing whitespace is
            stripped automatically.

        Returns
        -------
        dict
            ``{"response": str, "chunks": list[dict], "metrics": dict}``

            - ``response``  — generated answer text
            - ``chunks``    — top-k reranked retrieved chunks
            - ``metrics``   — combined retrieval and generation statistics
        """
        question = question.strip()

        if not question:
            logger.warning("Empty question received — returning empty response.")
            return {"response": "", "chunks": [], "metrics": {}}

        logger.info("ChatVeritas.ask() | mode=%s | question=%r", self._mode, question)
        return self._pipeline.run(question)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_pipeline(self, mode: str, use_lora: bool):
        """Instantiate and return the pipeline for the given mode."""
        if mode == "offline":
            # Lazy import: heavy libraries (torch, transformers, faiss) are
            # only imported when the pipeline is actually built, ensuring
            # thread-limit env vars have already been applied by the entry
            # point before this call.
            from pipelines.offline_pipeline import OfflinePipeline

            return OfflinePipeline(config=self._config, use_lora=use_lora)

        if mode == "deploy":
            from pipelines.deploy_pipeline import DeployPipeline

            return DeployPipeline(config=self._config)

        # Should be unreachable due to the mode validation above.
        raise ConfigurationError(f"Unknown mode: {mode}")
