"""Offline LCEL pipeline using the existing local Qwen plus LoRA backend."""

from core.logger import get_logger
from pipelines.rag_pipeline import RAGPipeline

logger = get_logger(__name__)


class OfflinePipeline(RAGPipeline):
    """Run the existing custom RAG flow with the local Qwen backend."""

    def __init__(self, config: dict, use_lora: bool = True) -> None:
        self.use_lora = use_lora
        logger.info("Initialising OfflinePipeline (use_lora=%s).", use_lora)
        super().__init__(config=config)
        logger.info("OfflinePipeline ready.")

    def _create_llm(self):
        """Create the unchanged Qwen model and optional LoRA adapter backend."""
        # Delay heavy model imports until the app has applied thread limits.
        from llm.offline import OfflineLLM

        return OfflineLLM(config=self.config, use_lora=self.use_lora)
