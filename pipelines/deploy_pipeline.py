"""Deploy LCEL pipeline using the existing Groq-backed LLM backend."""

from core.logger import get_logger
from pipelines.rag_pipeline import RAGPipeline

logger = get_logger(__name__)


class DeployPipeline(RAGPipeline):
    """Run the existing custom RAG flow with the Groq deployment backend."""

    def __init__(self, config: dict) -> None:
        logger.info("Initialising DeployPipeline.")
        super().__init__(config=config)
        logger.info("DeployPipeline ready.")

    def _create_llm(self):
        """Create the unchanged Groq OpenAI-compatible backend."""
        from llm.deploy import DeployLLM

        return DeployLLM(config=self.config)
