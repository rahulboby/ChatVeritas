"""Flat RAG pipelines for local/LM Studio and Groq generation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from operator import itemgetter
from typing import TYPE_CHECKING, Any, Mapping

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough

from configs import settings as default_settings
from core.llm import BaseLLM
from core.logger import get_logger
from core.prompts import RAG_PROMPT_TEMPLATE

if TYPE_CHECKING:
    from core.retrieval import Retriever

logger = get_logger(__name__)


class RAGPipeline(ABC):
    """Shared LCEL orchestration around ChatVeritas retrieval and generation."""

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        settings_module: Any = default_settings,
    ) -> None:
        # ``config`` retains the former programmatic injection point while
        # first-party callers use configs/settings.py directly.
        self.settings = (
            default_settings.from_legacy_config(config)
            if config is not None
            else settings_module
        )
        self.retriever = self._create_retriever()
        self.llm = self._create_llm()
        self.chain = self._build_chain()

    def _create_retriever(self) -> "Retriever":
        # Lazy import keeps cloud and LM Studio startup free of FAISS/model imports
        # until an actual pipeline is requested.
        from core.retrieval import Retriever

        return Retriever(
            index_path=self.settings.FAISS_INDEX_PATH,
            chunks_path=self.settings.CHUNKS_PATH,
            embedding_model=self.settings.EMBEDDING_MODEL_ID,
            top_k=self.settings.RETRIEVAL_TOP_K,
            faiss_candidates=self.settings.FAISS_CANDIDATES,
            embedding_device=self.settings.EMBEDDING_DEVICE,
            reranker_model=self.settings.RERANKER_MODEL_ID,
            reranker_device=self.settings.RERANKER_DEVICE,
        )

    @abstractmethod
    def _create_llm(self) -> BaseLLM:
        """Create the generation backend for this pipeline."""

    def _build_chain(self) -> Runnable[dict[str, str], dict[str, Any]]:
        retrieve = RunnableLambda(self._retrieve, name="retrieve_and_rerank")
        build_context = RunnableLambda(self._build_context, name="build_context")
        prompt_inputs = RunnableLambda(self._prompt_inputs, name="prompt_inputs")
        prompt = prompt_inputs | RAG_PROMPT_TEMPLATE
        generation = (
            itemgetter("prompt")
            | RunnableLambda(lambda value: value.to_string(), name="prompt_to_text")
            | self.llm.as_runnable()
        )
        parse_response = (
            itemgetter("generation")
            | RunnableLambda(lambda value: value["response"], name="generation_text")
            | StrOutputParser()
        )
        return (
            RunnablePassthrough.assign(retrieval=retrieve)
            | RunnablePassthrough.assign(context=build_context)
            | RunnablePassthrough.assign(prompt=prompt)
            | RunnablePassthrough.assign(generation=generation)
            | RunnablePassthrough.assign(response=parse_response)
            | RunnableLambda(self._format_result, name="format_result")
        )

    def run(self, question: str) -> dict[str, Any]:
        """Run retrieval, prompting, generation, and result formatting."""
        logger.info("%s running query: %r", self.__class__.__name__, question)
        result = self.chain.invoke({"question": question})
        logger.info(
            "%s complete - generation: %.2f s.",
            self.__class__.__name__,
            result["metrics"]["generation_time"],
        )
        return result

    def _retrieve(self, state: dict[str, Any]) -> dict[str, Any]:
        return self.retriever.retrieve(state["question"])

    @staticmethod
    def _build_context(state: dict[str, Any]) -> str:
        return "\n\n".join(item["chunk"] for item in state["retrieval"]["results"])

    @staticmethod
    def _prompt_inputs(state: dict[str, Any]) -> dict[str, str]:
        return {"question": state["question"], "context": state["context"]}

    @staticmethod
    def _format_result(state: dict[str, Any]) -> dict[str, Any]:
        metrics = {
            **state["retrieval"]["metrics"],
            **state["generation"]["metrics"],
        }
        return {
            "response": state["response"],
            "chunks": state["retrieval"]["results"],
            "metrics": metrics,
        }


class OfflinePipeline(RAGPipeline):
    """RAG pipeline for the local HF model or an LM Studio server."""

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        backend: str = "local",
        use_lora: bool = True,
        settings_module: Any = default_settings,
    ) -> None:
        self.backend = backend
        self.use_lora = use_lora
        super().__init__(config=config, settings_module=settings_module)

    def _create_llm(self) -> BaseLLM:
        from core.llm import OfflineLLM

        return OfflineLLM(
            backend=self.backend,
            use_lora=self.use_lora,
            settings_module=self.settings,
        )


class DeployPipeline(RAGPipeline):
    """RAG pipeline for Groq cloud generation."""

    def _create_llm(self) -> BaseLLM:
        from core.llm import DeployLLM

        return DeployLLM(settings_module=self.settings)
