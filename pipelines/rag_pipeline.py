"""Shared LCEL orchestration for the existing ChatVeritas RAG components.

This module deliberately orchestrates existing components instead of replacing
them. ``Retriever`` continues to perform query embedding, FAISS search, and
cross-encoder reranking; the selected LLM continues to load and generate with
the project's original backend code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from operator import itemgetter
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough

from core.logger import get_logger
from llm.base import BaseLLM
from prompts.templates import RAG_PROMPT_TEMPLATE

if TYPE_CHECKING:
    from retrieval.retriever import Retriever

logger = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class RAGPipeline(ABC):
    """LCEL orchestration around the project's custom RAG implementation."""

    def __init__(self, config: dict) -> None:
        self.config = config
        # Preserve the original startup order: retrieval resources first, then
        # the selected LLM backend.
        self.retriever = self._create_retriever()
        self.llm = self._create_llm()
        self.chain = self._build_chain()

    def _create_retriever(self) -> "Retriever":
        """Instantiate the unchanged custom FAISS plus cross-encoder retriever."""
        # Keep heavy FAISS and sentence-transformer imports deferred until a
        # real pipeline is created. Application entry points set thread limits
        # before instantiation.
        from retrieval.retriever import Retriever

        return Retriever(
            index_path=_PROJECT_ROOT / self.config["paths"]["vectorstore"] / "index.faiss",
            chunks_path=_PROJECT_ROOT / self.config["paths"]["vectorstore"] / "chunks.pkl",
            embedding_model=self.config["embedding"]["model"],
            top_k=self.config["retrieval"]["top_k"],
            faiss_candidates=self.config["retrieval"]["faiss_candidates"],
            embedding_device=self.config["embedding"].get("device", "cpu"),
            reranker_model=self.config["reranker"]["model"],
            reranker_device=self.config["reranker"].get("device", "cpu"),
        )

    @abstractmethod
    def _create_llm(self) -> BaseLLM:
        """Return the mode-specific existing LLM backend."""

    def _build_chain(self) -> Runnable[dict[str, str], dict[str, Any]]:
        """Compose retrieval, prompt rendering, generation, and parsing with LCEL."""
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
            | RunnableLambda(lambda generation: generation["response"], name="generation_text")
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

    def run(self, question: str) -> dict:
        """Execute the unchanged RAG flow through the LCEL orchestration chain."""
        logger.info("%s running query: %r", self.__class__.__name__, question)
        result = self.chain.invoke({"question": question})
        logger.info(
            "%s complete - generation: %.2f s.",
            self.__class__.__name__,
            result["metrics"]["generation_time"],
        )
        return result

    def _retrieve(self, state: dict[str, Any]) -> dict:
        """Call the custom two-stage retriever without adapting its algorithm."""
        return self.retriever.retrieve(state["question"])

    @staticmethod
    def _build_context(state: dict[str, Any]) -> str:
        """Preserve the original chunk-to-context conversion."""
        return "\n\n".join(item["chunk"] for item in state["retrieval"]["results"])

    @staticmethod
    def _prompt_inputs(state: dict[str, Any]) -> dict[str, str]:
        """Pass only the original prompt variables to PromptTemplate."""
        return {"question": state["question"], "context": state["context"]}

    @staticmethod
    def _format_result(state: dict[str, Any]) -> dict:
        """Return the existing public pipeline result contract unchanged."""
        metrics = state["retrieval"]["metrics"]
        metrics.update(state["generation"]["metrics"])
        return {
            "response": state["response"],
            "chunks": state["retrieval"]["results"],
            "metrics": metrics,
        }
