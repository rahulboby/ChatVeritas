"""LangChain adapters for ChatVeritas LLM backends.

The backends retain their existing ``generate(prompt)`` API. This adapter
preserves both the generated text and the project-specific metrics that would
be lost by forcing the backends into a generic chat-model abstraction.
"""

from typing import Protocol

from langchain_core.runnables import Runnable, RunnableLambda


class SupportsGeneration(Protocol):
    """The stable generation contract implemented by every ChatVeritas LLM."""

    def generate(self, prompt: str) -> tuple[str, dict]:
        """Return generated text and generation metrics."""


def create_generation_runnable(
    llm: SupportsGeneration,
) -> Runnable[str, dict[str, object]]:
    """Adapt a ChatVeritas LLM to an LCEL runnable without losing metrics."""

    def invoke_generation(prompt: str) -> dict[str, object]:
        response, metrics = llm.generate(prompt)
        return {"response": response, "metrics": metrics}

    return RunnableLambda(invoke_generation, name="chatveritas_generation")
