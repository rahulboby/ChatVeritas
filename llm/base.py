"""
llm/base.py

Abstract base class for all LLM backends in ChatVeritas.

Every backend must implement ``generate(prompt) -> (response, metrics)``.
This contract ensures the pipeline layer is completely unaware of the
underlying inference engine, while ``as_runnable`` makes each backend
available to LCEL without changing that public contract.

    - swap between local (OfflineLLM) and cloud (DeployLLM) backends,
    - add new backends (e.g. LangChain LLM wrappers) without touching
      any pipeline or interface code.
"""

from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """
    Abstract base class for ChatVeritas LLM backends.

    Subclasses must implement :meth:`generate`.
    """

    @abstractmethod
    def generate(self, prompt: str) -> tuple[str, dict]:
        """
        Generate a response from the given prompt.

        Parameters
        ----------
        prompt : str
            The assembled RAG prompt (system instructions + context +
            question), as returned by
            :func:`prompts.templates.build_rag_prompt`.

        Returns
        -------
        tuple[str, dict]
            A two-element tuple:

            - **response** (``str``): The generated answer text, stripped
              of leading and trailing whitespace.
            - **metrics** (``dict``): Timing and token-usage statistics.
              Guaranteed keys:
                - ``generation_time`` (``float``): Wall-clock seconds.
                - ``prompt_tokens``   (``int``):   Tokens in the prompt.
        """
        ...

    def as_runnable(self):
        """Expose this backend as an LCEL runnable while preserving metrics."""
        from llm.runnables import create_generation_runnable

        return create_generation_runnable(self)
