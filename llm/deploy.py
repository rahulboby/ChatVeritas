"""
llm/deploy.py

Deployment LLM backend for ChatVeritas.

Uses the Groq API (via the OpenAI-compatible client) for language model
inference.  Requires ``GROQ_API_KEY`` to be present in the environment
(typically loaded from ``.env`` by the entry point).
"""

import os
import time

from openai import OpenAI

from core.exceptions import ChatVeritasError
from core.logger import get_logger
from llm.base import BaseLLM

logger = get_logger(__name__)

# System message used for every Groq API call.
_SYSTEM_MESSAGE = (
    "You are ChatVeritas, a document-grounded AI assistant. "
    "Answer only using the supplied context. "
    "If the answer is not present, clearly state that there "
    "is insufficient information."
)


class DeployLLM(BaseLLM):
    """
    Cloud inference backend using the Groq API.

    Parameters
    ----------
    config : dict
        Loaded application configuration dictionary.  Reads:
            - ``config["generation"]["model"]``        — Groq model ID
            - ``config["generation"]["temperature"]``  — sampling temperature
            - ``config["generation"]["max_new_tokens"]``— token budget
    """

    def __init__(self, config: dict) -> None:
        self.config = config

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ChatVeritasError(
                "GROQ_API_KEY not found in environment variables. "
                "Ensure it is set in your .env file or shell environment."
            )

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )

        logger.info(
            "DeployLLM initialised | model=%s",
            config["generation"].get("model", "openai/gpt-oss-120b"),
        )

    # ------------------------------------------------------------------
    # BaseLLM implementation
    # ------------------------------------------------------------------

    def generate(self, prompt: str) -> tuple[str, dict]:
        """
        Generate a response via the Groq API.

        Parameters
        ----------
        prompt : str
            The assembled RAG prompt.

        Returns
        -------
        tuple[str, dict]
            ``(response_text, {"generation_time": float, "prompt_tokens": int})``

        Raises
        ------
        RuntimeError
            If the Groq API call fails.
        """
        model_id = self.config["generation"].get("model", "openai/gpt-oss-120b")
        temperature = self.config["generation"]["temperature"]
        max_tokens = self.config["generation"]["max_new_tokens"]

        logger.debug(
            "Groq API request | model=%s | temperature=%.2f | max_tokens=%d",
            model_id,
            temperature,
            max_tokens,
        )

        gen_start = time.perf_counter()

        try:
            completion = self.client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": _SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            raise RuntimeError(f"Groq API request failed: {e}") from e

        generation_time = time.perf_counter() - gen_start
        response = completion.choices[0].message.content

        metrics = {
            "generation_time": generation_time,
            "prompt_tokens": completion.usage.prompt_tokens,
        }

        logger.debug(
            "Groq API response received | %.2f s | %d prompt tokens",
            generation_time,
            metrics["prompt_tokens"],
        )

        return response.strip(), metrics
