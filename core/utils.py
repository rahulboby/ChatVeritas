"""Utility classes for ingestion and synthetic fine-tuning data generation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable

SYSTEM_PROMPT = """
You are helping generate a high-quality instruction tuning dataset.

You will be given ONE documentation paragraph.

Generate:

1. A concise topic (2-5 words).
2. Up to FIVE diverse user questions.

Rules:
- Every question MUST be answerable entirely from the paragraph.
- Questions should be naturally phrased.
- Do NOT copy sentences directly.
- Do NOT invent information.
- Do NOT answer the questions.

Include factual, conceptual, beginner, practical, and troubleshooting styles.
Return ONLY valid JSON with a non-empty ``topic`` and ``questions`` list.
""".strip()


class CacheManager:
    """Small atomic disk cache keyed by a SHA-256 hash of a source chunk."""

    def __init__(self, cache_directory: str | Path) -> None:
        self.cache_dir = Path(cache_directory)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _cache_path(self, text: str) -> Path:
        return self.cache_dir / f"{self._hash(text)}.json"

    def exists(self, text: str) -> bool:
        return self._cache_path(text).is_file()

    def load(self, text: str) -> dict[str, Any]:
        with self._cache_path(text).open("r", encoding="utf-8") as cache_file:
            data = json.load(cache_file)
        if not isinstance(data, dict):
            raise ValueError("Cache entries must be JSON objects.")
        return data

    def save(self, text: str, data: dict[str, Any]) -> None:
        path = self._cache_path(text)
        temporary_path = path.with_suffix(".tmp")
        with temporary_path.open("w", encoding="utf-8") as cache_file:
            json.dump(data, cache_file, ensure_ascii=False, indent=2)
        os.replace(temporary_path, path)

    def get(self, text: str) -> dict[str, Any] | None:
        return self.load(text) if self.exists(text) else None

    def get_or_create(
        self,
        text: str,
        generator_function: Callable[[str], dict[str, Any]],
        validator: Callable[[Any], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        try:
            cached = self.get(text)
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            cached = None
        if cached is not None:
            try:
                return validator(cached) if validator is not None else cached
            except (TypeError, ValueError):
                pass

        generated = generator_function(text)
        data = validator(generated) if validator is not None else generated
        self.save(text, data)
        return data

    def clear(self) -> None:
        """Remove only JSON entries owned by this cache directory."""
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()

    def count(self) -> int:
        return sum(1 for _ in self.cache_dir.glob("*.json"))


class ParagraphChunker:
    """Split text by paragraphs and sentence boundaries within a token limit."""

    def __init__(
        self,
        model_name: str,
        max_tokens: int = 450,
        min_paragraph_length: int = 40,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero.")
        if min_paragraph_length < 0:
            raise ValueError("min_paragraph_length cannot be negative.")
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.max_tokens = max_tokens
        self.min_paragraph_length = min_paragraph_length

    def chunk_document(self, text: str) -> list[str]:
        chunks: list[str] = []
        for paragraph in self._split_paragraphs(text):
            if len(paragraph) < self.min_paragraph_length:
                continue
            if self.token_count(paragraph) <= self.max_tokens:
                chunks.append(paragraph)
            else:
                chunks.extend(self._split_large_paragraph(paragraph))
        return chunks

    @staticmethod
    def _split_paragraphs(text: str) -> list[str]:
        return [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]

    def _split_large_paragraph(self, paragraph: str) -> list[str]:
        chunks: list[str] = []
        current: list[str] = []
        for sentence in self._split_sentences(paragraph):
            if self.token_count(sentence) > self.max_tokens:
                if current:
                    chunks.append(" ".join(current))
                    current = []
                chunks.extend(self._force_split_long_sentence(sentence))
                continue
            candidate = " ".join([*current, sentence])
            if self.token_count(candidate) <= self.max_tokens:
                current.append(sentence)
            else:
                if current:
                    chunks.append(" ".join(current))
                current = [sentence]
        if current:
            chunks.append(" ".join(current))
        return chunks

    @staticmethod
    def _split_sentences(paragraph: str) -> list[str]:
        return [part.strip() for part in re.split(r"(?<=[.!?])\s+", paragraph) if part.strip()]

    def _force_split_long_sentence(self, sentence: str) -> list[str]:
        chunks: list[str] = []
        current: list[str] = []
        for word in sentence.split():
            word_tokens = self.token_count(word)
            if self.token_count(" ".join([*current, word])) <= self.max_tokens:
                current.append(word)
                continue
            if current:
                chunks.append(" ".join(current))
                current = []
            if word_tokens <= self.max_tokens:
                current = [word]
                continue
            token_ids = self.tokenizer.encode(word, add_special_tokens=False)
            for start in range(0, len(token_ids), self.max_tokens):
                chunks.append(
                    self.tokenizer.decode(
                        token_ids[start : start + self.max_tokens], skip_special_tokens=True
                    )
                )
        if current:
            chunks.append(" ".join(current))
        return chunks

    def token_count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))


class QuestionGenerator:
    """Generate validated synthetic questions using an OpenAI-compatible API."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        temperature: float = 0.7,
        max_retries: int = 3,
        response_format: dict[str, Any] | None = None,
    ) -> None:
        if max_retries <= 0:
            raise ValueError("max_retries must be greater than zero.")
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.response_format = response_format

    def generate(self, paragraph: str) -> dict[str, Any]:
        if not isinstance(paragraph, str) or not paragraph.strip():
            raise ValueError("The source paragraph cannot be empty.")
        for attempt in range(self.max_retries):
            try:
                request: dict[str, Any] = {
                    "model": self.model,
                    "temperature": self.temperature,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": paragraph},
                    ],
                }
                if self.response_format is not None:
                    request["response_format"] = self.response_format
                response = self.client.chat.completions.create(**request)
                content = response.choices[0].message.content
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("The LLM returned an empty response.")
                return self._validate_response(content)
            except Exception as exc:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"Question generation failed: {exc}") from exc
                time.sleep(2**attempt)
        raise RuntimeError("Question generation failed.")

    def _validate_response(self, response: str) -> dict[str, Any]:
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            data = json.loads(re.sub(r"```json|```", "", response).strip())
        return self.validate_data(data)

    def validate_data(self, data: Any) -> dict[str, Any]:
        if not isinstance(data, dict):
            raise ValueError("The response must be a JSON object.")
        topic_value = data.get("topic", "")
        if not isinstance(topic_value, str) or not topic_value.strip():
            raise ValueError("Topic must be a non-empty string.")
        questions_value = data.get("questions", [])
        if not isinstance(questions_value, list):
            raise ValueError("Questions must be a list.")
        questions: list[str] = []
        seen: set[str] = set()
        for question in questions_value:
            if not isinstance(question, str) or not question.strip():
                continue
            question = question.strip()
            if question.lower() in seen:
                continue
            seen.add(question.lower())
            questions.append(question)
            if len(questions) == 5:
                break
        if not questions:
            raise ValueError("No questions generated.")
        return {"topic": topic_value.strip(), "questions": questions}
