"""
utils/paragraph_chunker.py

Utilities for splitting raw TXT files into semantic chunks suitable for
both RAG and fine-tuning.

Pipeline:
TXT File
    ↓
Paragraph Split
    ↓
Remove Empty Paragraphs
    ↓
If paragraph <= max_tokens:
    Keep
Else:
    Split by sentence boundaries
"""

from __future__ import annotations

import re
from typing import List

from transformers import AutoTokenizer


class ParagraphChunker:
    """
    Chunks documents paragraph-wise while respecting sentence boundaries.

    Large paragraphs are split into multiple chunks without breaking
    sentences.
    """

    def __init__(
        self,
        model_name: str,
        max_tokens: int = 450,
        min_paragraph_length: int = 40,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero.")

        if min_paragraph_length < 0:
            raise ValueError("min_paragraph_length cannot be negative.")

        self.max_tokens = max_tokens
        self.min_paragraph_length = min_paragraph_length

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def chunk_document(self, text: str) -> List[str]:
        """
        Split an entire document into semantic chunks.

        Parameters
        ----------
        text : str
            Raw document text.

        Returns
        -------
        list[str]
            Semantic chunks.
        """

        paragraphs = self._split_paragraphs(text)

        chunks: List[str] = []

        for paragraph in paragraphs:

            if len(paragraph) < self.min_paragraph_length:
                continue

            if self.token_count(paragraph) <= self.max_tokens:
                chunks.append(paragraph)

            else:
                chunks.extend(
                    self._split_large_paragraph(paragraph)
                )

        return chunks

    # ---------------------------------------------------------
    # Paragraph splitting
    # ---------------------------------------------------------

    def _split_paragraphs(self, text: str) -> List[str]:
        """
        Split text into paragraphs.

        Paragraphs are separated by one or more blank lines.
        """

        paragraphs = re.split(r"\n\s*\n", text)

        cleaned = [
            p.strip()
            for p in paragraphs
            if p.strip()
        ]

        return cleaned

    # ---------------------------------------------------------
    # Large paragraph handling
    # ---------------------------------------------------------

    def _split_large_paragraph(
        self,
        paragraph: str
    ) -> List[str]:
        """
        Split a large paragraph while preserving sentence boundaries.
        """

        sentences = self._split_sentences(paragraph)

        chunks = []

        current_sentences = []

        for sentence in sentences:

            sentence_tokens = self.token_count(sentence)

            # Extremely rare:
            # one sentence itself exceeds max_tokens
            if sentence_tokens > self.max_tokens:

                if current_sentences:

                    chunks.append(
                        " ".join(current_sentences).strip()
                    )

                    current_sentences = []

                chunks.extend(
                    self._force_split_long_sentence(sentence)
                )

                continue

            candidate = " ".join([*current_sentences, sentence])

            if self.token_count(candidate) <= self.max_tokens:

                current_sentences.append(sentence)

            else:

                chunks.append(
                    " ".join(current_sentences).strip()
                )

                current_sentences = [sentence]

        if current_sentences:

            chunks.append(
                " ".join(current_sentences).strip()
            )

        return chunks

    # ---------------------------------------------------------
    # Sentence splitting
    # ---------------------------------------------------------

    def _split_sentences(
        self,
        paragraph: str
    ) -> List[str]:
        """
        Lightweight sentence splitter.

        Splits on:

            .
            !
            ?

        while keeping punctuation.
        """

        sentences = re.split(
            r'(?<=[.!?])\s+',
            paragraph
        )

        sentences = [
            s.strip()
            for s in sentences
            if s.strip()
        ]

        return sentences

    # ---------------------------------------------------------
    # Extremely long sentence fallback
    # ---------------------------------------------------------

    def _force_split_long_sentence(
        self,
        sentence: str
    ) -> List[str]:
        """
        If a single sentence exceeds max_tokens,
        split it by words.

        This should almost never happen.
        """

        words = sentence.split()

        chunks = []

        current = []

        for word in words:

            word_tokens = self.token_count(word)

            candidate = " ".join([*current, word])

            if self.token_count(candidate) <= self.max_tokens:

                current.append(word)

            else:

                if current:
                    chunks.append(
                        " ".join(current)
                    )

                if word_tokens <= self.max_tokens:
                    current = [word]
                    continue

                token_ids = self.tokenizer.encode(
                    word,
                    add_special_tokens=False
                )

                for start in range(0, len(token_ids), self.max_tokens):
                    chunks.append(
                        self.tokenizer.decode(
                            token_ids[start:start + self.max_tokens],
                            skip_special_tokens=True
                        )
                    )

                current = []

        if current:

            chunks.append(
                " ".join(current)
            )

        return chunks

    # ---------------------------------------------------------
    # Token utilities
    # ---------------------------------------------------------

    def token_count(
        self,
        text: str
    ) -> int:
        """
        Return the number of tokens using the model tokenizer.
        """

        return len(
            self.tokenizer.encode(
                text,
                add_special_tokens=False
            )
        )
