"""
utils/question_generator.py

Generates synthetic instruction-tuning questions from a document chunk
using a stronger LLM hosted on Groq.

Returned format:

{
    "topic": "...",
    "questions": [
        "...",
        "...",
        ...
    ]
}
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from groq import Groq


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

Make the questions diverse.

Include different styles such as:

- factual
- conceptual
- beginner
- practical
- troubleshooting

Return ONLY valid JSON.

Example:

{
  "topic": "Data Trust Score",
  "questions": [
    "...",
    "...",
    "...",
    "...",
    "..."
  ]
}
""".strip()


class QuestionGenerator:

    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float = 0.7,
        max_retries: int = 3,
    ):

        self.client = Groq(api_key=api_key)

        self.model = model

        self.temperature = temperature

        if max_retries <= 0:
            raise ValueError("max_retries must be greater than zero.")

        self.max_retries = max_retries

    # ---------------------------------------------------------

    def generate(
        self,
        paragraph: str
    ) -> dict[str, Any]:
        """
        Generate topic + diverse questions.

        Returns
        -------
        dict
            {
                "topic": "...",
                "questions": [...]
            }
        """

        if not paragraph.strip():
            raise ValueError("The source paragraph cannot be empty.")

        for attempt in range(self.max_retries):

            try:

                response = self.client.chat.completions.create(

                    model=self.model,

                    temperature=self.temperature,

                    response_format={
                        "type": "json_object"
                    },

                    messages=[
                        {
                            "role": "system",
                            "content": SYSTEM_PROMPT
                        },
                        {
                            "role": "user",
                            "content": paragraph
                        }
                    ]
                )

                content = response.choices[0].message.content

                if not content:
                    raise ValueError("The LLM returned an empty response.")

                return self._validate_response(content)

            except Exception as e:

                if attempt == self.max_retries - 1:
                    raise RuntimeError(
                        f"Question generation failed: {e}"
                    )

                time.sleep(2 ** attempt)

        raise RuntimeError(
            "Question generation failed."
        )

    # ---------------------------------------------------------

    def _validate_response(
        self,
        response: str
    ) -> dict[str, Any]:
        """
        Validate the LLM response.
        """

        try:

            data = json.loads(response)

        except json.JSONDecodeError:

            # Occasionally models wrap JSON
            # in markdown code blocks.

            response = re.sub(
                r"```json|```",
                "",
                response
            ).strip()

            data = json.loads(response)

        return self.validate_data(data)

    def validate_data(
        self,
        data: Any
    ) -> dict[str, Any]:
        """Validate and normalize a parsed question-generation response."""

        if not isinstance(data, dict):
            raise ValueError("The response must be a JSON object.")

        topic_value = data.get("topic", "")

        if not isinstance(topic_value, str) or not topic_value.strip():
            raise ValueError("Topic must be a non-empty string.")

        topic = topic_value.strip()

        questions = data.get("questions", [])

        if not isinstance(questions, list):
            raise ValueError(
                "Questions must be a list."
            )

        cleaned_questions = []

        seen = set()

        for question in questions:

            if not isinstance(question, str):
                continue

            question = question.strip()

            if not question:
                continue

            if question.lower() in seen:
                continue

            seen.add(question.lower())

            cleaned_questions.append(question)

            if len(cleaned_questions) == 5:
                break

        if len(cleaned_questions) == 0:
            raise ValueError(
                "No questions generated."
            )

        return {
            "topic": topic,
            "questions": cleaned_questions
        }
