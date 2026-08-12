"""Prompt templates owned by the ChatVeritas RAG pipeline."""

from langchain_core.prompts import PromptTemplate

_RAG_PROMPT_TEXT = """You are an expert technical assistant answering questions about the provided documents.
Use the retrieved context as your PRIMARY source of information.
Guidelines:
1. Base your answer primarily on the provided context.
2. If the answer is explicitly stated in the context, answer confidently.
3. If the answer is not explicitly stated but can be reasonably inferred, clearly state it is an inference.
4. Only respond with "I don't have enough information in the provided documents." if the context is insufficient.
5. Never invent facts.

Context:
{context}

Question:
{question}

Answer:
"""

RAG_PROMPT_TEMPLATE = PromptTemplate.from_template(_RAG_PROMPT_TEXT)


def build_rag_prompt(question: str, context: str) -> str:
    """Render the standard document-grounded RAG prompt."""
    return RAG_PROMPT_TEMPLATE.format(context=context, question=question)
