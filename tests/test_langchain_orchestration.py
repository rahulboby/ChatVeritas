"""Behavioral tests for the LCEL orchestration layer.

These tests use the public custom-component contracts so no model, FAISS index,
or external API is required.
"""

import unittest
from unittest.mock import patch

from llm.base import BaseLLM
from pipelines.deploy_pipeline import DeployPipeline
from pipelines.offline_pipeline import OfflinePipeline
from pipelines.rag_pipeline import RAGPipeline
from prompts.templates import RAG_PROMPT_TEMPLATE, build_rag_prompt


class FakeRetriever:
    def retrieve(self, question: str) -> dict:
        self.question = question
        return {
            "results": [
                {
                    "chunk": "first context",
                    "source": "first.txt",
                    "chunk_id": 1,
                    "distance": 0.1,
                    "rerank_score": 1.0,
                },
                {
                    "chunk": "second context",
                    "source": "second.txt",
                    "chunk_id": 2,
                    "distance": 0.2,
                    "rerank_score": 0.9,
                },
            ],
            "metrics": {"embedding_time_ms": 1.0, "retrieval_time_ms": 2.0},
        }


class FakeLLM(BaseLLM):
    def generate(self, prompt: str) -> tuple[str, dict]:
        self.prompt = prompt
        return "unchanged response", {"generation_time": 0.5, "prompt_tokens": 12}


class FakePipeline(RAGPipeline):
    def _create_retriever(self):
        self.fake_retriever = FakeRetriever()
        return self.fake_retriever

    def _create_llm(self) -> BaseLLM:
        self.fake_llm = FakeLLM()
        return self.fake_llm


class LangChainOrchestrationTests(unittest.TestCase):
    def test_prompt_template_preserves_prompt_text(self):
        expected = """You are an expert technical assistant answering questions about the provided documents.
Use the retrieved context as your PRIMARY source of information.
Guidelines:
1. Base your answer primarily on the provided context.
2. If the answer is explicitly stated in the context, answer confidently.
3. If the answer is not explicitly stated but can be reasonably inferred, clearly state it is an inference.
4. Only respond with "I don't have enough information in the provided documents." if the context is insufficient.
5. Never invent facts.

Context:
context

Question:
question

Answer:
"""
        self.assertEqual(build_rag_prompt("question", "context"), expected)
        self.assertEqual(
            RAG_PROMPT_TEMPLATE.format(question="question", context="context"),
            expected,
        )

    def test_lcel_pipeline_preserves_custom_component_contracts(self):
        pipeline = FakePipeline(config={})

        result = pipeline.run("What happened?")

        self.assertEqual(pipeline.fake_retriever.question, "What happened?")
        self.assertEqual(result["response"], "unchanged response")
        self.assertEqual(result["chunks"][0]["rerank_score"], 1.0)
        self.assertEqual(result["metrics"]["generation_time"], 0.5)
        self.assertEqual(result["metrics"]["prompt_tokens"], 12)
        self.assertEqual(
            pipeline.fake_llm.prompt,
            build_rag_prompt("What happened?", "first context\n\nsecond context"),
        )

    def test_offline_and_deploy_pipelines_use_the_shared_lcel_orchestrator(self):
        for pipeline_type in (OfflinePipeline, DeployPipeline):
            with (
                patch.object(pipeline_type, "_create_retriever", return_value=FakeRetriever()),
                patch.object(pipeline_type, "_create_llm", return_value=FakeLLM()),
            ):
                if pipeline_type is OfflinePipeline:
                    pipeline = pipeline_type(config={}, use_lora=False)
                else:
                    pipeline = pipeline_type(config={})

            result = pipeline.run("Which backend?")

            self.assertEqual(result["response"], "unchanged response")
            self.assertIsInstance(pipeline, RAGPipeline)


if __name__ == "__main__":
    unittest.main()
