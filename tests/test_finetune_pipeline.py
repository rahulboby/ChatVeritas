import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from utils.cache import CacheManager
from utils.paragraph_chunker import ParagraphChunker
from utils.question_generator import QuestionGenerator
from scripts import prepare_finetune_dataset as prepare


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text)

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(token_ids)


class ParagraphChunkerTests(unittest.TestCase):
    @patch("utils.paragraph_chunker.AutoTokenizer.from_pretrained")
    def test_every_chunk_respects_token_limit(self, from_pretrained):
        from_pretrained.return_value = FakeTokenizer()
        chunker = ParagraphChunker("fake-model", max_tokens=12, min_paragraph_length=0)

        chunks = chunker.chunk_document(
            "A short one. Another sentence here.\n\nabcdefghijklmnopqrstuv"
        )

        self.assertTrue(chunks)
        self.assertTrue(all(chunks))
        self.assertTrue(all(chunker.token_count(chunk) <= 12 for chunk in chunks))

    @patch("utils.paragraph_chunker.AutoTokenizer.from_pretrained")
    def test_invalid_limits_are_rejected(self, from_pretrained):
        from_pretrained.return_value = FakeTokenizer()

        with self.assertRaises(ValueError):
            ParagraphChunker("fake-model", max_tokens=0)


class QuestionGeneratorTests(unittest.TestCase):
    def setUp(self):
        self.generator = QuestionGenerator.__new__(QuestionGenerator)

    def test_response_is_cleaned_deduplicated_and_limited(self):
        response = json.dumps(
            {
                "topic": "  Retrieval  ",
                "questions": [
                    " First? ",
                    "first?",
                    42,
                    "Second?",
                    "Third?",
                    "Fourth?",
                    "Fifth?",
                    "Sixth?",
                ],
            }
        )

        result = self.generator._validate_response(response)

        self.assertEqual(result["topic"], "Retrieval")
        self.assertEqual(len(result["questions"]), 5)
        self.assertEqual(result["questions"][0], "First?")

    def test_empty_topic_is_rejected(self):
        with self.assertRaises(ValueError):
            self.generator._validate_response('{"topic": "", "questions": ["Why?"]}')


class CacheManagerTests(unittest.TestCase):
    def test_corrupt_entry_is_regenerated(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = CacheManager(directory)
            cache_path = cache._cache_path("chunk")
            cache_path.write_text("not json", encoding="utf-8")

            result = cache.get_or_create(
                "chunk",
                lambda text: {"topic": "Test", "questions": [text]},
            )

            self.assertEqual(result["questions"], ["chunk"])
            self.assertEqual(cache.load("chunk"), result)

    def test_invalid_cached_schema_is_regenerated(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = CacheManager(directory)
            cache.save("chunk", {"unexpected": True})
            generator = QuestionGenerator.__new__(QuestionGenerator)

            result = cache.get_or_create(
                "chunk",
                lambda text: {"topic": "Test", "questions": [text]},
                validator=generator.validate_data,
            )

            self.assertEqual(result["topic"], "Test")


class ConfigurationTests(unittest.TestCase):
    def test_processed_dataset_path_is_configured(self):
        config_path = Path(__file__).resolve().parents[1] / "config" / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(config["paths"]["processed_data"], "data/processed/train.jsonl")


class DatasetPreparationTests(unittest.TestCase):
    def test_main_writes_chat_jsonl_without_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            raw_dir = project_root / "raw"
            raw_dir.mkdir()
            (raw_dir / "source.txt").write_text("Useful source text.", encoding="utf-8")

            config = {
                "paths": {
                    "raw_data": "raw",
                    "processed_data": "processed/train.jsonl",
                },
                "dataset": {
                    "max_chunk_tokens": 20,
                    "minimum_paragraph_length": 0,
                },
                "llm": {
                    "provider": "groq",
                    "tokenizer_model": "fake",
                    "model": "fake",
                    "temperature": 0.1,
                    "max_retries": 1,
                },
                "cache": {"enabled": False, "directory": "cache"},
            }

            with (
                patch.object(prepare, "PROJECT_ROOT", project_root),
                patch.object(prepare, "load_config", return_value=config),
                patch.object(prepare, "load_dotenv"),
                patch.object(prepare, "ParagraphChunker") as chunker_class,
                patch.object(prepare, "QuestionGenerator") as generator_class,
                patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}),
            ):
                chunker_class.return_value.chunk_document.return_value = ["Answer text"]
                generator_class.return_value.generate.return_value = {
                    "topic": "Topic",
                    "questions": ["Question?"],
                }

                prepare.main()

            output_path = project_root / "processed" / "train.jsonl"
            sample = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(sample["messages"][0]["content"], "Question?")
            self.assertEqual(sample["messages"][1]["content"], "Answer text")


if __name__ == "__main__":
    unittest.main()
