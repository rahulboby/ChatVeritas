import json
import tempfile
import unittest
from pathlib import Path

from scripts.fine_tune import split_by_chunk, validate_config, validate_dataset


class FineTuneDatasetTests(unittest.TestCase):
    def test_dataset_is_normalized(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset_path = Path(directory) / "train.jsonl"
            dataset_path.write_text(
                json.dumps(
                    {
                        "source_file": "source.txt",
                        "chunk_id": 7,
                        "messages": [
                            {"role": "user", "content": " Question? "},
                            {"role": "assistant", "content": " Answer. "},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            records = validate_dataset(dataset_path)

            self.assertEqual(records[0]["group"], "source.txt:7")
            self.assertEqual(records[0]["messages"][0]["content"], "Question?")

    def test_chunk_groups_do_not_leak_between_splits(self):
        records = [
            {"group": "a:1", "messages": [{"role": "user", "content": str(i)}]}
            for i in range(5)
        ] + [
            {"group": "b:2", "messages": [{"role": "user", "content": str(i)}]}
            for i in range(5, 10)
        ]

        train, validation = split_by_chunk(records, validation_fraction=0.5, seed=42)

        train_groups = {record["group"] for record in train}
        validation_groups = {record["group"] for record in validation}
        self.assertFalse(train_groups & validation_groups)
        self.assertEqual(len(train) + len(validation), len(records))


class FineTuneConfigTests(unittest.TestCase):
    def test_missing_lora_config_is_rejected(self):
        config = {
            "model": {"base_model": "model", "adapter_path": "models/adapters"},
            "paths": {
                "processed_data": "train.jsonl",
                "training_checkpoints": "models/checkpoints",
            },
            "training": {
                "epochs": 1,
                "batch_size": 1,
                "gradient_accumulation": 1,
                "learning_rate": 0.001,
                "max_sequence_length": 128,
            },
        }

        with self.assertRaises(ValueError):
            validate_config(config)


if __name__ == "__main__":
    unittest.main()
