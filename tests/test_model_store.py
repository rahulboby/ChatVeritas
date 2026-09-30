"""Purpose: Test model downloads, successful local reuse, and retry after failed downloads.
Dependencies: built-in: tempfile, unittest, pathlib; installed: huggingface-hub.
Custom: utils.model_store.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

from huggingface_hub.errors import LocalEntryNotFoundError

from utils.model_store import ensure_model_available


class EnsureModelAvailableTests(unittest.TestCase):

    @patch("huggingface_hub.snapshot_download")
    def test_copies_cached_model_and_reuses_completed_model(self, snapshot_download):
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            cached_snapshot = project_root / "hf-cache" / "snapshot"
            cached_snapshot.mkdir(parents=True)
            (cached_snapshot / "config.json").write_text("{}", encoding="utf-8")
            snapshot_download.return_value = str(cached_snapshot)

            model_path = ensure_model_available(
                "sentence-transformers/all-MiniLM-L6-v2",
                project_root=project_root,
            )
            cached_model_path = ensure_model_available(
                "sentence-transformers/all-MiniLM-L6-v2",
                project_root=project_root,
            )

            expected_path = (
                project_root
                / "data"
                / "models"
                / "sentence-transformers"
                / "all-MiniLM-L6-v2"
            )
            self.assertEqual(model_path, str(expected_path))
            self.assertEqual(cached_model_path, model_path)
            self.assertEqual(
                (expected_path / "config.json").read_text(encoding="utf-8"),
                "{}",
            )
            snapshot_download.assert_called_once_with(
                repo_id="sentence-transformers/all-MiniLM-L6-v2",
                local_files_only=True,
            )

    @patch("huggingface_hub.snapshot_download")
    def test_cache_miss_falls_back_to_download(self, snapshot_download):
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)

            def download(repo_id, **kwargs):
                if kwargs.get("local_files_only"):
                    raise LocalEntryNotFoundError("not cached")
                Path(kwargs["local_dir"], "config.json").write_text(
                    "{}",
                    encoding="utf-8",
                )

            snapshot_download.side_effect = download
            model_path = ensure_model_available(
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
                project_root=project_root,
            )

            self.assertTrue((Path(model_path) / "config.json").is_file())
            snapshot_download.assert_has_calls(
                [
                    call(
                        repo_id="cross-encoder/ms-marco-MiniLM-L-6-v2",
                        local_files_only=True,
                    ),
                    call(
                        repo_id="cross-encoder/ms-marco-MiniLM-L-6-v2",
                        local_dir=model_path,
                    ),
                ]
            )

    @patch("huggingface_hub.snapshot_download")
    def test_failed_download_is_not_marked_complete(self, snapshot_download):
        snapshot_download.side_effect = [
            LocalEntryNotFoundError("not cached"),
            RuntimeError("offline"),
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)

            with self.assertRaisesRegex(RuntimeError, "offline"):
                ensure_model_available(
                    "cross-encoder/ms-marco-MiniLM-L-6-v2",
                    project_root=project_root,
                )

            completion_marker = (
                project_root
                / "data"
                / "models"
                / "cross-encoder"
                / "ms-marco-MiniLM-L-6-v2"
                / ".chatveritas-download-complete"
            )
            self.assertFalse(completion_marker.exists())
            self.assertEqual(snapshot_download.call_count, 2)


if __name__ == "__main__":
    unittest.main()