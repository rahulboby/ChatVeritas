import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from utils.model_store import ensure_model_available


class EnsureModelAvailableTests(unittest.TestCase):

    @patch("huggingface_hub.snapshot_download")
    def test_downloads_once_and_reuses_completed_model(self, snapshot_download):
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)

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
            snapshot_download.assert_called_once_with(
                repo_id="sentence-transformers/all-MiniLM-L6-v2",
                local_dir=str(expected_path),
            )

    @patch("huggingface_hub.snapshot_download", side_effect=RuntimeError("offline"))
    def test_failed_download_is_not_marked_complete(self, snapshot_download):
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
            snapshot_download.assert_called_once()


if __name__ == "__main__":
    unittest.main()