"""
utils/cache.py

Simple disk-based cache for synthetic dataset generation.

Each document chunk is hashed using SHA-256 and stored as an
individual JSON file.

Benefits:
- Resume interrupted runs
- Avoid duplicate API calls
- Save API credits
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable


class CacheManager:
    """
    Simple SHA-256 file cache.
    """

    def __init__(self, cache_directory: str | Path):

        self.cache_dir = Path(cache_directory)

        self.cache_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    # ---------------------------------------------------------

    def _hash(self, text: str) -> str:
        """
        Compute SHA-256 hash for a chunk.
        """

        return hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest()

    # ---------------------------------------------------------

    def _cache_path(self, text: str) -> Path:
        """
        Return cache filename for a chunk.
        """

        filename = self._hash(text) + ".json"

        return self.cache_dir / filename

    # ---------------------------------------------------------

    def exists(self, text: str) -> bool:
        """
        Check whether the chunk has already been cached.
        """

        return self._cache_path(text).exists()

    # ---------------------------------------------------------

    def load(self, text: str) -> dict[str, Any]:
        """
        Load cached response.

        Raises
        ------
        FileNotFoundError
            If cache entry does not exist.
        """

        path = self._cache_path(text)

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    # ---------------------------------------------------------

    def save(
        self,
        text: str,
        data: dict[str, Any]
    ) -> None:
        """
        Save response to cache.
        """

        path = self._cache_path(text)

        temporary_path = path.with_suffix(".tmp")

        with open(
            temporary_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                data,
                f,
                ensure_ascii=False,
                indent=4
            )

        os.replace(temporary_path, path)

    # ---------------------------------------------------------

    def get(
        self,
        text: str
    ) -> dict[str, Any] | None:
        """
        Convenience method.

        Returns cached data if available,
        otherwise returns None.
        """

        if not self.exists(text):
            return None

        return self.load(text)

    # ---------------------------------------------------------

    def get_or_create(
        self,
        text: str,
        generator_function: Callable[[str], dict[str, Any]],
        validator: Callable[[Any], dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        """
        Get cached data if available.

        Otherwise call the provided generator function,
        cache its output, and return it.

        Parameters
        ----------
        text : str
            Input document chunk.

        generator_function : Callable[[str], dict]
            Function that generates the JSON response.

        Returns
        -------
        dict
            Cached or newly generated response.
        """

        try:
            cached = self.get(text)
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            cached = None

        if cached is not None:
            try:
                return validator(cached) if validator is not None else cached
            except (TypeError, ValueError):
                pass

        data = generator_function(text)

        if validator is not None:
            data = validator(data)

        self.save(text, data)

        return data

    # ---------------------------------------------------------

    def clear(self) -> None:
        """
        Delete all cached files.
        """

        for file in self.cache_dir.glob("*.json"):
            file.unlink()

    # ---------------------------------------------------------

    def count(self) -> int:
        """
        Number of cached entries.
        """

        return len(
            list(self.cache_dir.glob("*.json"))
        )
