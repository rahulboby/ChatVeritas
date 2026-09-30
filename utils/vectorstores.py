"""Purpose: Name vector-store directories and list stores with valid index metadata.
Dependencies: built-in: re, pathlib; installed: none.
Custom: none.
"""

import re
from pathlib import Path


def named_vectorstore_path(project_root, config, name):
    normalized_name = re.sub(r"\s+", "_", name.strip())
    if not normalized_name or not re.fullmatch(r"[\w-]+", normalized_name):
        raise ValueError("Use a name containing only letters, numbers, underscores, or hyphens.")

    base_path = project_root / config["paths"]["vectorstore"]
    return base_path.parent / f"{base_path.name}_{normalized_name}"


def list_vectorstores(project_root, config):
    base_path = project_root / config["paths"]["vectorstore"]
    candidates = []

    if base_path.is_dir():
        candidates.append(base_path)
    if base_path.parent.is_dir():
        candidates.extend(base_path.parent.glob(f"{base_path.name}_*"))

    return sorted(
        (
            path
            for path in candidates
            if path.is_dir()
            and (path / "index.faiss").is_file()
            and (path / "chunks.pkl").is_file()
        ),
        key=lambda path: path.name.lower(),
    )