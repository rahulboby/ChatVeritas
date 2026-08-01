"""
core/config.py

Configuration loader for the ChatVeritas application.

Loads ``config/config.json`` from the project root.  The project root is
resolved at import time as two directories up from this file:

    core/config.py
        → core/
        → <project root>

This is the single configuration loader used by runtime and preprocessing
entry points.
"""

import json
from pathlib import Path

# Two levels up: core/ → project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "config" / "config.json"


def load_config() -> dict:
    """
    Load and return the application configuration from ``config/config.json``.

    Returns
    -------
    dict
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If ``config/config.json`` does not exist.
    json.JSONDecodeError
        If the file is not valid JSON.
    """
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
