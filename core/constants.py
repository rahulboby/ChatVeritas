"""
core/constants.py

Project-wide constants and environment configuration helpers.

Thread limits
-------------
BLAS, OpenMP, MKL, and tqdm each spawn background monitor threads that
can crash in multi-threaded environments (e.g. Streamlit).  These must be
capped to 1 and disabled **before** NumPy, FAISS, sentence-transformers,
or HuggingFace Transformers are imported for the first time.

Usage in every entry point (must be the very first project-level call):

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from core.constants import apply_thread_limits
    apply_thread_limits()

    # — heavy imports follow —
"""

import os

# ---------------------------------------------------------------------------
# Thread-limit environment variables
# ---------------------------------------------------------------------------

THREAD_LIMIT_ENV_VARS: dict[str, str] = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "TQDM_DISABLE": "1",
    "TRANSFORMERS_VERBOSITY": "error",
}


def apply_thread_limits() -> None:
    """
    Set thread-limit environment variables.

    Prevents tqdm and BLAS monitor threads from crashing in multi-threaded
    runtime environments such as Streamlit.

    **Must be called before importing NumPy, FAISS, sentence-transformers,
    or HuggingFace Transformers.**
    """
    for key, value in THREAD_LIMIT_ENV_VARS.items():
        os.environ.setdefault(key, value)
