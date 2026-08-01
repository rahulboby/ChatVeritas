"""
core/logger.py

Centralised logging setup for the ChatVeritas application.

All runtime modules obtain their logger via:

    from core.logger import get_logger
    logger = get_logger(__name__)

This ensures a consistent format across the entire package and makes it
trivial to adjust log levels or add handlers in one place.
"""

import logging
import sys

_FORMAT = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger with a ``StreamHandler`` writing to stdout.

    Repeated calls with the same ``name`` return the same logger instance
    without adding duplicate handlers.

    Parameters
    ----------
    name : str
        Logger name — use ``__name__`` in every module for automatic
        hierarchy (e.g. ``retrieval.retriever``).

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(fmt=_FORMAT, datefmt=_DATE_FORMAT)
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        # Prevent messages from propagating to the root logger to avoid
        # duplicate output when a root handler is also configured.
        logger.propagate = False

    return logger
