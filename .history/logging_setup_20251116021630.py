import logging
import os

_DEF_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def setup_logging() -> None:
    """Configure root logging once.

    Environment variables:
    - GESTURE_LOG_LEVEL: logging level name (default INFO)
    - GESTURE_LOG_FORMAT: optional format string.
    """
    level_name = os.environ.get("GESTURE_LOG_LEVEL", "INFO").upper()
    fmt = os.environ.get("GESTURE_LOG_FORMAT", _DEF_FORMAT)
    level = getattr(logging, level_name, logging.INFO)
    if len(logging.getLogger().handlers) == 0:  # idempotent
        logging.basicConfig(level=level, format=fmt)
    else:
        logging.getLogger().setLevel(level)
