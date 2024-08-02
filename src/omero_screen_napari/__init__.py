import logging
import os
from pathlib import Path

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"


def setup_logging():
    # Set a less verbose level for the root logger
    logging.basicConfig(level=logging.WARNING)

    # Create and configure your application's main logger
    app_logger_name = "omero-screen-napari"
    app_logger = logging.getLogger(app_logger_name)
    app_logger.setLevel(logging.DEBUG)  # Set to DEBUG or any other level

    # Prevent propagation to the root logger
    app_logger.propagate = False

    # Create a console handler for the logger
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)  # Ensure it captures all levels processed by the logger
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s")
    ch.setFormatter(formatter)
    app_logger.addHandler(ch)

# Ensure this is called when your package is imported
setup_logging()

def set_env_vars():
    env_path = Path(__file__).resolve().parent.parent / ".env"
    localenv_path = Path(__file__).resolve().parent.parent / ".localenv"
    return localenv_path if os.getenv("USE_LOCAL_ENV") == "1" else env_path

