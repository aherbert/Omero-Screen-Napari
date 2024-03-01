import logging
import os
from pathlib import Path

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"


def setup_logging():
    # Set a less verbose level for the root logger to avoid noisy logs from external libraries
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    )

    # Create and configure your application's main logger
    app_logger_name = "omero-screen-napari"  # Use a unique name for your application's logger
    app_logger = logging.getLogger(app_logger_name)
    app_logger.setLevel(logging.DEBUG)  # Or DEBUG, as per your requirement

    # Optionally, add any specific handlers/formatters to your app logger here


# Ensure this is called when your package is imported
setup_logging()

def set_env_vars():
    env_path = Path(__file__).resolve().parent.parent / ".env"
    localenv_path = Path(__file__).resolve().parent.parent / ".localenv"
    return localenv_path if os.getenv("USE_LOCAL_ENV") == "1" else env_path

