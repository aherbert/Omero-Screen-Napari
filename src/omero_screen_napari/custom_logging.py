import logging


def setup_logging():
    # Configure your logging here (handlers, formatters, etc.)
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    )
    