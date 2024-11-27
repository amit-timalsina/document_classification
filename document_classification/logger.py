import sys
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger as loguru_logger


def setup_logger():  # noqa: ANN201
    """
    Set up and configure the logger for the application.

    This function creates necessary log directories, removes the default logger,
    and sets up multiple loggers for different purposes including stdout,
    run-specific logs, and daily rotating logs.

    Returns:
        loguru.Logger: Configured logger object

    """
    # Ensure the logs directory exists
    Path("logs").mkdir(parents=True, exist_ok=True)

    # Remove the default logger
    loguru_logger.remove()

    # Generate a unique identifier for this run
    run_id = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Add a logger that writes to stdout
    loguru_logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>"
        "{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )

    # Add a logger for this specific run
    loguru_logger.add(
        f"logs/ocr_run_{run_id}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
    )

    # Add a logger for daily rotation (for aggregated logs)
    loguru_logger.add(
        "logs/ocr_daily.log",
        rotation="00:00",
        retention="7 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        compression="zip",
        level="INFO",
    )

    return loguru_logger


# Initialize the logger
logger = setup_logger()
