from loguru import logger
import sys
import os
from datetime import datetime

# Ensure the logs directory exists
os.makedirs("logs", exist_ok=True)

# Remove the default logger
logger.remove()

# Generate a unique identifier for this run
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

# Add a logger that writes to stdout
logger.add(sys.stdout, colorize=True, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")

# Add a logger for this specific run
logger.add(
    f"logs/ocr_run_{run_id}.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG"
)

# Add a logger for daily rotation (for aggregated logs)
logger.add(
    "logs/ocr_daily.log",
    rotation="00:00",
    retention="7 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    compression="zip",
    level="INFO"
)

# Function to get the logger instance
def get_logger():
    return logger
