import logging
import os
from datetime import datetime

# Get current date and time
now = datetime.now()

# Define log directory with year and month subdirectories
log_dir = os.path.join(os.getcwd(), "logs", str(now.year), now.strftime("%B")).lower()

# Ensure the directory exists
os.makedirs(log_dir, exist_ok=True)

# Define log file name based on the current date and time in the preferred format
LOG_FILE = f"{now.strftime('%Y_%m_%d_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(log_dir, LOG_FILE)

# Basic logging configuration
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(levelname)s %(name)s - %(message)s",
    level=logging.INFO,
)

# Example usage
logging.info("This is a test log message.")
