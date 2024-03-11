import logging
import os
from datetime import datetime

# Define log directory and file
log_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(log_dir, exist_ok=True)  # Ensure the directory exists
LOG_FILE = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(log_dir, LOG_FILE)

# Basic logging configuration
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(levelname)s %(name)s - %(message)s",
    level=logging.INFO,
)