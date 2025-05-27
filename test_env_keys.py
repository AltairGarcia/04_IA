import os
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load the .env file explicitly
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
logger.debug(f"Loading .env file from: {dotenv_path}")
if not os.path.exists(dotenv_path):
    logger.error(".env file not found at the specified path.")
else:
    logger.debug(".env file loaded successfully.")
    load_dotenv(dotenv_path)

# Test retrieval of specific keys
stability_key = os.getenv('STABILITY_API_KEY', 'Not Found')
youtube_key = os.getenv('YOUTUBE_API_KEY', 'Not Found')

print(f"STABILITY_API_KEY: {stability_key}")
print(f"YOUTUBE_API_KEY: {youtube_key}")
