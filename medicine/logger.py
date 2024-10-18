"""Logger module.

Usage: `from medicine.logger import logger`. This will get a logger instance.
"""

import logging

# Create logger instance
logging.basicConfig(format="[%(levelname)s] - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
