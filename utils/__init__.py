import logging

LOG_FORMAT = "%(asctime)s [%(levelname)s]: %(message)s"
LOG_DATETIME_FORMAT = "%m/%d/%Y %I:%M:%S %p"
LOG_LEVEL = logging.INFO

logging.basicConfig(
    format=LOG_FORMAT, datefmt=LOG_DATETIME_FORMAT, level=LOG_LEVEL
)

logger = logging.getLogger()
