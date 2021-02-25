import logging

import tensorflow as tf

LOG_FORMAT = "%(asctime)s [%(levelname)s]: %(message)s"
LOG_DATETIME_FORMAT = "%m/%d/%Y %I:%M:%S %p"
LOG_LEVEL = logging.INFO

logging.basicConfig(
    format=LOG_FORMAT, datefmt=LOG_DATETIME_FORMAT, level=LOG_LEVEL
)

logger = logging.getLogger()


def init_tf():
    """
    Initializes Tensorflow and allows memory growth.
    """
    physical_devices = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
