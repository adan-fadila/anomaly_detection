import logging

class Logger:
    @staticmethod
    def setup_logger():
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        return logging.getLogger(__name__)

logger = Logger.setup_logger()