from loguru import logger

from src.utils.default_logger import DefaultLogger


class EpochLogger:
    @staticmethod
    def log(message: str):
        EpochLogger.apply_format()
        logger.info(message)
        DefaultLogger.apply_format()

    @staticmethod
    def apply_format():
        logger.remove()

        time = "<white>{time:HH:mm:ss}</white>"
        message = "<fg 52,158,0>{message}</fg 52,158,0>"
        logging_format = f"{time} {message}"

        logger.add(
            sink=lambda msg: print(msg, end=""),
            colorize=True,
            format=logging_format
        )
