from loguru import logger


class DefaultLogger:
    @staticmethod
    def apply_format():
        logger.remove()

        time = "<white>{time:HH:mm:ss}</white>"
        level = "<level>{level: ^8}</level>"

        file_name = "<white>{name}</white>"
        executed_function = "<white>{function}</white>"
        line_number = "<white>{line}</white>"
        location = f"{file_name}<red>:</red>{executed_function}<red>:</red>{line_number}"

        message = "<level>{message}</level>"
        logging_format = [time, level, f"{location} <red>-</red> {message}"]

        separator = " <red>|</red> "
        logger.add(
            sink=lambda msg: print(msg, end=""),
            colorize=True,
            format=separator.join(logging_format)
        )
