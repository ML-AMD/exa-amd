import logging
import sys

LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}


class MaxLevelFilter(logging.Filter):
    """ Filter that only allows log records up to a specified max level. """

    def __init__(self, max_level):
        super().__init__()
        self.max_level = max_level

    def filter(self, record):
        return record.levelno <= self.max_level


class ExitOnCriticalHandler(logging.Handler):
    """
    Exits the program whenever a CRITICAL-level log is emitted.
    """

    def emit(self, record):
        msg = self.format(record)
        sys.stderr.write(msg + "\n")
        if record.levelno == logging.CRITICAL:
            sys.exit(1)


def configure_logging(level_name="ERROR"):
    """
    Configures the root logger so that:
      - If 'level_name' is invalid, default to INFO.
      - DEBUG/INFO go to stdout, WARNING/ERROR to stderr, CRITICAL → exit.
      - All Parsl loggers (and sub-loggers) are forced to WARNING
        so they won't emit debug/info messages.
    """
    # Determine requested logging level
    level_name_upper = level_name.upper()
    level = LEVEL_MAP.get(level_name_upper, logging.INFO)

    # If invalid, default to INFO and warn
    if level_name_upper not in LEVEL_MAP:
        logging.basicConfig(stream=sys.stdout,
                            format="%(message)s", level=logging.INFO)
        logging.warning(
            f"Unsupported log level '{level_name}'. Falling back to INFO.")

    # Remove any existing handlers from the root logger
    root_logger = logging.getLogger()
    root_logger.name = "exa-amd"
    while root_logger.handlers:
        root_logger.removeHandler(root_logger.handlers[0])

    # Set the root logger's level
    root_logger.setLevel(level)

    formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")

    # stdout for DEBUG/INFO
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.addFilter(MaxLevelFilter(logging.INFO))
    stdout_handler.setFormatter(formatter)
    root_logger.addHandler(stdout_handler)

    # stderr for WARNING/ERROR
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(formatter)
    root_logger.addHandler(stderr_handler)

    # CRITICAL -> exit
    critical_handler = ExitOnCriticalHandler()
    critical_handler.setLevel(logging.CRITICAL)
    critical_handler.setFormatter(formatter)
    root_logger.addHandler(critical_handler)

    #Force all Parsl loggers and sub-loggers to WARNING,
    for logger_name in list(logging.Logger.manager.loggerDict.keys()):
        if logger_name.startswith("parsl"):
            plogger = logging.getLogger(logger_name)
            plogger.setLevel(logging.WARNING)
            while plogger.handlers:
                plogger.removeHandler(plogger.handlers[0])
