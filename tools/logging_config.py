import sys
import parsl


class ExaAmdLogger:
    """
    Lightweight logging utility.

    Supports five standard logging levels, in order of increasing severity:
    DEBUG, INFO, WARNING, ERROR, and CRITICAL.

    Args:
        level_name (str, optional): Logging level name.
        logger_name (str, optional): Name prefix for all log messages.
    """

    LEVEL_MAP = {
        "DEBUG": 10,
        "INFO": 20,
        "WARNING": 30,
        "ERROR": 40,
        "CRITICAL": 50
    }

    def __init__(self, level_name="INFO", logger_name="exa-amd"):
        self.logger_name = logger_name
        self.configure(level_name)

    def configure(self, level_name="INFO"):
        """
        Set the logging level.
        If unsupported level, fall back to INFO.
        """
        level_name = level_name.upper()
        if level_name in self.LEVEL_MAP:
            self._current_level = self.LEVEL_MAP[level_name]
        else:
            self._current_level = self.LEVEL_MAP["INFO"]
            sys.stdout.write(
                f"Unsupported log level '{level_name}'. Falling back to INFO.\n")

    def _log(self, level_name, message):
        """
         - DEBUG/INFO to stdout
         - WARNING/ERROR to stderr
         - CRITICAL to stderr and then exits
        """
        numeric_level = self.LEVEL_MAP[level_name]

        if numeric_level < self._current_level:
            return

        formatted_message = f"[{level_name}] {self.logger_name}: {message}"

        if numeric_level <= self.LEVEL_MAP["INFO"]:
            sys.stdout.write(formatted_message + "\n")
        elif numeric_level < self.LEVEL_MAP["CRITICAL"]:
            sys.stderr.write(formatted_message + "\n")
        else:
            sys.stderr.write(formatted_message + "\n")
            parsl.dfk().cleanup()
            sys.exit(1)

    def debug(self, message):
        """Log an debug-level message to stderr if permitted by current log level."""
        self._log("DEBUG", message)

    def info(self, message):
        """Log an info-level message to stderr if permitted by current log level."""
        self._log("INFO", message)

    def warning(self, message):
        """Log an warning-level message to stderr if permitted by current log level."""
        self._log("WARNING", message)

    def error(self, message):
        """Log an error-level message to stderr if permitted by current log level."""
        self._log("ERROR", message)

    def critical(self, message):
        """Log a critical-level message to stderr and terminate the program."""
        self._log("CRITICAL", message)


# global instance
amd_logger = ExaAmdLogger()
