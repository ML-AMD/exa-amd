"""Tests for the :class:`~tools.logging_config.ExaAmdLogger` utility.

These tests verify stream routing (stdout vs. stderr) per log level, the
critical-level exit behaviour, level-based filtering, and the defaults of the
module-level ``amd_logger`` instance.
"""

import pytest

from tools.logging_config import ExaAmdLogger, amd_logger


def check_output(
    capsys: pytest.CaptureFixture[str],
    logger: ExaAmdLogger,
    log_level: str,
    on_stdout: bool,
) -> None:
    """Emit a message and assert it lands on the expected stream.

    Parameters
    ----------
    capsys : pytest.CaptureFixture[str]
        Fixture capturing ``stdout`` and ``stderr``.
    logger : ExaAmdLogger
        Logger instance under test.
    log_level : str
        Lowercase level name and logger method to invoke (e.g. ``"info"``).
    on_stdout : bool
        When ``True`` the message is expected on ``stdout``; otherwise on
        ``stderr``. The complementary stream must be empty.

    Returns
    -------
    None
    """
    log_name = logger.logger_name
    msg = f"{log_level} message"
    getattr(logger, log_level)(msg)
    out, err = capsys.readouterr()
    non_empty = out if on_stdout else err
    empty = err if on_stdout else out

    assert f"[{log_level.upper()}] {log_name}: {msg}\n" == non_empty
    assert "" == empty


def test_stdout_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    """Log messages are routed to the correct stream per level.

    Parameters
    ----------
    capsys : pytest.CaptureFixture[str]
        Fixture capturing ``stdout`` and ``stderr``.

    Returns
    -------
    None
    """
    log_name = "test_debug_info"
    logger = ExaAmdLogger(level_name="DEBUG", logger_name=log_name)
    check_output(capsys, logger, "debug", on_stdout=True)
    check_output(capsys, logger, "info", on_stdout=True)
    check_output(capsys, logger, "warning", on_stdout=False)
    check_output(capsys, logger, "error", on_stdout=False)


def test_critical(capsys: pytest.CaptureFixture[str]) -> None:
    """A critical message is written to stderr and exits with code 1.

    Parameters
    ----------
    capsys : pytest.CaptureFixture[str]
        Fixture capturing ``stdout`` and ``stderr``.

    Returns
    -------
    None
    """
    log_name = "test_critical_exit"
    logger = ExaAmdLogger(level_name="DEBUG", logger_name=log_name)
    critical_msg = "critical message"
    with pytest.raises(SystemExit) as ex:
        logger.critical(critical_msg)
    assert ex.value.code == 1
    out, err = capsys.readouterr()
    assert f"[CRITICAL] {log_name}: {critical_msg}\n" == err
    assert "" == out


@pytest.mark.parametrize(
    "configured_log_level, lower_level",
    [
        ("INFO", "debug"),
        ("WARNING", "info"),
        ("ERROR", "warning"),
        ("CRITICAL", "error"),
    ],
)
def test_filtering_below_level(
    configured_log_level: str,
    lower_level: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Messages below the configured level produce no output.

    Parameters
    ----------
    configured_log_level : str
        Level the logger is configured with.
    lower_level : str
        Method whose severity is strictly below ``configured_log_level``.
    capsys : pytest.CaptureFixture[str]
        Fixture capturing ``stdout`` and ``stderr``.

    Returns
    -------
    None
    """
    logger = ExaAmdLogger(level_name=configured_log_level, logger_name="filter_test")
    msg = "should_not_trigger_an_output"
    getattr(logger, lower_level)(msg)
    out, err = capsys.readouterr()
    assert msg not in (out + err)


def test_global_logger(capsys: pytest.CaptureFixture[str]) -> None:
    """The module-level ``amd_logger`` uses INFO-level defaults.

    Parameters
    ----------
    capsys : pytest.CaptureFixture[str]
        Fixture capturing ``stdout`` and ``stderr``.

    Returns
    -------
    None
    """
    # should generate an output
    check_output(capsys, amd_logger, "info", on_stdout=True)
    check_output(capsys, amd_logger, "warning", on_stdout=False)
    check_output(capsys, amd_logger, "error", on_stdout=False)
    # should not generate an output
    amd_logger.debug("should_not_trigger_an_output")
    out, err = capsys.readouterr()
    assert "" == out
    assert "" == err
