import sys
import pytest
from tools.logging_config import ExaAmdLogger, amd_logger


def check_output(capsys, logger, log_level, on_stdout: bool):
    """
    helper for 'test_stdout_stderr'
    """
    log_name = getattr(logger, "logger_name")
    msg = f"{log_level} message"
    getattr(logger, log_level)(msg)
    out, err = capsys.readouterr()
    non_empty = out if on_stdout else err
    empty = err if on_stdout else out

    assert f"[{log_level.upper()}] {log_name}: {msg}\n" == non_empty
    assert "" == empty


def test_stdout_stderr(capsys):
    """
    test that the log messages are correctly printed on stdout and stderr
    """
    log_name = "test_debug_info"
    logger = ExaAmdLogger(level_name="DEBUG", logger_name=log_name)
    check_output(capsys, logger, "debug", on_stdout=True)
    check_output(capsys, logger, "info", on_stdout=True)
    check_output(capsys, logger, "warning", on_stdout=False)
    check_output(capsys, logger, "error", on_stdout=False)


def test_critical(capsys):
    """
    test that we exit after a critical message
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


@pytest.mark.parametrize("configured_log_level, lower_level", [
    ("INFO", "debug"),
    ("WARNING", "info"),
    ("ERROR", "warning"),
    ("CRITICAL", "error"),
])
def test_filtering_below_level(configured_log_level, lower_level, capsys):
    """
    verify that the log levels are respected
    """
    logger = ExaAmdLogger(
        level_name=configured_log_level,
        logger_name="filter_test")
    msg = "should_not_trigger_an_output"
    getattr(logger, lower_level)(msg)
    out, err = capsys.readouterr()
    assert msg not in (out + err)


def test_global_logger(capsys):
    """
    check the defaults of the global logger
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
