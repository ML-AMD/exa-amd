"""Tests for parsl_tasks.mlip_relaxation."""

from __future__ import annotations

import os
import shlex
from pathlib import Path
from typing import Iterator
from unittest import mock

import pytest

from parsl_tasks.mlip_relaxation import cmd_mlip_relaxation
from tools.config_labels import ConfigKeys as CK


def _make_config(tmp_path: Path) -> dict:
    """Create a minimal config with an existing working directory.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary directory.

    Returns
    -------
    dict
        Config dict containing only ``WORK_DIR``.
    """
    work_dir = tmp_path / "work"
    work_dir.mkdir(parents=True)

    return {
        CK.WORK_DIR: str(work_dir),
    }


@pytest.fixture
def restore_cwd() -> Iterator[None]:
    """Restore the process working directory after a test.

    Yields
    ------
    None
        Control is yielded to the test; the original cwd is restored on teardown.
    """
    cwd = os.getcwd()
    yield
    os.chdir(cwd)


def test_returns_command_string(tmp_path: Path, restore_cwd: None) -> None:
    """The command targets the packaged script and model."""
    config = _make_config(tmp_path)

    cmd = cmd_mlip_relaxation(config, ["a.cif", "b.cif"])

    assert cmd.startswith("python ")
    assert "mlip_relax.py" in cmd
    assert "uma-s-1p1.pt" in cmd


def test_single_path_is_normalized(tmp_path: Path, restore_cwd: None) -> None:
    """A single path argument is accepted and appears in the command."""
    config = _make_config(tmp_path)

    cmd = cmd_mlip_relaxation(config, "single.cif")

    assert cmd.endswith("single.cif")


def test_multiple_paths_are_joined_and_quoted(
    tmp_path: Path, restore_cwd: None
) -> None:
    """Multiple paths are shell-quoted and joined into the command."""
    config = _make_config(tmp_path)
    paths = ["with space.cif", "plain.cif"]

    cmd = cmd_mlip_relaxation(config, paths)

    for p in paths:
        assert shlex.quote(p) in cmd


@mock.patch("parsl_tasks.mlip_relaxation.os.chdir")
def test_changes_to_work_dir(mock_chdir: mock.MagicMock, tmp_path: Path) -> None:
    """The task changes into the configured working directory."""
    config = _make_config(tmp_path)

    cmd_mlip_relaxation(config, ["a.cif"])

    mock_chdir.assert_called_once_with(config[CK.WORK_DIR])


def test_creates_log_dir(tmp_path: Path, restore_cwd: None) -> None:
    """The MLIP log directory is created under the working directory."""
    config = _make_config(tmp_path)

    cmd_mlip_relaxation(config, ["a.cif"])

    log_dir = os.path.join(config[CK.WORK_DIR], CK.MLIP_LOG_DIR)
    assert os.path.isdir(log_dir)
