"""Shared pytest fixtures for the exa-amd test suite.

This module is discovered automatically by pytest and its fixtures are
available to every test in this directory and its subdirectories without
being imported.

Fixtures
--------
restore_cwd
    Autouse fixture that snapshots the working directory before each test
    and restores it afterwards, preventing global CWD mutations (e.g. from
    ``os.chdir`` inside Parsl tasks) from leaking between tests.
"""

import os
from typing import Iterator

import pytest


@pytest.fixture(autouse=True)
def restore_cwd() -> Iterator[None]:
    """Save the working directory before each test and restore it after.

    ``cmd_cgcnn_prediction`` (and potentially other tasks) call raw
    ``os.chdir``, which mutates global process state. This autouse fixture
    guarantees the working directory is reset after every test so state does
    not leak between tests.

    Yields
    ------
    None
        Control is yielded to the test; the original working directory is
        restored on teardown.
    """
    original = os.getcwd()
    yield
    os.chdir(original)
