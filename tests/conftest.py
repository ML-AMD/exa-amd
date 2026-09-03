"""Shared pytest fixtures and helpers for the exa-amd test suite.

This module is discovered automatically by pytest and its fixtures are
available to every test in this directory and its subdirectories without
being imported.

Importing this module also inserts the repository root onto ``sys.path`` so
that individual test modules do not need to repeat that boilerplate before
importing the packages under test.

Fixtures
--------
restore_cwd
    Autouse fixture that snapshots the working directory before each test
    and restores it afterwards, preventing global CWD mutations (e.g. from
    ``os.chdir`` inside Parsl tasks) from leaking between tests.

Helpers
-------
extract_tar
    Extract a tar archive using the safe ``filter="data"`` behaviour on
    Python 3.12+ with a graceful fallback on older interpreters.
pushd
    Context manager that temporarily changes the working directory and
    restores it on exit, replacing repeated ``os.chdir`` try/finally blocks.
ResourcePathCtx
    Context manager yielding a fixed path, used to double
    ``importlib.resources`` accessors in tests.
FakeFuture
    Minimal stand-in for a Parsl ``AppFuture`` exposing ``exception()``.
"""

import contextlib
import os
import sys
import tarfile
from pathlib import Path
from typing import Any, Iterator, Literal

import pytest

# Repository root (the parent of this ``tests`` directory). Inserted onto
# ``sys.path`` at import time so test modules can import the packages under
# test without repeating this boilerplate themselves.
REPO_ROOT = Path(__file__).parent.parent.resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(autouse=True)
def restore_cwd() -> Iterator[None]:
    """Snapshot and restore the working directory around each test.

    Several code paths under test (Parsl tasks, ``cmd_cgcnn_prediction``,
    etc.) call :func:`os.chdir`. Without restoration those mutations leak
    into subsequent tests. This autouse fixture records the working
    directory before the test and restores it afterwards.

    Yields
    ------
    None
        Control is yielded to the test; the original directory is restored
        on teardown, even if the test raises.
    """
    original = os.getcwd()
    try:
        yield
    finally:
        os.chdir(original)


@contextlib.contextmanager
def pushd(path: str | os.PathLike) -> Iterator[None]:
    """Temporarily change the working directory within a ``with`` block.

    Several tests need to run code that assumes a specific current working
    directory (for example, Parsl tasks that write relative to ``cwd``). This
    helper replaces the repeated ``cwd = os.getcwd(); try/finally:
    os.chdir(cwd)`` boilerplate with a single context manager.

    Parameters
    ----------
    path : str or os.PathLike
        Directory to switch into for the duration of the ``with`` block.

    Yields
    ------
    None
        Control is yielded with ``path`` as the working directory; the
        original directory is restored on exit, even if an exception is
        raised.

    Notes
    -----
    The autouse :func:`restore_cwd` fixture already restores the working
    directory after each test, so this helper is primarily for readability and
    for correctly restoring ``cwd`` *mid-test* (e.g. inside module-scoped
    fixtures that run several ``chdir`` blocks).
    """
    original = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original)


def extract_tar(archive_path: str | os.PathLike, dest: str | os.PathLike) -> None:
    """Extract ``archive_path`` into ``dest`` using safe extraction.

    Uses the ``filter="data"`` argument on Python 3.12+ (which rejects unsafe
    members) and falls back to a plain ``extractall`` on older interpreters
    that do not support the keyword.

    Parameters
    ----------
    archive_path : str or os.PathLike
        Path to the tar archive to extract.
    dest : str or os.PathLike
        Directory into which the archive is extracted.

    Returns
    -------
    None
    """
    with tarfile.open(archive_path) as tar:
        try:
            tar.extractall(path=dest, filter="data")  # Python 3.12+
        except TypeError:
            tar.extractall(path=dest)


def write_element_potcars(
    pot_dir: Path, elements: tuple[str, ...], prefix: str = "POTCAR"
) -> None:
    """Create per-element ``POTCAR`` files under ``pot_dir``.

    Parameters
    ----------
    pot_dir : pathlib.Path
        Root PAW potentials directory.
    elements : list[str]
        Element symbols for which to create ``<el>/POTCAR`` files.
    """
    for elem in elements:
        elem_dir = pot_dir / elem
        elem_dir.mkdir(parents=True)
        (elem_dir / "POTCAR").write_text(f"POTCAR-{elem}\n")


class ResourcePathCtx:
    """Context manager returning a fixed path, mimicking ``importlib.resources``.

    Test doubles for packaged-resource access (``importlib.resources.path`` and
    ``importlib.resources.as_file``) must behave as context managers yielding a
    filesystem path. This helper provides that behaviour with a pre-supplied
    path so tests can redirect packaged assets to temporary files.

    Parameters
    ----------
    path : pathlib.Path
        The path yielded on ``__enter__``.
    """

    def __init__(self, path: Path) -> None:
        self._path = path

    def __enter__(self) -> Path:
        """Return the wrapped path.

        Returns
        -------
        pathlib.Path
            The path supplied at construction time.
        """
        return self._path

    def __exit__(self, *exc: Any) -> Literal[False]:
        """Propagate any exception (never suppress).

        Returns
        -------
        bool
            Always ``False`` so exceptions are not swallowed.
        """
        return False


class FakeFuture:
    """Minimal stand-in for a Parsl ``AppFuture``.

    Parameters
    ----------
    exc : Exception or None, optional
        Exception to return from :meth:`exception`; ``None`` (the default)
        indicates a successful future.
    """

    def __init__(self, exc: Exception | None = None) -> None:
        self._exc = exc

    def exception(self) -> Exception | None:
        """Return the stored exception (or ``None``).

        Returns
        -------
        Exception or None
            The exception associated with this future.
        """
        return self._exc
