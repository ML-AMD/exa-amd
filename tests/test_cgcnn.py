"""Tests for :func:`parsl_tasks.cgcnn.cmd_cgcnn_prediction`.

These tests verify the constructed CGCNN prediction command string, the
copying of ``atom_init.json`` into the per-chunk structures directory, the
working-directory change, and argument-validation guards.
"""

import os
import shutil
from pathlib import Path

import pytest

import parsl_tasks.cgcnn as cgcnn_mod
from parsl_tasks.cgcnn import cmd_cgcnn_prediction
from tools.config_labels import ConfigKeys as CK

# Type alias for the config mappings used throughout this module.
Config = dict[str, object]


@pytest.fixture
def work_dir(tmp_path: Path) -> Path:
    """Create a work directory containing the per-chunk structures folder.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary directory.

    Returns
    -------
    pathlib.Path
        The work directory containing ``structures/0``.
    """
    (tmp_path / "structures" / "0").mkdir(parents=True)
    return tmp_path


@pytest.fixture
def config(work_dir: Path) -> Config:
    """Build a minimal CGCNN prediction config.

    Parameters
    ----------
    work_dir : pathlib.Path
        The work directory fixture.

    Returns
    -------
    dict
        Config dict with work directory, batch size and worker count.
    """
    return {
        CK.WORK_DIR: str(work_dir),
        CK.BATCH_SIZE: 32,
        CK.NUM_WORKERS: 4,
    }


@pytest.fixture(autouse=True)
def fake_pkg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the cgcnn package at a temporary directory with fake assets.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary directory.
    monkeypatch : pytest.MonkeyPatch
        Used to patch the packaged ``__file__`` location.

    Returns
    -------
    pathlib.Path
        The fake package directory containing the model and helper assets.
    """
    pkg_dir = tmp_path / "pkg"
    pkg_dir.mkdir()
    (pkg_dir / "form_1st.pth.tar").write_text("model")
    (pkg_dir / "atom_init.json").write_text("{}")
    (pkg_dir / "predict.py").write_text("# predict")
    monkeypatch.setattr(cgcnn_mod.cgcnn_pkg, "__file__", str(pkg_dir / "__init__.py"))
    return pkg_dir


def test_cmd_returns_expected_command(
    config: Config, fake_pkg: Path, work_dir: Path
) -> None:
    """The command references the model, structures dir, and CLI options.

    Parameters
    ----------
    config : dict
        The CGCNN prediction config fixture.
    fake_pkg : pathlib.Path
        The fake package directory fixture.
    work_dir : pathlib.Path
        The work directory fixture.

    Returns
    -------
    None
    """
    cmd = cmd_cgcnn_prediction(config, n_chunks=2, id=0)

    assert cmd.startswith(f"python {fake_pkg / 'predict.py'}")
    assert str(fake_pkg / "form_1st.pth.tar") in cmd
    assert str(work_dir / "structures" / "0") in cmd
    assert "--batch-size 32" in cmd
    assert "--workers 4" in cmd
    assert "--chunk_id 0" in cmd


def test_cmd_copies_atom_init(
    config: Config, fake_pkg: Path, work_dir: Path
) -> None:
    """``atom_init.json`` is copied into the per-chunk structures directory.

    Parameters
    ----------
    config : dict
        The CGCNN prediction config fixture.
    fake_pkg : pathlib.Path
        The fake package directory fixture.
    work_dir : pathlib.Path
        The work directory fixture.

    Returns
    -------
    None
    """
    cmd_cgcnn_prediction(config, n_chunks=1, id=0)
    assert (work_dir / "structures" / "0" / "atom_init.json").exists()


def test_cmd_changes_cwd(config: Config, work_dir: Path) -> None:
    """The command changes the working directory to the work directory.

    Parameters
    ----------
    config : dict
        The CGCNN prediction config fixture.
    work_dir : pathlib.Path
        The work directory fixture.

    Returns
    -------
    None
    """
    cmd_cgcnn_prediction(config, n_chunks=1, id=0)
    assert os.getcwd() == str(work_dir)


@pytest.mark.parametrize("n_chunks", [0, -1])
def test_invalid_n_chunks_raises(config: Config, n_chunks: int) -> None:
    """A non-positive ``n_chunks`` raises ``ValueError``.

    Parameters
    ----------
    config : dict
        The CGCNN prediction config fixture.
    n_chunks : int
        Invalid (non-positive) chunk count under test.

    Returns
    -------
    None
    """
    with pytest.raises(ValueError, match="n_chunks must be positive"):
        cmd_cgcnn_prediction(config, n_chunks=n_chunks, id=0)


@pytest.mark.parametrize("id", [-1, 5])
def test_id_out_of_range_raises(config: Config, id: int) -> None:
    """A chunk ``id`` outside ``[0, n_chunks)`` raises ``ValueError``.

    Parameters
    ----------
    config : dict
        The CGCNN prediction config fixture.
    id : int
        Out-of-range chunk identifier under test.

    Returns
    -------
    None
    """
    with pytest.raises(ValueError, match="id must satisfy"):
        cmd_cgcnn_prediction(config, n_chunks=2, id=id)


def test_missing_structures_dir_raises(config: Config, work_dir: Path) -> None:
    """A missing structures directory raises ``FileNotFoundError``.

    Parameters
    ----------
    config : dict
        The CGCNN prediction config fixture.
    work_dir : pathlib.Path
        The work directory fixture.

    Returns
    -------
    None

    Notes
    -----
    ``id=2`` passes the range guard (``0 <= 2 < 3``) but ``structures/2``
    does not exist. Removing the parent so ``shutil.copy`` cannot resolve the
    destination triggers the error.
    """
    shutil.rmtree(work_dir / "structures")
    with pytest.raises(FileNotFoundError):
        cmd_cgcnn_prediction(config, n_chunks=3, id=2)
