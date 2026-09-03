"""Tests for parsl_tasks.dft_optimization."""

import os
import types
from pathlib import Path
from typing import Iterator
from unittest import mock

import pytest

from parsl_tasks.dft_optimization import cmd_fused_vasp_calc
from tools.config_labels import ConfigKeys as CK
from tools.errors import VaspNonReached


def _make_config(tmp_path: Path, ntasks: int = 1) -> dict:
    """Build a VASP fused-calculation config with staged input files.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary directory.
    ntasks : int, optional
        Number of tasks per run (controls the ``srun`` prefix). Default ``1``.

    Returns
    -------
    dict
        Config dict with work directories, executable, timeout and NSW.
    """
    work_dir = tmp_path / "work"
    vasp_work = tmp_path / "vasp_work"
    (work_dir / "new").mkdir(parents=True)
    vasp_work.mkdir(parents=True)

    (work_dir / "new" / "POSCAR_1").write_text("poscar")
    (work_dir / "POTCAR").write_text("potcar")

    return {
        CK.WORK_DIR: str(work_dir),
        CK.VASP_WORK_DIR: str(vasp_work),
        CK.VASP_STD_EXE: "vasp_std",
        CK.VASP_TIMEOUT: 60,
        CK.VASP_NSW: 5,
        CK.VASP_NTASKS_PER_RUN: ntasks,
    }


class _FakeAssets:
    """Fake INCAR template files.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Directory in which to create the ``INCAR.rx`` and ``INCAR.en`` files.
    """

    def __init__(self, tmp_path: Path) -> None:
        self.incar_rx = tmp_path / "INCAR.rx"
        self.incar_en = tmp_path / "INCAR.en"
        self.incar_rx.write_text("NSW = 0\n")
        self.incar_en.write_text("energy\n")


@pytest.fixture
def assets(tmp_path: Path) -> _FakeAssets:
    """Provide fake INCAR template files.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary directory.

    Returns
    -------
    _FakeAssets
        The created fake INCAR assets.
    """
    return _FakeAssets(tmp_path)


@pytest.fixture
def patch_resources(
    monkeypatch: pytest.MonkeyPatch, assets: _FakeAssets
) -> _FakeAssets:
    """Patch importlib.resources to return the fake INCAR templates.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Used to patch ``importlib.resources``.
    assets : _FakeAssets
        The fake INCAR assets fixture.

    Returns
    -------
    _FakeAssets
        The same fake INCAR assets, for convenience.
    """
    import importlib.resources as iresources

    class _CtxFile:
        def __init__(self, path):
            self._path = path

        def __enter__(self):
            return self._path

        def __exit__(self, *exc):
            return False

    def fake_files(_pkg):
        class _Traversable:
            def __truediv__(self, name):
                if name == "INCAR.rx":
                    return assets.incar_rx
                if name == "INCAR.en":
                    return assets.incar_en
                raise KeyError(name)

        return _Traversable()

    monkeypatch.setattr(iresources, "files", fake_files)
    monkeypatch.setattr(iresources, "as_file", lambda path: _CtxFile(path))
    return assets


def _run_side_effect(
    reached: bool = False,
    nsw_hit: bool = False,
    nsw: int = 5,
    returncode: int = 0,
):
    """Build a subprocess.run replacement simulating VASP output/files.

    Parameters
    ----------
    reached : bool, optional
        Emit the "reached required accuracy" marker. Default ``False``.
    nsw_hit : bool, optional
        Emit an ``NSW`` step line to trigger a rerun. Default ``False``.
    nsw : int, optional
        The NSW step count used in the emitted line. Default ``5``.
    returncode : int, optional
        Return code of the simulated process. Default ``0``.

    Returns
    -------
    Callable
        A side-effect function suitable for ``mock.patch`` on ``subprocess.run``.
    """

    def side_effect(cmd, stdout=None, stderr=None, **kwargs):
        text = ""
        if reached:
            text += "reached required accuracy\n"
        if nsw_hit:
            text += f"   {nsw} F= -1.0\n"
        if stdout is not None:
            stdout.write(text)
        Path("OUTCAR").write_text("outcar")
        Path("CONTCAR").write_text("contcar")
        return types.SimpleNamespace(returncode=returncode)

    return side_effect


@pytest.fixture
def restore_cwd() -> Iterator[None]:
    """Restore the process working directory after a test.

    Yields
    ------
    None
        Control is yielded to the test; the cwd is restored on teardown.
    """
    cwd = os.getcwd()
    yield
    os.chdir(cwd)


@mock.patch("parsl_tasks.dft_optimization.subprocess.run")
def test_relaxation_reached_success(
    mock_run: mock.MagicMock,
    patch_resources: _FakeAssets,
    tmp_path: Path,
    restore_cwd: None,
) -> None:
    """A run that reaches accuracy writes ``DONE`` and the energy OUTCAR."""
    config = _make_config(tmp_path)
    mock_run.side_effect = _run_side_effect(reached=True)

    cmd_fused_vasp_calc(config, 1, walltime=120)

    work_subdir = Path(config[CK.VASP_WORK_DIR]) / "1"
    assert (work_subdir / "DONE").exists()
    assert (work_subdir / "OUTCAR_1.en").exists()
    assert mock_run.call_count == 2  # relaxation + energy


@mock.patch("parsl_tasks.dft_optimization.subprocess.run")
def test_nsw_hit_triggers_rerun(
    mock_run: mock.MagicMock,
    patch_resources: _FakeAssets,
    tmp_path: Path,
    restore_cwd: None,
) -> None:
    """Hitting the NSW step limit triggers a relaxation rerun."""
    config = _make_config(tmp_path)
    mock_run.side_effect = _run_side_effect(nsw_hit=True, nsw=config[CK.VASP_NSW])

    cmd_fused_vasp_calc(config, 1, walltime=120)

    work_subdir = Path(config[CK.VASP_WORK_DIR]) / "1"
    assert (work_subdir / "DONE").exists()
    assert mock_run.call_count == 3  # relaxation + rerun + energy


@mock.patch("parsl_tasks.dft_optimization.subprocess.run")
def test_non_convergence_raises(
    mock_run: mock.MagicMock,
    patch_resources: _FakeAssets,
    tmp_path: Path,
    restore_cwd: None,
) -> None:
    """Failure to reach accuracy raises :class:`VaspNonReached`."""
    config = _make_config(tmp_path)
    mock_run.side_effect = _run_side_effect(reached=False, nsw_hit=False)

    with pytest.raises(VaspNonReached):
        cmd_fused_vasp_calc(config, 1, walltime=120)

    work_subdir = Path(config[CK.VASP_WORK_DIR]) / "1"
    assert (work_subdir / "DONE").exists()


@mock.patch("parsl_tasks.dft_optimization.subprocess.run")
def test_timeout_raises(
    mock_run: mock.MagicMock,
    patch_resources: _FakeAssets,
    tmp_path: Path,
    restore_cwd: None,
) -> None:
    """A timeout return code raises :class:`VaspNonReached`."""
    config = _make_config(tmp_path)
    mock_run.side_effect = _run_side_effect(returncode=124)

    with pytest.raises(VaspNonReached):
        cmd_fused_vasp_calc(config, 1, walltime=120)

    work_subdir = Path(config[CK.VASP_WORK_DIR]) / "1"
    assert (work_subdir / "DONE").exists()


@mock.patch("parsl_tasks.dft_optimization.subprocess.run")
def test_no_srun_prefix_single_task(
    mock_run: mock.MagicMock,
    patch_resources: _FakeAssets,
    tmp_path: Path,
    restore_cwd: None,
) -> None:
    """A single task per run omits the ``srun`` launcher prefix."""
    config = _make_config(tmp_path, ntasks=1)
    mock_run.side_effect = _run_side_effect(reached=True)

    cmd_fused_vasp_calc(config, 1, walltime=120)

    first_cmd = mock_run.call_args_list[0].args[0]
    assert "srun" not in first_cmd
    assert first_cmd[:2] == ["timeout", str(config[CK.VASP_TIMEOUT])]


@mock.patch("parsl_tasks.dft_optimization.subprocess.run")
def test_srun_prefix_multiple_tasks(
    mock_run: mock.MagicMock,
    patch_resources: _FakeAssets,
    tmp_path: Path,
    restore_cwd: None,
) -> None:
    """Multiple tasks per run add an ``srun`` launcher prefix."""
    config = _make_config(tmp_path, ntasks=4)
    mock_run.side_effect = _run_side_effect(reached=True)

    cmd_fused_vasp_calc(config, 1, walltime=120)

    first_cmd = mock_run.call_args_list[0].args[0]
    assert "srun" in first_cmd
    assert "4" in first_cmd


@mock.patch("parsl_tasks.dft_optimization.subprocess.run")
def test_potcar_symlink_created(
    mock_run: mock.MagicMock,
    patch_resources: _FakeAssets,
    tmp_path: Path,
    restore_cwd: None,
) -> None:
    """A POTCAR symlink is created in the per-run work subdirectory."""
    config = _make_config(tmp_path)
    mock_run.side_effect = _run_side_effect(reached=True)

    cmd_fused_vasp_calc(config, 1, walltime=120)

    work_subdir = Path(config[CK.VASP_WORK_DIR]) / "1"
    assert (work_subdir / "POTCAR").is_symlink()
