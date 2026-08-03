import sys
from pathlib import Path

from tools.config_labels import ConfigKeys as CK

REPO_ROOT = Path(__file__).parent.parent.resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_cmd_vasp_hull_single_task_no_srun_prefix() -> None:
    """A single task per run omits the ``srun`` prefix.

    Verifies that ``cmd_vasp_hull`` builds a plain command without an
    ``srun`` launcher when ``VASP_NTASKS_PER_RUN`` equals one.
    """
    from parsl_tasks.hull import cmd_vasp_hull

    config = {
        CK.VASP_NTASKS_PER_RUN: 1,
        CK.VASP_STD_EXE: "/path/to/vasp_std",
    }
    work_subdir = "/tmp/calc_1"

    cmd = cmd_vasp_hull(config, work_subdir)

    assert "srun" not in cmd
    assert cmd.startswith(f"cd {work_subdir} && ")
    assert "/path/to/vasp_std" in cmd
    assert cmd.endswith(f"> {work_subdir}/output")


def test_cmd_vasp_hull_multi_task_uses_srun_prefix() -> None:
    """More than one task per run adds an ``srun`` prefix.

    Verifies that ``cmd_vasp_hull`` prepends the expected ``srun`` launcher
    with the correct task count when ``VASP_NTASKS_PER_RUN`` exceeds one.
    """
    from parsl_tasks.hull import cmd_vasp_hull

    config = {
        CK.VASP_NTASKS_PER_RUN: 4,
        CK.VASP_STD_EXE: "/path/to/vasp_std",
    }
    work_subdir = "/tmp/calc_2"

    cmd = cmd_vasp_hull(config, work_subdir)

    assert "srun -N 1 -n 4 --exact --cpu-bind=cores" in cmd
    assert cmd.startswith(f"cd {work_subdir} && ")
    assert "/path/to/vasp_std" in cmd
    assert cmd.endswith(f"> {work_subdir}/output")


def test_cmd_vasp_hull_output_path_joined(tmp_path: Path) -> None:
    """The output file is placed inside the work subdirectory.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary directory used as the work subdirectory.
    """
    from parsl_tasks.hull import cmd_vasp_hull

    config = {
        CK.VASP_NTASKS_PER_RUN: 1,
        CK.VASP_STD_EXE: "vasp_std",
    }
    work_subdir = str(tmp_path / "run")

    cmd = cmd_vasp_hull(config, work_subdir)

    expected_output = str(Path(work_subdir) / "output")
    assert f"> {expected_output}" in cmd
