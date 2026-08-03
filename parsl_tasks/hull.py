"""VASP convex-hull calculation commands and Parsl app.

This module builds the shell commands used to run VASP total-energy
calculations for reference phases and wraps them as a Parsl ``bash_app``.
The resulting energies are used to compute formation energies relative to
elemental reference phases when constructing the convex hull.
"""

from parsl import bash_app

from parsl_configs.parsl_executors_labels import POSTPROCESSING_LABEL
from tools.config_labels import ConfigKeys as CK


def cmd_vasp_hull(config: dict, work_subdir: str) -> str:
    """Build the shell command that runs VASP for a hull calculation.

    Using the total energies, computes the formation energies of each
    structure relative to reference elemental phases.

    Parameters
    ----------
    config : dict
        :class:`~tools.config_manager.ConfigManager` (or dict). Keys used:

        - ``VASP_STD_EXE`` (str): path to the VASP executable.
        - ``VASP_NTASKS_PER_RUN`` (int): number of MPI tasks per run. When
          greater than ``1`` an ``srun`` prefix is added.
    work_subdir : str
        Working subdirectory where the command should be executed.

    Returns
    -------
    str
        Shell command string.

    Raises
    ------
    Exception
        On directory navigation failures.
    """
    import os

    exec_cmd_prefix = (
        ""
        if config[CK.VASP_NTASKS_PER_RUN] == 1
        else f"srun -N 1 -n {config[CK.VASP_NTASKS_PER_RUN]} --exact --cpu-bind=cores"
    )
    output_file = os.path.join(work_subdir, "output")
    return f"cd {work_subdir} && {exec_cmd_prefix} {config[CK.VASP_STD_EXE]} > {output_file}"


@bash_app(executors=[POSTPROCESSING_LABEL])
def run_single_vasp_hull_calculation(config: dict, work_subdir: str) -> str:
    """Run a single VASP hull calculation as a Parsl ``bash_app``.

    Parameters
    ----------
    config : dict
        :class:`~tools.config_manager.ConfigManager` (or dict) forwarded to
        :func:`cmd_vasp_hull`.
    work_subdir : str
        Working subdirectory where the command should be executed.

    Returns
    -------
    str
        Shell command string produced by :func:`cmd_vasp_hull`.
    """
    return cmd_vasp_hull(config, work_subdir)
