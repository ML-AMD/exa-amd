from parsl import bash_app, python_app
from tools.config_labels import ConfigKeys as CK
from parsl_configs.parsl_executors_labels import POSTPROCESSING_LABEL


def cmd_vasp_hull(config, work_subdir):
    """
    Using the total energies, computes the formation energies of each structure
    relative to reference elemental phases.

    :param dict config:
        :class:`~tools.config_manager.ConfigManager` (or dict). Keys used:
        - ``vasp_std_exe`` (str): path to the VASP executable.

    :param str work_subdir:
        Working subdirectory where the command should be executed.

    :returns: Shell command string.
    :rtype: str

    :raises Exception: on directory navigation failures.
    """
    import os
    try:
        os.chdir(work_subdir)
        exec_cmd_prefix = (
            "" if config[CK.VASP_NTASKS_PER_RUN] == 1
            else f"srun -n {config[CK.VASP_NTASKS_PER_RUN]}"
        )
        output_file = os.path.join(work_subdir, "output")
    except Exception as e:
        raise
    return f" {exec_cmd_prefix} {config[CK.VASP_STD_EXE]} > {output_file}"


@bash_app(executors=[POSTPROCESSING_LABEL])
def run_single_vasp_hull_calculation(config, work_subdir):
    return cmd_vasp_hull(config, work_subdir)
