from parsl import bash_app
from tools.config_labels import ConfigKeys as CK
from parsl_configs.parsl_executors_labels import POSTPROCESSING_LABEL


def cmd_vasp_hull(config, work_subdir):
    """
    Using the total energies, computes the formation energies of each structure
    relative to reference elemental phases.

    Parameters
    ----------
    config : dict
        ConfigManager instance. The following fields are used:

        - ``CK.VASP_NTASKS_PER_RUN`` (*int*): Number of MPI ranks.
          ``1`` triggers a run with a single process; any other value results in
          ``srun -n <N>``.
        - ``CK.VASP_STD_EXE``: Path to the VASP executable.

        See :class:`~tools.config_manager.ConfigManager` for field descriptions.

    work_subdir : str
        Name of the working subdirectory where the hull calculation is run.

    Returns
    -------
    str
        A formatted shell command string to execute `run_single_vasp_hull_calculation.py`.

    Raises
    ------
    Exception
        If directory navigation or file operations fail.
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


def cmd_compile_vasp_hull(config, total_calcs, output_file, prefix):
    '''
    collect convex hull for all the structures
    '''
    import os
    bash_script = os.path.join(config[CK.CMS_DIR], "compile_vasp_hull.sh")
    return f"sh {bash_script} {total_calcs} {output_file} {prefix}"


@bash_app(executors=[POSTPROCESSING_LABEL])
def compile_vasp_hull(config, total_calcs, output_file, prefix):
    return cmd_compile_vasp_hull(config, total_calcs, output_file, prefix)


def cmd_calculate_ehul(config):
    """
    Select promising structures: Structures with low energy above the hull (Ehull) are identified as promising candidates. These are automatically copied to a dedicated folder for further analysis, such as evaluation of additional physical properties or preparation for experimental validation.

    Args:
        config (dict): ConfigManager. The following fields are used:
            - ``CK.ELEMENTS``
            - ``CK.CMS_DIR``
            - ``CK.VASP_WORK_DIR``
            - ``CK.ENERGY_DAT_OUT``
            - ``CK.POST_PROCESSING_OUT_DIR``
            - ``CK.MP_STABLE_OUT``

            See :class:`~tools.config_manager.ConfigManager` for field
            descriptions.

    Returns:
        str: A formatted shell command string to execute select_final_structures_ternary.py or select_final_structures_quaternary.py

    Raises:
        Exception: If directory navigation, file operations, or command
        composition fail.
    """
    import os
    try:
        elements = config[CK.ELEMENTS]
        nb_of_elements = len(elements.split('-'))

        script = "select_final_structures_ternary.py" if nb_of_elements == 3 else "select_final_structures_quaternary.py"
        ehull_script = os.path.join(config[CK.CMS_DIR], script)

        input_file = os.path.join(config[CK.VASP_WORK_DIR], CK.ENERGY_DAT_OUT)
        mp_stable_file = os.path.join(
            config[CK.POST_PROCESSING_OUT_DIR], CK.MP_STABLE_OUT)
        output_file = os.path.join(
            config[CK.POST_PROCESSING_OUT_DIR], "hull.dat")

    except Exception as e:
        raise

    return f"python {ehull_script} --elements {elements} --input {input_file} --stable {mp_stable_file} --output {output_file} --vasp_work_dir {config[CK.VASP_WORK_DIR]}"


@bash_app(executors=[POSTPROCESSING_LABEL])
def calculate_ehul(config):
    return cmd_calculate_ehul(config)


def cmd_convex_hull_color(config):
    """
    Generate updated phase diagrams by plotting the convex hull and
    highlighting the positions of all computed structures and compositions.

    Args:
        config (dict): ConfigManager. The following fields are used:
            - ``CK.ELEMENTS``
            - ``CK.CMS_DIR``
            - ``CK.POST_PROCESSING_OUT_DIR``
            - ``CK.MP_STABLE_OUT``
            - ``CK.POST_PROCESSING_FINAL_OUT``

            See :class:`~tools.config_manager.ConfigManager` for field descriptions.

    Returns:
        A formatted shell command string to execute `plot_convex_hull_ternary.py`
        or `plot_convex_hull_quaternary.py`.

    Raises:
        Exception: If directory navigation or file operations fail.
    """
    import os
    try:
        elements = config[CK.ELEMENTS]
        l_elements = elements.split('-')
        nb_of_elements = len(l_elements)
        script = "plot_convex_hull_ternary.py" if nb_of_elements == 3 else "plot_convex_hull_quaternary.py"

        convex_hull_color_script = os.path.join(config[CK.CMS_DIR], script)
        stable_dat = os.path.join(
            config[CK.POST_PROCESSING_OUT_DIR], CK.MP_STABLE_OUT)
        elename = ''.join(l_elements)
        input_csv = elename + '.csv' if nb_of_elements == 3 else elename + '_quaternary.csv'
        full_path_input_csv = os.path.join(
            config[CK.POST_PROCESSING_OUT_DIR], input_csv)
        output_file = os.path.join(
            config[CK.POST_PROCESSING_OUT_DIR], CK.POST_PROCESSING_FINAL_OUT)
    except Exception as e:
        raise

    return f"python {convex_hull_color_script} --elements {config[CK.ELEMENTS]} --stable {stable_dat} --input {full_path_input_csv} --threshold {config[CK.HULL_ENERGY_THR]} --output {output_file}"


@bash_app(executors=[POSTPROCESSING_LABEL])
def convex_hull_color(config):
    return cmd_convex_hull_color(config)
