from parsl import python_app, bash_app, join_app

from parsl_configs.parsl_executors_labels import VASP_EXECUTOR_LABEL
from tools.config_labels import ConfigKeys as CK


def cmd_fused_vasp_calc(config, id, walltime=(int)):
    """
    Run a two-stage VASP calculation via a Python Parsl task.

    It start by running a relaxation phase trying to find
    the lowest-energy configuration. If relaxation was successful,
    it runs the energy calculaction

    Args:
        config (ConfigManager or dict): Configuration object. The following fields are used:
            - vasp_ntasks_per_run
            - vasp_work_dir
            - work_dir
            - cms_dir
            - vasp_std_exe
            - vasp_timeout
            - vasp_nsw

            See :class:`~tools.config_manager.ConfigManager` for field descriptions.

        id (int): Identifier for the structure being processed. Used to name files and subdirectories.

        walltime (int, optional): Timeout in seconds for each VASP run.

    Raises:
        VaspNonReached: If the relaxation step fails to reach the convergence threshold.
        Exception: For general file I/O or execution failures.

    Side Effects:
        - Creates and modifies files in a per-structure working directory
        - Executes VASP via shell command (`os.system`)
        - Cleans intermediate VASP output files on completion or error
    """
    import os
    import shutil
    import time
    from tools.errors import VaspNonReached
    clean_work_dir = "rm DOSCAR PCDAT REPORT XDATCAR CHG CHGCAR EIGENVAL PROCAR WAVECAR vasprun.xml"

    try:
        exec_cmd_prefix = (
            "" if config[CK.VASP_NTASKS_PER_RUN] == 1
            else f"srun -n {config[CK.VASP_NTASKS_PER_RUN]}"
        )
        work_subdir = os.path.join(config[CK.VASP_WORK_DIR], str(id))
        if not os.path.exists(work_subdir):
            os.makedirs(work_subdir)
        os.chdir(work_subdir)

        #
        # prepare relaxation
        #
        output_file = os.path.join(work_subdir, "output.rx")

        vasp_std_exe = config[CK.VASP_STD_EXE]
        poscar = os.path.join(config[CK.WORK_DIR], "new", f"POSCAR_{id}")
        incar = os.path.join(config[CK.CMS_DIR], "INCAR.rx")
        os.symlink(os.path.join(config[CK.WORK_DIR], "POTCAR"), "POTCAR")

        # relaxation
        shutil.copy(poscar, os.path.join(work_subdir, "POSCAR"))
        shutil.copy(incar, os.path.join(work_subdir, "INCAR"))

        # Change NSW iterations
        VASP_NSW = config[CK.VASP_NSW]
        os.system(f"sed -i 's/NSW\\s*=\\s*[0-9]*/NSW = {VASP_NSW}/' INCAR")

        # run relaxation
        srun_cmd = (
            f"timeout {config[CK.VASP_TIMEOUT]} {exec_cmd_prefix} {vasp_std_exe} > {output_file} "
        )
        relaxation_status = os.system(srun_cmd)

        #
        # prepare energy calculation
        #
        output_rx = os.path.join(work_subdir, "output.rx")
        relaxation_criteria = os.system(
            f"grep -q -e 'reached' -e '{VASP_NSW} F=' output.rx")

        # check relaxation criteria
        if relaxation_status != 0 and relaxation_criteria != 0:
            raise VaspNonReached

        incar_en = os.path.join(config[CK.CMS_DIR], "INCAR.en")
        output_file_en = os.path.join(work_subdir, f"output_{id}.en")

        os.rename("OUTCAR", f"OUTCAR_{id}.rx")
        shutil.copy("CONTCAR", os.path.join(work_subdir, f"CONTCAR_{id}"))
        shutil.copy("CONTCAR", "POSCAR")
        shutil.copy(incar_en, "INCAR")

        # run relaxation
        srun_cmd = (
            f"timeout {config[CK.VASP_TIMEOUT]} {exec_cmd_prefix} {vasp_std_exe} > {output_file_en} "
        )
        os.system(srun_cmd)

        # clean
        os.system(clean_work_dir)
    except Exception as e:
        os.system(clean_work_dir)
        raise


@python_app(executors=[VASP_EXECUTOR_LABEL])
def fused_vasp_calc(config, id, walltime=(int)):
    cmd_fused_vasp_calc(config, id, walltime)


def run_vasp_calc(config, id):
    return fused_vasp_calc(config, id, walltime=2 * config[CK.VASP_TIMEOUT])
