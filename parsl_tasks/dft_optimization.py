import re
import subprocess
from pathlib import Path
from parsl import python_app, bash_app, join_app
import importlib.resources as pkg_resources

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
        - Cleans intermediate VASP output files on completion or error
    """
    import os
    import shutil
    import time
    from tools.errors import VaspNonReached

    def cleanup():
        cleanup_files = [
            "DOSCAR", "PCDAT", "REPORT", "XDATCAR", "CHG",
            "CHGCAR", "EIGENVAL", "PROCAR", "WAVECAR", "vasprun.xml"
        ]
        for fname in cleanup_files:
            try:
                os.remove(fname)
            except FileNotFoundError:
                pass

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
        with pkg_resources.path("workflows.vasp_assets", "INCAR.rx") as p:
            incar = str(p)
        os.symlink(os.path.join(config[CK.WORK_DIR], "POTCAR"), "POTCAR")

        # relaxation
        shutil.copy(poscar, os.path.join(work_subdir, "POSCAR"))
        shutil.copy(incar, os.path.join(work_subdir, "INCAR"))

        # Change NSW iterations
        VASP_NSW = config[CK.VASP_NSW]
        incar = Path("INCAR")

        text = incar.read_text()
        text = re.sub(r"NSW\s*=\s*\d*", f"NSW = {VASP_NSW}", text)
        incar.write_text(text)

        # run relaxation
        with open(output_file, "w") as out:
            result = subprocess.run(
                ["timeout", str(config[CK.VASP_TIMEOUT]), *exec_cmd_prefix.split(), vasp_std_exe],
                stdout=out,
                stderr=subprocess.STDOUT
            )
        relaxation_status = result.returncode

        #
        # prepare energy calculation
        #
        output_rx = Path(work_subdir) / "output.rx"
        relaxation_criteria = 1
        with open(output_rx, "r") as f:
            for line in f:
                if "reached" in line or f"{VASP_NSW} F=" in line:
                    relaxation_criteria = 0
                    break

        # check relaxation criteria
        if relaxation_status != 0 and relaxation_criteria != 0:
            raise VaspNonReached

        with pkg_resources.path("workflows.vasp_assets", "INCAR.en") as p:
            incar_en = str(p)
        
        output_file_en = os.path.join(work_subdir, f"output_{id}.en")

        os.rename("OUTCAR", f"OUTCAR_{id}.rx")
        shutil.copy("CONTCAR", os.path.join(work_subdir, f"CONTCAR_{id}"))
        shutil.copy("CONTCAR", "POSCAR")
        shutil.copy(incar_en, "INCAR")

        # run relaxation
        with open(output_file_en, "w") as out:
            result = subprocess.run(
                ["timeout", str(config[CK.VASP_TIMEOUT]), *exec_cmd_prefix.split(), vasp_std_exe],
                stdout=out,
                stderr=subprocess.STDOUT
            )
    finally:
        cleanup()


@python_app(executors=[VASP_EXECUTOR_LABEL])
def fused_vasp_calc(config, id, walltime=(int)):
    cmd_fused_vasp_calc(config, id, walltime)


def run_vasp_calc(config, id):
    return fused_vasp_calc(config, id, walltime=2 * config[CK.VASP_TIMEOUT])
