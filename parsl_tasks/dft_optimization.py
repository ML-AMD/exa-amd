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
    the lowest-energy configuration. If relaxation is successful,
    it runs the energy calculaction

    :param dict config:
        :class:`~tools.config_manager.ConfigManager` (or dict). Keys used:
        - ``vasp_work_dir`` (str): directory for per-structure work subdirs.
        - ``work_dir`` (str): project root holding inputs (e.g., ``new/``, ``POTCAR``).
        - ``vasp_std_exe`` (str): path to the VASP executable (e.g., ``vasp_std``).
        - ``vasp_timeout`` (int, s): max walltime per VASP invocation.
        - ``vasp_nsw`` (int): number of ionic steps (NSW) for relaxation.

    :param int id:
        Structure identifier: maps to ``POSCAR_{id}`` and names outputs.

    :param int walltime:
        Per-run timeout in seconds (unused; superseded by ``config[CK.VASP_TIMEOUT]``).

    :returns: None
    :rtype: None

    :raises VaspNonReached: if relaxation fails to meet criteria.
    :raises Exception: on file I/O or subprocess failures.

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
