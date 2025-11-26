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
    import re
    import subprocess
    from importlib import resources as iresources
    from pathlib import Path
    from tools.errors import VaspNonReached

    def _is_timeout(rc):
        return rc == 124

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
    success = False
    timed_out = False
    try:
        exec_cmd_prefix = (
            "" if config[CK.VASP_NTASKS_PER_RUN] == 1
            else f"srun -N 1 -n {config[CK.VASP_NTASKS_PER_RUN]} --exact --cpu-bind=cores"
        )
        work_subdir = os.path.join(config[CK.VASP_WORK_DIR], str(id))
        os.makedirs(work_subdir, exist_ok=True)
        os.chdir(work_subdir)

        #
        # prepare relaxation
        #
        output_rx = Path(work_subdir) / "output.rx"

        vasp_std_exe = config[CK.VASP_STD_EXE]
        poscar = os.path.join(config[CK.WORK_DIR], "new", f"POSCAR_{id}")
        with iresources.as_file(iresources.files("workflows.vasp_assets") / "INCAR.rx") as p:
            incar_src = str(p)

        # POTCAR symlink
        potcar_src = os.path.join(config[CK.WORK_DIR], "POTCAR")
        if not Path("POTCAR").exists():
            os.symlink(potcar_src, "POTCAR")

        # relaxation inputs
        shutil.copy(poscar, os.path.join(work_subdir, "POSCAR"))
        shutil.copy(incar_src, os.path.join(work_subdir, "INCAR"))

        # set NSW in INCAR
        VASP_NSW = config[CK.VASP_NSW]
        incar = Path("INCAR")

        text = incar.read_text()
        text = re.sub(r"NSW\s*=\s*\d*", f"NSW = {VASP_NSW}", text)
        incar.write_text(text)

        # run relaxation
        with open(output_rx, "w") as out:
            rc = subprocess.run(
                ["timeout", str(config[CK.VASP_TIMEOUT]), *exec_cmd_prefix.split(), vasp_std_exe],
                stdout=out,
                stderr=subprocess.STDOUT
            )
            if rc.returncode != 0:
                if _is_timeout(rc.returncode):
                    timed_out = True
                raise VaspNonReached(f"VASP exited with {rc} during relaxation")

        #  grep "reached"
        out_text = output_rx.read_text(errors="ignore")
        reached = ("reached" in out_text.lower())

        # grep "{NSW} F="
        re_nsw = re.compile(rf"(?m)^\s*{VASP_NSW}\s+F=")
        hit_nsw = bool(re_nsw.search(out_text))

        if reached:
            os.rename("OUTCAR", f"OUTCAR_{id}.rx")
            shutil.copy("CONTCAR", os.path.join(work_subdir, f"CONTCAR_{id}"))

        elif hit_nsw:
            # re-run relaxation from CONTCAR -> POSCAR
            shutil.copy("CONTCAR", "POSCAR")
            with open(output_rx, "w") as out:
                rc = subprocess.run(
                    ["timeout", str(config[CK.VASP_TIMEOUT]), *exec_cmd_prefix.split(), vasp_std_exe],
                    stdout=out,
                    stderr=subprocess.STDOUT
                )
                if rc.returncode != 0:
                    if _is_timeout(rc.returncode):
                        timed_out = True
                    raise VaspNonReached(f"VASP exited with {rc} during 2nd relaxation")

            shutil.copy("CONTCAR", os.path.join(work_subdir, f"CONTCAR_{id}"))
            os.rename("OUTCAR", f"OUTCAR_{id}.rx")

        else:
            raise VaspNonReached

        # energy calculation
        with iresources.as_file(iresources.files("workflows.vasp_assets") / "INCAR.en") as p:
            incar_en = str(p)

        output_file_en = os.path.join(work_subdir, f"output_{id}.en")

        shutil.copy("CONTCAR", "POSCAR")
        shutil.copy(incar_en, "INCAR")

        with open(output_file_en, "w") as out:
            subprocess.run(
                ["timeout", str(config[CK.VASP_TIMEOUT]), *exec_cmd_prefix.split(), vasp_std_exe],
                stdout=out,
                stderr=subprocess.STDOUT
            )
        os.rename("OUTCAR", f"OUTCAR_{id}.en")
        success = True
    finally:
        try:
            if success or timed_out:
                Path("DONE").touch()
        except Exception:
            pass
        cleanup()


@python_app(executors=[VASP_EXECUTOR_LABEL])
def fused_vasp_calc(config, id, walltime=(int)):
    cmd_fused_vasp_calc(config, id, walltime)


def run_vasp_calc(config, id):
    return fused_vasp_calc(config, id, walltime=2 * config[CK.VASP_TIMEOUT])
