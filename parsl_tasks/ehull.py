"""Energy-above-hull (Ehull) post-processing task.

This module computes formation energies and energy above the convex hull for
candidate structures produced by DFT (VASP) calculations. Structures with low
energy above the hull are flagged as promising and copied into a ``selected``
directory for further analysis.

It supports both ternary and quaternary chemical systems and exposes a Parsl
:func:`~parsl.app.app.python_app` (:func:`calculate_ehull`) for use within an
executor-based workflow.
"""

import subprocess

from parsl import python_app

from parsl_configs.parsl_executors_labels import POSTPROCESSING_LABEL
from tools.config_labels import ConfigKeys as CK
from tools.logging_config import amd_logger


def cmd_calculate_ehull(config: dict, gather_energy: bool = True) -> str:
    """Select promising structures based on energy above the convex hull.

    Structures with low energy above the hull (Ehull) are identified as
    promising candidates and automatically copied to a dedicated ``selected``
    folder for further analysis, such as evaluation of additional physical
    properties or preparation for experimental validation.

    Parameters
    ----------
    config : dict
        :class:`~tools.config_manager.ConfigManager` (or ``dict``). Keys used:

        - ``elements`` (str): system spec, e.g. ``"Ce-Co-B"``.
        - ``vasp_work_dir`` (str): directory holding per-ID subdirs with
          ``CONTCAR_{id}``.
        - ``energy_dat_out`` (str): filename (under ``vasp_work_dir``) listing
          total energies.
        - ``post_processing_out_dir`` (str): directory for outputs.
        - ``mp_stable_out`` (str): output filename (under
          ``post_processing_out_dir``) with reference stable phases.
    gather_energy : bool, optional
        If ``True`` (default), gather total energies from the DFT calculation
        output files before computing the hull.

    Returns
    -------
    str
        Absolute path to ``{post_processing_out_dir}/hull.dat``.
    """
    import os
    import re
    import shutil

    from pymatgen.core import Element, Structure

    elements = config[CK.ELEMENTS]
    nb_of_elements = len(elements.split("-"))

    input_file = os.path.join(config[CK.VASP_WORK_DIR], CK.ENERGY_DAT_OUT)
    mp_stable_file = os.path.join(config[CK.POST_PROCESSING_OUT_DIR], CK.MP_STABLE_OUT)
    output_file = os.path.join(config[CK.POST_PROCESSING_OUT_DIR], "hull.dat")
    vasp_work_dir = config[CK.VASP_WORK_DIR]

    # check that mp_stable_file exists
    if not os.path.exists(mp_stable_file):
        mp_stable_file = os.path.join(config[CK.VASP_WORK_DIR], CK.MP_STABLE_OUT)
        if not os.path.exists(mp_stable_file):
            amd_logger.critical(
                f"mp_stable_file not found in "
                f"{config[CK.POST_PROCESSING_OUT_DIR]} or "
                f"{config[CK.VASP_WORK_DIR]}"
            )
    # gather the energy from the dft calculations
    if gather_energy:
        os.chdir(config[CK.VASP_WORK_DIR])
        cmd_get_energy = (
            f"grep 'F=' */output_*.en | "
            r"sed 's/\/output\.en://g' > "
            f"{CK.ENERGY_DAT_OUT}"
        )
        result = subprocess.run(cmd_get_energy, shell=True)
        if result.returncode != 0:
            amd_logger.critical(f"{cmd_get_energy} failed")

    def read_energies(
        filename: str, vasp_work_dir: str
    ) -> tuple[list[float], list[int], list[str], list[str]]:
        """Read per-atom energies and structure metadata from an energy file.

        Parameters
        ----------
        filename : str
            Path to the energy data file listing total energies per structure.
        vasp_work_dir : str
            Directory containing per-ID subdirectories with ``CONTCAR_{id}``
            files.

        Returns
        -------
        energies : list of float
            Per-atom total energies for each valid structure.
        indices : list of int
            Structure indices corresponding to each energy.
        formulas : list of str
            Reduced chemical formulas for each structure.
        spgs : list of int
            Space group numbers for each structure.
        """
        energies, formulas, indices, spgs = [], [], [], []
        with open(filename, "r") as f:
            for line in f:
                m = re.search(r"^(\d+).*E0=\s*([-.\dE+]+)", line)
                if not m:
                    continue
                idx = int(m.group(1))
                energy = float(m.group(2))
                contcar = os.path.join(vasp_work_dir, f"{idx}/CONTCAR_{idx}")
                if not os.path.exists(contcar):
                    continue
                try:
                    s = Structure.from_file(contcar)
                    formulas.append(s.composition.reduced_formula)
                    natom = s.num_sites
                    if natom == 0:
                        continue
                    energies.append(energy / natom)
                    indices.append(idx)
                    spg = s.get_space_group_info(symprec=0.02)
                    spgs.append(spg[0])
                except Exception:
                    continue
        return energies, indices, formulas, spgs

    from parsl_tasks.ehull_utils import (
        judge_stable_quaternary,
        judge_stable_ternary,
        parse_stable_phases_quaternary,
        parse_stable_phases_ternary,
    )

    elements_str = elements.split("-")

    # Type hint for formation_energies here, is used twice
    formation_energies: list[float | None] = []

    if nb_of_elements == 3:
        # normalize symbols to match plotting’s filename convention
        elements_symbols = [Element(e).symbol for e in elements_str]
        stable_vec, _ = parse_stable_phases_ternary(mp_stable_file, elements_symbols)
        if not stable_vec:
            return output_file

        total_energies, indices, formulas, spgs = read_energies(
            input_file, vasp_work_dir
        )
        hull_phases: list[list[str] | None] = []

        for energy, formula in zip(total_energies, formulas, strict=True):
            try:
                d_hull, hull_vec = judge_stable_ternary(
                    stable_vec, elements_symbols, formula, energy
                )
                formation_energies.append(d_hull)
                hull_phases.append(hull_vec)
            except Exception:
                formation_energies.append(None)
                hull_phases.append(None)

        valid = [
            (e, f, idx, spg, te, hp)
            for e, f, idx, spg, te, hp in zip(
                formation_energies,
                formulas,
                indices,
                spgs,
                total_energies,
                hull_phases,
                strict=True,
            )
            if e is not None and hp is not None
        ]
        if not valid:
            return output_file

        valid.sort()

        prefix = os.path.dirname(output_file)
        dirname = os.path.join(prefix, "selected")
        os.makedirs(dirname, exist_ok=True)

        with (
            open(output_file, "w+") as f_out,
            open(
                os.path.join(prefix, "".join(elements_symbols) + ".csv"), "w+"
            ) as f_csv,
        ):
            count = 0
            for energy, formula, idx, spg, total_energy, _ in valid:
                count += 1
                f_out.write(f"{idx},{formula},{energy:.6f},{spg}\n")
                f_csv.write(f"{formula},{total_energy:.6f}\n")
                if energy <= 0 or count <= 20:
                    src = os.path.join(vasp_work_dir, f"{idx}/CONTCAR_{idx}")
                    dst = os.path.join(dirname, f"CONTCAR_{idx}")
                    if os.path.exists(src):
                        shutil.copy(src, dst)

        return output_file

    else:
        elements_objs = [Element(ele) for ele in elements_str]
        eles_symbols = [e.symbol for e in elements_objs]
        stable_vec, quaternary_phases = parse_stable_phases_quaternary(
            mp_stable_file, elements_objs
        )
        if not stable_vec:
            return output_file

        total_energies, indices, formulas, spgs = read_energies(
            input_file, vasp_work_dir
        )
        hull_phase_combinations: list[list[str] | None] = []

        for energy, formula in zip(total_energies, formulas, strict=True):
            try:
                d_hull, hull_vec = judge_stable_quaternary(
                    stable_vec, eles_symbols, formula, energy
                )
                if "Error" in hull_vec or "Composition Error" in hull_vec:
                    formation_energies.append(999.0)
                    hull_phase_combinations.append(["N/A"] * 4)
                else:
                    formation_energies.append(d_hull)
                    hull_phase_combinations.append(hull_vec)
            except Exception:
                formation_energies.append(None)
                hull_phase_combinations.append(None)

        valid = [
            (e, f, idx, spg, te, hp)
            for e, f, idx, spg, te, hp in zip(
                formation_energies,
                formulas,
                indices,
                spgs,
                total_energies,
                hull_phase_combinations,
                strict=True,
            )
            if e is not None and hp is not None
        ]
        if not valid:
            return output_file

        valid.sort()

        prefix = os.path.dirname(output_file)
        selected_dirname = os.path.join(prefix, "selected")
        os.makedirs(selected_dirname, exist_ok=True)
        csv_filename = os.path.join(prefix, "".join(eles_symbols) + "_quaternary.csv")

        with open(output_file, "w+") as f_out, open(csv_filename, "w+") as f_csv:
            f_out.write(
                "# Index,Formula,Formation_Energy_per_atom(eV/atom),Spacegroup\n"
            )
            f_csv.write(
                "Formula,Total_Energy_per_atom,Ehull,Hull_Phase1,Hull_Phase2,Hull_Phase3,Hull_Phase4\n"
            )
            count = 0
            for energy, formula, idx, spg, total_energy, phases in valid:
                count += 1
                f_out.write(f"{idx},{formula},{energy:.6f},{spg}\n")
                f_csv.write(
                    f"{formula},{total_energy:.6f},{energy:.6f},{','.join(phases)}\n"
                )
                if energy <= 1e-5 or count <= 20:
                    src = os.path.join(vasp_work_dir, f"{idx}/CONTCAR_{idx}")
                    dst = os.path.join(selected_dirname, f"CONTCAR_{idx}")
                    if os.path.exists(src):
                        shutil.copy(src, dst)

        return output_file


@python_app(executors=[POSTPROCESSING_LABEL])
def calculate_ehull(config: dict) -> str:
    """Parsl app wrapper for :func:`cmd_calculate_ehull`.

    Parameters
    ----------
    config : dict
        Configuration mapping passed through to :func:`cmd_calculate_ehull`.

    Returns
    -------
    str
        Absolute path to the generated ``hull.dat`` file.
    """
    return cmd_calculate_ehull(config)
