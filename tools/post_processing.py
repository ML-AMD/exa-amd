

import os
import numpy as np
import subprocess
import time
import math
import shutil
import re
from pathlib import Path
from shutil import copyfileobj
import subprocess

from mp_api.client import MPRester
from pymatgen.io.vasp.inputs import Incar
from pymatgen.core import Composition, Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from typing import List, Dict, Tuple
from itertools import combinations
from typing import List, Dict, Tuple
from tools.logging_config import amd_logger
from collections import defaultdict

from tools.config_labels import ConfigKeys as CK
from parsl_tasks.hull import run_single_vasp_hull_calculation, compile_vasp_hull


def get_stable_phases(elements: List[str], api_key: str) -> List[Dict]:
    """
    Fetch stable phases from Materials Project and categorize them.

    Args:
        elements: List of element symbols
        api_key: Materials Project API key

    Returns:
        List of dictionaries containing structure information and phase type
    """
    with MPRester(api_key) as mpr:
        criteria = {
            "elements": {"$in": elements, "$all": elements},
            "energy_above_hull": {"$lte": 0.01}
        }

        properties = ["material_id", "formula_pretty", "structure", "elements"]
        # docs = mpr.summary.search(criteria=criteria, properties=properties)
        phases = []
        for ii in range(len(elements)):
            for combo in combinations(elements, ii + 1):
                eles = list(combo)
                # data=mpr.summary.search(num_elements=ii+1,is_stable=True,elements=eles)
                docs = mpr.summary.search(
                    num_elements=ii + 1, is_stable=True, elements=eles)

                for doc in docs:
                    # Determine if phase is elementary, binary, or ternary
                    unique_elements = set(doc.elements)
                    phase_type = "elementary" if len(unique_elements) == 1 else \
                        "binary" if len(unique_elements) == 2 else "ternary"

                    phases.append({
                        "material_id": doc.material_id,
                        "formula": doc.formula_pretty,
                        "structure": doc.structure,
                        "elements": list(unique_elements),
                        "phase_type": phase_type
                    })

        return phases


def get_vasp_hull(config):
    """
    Construct or update the convex hull for the chemical system. Structures on or near the hull are considered potentially stable, while those significantly above the hull are likely metastable or unstable.

    Args:
        config (dict): ConfigManager. The following fields are used:

            * ``CK.ELEMENTS``
            * ``CK.MPRester_API_KEY``
            * ``CK.POT_DIR``
            * ``CK.CMS_DIR``
            * ``CK.VASP_WORK_DIR``
            * ``CK.SUBDIR_STABLE_PHASES``
            * ``CK.POST_PROCESSING_OUT_DIR``
            * ``CK.MP_STABLE_OUT``
            * ``CK.ENERGY_DAT_OUT``

            See :class:`~tools.config_manager.ConfigManager` for detailed
            field descriptions.

    Returns:
        None: The function performs its work through side effects—creating
        directories, launching VASP calculations, and emitting the final hull
        summary file. No value is returned.

    Raises:
        Exception: If directory navigation or file operations fail.
    """
    try:
        elements = config[CK.ELEMENTS].split("-")
        api_key = config[CK.MPRester_API_KEY]
        potcar_dir = config[CK.POT_DIR]
        incar_file = os.path.join(config[CK.CMS_DIR], "INCAR.en")
        incar_mag = os.path.join(config[CK.CMS_DIR], "INCAR_mag.en")
        mageles = ['Fe', 'Co', 'Ni', 'Mn']

        # gather the energy from the dft calculations
        os.chdir(config[CK.VASP_WORK_DIR])
        ENERGY_DAT_OUT = CK.ENERGY_DAT_OUT
        cmd_get_energy = (
            f"grep -h 'F=' */output_*.en | "
            r"sed 's/^[[:blank:]]\+//' | "
            f"sort -t/ -k1,1n > {CK.ENERGY_DAT_OUT}"
        )
        result = subprocess.run(cmd_get_energy, shell=True)
        if result.returncode != 0:
            amd_logger.critical(f"{cmd_get_energy} failed")

        # prepare the input and output paths
        WORK_DIR = os.path.join(
            config[CK.VASP_WORK_DIR], CK.SUBDIR_STABLE_PHASES)
        VASP_CALCS_DIR = os.path.join(WORK_DIR, "vasp_calcs")
        os.makedirs(VASP_CALCS_DIR, exist_ok=True)

        output_file = os.path.join(
            config[CK.POST_PROCESSING_OUT_DIR], CK.MP_STABLE_OUT)
        mp_file = os.path.join(config[CK.VASP_WORK_DIR], CK.MP_STABLE_OUT)

        phases = get_stable_phases(elements, api_key)
        n_structures = len(phases)

        l_futures = []
        for i, phase in enumerate(phases):
            calc_id = i + 1
            calc_dir = os.path.join(VASP_CALCS_DIR, f"calc_{calc_id}")
            os.makedirs(calc_dir, exist_ok=True)
            phase['structure'].to(filename=os.path.join(calc_dir, "POSCAR"))

            # Copy INCAR
            if any(Element(magele) in phase['elements'] for magele in mageles):
                shutil.copy(incar_mag, f"{calc_dir}/INCAR")
            else:
                shutil.copy(incar_file, f"{calc_dir}/INCAR")

            incar_path = Path(calc_dir) / "INCAR"
            text = incar_path.read_text()
            text = re.sub(r"^SYSTEM[ \t]*=[ \t]*.*", f"SYSTEM = {phase['formula']}", text, flags=re.MULTILINE)
            incar_path.write_text(text)

            # Create POTCAR
            potcar_paths = [Path(potcar_dir) / str(elem) / "POTCAR" for elem in phase["structure"].elements]
            out_path = Path(calc_dir) / "POTCAR"

            with out_path.open("wb") as out:
                for p in potcar_paths:
                    with p.open("rb") as inp:
                        copyfileobj(inp, out)

            l_futures.append(
                run_single_vasp_hull_calculation(
                    config, calc_dir))

        for future in l_futures:
            err = future.exception()
            if err:
                amd_logger.warning(f"Post-processing VASP: {err}")
        amd_logger.info("vasp hull calculation done")

        prefix = os.path.join(VASP_CALCS_DIR, f"calc_")
        err = compile_vasp_hull(
            config,
            n_structures,
            output_file,
            prefix).exception()
        if err:
            amd_logger.warning(f"Post-processing VASP: {err}")

        shutil.copy(output_file, mp_file)
        amd_logger.info("vasp hull compilation done")

    except Exception as e:
        amd_logger.critical(f"An exception occurred: {e}")
