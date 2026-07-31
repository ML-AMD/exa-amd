"""Post-processing utilities for convex-hull construction.

This module fetches stable phases from the Materials Project and builds (or
updates) the convex hull for a chemical system by staging and launching VASP
calculations, then compiling their energies into a hull summary file.
"""

import importlib.resources as pkg_resources
import os
import re
import shutil
from itertools import combinations
from pathlib import Path
from shutil import copyfileobj
from typing import Any, Dict, List

from mp_api.client import MPRester  # type: ignore[import-not-found]
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from parsl_tasks.compile_hull import compile_vasp_hull
from parsl_tasks.hull import run_single_vasp_hull_calculation
from tools.config_labels import ConfigKeys as CK
from tools.logging_config import amd_logger


def get_stable_phases(elements: List[str], api_key: str) -> List[Dict[str, Any]]:
    """Fetch and categorize stable phases from the Materials Project.

    Every non-empty combination of the supplied elements is queried for
    thermodynamically stable entries, which are then labelled by the number
    of distinct constituent elements.

    Parameters
    ----------
    elements : list[str]
        Element symbols composing the chemical system (e.g. ``["Na", "B"]``).
    api_key : str
        Materials Project API key used to authenticate the ``MPRester``.

    Returns
    -------
    list[dict]
        One dict per stable phase with the keys ``material_id``, ``formula``,
        ``structure``, ``elements`` and ``phase_type`` (one of
        ``"elementary"``, ``"binary"`` or ``"ternary"``).
    """
    properties = ["material_id", "formula_pretty", "structure", "elements"]
    phases: List[Dict[str, Any]] = []

    # Return plain dicts (avoid document models to bypass pydantic/env skew issues)
    with MPRester(api_key, use_document_model=False) as mpr:
        for r in range(1, len(elements) + 1):
            for combo in combinations(elements, r):
                eles = list(combo)

                # Prefer 'fields=', fall back to 'properties='
                # (both seen across mp-api versions)
                try:
                    docs = mpr.summary.search(
                        num_elements=r, is_stable=True, elements=eles, fields=properties
                    )
                except TypeError:
                    docs = mpr.summary.search(
                        num_elements=r,
                        is_stable=True,
                        elements=eles,
                        properties=properties,
                    )

                for doc in docs:
                    d = doc if isinstance(doc, dict) else dict(doc)
                    unique_elements = set(d.get("elements", []) or [])
                    if not unique_elements:
                        continue
                    phase_type = (
                        "elementary"
                        if len(unique_elements) == 1
                        else ("binary" if len(unique_elements) == 2 else "ternary")
                    )

                    phases.append(
                        {
                            "material_id": d.get("material_id"),
                            "formula": d.get("formula_pretty"),
                            "structure": d.get("structure"),
                            "elements": list(unique_elements),
                            "phase_type": phase_type,
                        }
                    )

    return phases


def get_vasp_hull(config: Dict[str, Any]) -> None:
    """Construct or update the convex hull for the chemical system.

    Structures on or near the hull are considered potentially stable, while
    those significantly above it are likely metastable or unstable. Stable
    phases are fetched from the Materials Project, staged as VASP inputs
    (POSCAR, INCAR and POTCAR), relaxed, and compiled into a hull summary. The
    function returns early if a compiled hull already exists.

    Parameters
    ----------
    config : dict
        Configuration mapping (a ``ConfigManager``). The following fields are
        used:

        * ``CK.ELEMENTS``
        * ``CK.MPRester_API_KEY``
        * ``CK.POT_DIR``
        * ``CK.VASP_WORK_DIR``
        * ``CK.SUBDIR_STABLE_PHASES``
        * ``CK.POST_PROCESSING_OUT_DIR``
        * ``CK.MP_STABLE_OUT``
        * ``CK.ENERGY_DAT_OUT``

        See :class:`~tools.config_manager.ConfigManager` for detailed field
        descriptions.

    Returns
    -------
    None
        The function works through side effects—creating directories,
        launching VASP calculations and emitting the final hull summary file.

    Raises
    ------
    Exception
        Any error during directory navigation or file operations is caught
        and logged rather than propagated.
    """
    try:
        elements = config[CK.ELEMENTS].split("-")
        api_key = config[CK.MPRester_API_KEY]
        potcar_dir = config[CK.POT_DIR]
        with pkg_resources.path("workflows.vasp_assets", "INCAR.en") as p:
            incar_file = str(p)
        with pkg_resources.path("workflows.vasp_assets", "INCAR_mag.en") as p:
            incar_mag = str(p)
        mageles = ["Fe", "Co", "Ni", "Mn"]

        # prepare the input and output paths
        WORK_DIR = os.path.join(config[CK.VASP_WORK_DIR], CK.SUBDIR_STABLE_PHASES)
        VASP_CALCS_DIR = os.path.join(WORK_DIR, "vasp_calcs")
        os.makedirs(VASP_CALCS_DIR, exist_ok=True)

        output_file = os.path.join(config[CK.POST_PROCESSING_OUT_DIR], CK.MP_STABLE_OUT)
        mp_file = os.path.join(config[CK.VASP_WORK_DIR], CK.MP_STABLE_OUT)

        if os.path.isfile(mp_file) and os.path.getsize(mp_file) > 0:
            amd_logger.info(f"compiled hull exists already: {mp_file}")
            return

        phases = get_stable_phases(elements, api_key)
        n_structures = len(phases)

        l_futures = []
        for i, phase in enumerate(phases):
            calc_id = i + 1
            calc_dir = os.path.join(VASP_CALCS_DIR, f"calc_{calc_id}")
            os.makedirs(calc_dir, exist_ok=True)

            sga = SpacegroupAnalyzer(phase["structure"], symprec=0.05)
            symm_structure = sga.get_refined_structure()
            symm_structure.to(filename=os.path.join(calc_dir, "POSCAR"))

            # Copy INCAR
            if any(magele in phase["elements"] for magele in mageles):
                shutil.copy(incar_mag, f"{calc_dir}/INCAR")
            else:
                shutil.copy(incar_file, f"{calc_dir}/INCAR")

            incar_path = Path(calc_dir) / "INCAR"
            text = incar_path.read_text()
            text = re.sub(
                r"^SYSTEM[ \t]*=[ \t]*.*",
                f"SYSTEM = {phase['formula']}",
                text,
                flags=re.MULTILINE,
            )
            incar_path.write_text(text)

            # Create POTCAR
            potcar_paths = [
                Path(potcar_dir) / str(elem) / "POTCAR"
                for elem in phase["structure"].elements
            ]
            out_path = Path(calc_dir) / "POTCAR"

            with out_path.open("wb") as out:
                for p in potcar_paths:
                    with p.open("rb") as inp:
                        copyfileobj(inp, out)

            l_futures.append(run_single_vasp_hull_calculation(config, calc_dir))

        for future in l_futures:
            err = future.exception()
            if err:
                amd_logger.critical(f"get_vasp_hull VASP: {err}")
        amd_logger.info("vasp hull calculation done")

        prefix = os.path.join(VASP_CALCS_DIR, "calc_")
        err = compile_vasp_hull(n_structures, output_file, prefix).exception()
        if err:
            amd_logger.warning(f"get_vasp_hull VASP: {err}")

        shutil.copy(output_file, mp_file)
        amd_logger.info("vasp hull compilation done")

    except Exception as e:
        amd_logger.critical(f"An exception occurred: {e}")
