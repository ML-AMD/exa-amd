"""Generation of hypothetical crystal structures for the exa-amd workflow.

This module produces candidate structures from a set of initial CIF files by
permuting the target elements over the crystallographic sites and scaling the
lattice by a fixed set of factors. The work is partitioned into chunks so that
it can be distributed across Parsl executors.

Attributes
----------
badele_vec : list of str
    Element symbols that are disallowed. Any initial structure containing one
    of these elements is skipped.
LATTICE_SCALES : list of float
    Linear scaling factors applied to the lattice for each permutation.
"""

import math
import os
import warnings
from itertools import permutations
from multiprocessing import Pool
from typing import Any, Mapping

from parsl import python_app
from pymatgen.core import Structure

from parsl_configs.parsl_executors_labels import GENERATE_EXECUTOR_LABEL
from tools.config_labels import ConfigKeys as CK

badele_vec = [
    "D",
    "He",
    "Ne",
    "Ar",
    "Br",
    "Kr",
    "Tc",
    "Xe",
    "At",
    "Rn",
    "Pm",
    "Fr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
    "Ac",
    "Th",
    "Pa",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "F",
    "Cl",
    "Br",
    "I",
    "O",
]

LATTICE_SCALES = [0.96, 0.98, 1.0, 1.02, 1.04]


def _generate_structures(
    structure_file: str, elements: list[str], dirs: str
) -> list[Structure]:
    """Generate structures by permuting elements and scaling lattices.

    Parameters
    ----------
    structure_file : str
        File name of the initial CIF structure (relative to ``dirs``).
    elements : list of str
        Target element symbols to permute over the structure sites.
    dirs : str
        Directory containing ``structure_file``.

    Returns
    -------
    list of pymatgen.core.Structure
        Generated structures. Empty if the initial structure contains any
        element listed in :data:`badele_vec`.
    """
    element_permutations = list(permutations(elements))
    structures = []
    original_structure = Structure.from_file(os.path.join(dirs, structure_file))

    # Skip if any disallowed element present
    if any(element.symbol in badele_vec for element in original_structure.composition):
        return []

    elements_to_substitute = [el.symbol for el in original_structure.composition]

    for perm in element_permutations:
        for scale in LATTICE_SCALES:
            new_structure = original_structure.copy()
            for i, site in enumerate(new_structure):
                if site.specie.symbol in elements_to_substitute:
                    new_structure.replace(
                        i, perm[elements_to_substitute.index(site.specie.symbol)]
                    )
            new_structure.scale_lattice(new_structure.volume * (scale**3))
            structures.append(new_structure)

    return structures


def _process_structure(
    args: tuple[str, int, str, list[str], int],
) -> int:
    """Process a single structure file and write generated CIFs.

    Parameters
    ----------
    args : tuple
        A 5-tuple ``(structure_file, start_index, dirs, elements, chunk_id)``
        where ``structure_file`` is the CIF file name, ``start_index`` is the
        first output index, ``dirs`` is the input directory, ``elements`` is
        the list of target element symbols, and ``chunk_id`` is the chunk
        identifier used in output file names.

    Returns
    -------
    int
        Number of structures generated and written to disk.
    """
    structure_file, start_index, dirs, elements, chunk_id = args
    structures = _generate_structures(structure_file, elements, dirs)
    for i, structure in enumerate(structures, start=start_index):
        structure.to(filename=f"{chunk_id}_{i}.cif")
    return len(structures)


def run_gen_structures(config: Mapping[str, Any], n_chunks: int, chunk_id: int) -> str:
    """
    Parsl task that generates hypothetical structures from initial crystal structures.

    The total search space is partitioned into ``n_chunks`` disjoint segments.
    This task processes the segment identified by ``chunk_id``.

    Parameters
    ----------
    config : Mapping[str, Any]
        A :class:`~tools.config_manager.ConfigManager` (or mapping with the same
        fields). The following keys are read:

        - ``work_dir`` (str): root working directory where outputs are written
        - ``num_workers`` (int): number of parallel workers for the inner loop
        - ``elements`` (str): target system (e.g., "Ce-Co-B")
        - ``initial_structures_dir`` (str): directory containing initial structures

        See :class:`~tools.config_manager.ConfigManager` for complete field
        descriptions and defaults.
    n_chunks : int
        Total number of chunks for the workload.
    chunk_id : int
        One-based index of the partition to execute, where
        ``1 <= chunk_id <= n_chunks``.

    Returns
    -------
    str
        Absolute path to this chunk's ``id_prop.csv``.

    Raises
    ------
    SystemExit
        If ``chunk_id`` is not between 1 and ``n_chunks``.
    Exception
        On directory navigation or file I/O failures.
    """
    dir_structures = os.path.join(config[CK.WORK_DIR], "structures", str(chunk_id))
    input_dir = config[CK.INITIAL_STRS]
    num_workers = int(config[CK.NUM_WORKERS])

    if not os.path.exists(dir_structures):
        os.makedirs(dir_structures)
    os.chdir(dir_structures)

    if chunk_id < 1 or chunk_id > n_chunks:
        raise SystemExit("chunk_id must be between 1 and n_chunks.")

    warnings.filterwarnings("ignore")

    dirs = os.path.abspath(input_dir)
    structure_files = [f for f in os.listdir(dirs) if f.endswith(".cif")]
    elements = [ele for ele in str(config[CK.ELEMENTS]).split("-")]

    # Divide work into chunks
    chunk_size = math.ceil(len(structure_files) / n_chunks) if n_chunks > 0 else 0
    start_file = (chunk_id - 1) * chunk_size
    end_file = min(start_file + chunk_size, len(structure_files))
    sel_files = structure_files[start_file:end_file]

    element_permutations = list(permutations(elements))
    numall = len(element_permutations) * len(LATTICE_SCALES)

    args_list = []
    for i, f in enumerate(structure_files):
        if f not in sel_files:
            continue
        args_list.append((f, i * numall + 1, dirs, elements, chunk_id))

    results = []
    if args_list:
        with Pool(num_workers) as pool:
            results = pool.map(_process_structure, args_list)

    generated_ids = []
    for (f, start_idx, _, _, _), n in zip(args_list, results):
        generated_ids.extend(range(start_idx, start_idx + n))

    out_csv = "id_prop.csv"
    with open(out_csv, "w", newline="") as f:
        for idx in generated_ids:
            f.write(f"{chunk_id}_{idx},0.5\n")

    return os.path.abspath(out_csv)


@python_app(executors=[GENERATE_EXECUTOR_LABEL])
def gen_structures(config: Mapping[str, Any], n_chunks: int, chunk_id: int) -> str:
    """Parsl app wrapper around :func:`run_gen_structures`.

    Parameters
    ----------
    config : Mapping[str, Any]
        Configuration mapping; see :func:`run_gen_structures`.
    n_chunks : int
        Total number of chunks for the workload.
    chunk_id : int
        One-based index of the partition to execute.

    Returns
    -------
    str
        Absolute path to this chunk's ``id_prop.csv``.
    """
    return run_gen_structures(config, n_chunks, chunk_id)
