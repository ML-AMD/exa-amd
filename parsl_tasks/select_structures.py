"""Select and deduplicate candidate structures for downstream workflows.

This module reads candidate structures ranked by formation energy from a CSV
file, applies filtering criteria (formation-energy threshold, maximum atom
count, and per-element minimum fractions), and deduplicates the survivors on a
per-composition basis using
:class:`pymatgen.analysis.structure_matcher.StructureMatcher`.

The selected structures are written to an output directory as an
``id_prop.csv`` index file and individual ``POSCAR_{i}`` files. A Parsl
``python_app`` (:func:`select_structures`) wraps the core logic for use within
a Parsl workflow.
"""

import csv
import math
import multiprocessing as mp
import os
from collections import defaultdict
from multiprocessing import Process, Queue
from typing import Any

from parsl import python_app
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Element, Structure

from parsl_configs.parsl_executors_labels import SELECT_EXECUTOR_LABEL
from tools.config_labels import ConfigKeys as CK


def read_csv(csv_file: str, ef_threshold: float) -> dict[str, float]:
    """Read and rank candidate structures from a CSV file.

    Structures are sorted by formation energy and filtered by ``ef_threshold``.
    If fewer than the internal minimum are retained, the threshold is relaxed
    so that at least the minimum number of structures is returned (subject to
    availability). An internal maximum caps the number returned.

    Parameters
    ----------
    csv_file : str
        Path to the input CSV with three columns ``index, _, Ef`` where
        ``index`` matches CIF filenames and ``Ef`` is a float used for ranking.
    ef_threshold : float
        Maximum allowed ``Ef`` for initial filtering.

    Returns
    -------
    dict of str to float
        Mapping from structure ``index`` to its formation energy ``Ef``.
    """
    # First, read all structures and their energies
    all_structures: list[tuple[str, float]] = []
    with open(csv_file, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            index, _, ef = row[0], row[1], float(row[2])
            all_structures.append((index, ef))

    # Sort structures by formation energy
    all_structures.sort(key=lambda x: x[1])

    # Initialize parameters
    min_structures = 20000
    max_structures = 300000
    structures_data: dict[str, float] = {}

    # First try with original ef_threshold
    for index, ef in all_structures:
        if ef >= ef_threshold or len(structures_data) >= max_structures:
            break
        structures_data[index] = ef

    # If we have fewer than min_structures, take the first min_structures
    # regardless of ef_threshold
    if len(structures_data) < min_structures:
        structures_data = {}
        for i, (index, ef) in enumerate(all_structures):
            if i >= min_structures:
                break
            structures_data[index] = ef
        actual_ef_threshold = (
            all_structures[min_structures - 1][1]
            if len(all_structures) >= min_structures
            else all_structures[-1][1]
        )
        print(
            f"Adjusted Ef threshold to {actual_ef_threshold:.3f} to ensure "
            f"minimum of {min_structures} structures"
        )
    else:
        print(
            f"Selected {len(structures_data)} structures with original "
            f"Ef threshold {ef_threshold}"
        )

    return structures_data


def process_structures(
    task_queue: "Queue",
    result_queue: "Queue",
    nomix_dir: str,
    natom_threshold: int,
    element_fractions: dict[str, float],
) -> None:
    """Worker: load and filter structures from a task queue.

    Consumes ``(index, ef)`` tasks, loads the corresponding CIF, applies the
    atom-count and per-element fraction filters, and pushes surviving results
    onto ``result_queue``. Terminates on receiving ``None`` and then emits the
    sentinel ``'DONE'``.

    Parameters
    ----------
    task_queue : multiprocessing.Queue
        Queue yielding ``(index, ef)`` tuples, or ``None`` to stop.
    result_queue : multiprocessing.Queue
        Queue receiving ``(index, ef, reduced_formula, Structure)`` tuples and
        a final ``'DONE'`` sentinel.
    nomix_dir : str
        Root directory containing CIFs laid out as
        ``{chunk_prefix}/{index}.cif``.
    natom_threshold : int
        Maximum total atoms (reduced formula) allowed per structure.
    element_fractions : dict of str to float
        Minimum atomic-fraction constraints keyed by element symbol.

    Returns
    -------
    None
    """
    while True:
        task = task_queue.get()
        if task is None:
            break
        index, ef = task
        prefix_chunk_dir = index.split("_")[0]
        structure = Structure.from_file(
            os.path.join(nomix_dir, prefix_chunk_dir, f"{index}.cif")
        )
        composition = structure.composition
        reduced_formula = composition.reduced_formula
        flag = 0

        # Check total number of atoms in the reduced formula
        total_atoms = sum(composition.get_reduced_composition_and_factor()[0].values())
        if total_atoms > natom_threshold:
            continue

        # Check element fractions
        if len(element_fractions) > 0:
            for element, min_fraction in element_fractions.items():
                if composition.get_atomic_fraction(Element(element)) < min_fraction:
                    flag = 1
                    break

        if flag == 0:
            result_queue.put((index, ef, reduced_formula, structure))

    result_queue.put("DONE")


def select_structures_for_compositions(
    task_queue: "Queue", result_queue: "Queue", matcher: StructureMatcher
) -> None:
    """Worker: deduplicate structures within each composition group.

    Consumes ``(composition, structures, n_per_composition)`` tasks, selects up
    to ``n_per_composition`` structurally distinct structures (lowest ``Ef``
    first) using ``matcher``, and pushes ``(composition, selected)`` onto
    ``result_queue``. Terminates on receiving ``None``.

    Parameters
    ----------
    task_queue : multiprocessing.Queue
        Queue yielding ``(composition, structures, n_per_composition)`` tuples,
        or ``None`` to stop. Here ``structures`` is a list of
        ``(index, ef, Structure)`` tuples.
    result_queue : multiprocessing.Queue
        Queue receiving ``(composition, selected)`` tuples, where ``selected``
        is a list of ``(index, ef, Structure)`` tuples.
    matcher : pymatgen.analysis.structure_matcher.StructureMatcher
        Matcher used to detect near-duplicate structures.

    Returns
    -------
    None
    """
    while True:
        task = task_queue.get()
        if task is None:
            break
        composition, structures, n_per_composition = task
        selected: list[tuple[str, float, Structure]] = []
        for index, ef, structure in sorted(structures, key=lambda x: x[1]):
            if not any(matcher.fit(structure, s) for _, _, s in selected):
                selected.append((index, ef, structure))
                if len(selected) == n_per_composition:
                    break
        result_queue.put((composition, selected))


def select_structures_core(
    nomix_dir: str,
    output_dir: str,
    csv_file: str,
    ef_threshold: float,
    min_total: int,
    max_total: int,
    num_workers: int,
    natom_threshold: int,
    element_fractions: dict[str, float],
) -> None:
    """Run the full filter-and-deduplicate selection pipeline.

    Reads and ranks candidates, filters them across worker processes, groups by
    composition, deduplicates each group, trims to ``max_total``, and writes the
    results to ``output_dir``.

    Parameters
    ----------
    nomix_dir : str
        Root directory containing input CIFs laid out as
        ``{chunk_prefix}/{index}.cif``.
    output_dir : str
        Directory to write outputs (created if missing). Writes
        ``id_prop.csv`` and ``POSCAR_{i}`` files.
    csv_file : str
        Path to the input CSV with columns ``index, _, Ef``.
    ef_threshold : float
        Maximum allowed ``Ef`` for initial filtering.
    min_total : int
        Desired minimum number of selected structures; a warning is printed if
        the final count is smaller.
    max_total : int
        Hard cap on the number of selected structures.
    num_workers : int
        Number of worker processes for filtering and selection.
    natom_threshold : int
        Maximum total atoms (reduced formula) allowed per structure.
    element_fractions : dict of str to float
        Minimum atomic-fraction constraints keyed by element symbol.

    Returns
    -------
    None
    """
    os.makedirs(output_dir, exist_ok=True)

    structures_data = read_csv(csv_file, ef_threshold)
    print(f"Loaded {len(structures_data)} structures from CSV")

    # Set up queues
    task_queue: "Queue" = mp.Queue()
    result_queue: "Queue" = mp.Queue()

    # Start worker processes for structure processing
    processes: list[Process] = []
    for _ in range(num_workers):
        p = mp.Process(
            target=process_structures,
            args=(
                task_queue,
                result_queue,
                nomix_dir,
                natom_threshold,
                element_fractions,
            ),
        )
        p.start()
        processes.append(p)

    # Add tasks to the queue
    for index, ef in structures_data.items():
        task_queue.put((index, ef))

    # Add termination signals
    for _ in range(num_workers):
        task_queue.put(None)

    # Collect results
    composition_groups: dict[str, list[tuple[str, float, Structure]]] = defaultdict(
        list
    )
    processed_count = 0
    finished_workers = 0
    while finished_workers < num_workers:
        result = result_queue.get()
        if result == "DONE":
            finished_workers += 1
        else:
            index, ef, composition, structure = result
            composition_groups[composition].append((index, ef, structure))
            processed_count += 1

    # Wait for all processes to complete
    for p in processes:
        p.join()

    print("Finished processing structures")

    # Sort compositions by their lowest Ef
    sorted_compositions = sorted(
        composition_groups.keys(),
        key=lambda x: min(s[1] for s in composition_groups[x]),
    )

    num_compositions = len(sorted_compositions)
    n_per_composition = math.ceil(max_total / num_compositions)
    print(f"Number of compositions: {num_compositions}")
    print(f"Estimated structures per composition: {n_per_composition}")

    # Clear queues
    task_queue = mp.Queue()
    result_queue = mp.Queue()

    # Start worker processes for structure selection
    matcher = StructureMatcher()
    processes = []
    for _ in range(num_workers):
        p = mp.Process(
            target=select_structures_for_compositions,
            args=(task_queue, result_queue, matcher),
        )
        p.start()
        processes.append(p)

    # Add tasks to the queue
    for composition in sorted_compositions:
        task_queue.put(
            (composition, composition_groups[composition], n_per_composition)
        )

    # Add termination signals
    for _ in range(num_workers):
        task_queue.put(None)

    # Collect results
    selected_structures = []
    for _ in range(len(sorted_compositions)):
        composition, selected = result_queue.get()
        selected_structures.extend(selected)

    # Wait for all processes to complete
    for p in processes:
        p.join()

    # Sort selected structures by Ef
    selected_structures.sort(key=lambda x: x[1])

    # Trim to max_total if exceeded
    selected_structures = selected_structures[:max_total]

    print(f"Selected {len(selected_structures)} structures")

    # Write selected structures to output
    with open(os.path.join(output_dir, "id_prop.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "Ef"])
        for i, (_, ef, structure) in enumerate(selected_structures, 1):
            writer.writerow([str(i), ef])
            structure.to(filename=os.path.join(output_dir, f"POSCAR_{i}"), fmt="poscar")

    print("Finished writing output files")

    if len(selected_structures) < min_total:
        print(
            "Warning: The final number of selected structures "
            f"({len(selected_structures)}) "
            f"is less than the specified minimum ({min_total})."
        )
        print(
            "This may be due to a lack of sufficiently diverse structures in the"
            " dataset or the filtering criteria."
        )


def run_select_structures(
    output_dir: str,
    nomix_dir: str = "nomix/",
    csv_file: str = "test_results.csv",
    ef_threshold: float = 1.0,
    min_total: int = 1000,
    max_total: int = 4000,
    num_workers: int = mp.cpu_count(),
    natom_threshold: int = 50,
    element_fractions: str = "",
) -> None:
    """Identify and remove duplicate or near-duplicate structures.

    Reads candidates from a CSV file, sorts data by formation energy (Ef) and
    eliminates the structures above the ``ef_threshold``. It then deduplicates
    per composition using
    :class:`pymatgen.analysis.structure_matcher.StructureMatcher`, and writes
    the selected set to ``output_dir``.

    Parameters
    ----------
    output_dir : str
        Directory to write outputs (created if missing). Writes
        ``id_prop.csv`` and ``POSCAR_{i}`` files for selected structures.
    nomix_dir : str, optional
        Root directory containing input CIFs laid out as
        ``{chunk_prefix}/{index}.cif``. Default is ``"nomix/"``.
    csv_file : str, optional
        Path to the input CSV with three columns ``index, _, Ef`` — where
        ``index`` matches CIF filenames and ``Ef`` is a float (formation energy
        or score used for ranking). Default is ``"test_results.csv"``.
    ef_threshold : float, optional
        Maximum allowed ``Ef`` for initial filtering. Default is ``1.0``.
    min_total : int, optional
        Desired minimum number of selected structures. A warning is printed if
        the final count is smaller. Default is ``1000``.
    max_total : int, optional
        Hard cap on the number of selected structures. Default is ``4000``.
    num_workers : int, optional
        Number of worker processes for filtering and selection. Default is
        ``multiprocessing.cpu_count()``.
    natom_threshold : int, optional
        Maximum total atoms (reduced formula) allowed per structure. Default is
        ``50``.
    element_fractions : str, optional
        Comma-separated minimum fraction constraints by element, each formatted
        as ``element:fraction``. Structures with any listed element below its
        fraction are discarded. Empty string disables this filter. Default is
        ``""``.

    Returns
    -------
    None
    """
    element_fractions_dict = {
        elem: float(frac)
        for elem, frac in [
            pair.split(":") for pair in element_fractions.split(",") if pair
        ]
    }

    select_structures_core(
        nomix_dir,
        output_dir,
        csv_file,
        ef_threshold,
        min_total,
        max_total,
        num_workers,
        natom_threshold,
        element_fractions_dict,
    )


@python_app(executors=[SELECT_EXECUTOR_LABEL])
def select_structures(
    config: dict[str, Any],
    out_dir: str,
    min_total: int,
    max_total: int,
) -> None:
    """Parsl app that runs structure selection within a workflow.

    Changes into the configured working directory and invokes
    :func:`run_select_structures` using paths and parameters derived from
    ``config``.

    Parameters
    ----------
    config : dict
        Workflow configuration mapping, keyed by :class:`ConfigKeys` values.
        Must provide the working directory, formation-energy threshold, and
        worker count.
    out_dir : str
        Directory to write selected structures into.
    min_total : int
        Desired minimum number of selected structures.
    max_total : int
        Hard cap on the number of selected structures.

    Returns
    -------
    None
    """
    os.chdir(config[CK.WORK_DIR])
    tr_csv_file = os.path.join(config[CK.WORK_DIR], "test_results.csv")
    dir_structures = os.path.join(config[CK.WORK_DIR], "structures")
    run_select_structures(
        out_dir,
        nomix_dir=dir_structures,
        csv_file=tr_csv_file,
        ef_threshold=float(config[CK.EF_THR]),
        min_total=min_total,
        max_total=max_total,
        num_workers=int(config[CK.NUM_WORKERS]),
    )
