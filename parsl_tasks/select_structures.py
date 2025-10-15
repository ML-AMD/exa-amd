import os
import csv
import argparse
from collections import defaultdict
from pymatgen.core import Structure, Element
from pymatgen.analysis.structure_matcher import StructureMatcher
import multiprocessing as mp
import math
from parsl import python_app
from parsl_configs.parsl_executors_labels import SELECT_EXECUTOR_LABEL
from tools.config_labels import ConfigKeys as CK


def read_csv(csv_file, ef_threshold):
    # First, read all structures and their energies
    all_structures = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            index, _, ef = row[0], row[1], float(row[2])
            all_structures.append((index, ef))

    # Sort structures by formation energy
    all_structures.sort(key=lambda x: x[1])

    # Initialize parameters
    min_structures = 20000
    max_structures = 300000
    structures_data = {}

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
        actual_ef_threshold = all_structures[min_structures - 1][1] if len(
            all_structures) >= min_structures else all_structures[-1][1]
        print(
            f"Adjusted Ef threshold to {actual_ef_threshold:.3f} to ensure minimum of {min_structures} structures")
    else:
        print(
            f"Selected {len(structures_data)} structures with original Ef threshold {ef_threshold}")

    return structures_data


def process_structures(task_queue, result_queue, nomix_dir,
                       natom_threshold, element_fractions):
    while True:
        task = task_queue.get()
        if task is None:
            break
        index, ef = task
        prefix_chunk_dir = index.split("_")[0]
        structure = Structure.from_file(
            os.path.join(nomix_dir, prefix_chunk_dir, f"{index}.cif"))
        composition = structure.composition
        reduced_formula = composition.reduced_formula
        flag = 0

        # Check total number of atoms in the reduced formula
        total_atoms = sum(
            composition.get_reduced_composition_and_factor()[0].values())
        if total_atoms > natom_threshold:
            continue

        # Check element fractions
        if len(element_fractions) > 0:
            for element, min_fraction in element_fractions.items():
                if composition.get_atomic_fraction(
                        Element(element)) < min_fraction:
                    flag = 1
                    break

        if flag == 0:
            result_queue.put((index, ef, reduced_formula, structure))

    result_queue.put('DONE')


def select_structures_for_compositions(task_queue, result_queue, matcher):
    while True:
        task = task_queue.get()
        if task is None:
            break
        composition, structures, n_per_composition = task
        selected = []
        for index, ef, structure in sorted(structures, key=lambda x: x[1]):
            if not any(matcher.fit(structure, s) for _, _, s in selected):
                selected.append((index, ef, structure))
                if len(selected) == n_per_composition:
                    break
        result_queue.put((composition, selected))


def select_structures_core(nomix_dir, output_dir, csv_file, ef_threshold,
                           min_total, max_total, num_workers, natom_threshold, element_fractions):
    os.makedirs(output_dir, exist_ok=True)

    structures_data = read_csv(csv_file, ef_threshold)
    print(f"Loaded {len(structures_data)} structures from CSV")

    # Set up queues
    task_queue = mp.Queue()
    result_queue = mp.Queue()

    # Start worker processes for structure processing
    processes = []
    for _ in range(num_workers):
        p = mp.Process(target=process_structures, args=(
            task_queue, result_queue, nomix_dir, natom_threshold, element_fractions))
        p.start()
        processes.append(p)

    # Add tasks to the queue
    for index, ef in structures_data.items():
        task_queue.put((index, ef))

    # Add termination signals
    for _ in range(num_workers):
        task_queue.put(None)

    # Collect results
    composition_groups = defaultdict(list)
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
    sorted_compositions = sorted(composition_groups.keys(),
                                 key=lambda x: min(s[1] for s in composition_groups[x]))

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
        p = mp.Process(target=select_structures_for_compositions,
                       args=(task_queue, result_queue, matcher))
        p.start()
        processes.append(p)

    # Add tasks to the queue
    for composition in sorted_compositions:
        task_queue.put(
            (composition, composition_groups[composition], n_per_composition))

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
    with open(os.path.join(output_dir, 'id_prop.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['index', 'Ef'])
        for i, (index, ef, structure) in enumerate(selected_structures, 1):
            writer.writerow([str(i), ef])
            structure.to(filename=os.path.join(
                output_dir, f"POSCAR_{i}"), fmt="poscar")

    print("Finished writing output files")

    if len(selected_structures) < min_total:
        print(
            f"Warning: The final number of selected structures ({len(selected_structures)}) is less than the specified minimum ({min_total}).")
        print("This may be due to a lack of sufficiently diverse structures in the dataset or the filtering criteria.")


def run_select_structures(nomix_dir="nomix/",
                          output_dir="new/",
                          csv_file="test_results.csv",
                          ef_threshold=-0.2,
                          min_total=1000,
                          max_total=4000,
                          num_workers=mp.cpu_count(),
                          natom_threshold=50,
                          element_fractions=""):
    """
    Identify and remove duplicate or near-duplicate structures,
    based on a structural similarity threshold.

    Reads candidates from a CSV file, sort data by formation energy (Ef) and
    eliminates the structures above the `ef_threshold`.
    It then deduplicates per composition using
    :class:`pymatgen.analysis.structure_matcher.StructureMatcher`, and writes
    the selected set to ``output_dir``.

    :param str nomix_dir:
        Root directory containing input CIFs laid out as
        ``{chunk_prefix}/{index}.cif``.

    :param str output_dir:
        Directory to write outputs (created if missing). Writes
        ``id_prop.csv`` and ``POSCAR_{i}`` files for selected structures.

    :param str csv_file:
        Path to the input CSV with three columns:
        ``index, _ , Ef`` — where ``index`` matches CIF filenames and
        ``Ef`` is a float (formation energy or score used for ranking).

    :param float ef_threshold:
        Maximum allowed ``Ef`` for initial filtering.

    :param int min_total:
        Desired minimum number of selected structures. A warning is printed
        if the final count is smaller.

    :param int max_total:
        Hard cap on the number of selected structures.

    :param int num_workers:
        Number of worker processes (threads) for filtering and selection.

    :param int natom_threshold:
        Maximum total atoms (reduced formula) allowed per structure.

    :param str element_fractions:
        Comma-separated minimum fraction constraints by element.
        Structures with any listed element below
        its fraction are discarded. Empty string disables this filter.

    :returns: None
    :rtype: None

    """
    element_fractions = {elem: float(frac) for elem, frac in [pair.split(
        ':') for pair in element_fractions.split(',') if pair]}

    select_structures_core(nomix_dir, output_dir, csv_file,
                           ef_threshold, min_total, max_total,
                           num_workers, natom_threshold, element_fractions)


@python_app(executors=[SELECT_EXECUTOR_LABEL])
def select_structures(config):
    try:
        os.chdir(config[CK.WORK_DIR])
        tr_csv_file = os.path.join(config[CK.WORK_DIR], "test_results.csv")
        dir_structures = os.path.join(config[CK.WORK_DIR], "structures")
        run_select_structures(
            nomix_dir=dir_structures,
            csv_file=tr_csv_file,
            ef_threshold=float(config[CK.EF_THR]),
            num_workers=int(config[CK.NUM_WORKERS])
        )
    except Exception as e:
        raise
