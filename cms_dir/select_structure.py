import os
import csv
import argparse
from collections import defaultdict
from pymatgen.core import Structure, Element
from pymatgen.analysis.structure_matcher import StructureMatcher
import multiprocessing as mp
import math


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

    # If we have fewer than min_structures, take the first min_structures regardless of ef_threshold
    if len(structures_data) < min_structures:
        structures_data = {}
        for i, (index, ef) in enumerate(all_structures):
            if i >= min_structures:
                break
            structures_data[index] = ef
        actual_ef_threshold = all_structures[min_structures-1][1] if len(
            all_structures) >= min_structures else all_structures[-1][1]
        print(
            f"Adjusted Ef threshold to {actual_ef_threshold:.3f} to ensure minimum of {min_structures} structures")
    else:
        print(
            f"Selected {len(structures_data)} structures with original Ef threshold {ef_threshold}")

    return structures_data


def process_structures(task_queue, result_queue, nomix_dir, natom_threshold, element_fractions):
    while True:
        task = task_queue.get()
        if task is None:
            break
        index, ef = task
        structure = Structure.from_file(
            os.path.join(nomix_dir, f"{index}.cif"))
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
                if composition.get_atomic_fraction(Element(element)) < min_fraction:
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


def select_structures(nomix_dir, output_dir, csv_file, ef_threshold, min_total, max_total, num_workers, natom_threshold, element_fractions):
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

    # Collect results
    # composition_groups = defaultdict(list)
    # for _ in range(len(structures_data)):
    #    index, ef, composition, structure = result_queue.get()
    #    composition_groups[composition].append((index, ef, structure))

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


def main():
    parser = argparse.ArgumentParser(
        description="Optimized Multiprocessing Structure Selection Script")
    parser.add_argument("--nomix_dir", default="nomix/",
                        help="Directory containing hypo structures")
    parser.add_argument("--output_dir", default="new/",
                        help="Output directory for selected structures")
    parser.add_argument("--csv_file", default="test_results.csv",
                        help="Path to test_results.csv")
    parser.add_argument("--ef_threshold", type=float,
                        default=-0.2, help="Ef threshold for selection")
    parser.add_argument("--min_total", type=int, default=1000,
                        help="Minimum total number of structures to select")
    parser.add_argument("--max_total", type=int, default=4000,
                        help="Maximum total number of structures to select")
    parser.add_argument("--num_workers", type=int, default=mp.cpu_count(),
                        help="Number of worker processes to use")
    parser.add_argument("--natom_threshold", type=int, default=50,
                        help="Maximum number of atoms in reduced formula")
    parser.add_argument("--element_fractions", type=str, default="",
                        help="Minimum atomic fractions for elements, format: 'Element1:fraction1,Element2:fraction2,...'")

    args = parser.parse_args()

    # Parse element fractions
    element_fractions = {elem: float(frac) for elem, frac in [pair.split(
        ':') for pair in args.element_fractions.split(',') if pair]}

    select_structures(args.nomix_dir, args.output_dir, args.csv_file,
                      args.ef_threshold, args.min_total, args.max_total,
                      args.num_workers, args.natom_threshold, element_fractions)


if __name__ == "__main__":
    main()
