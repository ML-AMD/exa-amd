import os
import csv
import sys
import math
import argparse
from multiprocessing import Pool
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar
from itertools import permutations
import warnings

badele_vec = ['D', 'He', 'Ne', 'Ar', 'Br', 'Kr', 'Tc', 'Xe', 'At', 'Rn', 'Pm', 'Fr', 'Rf',
              'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og',
              'Ac', 'Th', 'Pa', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
              'F', 'Cl', 'Br', 'I', 'O']

# badele_vec = []

lattice_scales = None


def generate_structures(structure_file, elements, dirs):
    element_permutations = list(permutations(elements))

    structures = []

    # Read the original structure
    original_structure = Structure.from_file(
        os.path.join(dirs, structure_file))

    # Check if the structure contains any bad elements
    if any(element.symbol in badele_vec for element in original_structure.composition):
        return []  # Skip this structure

    # Identify elements to be substituted (those not in badele_vec)
    elements_to_substitute = [
        element.symbol for element in original_structure.composition]

    for perm in element_permutations:
        for scale in lattice_scales:
            new_structure = original_structure.copy()

            # Replace elements
            for i, site in enumerate(new_structure):
                if site.specie.symbol in elements_to_substitute:
                    new_structure.replace(
                        i, perm[elements_to_substitute.index(site.specie.symbol)])

            # Scale lattice
            new_structure.scale_lattice(new_structure.volume * scale**3)

            structures.append(new_structure)

    return structures


def process_structure(args):
    structure_file, start_index, dirs, elements, chunk_id = args
    structures = generate_structures(structure_file, elements, dirs)
    for i, structure in enumerate(structures, start=start_index):
        structure.to(filename=f"{chunk_id}_{i}.cif")
    return len(structures)


def main(args):
    n_chunks = args.n_chunks
    chunk_id = args.chunk_id

    # sanity check
    if chunk_id < 1 or chunk_id > n_chunks:
        sys.exit("chunk_id must be between 1 and n_chunks.")

    # Suppress UserWarnings
    warnings.filterwarnings("ignore")

    dirs = os.path.abspath(args.input_dir)
    num_workers = args.num_workers
    structure_files = [f for f in os.listdir(dirs) if f.endswith('.cif')]
    elements = [ele for ele in args.elements.split('-')]
    global lattice_scales
    lattice_scales = [0.96, 0.98, 1.0, 1.02, 1.04]

    # calculate the index of the first and last files to process
    chunk_size = math.ceil(len(structure_files) / n_chunks)
    start_file = (chunk_id - 1) * chunk_size
    end_file = min(start_file + chunk_size, len(structure_files))
    sel_files = structure_files[start_file:end_file]

    element_permutations = list(permutations(elements))
    numall = len(element_permutations) * len(lattice_scales)

    args_list = []
    for i, f in enumerate(structure_files):
        if f not in sel_files:
            continue
        args_list.append((f, i * numall + 1, dirs, elements, chunk_id))

    with Pool(num_workers) as pool:
        results = pool.map(process_structure, args_list)

    generated_ids = []
    for (f, start_idx, _, _, _), n in zip(args_list, results):
        generated_ids.extend(range(start_idx, start_idx + n))

    # out_csv = f"id_prop_chunk_{chunk_id}.csv"
    out_csv = f"id_prop.csv"
    with open(out_csv, 'w', newline='') as f:
        for idx in generated_ids:
            f.write(f"{chunk_id}_{idx},0.5\n")


if __name__ == "__main__":
    # num_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    parser = argparse.ArgumentParser(
        description="Generate structures in parallel")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of worker processes")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory containing MP structures")
    parser.add_argument("--elements", type=str, required=True, help="Elements")
    parser.add_argument("--n_chunks", type=int, default=1, help="Total number of chunks")
    parser.add_argument("--chunk_id", type=int, default=1, help="Chunk index (1-based)")
    args = parser.parse_args()
    main(args)
