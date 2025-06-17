import os
import csv
import sys
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
    structure_file, start_index, dirs, elements = args
    structures = generate_structures(structure_file, elements, dirs)
    for i, structure in enumerate(structures, start=start_index):
        structure.to(filename=f"{i}.cif")
    return len(structures)


def main(args):
    # Suppress UserWarnings
    warnings.filterwarnings("ignore")

    dirs = os.path.abspath(args.input_dir)
    num_workers = args.num_workers
    structure_files = [f for f in os.listdir(dirs) if f.endswith('.cif')]
    elements = [ele for ele in args.elements.split('-')]
    global lattice_scales
    lattice_scales = [0.96, 0.98, 1.0, 1.02, 1.04]
    element_permutations = list(permutations(elements))
    numall = len(element_permutations) * len(lattice_scales)

    args_list = [(f, i * numall + 1, dirs, elements)
                 for i, f in enumerate(structure_files)]

    with Pool(num_workers) as pool:
        results = pool.map(process_structure, args_list)

    total_structures = sum(results)
    print(f"Total structures generated: {total_structures}")
    with open('id_prop.csv', 'w+') as f:
        for i in range(1, total_structures + 1):
            f.write(str(i) + ',0.5' + '\n')


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
    args = parser.parse_args()
    main(args)
