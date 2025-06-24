import os
import re
import shutil
import numpy as np
from pymatgen.core import Structure, Composition, Element
from pymatgen.io.cif import CifWriter
import argparse


def det(v1, v2, v3):
    """Calculate determinant for 3 composition points."""
    matrix = np.array([[v1[1], v1[2], v1[3]],
                      [v2[1], v2[2], v2[3]],
                      [v3[1], v3[2], v3[3]]])
    return np.linalg.det(matrix)


def dhull(struc, hull):
    """Calculate distance to hull using determinant method."""
    tol = 1.e-5
    d_hull_max = -100
    hull_vec = ["0", "0", "0"]
    le = len(hull)

    for i in range(le):
        for j in range(i + 1, le):
            for k in range(j + 1, le):
                det_a = det(hull[i], hull[j], hull[k])
                if abs(det_a) - tol > 0:
                    s1 = det(struc, hull[j], hull[k]) / det_a
                    s2 = det(hull[i], struc, hull[k]) / det_a
                    s3 = det(hull[i], hull[j], struc) / det_a
                    if s1 >= -0.003 and s2 >= -0.003 and s3 >= -0.003:
                        d_convex = struc[4] - s1 * hull[i][4] - \
                            s2 * hull[j][4] - s3 * hull[k][4]
                        if d_hull_max < d_convex:
                            d_hull_max = d_convex
                            hull_vec[0] = hull[i][0]
                            hull_vec[1] = hull[j][0]
                            hull_vec[2] = hull[k][0]

    return d_hull_max, hull_vec


def judge_stable(stable_vec, system, comp_struc, predict_Eper):
    """Calculate stability for a given structure."""
    Comp = Composition(comp_struc)
    xcomp = Comp.get_atomic_fraction(Element(system[0]))
    ycomp = Comp.get_atomic_fraction(Element(system[1]))
    zcomp = Comp.get_atomic_fraction(Element(system[2]))
    struc_vec = [comp_struc, xcomp, ycomp, zcomp, predict_Eper]
    return dhull(struc_vec, stable_vec)


def parse_stable_phases(filename, elements):
    """Parse the stable phases file and create hull vectors."""
    stable_vec = []
    ternary_vec = []
    with open(filename, 'r') as f:
        for line in f:
            formula, energy = line.strip().split()
            comp = Composition(formula)
            # Check if composition contains only our elements of interest
            if all(elem in elements for elem in comp.elements) and \
               all(elem in comp.elements or comp.get(elem, 0) == 0 for elem in elements):
                xcomp = comp.get_atomic_fraction(Element(elements[0]))
                ycomp = comp.get_atomic_fraction(Element(elements[1]))
                zcomp = comp.get_atomic_fraction(Element(elements[2]))
                stable_vec.append(
                    [formula, xcomp, ycomp, zcomp, float(energy)])
                if xcomp > 0 and ycomp > 0 and zcomp > 0:
                    ternary_vec.append(
                        [formula, xcomp, ycomp, zcomp, float(energy)])
    return stable_vec, ternary_vec

# Read elemental energies from mp_element.dat


def read_elemental_energies(filename):
    elemental_energies = {}
    with open(filename, 'r') as f:
        for line in f:
            element, energy = line.split()
            elemental_energies[element] = float(energy)
    return elemental_energies

# Read energies from ener.dat


def read_energies(filename, vasp_work_dir):
    energies = []
    formulas = []
    indices = []
    spgs = []
    with open(filename, 'r') as f:
        for line in f:
            match = re.search(r'^(\d+).*E0=\s*([-.\dE+]+)', line)
            if match:
                index = int(match.group(1))
                energy = float(match.group(2))
                strs = Structure.from_file(
                    f'{vasp_work_dir}/{index}/CONTCAR_' + str(index))
                formulas.append(strs.composition.reduced_formula)
                natom = strs.num_sites
                energies.append(energy / natom)
                indices.append(index)
                spg = strs.get_space_group_info(symprec=0.02)
                spgs.append(spg[0])

    return energies, indices, formulas, spgs


def main():
    parser = argparse.ArgumentParser(
        description='Calculate formation energies relative to convex hull.')
    parser.add_argument("--elements", type=str, required=True, help="Elements")
    parser.add_argument(
        '--input',
        required=True,
        help='Input structure file (default: input_file.txt)')
    parser.add_argument(
        '--stable',
        required=True,
        help='Stable phases file (default: mp_int_stable.dat)')
    parser.add_argument(
        '--output',
        required=True,
        help='Output file (default: formation_energies.txt)')
    parser.add_argument(
        '--vasp_work_dir',
        required=True,
        help='Path to a work directory for VASP-specific operations (required).')

    args = parser.parse_args()
    elements = [Element(ele) for ele in args.elements.split('-')]
    eles = [ele.symbol for ele in elements]

    elename = ''.join(eles)

    # Read stable phases
    stable_vec, ternary_vec = parse_stable_phases(args.stable, elements)

    if not stable_vec:
        print(f"Error: No stable phases found containing elements {eles}")
        return

    print(f"Found {len(stable_vec)} stable phases for system {'-'.join(eles)}")
    for ternaries in ternary_vec:
        print(f"{ternaries[0]}")

    # Process structures
    total_energies, indices, formulas, spgs = read_energies(
        args.input, args.vasp_work_dir)
    formation_energies = []
    hull_phases = []

    for i, energy, formula in zip(indices, total_energies, formulas):
        try:
            d_hull, hull_vec = judge_stable(
                stable_vec, elements, formula, energy)
            formation_energies.append(d_hull)
            hull_phases.append(hull_vec)
        except Exception as e:
            print(f"Warning for structure {i}: {e}")
            formation_energies.append(None)
            hull_phases.append(None)

    # Write formation energies file
    formation_energies, formulas, indices, spgs, total_energies = zip(
        *sorted(zip(formation_energies, formulas, indices, spgs, total_energies)))

    count = 0
    # Create absolute path for selected
    prefix = os.path.dirname(args.output)
    vasp_workdir = args.vasp_work_dir
    dirname = os.path.join(prefix, "selected")
    os.makedirs(dirname, exist_ok=True)

    with open(args.output, 'w+') as f:
        with open(os.path.join(prefix, elename + '.csv'), 'w+') as f0:
            for idx, energy, phases, formula, spg, total_energy in zip(
                    indices, formation_energies, hull_phases, formulas, spgs, total_energies):
                if energy is not None:
                    count += 1
                    f.write(f'{idx},{formula},{energy:.6f},{spg}\n')
                    f0.write(f'{formula},{total_energy:.6f}\n')
                if energy <= 0 or count <= 20:
                    shutil.copy(
                        os.path.join(
                            vasp_workdir,
                            f"{idx}/CONTCAR_{idx}"),
                        os.path.join(
                            dirname,
                            f"CONTCAR_{idx}"))

    print(f"\nProcessed {len(indices)} structures")


if __name__ == '__main__':
    main()
