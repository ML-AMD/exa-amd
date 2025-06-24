import os
import re
import shutil
import numpy as np
from pymatgen.core import Structure, Composition, Element
from pymatgen.io.cif import CifWriter
import argparse
from itertools import combinations  # Added for combinations


def det4(v1, v2, v3, v4):
    """Calculate determinant for 4 composition points (quaternary)."""
    # Vectors should be [x, y, z, w] fractions
    matrix = np.array([
        [v1[1], v1[2], v1[3], v1[4]],
        [v2[1], v2[2], v2[3], v2[4]],
        [v3[1], v3[2], v3[3], v3[4]],
        [v4[1], v4[2], v4[3], v4[4]]
    ])
    return np.linalg.det(matrix)


def dhull_quaternary(struc, hull):
    """Calculate distance to hull for quaternary systems."""
    # struc should be [formula, x, y, z, w, energy]
    # hull entries should be [formula, x, y, z, w, energy]
    tol = 1.e-5  # Tolerance for checking if point is on or inside the facet
    d_hull_max = -100  # Initialize with a very small number
    hull_vec = ["0"] * 4  # Store the 4 points defining the hull facet
    le = len(hull)

    # Iterate through all combinations of 4 points from the hull
    for combo in combinations(range(le), 4):
        i, j, k, l_idx = combohull_points = [hull[i], hull[j], hull[k], hull[l_idx]]

        # Check if the 4 points form a non-degenerate tetrahedron in composition space
        # We use a simplified check: ensure the points are not collinear/coplanar
        # A full check involves checking volumes, but determinant serves as a proxy here
        # We need a reference point (e.g., origin or average) to calculate volume robustly,
        # but for stability checks, focusing on barycentric coordinates is
        # often sufficient.

        # Represent points relative to one of them to check for degeneracy
        # using a 3x3 det
        try:
            matrix_3d_check = np.array([
                [hull[j][1] - hull[i][1], hull[j][2] -
                    hull[i][2], hull[j][3] - hull[i][3]],
                [hull[k][1] - hull[i][1], hull[k][2] -
                    hull[i][2], hull[k][3] - hull[i][3]],
                [hull[l_idx][1] - hull[i][1], hull[l_idx][2] -
                    hull[i][2], hull[l_idx][3] - hull[i][3]]
            ])
            det_check = np.linalg.det(matrix_3d_check)
        except IndexError:
            print(
                f"Error checking degeneracy for hull points: "
                f"{hull[i][0]}, {hull[j][0]}, {hull[k][0]}, {hull[l_idx][0]}. "
                "Ensure they are 4-component."
            )
            continue  # Skip this combination if dimensions are wrong

        if abs(det_check) > tol:  # Check if points form a volume (non-degenerate)
            # Calculate barycentric coordinates using matrix inversion or Cramer's rule
            # System: s1*p1 + s2*p2 + s3*p3 + s4*p4 = p_struc
            #         s1 + s2 + s3 + s4 = 1
            # Where pi = [xi, yi, zi, wi] compositional coordinates
            A = np.array([
                [hull[i][1], hull[j][1], hull[k][1], hull[l_idx][1]],
                [hull[i][2], hull[j][2], hull[k][2], hull[l_idx][2]],
                [hull[i][3], hull[j][3], hull[k][3], hull[l_idx][3]],
                [hull[i][4], hull[j][4], hull[k][4], hull[l_idx][4]]
            ])
            b = np.array([struc[1], struc[2], struc[3], struc[4]])

            try:
                # Solve Ax = b for x (barycentric coords s1, s2, s3, s4)
                coords = np.linalg.solve(A, b)
                s1, s2, s3, s4 = coords

                # Check if the structure's composition lies within the tetrahedron
                # Add tolerance for numerical stability
                if s1 >= -tol and s2 >= -tol and s3 >= -tol and s4 >= - \
                        tol and abs(s1 + s2 + s3 + s4 - 1.0) < tol:
                    # Calculate the energy difference to the hull facet
                    d_convex = struc[5] - (s1 * hull[i][5] + s2 *
                                           hull[j][5] + s3 * hull[k][5] + s4 * hull[l_idx][5])

                    # Update if this is the largest positive distance (least stable)
                    # or smallest negative distance (most stable below hull)
                    # For Ehull, we want the distance *above* the hull, so max
                    # positive value
                    if d_hull_max < d_convex:
                        d_hull_max = d_convex
                        hull_vec[0] = hull[i][0]
                        hull_vec[1] = hull[j][0]
                        hull_vec[2] = hull[k][0]
                        hull_vec[3] = hull[l_idx][0]

            except np.linalg.LinAlgError:
                # Matrix A is singular, points are likely degenerate, skip
                continue

    # If d_hull_max remains -100, no valid hull facet was found containing the
    # composition
    if abs(d_hull_max - (-100)) < tol:
        # This might happen if the point is outside the compositional space covered by the hull input
        # or if the hull itself is not comprehensive. Handle as needed (e.g., return None or raise error)
        # print(f"Warning: Could not find containing hull facet for {struc[0]}.")
        # For simplicity, return a large negative number indicating instability
        # or error
        return -100, ["Error"] * 4

    return d_hull_max, hull_vec


def judge_stable_quaternary(
        stable_vec, system_elements, comp_struc, predict_Eper):
    """Calculate stability for a given structure in a quaternary system."""
    Comp = Composition(comp_struc)
    if len(system_elements) != 4:
        raise ValueError(
            "System must contain exactly 4 elements for quaternary calculation.")

    # Get atomic fractions for the 4 elements
    xcomp = Comp.get_atomic_fraction(Element(system_elements[0]))
    ycomp = Comp.get_atomic_fraction(Element(system_elements[1]))
    zcomp = Comp.get_atomic_fraction(Element(system_elements[2]))
    wcomp = Comp.get_atomic_fraction(Element(system_elements[3]))

    # Ensure fractions sum to 1 (within tolerance)
    if not abs(xcomp + ycomp + zcomp + wcomp - 1.0) < 1e-6:
        # Handle cases where the structure might contain other elements
        # print(f"Warning: Composition {comp_struc} fractions do not sum to 1 for elements {system_elements}. Skipping.")
        # Depending on use case, might raise error or return default
        # instability
        return -100, ["Composition Error"] * 4  # Indicate error/instability

    struc_vec = [comp_struc, xcomp, ycomp, zcomp, wcomp, predict_Eper]
    return dhull_quaternary(struc_vec, stable_vec)


def parse_stable_phases_quaternary(filename, elements):
    """Parse the stable phases file and create hull vectors for quaternary systems."""
    if len(elements) != 4:
        raise ValueError(
            "Must provide exactly 4 elements for quaternary parsing.")

    stable_vec = []
    quaternary_vec = []  # Store phases containing all 4 elements
    # Set for faster lookup
    element_symbols = {elem.symbol for elem in elements}

    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue  # Skip empty or invalid lines
            formula = parts[0]
            try:
                energy = float(parts[-1])  # Assume energy is the last part
            except ValueError:
                print(
                    f"Warning: Could not parse energy for line: {
                        line.strip()}. Skipping.")
                continue

            try:
                comp = Composition(formula)
                # Check if composition contains ONLY our elements of interest
                comp_elements = {elem.symbol for elem in comp.elements}
                if comp_elements.issubset(element_symbols):
                    # All elements in the formula are within our system
                    xcomp = comp.get_atomic_fraction(elements[0])
                    ycomp = comp.get_atomic_fraction(elements[1])
                    zcomp = comp.get_atomic_fraction(elements[2])
                    wcomp = comp.get_atomic_fraction(elements[3])
                    # Ensure the composition fractions sum to 1
                    if abs(xcomp + ycomp + zcomp + wcomp - 1.0) < 1e-6:
                        stable_vec.append(
                            [formula, xcomp, ycomp, zcomp, wcomp, energy])
                        # Check if it contains all 4 elements (is truly
                        # quaternary)
                        if xcomp > 1e-6 and ycomp > 1e-6 and zcomp > 1e-6 and wcomp > 1e-6:
                            quaternary_vec.append(
                                [formula, xcomp, ycomp, zcomp, wcomp, energy])

            except Exception as e:
                print(
                    f"Warning: Could not process formula {formula} from stable phases file: {e}. Skipping.")

    return stable_vec, quaternary_vec

# read_elemental_energies remains the same


def read_elemental_energies(filename):
    elemental_energies = {}
    with open(filename, 'r') as f:
        for line in f:
            element, energy = line.split()
            elemental_energies[element] = float(energy)
    return elemental_energies

# read_energies remains the same


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
                contcar_file = f'{vasp_work_dir}/{index}/CONTCAR_{index}'
                if not os.path.exists(contcar_file):
                    print(
                        f"Warning: {contcar_file} not found. Skipping index {index}.")
                    continue
                try:
                    strs = Structure.from_file(contcar_file)
                    formulas.append(strs.composition.reduced_formula)
                    natom = strs.num_sites
                    if natom == 0:
                        print(
                            f"Warning: Structure {index} has 0 atoms. Skipping.")
                        continue
                    energies.append(energy / natom)
                    indices.append(index)
                    spg = strs.get_space_group_info(
                        symprec=0.02)  # Consider adjusting symprec
                    spgs.append(spg[0])
                except Exception as e:
                    print(
                        f"Warning: Could not read or process {contcar_file}: {e}. Skipping index {index}.")

    return energies, indices, formulas, spgs


def main():
    parser = argparse.ArgumentParser(
        description='Calculate formation energies relative to convex hull for QUATERNARY systems.')
    parser.add_argument(
        "--elements",
        type=str,
        required=True,
        help="Elements (4 required, separated by hyphen, e.g., A-B-C-D)")
    parser.add_argument(
        '--input',
        required=True,
        help='Input energy file (default: ener.dat)')
    parser.add_argument(
        '--stable',
        required=True,
        help='Stable phases file (default: mp_int_stable.dat)')
    parser.add_argument(
        '--output',
        required=True,
        help='Output file (default: hull_quaternary.dat)')
    parser.add_argument(
        '--vasp_work_dir',
        required=True,
        help='Path to a work directory for VASP-specific operations (required).')

    args = parser.parse_args()

    # Use provided elements
    elements_str = args.elements.split('-')
    if len(elements_str) != 4:
        print(
            "Error: Exactly 4 elements must be provided via --elements flag (e.g., A-B-C-D)")
        return
    elements = [Element(ele) for ele in elements_str]
    eles_symbols = [ele.symbol for ele in elements]  # List of symbols

    elename = ''.join(eles_symbols)  # For output file naming

    # Read stable phases using the quaternary parser
    try:
        stable_vec, quaternary_phases = parse_stable_phases_quaternary(
            args.stable, elements)
    except ValueError as e:
        print(f"Error parsing stable phases: {e}")
        return
    except FileNotFoundError:
        print(f"Error: Stable phases file '{args.stable}' not found.")
        return

    if not stable_vec:
        print(
            f"Error: No stable phases found containing only elements {eles_symbols} in {
                args.stable}")
        return

    print(
        f"Found {
            len(stable_vec)} stable phases for system {
            '-'.join(eles_symbols)}")
    print(f"Found {len(quaternary_phases)} specifically quaternary phases:")

    # Process structures from ener.dat
    try:
        total_energies, indices, formulas, spgs = read_energies(
            args.input, args.vasp_work_dir)
    except FileNotFoundError:
        print(f"Error: Input energy file '{args.input}' not found.")
        return

    formation_energies = []
    hull_phase_combinations = []  # Store the 4 hull points

    print(f"\nProcessing {len(indices)} structures from {args.input}...")
    processed_count = 0
    for i, energy, formula in zip(indices, total_energies, formulas):
        try:
            # Use the quaternary stability function
            d_hull, hull_vec = judge_stable_quaternary(
                stable_vec, eles_symbols, formula, energy)

            # Check if hull calculation was successful
            if "Error" in hull_vec or "Composition Error" in hull_vec:
                print(
                    f"Warning for structure {i}({formula}): Could not determine stability({hull_vec[0]}). Setting Ehull to large positive.")
                # Assign large positive value for plotting/sorting
                formation_energies.append(999.0)
                hull_phase_combinations.append(["N/A"] * 4)
            else:
                formation_energies.append(d_hull)
                hull_phase_combinations.append(hull_vec)
                processed_count += 1

        # Catch errors from judge_stable(e.g., wrong element count)
        except ValueError as e:
            print(f"Error processing structure {i}({formula}): {e}")
            formation_energies.append(None)  # Mark as None if error occurs
            hull_phase_combinations.append(None)
        except Exception as e:  # Catch unexpected errors
            print(f"Unexpected error for structure {i}({formula}): {e}")
            formation_energies.append(None)
            hull_phase_combinations.append(None)

    print(
        f"Finished processing. Calculated Ehull for {processed_count} structures.")

    # Combine results and filter out None values before sorting
    valid_results = []
    for i in range(len(indices)):
        if formation_energies[i] is not None and hull_phase_combinations[i] is not None:
            valid_results.append(
                (formation_energies[i],
                 formulas[i],
                 indices[i],
                 spgs[i],
                 total_energies[i],
                 hull_phase_combinations[i]))

    if not valid_results:
        print("\nNo valid formation energies could be calculated.")
        return

    # Sort by formation energy(Ehull)
    valid_results.sort()  # Sorts by the first element(formation_energy)

    # Prepare for output
    count = 0
    vasp_workdir = args.vasp_work_dir
    output_filename = args.output

    prefix = os.path.dirname(output_filename)
    selected_dirname = os.path.join(prefix, "selected")
    os.makedirs(selected_dirname, exist_ok=True)

    csv_filename = os.path.join(prefix, elename + '_quaternary.csv')
    print(f"\nWriting results to {output_filename} and {csv_filename}...")
    with open(output_filename, 'w+') as f_out, open(csv_filename, 'w+') as f_csv:
        f_out.write(
            "# Index,Formula,Formation_Energy_per_atom(eV/atom),Spacegroup\n")
        # Example CSV header:
        # Formula,Total_Energy_per_atom,Ehull,Hull_Phase1,Hull_Phase2,Hull_Phase3,Hull_Phase4
        f_csv.write(
            "Formula,Total_Energy_per_atom,Ehull,Hull_Phase1,Hull_Phase2,Hull_Phase3,Hull_Phase4\n")

        for energy, formula, idx, spg, total_energy, phases in valid_results:
            count += 1
            f_out.write(f'{idx},{formula},{energy:.6f},{spg}\n')
            phase_str = ",".join(phases)  # Join the 4 hull phases for CSV
            f_csv.write(f'{formula},{total_energy:.6f},{energy:.6f},{phase_str}\n')

            # Copy CONTCAR file if stable(Ehull <= 0) or among the lowest 20
            # unstable
            if energy <= 1e-5 or count <= 20:  # Use tolerance for stability check
                src_contcar = os.path.join(
                    vasp_workdir, f"{idx}/CONTCAR_{idx}")
                dest_contcar = os.path.join(selected_dirname, f"CONTCAR_{idx}")
                if os.path.exists(src_contcar):
                    shutil.copy(src_contcar, dest_contcar)
                else:
                    print(f"Warning: CONTCAR_{idx} not found for copying.")

    print(f"\nProcessed and wrote {count} structures.")
    print(f"Stable(Ehull <= 0) and lowest 20 unstable CONTCARs copied to '{selected_dirname}'.")


if __name__ == '__main__':
    main()
