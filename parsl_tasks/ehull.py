from parsl import python_app
from tools.config_labels import ConfigKeys as CK
from parsl_configs.parsl_executors_labels import POSTPROCESSING_LABEL
from parsl import python_app
from tools.config_labels import ConfigKeys as CK
from parsl_configs.parsl_executors_labels import POSTPROCESSING_LABEL


def cmd_calculate_ehul(config):
    """
    Select promising structures: Structures with low energy above the hull (Ehull) are identified as promising candidates. These are automatically copied to a dedicated folder for further analysis, such as evaluation of additional physical properties or preparation for experimental validation.

    :param dict config:
        :class:`~tools.config_manager.ConfigManager` (or dict). Keys used:
        - ``elements`` (str): system spec, e.g. ``"Ce-Co-B"``.
        - ``vasp_work_dir`` (str): directory holding per-ID subdirs with ``CONTCAR_{id}``.
        - ``energy_dat_out`` (str): filename (under ``vasp_work_dir``) listing total energies.
        - ``post_processing_out_dir`` (str): directory for outputs.
        - ``mp_stable_out`` (str): output filename (under ``post_processing_out_dir``) with reference stable phases.

    :returns: Absolute path to ``{post_processing_out_dir}/hull.dat``.
    :rtype: str

    """
    import os
    import re
    import shutil
    import numpy as np
    from pymatgen.core import Structure, Composition, Element

    elements = config[CK.ELEMENTS]
    nb_of_elements = len(elements.split('-'))

    input_file = os.path.join(config[CK.VASP_WORK_DIR], CK.ENERGY_DAT_OUT)
    mp_stable_file = os.path.join(config[CK.POST_PROCESSING_OUT_DIR], CK.MP_STABLE_OUT)
    output_file = os.path.join(config[CK.POST_PROCESSING_OUT_DIR], "hull.dat")
    vasp_work_dir = config[CK.VASP_WORK_DIR]

    def read_energies(filename, vasp_work_dir):
        energies, formulas, indices, spgs = [], [], [], []
        with open(filename, 'r') as f:
            for line in f:
                m = re.search(r'^(\d+).*E0=\s*([-.\dE+]+)', line)
                if not m:
                    continue
                idx = int(m.group(1))
                energy = float(m.group(2))
                contcar = os.path.join(vasp_work_dir, f"{idx}/CONTCAR_{idx}")
                if not os.path.exists(contcar):
                    continue
                try:
                    s = Structure.from_file(contcar)
                    formulas.append(s.composition.reduced_formula)
                    natom = s.num_sites
                    if natom == 0:
                        continue
                    energies.append(energy / natom)
                    indices.append(idx)
                    spg = s.get_space_group_info(symprec=0.02)
                    spgs.append(spg[0])
                except Exception:
                    continue
        return energies, indices, formulas, spgs

    def parse_stable_phases_ternary(filename, elements_symbols):
        stable_vec, ternary_vec = [], []
        with open(filename, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                formula, energy = parts
                energy = float(energy)
                comp = Composition(formula)
                comp_symbols = {e.symbol for e in comp.elements}
                if not comp_symbols.issubset(set(elements_symbols)):
                    continue
                x = comp.get_atomic_fraction(Element(elements_symbols[0]))
                y = comp.get_atomic_fraction(Element(elements_symbols[1]))
                z = comp.get_atomic_fraction(Element(elements_symbols[2]))
                stable_vec.append([formula, x, y, z, energy])
                if x > 0 and y > 0 and z > 0:
                    ternary_vec.append([formula, x, y, z, energy])
        return stable_vec, ternary_vec

    def parse_stable_phases_quaternary(filename, elements):
        if len(elements) != 4:
            raise ValueError("Must provide exactly 4 elements for quaternary parsing.")
        stable_vec, quaternary_vec = [], []
        element_symbols = {elem.symbol for elem in elements}
        with open(filename, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                formula = parts[0]
                try:
                    energy = float(parts[-1])
                except ValueError:
                    continue
                try:
                    comp = Composition(formula)
                    comp_elements = {e.symbol for e in comp.elements}
                    if comp_elements.issubset(element_symbols):
                        x = comp.get_atomic_fraction(elements[0])
                        y = comp.get_atomic_fraction(elements[1])
                        z = comp.get_atomic_fraction(elements[2])
                        w = comp.get_atomic_fraction(elements[3])
                        if abs(x + y + z + w - 1.0) < 1e-6:
                            stable_vec.append([formula, x, y, z, w, energy])
                            if x > 1e-6 and y > 1e-6 and z > 1e-6 and w > 1e-6:
                                quaternary_vec.append([formula, x, y, z, w, energy])
                except Exception:
                    continue
        return stable_vec, quaternary_vec

    def det_tern(v1, v2, v3):
        m = np.array([[v1[1], v1[2], v1[3]],
                      [v2[1], v2[2], v2[3]],
                      [v3[1], v3[2], v3[3]]])
        return np.linalg.det(m)

    def dhull_ternary(struc, hull):
        tol = 1.e-5
        d_hull_max = -100
        hull_vec = ["0", "0", "0"]
        le = len(hull)
        for i in range(le):
            for j in range(i + 1, le):
                for k in range(j + 1, le):
                    det_a = det_tern(hull[i], hull[j], hull[k])
                    if abs(det_a) - tol > 0:
                        s1 = det_tern(struc, hull[j], hull[k]) / det_a
                        s2 = det_tern(hull[i], struc, hull[k]) / det_a
                        s3 = det_tern(hull[i], hull[j], struc) / det_a
                        if s1 >= -0.003 and s2 >= -0.003 and s3 >= -0.003:
                            d_convex = struc[4] - s1 * hull[i][4] - s2 * hull[j][4] - s3 * hull[k][4]
                            if d_hull_max < d_convex:
                                d_hull_max = d_convex
                                hull_vec = [hull[i][0], hull[j][0], hull[k][0]]
        return d_hull_max, hull_vec

    def judge_stable_ternary(stable_vec, system_symbols, comp_struc, predict_Eper):
        Comp = Composition(comp_struc)
        x = Comp.get_atomic_fraction(Element(system_symbols[0]))
        y = Comp.get_atomic_fraction(Element(system_symbols[1]))
        z = Comp.get_atomic_fraction(Element(system_symbols[2]))
        struc_vec = [comp_struc, x, y, z, predict_Eper]
        return dhull_ternary(struc_vec, stable_vec)

    def dhull_quaternary(struc, hull):
        tol = 1.e-5
        d_hull_max = -100
        hull_vec = ["0"] * 4
        le = len(hull)
        from itertools import combinations
        for i, j, k, l_idx in combinations(range(le), 4):
            try:
                m3 = np.array([
                    [hull[j][1] - hull[i][1], hull[j][2] - hull[i][2], hull[j][3] - hull[i][3]],
                    [hull[k][1] - hull[i][1], hull[k][2] - hull[i][2], hull[k][3] - hull[i][3]],
                    [hull[l_idx][1] - hull[i][1], hull[l_idx][2] - hull[i][2], hull[l_idx][3] - hull[i][3]],
                ])
                det_check = np.linalg.det(m3)
            except IndexError:
                continue
            if abs(det_check) > tol:
                A = np.array([
                    [hull[i][1], hull[j][1], hull[k][1], hull[l_idx][1]],
                    [hull[i][2], hull[j][2], hull[k][2], hull[l_idx][2]],
                    [hull[i][3], hull[j][3], hull[k][3], hull[l_idx][3]],
                    [hull[i][4], hull[j][4], hull[k][4], hull[l_idx][4]],
                ])
                b = np.array([struc[1], struc[2], struc[3], struc[4]])
                try:
                    s1, s2, s3, s4 = np.linalg.solve(A, b)
                    if s1 >= -tol and s2 >= -tol and s3 >= -tol and s4 >= -tol and abs(s1 + s2 + s3 + s4 - 1.0) < tol:
                        d_convex = struc[5] - (s1 * hull[i][5] + s2 * hull[j][5] + s3 * hull[k][5] + s4 * hull[l_idx][5])
                        if d_hull_max < d_convex:
                            d_hull_max = d_convex
                            hull_vec = [hull[i][0], hull[j][0], hull[k][0], hull[l_idx][0]]
                except np.linalg.LinAlgError:
                    continue
        if abs(d_hull_max - (-100)) < tol:
            return -100, ["Error"] * 4
        return d_hull_max, hull_vec

    def judge_stable_quaternary(stable_vec, system_elements, comp_struc, predict_Eper):
        Comp = Composition(comp_struc)
        if len(system_elements) != 4:
            raise ValueError("System must contain exactly 4 elements for quaternary calculation.")
        x = Comp.get_atomic_fraction(Element(system_elements[0]))
        y = Comp.get_atomic_fraction(Element(system_elements[1]))
        z = Comp.get_atomic_fraction(Element(system_elements[2]))
        w = Comp.get_atomic_fraction(Element(system_elements[3]))
        if not abs(x + y + z + w - 1.0) < 1e-6:
            return -100, ["Composition Error"] * 4
        struc_vec = [comp_struc, x, y, z, w, predict_Eper]
        return dhull_quaternary(struc_vec, stable_vec)

    elements_str = elements.split('-')

    if nb_of_elements == 3:
        # normalize symbols to match plotting’s filename convention
        elements_symbols = [Element(e).symbol for e in elements_str]
        stable_vec, _ = parse_stable_phases_ternary(mp_stable_file, elements_symbols)
        if not stable_vec:
            return output_file

        total_energies, indices, formulas, spgs = read_energies(input_file, vasp_work_dir)
        formation_energies, hull_phases = [], []

        for i, energy, formula in zip(indices, total_energies, formulas):
            try:
                d_hull, hull_vec = judge_stable_ternary(stable_vec, elements_symbols, formula, energy)
                formation_energies.append(d_hull)
                hull_phases.append(hull_vec)
            except Exception:
                formation_energies.append(None)
                hull_phases.append(None)

        valid = [(e, f, idx, spg, te, hp)
                 for e, f, idx, spg, te, hp in zip(formation_energies, formulas, indices, spgs, total_energies, hull_phases)
                 if e is not None and hp is not None]
        if not valid:
            return output_file

        valid.sort()

        prefix = os.path.dirname(output_file)
        dirname = os.path.join(prefix, "selected")
        os.makedirs(dirname, exist_ok=True)

        with open(output_file, 'w+') as f_out, open(os.path.join(prefix, ''.join(elements_symbols) + '.csv'), 'w+') as f_csv:
            count = 0
            for energy, formula, idx, spg, total_energy, phases in valid:
                count += 1
                f_out.write(f'{idx},{formula},{energy:.6f},{spg}\n')
                f_csv.write(f'{formula},{total_energy:.6f}\n')
                if energy <= 0 or count <= 20:
                    src = os.path.join(vasp_work_dir, f"{idx}/CONTCAR_{idx}")
                    dst = os.path.join(dirname, f"CONTCAR_{idx}")
                    if os.path.exists(src):
                        shutil.copy(src, dst)

        return output_file

    else:
        elements_objs = [Element(ele) for ele in elements_str]
        eles_symbols = [e.symbol for e in elements_objs]
        stable_vec, quaternary_phases = parse_stable_phases_quaternary(mp_stable_file, elements_objs)
        if not stable_vec:
            return output_file

        total_energies, indices, formulas, spgs = read_energies(input_file, vasp_work_dir)
        formation_energies, hull_phase_combinations = [], []

        for i, energy, formula in zip(indices, total_energies, formulas):
            try:
                d_hull, hull_vec = judge_stable_quaternary(stable_vec, eles_symbols, formula, energy)
                if "Error" in hull_vec or "Composition Error" in hull_vec:
                    formation_energies.append(999.0)
                    hull_phase_combinations.append(["N/A"] * 4)
                else:
                    formation_energies.append(d_hull)
                    hull_phase_combinations.append(hull_vec)
            except Exception:
                formation_energies.append(None)
                hull_phase_combinations.append(None)

        valid = [(e, f, idx, spg, te, hp)
                 for e, f, idx, spg, te, hp in zip(formation_energies, formulas, indices, spgs, total_energies, hull_phase_combinations)
                 if e is not None and hp is not None]
        if not valid:
            return output_file

        valid.sort()

        prefix = os.path.dirname(output_file)
        selected_dirname = os.path.join(prefix, "selected")
        os.makedirs(selected_dirname, exist_ok=True)
        csv_filename = os.path.join(prefix, ''.join(eles_symbols) + '_quaternary.csv')

        with open(output_file, 'w+') as f_out, open(csv_filename, 'w+') as f_csv:
            f_out.write("# Index,Formula,Formation_Energy_per_atom(eV/atom),Spacegroup\n")
            f_csv.write("Formula,Total_Energy_per_atom,Ehull,Hull_Phase1,Hull_Phase2,Hull_Phase3,Hull_Phase4\n")
            count = 0
            for energy, formula, idx, spg, total_energy, phases in valid:
                count += 1
                f_out.write(f'{idx},{formula},{energy:.6f},{spg}\n')
                f_csv.write(f'{formula},{total_energy:.6f},{energy:.6f},{",".join(phases)}\n')
                if energy <= 1e-5 or count <= 20:
                    src = os.path.join(vasp_work_dir, f"{idx}/CONTCAR_{idx}")
                    dst = os.path.join(selected_dirname, f"CONTCAR_{idx}")
                    if os.path.exists(src):
                        shutil.copy(src, dst)

        return output_file


@python_app(executors=[POSTPROCESSING_LABEL])
def calculate_ehul(config):
    return cmd_calculate_ehul(config)
