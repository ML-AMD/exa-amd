from parsl import python_app
from parsl_configs.parsl_executors_labels import POSTPROCESSING_LABEL
from tools.config_labels import ConfigKeys as CK


def plot_convex_hull_ternary(elements_list, stable_dat, full_path_input_csv, threshold, output_file):
    import os
    import csv
    import numpy as np
    from pymatgen.core import Element, Composition
    from scipy.spatial import ConvexHull

    system = []  # system we want to get PD for
    ene = []

    # Read elemental energies from mp_element.dat
    def read_elemental_energies(filename):
        elemental_energies = {}
        with open(filename, 'r') as f:
            for line in f:
                element, energy = line.replace(',', ' ').split()
                try:
                    eles = Composition(element).elements
                    if len(eles) == 1:
                        elemental_energies[eles[0].symbol] = float(energy)
                except BaseException:
                    continue
        return elemental_energies

    def read_mp(file_in):
        processed_entries = []
        ef_large0 = []
        with open(file_in, "r") as fin:
            lines = fin.readlines()
            for line in lines:
                formula = line.split()[0]
                comp = Composition(formula)
                formreduce = comp.reduced_formula
                natom_1 = int(comp.element_composition.get(system[0]))
                natom_2 = int(comp.element_composition.get(system[1]))
                natom_3 = int(comp.element_composition.get(system[2]))
                et = float(line.split()[1])
                natom = natom_1 + natom_2 + natom_3
                ef = et - (natom_1 * ene[0] + natom_2 *
                           ene[1] + natom_3 * ene[2]) / natom
                my_entry = [natom_1, natom_2, natom_3, ef]
                processed_entries.append(my_entry)
        return processed_entries, ef_large0

    def read_all(file_in):
        processed_entries = []
        ef_large0 = []
        with open(file_in, "r") as fin:
            lines = csv.reader(fin)
            for line in lines:
                comp = Composition(line[0])

                natom_1 = int(comp.element_composition.get(system[0]))
                natom_2 = int(comp.element_composition.get(system[1]))
                natom_3 = int(comp.element_composition.get(system[2]))

                et = float(line[1])
                natom = natom_1 + natom_2 + natom_3
                ef = et - (natom_1 * ene[0] + natom_2 *
                           ene[1] + natom_3 * ene[2]) / natom
                my_entry = [natom_1, natom_2, natom_3, ef]
                if (ef > 0):
                    ef_large0.append(my_entry)
                    continue
                elif ef < -1:
                    continue
                processed_entries.append(my_entry)
        return processed_entries, ef_large0

    # input: pts as list with element [nA,nB,nC,Ef]
    def area(a, b, c):
        import numpy as np
        from numpy.linalg import norm
        if (isinstance(a, list)):
            a = np.array(a, dtype=np.float32)
        if (isinstance(b, list)):
            b = np.array(b, dtype=np.float32)
        if (isinstance(c, list)):
            c = np.array(c, dtype=np.float32)
        return 0.5 * norm(np.cross(b - a, c - a))

    def get_plane(p1, p2, p3):
        import numpy as np
        if (isinstance(p1, list)):
            p1 = np.array(p1, dtype=np.float32)
        if (isinstance(p2, list)):
            p2 = np.array(p2, dtype=np.float32)
        if (isinstance(p3, list)):
            p3 = np.array(p3, dtype=np.float32)
        v1 = p3 - p1
        v2 = p2 - p1
        cp = np.cross(v1, v2)
        a, b, c = cp
        d = np.dot(cp, p3)
        return a, b, c, d

    def draw_ternary_convex(pts, pts_aga, pts_exp, pts_mp,
                            pts_l0, ele, string, hullmax=0.1, output_file=None):
        import matplotlib
        import ternary
        import numpy as np
        from scipy.spatial import ConvexHull
        matplotlib.rcParams['figure.dpi'] = 200
        matplotlib.rcParams['figure.figsize'] = (4, 4)

        # scales
        figure, tax = ternary.figure(scale=1.0)
        # boundary
        tax.boundary(linewidth=0.5)
        tax.gridlines(color="grey", multiple=0.1)

        fontsize = 12
        ter = ele[0] + "-" + ele[1] + "-" + ele[2]
        tax.right_corner_label(ele[0], fontsize=fontsize + 2)
        tax.top_corner_label(ele[1], fontsize=fontsize + 2)
        tax.left_corner_label(ele[2], fontsize=fontsize + 2)

        # convert data to trianle set
        pts = np.array(pts)
        pts_aga = np.array(pts_aga)
        pts_exp = np.array(pts_exp)
        pts_mp = np.array(pts_mp)
        pts_l0 = np.array(pts_l0)
        tpts = []
        mpts = []
        for ipt in pts:
            comp = np.array([int(ii) for ii in ipt[:3]])
            comp = comp / sum(comp)
            x = comp[0] + comp[1] / 2.
            y = comp[1] * np.sqrt(3) / 2
            tpts.append([x, y, float(ipt[3])])

        tpts_l0 = []
        for ipt in pts_l0:
            comp = np.array([int(ii) for ii in ipt[:3]])
            comp = comp / sum(comp)
            x = comp[0] + comp[1] / 2.
            y = comp[1] * np.sqrt(3) / 2
            tpts_l0.append([x, y, float(ipt[3])])

        comps = []
        ehulls = []

        hull = ConvexHull(tpts)
        fout = open("./convex-hull.dat", "w+")
        print("# of stable structures", len(hull.vertices), ":", end=" ", file=fout)
        fout.write("\n")
        print(*ele, "Ef(eV/atom)", end=" ", file=fout)
        fout.write("\n")
        # plot data
        pdata = []

        for pt in pts:
            mm = np.array([int(ii) for ii in pt[:3]])
            pdata.append(1.0 * mm / sum(mm))

        # 1 plot stable and connect them
        for isimp in hull.simplices:
            tax.line(pdata[isimp[0]], pdata[isimp[1]], linewidth=0.7,
                     marker='.', markersize=8., color='black')
            tax.line(pdata[isimp[0]], pdata[isimp[2]], linewidth=0.7,
                     marker='.', markersize=8., color='black')
            tax.line(pdata[isimp[1]], pdata[isimp[2]], linewidth=0.7,
                     marker='.', markersize=8., color='black')

        stables = []
        stables_extra = []
        for iv in hull.vertices:
            name = ele[0] + str(int(pts[iv][0])) + ele[1] + \
                str(int(pts[iv][1])) + ele[2] + str(int(pts[iv][2]))
            aaa = pts[iv]
            # still not sure how to plot names on the figure 06/24
            stables.append([tpts[iv][0], tpts[iv][1], name])
            name = ele[0] + str(int(pts[iv][0])) + ele[1] + \
                str(int(pts[iv][1])) + ele[2] + str(int(pts[iv][2]))

            matches_first_three = np.all(np.isclose(
                pts_exp[:, :3], aaa[:3], atol=1e-4), axis=1)
            matches_fourth = np.isclose(pts_exp[:, 3], aaa[3], atol=1e-1)
            # if not np.any(np.all(np.isclose(pts_exp, aaa, atol=1e-1), axis=1)):
            if not np.any(matches_first_three & matches_fourth):
                tax.scatter([pdata[iv]], marker='.', s=64., color='red', zorder=10)
                comps.append(iv)
                ehulls.append(0)
            else:
                formula = Composition(name).reduced_formula
                fout.write(formula + '\n')

        # 2 get meta-stable phases
        mstables = []
        for i in range(len(pdata)):
            if (i not in hull.vertices):
                mstables.append(pdata[i])

        aga_meta_stables = []
        exp_meta_stables = []
        mp_meta_stables = []
        l0_meta_stables = []
        # 4 find the distance to the convex hull
        print("# of metastable structures", len(mstables), ":", end=" ", file=fout)
        fout.write("\n")
        print(*ele, "Ef(eV/atom) E_to_convex_hull(eV/atom)", end=" ", file=fout)
        fout.write("\n")
        #  4.1 get nearest 3 points
        for k in range(len(tpts)):
            if (k in hull.vertices):
                h = 0
                # continue # jump the stable ones
            else:
                x = tpts[k][:2]  # metastable, as [x,y,Ef]
                for isimp in hull.simplices:  # loop the simplices
                    A = tpts[isimp[0]][:2]
                    B = tpts[isimp[1]][:2]
                    C = tpts[isimp[2]][:2]
                    # find if x in the A-B-C triangle
                    area_ABC = area(A, B, C)
                    sum_a = area(A, B, x) + area(A, C, x) + area(B, C, x)
                    if (sum_a - area_ABC <= 0.001):
                        # in the ABC, get the ABC plane
                        a, b, c, d = get_plane(
                            tpts[isimp[0]], tpts[isimp[1]], tpts[isimp[2]])
                        if (a == 0 and b == 0 and d == 0):
                            continue
                        if (c == 0):
                            continue
                        # get the cross point with ABC plane
                        z = (d - a * x[0] - b * x[1]) / c
                        # height to convex hull
                        h = tpts[k][2] - z

            name = ele[0] + str(int(pts[k][0])) + ele[1] + \
                str(int(pts[k][1])) + ele[2] + str(int(pts[k][2]))
            formula = Composition(name).reduced_formula
            comps.append(k)
            ehulls.append(h)
        # judge the label for aga, exp and mp
        label = "000"
        for ss in range(len(pts_aga)):
            if (pts[k][-1] == pts_aga[ss][-1]):
                label = "aga"
                aga_meta_stables.append([float(pts[k][0]), float(
                    pts[k][1]), float(pts[k][2]), pts[k][3], h])
                break
        for ss in range(len(pts_exp)):
            if (pts[k][-1] == pts_exp[ss][-1]):
                label = "exp"
                exp_meta_stables.append([float(pts[k][0]), float(
                    pts[k][1]), float(pts[k][2]), pts[k][3], h])
        for ss in range(len(pts_mp)):
            if (pts[k][-1] == pts_mp[ss][-1]):
                label = "mp"
                mp_meta_stables.append([float(pts[k][0]), float(
                    pts[k][1]), float(pts[k][2]), pts[k][3], h])

        for ka in range(len(tpts_l0)):
            x = tpts_l0[ka][:2]  # metastable, as [x,y,Ef]
            for isimp in hull.simplices:  # loop the simplices
                A = tpts[isimp[0]][:2]
                B = tpts[isimp[1]][:2]
                C = tpts[isimp[2]][:2]
                # find if x in the A-B-C triangle
                area_ABC = area(A, B, C)
                sum_a = area(A, B, x) + area(A, C, x) + area(B, C, x)
                if (sum_a - area_ABC <= 0.001):
                    # in the ABC, get the ABC plane
                    a, b, c, d = get_plane(
                        tpts[isimp[0]], tpts[isimp[1]], tpts[isimp[2]])
                    if (a == 0 and b == 0 and d == 0):
                        continue
                    if (c == 0):
                        continue
                    # get the cross point with ABC plane
                    z = (d - a * x[0] - b * x[1]) / c
                    # height to convex hull
                    h = tpts_l0[ka][2] - z
                    l0_meta_stables.append([float(pts_l0[ka][0]), float(pts_l0[ka][1]), float(pts_l0[ka][2]),
                                            pts_l0[ka][3], h])

        all_meta_stables = [
            aga_meta_stables,
            exp_meta_stables,
            mp_meta_stables,
            l0_meta_stables]
        if hullmax > 0:
            ehulls, comps = zip(*sorted(zip(ehulls, comps)))
            for eh, kkk in zip(ehulls, comps):
                aaa = pts[kkk]
                name = ele[0] + str(int(aaa[0])) + ele[1] + \
                    str(int(aaa[1])) + ele[2] + str(int(aaa[2]))
                formula = Composition(name).reduced_formula

                fout.write(formula + '   ' + str(eh * 1000) + '\n')

        fout.close()

        marker_vec = [6, 7, ".", "."]
        s_vec = [50, 50, 50, 50]
        all_color_data = []
        all_meta_data = []
        # cm = ternary.plt.cm.get_cmap('tab20c')
        cm = ternary.plt.cm.get_cmap('rainbow')

        for ms in range(len(all_meta_stables)):
            # all_meta_stables[ms]=np.array(all_meta_stables[ms])
            if (len(all_meta_stables[ms]) != 0):
                meta_data = []
                color_data = []
                for mpt in all_meta_stables[ms]:
                    if mpt[-1] < hullmax:
                        mm = np.array([float(ii) for ii in mpt[:3]])
                        point_t = 1.0 * mm / sum(mm)
                        meta_data.append(point_t)
                        all_meta_data.append(point_t)
                        color_data.append(mpt[-1])
                        all_color_data.append(mpt[-1])

        if hullmax > 0:
            tax.scatter(
                all_meta_data,
                s=7,
                marker='s',
                colormap=cm.reversed(),
                vmin=0,
                vmax=hullmax,
                colorbar=False,
                c=all_color_data,
                cmap=cm.reversed())

        # remove matplotlib axes
        tax.clear_matplotlib_ticks()
        tax.get_axes().axis('off')

        ternary.plt.show()

        if output_file:
            ternary.plt.savefig(output_file, dpi=300)
        else:
            ternary.plt.show()

    elements = [Element(ele) for ele in elements_list]
    eles = [ele.symbol for ele in elements]
    elename = ''.join(eles)
    system.append(eles[2])
    system.append(eles[0])
    system.append(eles[1])

    ef_file = stable_dat
    elemental_energies = read_elemental_energies(ef_file)
    ene = [float(elemental_energies[i]) for i in system]

    mp_file = full_path_input_csv
    ef_l0 = []
    pre_xyze, ef_l0 = read_mp(ef_file)
    mp_xyze, ef_l0 = read_all(mp_file)

    if threshold <= 0:
        mp_xyze = []
    all_xyze = mp_xyze

    for j in range(len(pre_xyze)):
        all_xyze.append(pre_xyze[j])

    aga_xyze = []
    exp_xyze = []

    draw_ternary_convex(
        all_xyze,
        aga_xyze,
        pre_xyze,
        mp_xyze,
        ef_l0,
        system,
        elename,
        threshold,
        output_file)

    return output_file


def plot_convex_hull_quaternary(elements_str, stable_path, input_csv_path, ehull_threshold, output_file=None):
    import argparse
    import os
    import re
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib.colors as mcolors
    from pymatgen.core import Composition, Element
    from scipy.spatial import ConvexHull

    # --- Tetrahedral Projection ---
    # Define standard coordinates for the 4 vertices (representing pure elements)
    # A at (0, 0, 0), B at (1, 0, 0), C at (0.5, sqrt(3)/2, 0), D at (0.5,
    # sqrt(3)/6, sqrt(6)/3)
    TETRA_CORNERS = {
        0: np.array([0.0, 0.0, 0.0]),                   # Element A (index 0)
        1: np.array([1.0, 0.0, 0.0]),                   # Element B (index 1)
        2: np.array([0.5, np.sqrt(3) / 2, 0.0]),          # Element C (index 2)
        3: np.array([0.5, np.sqrt(3) / 6, np.sqrt(6) / 3]),  # Element D (index 3)
    }

    def composition_to_tetrahedral_coords(comp, element_map):
        """
        Converts a pymatgen Composition object to 3D tetrahedral coordinates.

        Args:
            comp (Composition): The composition to convert.
            element_map (dict): A dictionary mapping Element objects to their
                                vertex index (0, 1, 2, 3).

        Returns:
            np.ndarray: The 3D coordinates (x, y, z), or None if composition
                        doesn't match the 4 elements.
        """
        coords = np.zeros(3)
        total_fraction = 0.0
        try:
            # Calculate weighted average of corner coordinates based on atomic fractions
            # We skip the element at index 0 as it's implicitly represented
            # (origin)
            for element, index in element_map.items():
                fraction = comp.get_atomic_fraction(element)
                if index != 0:  # Don't add contribution from the origin element explicitly
                    coords += fraction * TETRA_CORNERS[index]
                total_fraction += fraction

            # Basic check if the composition belongs to the system
            if not np.isclose(total_fraction, 1.0):
                # This might happen if comp contains elements outside the map
                # Or if it's an empty composition
                # print(f"Warning: Composition {comp.reduced_formula} fractions don't sum to 1 for the system. Skipping.")
                return None
            # Check if all elements in the comp are in our system
            if not all(el in element_map for el in comp.elements):
                return None

            return coords
        except Exception as e:
            print(f"Error converting composition {comp.reduced_formula}: {e}")
            return None

    def parse_stable_phases(filename, element_map):
        """
        Parses the stable phases file (e.g., mp_int_stable.dat).

        Args:
            filename (str): Path to the stable phases file.
            element_map (dict): Dictionary mapping Element objects to vertex indices.

        Returns:
            list: A list of tuples, where each tuple is
                  (formula, energy_per_atom, np.ndarray_coords).
                  Returns only phases belonging to the A-B-C-D system.
        """
        stable_phases = []
        elements_in_system = set(element_map.keys())
        print(f"Parsing stable phases from: {filename}")
        try:
            with open(filename, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    formula = parts[0]
                    try:
                        energy = float(parts[-1])  # Assume energy is the last part
                        comp = Composition(formula)

                        # Check if the composition's elements are a subset of our
                        # system
                        if set(comp.elements).issubset(elements_in_system):
                            coords = composition_to_tetrahedral_coords(
                                comp, element_map)
                            if coords is not None:
                                stable_phases.append((formula, energy, coords))
                    # else:
                        # print(f"  Skipping stable phase {formula} (elements outside system).")

                    except (ValueError, TypeError) as e:
                        print(
                            f"  Warning: Could not parse line: '{line}'. Error: {e}")
                    except Exception as e:
                        print(
                            f"  Warning: Could not process composition {formula}: {e}")

        except FileNotFoundError:
            print(f"Error: Stable phases file '{filename}' not found.")
            return []
        except Exception as e:
            print(f"An error occurred reading {filename}: {e}")
            return []

        print(f"Found {len(stable_phases)} stable phases within the specified element system.")
        return stable_phases

    def parse_results_csv(filename, element_map):
        """
        Parses the results CSV file (e.g., *_quaternary.csv).
        Assumes columns: Formula,Total_Energy_per_atom,Ehull,...

        Args:
            filename (str): Path to the results CSV file.
            element_map (dict): Dictionary mapping Element objects to vertex indices.

        Returns:
            list: A list of tuples, where each tuple is
                  (formula, ehull, np.ndarray_coords).
        """
        results = []
        print(f"Parsing calculated results from: {filename}")
        try:
            with open(filename, 'r') as f:
                header = f.readline().strip().lower()  # Read header
                if not header.startswith('formula'):
                    print(
                        "Warning: CSV file does not seem to have the expected header (Formula,...). Trying to parse anyway.")

                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split(',')
                    if len(parts) < 3:  # Need at least Formula, Total_Energy, Ehull
                        print(f"  Warning: Skipping malformed line: '{line}'")
                        continue
                    formula = parts[0]
                    try:
                        # Ehull is expected to be the 3rd column (index 2)
                        ehull = float(parts[2])
                        comp = Composition(formula)
                        coords = composition_to_tetrahedral_coords(
                            comp, element_map)
                        if coords is not None:
                            results.append((formula, ehull, coords))

                    except (ValueError, TypeError) as e:
                        print(f"  Warning: Could not parse Ehull or composition for line: '{line}'. Error: {e}")
                    except Exception as e:
                        print(f"  Warning: Could not process composition {formula} from results: {e}")

        except FileNotFoundError:
            print(f"Error: Results file '{filename}' not found.")
            return []
        except Exception as e:
            print(f"An error occurred reading {filename}: {e}")
            return []

        print(f"Found {len(results)} calculated results within the specified element system.")
        return results

    def plot_quaternary_hull(elements_str, stable_phases,
                             calculated_results, ehull_threshold, output_file=None):
        """
        Generates the 3D plot of the quaternary convex hull.

        Args:
            elements_str (list): List of 4 element symbols (e.g., ['Si','Ge','Sn','Pb']).
            stable_phases (list): List of (formula, energy, coords) for stable phases.
            calculated_results (list): List of (formula, ehull, coords) for calculated phases.
            ehull_threshold (float): Max Ehull value to plot for calculated phases.
            output_file (str, optional): Path to save the plot image. If None, displays plot.
        """
        if len(elements_str) != 4:
            raise ValueError("Exactly 4 element symbols are required.")

        elements = [Element(el) for el in elements_str]
        element_map = {el: i for i, el in enumerate(elements)}

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        print("Plotting tetrahedron edges...")
        corners_3d = list(TETRA_CORNERS.values())
        edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        for i, j in edges:
            ax.plot([corners_3d[i][0], corners_3d[j][0]],
                    [corners_3d[i][1], corners_3d[j][1]],
                    [corners_3d[i][2], corners_3d[j][2]], 'k-', lw=1.0, alpha=0.6)

        # Label corners
        corner_labels = elements_str
        for i, label in enumerate(corner_labels):
            ax.text(corners_3d[i][0] * 1.05, corners_3d[i][1] * 1.05, corners_3d[i][2] * 1.05, label,
                    fontsize=15, ha='center', va='center')

        print("Computing and plotting convex hull facets...")
        stable_coords = np.array([p[2] for p in stable_phases if p[2] is not None])

        if len(stable_coords) >= 4:  # Need at least 4 points for a 3D hull
            try:
                hull = ConvexHull(stable_coords)
                # Plot the triangular faces of the hull
                for simplex in hull.simplices:
                    triangle = stable_coords[simplex]
                    face = Poly3DCollection(
                        [triangle],
                        alpha=0.2,
                        facecolor='lightblue',
                        edgecolor='grey',
                        lw=0.5)
                    ax.add_collection3d(face)
                print(
                    f"  Successfully computed and plotted hull with {len(hull.simplices)} facets.")
            except Exception as e:
                print(
                    f"  Warning: Could not compute or plot convex hull: {e}. Only plotting points.")
        else:
            print("  Warning: Not enough stable points (need >= 4) to compute 3D convex hull.")

        print("Plotting stable phase points...")
        if stable_coords.any():  # Check if there are any stable coordinates to plot
            ax.scatter(stable_coords[:, 0], stable_coords[:, 1], stable_coords[:, 2],
                       c='black', marker='o', s=60, label='Stable Phases (Input)', depthshade=False, alpha=0.8)
        else:
            print("  No stable phase coordinates found to plot.")

        print(
            f"Plotting calculated results with Ehull <= {ehull_threshold} eV/atom...")
        calculated_coords = []
        calculated_ehull = []
        calculated_labels = []

        for formula, ehull, coords in calculated_results:
            if coords is not None and ehull <= ehull_threshold:
                calculated_coords.append(coords)
                calculated_ehull.append(ehull)
                calculated_labels.append(formula)

        if calculated_coords:
            calculated_coords = np.array(calculated_coords)
            calculated_ehull = np.array(calculated_ehull)

            # Normalize Ehull values for colormap
            norm = mcolors.Normalize(vmin=0, vmax=ehull_threshold)
            # Reversed viridis: blue (low Ehull) to yellow (high Ehull)
            cmap = plt.cm.rainbow_r

            sc = ax.scatter(calculated_coords[:, 0], calculated_coords[:, 1], calculated_coords[:, 2],
                            c=calculated_ehull, cmap=cmap, norm=norm,
                            marker='^', s=40, label=f'Calculated (Ehull <= {ehull_threshold:.3f})',
                            depthshade=True, alpha=0.9)  # Use depthshade for better 3D perception

            # Add Colorbar
            cbar = fig.colorbar(sc, shrink=0.6, aspect=20, pad=0.1)
            cbar.set_label('Formation Energy above Hull (eV/atom)')
            print(f"  Plotted {len(calculated_coords)} calculated points.")
        else:
            print("  No calculated results found within the Ehull threshold.")

        ax.set_xlabel('Composition Space X')
        ax.set_ylabel('Composition Space Y')
        ax.set_zlabel('Composition Space Z')

        # Remove axis ticks/grid for cleaner compositional space view
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.grid(False)
        plt.axis('off')  # Turn off the axis frame

        # Adjust view angle (elevation, azimuth)
        ax.view_init(elev=20, azim=30)
        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=300)
            print(f"Plot saved to {output_file}")
        else:
            plt.show()

    element_symbols = elements_str
    element_map_main = {Element(el): i for i, el in enumerate(element_symbols)}
    results_data = parse_results_csv(input_csv_path, element_map_main)
    stable_data = parse_stable_phases(stable_path, element_map_main)
    if not stable_data and not results_data:
        print("\nError: No data could be parsed from input files for the specified elements. Cannot generate plot.")
        return output_file
    plot_quaternary_hull(
        element_symbols,
        stable_data,
        results_data,
        ehull_threshold,
        output_file)
    return output_file


@python_app(executors=[POSTPROCESSING_LABEL])
def convex_hull_color(config):
    try:
        import os
        elements = config[CK.ELEMENTS]
        l_elements = elements.split('-')
        nb_of_elements = len(l_elements)
        stable_dat = os.path.join(config[CK.POST_PROCESSING_OUT_DIR], CK.MP_STABLE_OUT)
        elename = ''.join(l_elements)
        input_csv = elename + '.csv' if nb_of_elements == 3 else elename + '_quaternary.csv'
        full_path_input_csv = os.path.join(config[CK.POST_PROCESSING_OUT_DIR], input_csv)
        output_file = os.path.join(config[CK.POST_PROCESSING_OUT_DIR], CK.POST_PROCESSING_FINAL_OUT)
        threshold = float(config[CK.HULL_ENERGY_THR])
        if nb_of_elements == 3:
            plot_convex_hull_ternary(l_elements, stable_dat, full_path_input_csv, threshold, output_file)
        else:
            plot_convex_hull_quaternary(l_elements, stable_dat, full_path_input_csv, threshold, output_file)
    except Exception as e:
        raise
