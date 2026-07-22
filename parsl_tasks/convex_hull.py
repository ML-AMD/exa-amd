"""Convex hull plotting tasks for ternary and quaternary chemical systems.

This module provides Parsl-compatible functions to compute and visualize
formation-energy convex hulls. It supports two cases:

* Ternary (3-element) systems, rendered as a 2D ternary diagram with
  metastable points colored by their distance above the hull.
* Quaternary (4-element) systems, rendered as a 3D tetrahedral projection.

The public entry point :func:`convex_hull_color` is a Parsl ``python_app``
that reads a configuration mapping and dispatches to either
:func:`plot_convex_hull_ternary` or :func:`plot_convex_hull_quaternary`.
"""

from typing import Any

from parsl import python_app

from parsl_configs.parsl_executors_labels import POSTPROCESSING_LABEL
from tools.config_labels import ConfigKeys as CK


def plot_convex_hull_ternary(
    elements_list: list[str],
    stable_dat: str,
    full_path_input_csv: str,
    threshold: float,
    output_file: str,
) -> str:
    """Plot the ternary convex hull and metastable points for a 3-element system.

    Parameters
    ----------
    elements_list : list of str
        Three element symbols (e.g., ``["Ce", "Co", "B"]``).
    stable_dat : str
        Path to a text file with elemental reference energies.
    full_path_input_csv : str
        CSV with rows ``Formula,Total_Energy_per_atom`` for calculated phases.
    threshold : float
        Max Ehull (eV/atom) to display for metastable points
        (``<= 0`` hides them).
    output_file : str
        Path for the saved image.

    Returns
    -------
    str
        The ``output_file`` path (for convenience).
    """
    import csv

    import numpy as np
    from pymatgen.core import Composition, Element
    from scipy.spatial import ConvexHull

    system: list[str] = []  # system we want to get PD for
    ene: list[float] = []

    # Read elemental energies from mp_element.dat
    def read_elemental_energies(filename: str) -> dict[str, float]:
        """Read single-element reference energies from a data file.

        Parameters
        ----------
        filename : str
            Path to the file containing ``element energy`` pairs.

        Returns
        -------
        dict
            Mapping of element symbol to its reference energy (float).
        """
        elemental_energies = {}
        with open(filename, "r") as f:
            for line in f:
                element, energy = line.replace(",", " ").split()
                try:
                    eles = Composition(element).elements
                    if len(eles) == 1:
                        elemental_energies[eles[0].symbol] = float(energy)
                except BaseException:
                    continue
        return elemental_energies

    def read_mp(file_in: str) -> tuple[list[list[float]], list[list[float]]]:
        """Read Materials Project phases and compute formation energies.

        Parameters
        ----------
        file_in : str
            Path to a whitespace-separated file with ``Formula Energy`` rows.

        Returns
        -------
        processed_entries : list of list
            Entries of the form ``[nA, nB, nC, Ef]``.
        ef_large0 : list of list
            Entries with positive formation energy (always empty here).
        """
        processed_entries: list[list[float]] = []
        ef_large0: list[list[float]] = []
        with open(file_in, "r") as fin:
            lines = fin.readlines()
            for line in lines:
                formula = line.split()[0]
                comp = Composition(formula)
                natom_1 = int(comp.element_composition.get(system[0]) or 0)
                natom_2 = int(comp.element_composition.get(system[1]) or 0)
                natom_3 = int(comp.element_composition.get(system[2]) or 0)
                et = float(line.split()[1])
                natom = natom_1 + natom_2 + natom_3
                ef = (
                    et
                    - (natom_1 * ene[0] + natom_2 * ene[1] + natom_3 * ene[2]) / natom
                )
                my_entry = [natom_1, natom_2, natom_3, ef]
                processed_entries.append(my_entry)
        return processed_entries, ef_large0

    def read_all(file_in: str) -> tuple[list[list[float]], list[list[float]]]:
        """Read all calculated phases from a CSV and compute formation energies.

        Parameters
        ----------
        file_in : str
            Path to a CSV with ``Formula,Total_Energy_per_atom`` rows.

        Returns
        -------
        processed_entries : list of list
            Entries ``[nA, nB, nC, Ef]`` with ``-1 <= Ef <= 0``.
        ef_large0 : list of list
            Entries with positive formation energy.
        """
        processed_entries: list[list[float]] = []
        ef_large0: list[list[float]] = []
        with open(file_in, "r") as fin:
            lines = csv.reader(fin)
            for line in lines:
                comp = Composition(line[0])

                natom_1 = int(comp.element_composition.get(system[0]) or 0)
                natom_2 = int(comp.element_composition.get(system[1]) or 0)
                natom_3 = int(comp.element_composition.get(system[2]) or 0)

                et = float(line[1])
                natom = natom_1 + natom_2 + natom_3
                ef = (
                    et
                    - (natom_1 * ene[0] + natom_2 * ene[1] + natom_3 * ene[2]) / natom
                )
                my_entry = [natom_1, natom_2, natom_3, ef]
                if ef > 0:
                    ef_large0.append(my_entry)
                    continue
                elif ef < -1:
                    continue
                processed_entries.append(my_entry)
        return processed_entries, ef_large0

    # input: pts as list with element [nA,nB,nC,Ef]
    def area(
        a: list | np.ndarray,
        b: list | np.ndarray,
        c: list | np.ndarray,
    ) -> float:
        """Compute the area of the triangle defined by three 2D points.

        Parameters
        ----------
        a, b, c : list or numpy.ndarray
            2D coordinates of the triangle vertices.

        Returns
        -------
        float
            Area of the triangle.
        """
        from numpy.linalg import norm

        if isinstance(a, list):
            a = np.array(a, dtype=np.float32)
        if isinstance(b, list):
            b = np.array(b, dtype=np.float32)
        if isinstance(c, list):
            c = np.array(c, dtype=np.float32)
        # ensure 3D vectors for np.cross (NumPy 2.0)
        ab = np.append(b - a, 0.0)
        ac = np.append(c - a, 0.0)
        return 0.5 * norm(np.cross(ab, ac))

    def get_plane(
        p1: list | np.ndarray,
        p2: list | np.ndarray,
        p3: list | np.ndarray,
    ) -> tuple[float, float, float, float]:
        """Compute the plane equation coefficients through three 3D points.

        Parameters
        ----------
        p1, p2, p3 : list or numpy.ndarray
            3D coordinates defining the plane.

        Returns
        -------
        a, b, c, d : float
            Coefficients of the plane equation ``a*x + b*y + c*z = d``.
        """
        import numpy as np

        if isinstance(p1, list):
            p1 = np.array(p1, dtype=np.float32)
        if isinstance(p2, list):
            p2 = np.array(p2, dtype=np.float32)
        if isinstance(p3, list):
            p3 = np.array(p3, dtype=np.float32)
        v1 = p3 - p1
        v2 = p2 - p1
        cp = np.cross(v1, v2)
        a, b, c = cp
        d = np.dot(cp, p3)
        return a, b, c, d

    def draw_ternary_convex(
        pts: list[list[float]],
        pts_aga: list[list[float]],
        pts_exp: list[list[float]],
        pts_mp: list[list[float]],
        pts_l0: list[list[float]],
        ele: list[str],
        string: str,
        hullmax: float = 0.1,
        output_file: str | None = None,
    ) -> None:
        """Draw the ternary convex hull and scatter metastable phases.

        Parameters
        ----------
        pts : list of list
            All phase entries ``[nA, nB, nC, Ef]`` used to build the hull.
        pts_aga : list of list
            AGA phase entries used to classify metastable points.
        pts_exp : list of list
            Experimental/stable phase entries.
        pts_mp : list of list
            Materials Project phase entries.
        pts_l0 : list of list
            Phase entries with positive formation energy.
        ele : list of str
            Element symbols used for labeling.
        string : str
            System name (currently unused in the plot).
        hullmax : float, optional
            Max Ehull (eV/atom) to display for metastable points, by default
            ``0.1``.
        output_file : str, optional
            Path to save the figure. If ``None``, the plot is shown.

        Returns
        -------
        None
        """
        import matplotlib
        import numpy as np
        import ternary  # type: ignore[import-untyped]

        matplotlib.rcParams["figure.dpi"] = 200
        matplotlib.rcParams["figure.figsize"] = (4, 4)

        # scales
        figure, tax = ternary.figure(scale=1.0)
        # boundary
        tax.boundary(linewidth=0.5)
        tax.gridlines(color="grey", multiple=0.1)

        fontsize = 12
        tax.right_corner_label(ele[0], fontsize=fontsize + 2)
        tax.top_corner_label(ele[1], fontsize=fontsize + 2)
        tax.left_corner_label(ele[2], fontsize=fontsize + 2)

        # convert data to trianle set
        pts_arr = np.array(pts)
        pts_aga_arr = np.array(pts_aga)
        pts_exp_arr = np.array(pts_exp)
        pts_mp_arr = np.array(pts_mp)
        pts_l0_arr = np.array(pts_l0)
        tpts = []
        for ipt in pts_arr:
            comp = np.array([int(ii) for ii in ipt[:3]])
            comp = comp / sum(comp)
            x = comp[0] + comp[1] / 2.0
            y = comp[1] * np.sqrt(3) / 2
            tpts.append([x, y, float(ipt[3])])

        tpts_l0: list[list[float]] = []
        for ipt in pts_l0_arr:
            comp = np.array([int(ii) for ii in ipt[:3]])
            comp = comp / sum(comp)
            x = comp[0] + comp[1] / 2.0
            y = comp[1] * np.sqrt(3) / 2
            tpts_l0.append([x, y, float(ipt[3])])

        comps: list[int] = []
        ehulls: list[float] = []

        hull = ConvexHull(tpts)
        fout = open("./convex-hull.dat", "w+")
        print("# of stable structures", len(hull.vertices), ":", end=" ", file=fout)
        fout.write("\n")
        print(*ele, "Ef(eV/atom)", end=" ", file=fout)
        fout.write("\n")
        # plot data
        pdata: list = []

        for pt in pts_arr:
            mm = np.array([int(ii) for ii in pt[:3]])
            pdata.append(1.0 * mm / sum(mm))

        # 1 plot stable and connect them
        for isimp in hull.simplices:
            tax.line(
                pdata[isimp[0]],
                pdata[isimp[1]],
                linewidth=0.7,
                marker=".",
                markersize=8.0,
                color="black",
            )
            tax.line(
                pdata[isimp[0]],
                pdata[isimp[2]],
                linewidth=0.7,
                marker=".",
                markersize=8.0,
                color="black",
            )
            tax.line(
                pdata[isimp[1]],
                pdata[isimp[2]],
                linewidth=0.7,
                marker=".",
                markersize=8.0,
                color="black",
            )

        stables: list[list] = []
        for iv in hull.vertices:
            name = (
                ele[0]
                + str(int(pts_arr[iv][0]))
                + ele[1]
                + str(int(pts_arr[iv][1]))
                + ele[2]
                + str(int(pts_arr[iv][2]))
            )
            aaa = pts_arr[iv]
            # still not sure how to plot names on the figure 06/24
            stables.append([tpts[iv][0], tpts[iv][1], name])
            name = (
                ele[0]
                + str(int(pts_arr[iv][0]))
                + ele[1]
                + str(int(pts_arr[iv][1]))
                + ele[2]
                + str(int(pts_arr[iv][2]))
            )

            matches_first_three = np.all(
                np.isclose(pts_exp_arr[:, :3], aaa[:3], atol=1e-4), axis=1
            )
            matches_fourth = np.isclose(pts_exp_arr[:, 3], aaa[3], atol=1e-1)
            if not np.any(matches_first_three & matches_fourth):
                tax.scatter([pdata[iv]], marker=".", s=64.0, color="red", zorder=10)
                comps.append(int(iv))
                ehulls.append(0)
            else:
                formula = Composition(name).reduced_formula
                fout.write(formula + "\n")

        # 2 get meta-stable phases
        mstables: list = []
        for i in range(len(pdata)):
            if i not in hull.vertices:
                mstables.append(pdata[i])

        aga_meta_stables: list[list[float]] = []
        exp_meta_stables: list[list[float]] = []
        mp_meta_stables: list[list[float]] = []
        l0_meta_stables: list[list[float]] = []
        # 4 find the distance to the convex hull
        print("# of metastable structures", len(mstables), ":", end=" ", file=fout)
        fout.write("\n")
        print(*ele, "Ef(eV/atom) E_to_convex_hull(eV/atom)", end=" ", file=fout)
        fout.write("\n")
        #  4.1 get nearest 3 points
        for k in range(len(tpts)):
            if k in hull.vertices:
                h: float = 0
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
                    if sum_a - area_ABC <= 0.001:
                        # in the ABC, get the ABC plane
                        a, b, c, d = get_plane(
                            tpts[isimp[0]], tpts[isimp[1]], tpts[isimp[2]]
                        )
                        if a == 0 and b == 0 and d == 0:
                            continue
                        if c == 0:
                            continue
                        # get the cross point with ABC plane
                        z = (d - a * x[0] - b * x[1]) / c
                        # height to convex hull
                        h = tpts[k][2] - z

            name = (
                ele[0]
                + str(int(pts_arr[k][0]))
                + ele[1]
                + str(int(pts_arr[k][1]))
                + ele[2]
                + str(int(pts_arr[k][2]))
            )
            formula = Composition(name).reduced_formula
            comps.append(int(k))
            ehulls.append(h)
            # judge the label for aga, exp and mp
            for ss in range(len(pts_aga_arr)):
                if pts_arr[k][-1] == pts_aga_arr[ss][-1]:
                    aga_meta_stables.append(
                        [
                            float(pts_arr[k][0]),
                            float(pts_arr[k][1]),
                            float(pts_arr[k][2]),
                            float(pts_arr[k][3]),
                            h,
                        ]
                    )
                    break
            for ss in range(len(pts_exp_arr)):
                if pts_arr[k][-1] == pts_exp_arr[ss][-1]:
                    exp_meta_stables.append(
                        [
                            float(pts_arr[k][0]),
                            float(pts_arr[k][1]),
                            float(pts_arr[k][2]),
                            float(pts_arr[k][3]),
                            h,
                        ]
                    )
            for ss in range(len(pts_mp_arr)):
                if pts_arr[k][-1] == pts_mp_arr[ss][-1]:
                    mp_meta_stables.append(
                        [
                            float(pts_arr[k][0]),
                            float(pts_arr[k][1]),
                            float(pts_arr[k][2]),
                            float(pts_arr[k][3]),
                            h,
                        ]
                    )

        for ka in range(len(tpts_l0)):
            x = tpts_l0[ka][:2]  # metastable, as [x,y,Ef]
            for isimp in hull.simplices:  # loop the simplices
                A = tpts[isimp[0]][:2]
                B = tpts[isimp[1]][:2]
                C = tpts[isimp[2]][:2]
                # find if x in the A-B-C triangle
                area_ABC = area(A, B, C)
                sum_a = area(A, B, x) + area(A, C, x) + area(B, C, x)
                if sum_a - area_ABC <= 0.001:
                    # in the ABC, get the ABC plane
                    a, b, c, d = get_plane(
                        tpts[isimp[0]], tpts[isimp[1]], tpts[isimp[2]]
                    )
                    if a == 0 and b == 0 and d == 0:
                        continue
                    if c == 0:
                        continue
                    # get the cross point with ABC plane
                    z = (d - a * x[0] - b * x[1]) / c
                    # height to convex hull
                    h = tpts_l0[ka][2] - z
                    l0_meta_stables.append(
                        [
                            float(pts_l0_arr[ka][0]),
                            float(pts_l0_arr[ka][1]),
                            float(pts_l0_arr[ka][2]),
                            float(pts_l0_arr[ka][3]),
                            h,
                        ]
                    )

        all_meta_stables = [
            aga_meta_stables,
            exp_meta_stables,
            mp_meta_stables,
            l0_meta_stables,
        ]
        if hullmax > 0:
            pairs = list(sorted(zip(ehulls, comps, strict=True)))
            if pairs:
                sorted_ehulls, sorted_comps = zip(
                    *sorted(zip(ehulls, comps, strict=True)), strict=True
                )
                for eh, kkk in zip(sorted_ehulls, sorted_comps, strict=True):
                    aaa = pts_arr[kkk]
                    name = (
                        ele[0]
                        + str(int(aaa[0]))
                        + ele[1]
                        + str(int(aaa[1]))
                        + ele[2]
                        + str(int(aaa[2]))
                    )
                    formula = Composition(name).reduced_formula

                    fout.write(formula + "   " + str(eh * 1000) + "\n")
        fout.close()

        all_color_data: list[float] = []
        all_meta_data: list = []
        cm = ternary.plt.colormaps["rainbow"]

        for ms in range(len(all_meta_stables)):
            if len(all_meta_stables[ms]) != 0:
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
                marker="s",
                colormap=cm.reversed(),
                vmin=0,
                vmax=hullmax,
                colorbar=False,
                c=all_color_data,
                cmap=cm.reversed(),
            )

        # remove matplotlib axes
        tax.clear_matplotlib_ticks()
        tax.get_axes().axis("off")
        if output_file:
            ternary.plt.savefig(output_file, dpi=300)
        else:
            ternary.plt.show()

    elements = [Element(ele) for ele in elements_list]
    eles = [ele.symbol for ele in elements]
    elename = "".join(eles)
    system.append(eles[2])
    system.append(eles[0])
    system.append(eles[1])

    ef_file = stable_dat
    elemental_energies = read_elemental_energies(ef_file)
    ene[:] = [float(elemental_energies[i]) for i in system]

    mp_file = full_path_input_csv
    ef_l0: list[list[float]] = []
    pre_xyze, ef_l0 = read_mp(ef_file)
    mp_xyze, ef_l0 = read_all(mp_file)

    if threshold <= 0:
        mp_xyze = []
    all_xyze = mp_xyze

    for j in range(len(pre_xyze)):
        all_xyze.append(pre_xyze[j])

    aga_xyze: list[list[float]] = []

    draw_ternary_convex(
        all_xyze,
        aga_xyze,
        pre_xyze,
        mp_xyze,
        ef_l0,
        system,
        elename,
        threshold,
        output_file,
    )

    return output_file


def plot_convex_hull_quaternary(
    elements_str: list[str],
    stable_path: str,
    input_csv_path: str,
    ehull_threshold: float,
    output_file: str | None = None,
) -> str | None:
    """Plot the quaternary convex hull and metastable points for a 4-element system.

    Parameters
    ----------
    elements_str : list of str
        List of 4 element symbols (e.g., ``['Si', 'Ge', 'Sn', 'Pb']``).
    stable_path : str
        Path to a text file with elemental reference energies.
    input_csv_path : str
        CSV with rows ``Formula,Total_Energy_per_atom`` for calculated phases.
    ehull_threshold : float
        Max Ehull (eV/atom) to display for metastable points
        (``<= 0`` hides them).
    output_file : str, optional
        Path for the saved image. If ``None``, the plot is shown.

    Returns
    -------
    str
        The ``output_file`` path (for convenience).
    """
    import matplotlib.colormaps as mcmaps  # type: ignore[import-not-found]
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # type: ignore[import-untyped]
    from pymatgen.core import Composition, Element
    from scipy.spatial import ConvexHull

    # --- Tetrahedral Projection ---
    # Define standard coordinates for the 4 vertices (representing pure elements)
    # A at (0, 0, 0), B at (1, 0, 0), C at (0.5, sqrt(3)/2, 0), D at (0.5,
    # sqrt(3)/6, sqrt(6)/3)
    TETRA_CORNERS = {
        0: np.array([0.0, 0.0, 0.0]),  # Element A (index 0)
        1: np.array([1.0, 0.0, 0.0]),  # Element B (index 1)
        2: np.array([0.5, np.sqrt(3) / 2, 0.0]),  # Element C (index 2)
        3: np.array([0.5, np.sqrt(3) / 6, np.sqrt(6) / 3]),  # Element D (index 3)
    }

    def composition_to_tetrahedral_coords(
        comp: "Composition", element_map: dict["Element", int]
    ) -> np.ndarray | None:
        """Convert a pymatgen Composition to 3D tetrahedral coordinates.

        Parameters
        ----------
        comp : pymatgen.core.Composition
            The composition to convert.
        element_map : dict
            Mapping of :class:`Element` objects to their vertex index
            (0, 1, 2, 3).

        Returns
        -------
        numpy.ndarray or None
            The 3D coordinates ``(x, y, z)``, or ``None`` if the composition
            does not match the 4 elements.
        """
        coords = np.zeros(3)
        total_fraction = 0.0
        try:
            # Calculate weighted average of corner coordinates based on atomic fractions
            # We skip the element at index 0 as it's implicitly represented
            # (origin)
            for element, index in element_map.items():
                fraction = comp.get_atomic_fraction(element)
                if (
                    index != 0
                ):  # Don't add contribution from the origin element explicitly
                    coords += fraction * TETRA_CORNERS[index]
                total_fraction += fraction

            # Basic check if the composition belongs to the system
            if not np.isclose(total_fraction, 1.0):
                # This might happen if comp contains elements outside the map
                # or if it's an empty composition print(f"Warning: Composition
                # {comp.reduced_formula} fractions don't sum to 1 for the
                # system. Skipping.")
                return None
            # Check if all elements in the comp are in our system
            if not all(el in element_map for el in comp.elements):
                return None

            return coords
        except Exception as e:
            print(f"Error converting composition {comp.reduced_formula}: {e}")
            return None

    def parse_stable_phases(
        filename: str, element_map: dict["Element", int]
    ) -> list[tuple[str, float, "np.ndarray"]]:
        """Parse the stable phases file (e.g., ``mp_int_stable.dat``).

        Parameters
        ----------
        filename : str
            Path to the stable phases file.
        element_map : dict
            Mapping of :class:`Element` objects to vertex indices.

        Returns
        -------
        list of tuple
            Tuples ``(formula, energy_per_atom, coords)`` for phases belonging
            to the A-B-C-D system.
        """
        stable_phases = []
        elements_in_system = set(element_map.keys())
        print(f"Parsing stable phases from: {filename}")
        try:
            with open(filename, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
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
                                comp, element_map
                            )
                            if coords is not None:
                                stable_phases.append((formula, energy, coords))

                    except (ValueError, TypeError) as e:
                        print(f"  Warning: Could not parse line: '{line}'. Error: {e}")
                    except Exception as e:
                        print(
                            f"  Warning: Could not process composition {formula}: {e}"
                        )

        except FileNotFoundError:
            print(f"Error: Stable phases file '{filename}' not found.")
            return []
        except Exception as e:
            print(f"An error occurred reading {filename}: {e}")
            return []

        print(
            f"Found {len(stable_phases)} stable phases within the specified"
            " element system."
        )
        return stable_phases

    def parse_results_csv(
        filename: str, element_map: dict[Element, int]
    ) -> list[tuple[str, float, np.ndarray]]:
        """Parse the results CSV file (e.g., ``*_quaternary.csv``).

        Assumes columns ``Formula,Total_Energy_per_atom,Ehull,...``.

        Parameters
        ----------
        filename : str
            Path to the results CSV file.
        element_map : dict
            Mapping of :class:`Element` objects to vertex indices.

        Returns
        -------
        list of tuple
            Tuples ``(formula, ehull, coords)``.
        """
        results = []
        print(f"Parsing calculated results from: {filename}")
        try:
            with open(filename, "r") as f:
                header = f.readline().strip().lower()  # Read header
                if not header.startswith("formula"):
                    print(
                        "Warning: CSV file does not seem to have the expected"
                        " header (Formula,...). Trying to parse anyway."
                    )

                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split(",")
                    if len(parts) < 3:  # Need at least Formula, Total_Energy, Ehull
                        print(f"  Warning: Skipping malformed line: '{line}'")
                        continue
                    formula = parts[0]
                    try:
                        # Ehull is expected to be the 3rd column (index 2)
                        ehull = float(parts[2])
                        comp = Composition(formula)
                        coords = composition_to_tetrahedral_coords(comp, element_map)
                        if coords is not None:
                            results.append((formula, ehull, coords))

                    except (ValueError, TypeError) as e:
                        print(
                            "  Warning: Could not parse Ehull or composition "
                            f"for line: '{line}'. Error: {e}"
                        )
                    except Exception as e:
                        print(
                            f"  Warning: Could not process composition {formula} "
                            f"from results: {e}"
                        )

        except FileNotFoundError:
            print(f"Error: Results file '{filename}' not found.")
            return []
        except Exception as e:
            print(f"An error occurred reading {filename}: {e}")
            return []

        print(
            f"Found {len(results)} calculated results within the specified "
            "element system."
        )
        return results

    def plot_quaternary_hull(
        elements_str: list[str],
        stable_phases: list[tuple[str, float, np.ndarray]],
        calculated_results: list[tuple[str, float, np.ndarray]],
        ehull_threshold: float,
        output_file: str | None = None,
    ) -> None:
        """Generate the 3D plot of the quaternary convex hull.

        Parameters
        ----------
        elements_str : list of str
            List of 4 element symbols (e.g., ``['Si', 'Ge', 'Sn', 'Pb']``).
        stable_phases : list of tuple
            Tuples ``(formula, energy, coords)`` for stable phases.
        calculated_results : list of tuple
            Tuples ``(formula, ehull, coords)`` for calculated phases.
        ehull_threshold : float
            Max Ehull value to plot for calculated phases.
        output_file : str, optional
            Path to save the plot image. If ``None``, the plot is shown.

        Raises
        ------
        ValueError
            If ``elements_str`` does not contain exactly 4 symbols.

        Returns
        -------
        None
        """
        if len(elements_str) != 4:
            raise ValueError("Exactly 4 element symbols are required.")

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")

        print("Plotting tetrahedron edges...")
        corners_3d = list(TETRA_CORNERS.values())
        edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        for i, j in edges:
            ax.plot(
                [corners_3d[i][0], corners_3d[j][0]],
                [corners_3d[i][1], corners_3d[j][1]],
                [corners_3d[i][2], corners_3d[j][2]],
                "k-",
                lw=1.0,
                alpha=0.6,
            )

        # Label corners
        corner_labels = elements_str
        for i, label in enumerate(corner_labels):
            ax.text(
                corners_3d[i][0] * 1.05,
                corners_3d[i][1] * 1.05,
                corners_3d[i][2] * 1.05,
                label,
                fontsize=15,
                ha="center",
                va="center",
            )

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
                        facecolor="lightblue",
                        edgecolor="grey",
                        lw=0.5,
                    )
                    ax.add_collection3d(face)
                print(
                    "  Successfully computed and plotted hull with "
                    f"{len(hull.simplices)} facets."
                )
            except Exception as e:
                print(
                    f"  Warning: Could not compute or plot convex hull: {e}. "
                    "Only plotting points."
                )
        else:
            print(
                "  Warning: Not enough stable points (need >= 4) to compute "
                "3D convex hull."
            )

        print("Plotting stable phase points...")
        if stable_coords.any():  # Check if there are any stable coordinates to plot
            ax.scatter(
                stable_coords[:, 0],
                stable_coords[:, 1],
                stable_coords[:, 2],
                c="black",
                marker="o",
                s=60,
                label="Stable Phases (Input)",
                depthshade=False,
                alpha=0.8,
            )
        else:
            print("  No stable phase coordinates found to plot.")

        print(f"Plotting calculated results with Ehull <= {ehull_threshold} eV/atom...")
        calculated_coords: list[np.ndarray] = []
        calculated_ehull: list[float] = []
        calculated_labels: list[str] = []

        for formula, ehull, coords in calculated_results:
            if coords is not None and ehull <= ehull_threshold:
                calculated_coords.append(coords)
                calculated_ehull.append(ehull)
                calculated_labels.append(formula)

        if calculated_coords:
            calculated_coords_arr = np.array(calculated_coords)
            calculated_ehull_arr = np.array(calculated_ehull)

            # Normalize Ehull values for colormap
            norm = mcolors.Normalize(vmin=0, vmax=ehull_threshold)
            # Reversed viridis: blue (low Ehull) to yellow (high Ehull)
            cmap = mcmaps["rainbow_r"]

            sc = ax.scatter(
                calculated_coords_arr[:, 0],
                calculated_coords_arr[:, 1],
                calculated_coords_arr[:, 2],
                c=calculated_ehull_arr,
                cmap=cmap,
                norm=norm,
                marker="^",
                s=40,
                label=f"Calculated (Ehull <= {ehull_threshold:.3f})",
                depthshade=True,
                alpha=0.9,
            )  # Use depthshade for better 3D perception

            # Add Colorbar
            cbar = fig.colorbar(sc, shrink=0.6, aspect=20, pad=0.1)
            cbar.set_label("Formation Energy above Hull (eV/atom)")
            print(f"  Plotted {len(calculated_coords)} calculated points.")
        else:
            print("  No calculated results found within the Ehull threshold.")

        ax.set_xlabel("Composition Space X")
        ax.set_ylabel("Composition Space Y")
        ax.set_zlabel("Composition Space Z")

        # Remove axis ticks/grid for cleaner compositional space view
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.grid(False)
        plt.axis("off")  # Turn off the axis frame

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
        print(
            "\nError: No data could be parsed from input files for the specified"
            " elements. Cannot generate plot."
        )
        return output_file
    plot_quaternary_hull(
        element_symbols, stable_data, results_data, ehull_threshold, output_file
    )
    return output_file


@python_app(executors=[POSTPROCESSING_LABEL])
def convex_hull_color(config: dict[str, Any]) -> None:
    """Parsl app that dispatches ternary or quaternary convex hull plotting.

    Parameters
    ----------
    config : dict
        Configuration mapping using :class:`ConfigKeys`. Relevant keys are
        the element system, post-processing output directory, and the hull
        energy threshold.

    Returns
    -------
    None

    Raises
    ------
    Exception
        Re-raised if any error occurs during plotting.
    """
    try:
        import os

        elements = config[CK.ELEMENTS]
        l_elements = elements.split("-")
        nb_of_elements = len(l_elements)
        stable_dat = os.path.join(config[CK.POST_PROCESSING_OUT_DIR], CK.MP_STABLE_OUT)
        elename = "".join(l_elements)
        input_csv = (
            elename + ".csv" if nb_of_elements == 3 else elename + "_quaternary.csv"
        )
        full_path_input_csv = os.path.join(
            config[CK.POST_PROCESSING_OUT_DIR], input_csv
        )
        output_file = os.path.join(
            config[CK.POST_PROCESSING_OUT_DIR], CK.POST_PROCESSING_FINAL_OUT
        )
        threshold = float(config[CK.HULL_ENERGY_THR])
        if nb_of_elements == 3:
            plot_convex_hull_ternary(
                l_elements, stable_dat, full_path_input_csv, threshold, output_file
            )
        else:
            plot_convex_hull_quaternary(
                l_elements, stable_dat, full_path_input_csv, threshold, output_file
            )
    except Exception:
        raise
