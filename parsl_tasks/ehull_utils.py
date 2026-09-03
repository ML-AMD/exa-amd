"""Convex-hull energy-above-hull (Ehull) utilities for ternary and quaternary systems.

Hull vertices and structure vectors use a positional layout:

- Ternary: ``[formula, x, y, z, energy]`` (index 4 is energy).
- Quaternary: ``[formula, x, y, z, w, energy]`` (index 5 is energy).

where ``x, y, z, w`` are atomic fractions of the system elements.
"""

from itertools import combinations
from typing import Any, Sequence

import numpy as np
from pymatgen.core import Composition, Element

# A hull/structure vector: [formula, *fractions, energy]
HullVec = list[Any]
# Element specification: symbols or pymatgen Element objects.
ElementSpec = Sequence[str | Element]


def det_tern(v1: HullVec, v2: HullVec, v3: HullVec) -> float:
    """Compute the determinant of three ternary fraction rows.

    Parameters
    ----------
    v1, v2, v3 : list
        Hull vectors; indices 1..3 hold the atomic fractions used as rows.

    Returns
    -------
    float
        Determinant of the 3x3 matrix formed from the fraction components.
    """
    m = np.array([[v1[1], v1[2], v1[3]], [v2[1], v2[2], v2[3]], [v3[1], v3[2], v3[3]]])
    return float(np.linalg.det(m))


def dhull_ternary(struc: HullVec, hull: Sequence[HullVec]) -> tuple[float, list[str]]:
    """Compute a structure's energy above the ternary convex hull.

    Searches all triangular facets of the hull, solving barycentric
    coordinates and keeping the maximum convex distance.

    Parameters
    ----------
    struc : list
        Structure vector ``[formula, x, y, z, energy]``.
    hull : sequence of list
        Stable-phase hull vectors with the same layout.

    Returns
    -------
    tuple of (float, list of str)
        ``(d_hull_max, hull_vec)`` where ``hull_vec`` is the formulas of the
        three vertices defining the facet. Defaults to ``(-100, ["0", "0", "0"])``
        if no valid facet is found.
    """
    tol = 1.0e-5
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
                        d_convex = (
                            struc[4]
                            - s1 * hull[i][4]
                            - s2 * hull[j][4]
                            - s3 * hull[k][4]
                        )
                        if d_hull_max < d_convex:
                            d_hull_max = d_convex
                            hull_vec = [hull[i][0], hull[j][0], hull[k][0]]
    return d_hull_max, hull_vec


def judge_stable_ternary(
    stable_vec: Sequence[HullVec],
    system_symbols: ElementSpec,
    comp_struc: str,
    predict_Eper: float,
) -> tuple[float, list[str]]:
    """Evaluate a ternary structure against the stable-phase hull.

    Parameters
    ----------
    stable_vec : sequence of list
        Stable-phase hull vectors.
    system_symbols : sequence of str or pymatgen.core.Element
        The three system elements.
    comp_struc : str
        Chemical formula of the structure to evaluate.
    predict_Eper : float
        Predicted per-atom energy of the structure.

    Returns
    -------
    tuple of (float, list of str)
        The hull distance and defining vertex formulas, as returned by
        :func:`dhull_ternary`.
    """
    Comp = Composition(comp_struc)
    symbols = [e.symbol if isinstance(e, Element) else str(e) for e in system_symbols]
    x = Comp.get_atomic_fraction(Element(symbols[0]))
    y = Comp.get_atomic_fraction(Element(symbols[1]))
    z = Comp.get_atomic_fraction(Element(symbols[2]))
    struc_vec = [comp_struc, x, y, z, predict_Eper]
    return dhull_ternary(struc_vec, stable_vec)


def dhull_quaternary(
    struc: HullVec, hull: Sequence[HullVec]
) -> tuple[float, list[str]]:
    """Compute a structure's energy above the quaternary convex hull.

    Iterates over all 4-vertex simplices of the hull, solving the barycentric
    system and keeping the maximum convex distance.

    Parameters
    ----------
    struc : list
        Structure vector ``[formula, x, y, z, w, energy]``.
    hull : sequence of list
        Stable-phase hull vectors with the same layout.

    Returns
    -------
    tuple of (float, list of str)
        ``(d_hull_max, hull_vec)``. Returns ``(-100, ["Error"] * 4)`` when no
        valid simplex is found.
    """
    tol = 1.0e-5
    d_hull_max = -100
    hull_vec = ["0"] * 4
    le = len(hull)

    for i, j, k, l_idx in combinations(range(le), 4):
        try:
            m3 = np.array(
                [
                    [
                        hull[j][1] - hull[i][1],
                        hull[j][2] - hull[i][2],
                        hull[j][3] - hull[i][3],
                    ],
                    [
                        hull[k][1] - hull[i][1],
                        hull[k][2] - hull[i][2],
                        hull[k][3] - hull[i][3],
                    ],
                    [
                        hull[l_idx][1] - hull[i][1],
                        hull[l_idx][2] - hull[i][2],
                        hull[l_idx][3] - hull[i][3],
                    ],
                ]
            )
            det_check = np.linalg.det(m3)
        except IndexError:
            continue
        if abs(det_check) > tol:
            A = np.array(
                [
                    [hull[i][1], hull[j][1], hull[k][1], hull[l_idx][1]],
                    [hull[i][2], hull[j][2], hull[k][2], hull[l_idx][2]],
                    [hull[i][3], hull[j][3], hull[k][3], hull[l_idx][3]],
                    [hull[i][4], hull[j][4], hull[k][4], hull[l_idx][4]],
                ]
            )
            b = np.array([struc[1], struc[2], struc[3], struc[4]])
            try:
                s1, s2, s3, s4 = np.linalg.solve(A, b)
                if (
                    s1 >= -tol
                    and s2 >= -tol
                    and s3 >= -tol
                    and s4 >= -tol
                    and abs(s1 + s2 + s3 + s4 - 1.0) < tol
                ):
                    d_convex = struc[5] - (
                        s1 * hull[i][5]
                        + s2 * hull[j][5]
                        + s3 * hull[k][5]
                        + s4 * hull[l_idx][5]
                    )
                    if d_hull_max < d_convex:
                        d_hull_max = d_convex
                        hull_vec = [hull[i][0], hull[j][0], hull[k][0], hull[l_idx][0]]
            except np.linalg.LinAlgError:
                continue
    if abs(d_hull_max - (-100)) < tol:
        return -100, ["Error"] * 4
    return d_hull_max, hull_vec


def judge_stable_quaternary(
    stable_vec: Sequence[HullVec],
    system_elements: ElementSpec,
    comp_struc: str,
    predict_Eper: float,
) -> tuple[float, list[str]]:
    """Evaluate a quaternary structure against the stable-phase hull.

    Parameters
    ----------
    stable_vec : sequence of list
        Stable-phase hull vectors.
    system_elements : sequence of str or pymatgen.core.Element
        The four system elements.
    comp_struc : str
        Chemical formula of the structure to evaluate.
    predict_Eper : float
        Predicted per-atom energy of the structure.

    Returns
    -------
    tuple of (float, list of str)
        The hull distance and defining vertex formulas. Returns
        ``(-100, ["Composition Error"] * 4)`` if the fractions do not sum to 1.

    Raises
    ------
    ValueError
        If ``system_elements`` does not contain exactly four elements.
    """
    Comp = Composition(comp_struc)
    if len(system_elements) != 4:
        raise ValueError(
            "System must contain exactly 4 elements for quaternary calculation."
        )
    elements = [e.symbol if isinstance(e, Element) else str(e) for e in system_elements]
    x = Comp.get_atomic_fraction(Element(elements[0]))
    y = Comp.get_atomic_fraction(Element(elements[1]))
    z = Comp.get_atomic_fraction(Element(elements[2]))
    w = Comp.get_atomic_fraction(Element(elements[3]))
    if not abs(x + y + z + w - 1.0) < 1e-6:
        return -100, ["Composition Error"] * 4
    struc_vec = [comp_struc, x, y, z, w, predict_Eper]
    return dhull_quaternary(struc_vec, stable_vec)


def parse_stable_phases_ternary(
    filename: str, elements: ElementSpec
) -> tuple[list[HullVec], list[HullVec]]:
    """Parse the stable phases file and create ternary hull vectors.

    Parameters
    ----------
    filename : str
        Path to file containing lines: ``"formula energy"``.
    elements : sequence of str or pymatgen.core.Element
        Element symbols for the ternary system, e.g. ``["A", "B", "C"]``.

    Returns
    -------
    tuple of (list of list, list of list)
        ``(stable_vec, ternary_vec)``. ``stable_vec`` holds all in-system
        phases; ``ternary_vec`` holds only those with all three fractions > 0.
    """
    stable_vec, ternary_vec = [], []

    elements_symbols = [e.symbol if hasattr(e, "symbol") else str(e) for e in elements]
    allowed = set(elements_symbols)

    with open(filename, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            formula, energy_str = parts
            energy = float(energy_str)
            comp = Composition(formula)
            comp_symbols = {e.symbol for e in comp.elements}

            # composition must only contain our elements of interest
            if not comp_symbols.issubset(allowed):
                continue
            x = comp.get_atomic_fraction(Element(elements_symbols[0]))
            y = comp.get_atomic_fraction(Element(elements_symbols[1]))
            z = comp.get_atomic_fraction(Element(elements_symbols[2]))

            stable_vec.append([formula, x, y, z, energy])
            if x > 0 and y > 0 and z > 0:
                ternary_vec.append([formula, x, y, z, energy])

    return stable_vec, ternary_vec


def parse_stable_phases_quaternary(
    filename: str, elements: ElementSpec
) -> tuple[list[HullVec], list[HullVec]]:
    """Parse the stable phases file and create quaternary hull vectors.

    Parameters
    ----------
    filename : str
        Path to file containing lines with a leading formula and a trailing
        energy value.
    elements : sequence of pymatgen.core.Element
        The four system elements. ``get_atomic_fraction`` is called with these
        directly, so ``Element`` objects are expected.

    Returns
    -------
    tuple of (list of list, list of list)
        ``(stable_vec, quaternary_vec)``. ``stable_vec`` holds all in-system
        phases whose fractions sum to 1; ``quaternary_vec`` holds only those
        with all four fractions > 0.

    Raises
    ------
    ValueError
        If ``elements`` does not contain exactly four elements.
    """
    if len(elements) != 4:
        raise ValueError("Must provide exactly 4 elements for quaternary parsing.")
    stable_vec, quaternary_vec = [], []
    symbols = [e.symbol if isinstance(e, Element) else str(e) for e in elements]
    element_symbols = set(symbols)
    with open(filename, "r") as f:
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
                    x = float(comp.get_atomic_fraction(Element(symbols[0])))
                    y = float(comp.get_atomic_fraction(Element(symbols[1])))
                    z = float(comp.get_atomic_fraction(Element(symbols[2])))
                    w = float(comp.get_atomic_fraction(Element(symbols[3])))
                    if abs(x + y + z + w - 1.0) < 1e-6:
                        stable_vec.append([formula, x, y, z, w, energy])
                        if x > 1e-6 and y > 1e-6 and z > 1e-6 and w > 1e-6:
                            quaternary_vec.append([formula, x, y, z, w, energy])
            except Exception:
                continue
    return stable_vec, quaternary_vec
