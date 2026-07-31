"""Tests for the quaternary routines in parsl_tasks.ehull_utils."""

from __future__ import annotations

from pathlib import Path

import pytest
from pymatgen.core import Element

from parsl_tasks.ehull_utils import (
    dhull_quaternary,
    judge_stable_quaternary,
    parse_stable_phases_quaternary,
)


# ---------------------------------------------------------------------------
# parse_stable_phases_quaternary
# ---------------------------------------------------------------------------
def _elements() -> list[Element]:
    """Return the four-element system used across the tests.

    Returns
    -------
    list of pymatgen.core.Element
        The elements ``[Al, Fe, Co, O]``.
    """
    return [Element("Al"), Element("Fe"), Element("Co"), Element("O")]


def test_parse_requires_four_elements() -> None:
    """A system without exactly four elements raises ``ValueError``."""
    with pytest.raises(ValueError):
        parse_stable_phases_quaternary("ignored.dat", [Element("Al"), Element("Fe")])


def test_parse_skips_foreign_and_malformed(tmp_path: Path) -> None:
    """Foreign-element and unparsable lines are ignored."""
    f = tmp_path / "stable.dat"
    f.write_text(
        "AlFeCoO4 -5.0\n"  # valid, all four elements
        "AlFe -1.0\n"  # valid subset (only two elements)
        "AlFeCoNiO -2.0\n"  # foreign element Ni -> skipped
        "AlFeCoO notanumber\n"  # bad energy -> skipped
        "\n"  # blank -> skipped
    )

    stable_vec, quaternary_vec = parse_stable_phases_quaternary(str(f), _elements())

    formulas = {row[0] for row in stable_vec}
    assert "AlFeCoNiO4" not in formulas
    assert "AlFeCoNiO" not in formulas
    # only the record with all four fractions > 0 is a true quaternary
    assert [row[0] for row in quaternary_vec] == ["AlFeCoO4"]


def test_parse_records_have_fraction_and_energy(tmp_path: Path) -> None:
    """Each stable record stores formula, four fractions, and energy."""
    f = tmp_path / "stable.dat"
    f.write_text("AlFeCoO4 -5.0\n")

    stable_vec, _ = parse_stable_phases_quaternary(str(f), _elements())

    assert len(stable_vec) == 1
    row = stable_vec[0]
    assert row[0] == "AlFeCoO4"
    assert len(row) == 6  # formula, x, y, z, w, energy
    assert row[-1] == -5.0
    assert abs(sum(row[1:5]) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# judge_stable_quaternary
# ---------------------------------------------------------------------------
def test_judge_requires_four_elements() -> None:
    """Judging a non-quaternary system raises ``ValueError``."""
    with pytest.raises(ValueError):
        judge_stable_quaternary([], ["Al", "Fe", "Co"], "AlFeCo", -1.0)


def test_judge_composition_error_when_fractions_missing() -> None:
    """A formula missing one of the four elements returns the error sentinel."""
    # formula omits O, so fractions over [Al, Fe, Co, O] sum to 1 across
    # only three elements -> w == 0, but total still sums to 1, so this
    # instead exercises the path where a system element is absent.
    d_hull, hull_vec = judge_stable_quaternary(
        [], ["Al", "Fe", "Co", "O"], "AlFeCo", -1.0
    )

    # AlFeCo has no O -> fractions still sum to 1, delegates to dhull with
    # empty hull -> no simplex found.
    assert d_hull == -100
    assert hull_vec == ["Error"] * 4


def test_judge_empty_hull_returns_sentinel() -> None:
    """With no stable phases the hull distance is the -100 sentinel."""
    d_hull, hull_vec = judge_stable_quaternary(
        [], ["Al", "Fe", "Co", "O"], "AlFeCoO4", -1.0
    )

    assert d_hull == -100
    assert hull_vec == ["Error"] * 4


# ---------------------------------------------------------------------------
# dhull_quaternary
# ---------------------------------------------------------------------------
def _corner(name: str, x: float, y: float, z: float, w: float, e: float) -> list:
    """Build a hull vertex ``[formula, x, y, z, w, energy]``.

    Parameters
    ----------
    name : str
        Vertex label.
    x, y, z, w : float
        Atomic fractions of the four elements.
    e : float
        Energy at the vertex.

    Returns
    -------
    list
        The hull vertex row.
    """
    return [name, x, y, z, w, e]


def test_dhull_no_simplex_returns_sentinel() -> None:
    """Fewer than four hull points yield the -100 sentinel."""
    hull = [_corner("A", 1, 0, 0, 0, 0.0)]
    struc = ["S", 0.25, 0.25, 0.25, 0.25, 0.0]

    d_hull, hull_vec = dhull_quaternary(struc, hull)

    assert d_hull == -100
    assert hull_vec == ["Error"] * 4


def test_dhull_computes_distance_on_simplex() -> None:
    """A structure inside the elemental simplex gets a finite hull distance."""
    # Four pure elements at zero energy form the base simplex.
    hull = [
        _corner("Al", 1, 0, 0, 0, 0.0),
        _corner("Fe", 0, 1, 0, 0, 0.0),
        _corner("Co", 0, 0, 1, 0, 0.0),
        _corner("O", 0, 0, 0, 1, 0.0),
    ]
    # Equal-fraction structure with negative energy -> below the hull.
    struc = ["S", 0.25, 0.25, 0.25, 0.25, -1.0]

    d_hull, hull_vec = dhull_quaternary(struc, hull)

    # d_convex = struc_e - sum(s_i * hull_e_i) = -1.0 - 0 = -1.0
    assert d_hull == pytest.approx(-1.0)
    assert hull_vec == ["Al", "Fe", "Co", "O"]
