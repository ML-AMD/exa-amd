"""Tests for :mod:`parsl_tasks.ml_ehull`.

These tests cover the energy-file reader (:func:`read_energies`), the ternary
and quaternary per-structure judging wrappers, the parallel ternary/quaternary
e-hull routines (sorting, output formatting, failure skipping and sentinel
filtering), and the element-count based dispatch in ``ehull_ml_parallel``.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from parsl_tasks import ml_ehull
from parsl_tasks.ml_ehull import (
    parallel_quaternary_ehull,
    parallel_ternary_ehull,
    process_structure_wrapper_quaternary,
    process_structure_wrapper_ternary,
    read_energies,
)
from tools.config_labels import ConfigKeys as CK


class _FakeElement:
    """Minimal :class:`pymatgen.core.Element` double exposing ``symbol``.

    Parameters
    ----------
    symbol : str
        Chemical element symbol (e.g. ``"Al"``).
    """

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol


def _write_energies(path: Path, rows: list[tuple[int, float, str]]) -> Path:
    """Write an energy input file with ``index,energy,formula`` lines.

    Parameters
    ----------
    path : pathlib.Path
        Destination file path.
    rows : list of tuple of (int, float, str)
        The ``(index, energy, formula)`` records to write.

    Returns
    -------
    pathlib.Path
        The written file path.
    """
    path.write_text("".join(f"{i},{e},{f}\n" for i, e, f in rows))
    return path


# ---------------------------------------------------------------------------
# read_energies
# ---------------------------------------------------------------------------
def test_read_energies_parses_rows(tmp_path: Path) -> None:
    """Fields are parsed into typed, order-preserving lists."""
    f = _write_energies(
        tmp_path / "ener.dat",
        [(0, -1.5, "AlFeO"), (1, -2.0, "AlFe2O")],
    )

    energies, indices, formulas = read_energies(str(f))

    assert energies == [-1.5, -2.0]
    assert indices == [0, 1]
    assert formulas == ["AlFeO", "AlFe2O"]


def test_read_energies_strips_formula_whitespace(tmp_path: Path) -> None:
    """Trailing whitespace/newlines are stripped from formulas."""
    f = _write_energies(tmp_path / "ener.dat", [(3, -0.1, "AlFeO")])

    _, _, formulas = read_energies(str(f))

    assert formulas == ["AlFeO"]


# ---------------------------------------------------------------------------
# process_structure_wrapper_*
# ---------------------------------------------------------------------------
def test_ternary_wrapper_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Successful judging returns the index, hull data, and formula."""
    monkeypatch.setattr(ml_ehull, "judge_stable_ternary", lambda *a: (-0.25, [1, 0, 0]))

    result = process_structure_wrapper_ternary(7, "AlFeO", -1.0, {}, ["Al", "Fe", "O"])

    assert result == (7, -0.25, [1, 0, 0], "AlFeO")


def test_ternary_wrapper_failure_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exceptions are swallowed and produce ``None`` results."""

    def boom(*_a):
        raise ValueError("bad")

    monkeypatch.setattr(ml_ehull, "judge_stable_ternary", boom)

    result = process_structure_wrapper_ternary(7, "AlFeO", -1.0, {}, ["Al", "Fe", "O"])

    assert result == (7, None, None, "AlFeO")


def test_quaternary_wrapper_failure_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Quaternary wrapper also swallows exceptions."""

    def boom(*_a):
        raise RuntimeError("bad")

    monkeypatch.setattr(ml_ehull, "judge_stable_quaternary", boom)

    result = process_structure_wrapper_quaternary(
        2, "AlFeCoO", -1.0, {}, ["Al", "Fe", "Co", "O"]
    )

    assert result == (2, None, None, "AlFeCoO")


# ---------------------------------------------------------------------------
# parallel_ternary_ehull
# ---------------------------------------------------------------------------
@mock.patch("parsl_tasks.ml_ehull.Element", side_effect=_FakeElement)
def test_parallel_ternary_sorts_and_writes(
    _mock_element,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Output is sorted by Ehull ascending and written as ``index,Ehull``."""
    in_file = _write_energies(
        tmp_path / "ener.dat",
        [(0, -1.0, "A"), (1, -1.0, "B"), (2, -1.0, "C")],
    )
    out_file = tmp_path / "hull.dat"

    monkeypatch.setattr(
        ml_ehull,
        "parse_stable_phases_ternary",
        lambda *a: ({"stub": 1}, [("phase",)]),
    )
    hulls = {"A": 0.5, "B": -0.5, "C": 0.1}
    monkeypatch.setattr(
        ml_ehull,
        "judge_stable_ternary",
        lambda sv, el, formula, en: (hulls[formula], []),
    )

    result = parallel_ternary_ehull(
        str(in_file), "stable.dat", str(out_file), "Al-Fe-O", workers=1
    )

    assert result == str(out_file)
    lines = out_file.read_text().splitlines()
    # sorted by energy: B(-0.5), C(0.1), A(0.5)
    assert lines == ["1,-0.500000", "2,0.100000", "0,0.500000"]


@mock.patch("parsl_tasks.ml_ehull.Element", side_effect=_FakeElement)
def test_parallel_ternary_skips_failed(
    _mock_element,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Rows whose judging returns ``None`` are excluded from output."""
    in_file = _write_energies(tmp_path / "ener.dat", [(0, -1.0, "A"), (1, -1.0, "B")])
    out_file = tmp_path / "hull.dat"

    monkeypatch.setattr(
        ml_ehull, "parse_stable_phases_ternary", lambda *a: ({"s": 1}, [])
    )

    def judge(sv, el, formula, en):
        if formula == "A":
            raise ValueError("fail")
        return (-0.3, [])

    monkeypatch.setattr(ml_ehull, "judge_stable_ternary", judge)

    parallel_ternary_ehull(
        str(in_file), "stable.dat", str(out_file), "Al-Fe-O", workers=1
    )

    assert out_file.read_text().splitlines() == ["1,-0.300000"]


# ---------------------------------------------------------------------------
# parallel_quaternary_ehull
# ---------------------------------------------------------------------------
@mock.patch("parsl_tasks.ml_ehull.Element", side_effect=_FakeElement)
def test_parallel_quaternary_filters_sentinel(
    _mock_element,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Values at/below the -99.0 sentinel are filtered out."""
    in_file = _write_energies(tmp_path / "ener.dat", [(0, -1.0, "A"), (1, -1.0, "B")])
    out_file = tmp_path / "hull.dat"

    monkeypatch.setattr(
        ml_ehull, "parse_stable_phases_quaternary", lambda *a: ({"s": 1}, None)
    )
    hulls = {"A": -100.0, "B": -0.2}
    monkeypatch.setattr(
        ml_ehull,
        "judge_stable_quaternary",
        lambda sv, el, formula, en: (hulls[formula], []),
    )

    parallel_quaternary_ehull(
        str(in_file), "stable.dat", str(out_file), "Al-Fe-Co-O", workers=1
    )

    assert out_file.read_text().splitlines() == ["1,-0.200000"]


# ---------------------------------------------------------------------------
# ehull_ml_parallel (dispatch)
# ---------------------------------------------------------------------------
def _dispatch_config() -> dict:
    """Build a config for the dispatch app tests.

    Returns
    -------
    dict
        Config with work dirs and an element specification.
    """
    return {
        CK.WORK_DIR: "/work",
        CK.VASP_WORK_DIR: "/vasp",
        CK.ELEMENTS: "Al-Fe-O",
    }


@mock.patch(
    "parsl_tasks.ml_ehull.parallel_ternary_ehull", return_value="/work/hull.dat"
)
def test_dispatch_ternary(mock_ternary: mock.MagicMock) -> None:
    """A 3-element system dispatches to the ternary routine."""
    config = _dispatch_config()

    result = ml_ehull.ehull_ml_parallel.func(config)

    assert result == "/work/hull.dat"
    mock_ternary.assert_called_once()
    # elements arg is passed through unchanged
    assert mock_ternary.call_args.args[-1] == "Al-Fe-O"


@mock.patch(
    "parsl_tasks.ml_ehull.parallel_quaternary_ehull", return_value="/work/hull.dat"
)
def test_dispatch_quaternary(mock_quaternary: mock.MagicMock) -> None:
    """A 4-element system dispatches to the quaternary routine."""
    config = _dispatch_config()
    config[CK.ELEMENTS] = "Al-Fe-Co-O"

    result = ml_ehull.ehull_ml_parallel.func(config)

    assert result == "/work/hull.dat"
    mock_quaternary.assert_called_once()


def test_dispatch_unsupported_raises() -> None:
    """A system size other than 3 or 4 causes a critical exit."""
    config = _dispatch_config()
    config[CK.ELEMENTS] = "Al-Fe"

    with pytest.raises(SystemExit):
        ml_ehull.ehull_ml_parallel.func(config)
