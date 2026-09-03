"""Tests for :mod:`tools.post_processing` and the convex-hull utilities.

These tests cover e-hull calculation and CSV/plot generation
(:func:`cmd_calculate_ehull`, :func:`plot_convex_hull_ternary`,
:func:`plot_convex_hull_quaternary`), the VASP hull compilation
(:func:`cmd_compile_vasp_hull`, :func:`get_vasp_hull`), Materials Project
stable-phase retrieval (:func:`get_stable_phases`), and the
``convex_hull_color`` dispatch based on the number of elements.
"""

import os
import re
from pathlib import Path
from typing import Any, Literal
from unittest import mock

import pytest
from conftest import (
    FakeFuture,
    ResourcePathCtx,
    extract_tar,
    pushd,
    write_element_potcars,
)

from tools.config_labels import ConfigKeys as CK


@pytest.fixture(scope="module")
def ehull_env(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Extract the e-hull fixture archive and build a matching config.

    Parameters
    ----------
    tmp_path_factory : pytest.TempPathFactory
        Factory used to create a module-scoped temporary directory.

    Returns
    -------
    dict
        Mapping with ``ehull_dir`` (the extracted directory) and ``config``
        (a config dict pointing at that directory).
    """
    tmp = tmp_path_factory.mktemp("ehull_fixture")
    tar_path = Path(__file__).parent / "post_processing.tar"

    extract_tar(tar_path, tmp)

    ehull_dir = tmp / "post_processing"
    assert (ehull_dir / "energy.dat").exists(), "energy.dat missing"
    assert (ehull_dir / "mp_int_stable.dat").exists(), "mp_int_stable.dat missing"

    config = {
        CK.ELEMENTS: "Na-B-C",
        CK.VASP_WORK_DIR: str(ehull_dir),
        CK.ENERGY_DAT_OUT: "energy.dat",
        CK.POST_PROCESSING_OUT_DIR: str(ehull_dir),
        CK.MP_STABLE_OUT: "mp_int_stable.dat",
    }
    return {"ehull_dir": ehull_dir, "config": config}


def test_calculate_ehull_outputs(ehull_env: dict[str, Any]) -> None:
    """Running ``cmd_calculate_ehull`` writes hull, CSV and selected outputs.

    Parameters
    ----------
    ehull_env : dict
        Fixture providing the extracted directory and config.
    """
    from parsl_tasks.ehull import cmd_calculate_ehull

    ehull_dir = ehull_env["ehull_dir"]
    config = ehull_env["config"]

    out_path = Path(cmd_calculate_ehull(config, False))

    assert out_path == ehull_dir / "hull.dat"

    assert out_path.exists(), "hull.dat not created"
    assert out_path.stat().st_size > 0, "hull.dat is empty"

    csv_path = ehull_dir / "NaBC.csv"
    assert csv_path.exists(), "NaBC.csv not created"
    with open(csv_path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    assert lines, "NaBC.csv is empty"

    sel_dir = ehull_dir / "selected"
    assert sel_dir.exists(), '"selected/" directory not created'
    assert list(sel_dir.glob("CONTCAR_*")), "No CONTCAR_* copied into selected/"

    with open(out_path) as f:
        first = f.readline().strip()
    assert first.count(",") >= 3, "Unexpected hull.dat line format"


def test_convex_hull_color_ternary(
    ehull_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``plot_convex_hull_ternary`` produces a PNG and hull data file.

    Parameters
    ----------
    ehull_env : dict
        Fixture providing the extracted directory and config.
    monkeypatch : pytest.MonkeyPatch
        Used to force a non-interactive matplotlib backend.
    """
    from parsl_tasks.convex_hull import plot_convex_hull_ternary
    from parsl_tasks.ehull import cmd_calculate_ehull

    # ensure non-interactive matplotlib
    monkeypatch.setenv("MPLBACKEND", "Agg")

    ehull_dir = ehull_env["ehull_dir"]
    config = ehull_env["config"]

    # Produce NaBC.csv independently of test ordering.
    cmd_calculate_ehull(config, False)

    elements_list = config[CK.ELEMENTS].split("-")
    stable_dat = os.path.join(ehull_dir, config[CK.MP_STABLE_OUT])
    input_csv = os.path.join(ehull_dir, "NaBC.csv")
    assert Path(input_csv).exists(), "NaBC.csv missing"

    output_png = os.path.join(ehull_dir, "convex_hull.png")
    threshold = 0.10

    with pushd(ehull_dir):
        str_out_path = plot_convex_hull_ternary(
            elements_list=elements_list,
            stable_dat=stable_dat,
            full_path_input_csv=input_csv,
            threshold=threshold,
            output_file=output_png,
        )

    out_path = Path(str_out_path)
    assert out_path == Path(output_png)
    assert out_path.exists() and out_path.stat().st_size > 0

    hull_txt = Path(ehull_dir) / "convex-hull.dat"
    assert hull_txt.exists() and hull_txt.stat().st_size > 0


@pytest.fixture(scope="module")
def compile_vasp_hull_inputs(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, Any]:
    """Extract the compile-hull fixture and describe its calc directories.

    Parameters
    ----------
    tmp_path_factory : pytest.TempPathFactory
        Factory used to create a module-scoped temporary directory.

    Returns
    -------
    dict
        Mapping with ``root``, ``total`` (highest calc index), ``prefix``
        (calc path prefix) and ``out_file`` (compiled output path).
    """
    tmp = tmp_path_factory.mktemp("compile_hull_in")
    tar_path = Path(__file__).parent / "compile_hull_in.tar"
    assert tar_path.exists(), f"Missing {tar_path}"

    extract_tar(tar_path, tmp)

    root = tmp / "compile_hull_in"
    assert root.exists()

    calc_indices = []
    for p in root.iterdir():
        if p.is_dir():
            m = re.fullmatch(r"calc_(\d+)", p.name)
            if m:
                calc_indices.append(int(m.group(1)))
                assert (p / "INCAR").exists()
                assert (p / "OUTCAR").exists()
                assert (p / "output").exists()
    assert calc_indices, "No calc_* directories found"
    total = max(calc_indices)

    return {
        "root": root,
        "total": total,
        "prefix": str(root / "calc_"),
        "out_file": root / "mp_int_stable.dat",
    }


def _expected_compile_vasp_hull(root: Path, total: int) -> dict[str, float]:
    """Compute the expected ``{formula: min_energy_per_atom}`` mapping.

    Parameters
    ----------
    root : pathlib.Path
        Directory containing the ``calc_*`` subdirectories.
    total : int
        Highest calc index to inspect (inclusive).

    Returns
    -------
    dict
        Minimum energy per atom keyed by chemical formula.
    """
    expected: dict[str, float] = {}
    for i in range(1, total + 1):
        d = root / f"calc_{i}"
        if not d.exists():
            continue

        incar, out, outcar = d / "INCAR", d / "output", d / "OUTCAR"
        formula, energy, natoms = "", None, None

        try:
            last = None
            with open(incar) as f:
                for ln in f:
                    if "SYSTEM" in ln:
                        last = ln
            if last:
                toks = last.split()
                if len(toks) >= 3:
                    formula = toks[2]
        except FileNotFoundError:
            pass

        try:
            last = None
            with open(out) as f:
                for ln in f:
                    if "F=" in ln:
                        last = ln
            if last:
                toks = last.split()
                if len(toks) >= 5:
                    energy = float(toks[4])
        except FileNotFoundError:
            pass

        try:
            last = None
            with open(outcar) as f:
                for ln in f:
                    if "NIONS" in ln:
                        last = ln
            if last:
                natoms = int(last.split()[-1])
        except FileNotFoundError:
            pass

        if formula and (energy is not None) and natoms:
            tenergy = float(f"{energy:.6f}")
            epa = float(f"{tenergy / natoms:.6f}")
            prev = expected.get(formula)
            if prev is None or epa < prev:
                expected[formula] = epa
    return expected


def test_cmd_compile_vasp_hull(
    compile_vasp_hull_inputs: dict[str, Any],
) -> None:
    """``cmd_compile_vasp_hull`` writes alphabetized minimum energies per atom.

    Parameters
    ----------
    compile_vasp_hull_inputs : dict
        Fixture describing the extracted calc directories and output path.
    """
    from parsl_tasks.compile_hull import cmd_compile_vasp_hull

    root = compile_vasp_hull_inputs["root"]
    total = compile_vasp_hull_inputs["total"]
    prefix = compile_vasp_hull_inputs["prefix"]
    out_file = compile_vasp_hull_inputs["out_file"]

    cmd_compile_vasp_hull(total_calcs=total, output_file=str(out_file), prefix=prefix)

    assert out_file.exists() and out_file.stat().st_size > 0

    # Parse output -> {formula: epa}
    lines = [ln.strip() for ln in out_file.read_text().splitlines() if ln.strip()]
    got = {}
    for ln in lines:
        toks = ln.split()
        assert len(toks) == 2, f"Unexpected line: {ln}"
        got[toks[0]] = float(toks[1])

    expected = _expected_compile_vasp_hull(root, total)
    assert set(got) == set(expected), "Formula set mismatch"
    for k in expected:
        assert abs(got[k] - expected[k]) <= 1e-6, f"{k}: {got[k]} != {expected[k]}"

    # Output in alphabetical order
    formulas = [ln.split()[0] for ln in lines]
    assert formulas == sorted(formulas)


def _write_quaternary_inputs(root: Path, elements: list[str]) -> tuple[str, str]:
    """Create stable ``.dat`` and results ``.csv`` files for a 4-element system.

    Parameters
    ----------
    root : pathlib.Path
        Directory in which to write the fixture files.
    elements : list[str]
        The four element symbols composing the system.

    Returns
    -------
    tuple[str, str]
        Paths to the written stable ``.dat`` and results ``.csv`` files.
    """
    a, b, c, d = elements
    stable = root / "mp_int_stable.dat"
    # Pure elements + a couple of binaries so ConvexHull has >= 4 points.
    stable.write_text(
        "\n".join(
            [
                f"{a} -1.0",
                f"{b} -1.5",
                f"{c} -2.0",
                f"{d} -2.5",
                f"{a}{b} -3.0",
                f"{c}{d} -4.0",
            ]
        )
        + "\n"
    )

    csv = root / "results_quaternary.csv"
    csv.write_text(
        "\n".join(
            [
                "Formula,Total_Energy_per_atom,Ehull",
                f"{a}2{b}2,-2.0,0.02",
                f"{a}{b}{c}{d},-3.0,0.5",  # above threshold
                f"{c}2{d},-3.5,0.05",
            ]
        )
        + "\n"
    )
    return str(stable), str(csv)


def test_plot_convex_hull_quaternary_creates_plot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A valid 4-element system yields a non-empty PNG.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary directory.
    monkeypatch : pytest.MonkeyPatch
        Used to force a non-interactive matplotlib backend.
    """
    monkeypatch.setenv("MPLBACKEND", "Agg")
    from parsl_tasks.convex_hull import plot_convex_hull_quaternary

    elements = ["Si", "Ge", "Sn", "Pb"]
    stable_path, csv_path = _write_quaternary_inputs(tmp_path, elements)
    output_png = os.path.join(str(tmp_path), "quaternary.png")

    out = plot_convex_hull_quaternary(
        elements_str=elements,
        stable_path=stable_path,
        input_csv_path=csv_path,
        ehull_threshold=0.10,
        output_file=output_png,
    )

    assert out == output_png
    assert Path(output_png).exists() and Path(output_png).stat().st_size > 0


def test_plot_convex_hull_quaternary_no_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No parsable data returns the output path without writing a file.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary directory.
    monkeypatch : pytest.MonkeyPatch
        Used to force a non-interactive matplotlib backend.
    """
    monkeypatch.setenv("MPLBACKEND", "Agg")
    from parsl_tasks.convex_hull import plot_convex_hull_quaternary

    stable = tmp_path / "empty_stable.dat"
    stable.write_text("")
    csv = tmp_path / "empty.csv"
    csv.write_text("Formula,Total_Energy_per_atom,Ehull\n")

    output_png = os.path.join(str(tmp_path), "none.png")
    out = plot_convex_hull_quaternary(
        elements_str=["Si", "Ge", "Sn", "Pb"],
        stable_path=str(stable),
        input_csv_path=str(csv),
        ehull_threshold=0.10,
        output_file=output_png,
    )

    assert out == output_png
    assert not Path(output_png).exists()


def test_plot_convex_hull_quaternary_few_stable_points(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fewer than four stable points still plots results without a hull.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary directory.
    monkeypatch : pytest.MonkeyPatch
        Used to force a non-interactive matplotlib backend.
    """
    monkeypatch.setenv("MPLBACKEND", "Agg")
    from parsl_tasks.convex_hull import plot_convex_hull_quaternary

    elements = ["Si", "Ge", "Sn", "Pb"]
    stable = tmp_path / "few_stable.dat"
    stable.write_text(f"{elements[0]} -1.0\n{elements[1]} -1.5\n")

    csv = tmp_path / "few.csv"
    csv.write_text(
        f"Formula,Total_Energy_per_atom,Ehull\n{elements[0]}2{elements[1]}2,-2.0,0.02\n"
    )

    output_png = os.path.join(str(tmp_path), "few.png")
    out = plot_convex_hull_quaternary(
        elements_str=elements,
        stable_path=str(stable),
        input_csv_path=str(csv),
        ehull_threshold=0.10,
        output_file=output_png,
    )

    assert out == output_png
    assert Path(output_png).exists() and Path(output_png).stat().st_size > 0


def test_plot_convex_hull_quaternary_requires_four_elements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``ValueError`` is raised when the element count is not four.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary directory.
    monkeypatch : pytest.MonkeyPatch
        Used to force a non-interactive matplotlib backend.
    """
    monkeypatch.setenv("MPLBACKEND", "Agg")
    from parsl_tasks.convex_hull import plot_convex_hull_quaternary

    elements = ["Si", "Ge", "Sn"]
    stable = tmp_path / "s.dat"
    stable.write_text(f"{elements[0]} -1.0\n{elements[1]} -1.5\n{elements[2]} -2.0\n")
    csv = tmp_path / "r.csv"
    csv.write_text(
        "Formula,Total_Energy_per_atom,Ehull\n"
        f"{elements[0]}{elements[1]}{elements[2]},-2.0,0.02\n"
    )

    with pytest.raises(ValueError):
        plot_convex_hull_quaternary(
            elements_str=elements,
            stable_path=str(stable),
            input_csv_path=str(csv),
            ehull_threshold=0.10,
            output_file=os.path.join(str(tmp_path), "bad.png"),
        )


def _run_convex_hull_color(config: dict[str, Any]) -> Any:
    """Invoke the underlying function of the ``convex_hull_color`` python_app.

    Parameters
    ----------
    config : dict
        Configuration passed through to the wrapped function.

    Returns
    -------
    Any
        Whatever the wrapped ``convex_hull_color`` function returns.
    """
    from parsl_tasks import convex_hull

    return convex_hull.convex_hull_color.func(config)


def test_convex_hull_color_dispatch_ternary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three elements dispatch to ``plot_convex_hull_ternary``.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Used to replace the plotting functions with call recorders.
    """
    from parsl_tasks import convex_hull

    calls = {}

    def fake_ternary(elements_list, stable_dat, csv, threshold, output_file):
        calls["ternary"] = (elements_list, threshold)
        return output_file

    def fake_quaternary(*args, **kwargs):
        calls["quaternary"] = True

    monkeypatch.setattr(convex_hull, "plot_convex_hull_ternary", fake_ternary)
    monkeypatch.setattr(convex_hull, "plot_convex_hull_quaternary", fake_quaternary)

    config = {
        CK.ELEMENTS: "Na-B-C",
        CK.POST_PROCESSING_OUT_DIR: "/tmp/pp",
        CK.HULL_ENERGY_THR: "0.1",
    }

    _run_convex_hull_color(config)

    assert "ternary" in calls
    assert "quaternary" not in calls
    assert calls["ternary"][0] == ["Na", "B", "C"]
    assert calls["ternary"][1] == 0.1


def test_convex_hull_color_dispatch_quaternary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Four elements dispatch to ``plot_convex_hull_quaternary``.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Used to replace the plotting functions with call recorders.
    """
    from parsl_tasks import convex_hull

    calls = {}

    def fake_ternary(*args, **kwargs):
        calls["ternary"] = True

    def fake_quaternary(elements_list, stable_dat, csv, threshold, output_file):
        calls["quaternary"] = (elements_list, csv, threshold)
        return output_file

    monkeypatch.setattr(convex_hull, "plot_convex_hull_ternary", fake_ternary)
    monkeypatch.setattr(convex_hull, "plot_convex_hull_quaternary", fake_quaternary)

    config = {
        CK.ELEMENTS: "Si-Ge-Sn-Pb",
        CK.POST_PROCESSING_OUT_DIR: "/tmp/pp",
        CK.HULL_ENERGY_THR: "0.2",
        CK.MP_STABLE_OUT: "mp_int_stable.dat",
        CK.POST_PROCESSING_FINAL_OUT: "convex_hull.png",
    }

    _run_convex_hull_color(config)

    assert "quaternary" in calls
    assert "ternary" not in calls
    assert calls["quaternary"][0] == ["Si", "Ge", "Sn", "Pb"]
    assert calls["quaternary"][1].endswith("SiGeSnPb_quaternary.csv")
    assert calls["quaternary"][2] == 0.2


def test_convex_hull_color_reraises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Errors from the ternary plotter propagate through the dispatch.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Used to inject a failing plotting function.
    """
    from parsl_tasks import convex_hull

    called = {"ran": False}

    def boom(*args, **kwargs):
        called["ran"] = True
        raise RuntimeError("plot failed")

    monkeypatch.setattr(convex_hull, "plot_convex_hull_ternary", boom)

    config = {
        CK.ELEMENTS: "Na-B-C",
        CK.POST_PROCESSING_OUT_DIR: "/tmp/pp",
        CK.HULL_ENERGY_THR: "0.1",
        CK.MP_STABLE_OUT: "mp_int_stable.dat",
        CK.POST_PROCESSING_FINAL_OUT: "convex_hull.png",
    }

    _run_convex_hull_color(config)

    assert called["ran"], "plot_convex_hull_ternary was never called"


class _FakeMPRester:
    """Context-manager double for ``MPRester``.

    Documents are bound per instance (via the class attribute set by
    :func:`_make_fake_mprester`) rather than through shared mutable state, so
    tests remain isolated from one another.
    """

    docs: list[dict[str, Any]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.summary = mock.MagicMock()
        self.summary.search.return_value = list(type(self).docs)

    def __enter__(self) -> "_FakeMPRester":
        return self

    def __exit__(self, *exc: Any) -> Literal[False]:
        return False


def _make_fake_mprester(docs: list[dict[str, Any]]) -> type[_FakeMPRester]:
    """Build an ``MPRester`` double bound to ``docs``.

    Parameters
    ----------
    docs : list[dict]
        Documents returned by every ``summary.search`` call.

    Returns
    -------
    type[_FakeMPRester]
        A subclass whose instances return ``docs`` from ``summary.search``.
    """
    return type("_BoundFakeMPRester", (_FakeMPRester,), {"docs": list(docs)})


def test_get_stable_phases_categorizes_by_element_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Phases are labelled elementary, binary or ternary by element count."""
    from tools import post_processing

    docs = [
        {
            "material_id": "mp-1",
            "formula_pretty": "Na",
            "structure": "S1",
            "elements": ["Na"],
        },
        {
            "material_id": "mp-2",
            "formula_pretty": "NaB",
            "structure": "S2",
            "elements": ["Na", "B"],
        },
        {
            "material_id": "mp-3",
            "formula_pretty": "NaBC",
            "structure": "S3",
            "elements": ["Na", "B", "C"],
        },
    ]
    monkeypatch.setattr(post_processing, "MPRester", _make_fake_mprester(docs))

    phases = post_processing.get_stable_phases(["Na", "B", "C"], "KEY")

    types = {p["formula"]: p["phase_type"] for p in phases}
    assert types["Na"] == "elementary"
    assert types["NaB"] == "binary"
    assert types["NaBC"] == "ternary"


def test_get_stable_phases_skips_docs_without_elements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Documents lacking an ``elements`` field are ignored."""
    from tools import post_processing

    docs = [
        {
            "material_id": "mp-x",
            "formula_pretty": "?",
            "structure": "S",
            "elements": [],
        },
    ]
    monkeypatch.setattr(post_processing, "MPRester", _make_fake_mprester(docs))

    phases = post_processing.get_stable_phases(["Na"], "KEY")

    assert phases == []


def test_get_stable_phases_falls_back_to_properties(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``TypeError`` on ``fields=`` falls back to ``properties=``."""
    from tools import post_processing

    class _Rester(_FakeMPRester):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)

            def search(**kw: Any) -> list[dict[str, Any]]:
                if "fields" in kw:
                    raise TypeError("no fields kwarg")
                return [
                    {
                        "material_id": "mp-1",
                        "formula_pretty": "Na",
                        "structure": "S",
                        "elements": ["Na"],
                    },
                ]

            self.summary.search.side_effect = search

    monkeypatch.setattr(post_processing, "MPRester", _Rester)

    phases = post_processing.get_stable_phases(["Na"], "KEY")

    assert len(phases) == 1
    assert phases[0]["phase_type"] == "elementary"


class _NamedElement:
    """Element double whose ``str()`` yields its symbol.

    Parameters
    ----------
    symbol : str
        Chemical element symbol (e.g. ``"Na"``).
    """

    def __init__(self, symbol: str) -> None:
        self._symbol = symbol

    def __str__(self) -> str:
        """Return the element symbol.

        Returns
        -------
        str
            The chemical symbol.
        """
        return self._symbol


class _FakeStructure:
    """Structure double exposing ``elements`` and a ``to`` writer.

    Parameters
    ----------
    elements : list[str]
        Element symbols composing the structure.
    """

    def __init__(self, elements: list[str | _NamedElement]) -> None:
        self.elements = elements

    def to(self, filename: str) -> None:
        """Write a placeholder POSCAR file.

        Parameters
        ----------
        filename : str
            Destination path for the written file.
        """
        Path(filename).write_text("POSCAR")


class _FakeSGA:
    """SpacegroupAnalyzer double returning its input structure.

    Parameters
    ----------
    structure : _FakeStructure
        The structure passed through unchanged.
    """

    def __init__(self, structure: _FakeStructure, symprec: float) -> None:
        self._structure = structure

    def get_refined_structure(self) -> _FakeStructure:
        """Return the wrapped structure unchanged.

        Returns
        -------
        _FakeStructure
            The original structure.
        """
        return self._structure


@pytest.fixture
def hull_config(tmp_path: Path) -> dict[str, Any]:
    """Build a config and INCAR/POTCAR assets for ``get_vasp_hull``.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary directory.

    Returns
    -------
    dict
        Config dict targeting temporary work and output directories.
    """
    work = tmp_path / "work"
    out = tmp_path / "out"
    pot = tmp_path / "pot"
    write_element_potcars(pot, ("Na", "B", "Fe"))
    work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    return {
        CK.ELEMENTS: "Na-B-Fe",
        CK.MPRester_API_KEY: "KEY",
        CK.POT_DIR: str(pot),
        CK.VASP_WORK_DIR: str(work),
        CK.POST_PROCESSING_OUT_DIR: str(out),
        CK.MP_STABLE_OUT: "mp_int_stable.dat",
    }


def _patch_hull_assets(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Redirect packaged INCAR assets to temporary files.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to patch module attributes.
    tmp_path : pathlib.Path
        Directory used to store the fake INCAR files.
    """
    from tools import post_processing

    incar = tmp_path / "INCAR.en"
    incar.write_text("SYSTEM = placeholder\nENCUT = 500\n")
    incar_mag = tmp_path / "INCAR_mag.en"
    incar_mag.write_text("SYSTEM = placeholder\nISPIN = 2\n")

    def fake_path(pkg: str, name: str) -> "ResourcePathCtx":
        return ResourcePathCtx(incar_mag if "mag" in name else incar)

    monkeypatch.setattr(post_processing.pkg_resources, "path", fake_path)


def test_get_vasp_hull_returns_early_when_output_exists(
    monkeypatch: pytest.MonkeyPatch, hull_config: dict[str, Any]
) -> None:
    """No calculations run when the compiled hull already exists."""
    from tools import post_processing

    mp_file = os.path.join(hull_config[CK.VASP_WORK_DIR], hull_config[CK.MP_STABLE_OUT])
    Path(mp_file).write_text("Na -1.0\n")

    get_phases = mock.MagicMock()
    monkeypatch.setattr(post_processing, "get_stable_phases", get_phases)

    post_processing.get_vasp_hull(hull_config)

    get_phases.assert_not_called()


def test_get_vasp_hull_builds_inputs_and_compiles(
    monkeypatch: pytest.MonkeyPatch,
    hull_config: dict[str, Any],
    tmp_path: Path,
) -> None:
    """Inputs are staged, VASP is launched, and the hull is compiled."""
    from tools import post_processing

    _patch_hull_assets(monkeypatch, tmp_path)

    phases = [
        {
            "formula": "NaB",
            "structure": _FakeStructure([_NamedElement("Na"), _NamedElement("B")]),
            "elements": ["Na", "B"],
            "phase_type": "binary",
        },
        {
            "formula": "Fe",
            "structure": _FakeStructure([_NamedElement("Fe")]),
            "elements": ["Fe"],
            "phase_type": "elementary",
        },
    ]
    monkeypatch.setattr(
        post_processing,
        "get_stable_phases",
        lambda elements, api_key: phases,
    )
    monkeypatch.setattr(post_processing, "SpacegroupAnalyzer", _FakeSGA)
    monkeypatch.setattr(
        post_processing,
        "run_single_vasp_hull_calculation",
        lambda config, calc_dir: FakeFuture(),
    )

    compiled = mock.MagicMock(return_value=FakeFuture())
    monkeypatch.setattr(post_processing, "compile_vasp_hull", compiled)

    output_file = os.path.join(
        hull_config[CK.POST_PROCESSING_OUT_DIR], hull_config[CK.MP_STABLE_OUT]
    )
    Path(output_file).write_text("NaB -3.0\n")

    post_processing.get_vasp_hull(hull_config)

    calcs = Path(hull_config[CK.VASP_WORK_DIR]) / CK.SUBDIR_STABLE_PHASES / "vasp_calcs"
    calc1, calc2 = calcs / "calc_1", calcs / "calc_2"

    # Non-magnetic INCAR for NaB, magnetic INCAR for Fe.
    assert "ISPIN" not in (calc1 / "INCAR").read_text()
    assert "ISPIN = 2" in (calc2 / "INCAR").read_text()

    # SYSTEM lines rewritten to each phase formula.
    assert "SYSTEM = NaB" in (calc1 / "INCAR").read_text()
    assert "SYSTEM = Fe" in (calc2 / "INCAR").read_text()

    # POSCAR written for every calc.
    assert (calc1 / "POSCAR").exists()
    assert (calc2 / "POSCAR").exists()

    # POTCAR is the concatenation of per-element POTCARs.
    assert (calc1 / "POTCAR").read_text() == "POTCAR-Na\nPOTCAR-B\n"
    assert (calc2 / "POTCAR").read_text() == "POTCAR-Fe\n"

    # Hull compilation invoked with the full structure count.
    compiled.assert_called_once()
    args, _ = compiled.call_args
    assert args[0] == len(phases)

    # Compiled hull copied into the work directory.
    mp_file = os.path.join(hull_config[CK.VASP_WORK_DIR], hull_config[CK.MP_STABLE_OUT])
    assert Path(mp_file).exists()


def test_calculate_ehull_gather_energy_branch(
    ehull_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ``gather_energy=True`` path runs the grep/sed subprocess command.

    Parameters
    ----------
    ehull_env : dict
        Fixture providing the extracted directory and config.
    monkeypatch : pytest.MonkeyPatch
        Used to intercept the subprocess call.
    """
    from parsl_tasks import ehull as ehull_mod
    from parsl_tasks.ehull import cmd_calculate_ehull

    ehull_dir = ehull_env["ehull_dir"]
    config = ehull_env["config"]

    captured: dict[str, Any] = {}

    class _Result:
        returncode = 0

    def fake_run(cmd: str, shell: bool = False):  # noqa: ANN001
        captured["cmd"] = cmd
        captured["shell"] = shell
        # Recreate a valid energy file so downstream logic still works.
        src = os.path.join(str(ehull_dir), config[CK.ENERGY_DAT_OUT])
        with open(src) as f:
            data = f.read()
        with open(src, "w") as f:
            f.write(data)
        return _Result()

    monkeypatch.setattr(ehull_mod.subprocess, "run", fake_run)

    out_path = Path(cmd_calculate_ehull(config, gather_energy=True))

    assert "grep 'F='" in captured["cmd"]
    assert captured["shell"] is True
    assert out_path == ehull_dir / "hull.dat"


def test_calculate_ehull_gather_energy_failure_logs_critical(
    ehull_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-zero return code from the gather command logs a critical message.

    Parameters
    ----------
    ehull_env : dict
        Fixture providing the extracted directory and config.
    monkeypatch : pytest.MonkeyPatch
        Used to intercept the subprocess call and logger.
    """
    from parsl_tasks import ehull as ehull_mod
    from parsl_tasks.ehull import cmd_calculate_ehull

    config = ehull_env["config"]

    class _Result:
        returncode = 1

    monkeypatch.setattr(ehull_mod.subprocess, "run", lambda cmd, shell=False: _Result())

    critical = mock.MagicMock()
    monkeypatch.setattr(ehull_mod.amd_logger, "critical", critical)

    cmd_calculate_ehull(config, gather_energy=True)

    assert critical.called


def test_calculate_ehull_mp_stable_missing_logs_critical(
    ehull_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A missing mp_stable_file (in both locations) logs a critical message.

    Parameters
    ----------
    ehull_env : dict
        Fixture providing the extracted directory and config.
    monkeypatch : pytest.MonkeyPatch
        Used to intercept the logger.
    tmp_path : pathlib.Path
        Pytest-provided temporary directory.
    """
    from parsl_tasks import ehull as ehull_mod
    from parsl_tasks.ehull import cmd_calculate_ehull

    work = tmp_path / "work"
    out = tmp_path / "out"
    work.mkdir()
    out.mkdir()

    # Provide an empty energy file so read_energies has something to read.
    (work / "energy.dat").write_text("")

    config = {
        CK.ELEMENTS: "Na-B-C",
        CK.VASP_WORK_DIR: str(work),
        CK.ENERGY_DAT_OUT: "energy.dat",
        CK.POST_PROCESSING_OUT_DIR: str(out),
        CK.MP_STABLE_OUT: "does_not_exist.dat",
    }

    critical = mock.MagicMock()
    monkeypatch.setattr(ehull_mod.amd_logger, "critical", critical)

    # Source logs a critical message then continues; the subsequent
    # parse step raises because the file is genuinely absent.
    with pytest.raises(FileNotFoundError):
        cmd_calculate_ehull(config, gather_energy=False)

    assert critical.called


@pytest.fixture(scope="module")
def ehull_quaternary_env(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, Any]:
    """Reuse the ternary fixture archive but drive the quaternary branch.

    Parameters
    ----------
    tmp_path_factory : pytest.TempPathFactory
        Factory used to create a module-scoped temporary directory.

    Returns
    -------
    dict
        Mapping with ``ehull_dir`` and a 4-element ``config``.
    """
    tmp = tmp_path_factory.mktemp("ehull_quaternary_fixture")
    tar_path = Path(__file__).parent / "post_processing.tar"

    extract_tar(tar_path, tmp)

    ehull_dir = tmp / "post_processing"

    config = {
        CK.ELEMENTS: "Na-B-C-Fe",
        CK.VASP_WORK_DIR: str(ehull_dir),
        CK.ENERGY_DAT_OUT: "energy.dat",
        CK.POST_PROCESSING_OUT_DIR: str(ehull_dir),
        CK.MP_STABLE_OUT: "mp_int_stable.dat",
    }
    return {"ehull_dir": ehull_dir, "config": config}


def test_calculate_ehull_quaternary_branch(
    ehull_quaternary_env: dict[str, Any],
) -> None:
    """The quaternary path returns the hull path (or early-exits gracefully).

    Parameters
    ----------
    ehull_quaternary_env : dict
        Fixture providing the extracted directory and 4-element config.
    """
    from parsl_tasks.ehull import cmd_calculate_ehull

    ehull_dir = ehull_quaternary_env["ehull_dir"]
    config = ehull_quaternary_env["config"]

    out_path = Path(cmd_calculate_ehull(config, gather_energy=False))

    # The quaternary branch always returns the hull.dat path.
    assert out_path == ehull_dir / "hull.dat"
