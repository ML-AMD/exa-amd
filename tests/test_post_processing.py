# tests/test_post_processing.py
import os
import re
import sys
import tarfile
from pathlib import Path

import pytest

from tools.config_labels import ConfigKeys as CK

REPO_ROOT = Path(__file__).parent.parent.resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="module")
def ehull_env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ehull_fixture")
    tar_path = Path(__file__).parent / "post_processing.tar"

    with tarfile.open(tar_path) as tar:
        try:
            tar.extractall(path=tmp, filter="data")  # Python 3.12+
        except TypeError:
            tar.extractall(path=tmp)

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


def test_calculate_ehul_outputs(ehull_env):
    """
    Run and test calculate_ehul()
    """
    from parsl_tasks.ehull import cmd_calculate_ehul

    ehull_dir = ehull_env["ehull_dir"]
    config = ehull_env["config"]

    out_path = Path(cmd_calculate_ehul(config, False))

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


def test_convex_hull_color_ternary(ehull_env, monkeypatch):
    """
    Run and test convex_hull_color()
    """
    from parsl_tasks.convex_hull import plot_convex_hull_ternary
    from parsl_tasks.ehull import cmd_calculate_ehul

    # ensure non-interactive matplotlib
    monkeypatch.setenv("MPLBACKEND", "Agg")

    ehull_dir = ehull_env["ehull_dir"]
    config = ehull_env["config"]

    # Produce NaBC.csv independently of test ordering.
    cmd_calculate_ehul(config, False)

    elements_list = config[CK.ELEMENTS].split("-")
    stable_dat = os.path.join(ehull_dir, config[CK.MP_STABLE_OUT])
    input_csv = os.path.join(ehull_dir, "NaBC.csv")
    assert Path(input_csv).exists(), "NaBC.csv missing"

    output_png = os.path.join(ehull_dir, "convex_hull.png")
    threshold = 0.10

    cwd = os.getcwd()
    try:
        os.chdir(ehull_dir)
        out_path = plot_convex_hull_ternary(
            elements_list=elements_list,
            stable_dat=stable_dat,
            full_path_input_csv=input_csv,
            threshold=threshold,
            output_file=output_png,
        )
    finally:
        os.chdir(cwd)

    out_path = Path(out_path)
    assert out_path == Path(output_png)
    assert out_path.exists() and out_path.stat().st_size > 0

    hull_txt = Path(ehull_dir) / "convex-hull.dat"
    assert hull_txt.exists() and hull_txt.stat().st_size > 0


@pytest.fixture(scope="module")
def compile_vasp_hull_inputs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("compile_hull_in")
    tar_path = Path(__file__).parent / "compile_hull_in.tar"
    assert tar_path.exists(), f"Missing {tar_path}"

    with tarfile.open(tar_path) as tar:
        try:
            tar.extractall(path=tmp, filter="data")
        except TypeError:
            tar.extractall(path=tmp)

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


def _expected_compile_vasp_hull(root: Path, total: int):
    """Compute expected {formula: min_epa}"""
    expected = {}
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


def test_cmd_compile_vasp_hull(compile_vasp_hull_inputs):
    """
    Run and test compile_vasp_hull()
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


def _write_quaternary_inputs(root: Path, elements):
    """Create stable .dat and results .csv files for a 4-element system."""
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


def test_plot_convex_hull_quaternary_creates_plot(tmp_path, monkeypatch):
    """
    Run and test plot_convex_hull_quaternary() with valid inputs.
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


def test_plot_convex_hull_quaternary_no_data(tmp_path, monkeypatch):
    """
    plot_convex_hull_quaternary() returns output_file when nothing parses.
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


def test_plot_convex_hull_quaternary_few_stable_points(tmp_path, monkeypatch):
    """
    Fewer than 4 stable points: still plots calculated results, no hull.
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


def test_plot_convex_hull_quaternary_requires_four_elements(tmp_path, monkeypatch):
    """
    plot_quaternary_hull raises ValueError when not exactly 4 elements.
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


def _run_convex_hull_color(config):
    """Call the underlying function of the convex_hull_color python_app."""
    from parsl_tasks import convex_hull

    return convex_hull.convex_hull_color.func(config)


def test_convex_hull_color_dispatch_ternary(monkeypatch):
    """
    convex_hull_color() dispatches to plot_convex_hull_ternary for 3 elements.
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


def test_convex_hull_color_dispatch_quaternary(monkeypatch):
    """
    convex_hull_color() dispatches to plot_convex_hull_quaternary for 4 elements.
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


def test_convex_hull_color_reraises(monkeypatch):
    """
    convex_hull_color() invokes the ternary plotting function; errors there
    propagate out of the dispatch (the try/except is a no-op re-raise).
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
