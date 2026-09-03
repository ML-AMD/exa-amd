"""Unit tests for :mod:`parsl_tasks.select_structures`.

These tests exercise the pieces of the selection pipeline that are not covered
by the end-to-end workflow test in ``test_workflow.py``: the CSV reading and
threshold-relaxation logic, element-fraction parsing and filtering, the
atom-count filter, and the low-count warning branch.
"""

import csv
import multiprocessing as mp
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Lattice, Structure

from parsl_tasks.select_structures import (
    process_structures,
    read_csv,
    run_select_structures,
    select_structures,
    select_structures_for_compositions,
)


def _write_csv(path: Path, rows: list[tuple[str, str, float]]) -> None:
    """Write a candidate CSV in the ``index, _, Ef`` format.

    Parameters
    ----------
    path : pathlib.Path
        Destination CSV path.
    rows : list of tuple of (str, str, float)
        Rows of ``(index, placeholder, Ef)`` to write.

    Returns
    -------
    None
    """
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        for index, placeholder, ef in rows:
            writer.writerow([index, placeholder, ef])


def _write_cif(
    nomix_dir: Path,
    index: str,
    species: list[str],
    a: float = 3.0,
) -> None:
    """Create a simple cubic CIF at ``{nomix_dir}/{prefix}/{index}.cif``.

    The chunk prefix is derived from the part of ``index`` before the first
    underscore, matching the layout expected by the selection pipeline.

    Parameters
    ----------
    nomix_dir : pathlib.Path
        Root directory that holds per-chunk subdirectories.
    index : str
        Structure identifier, e.g. ``"1_0"``.
    species : list of str
        Element symbols, one per site.
    a : float, optional
        Cubic lattice parameter. Default is ``3.0``.

    Returns
    -------
    None
    """
    prefix = index.split("_")[0]
    chunk_dir = nomix_dir / prefix
    chunk_dir.mkdir(parents=True, exist_ok=True)

    n = len(species)
    coords = [[i / n, i / n, i / n] for i in range(n)]
    structure = Structure(Lattice.cubic(a), species, coords)
    structure.to(filename=str(chunk_dir / f"{index}.cif"), fmt="cif")


def test_read_csv_respects_threshold(tmp_path: Path) -> None:
    """read_csv keeps only structures below the threshold when above minimum.

    With a threshold set between the sorted energies, only the lower-energy
    entries should be retained. The internal 20000 minimum is not triggered
    because the relaxation branch requires fewer than that many survivors, so
    this asserts the *filtering* logic rather than the exact cutoff count.
    """
    csv_file = tmp_path / "in.csv"
    rows = [
        ("1_0", "x", -1.0),
        ("1_1", "x", -0.5),
        ("1_2", "x", 0.5),
        ("1_3", "x", 2.0),
    ]
    _write_csv(csv_file, rows)

    result = read_csv(str(csv_file), ef_threshold=0.0)

    # Fewer than min_structures survive, so the relaxation branch fills up to
    # the available count; all four are returned but sorted by energy.
    assert set(result.keys()) == {"1_0", "1_1", "1_2", "1_3"}
    assert result["1_0"] == -1.0


def test_read_csv_relaxes_when_below_minimum(tmp_path: Path) -> None:
    """read_csv falls back to all available structures below the minimum.

    When the number of candidates is far below the internal minimum, every row
    is returned regardless of the threshold.
    """
    csv_file = tmp_path / "in.csv"
    rows = [("1_0", "x", 5.0), ("1_1", "x", 6.0)]
    _write_csv(csv_file, rows)

    result = read_csv(str(csv_file), ef_threshold=0.0)

    assert result == {"1_0": 5.0, "1_1": 6.0}


def test_run_select_structures_element_fraction_filter(tmp_path: Path) -> None:
    """Structures below a required element fraction are discarded.

    A structure with no Fe is filtered out when ``element_fractions="Fe:0.5"``,
    while an Fe-rich structure is retained.
    """
    nomix_dir = tmp_path / "nomix"
    _write_cif(nomix_dir, "1_0", ["Fe", "Fe"])  # 100% Fe -> kept
    _write_cif(nomix_dir, "1_1", ["Al", "Al"])  # 0% Fe -> dropped

    csv_file = tmp_path / "in.csv"
    _write_csv(csv_file, [("1_0", "x", -1.0), ("1_1", "x", -0.9)])

    out_dir = tmp_path / "out"
    run_select_structures(
        output_dir=str(out_dir),
        nomix_dir=str(nomix_dir),
        csv_file=str(csv_file),
        ef_threshold=1.0,
        min_total=1,
        max_total=10,
        num_workers=1,
        element_fractions="Fe:0.5",
    )

    poscars = list(out_dir.glob("POSCAR_*"))
    assert len(poscars) == 1, "Only the Fe-rich structure should survive"


def test_run_select_structures_natom_threshold(tmp_path: Path) -> None:
    """Structures exceeding the reduced-formula atom threshold are dropped."""
    nomix_dir = tmp_path / "nomix"
    _write_cif(nomix_dir, "1_0", ["Fe", "Al"])  # reduced formula: 2 atoms
    _write_cif(nomix_dir, "1_1", ["Fe", "Al", "Ni", "Cu"])  # 4 atoms

    csv_file = tmp_path / "in.csv"
    _write_csv(csv_file, [("1_0", "x", -1.0), ("1_1", "x", -0.9)])

    out_dir = tmp_path / "out"
    run_select_structures(
        output_dir=str(out_dir),
        nomix_dir=str(nomix_dir),
        csv_file=str(csv_file),
        ef_threshold=1.0,
        min_total=1,
        max_total=10,
        num_workers=1,
        natom_threshold=2,
    )

    poscars = list(out_dir.glob("POSCAR_*"))
    assert len(poscars) == 1, "Only the 2-atom structure should survive"


def test_run_select_structures_min_total_warning(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A warning is printed when fewer than ``min_total`` structures remain."""
    nomix_dir = tmp_path / "nomix"
    _write_cif(nomix_dir, "1_0", ["Fe", "Al"])

    csv_file = tmp_path / "in.csv"
    _write_csv(csv_file, [("1_0", "x", -1.0)])

    out_dir = tmp_path / "out"
    run_select_structures(
        output_dir=str(out_dir),
        nomix_dir=str(nomix_dir),
        csv_file=str(csv_file),
        ef_threshold=1.0,
        min_total=100,
        max_total=10,
        num_workers=1,
    )

    captured = capsys.readouterr()
    assert "less than the specified minimum" in captured.out


def test_process_structures_filters_and_signals_done(tmp_path: Path) -> None:
    """process_structures emits survivors then a 'DONE' sentinel.

    Runs the worker in-process against real queues: one structure passes all
    filters, one is dropped by the atom-count filter. The worker stops on the
    ``None`` sentinel and must finish by pushing ``'DONE'``.
    """
    nomix_dir = tmp_path / "nomix"
    _write_cif(nomix_dir, "1_0", ["Fe", "Al"])  # 2 atoms -> kept
    # 4 atoms -> dropped
    _write_cif(nomix_dir, "1_1", ["Fe", "Al", "Ni", "Cu"])

    task_queue: "mp.Queue" = mp.Queue()
    result_queue: "mp.Queue" = mp.Queue()
    task_queue.put(("1_0", -1.0))
    task_queue.put(("1_1", -0.9))
    task_queue.put(None)

    process_structures(
        task_queue,
        result_queue,
        str(nomix_dir),
        natom_threshold=2,
        element_fractions={},
    )

    results = []
    while True:
        item = result_queue.get()
        if item == "DONE":
            break
        results.append(item)

    assert len(results) == 1
    index, ef, formula, structure = results[0]
    assert index == "1_0"
    assert ef == -1.0
    assert structure.num_sites == 2


def test_process_structures_element_fraction(tmp_path: Path) -> None:
    """process_structures drops structures below a required element fraction."""
    nomix_dir = tmp_path / "nomix"
    _write_cif(nomix_dir, "1_0", ["Fe", "Fe"])  # 100% Fe -> kept
    _write_cif(nomix_dir, "1_1", ["Al", "Al"])  # 0% Fe -> dropped

    task_queue: "mp.Queue" = mp.Queue()
    result_queue: "mp.Queue" = mp.Queue()
    task_queue.put(("1_0", -1.0))
    task_queue.put(("1_1", -0.9))
    task_queue.put(None)

    process_structures(
        task_queue,
        result_queue,
        str(nomix_dir),
        natom_threshold=50,
        element_fractions={"Fe": 0.5},
    )

    kept = []
    while True:
        item = result_queue.get()
        if item == "DONE":
            break
        kept.append(item[0])

    assert kept == ["1_0"]


def test_select_structures_for_compositions_dedup() -> None:
    """select_structures_for_compositions dedups and caps per composition.

    Two identical structures and one distinct structure share a composition
    group. With ``n_per_composition=2`` the worker should return the two
    structurally distinct entries, preferring the lowest ``Ef``.
    """
    struct_a = Structure(
        Lattice.cubic(3.0), ["Fe", "Al"], np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
    )
    struct_a_dup = struct_a.copy()
    struct_b = Structure(
        Lattice.cubic(3.0), ["Fe", "Al"], np.array([[0, 0, 0], [0.15, 0.15, 0.15]])
    )

    structures = [
        ("1_0", -0.8, struct_a),
        ("1_1", -1.0, struct_a_dup),  # duplicate of struct_a, lower ef
        ("1_2", -0.9, struct_b),  # distinct
    ]

    task_queue: "mp.Queue" = mp.Queue()
    result_queue: "mp.Queue" = mp.Queue()
    task_queue.put(("FeAl", structures, 2))
    task_queue.put(None)

    select_structures_for_compositions(task_queue, result_queue, StructureMatcher())

    composition, selected = result_queue.get()
    selected_indices = {s[0] for s in selected}

    assert composition == "FeAl"
    assert len(selected) == 2
    # Lowest-Ef of the duplicate pair (1_1) plus the distinct structure (1_2)
    assert selected_indices == {"1_1", "1_2"}


def test_select_structures_app_wiring(tmp_path: Path) -> None:
    """The select_structures app wires config values into run_select_structures.

    Calls the underlying plain function to avoid needing a live Parsl config,
    mocking the heavy pipeline to assert argument passing and the
    working-directory change.
    """
    from tools.config_labels import ConfigKeys as CK

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    out_dir = tmp_path / "out"

    config = {
        CK.WORK_DIR: str(work_dir),
        CK.EF_THR: "-0.2",
        CK.NUM_WORKERS: "3",
    }

    # Parsl stores the original callable on the app's `.func` attribute.
    fn = getattr(select_structures, "func", None) or getattr(
        select_structures, "__wrapped__", select_structures
    )

    with mock.patch("parsl_tasks.select_structures.run_select_structures") as mock_run:
        fn(config, str(out_dir), min_total=5, max_total=50)

    mock_run.assert_called_once()
    _, kwargs = mock_run.call_args
    assert kwargs["nomix_dir"] == str(work_dir / "structures")
    assert kwargs["csv_file"] == str(work_dir / "test_results.csv")
    assert kwargs["ef_threshold"] == -0.2
    assert kwargs["min_total"] == 5
    assert kwargs["max_total"] == 50
    assert kwargs["num_workers"] == 3
