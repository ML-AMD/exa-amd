"""Tests for :mod:`parsl_tasks.gen_structures`.

These tests exercise structure generation via element substitution, the
disallowed-element skipping logic, per-structure CIF writing, and the
chunk-level entry point that produces the ``id_prop.csv`` manifest alongside
the generated CIF files.
"""

import os
import sys
import tarfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="module")
def gen_env(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Extract the test archive and locate the CIF input directory.

    Parameters
    ----------
    tmp_path_factory : pytest.TempPathFactory
        Factory used to create the module-scoped temporary directory.

    Returns
    -------
    dict of str to pathlib.Path
        Mapping with keys ``"tmp"`` (the extraction root) and ``"input_dir"``
        (the directory containing the CIF file).
    """
    tmp = tmp_path_factory.mktemp("gen_structures")
    tar_path = Path(__file__).parent / "initial_structures_in.tar"
    assert tar_path.exists(), f"Missing {tar_path}"

    with tarfile.open(tar_path) as tar:
        try:
            tar.extractall(path=tmp, filter="data")  # Python 3.12+
        except TypeError:
            tar.extractall(path=tmp)

    input_dir = None
    for cand in [tmp, *tmp.iterdir()]:
        if cand.is_dir() and any(p.suffix == ".cif" for p in cand.iterdir()):
            input_dir = cand
            break
    assert input_dir is not None, "Could not find directory containing CIF files"

    # sanity: we provide a single initial structure
    cif_files = [p for p in input_dir.iterdir() if p.suffix == ".cif"]
    assert len(cif_files) == 1, f"Expected 1 CIF, found {len(cif_files)}"

    return {"tmp": tmp, "input_dir": input_dir}


def test_generate_structures(gen_env: dict[str, Path]) -> None:
    """Verify structure count and element substitution.

    Parameters
    ----------
    gen_env : dict of str to pathlib.Path
        Fixture providing the extracted input directory.

    Notes
    -----
    Asserts the number of generated structures equals
    ``len(permutations(elements)) * len(LATTICE_SCALES)`` and that each
    structure only contains the substituted elements.
    """
    from parsl_tasks.gen_structures import LATTICE_SCALES, _generate_structures

    input_dir = gen_env["input_dir"]
    cif_files = [p for p in input_dir.iterdir() if p.suffix == ".cif"]
    structure_file = cif_files[0].name
    elements = ["Na", "B", "C"]

    from itertools import permutations

    expected = len(list(permutations(elements))) * len(LATTICE_SCALES)

    structures = _generate_structures(structure_file, elements, str(input_dir))

    assert len(structures) == expected, (
        f"Expected {expected} structures, got {len(structures)}"
    )

    # each generated structure only contains substituted elements
    allowed = set(elements)
    for s in structures:
        symbols = {el.symbol for el in s.composition}
        assert symbols.issubset(allowed), f"Unexpected elements {symbols - allowed}"


def test_generate_structures_skips_bad_elements(
    gen_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure structures with disallowed elements are skipped.

    Parameters
    ----------
    gen_env : dict of str to pathlib.Path
        Fixture providing the extracted input directory.
    tmp_path : pathlib.Path
        Per-test temporary directory.
    monkeypatch : pytest.MonkeyPatch
        Used to inject a present element into ``badele_vec``.

    Notes
    -----
    Patches :data:`~parsl_tasks.gen_structures.badele_vec` so one of the
    structure's elements is disallowed and asserts an empty result.
    """
    import parsl_tasks.gen_structures as gs
    from parsl_tasks.gen_structures import _generate_structures

    input_dir = gen_env["input_dir"]
    cif_files = [p for p in input_dir.iterdir() if p.suffix == ".cif"]
    structure_file = cif_files[0].name

    # Force one of the present elements to be treated as disallowed
    from pymatgen.core import Structure

    original = Structure.from_file(str(cif_files[0]))
    present = [el.symbol for el in original.composition]

    monkeypatch.setattr(gs, "badele_vec", gs.badele_vec + present[:1])

    structures = _generate_structures(structure_file, ["Na", "B", "C"], str(input_dir))
    assert structures == [], "Structures with disallowed elements must be skipped"


def test_process_structure(gen_env: dict[str, Path], tmp_path: Path) -> None:
    """Verify CIF files are written with the expected names and count.

    Parameters
    ----------
    gen_env : dict of str to pathlib.Path
        Fixture providing the extracted input directory.
    tmp_path : pathlib.Path
        Per-test temporary directory used as the CIF output location.

    Notes
    -----
    Asserts the returned count, the number of written CIF files, and that the
    generated file names match ``<chunk_id>_<index>.cif``.
    """
    from itertools import permutations

    from parsl_tasks.gen_structures import LATTICE_SCALES, _process_structure

    input_dir = gen_env["input_dir"]
    cif_files = [p for p in input_dir.iterdir() if p.suffix == ".cif"]
    structure_file = cif_files[0].name
    elements = ["Na", "B", "C"]
    chunk_id = 1
    start_index = 1

    expected = len(list(permutations(elements))) * len(LATTICE_SCALES)

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    cwd = os.getcwd()
    try:
        os.chdir(out_dir)
        count = _process_structure(
            (structure_file, start_index, str(input_dir), elements, chunk_id)
        )
    finally:
        os.chdir(cwd)

    assert count == expected, f"Expected {expected} structures, got {count}"

    written = sorted(out_dir.glob(f"{chunk_id}_*.cif"))
    assert len(written) == expected, (
        f"Expected {expected} CIF files, found {len(written)}"
    )

    expected_names = {
        f"{chunk_id}_{i}.cif" for i in range(start_index, start_index + expected)
    }
    assert {p.name for p in written} == expected_names


def test_run_gen_structures(
    gen_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify the chunk-level entry point end to end.

    Parameters
    ----------
    gen_env : dict of str to pathlib.Path
        Fixture providing the extracted input directory.
    monkeypatch : pytest.MonkeyPatch
        Available for patching if needed.

    Notes
    -----
    Runs :func:`~parsl_tasks.gen_structures.run_gen_structures` for a single
    chunk and asserts that ``id_prop.csv`` and the generated CIF files agree in
    both count (30 rows) and identifiers.
    """
    from parsl_tasks.gen_structures import run_gen_structures
    from tools.config_labels import ConfigKeys as CK

    work_dir = gen_env["tmp"] / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    config = {
        CK.WORK_DIR: str(work_dir),
        CK.INITIAL_STRS: str(gen_env["input_dir"]),
        CK.NUM_WORKERS: 1,
        CK.ELEMENTS: "Na-B-C",
    }

    n_chunks = 1
    chunk_id = 1

    cwd = os.getcwd()
    try:
        os.chdir(work_dir)
        out_csv = run_gen_structures(config, n_chunks=n_chunks, chunk_id=chunk_id)
    finally:
        os.chdir(cwd)

    out_csv = Path(out_csv)
    assert out_csv.exists(), "id_prop.csv was not created"

    # id_prop.csv should contain 30 lines: "<chunk>_<idx>,0.5"
    lines = [ln.strip() for ln in out_csv.read_text().splitlines() if ln.strip()]
    assert len(lines) == 30, f"Expected 30 rows in csv, found {len(lines)}"
    assert all("," in ln for ln in lines), "Malformed csv rows"

    # verify generated CIF files
    out_dir = work_dir / "structures" / str(chunk_id)
    assert out_dir.exists(), "Output structures directory missing"

    cif_files = sorted(out_dir.glob(f"{chunk_id}_*.cif"))
    assert len(cif_files) == 30, f"Expected 30 CIF files, found {len(cif_files)}"

    # compare ids in csv vs CIF filenames
    csv_ids = {ln.split(",")[0] for ln in lines}
    cif_ids = {p.stem for p in cif_files}
    assert csv_ids == cif_ids, "csv ids do not match generated CIF filenames"
