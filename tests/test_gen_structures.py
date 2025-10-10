import os
import tarfile
from pathlib import Path
import sys
import pytest

REPO_ROOT = Path(__file__).parent.parent.resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="module")
def gen_env(tmp_path_factory):
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


def test_run_gen_structures(gen_env, monkeypatch):
    from tools.config_labels import ConfigKeys as CK
    from parsl_tasks.gen_structures import run_gen_structures

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
