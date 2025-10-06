import sys
import pytest
import tarfile
import os
import shutil
from pathlib import Path

# Ensure repo root is importable
REPO_ROOT = Path(__file__).parent.parent.resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="module")
def cgcnn_output(tmp_path_factory):
    """
    runs the cgcnn prediction and returns paths for follow-up tests.
    """
    tmp_path = tmp_path_factory.mktemp("cgcnn_test")
    archive_path = Path(__file__).parent / "test_structures.tar"
    test_structures_dir = tmp_path / "test_structures"
    pkg_root = REPO_ROOT / "ml_models" / "cgcnn"
    model_path = pkg_root / "form_1st.pth.tar"
    atom_init_src = pkg_root / "atom_init.json"
    cgcnn_output_csv = tmp_path / "test_results_1.csv"

    # extract test_structures.tar
    with tarfile.open(archive_path) as tar:
        try:
            tar.extractall(path=tmp_path, filter="data")  # Python 3.12+
        except TypeError:
            tar.extractall(path=tmp_path)

    assert test_structures_dir.exists(), "Extraction failed"

    # ensure atom_init.json is in the correct place
    cif_dir = test_structures_dir / "1"
    atom_init_dst = cif_dir / "atom_init.json"
    if not atom_init_dst.exists():
        shutil.copyfile(atom_init_src, atom_init_dst)

    from ml_models.cgcnn.predict import predict_cgcnn

    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        out_csv = predict_cgcnn(
            modelpath=str(model_path),
            cifpath=str(cif_dir),
            batch_size=256,
            workers=0,
            chunk_id=1,
            output_csv=None,
        )
    finally:
        os.chdir(cwd)

    assert Path(out_csv) == cgcnn_output_csv, "Output CSV path mismatch"
    assert cgcnn_output_csv.exists(), "test_results_1.csv not created"

    return {
        "csv": cgcnn_output_csv,
        "structures": test_structures_dir,
        "base": tmp_path
    }


def test_cgcnn(cgcnn_output):
    """
    test that test_results_1.csv has correct format
    """
    cgcnn_output_csv = cgcnn_output["csv"]
    with open(cgcnn_output_csv) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    assert lines, "CSV is empty"
    assert all(len(line.split(",")) == 3 for line in lines), "Unexpected line in CSV"


def test_select_structure(cgcnn_output):
    """
    test select structures using the callable (no subprocess, no cms_dir)
    """
    from parsl_tasks.select_structures import run_select_structures

    output_dir = cgcnn_output["base"]
    cgcnn_output_csv = cgcnn_output["csv"]
    structures_dir = cgcnn_output["structures"]
    new_dir = output_dir / "new"

    run_select_structures(
        nomix_dir=str(structures_dir),
        output_dir=str(new_dir),
        csv_file=str(cgcnn_output_csv),
        ef_threshold=-0.2,
        num_workers=1,
    )

    assert new_dir.exists(), "'new' directory not created"
    poscars = list(new_dir.glob("POSCAR_*"))
    assert len(poscars) > 0, "No POSCAR_* files produced"

    id_prop = new_dir / "id_prop.csv"
    assert id_prop.exists(), "id_prop.csv not found in new directory"
