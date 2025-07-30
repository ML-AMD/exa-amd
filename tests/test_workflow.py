import pytest
import tarfile
import subprocess
import os
from pathlib import Path


@pytest.fixture(scope="module")
def cgcnn_output(tmp_path_factory):
    """
    runs the cgcnn prediction script and returns paths for follow-up tests.
    """
    tmp_path = tmp_path_factory.mktemp("cgcnn_test")
    archive_path = Path(__file__).parent / "test_structures.tar"
    test_structures_dir = tmp_path / "test_structures"
    model_path = Path(__file__).parent.parent / "cms_dir/form_1st.pth.tar"
    predict_script_path = Path(__file__).parent.parent / "cms_dir/predict.py"
    cgcnn_output_csv = tmp_path / "test_results_1.csv"

    # extract test_structures.tar
    with tarfile.open(archive_path) as tar:
        tar.extractall(path=tmp_path, filter="data")

    assert test_structures_dir.exists(), "Extraction failed"

    # Run predict.py
    result = subprocess.run(
        [
            "python", str(predict_script_path),
            str(model_path),
            os.path.join(str(test_structures_dir), "1"),
            "--batch-size", "256"
        ],
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    print("# STDOUT:", result.stdout)
    print("# STDERR:", result.stderr)
    assert result.returncode == 0, f"predict.py failed\n{result.stderr}"
    assert cgcnn_output_csv.exists(), "test_results.csv not created"

    return {
        "csv": cgcnn_output_csv,
        "structures": test_structures_dir,
        "base": tmp_path
    }


def test_cgcnn(cgcnn_output):
    """
    test that test_results.csv has correct format
    """
    cgcnn_output_csv = cgcnn_output["csv"]
    with open(cgcnn_output_csv) as f:
        lines = f.readlines()
        assert all(len(line.strip().split(",")) ==
                   3 for line in lines), "Unexpected line in CSV"


def test_select_structure(cgcnn_output):
    """
    test select structures
    """
    select_script = Path(__file__).parent.parent / \
        "cms_dir/select_structure.py"
    output_dir = cgcnn_output["base"]
    cgcnn_output_csv = cgcnn_output["csv"]
    structures_dir = cgcnn_output["structures"]
    new_dir = output_dir / "new"

    result = subprocess.run(
        [
            "python", str(select_script),
            "--ef_threshold", "-0.2",
            "--csv_file", str(cgcnn_output_csv),
            "--nomix_dir", str(structures_dir)
        ],
        cwd=output_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    print("# STDOUT:", result.stdout)
    print("# STDERR:", result.stderr)
    assert result.returncode == 0, "select_structure.py failed"

    # check the existence of the 'Selected N structures' message
    assert "Selected" in result.stdout
    assert "structures" in result.stdout
    selected_count = None
    for line in result.stdout.splitlines():
        if line.strip().startswith("Selected"):
            selected_count = int(line.strip().split()[1])
            break

    assert selected_count is not None, "Fid not find 'Selected N structures' message"

    # check that the directory "./new" was created
    assert new_dir.exists(), "'new' directory not created"
    poscars = list(new_dir.glob("POSCAR_*"))
    assert len(poscars) == selected_count, f"Expected {selected_count} POSCAR files, found {
        len(poscars)}"

    id_prop = new_dir / "id_prop.csv"
    assert id_prop.exists(), "id_prop.csv not found in new directory"
