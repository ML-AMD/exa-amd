"""End-to-end workflow tests for the CGCNN prediction and structure selection.

These tests run :func:`ml_models.cgcnn.predict.predict_cgcnn` on a bundled set
of test structures, verify that predictions are reproducible across repeated
runs, and exercise :func:`parsl_tasks.select_structures.run_select_structures`
on the resulting predictions CSV.
"""

import os
import shutil
import sys
import tarfile
from pathlib import Path

import pytest

# Ensure repo root is importable
REPO_ROOT = Path(__file__).parent.parent.resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="module")
def cgcnn_output(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, Path]:
    """Run the CGCNN prediction and return paths for follow-up tests.

    Parameters
    ----------
    tmp_path_factory : pytest.TempPathFactory
        Pytest factory for module-scoped temporary directories.

    Returns
    -------
    dict of str to pathlib.Path
        A mapping with keys ``"csv"`` (predictions CSV path), ``"structures"``
        (extracted structures directory), and ``"base"`` (working directory).
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
        "base": tmp_path,
    }


def test_cgcnn_reproducible_predictions(
    cgcnn_output: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Predictions are stable across repeated ``predict_cgcnn`` runs.

    Parameters
    ----------
    cgcnn_output : dict of str to pathlib.Path
        Fixture providing the base directory and extracted structures.
    tmp_path : pathlib.Path
        Per-test temporary directory.
    monkeypatch : pytest.MonkeyPatch
        Available for patching if needed.

    Notes
    -----
    Runs :func:`predict_cgcnn` five times and asserts the per-structure
    predictions agree to within ``1e-6``.
    """
    import os
    from pathlib import Path

    import ml_models.cgcnn as cgcnn_pkg
    from ml_models.cgcnn.predict import predict_cgcnn

    base = cgcnn_output["base"]
    cif_dir = cgcnn_output["structures"] / "1"
    model_path = Path(cgcnn_pkg.__file__).parent / "form_1st.pth.tar"
    assert model_path.exists()

    def read_preds(csv_path: os.PathLike | str) -> dict[str, float]:
        """Read a predictions CSV into a ``{cif_id: prediction}`` mapping.

        Parameters
        ----------
        csv_path : os.PathLike or str
            Path to the predictions CSV.

        Returns
        -------
        dict of str to float
            Mapping from CIF id to predicted value.
        """
        out = {}
        with open(csv_path) as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                cid, _target, pred = ln.split(",")
                out[cid] = float(pred)
        return out

    runs, outputs = 5, []
    cwd = os.getcwd()
    try:
        os.chdir(base)
        for i in range(runs):
            out_csv = base / f"repro_{i}.csv"
            _ = predict_cgcnn(
                modelpath=str(model_path),
                cifpath=str(cif_dir),
                batch_size=256,
                workers=0,
                disable_cuda=True,
                chunk_id=1,
                output_csv=str(out_csv),
            )
            assert out_csv.exists()
            outputs.append(read_preds(out_csv))
    finally:
        os.chdir(cwd)

    first = outputs[0]
    for j, cur in enumerate(outputs[1:], start=2):
        for k in first:
            assert abs(first[k] - cur[k]) <= 1e-6, (
                f"Unstable CGCNN prediction for {k} on run {j}"
            )


def test_select_structure(cgcnn_output: dict[str, Path]) -> None:
    """Selection runs on the CGCNN predictions and writes POSCAR outputs.

    Parameters
    ----------
    cgcnn_output : dict of str to pathlib.Path
        Fixture providing the predictions CSV and extracted structures.

    Notes
    -----
    Calls :func:`run_select_structures` directly (no subprocess) and asserts
    that the ``new`` directory, ``POSCAR_*`` files, and ``id_prop.csv`` are
    produced.
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
