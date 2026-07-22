import os

import pytest

import parsl_tasks.cgcnn as cgcnn_mod
from parsl_tasks.cgcnn import cmd_cgcnn_prediction
from tools.config_labels import ConfigKeys as CK


@pytest.fixture
def work_dir(tmp_path):
    """Create a work directory containing the per-chunk structures folder."""
    (tmp_path / "structures" / "0").mkdir(parents=True)
    return tmp_path


@pytest.fixture
def config(work_dir):
    return {
        CK.WORK_DIR: str(work_dir),
        CK.BATCH_SIZE: 32,
        CK.NUM_WORKERS: 4,
    }


@pytest.fixture(autouse=True)
def fake_pkg(tmp_path, monkeypatch):
    """Point the cgcnn package at a temporary directory with fake assets."""
    pkg_dir = tmp_path / "pkg"
    pkg_dir.mkdir()
    (pkg_dir / "form_1st.pth.tar").write_text("model")
    (pkg_dir / "atom_init.json").write_text("{}")
    (pkg_dir / "predict.py").write_text("# predict")
    monkeypatch.setattr(cgcnn_mod.cgcnn_pkg, "__file__", str(pkg_dir / "__init__.py"))
    return pkg_dir


def test_cmd_returns_expected_command(config, fake_pkg, work_dir):
    cmd = cmd_cgcnn_prediction(config, n_chunks=2, id=0)

    assert cmd.startswith(f"python {fake_pkg / 'predict.py'}")
    assert str(fake_pkg / "form_1st.pth.tar") in cmd
    assert str(work_dir / "structures" / "0") in cmd
    assert "--batch-size 32" in cmd
    assert "--workers 4" in cmd
    assert "--chunk_id 0" in cmd


def test_cmd_copies_atom_init(config, fake_pkg, work_dir):
    cmd_cgcnn_prediction(config, n_chunks=1, id=0)
    assert (work_dir / "structures" / "0" / "atom_init.json").exists()


def test_cmd_changes_cwd(config, work_dir):
    cmd_cgcnn_prediction(config, n_chunks=1, id=0)
    assert os.getcwd() == str(work_dir)


@pytest.mark.parametrize("n_chunks", [0, -1])
def test_invalid_n_chunks_raises(config, n_chunks):
    with pytest.raises(ValueError, match="n_chunks must be positive"):
        cmd_cgcnn_prediction(config, n_chunks=n_chunks, id=0)


@pytest.mark.parametrize("id", [-1, 5])
def test_id_out_of_range_raises(config, id):
    with pytest.raises(ValueError, match="id must satisfy"):
        cmd_cgcnn_prediction(config, n_chunks=2, id=id)


def test_missing_structures_dir_raises(config, work_dir):
    # id=2 passes the range guard (0 <= 2 < 3) but structures/2 does not
    # exist. Remove the parent so shutil.copy cannot resolve the destination.
    import shutil as _sh

    _sh.rmtree(work_dir / "structures")
    with pytest.raises(FileNotFoundError):
        cmd_cgcnn_prediction(config, n_chunks=3, id=2)
