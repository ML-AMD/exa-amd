import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ml_models.mlip import mlip_relax

# Module path matching the imported module object (used for patch targets)
MODULE = mlip_relax.__name__


@pytest.fixture(autouse=True)
def reset_globals():
    """Reset module globals before and after each test to avoid leakage."""
    mlip_relax.global_predictor = None
    mlip_relax.global_calc = None
    mlip_relax.global_model_path = None
    yield
    mlip_relax.global_predictor = None
    mlip_relax.global_calc = None
    mlip_relax.global_model_path = None


@pytest.fixture
def mock_atoms():
    """Create a mock ASE Atoms object."""
    atoms = MagicMock()
    atoms.get_potential_energy.return_value = -50.0
    atoms.__len__.return_value = 10
    atoms.symbols.get_chemical_formula.return_value = "H2O"
    return atoms


@pytest.fixture
def mock_calculator():
    """Create a mock FAIRChemCalculator."""
    return MagicMock()


@pytest.fixture
def mock_predictor():
    """Create a mock predictor."""
    return MagicMock()


@pytest.fixture
def temp_energy_log_dir(tmp_path):
    """Create a temporary directory for energy logs."""
    log_dir = tmp_path / "energy_logs"
    log_dir.mkdir()
    return str(log_dir)


@pytest.fixture
def sample_cif_file(tmp_path):
    """Create a sample CIF file path."""
    cif_file = tmp_path / "structure_42.cif"
    cif_file.write_text("mock cif content")
    return str(cif_file)


class TestWorkerInitializer:
    """Tests for worker_initializer function."""

    @patch(f"{MODULE}.FAIRChemCalculator")
    @patch(f"{MODULE}.pretrained_mlip")
    @patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"})
    def test_worker_initializer_success(
        self, mock_mlip, mock_calc_class, mock_predictor
    ):
        """Test successful worker initialization."""
        mock_mlip.load_predict_unit.return_value = mock_predictor
        mock_calc_instance = MagicMock()
        mock_calc_class.return_value = mock_calc_instance

        model_path = "/path/to/model.pt"
        mlip_relax.worker_initializer(model_path)

        # Verify model loading
        mock_mlip.load_predict_unit.assert_called_once_with(model_path, device="cuda")
        mock_calc_class.assert_called_once_with(mock_predictor, task_name="omat")

        # Verify globals are set
        assert mlip_relax.global_predictor == mock_predictor
        assert mlip_relax.global_calc == mock_calc_instance
        assert mlip_relax.global_model_path == model_path

    @patch(f"{MODULE}.FAIRChemCalculator")
    @patch(f"{MODULE}.pretrained_mlip")
    @patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "Unknown"})
    def test_worker_initializer_no_gpu_env(
        self, mock_mlip, mock_calc_class, mock_predictor
    ):
        """Test worker initialization without GPU environment variable."""
        mock_mlip.load_predict_unit.return_value = mock_predictor
        mock_calc_class.return_value = MagicMock()

        mlip_relax.worker_initializer("/path/to/model.pt")

        # Should still proceed despite no GPU ID
        assert mlip_relax.global_predictor is not None

    @patch(f"{MODULE}.pretrained_mlip")
    @patch("builtins.print")
    def test_worker_initializer_model_load_error(self, mock_print, mock_mlip):
        """Test worker initialization with model loading error."""
        mock_mlip.load_predict_unit.side_effect = Exception("CUDA out of memory")

        with pytest.raises(SystemExit) as exc_info:
            mlip_relax.worker_initializer("/path/to/model.pt")

        assert exc_info.value.code == 1


class TestRelaxAndLog:
    """Tests for relax_and_log function."""

    @patch(f"{MODULE}.write")
    @patch(f"{MODULE}.read")
    @patch(f"{MODULE}.FIRE")
    @patch(f"{MODULE}.FrechetCellFilter")
    def test_relax_and_log_success(
        self,
        mock_filter,
        mock_fire,
        mock_read,
        mock_write,
        sample_cif_file,
        temp_energy_log_dir,
        mock_atoms,
        mock_calculator,
    ):
        """Test successful relaxation and logging."""
        # Setup mocks
        mlip_relax.global_calc = mock_calculator
        mock_read.return_value = mock_atoms
        mock_opt = MagicMock()
        mock_fire.return_value = mock_opt

        # Run relaxation
        result = mlip_relax.relax_and_log(sample_cif_file, temp_energy_log_dir)

        # Verify file operations
        mock_read.assert_called_once_with(sample_cif_file)
        assert mock_atoms.calc == mock_calculator

        # Verify optimization
        mock_filter.assert_called_once_with(mock_atoms)
        mock_opt.run.assert_called_once_with(fmax=0.05, steps=100)

        # Verify output files
        mock_write.assert_called_once()
        assert "CONTCAR_42" in mock_write.call_args[0][0]

        # Verify energy log
        energy_log = Path(temp_energy_log_dir) / "energy_42.tmp"
        assert energy_log.exists()
        content = energy_log.read_text()
        assert "42,-5.0,H2O" in content

        # Verify return message
        assert "Successfully relaxed" in result
        assert "-5.0000 eV/atom" in result

    @patch(f"{MODULE}.read")
    def test_relax_and_log_file_not_found(self, mock_read, temp_energy_log_dir):
        """Test relaxation with missing input file."""
        mlip_relax.global_calc = MagicMock()
        mock_read.side_effect = FileNotFoundError("File not found")

        result = mlip_relax.relax_and_log("nonexistent.cif", temp_energy_log_dir)

        assert "Error relaxing" in result
        assert "output directory not found" in result

    @patch(f"{MODULE}.write")
    @patch(f"{MODULE}.read")
    @patch(f"{MODULE}.FIRE")
    def test_relax_and_log_optimization_error(
        self,
        mock_fire,
        mock_read,
        mock_write,
        sample_cif_file,
        temp_energy_log_dir,
        mock_atoms,
        mock_calculator,
    ):
        """Test relaxation with optimization error."""
        mlip_relax.global_calc = mock_calculator
        mock_read.return_value = mock_atoms
        mock_opt = MagicMock()
        mock_opt.run.side_effect = RuntimeError("Optimization failed")
        mock_fire.return_value = mock_opt

        result = mlip_relax.relax_and_log(sample_cif_file, temp_energy_log_dir)

        # Verify error log
        energy_log = Path(temp_energy_log_dir) / "energy_42.tmp"
        assert energy_log.exists()
        content = energy_log.read_text()
        assert "42,ERROR" in content

        assert "Error relaxing" in result

    @patch(f"{MODULE}.write")
    @patch(f"{MODULE}.read")
    @patch(f"{MODULE}.FIRE")
    @patch(f"{MODULE}.FrechetCellFilter")
    def test_relax_and_log_no_index_in_filename(
        self,
        mock_filter,
        mock_fire,
        mock_read,
        mock_write,
        temp_energy_log_dir,
        mock_atoms,
        mock_calculator,
        tmp_path,
    ):
        """Test relaxation with filename without numeric index."""
        mlip_relax.global_calc = mock_calculator
        mock_read.return_value = mock_atoms
        mock_fire.return_value = MagicMock()

        # Create file without numeric index
        cif_file = tmp_path / "structure.cif"
        cif_file.write_text("mock content")

        result = mlip_relax.relax_and_log(str(cif_file), temp_energy_log_dir)

        # Should use base name as index
        assert "Successfully relaxed" in result
        energy_log = Path(temp_energy_log_dir) / "energy_structure.tmp"
        assert energy_log.exists()


class TestMain:
    """Tests for main function."""

    @patch(f"{MODULE}.relax_and_log")
    @patch(f"{MODULE}.worker_initializer")
    @patch(f"{MODULE}.warnings")
    def test_main_success(
        self, mock_warnings, mock_init, mock_relax, sample_cif_file, temp_energy_log_dir
    ):
        """Test successful main execution."""
        test_args = [
            "mlip_relax.py",
            "/path/to/model.pt",
            temp_energy_log_dir,
            sample_cif_file,
        ]

        with patch.object(sys, "argv", test_args):
            mlip_relax.main()

        # Verify initialization
        mock_init.assert_called_once_with("/path/to/model.pt")
        mock_warnings.filterwarnings.assert_called_once_with("ignore")

        # Verify relaxation
        mock_relax.assert_called_once_with(sample_cif_file, temp_energy_log_dir)

    @patch(f"{MODULE}.relax_and_log")
    @patch(f"{MODULE}.worker_initializer")
    def test_main_multiple_files(
        self, mock_init, mock_relax, tmp_path, temp_energy_log_dir
    ):
        """Test main with multiple input files."""
        file1 = tmp_path / "struct1.cif"
        file2 = tmp_path / "struct2.cif"
        file1.write_text("content1")
        file2.write_text("content2")

        test_args = [
            "mlip_relax.py",
            "/path/to/model.pt",
            temp_energy_log_dir,
            str(file1),
            str(file2),
        ]

        with patch.object(sys, "argv", test_args):
            mlip_relax.main()

        # Verify both files processed
        assert mock_relax.call_count == 2
        mock_relax.assert_any_call(str(file1), temp_energy_log_dir)
        mock_relax.assert_any_call(str(file2), temp_energy_log_dir)

    @patch("builtins.print")
    def test_main_insufficient_arguments(self, mock_print):
        """Test main with insufficient arguments."""
        test_args = ["mlip_relax.py", "/path/to/model.pt"]

        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit) as exc_info:
                mlip_relax.main()

        assert exc_info.value.code == 1
        # Verify usage message printed
        assert mock_print.called
        printed_text = str(mock_print.call_args_list[0])
        assert "Usage:" in printed_text

    @patch("builtins.print")
    def test_main_no_arguments(self, mock_print):
        """Test main with no arguments."""
        test_args = ["mlip_relax.py"]

        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit) as exc_info:
                mlip_relax.main()

        assert exc_info.value.code == 1


class TestGlobalVariables:
    """Tests for global variable behavior."""

    def test_global_variables_initial_state(self):
        """Test that global variables start as None."""
        # Import fresh module or reset
        import importlib

        importlib.reload(mlip_relax)

        assert mlip_relax.global_predictor is None
        assert mlip_relax.global_calc is None
        assert mlip_relax.global_model_path is None
