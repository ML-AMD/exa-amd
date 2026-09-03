"""Unit tests for the CGCNN prediction utilities in ``ml_models.cgcnn.predict``.

These tests cover the standalone helpers and classes (``Normalizer``,
``AverageMeter``, ``mae``, ``class_eval``, ``save_checkpoint``,
``_load_model_args``, ``_build_argparser``) that are not exercised by the
end-to-end workflow tests.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from ml_models.cgcnn.predict import (
    AverageMeter,
    Normalizer,
    _build_argparser,
    _load_model_args,
    _validate,
    class_eval,
    mae,
    save_checkpoint,
)

# Ensure repo root is importable
REPO_ROOT = Path(__file__).parent.parent.resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_normalizer_norm_denorm_roundtrip() -> None:
    """Normalizing then denormalizing should recover the original tensor."""
    sample = torch.tensor([1.0, 2.0, 3.0, 4.0])
    norm = Normalizer(sample)

    data = torch.tensor([2.5, 5.0, -1.0])
    normed = norm.norm(data)
    restored = norm.denorm(normed)

    assert torch.allclose(restored, data, atol=1e-6)
    assert torch.isclose(norm.mean, torch.mean(sample))
    assert torch.isclose(norm.std, torch.std(sample))


def test_normalizer_state_dict_roundtrip() -> None:
    """State dict should round-trip mean and std through load_state_dict."""
    norm = Normalizer(torch.tensor([0.0, 10.0]))
    state = norm.state_dict()

    other = Normalizer(torch.zeros(3))
    other.load_state_dict(state)

    assert torch.isclose(other.mean, norm.mean)
    assert torch.isclose(other.std, norm.std)


def test_mae_matches_manual() -> None:
    """mae should equal the manual mean of absolute differences."""
    pred = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.5, 2.0, 1.0])
    result = mae(pred, target)
    assert pytest.approx(result.item(), abs=1e-6) == (0.5 + 0.0 + 2.0) / 3


def test_average_meter_scalar_updates() -> None:
    """AverageMeter should track val, sum, count, and running average."""
    meter = AverageMeter()
    meter.update(2.0, n=1)
    meter.update(4.0, n=3)

    assert meter.val == 4.0
    assert meter.count == 4
    assert pytest.approx(meter.sum, abs=1e-6) == 2.0 + 4.0 * 3
    assert pytest.approx(meter.avg, abs=1e-6) == (2.0 + 12.0) / 4


def test_average_meter_reset() -> None:
    """reset should clear all accumulated statistics."""
    meter = AverageMeter()
    meter.update(5.0, n=2)
    meter.reset()
    assert (meter.val, meter.avg, meter.sum, meter.count) == (0, 0, 0, 0)


def test_class_eval_binary() -> None:
    """class_eval should return sensible metrics for a binary problem."""
    # Log-probabilities: rows favor class 1, 0, 1, 0 respectively.
    prediction = torch.log(
        torch.tensor(
            [
                [0.1, 0.9],
                [0.8, 0.2],
                [0.3, 0.7],
                [0.6, 0.4],
            ]
        )
    )
    target = torch.tensor([1, 0, 1, 0])

    accuracy, precision, recall, fscore, auc = class_eval(prediction, target)

    assert accuracy == pytest.approx(1.0)
    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(1.0)
    assert fscore == pytest.approx(1.0)
    assert 0.0 <= auc <= 1.0


def test_class_eval_non_binary_raises() -> None:
    """class_eval should raise NotImplementedError for >2 classes."""
    prediction = torch.log(torch.full((2, 3), 1.0 / 3.0))
    target = torch.tensor([0, 1])
    with pytest.raises(NotImplementedError):
        class_eval(prediction, target)


def test_load_model_args_missing_file(tmp_path: Path) -> None:
    """Missing checkpoint should fall back to regression defaults."""
    missing = tmp_path / "does_not_exist.pth.tar"
    args = _load_model_args(str(missing))
    assert args.task == "regression"


def test_load_model_args_from_checkpoint(tmp_path: Path) -> None:
    """Existing checkpoint args should be loaded into a SimpleNamespace."""
    ckpt_path = tmp_path / "model.pth.tar"
    torch.save({"args": {"task": "classification", "n_conv": 4}}, ckpt_path)

    args = _load_model_args(str(ckpt_path))
    assert args.task == "classification"
    assert args.n_conv == 4


def test_save_checkpoint_writes_file(tmp_path: Path) -> None:
    """save_checkpoint should write the checkpoint to disk."""
    dest = tmp_path / "checkpoint.pth.tar"
    payload = {"epoch": 1, "state_dict": {}}
    save_checkpoint(payload, is_best=False, filename=str(dest))
    assert dest.exists()
    loaded = torch.load(dest, weights_only=False)
    assert loaded["epoch"] == 1


def test_save_checkpoint_best_copies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When is_best is True, model_best.pth.tar should also be created."""
    monkeypatch.chdir(tmp_path)
    dest = tmp_path / "checkpoint.pth.tar"
    save_checkpoint({"epoch": 2}, is_best=True, filename=str(dest))
    assert dest.exists()
    assert (tmp_path / "model_best.pth.tar").exists()


def test_build_argparser_defaults() -> None:
    """The argument parser should apply documented default values."""
    parser = _build_argparser()
    ns = parser.parse_args(["model.pth.tar", "cif_dir"])
    assert ns.modelpath == "model.pth.tar"
    assert ns.cifpath == "cif_dir"
    assert ns.batch_size == 256
    assert ns.workers == 0
    assert ns.disable_cuda is False
    assert ns.print_freq == 10
    assert ns.chunk_id == 1
    assert ns.output_csv is None


def test_build_argparser_overrides() -> None:
    """The argument parser should honor explicit CLI overrides."""
    parser = _build_argparser()
    ns = parser.parse_args(
        [
            "m.pth.tar",
            "cifs",
            "-b",
            "32",
            "-j",
            "2",
            "--disable-cuda",
            "-p",
            "5",
            "--chunk_id",
            "3",
            "--output-csv",
            "out.csv",
        ]
    )
    assert ns.batch_size == 32
    assert ns.workers == 2
    assert ns.disable_cuda is True
    assert ns.print_freq == 5
    assert ns.chunk_id == 3
    assert ns.output_csv == "out.csv"


class _StubClassModel(nn.Module):
    """Model stub returning fixed log-probabilities for classification."""

    def __init__(self, log_probs: torch.Tensor) -> None:
        super().__init__()
        self._log_probs = log_probs

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Return the fixed classification log-probabilities."""
        return self._log_probs


class _StubRegModel(nn.Module):
    """Model stub returning fixed values for regression."""

    def __init__(self, values: torch.Tensor) -> None:
        super().__init__()
        self._values = values

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Return the fixed regression values."""
        return self._values


def _make_input(batch_size: int) -> tuple:
    """Build a dummy 4-tuple input compatible with the validation loop.

    Parameters
    ----------
    batch_size : int
        Number of atoms/rows in the dummy batch.

    Returns
    -------
    tuple
        A ``(atom_fea, nbr_fea, nbr_idx, crystal_atom_idx)`` input tuple.
    """
    atom_fea = torch.zeros(batch_size, 1)
    nbr_fea = torch.zeros(batch_size, 1, 1)
    nbr_idx = torch.zeros(batch_size, 1, dtype=torch.long)
    crys_idx = [torch.arange(batch_size)]
    return (atom_fea, nbr_fea, nbr_idx, crys_idx)


def test_validate_classification_branch(tmp_path: Path) -> None:
    """Exercise the classification else-branch of _validate."""
    log_probs = torch.log(
        torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])
    )
    target = torch.tensor([[1], [0], [1], [0]])
    cif_ids = ["a", "b", "c", "d"]
    loader = [(_make_input(4), target, cif_ids)]

    out_csv = tmp_path / "cls_results.csv"
    args = SimpleNamespace(cuda=False, print_freq=1, output_csv=str(out_csv))
    model_args = SimpleNamespace(task="classification")
    model = _StubClassModel(log_probs)

    auc = _validate(
        args,
        model_args,
        loader,
        model,
        nn.NLLLoss(),
        Normalizer(torch.zeros(3)),
        test=True,
    )

    assert 0.0 <= auc <= 1.0
    assert out_csv.exists()
    rows = out_csv.read_text().strip().splitlines()
    assert len(rows) == 4


def test_validate_regression_branch(tmp_path: Path) -> None:
    """Exercise the regression branch of _validate for completeness."""
    values = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    target = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    cif_ids = ["a", "b", "c", "d"]
    loader = [(_make_input(4), target, cif_ids)]

    out_csv = tmp_path / "reg_results.csv"
    args = SimpleNamespace(cuda=False, print_freq=1, output_csv=str(out_csv))
    model_args = SimpleNamespace(task="regression")

    normalizer = Normalizer(torch.tensor([0.0, 1.0, 2.0, 3.0]))
    model = _StubRegModel(normalizer.norm(values))

    metric = _validate(
        args,
        model_args,
        loader,
        model,
        nn.MSELoss(),
        normalizer,
        test=True,
    )

    assert metric >= 0.0
    assert out_csv.exists()
