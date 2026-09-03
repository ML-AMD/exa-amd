"""Tests for :mod:`ml_models.cgcnn.data`.

These tests cover the CGCNN data utilities: the train/val/test data-loader
splitter (:func:`get_train_val_test_loader`), the atom feature initializers
(:class:`AtomInitializer`, :class:`AtomCustomJSONInitializer`), the Gaussian
distance expansion (:class:`GaussianDistance`), the batch collation helper
(:func:`collate_pool`), and the :class:`CIFData` dataset.
"""

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from ml_models.cgcnn.data import (
    AtomCustomJSONInitializer,
    AtomInitializer,
    CIFData,
    GaussianDistance,
    collate_pool,
    get_train_val_test_loader,
)


class DummyDataset(Dataset):
    """Simple dataset for testing purposes.

    Parameters
    ----------
    size : int, optional
        Number of samples in the dataset. Default is ``100``.
    """

    def __init__(self, size: int = 100) -> None:
        self.size = size
        self.data = torch.randn(size, 10)
        self.targets = torch.randn(size, 1)

    def __len__(self) -> int:
        """Return the number of samples.

        Returns
        -------
        int
            The dataset size.
        """
        return self.size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the ``(data, target)`` pair at ``idx``.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        tuple of torch.Tensor
            The feature tensor and target tensor.
        """
        return self.data[idx], self.targets[idx]


class TestGetTrainValTestLoader:
    """Tests for :func:`get_train_val_test_loader`."""

    def test_basic_split_without_test(self) -> None:
        """A basic train/val split omits the test loader."""
        dataset = DummyDataset(size=100)
        train_loader, val_loader = get_train_val_test_loader(
            dataset, batch_size=16, val_ratio=0.1, test_ratio=0.1, return_test=False
        )

        assert train_loader is not None
        assert val_loader is not None
        assert len(train_loader.sampler) == 80  # 80% of 100
        assert len(val_loader.sampler) == 10  # 10% of 100

    def test_basic_split_with_test(self) -> None:
        """A train/val/test split returns the test loader."""
        dataset = DummyDataset(size=100)
        train_loader, val_loader, test_loader = get_train_val_test_loader(
            dataset, batch_size=16, val_ratio=0.1, test_ratio=0.1, return_test=True
        )

        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None
        assert len(train_loader.sampler) == 80
        assert len(val_loader.sampler) == 10
        assert len(test_loader.sampler) == 10

    def test_explicit_train_ratio(self) -> None:
        """An explicit ``train_ratio`` is honoured."""
        dataset = DummyDataset(size=100)
        train_loader, val_loader = get_train_val_test_loader(
            dataset,
            batch_size=16,
            train_ratio=0.7,
            val_ratio=0.2,
            test_ratio=0.1,
            return_test=False,
        )

        assert len(train_loader.sampler) == 70
        assert len(val_loader.sampler) == 20

    def test_custom_train_size(self) -> None:
        """An explicit ``train_size`` overrides the ratio."""
        dataset = DummyDataset(size=100)
        train_loader, val_loader = get_train_val_test_loader(
            dataset,
            batch_size=16,
            val_ratio=0.1,
            test_ratio=0.1,
            train_size=50,
            return_test=False,
        )

        assert len(train_loader.sampler) == 50

    def test_custom_val_size(self) -> None:
        """An explicit ``val_size`` overrides the ratio."""
        dataset = DummyDataset(size=100)
        train_loader, val_loader = get_train_val_test_loader(
            dataset,
            batch_size=16,
            val_ratio=0.1,
            test_ratio=0.1,
            val_size=15,
            return_test=False,
        )

        assert len(val_loader.sampler) == 15

    def test_custom_test_size(self) -> None:
        """An explicit ``test_size`` overrides the ratio."""
        dataset = DummyDataset(size=100)
        train_loader, val_loader, test_loader = get_train_val_test_loader(
            dataset,
            batch_size=16,
            val_ratio=0.1,
            test_ratio=0.1,
            test_size=20,
            return_test=True,
        )

        assert len(test_loader.sampler) == 20

    def test_all_custom_sizes(self) -> None:
        """All explicit size parameters are applied together."""
        dataset = DummyDataset(size=100)
        train_loader, val_loader, test_loader = get_train_val_test_loader(
            dataset,
            batch_size=16,
            train_size=60,
            val_size=20,
            test_size=20,
            return_test=True,
        )

        assert len(train_loader.sampler) == 60
        assert len(val_loader.sampler) == 20
        assert len(test_loader.sampler) == 20

    def test_batch_size(self) -> None:
        """The ``batch_size`` is propagated to the loaders."""
        dataset = DummyDataset(size=100)
        train_loader, _ = get_train_val_test_loader(
            dataset, batch_size=32, val_ratio=0.1, test_ratio=0.1, return_test=False
        )

        assert train_loader.batch_size == 32

    def test_num_workers(self) -> None:
        """The ``num_workers`` value is propagated to the loaders."""
        dataset = DummyDataset(size=100)
        train_loader, _ = get_train_val_test_loader(
            dataset,
            batch_size=16,
            val_ratio=0.1,
            test_ratio=0.1,
            num_workers=2,
            return_test=False,
        )

        assert train_loader.num_workers == 2

    def test_pin_memory(self) -> None:
        """The ``pin_memory`` flag is propagated to the loaders."""
        dataset = DummyDataset(size=100)
        train_loader, _ = get_train_val_test_loader(
            dataset,
            batch_size=16,
            val_ratio=0.1,
            test_ratio=0.1,
            pin_memory=True,
            return_test=False,
        )

        assert train_loader.pin_memory is True

    def test_custom_collate_fn(self) -> None:
        """A custom collate function is used by the train loader."""

        def custom_collate(batch: list) -> list:
            return batch

        dataset = DummyDataset(size=100)
        train_loader, _ = get_train_val_test_loader(
            dataset,
            collate_fn=custom_collate,
            batch_size=16,
            val_ratio=0.1,
            test_ratio=0.1,
            return_test=False,
        )

        assert train_loader.collate_fn == custom_collate

    def test_ratios_sum_validation(self) -> None:
        """Ratios summing above one raise ``AssertionError``."""
        dataset = DummyDataset(size=100)

        with pytest.raises(AssertionError):
            get_train_val_test_loader(
                dataset,
                train_ratio=0.8,
                val_ratio=0.3,
                test_ratio=0.3,
                return_test=False,
            )

    def test_train_ratio_none_validation(self) -> None:
        """With ``train_ratio=None``, ``val + test`` must be below one."""
        dataset = DummyDataset(size=100)

        with pytest.raises(AssertionError):
            get_train_val_test_loader(
                dataset,
                train_ratio=None,
                val_ratio=0.6,
                test_ratio=0.6,
                return_test=False,
            )

    def test_small_dataset(self) -> None:
        """Splitting works for a very small dataset."""
        dataset = DummyDataset(size=10)
        train_loader, val_loader = get_train_val_test_loader(
            dataset, batch_size=2, val_ratio=0.1, test_ratio=0.1, return_test=False
        )

        assert len(train_loader.sampler) == 8
        assert len(val_loader.sampler) == 1

    def test_loader_iteration(self) -> None:
        """The returned loaders can be iterated and yield sized batches."""
        dataset = DummyDataset(size=50)
        train_loader, val_loader = get_train_val_test_loader(
            dataset, batch_size=8, val_ratio=0.2, test_ratio=0.2, return_test=False
        )

        # Test that we can iterate over the loaders
        train_batches = list(train_loader)
        val_batches = list(val_loader)

        assert len(train_batches) > 0
        assert len(val_batches) > 0

        # Check batch shapes
        data, target = train_batches[0]
        assert data.shape[0] <= 8  # batch_size or less for last batch
        assert target.shape[0] <= 8

    def test_zero_test_ratio(self) -> None:
        """A ``test_ratio`` of zero exercises the val-sampler slicing edge case."""
        dataset = DummyDataset(size=100)
        train_loader, val_loader = get_train_val_test_loader(
            dataset, batch_size=16, val_ratio=0.2, test_ratio=0.0, return_test=False
        )

        assert len(train_loader.sampler) == 80
        assert len(val_loader.sampler) == 20

    def test_train_ratio_none_with_warning(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A ``train_ratio`` of ``None`` prints a warning message.

        Parameters
        ----------
        capsys : pytest.CaptureFixture[str]
            Fixture capturing ``stdout`` and ``stderr``.
        """
        dataset = DummyDataset(size=100)
        train_loader, val_loader = get_train_val_test_loader(
            dataset,
            batch_size=16,
            train_ratio=None,
            val_ratio=0.1,
            test_ratio=0.1,
            return_test=False,
        )

        captured = capsys.readouterr()
        assert "[Warning] train_ratio is None" in captured.out
        assert len(train_loader.sampler) == 80

    def test_exact_ratio_sum(self) -> None:
        """Ratios summing to exactly one are accepted."""
        dataset = DummyDataset(size=100)
        train_loader, val_loader, test_loader = get_train_val_test_loader(
            dataset,
            batch_size=16,
            train_ratio=0.7,
            val_ratio=0.2,
            test_ratio=0.1,
            return_test=True,
        )

        assert len(train_loader.sampler) == 70
        assert len(val_loader.sampler) == 20
        assert len(test_loader.sampler) == 10

    def test_return_test_with_zero_test_size(self) -> None:
        """``return_test=True`` with ``test_size=0`` yields an empty test loader."""
        dataset = DummyDataset(size=100)
        train_loader, val_loader, test_loader = get_train_val_test_loader(
            dataset,
            batch_size=16,
            val_ratio=0.2,
            test_ratio=0.0,
            test_size=0,
            return_test=True,
        )

        assert len(train_loader.sampler) == 80
        assert len(val_loader.sampler) == 20
        assert len(test_loader.sampler) == 0

    def test_nonzero_test_with_explicit_sizes(self) -> None:
        """Explicit sizes with a small nonzero test slice are respected."""
        dataset = DummyDataset(size=100)
        train_loader, val_loader, test_loader = get_train_val_test_loader(
            dataset,
            batch_size=16,
            train_size=70,
            val_size=20,
            test_size=10,
            return_test=True,
        )

        assert len(train_loader.sampler) == 70
        assert len(val_loader.sampler) == 20
        assert len(test_loader.sampler) == 10


class TestAtomInitializer:
    """Tests for :class:`AtomInitializer`."""

    def test_initialization(self) -> None:
        """Construction records the atom types and empty embedding."""
        atom_types = [1, 6, 8]
        initializer = AtomInitializer(atom_types)
        assert initializer.atom_types == set(atom_types)
        assert initializer._embedding == {}

    def test_get_atom_fea_assertion(self) -> None:
        """Requesting a feature for an unknown atom raises ``AssertionError``."""
        initializer = AtomInitializer([1, 6, 8])
        with pytest.raises(AssertionError):
            initializer.get_atom_fea(7)

    def test_state_dict_operations(self) -> None:
        """``state_dict`` round-trips through ``load_state_dict``."""
        initializer = AtomInitializer([1, 6, 8])
        state = {1: 0, 6: 1, 8: 2}
        initializer.load_state_dict(state)

        assert initializer.state_dict() == state
        assert initializer.atom_types == {1, 6, 8}

    def test_decode(self) -> None:
        """``decode`` maps embedding indices back to atom types."""
        initializer = AtomInitializer([1, 6, 8])
        state = {1: 0, 6: 1, 8: 2}
        initializer.load_state_dict(state)

        assert initializer.decode(0) == 1
        assert initializer.decode(1) == 6
        assert initializer.decode(2) == 8

    def test_decode_builds_decode_dict(self) -> None:
        """``decode`` lazily builds ``_decodedict`` on first use."""
        initializer = AtomInitializer([1, 6])
        initializer._embedding = {1: 0, 6: 1}
        initializer.atom_types = {1, 6}

        # First call should build _decodedict
        assert not hasattr(initializer, "_decodedict")
        result = initializer.decode(0)
        assert hasattr(initializer, "_decodedict")
        assert result == 1


class TestAtomCustomJSONInitializer:
    """Tests for :class:`AtomCustomJSONInitializer`."""

    def test_initialization(self, tmp_path: Path) -> None:
        """Initialisation loads per-atom features from a JSON file.

        Parameters
        ----------
        tmp_path : pathlib.Path
            Pytest-provided temporary directory.
        """
        temp_file = tmp_path / "atom_init.json"
        temp_file.write_text(json.dumps({"1": [0.1, 0.2, 0.3], "6": [0.4, 0.5, 0.6]}))

        initializer = AtomCustomJSONInitializer(str(temp_file))
        assert 1 in initializer.atom_types
        assert 6 in initializer.atom_types

        fea_1 = initializer.get_atom_fea(1)
        assert isinstance(fea_1, np.ndarray)
        assert np.allclose(fea_1, [0.1, 0.2, 0.3])


class TestGaussianDistance:
    """Tests for :class:`GaussianDistance`."""

    def test_initialization(self) -> None:
        """Construction derives the filter length and default variance."""
        gd = GaussianDistance(dmin=0, dmax=6, step=0.2)
        assert len(gd.filter) == 31  # (6-0)/0.2 + 1
        assert gd.var == 0.2

    def test_initialization_with_custom_var(self) -> None:
        """A custom variance is stored on the instance."""
        gd = GaussianDistance(dmin=0, dmax=6, step=0.2, var=0.5)
        assert gd.var == 0.5

    def test_initialization_assertions(self) -> None:
        """Invalid distance ranges raise ``AssertionError``."""
        with pytest.raises(AssertionError):
            GaussianDistance(dmin=6, dmax=0, step=0.2)

        with pytest.raises(AssertionError):
            GaussianDistance(dmin=0, dmax=0.1, step=0.2)

    def test_expand(self) -> None:
        """Expanding 1D distances yields the expected shape and dtype."""
        gd = GaussianDistance(dmin=0, dmax=4, step=1.0)
        distances = np.array([1.0, 2.0, 3.0])
        expanded = gd.expand(distances)

        assert expanded.shape == (3, 5)  # 3 distances, 5 filter points
        assert expanded.dtype == np.float64

    def test_expand_2d(self) -> None:
        """Expanding a 2D distance array adds a trailing filter axis."""
        gd = GaussianDistance(dmin=0, dmax=4, step=1.0)
        distances = np.array([[1.0, 2.0], [3.0, 4.0]])
        expanded = gd.expand(distances)

        assert expanded.shape == (2, 2, 5)


class TestCollatePool:
    """Tests for :func:`collate_pool`."""

    def test_collate_pool(self) -> None:
        """Batches are concatenated with correct shapes and cif ids."""
        # Create mock data
        atom_fea_1 = torch.randn(3, 5)
        nbr_fea_1 = torch.randn(3, 12, 10)
        nbr_fea_idx_1 = torch.randint(0, 3, (3, 12))
        target_1 = torch.tensor([1.0])

        atom_fea_2 = torch.randn(2, 5)
        nbr_fea_2 = torch.randn(2, 12, 10)
        nbr_fea_idx_2 = torch.randint(0, 2, (2, 12))
        target_2 = torch.tensor([2.0])

        dataset_list = [
            ((atom_fea_1, nbr_fea_1, nbr_fea_idx_1), target_1, "cif_1"),
            ((atom_fea_2, nbr_fea_2, nbr_fea_idx_2), target_2, "cif_2"),
        ]

        (
            (batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx, crystal_atom_idx),
            batch_target,
            batch_cif_ids,
        ) = collate_pool(dataset_list)

        assert batch_atom_fea.shape == (5, 5)  # 3 + 2 atoms
        assert batch_nbr_fea.shape == (5, 12, 10)
        assert batch_nbr_fea_idx.shape == (5, 12)
        assert len(crystal_atom_idx) == 2
        assert batch_target.shape == (2, 1)
        assert batch_cif_ids == ["cif_1", "cif_2"]

    def test_collate_pool_index_offset(self) -> None:
        """Neighbour indices are offset per crystal during collation."""
        atom_fea_1 = torch.randn(3, 5)
        nbr_fea_1 = torch.randn(3, 12, 10)
        nbr_fea_idx_1 = torch.zeros((3, 12), dtype=torch.long)
        target_1 = torch.tensor([1.0])

        atom_fea_2 = torch.randn(2, 5)
        nbr_fea_2 = torch.randn(2, 12, 10)
        nbr_fea_idx_2 = torch.zeros((2, 12), dtype=torch.long)
        target_2 = torch.tensor([2.0])

        dataset_list = [
            ((atom_fea_1, nbr_fea_1, nbr_fea_idx_1), target_1, "cif_1"),
            ((atom_fea_2, nbr_fea_2, nbr_fea_idx_2), target_2, "cif_2"),
        ]

        (_, _, batch_nbr_fea_idx, crystal_atom_idx), _, _ = collate_pool(dataset_list)

        # First crystal indices should be 0-2
        assert torch.equal(crystal_atom_idx[0], torch.LongTensor([0, 1, 2]))
        # Second crystal indices should be offset by 3
        assert torch.equal(crystal_atom_idx[1], torch.LongTensor([3, 4]))
        # Second crystal neighbor indices should be offset
        assert batch_nbr_fea_idx[3:].min() >= 3


class TestGaussianDistanceExpandValues:
    """Value-level tests for GaussianDistance.expand."""

    def test_expand_peak_at_filter(self) -> None:
        """Distance equal to a filter point should produce a value of 1.0."""
        gd = GaussianDistance(dmin=0, dmax=4, step=1.0)
        expanded = gd.expand(np.array([0.0]))
        # First filter point is 0.0, so exp(0) == 1.0
        assert np.isclose(expanded[0, 0], 1.0)

    def test_expand_monotonic_decay(self) -> None:
        """Values should decay away from the matching filter point."""
        gd = GaussianDistance(dmin=0, dmax=4, step=1.0)
        expanded = gd.expand(np.array([0.0]))
        assert expanded[0, 0] > expanded[0, 1] > expanded[0, 2]


class TestCIFData:
    """Tests for the :class:`CIFData` dataset."""

    @staticmethod
    def _write_cif(path: str, cif_id: str) -> None:
        """Write a minimal cubic NaCl-like CIF file.

        Parameters
        ----------
        path : str
            Directory in which to write the CIF file.
        cif_id : str
            Identifier used for both the ``data_`` block and file name.
        """
        cif = f"""data_{cif_id}
_cell_length_a 4.0
_cell_length_b 4.0
_cell_length_c 4.0
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0
_symmetry_space_group_name_H-M 'P 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Na1 Na 0.0 0.0 0.0
Cl1 Cl 0.5 0.5 0.5
"""
        (Path(path) / f"{cif_id}.cif").write_text(cif)

    @staticmethod
    def _write_atom_init(path: str, numbers: list[int], fea_len: int = 4) -> None:
        """Write an ``atom_init.json`` embedding file.

        Parameters
        ----------
        path : str
            Directory in which to write the file.
        numbers : list of int
            Atomic numbers to include in the embedding.
        fea_len : int, optional
            Length of each per-atom feature vector. Default is ``4``.
        """
        embedding = {str(n): [float(i)] * fea_len for i, n in enumerate(numbers)}
        (Path(path) / "atom_init.json").write_text(json.dumps(embedding))

    @staticmethod
    def _write_id_prop(path: str, rows: list[list[str]]) -> None:
        """Write an ``id_prop.csv`` manifest.

        Parameters
        ----------
        path : str
            Directory in which to write the file.
        rows : list of list of str
            Rows of ``[cif_id, target]`` values to write.
        """
        with open(Path(path) / "id_prop.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

    def test_missing_root_dir(self) -> None:
        """Nonexistent root_dir should raise AssertionError."""
        with pytest.raises(AssertionError):
            CIFData("/nonexistent/path/for/testing")

    def test_missing_id_prop(self, tmp_path: Path) -> None:
        """Missing id_prop.csv should raise AssertionError."""
        with pytest.raises(AssertionError):
            CIFData(str(tmp_path))

    def test_missing_atom_init(self, tmp_path: Path) -> None:
        """Missing atom_init.json should raise AssertionError."""
        self._write_id_prop(str(tmp_path), [["mat0", "1.0"]])
        with pytest.raises(AssertionError):
            CIFData(str(tmp_path))

    def test_len(self, tmp_path: Path) -> None:
        """__len__ should reflect the number of entries in id_prop.csv."""
        tmp = str(tmp_path)
        self._write_id_prop(tmp, [["m0", "1.0"], ["m1", "2.0"]])
        self._write_atom_init(tmp, [11, 17])
        self._write_cif(tmp, "m0")
        self._write_cif(tmp, "m1")
        dataset = CIFData(tmp, max_num_nbr=6, radius=6)
        assert len(dataset) == 2

    def test_getitem_shapes(self, tmp_path: Path) -> None:
        """__getitem__ should return correctly shaped tensors and cif_id."""
        tmp = str(tmp_path)
        self._write_id_prop(tmp, [["m0", "3.5"]])
        self._write_atom_init(tmp, [11, 17], fea_len=4)
        self._write_cif(tmp, "m0")
        dataset = CIFData(tmp, max_num_nbr=6, radius=6, step=0.5)
        (atom_fea, nbr_fea, nbr_fea_idx), target, cif_id = dataset[0]

        assert atom_fea.shape[0] == 2  # two atoms
        assert atom_fea.shape[1] == 4  # feature length
        assert nbr_fea.shape[0] == 2
        assert nbr_fea.shape[1] == 6  # max_num_nbr
        assert nbr_fea_idx.shape == (2, 6)
        assert torch.isclose(target, torch.tensor([3.5]))
        assert cif_id == "m0"

    def test_getitem_insufficient_neighbors_warns(self, tmp_path: Path) -> None:
        """Requesting more neighbors than available should warn and pad."""
        tmp = str(tmp_path)
        self._write_id_prop(tmp, [["m0", "1.0"]])
        self._write_atom_init(tmp, [11, 17], fea_len=4)
        self._write_cif(tmp, "m0")
        # Small radius + large max_num_nbr forces padding
        dataset = CIFData(tmp, max_num_nbr=50, radius=3, step=0.5)
        with pytest.warns(UserWarning, match="not find enough neighbors"):
            (_, nbr_fea, nbr_fea_idx), _, _ = dataset[0]
        assert nbr_fea.shape[1] == 50
        assert nbr_fea_idx.shape[1] == 50
