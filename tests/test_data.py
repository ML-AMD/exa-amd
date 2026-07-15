import json
import os
import tempfile

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from ml_models.cgcnn.data import (
    AtomCustomJSONInitializer,
    AtomInitializer,
    GaussianDistance,
    collate_pool,
    get_train_val_test_loader,
)


class DummyDataset(Dataset):
    """Simple dataset for testing purposes."""

    def __init__(self, size=100):
        self.size = size
        self.data = torch.randn(size, 10)
        self.targets = torch.randn(size, 1)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class TestGetTrainValTestLoader:
    """Test suite for get_train_val_test_loader function."""

    def test_basic_split_without_test(self):
        """Test basic train/val split without returning test loader."""
        dataset = DummyDataset(size=100)
        train_loader, val_loader = get_train_val_test_loader(
            dataset, batch_size=16, val_ratio=0.1, test_ratio=0.1, return_test=False
        )

        assert train_loader is not None
        assert val_loader is not None
        assert len(train_loader.sampler) == 80  # 80% of 100
        assert len(val_loader.sampler) == 10  # 10% of 100

    def test_basic_split_with_test(self):
        """Test train/val/test split with test loader returned."""
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

    def test_explicit_train_ratio(self):
        """Test with explicitly specified train_ratio."""
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

    def test_custom_train_size(self):
        """Test with custom train_size parameter."""
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

    def test_custom_val_size(self):
        """Test with custom val_size parameter."""
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

    def test_custom_test_size(self):
        """Test with custom test_size parameter."""
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

    def test_all_custom_sizes(self):
        """Test with all custom size parameters."""
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

    def test_batch_size(self):
        """Test that batch_size is correctly applied."""
        dataset = DummyDataset(size=100)
        train_loader, _ = get_train_val_test_loader(
            dataset, batch_size=32, val_ratio=0.1, test_ratio=0.1, return_test=False
        )

        assert train_loader.batch_size == 32

    def test_num_workers(self):
        """Test that num_workers is correctly applied."""
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

    def test_pin_memory(self):
        """Test that pin_memory is correctly applied."""
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

    def test_custom_collate_fn(self):
        """Test with custom collate function."""

        def custom_collate(batch):
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

    def test_ratios_sum_validation(self):
        """Test that ratios summing to more than 1 raises assertion."""
        dataset = DummyDataset(size=100)

        with pytest.raises(AssertionError):
            get_train_val_test_loader(
                dataset,
                train_ratio=0.8,
                val_ratio=0.3,
                test_ratio=0.3,
                return_test=False,
            )

    def test_train_ratio_none_validation(self):
        """Test that val_ratio + test_ratio must be < 1 when train_ratio is None."""
        dataset = DummyDataset(size=100)

        with pytest.raises(AssertionError):
            get_train_val_test_loader(
                dataset,
                train_ratio=None,
                val_ratio=0.6,
                test_ratio=0.6,
                return_test=False,
            )

    def test_small_dataset(self):
        """Test with a very small dataset."""
        dataset = DummyDataset(size=10)
        train_loader, val_loader = get_train_val_test_loader(
            dataset, batch_size=2, val_ratio=0.1, test_ratio=0.1, return_test=False
        )

        assert len(train_loader.sampler) == 8
        assert len(val_loader.sampler) == 1

    def test_loader_iteration(self):
        """Test that loaders can be iterated over."""
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

    def test_zero_test_ratio(self):
        """Test with test_ratio=0 to cover edge case in val_sampler slicing."""
        dataset = DummyDataset(size=100)
        train_loader, val_loader = get_train_val_test_loader(
            dataset, batch_size=16, val_ratio=0.2, test_ratio=0.0, return_test=False
        )

        assert len(train_loader.sampler) == 80
        assert len(val_loader.sampler) == 20

    def test_train_ratio_none_with_warning(self, capsys):
        """Test that train_ratio=None prints warning message."""
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

    def test_exact_ratio_sum(self):
        """Test when train_ratio + val_ratio + test_ratio equals exactly 1."""
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

    def test_return_test_with_zero_test_size(self):
        """Test return_test=True with test_size=0."""
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

    def test_nonzero_test_with_explicit_sizes(self):
        """Test with small nonzero test_size to ensure val_sampler works correctly."""
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
    """Test suite for AtomInitializer class."""

    def test_initialization(self):
        """Test AtomInitializer initialization."""
        atom_types = [1, 6, 8]
        initializer = AtomInitializer(atom_types)
        assert initializer.atom_types == set(atom_types)
        assert initializer._embedding == {}

    def test_get_atom_fea_assertion(self):
        """Test that getting feature for unknown atom raises assertion."""
        initializer = AtomInitializer([1, 6, 8])
        with pytest.raises(AssertionError):
            initializer.get_atom_fea(7)

    def test_state_dict_operations(self):
        """Test state_dict and load_state_dict."""
        initializer = AtomInitializer([1, 6, 8])
        state = {1: 0, 6: 1, 8: 2}
        initializer.load_state_dict(state)

        assert initializer.state_dict() == state
        assert initializer.atom_types == {1, 6, 8}

    def test_decode(self):
        """Test decode method."""
        initializer = AtomInitializer([1, 6, 8])
        state = {1: 0, 6: 1, 8: 2}
        initializer.load_state_dict(state)

        assert initializer.decode(0) == 1
        assert initializer.decode(1) == 6
        assert initializer.decode(2) == 8

    def test_decode_builds_decode_dict(self):
        """Test that decode builds _decodedict on first call."""
        initializer = AtomInitializer([1, 6])
        initializer._embedding = {1: 0, 6: 1}
        initializer.atom_types = {1, 6}

        # First call should build _decodedict
        assert not hasattr(initializer, "_decodedict")
        result = initializer.decode(0)
        assert hasattr(initializer, "_decodedict")
        assert result == 1


class TestAtomCustomJSONInitializer:
    """Test suite for AtomCustomJSONInitializer class."""

    def test_initialization(self):
        """Test AtomCustomJSONInitializer with JSON file."""
        # Create temporary JSON file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"1": [0.1, 0.2, 0.3], "6": [0.4, 0.5, 0.6]}, f)
            temp_file = f.name

        try:
            initializer = AtomCustomJSONInitializer(temp_file)
            assert 1 in initializer.atom_types
            assert 6 in initializer.atom_types

            fea_1 = initializer.get_atom_fea(1)
            assert isinstance(fea_1, np.ndarray)
            assert np.allclose(fea_1, [0.1, 0.2, 0.3])
        finally:
            os.unlink(temp_file)


class TestGaussianDistance:
    """Test suite for GaussianDistance class."""

    def test_initialization(self):
        """Test GaussianDistance initialization."""
        gd = GaussianDistance(dmin=0, dmax=6, step=0.2)
        assert len(gd.filter) == 31  # (6-0)/0.2 + 1
        assert gd.var == 0.2

    def test_initialization_with_custom_var(self):
        """Test GaussianDistance with custom variance."""
        gd = GaussianDistance(dmin=0, dmax=6, step=0.2, var=0.5)
        assert gd.var == 0.5

    def test_initialization_assertions(self):
        """Test GaussianDistance initialization assertions."""
        with pytest.raises(AssertionError):
            GaussianDistance(dmin=6, dmax=0, step=0.2)

        with pytest.raises(AssertionError):
            GaussianDistance(dmin=0, dmax=0.1, step=0.2)

    def test_expand(self):
        """Test Gaussian distance expansion."""
        gd = GaussianDistance(dmin=0, dmax=4, step=1.0)
        distances = np.array([1.0, 2.0, 3.0])
        expanded = gd.expand(distances)

        assert expanded.shape == (3, 5)  # 3 distances, 5 filter points
        assert expanded.dtype == np.float64

    def test_expand_2d(self):
        """Test Gaussian expansion with 2D array."""
        gd = GaussianDistance(dmin=0, dmax=4, step=1.0)
        distances = np.array([[1.0, 2.0], [3.0, 4.0]])
        expanded = gd.expand(distances)

        assert expanded.shape == (2, 2, 5)


class TestCollatePool:
    """Test suite for collate_pool function."""

    def test_collate_pool(self):
        """Test collate_pool function."""
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

    def test_collate_pool_index_offset(self):
        """Test that collate_pool correctly offsets neighbor indices."""
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
