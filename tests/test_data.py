import pytest
import torch
from torch.utils.data import Dataset

from ml_models.cgcnn.data import get_train_val_test_loader


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
