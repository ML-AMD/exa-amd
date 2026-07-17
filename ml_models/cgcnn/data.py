"""Data loading and graph-construction utilities for the CGCNN model.

This module provides dataset wrappers and helper functions to turn crystal
structures stored as CIF files into batched graph representations suitable
for the Crystal Graph Convolutional Neural Network (CGCNN). It includes:

- :func:`get_train_val_test_loader`: split a dataset into data loaders.
- :func:`collate_pool`: collate variable-sized crystal graphs into a batch.
- :class:`GaussianDistance`: expand distances onto a Gaussian basis.
- :class:`AtomInitializer` / :class:`AtomCustomJSONInitializer`: build atom
  feature vectors.
- :class:`CIFData`: a :class:`torch.utils.data.Dataset` backed by CIF files.
"""
from __future__ import division, print_function

import csv
import json
import os
import random
import warnings

import numpy as np
import torch
from pymatgen.core.structure import Structure
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


def get_train_val_test_loader(dataset, collate_fn=default_collate,
                              batch_size=64, train_ratio=None,
                              val_ratio=0.1, test_ratio=0.1, return_test=False,
                              num_workers=1, pin_memory=False, **kwargs):
    """Divide a dataset into train, validation, and test data loaders.

    !!! The dataset needs to be shuffled before using the function !!!

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        The full dataset to be divided.
    collate_fn : callable, optional
        Function to merge a list of samples into a batch. Defaults to
        ``default_collate``.
    batch_size : int, optional
        The batch size for the DataLoader. Defaults to 64.
    train_ratio : float, optional
        Fraction of the dataset used for training. If ``None``, it is inferred
        from ``val_ratio`` and ``test_ratio``. Defaults to ``None``.
    val_ratio : float, optional
        Fraction of the dataset used for validation. Defaults to 0.1.
    test_ratio : float, optional
        Fraction of the dataset used for testing. Defaults to 0.1.
    return_test : bool, optional
        Whether to return the test data loader. If ``False``, the last
        ``test_size`` samples are hidden. Defaults to ``False``.
    num_workers : int, optional
        Number of subprocesses used for data loading. Defaults to 1.
    pin_memory : bool, optional
        Whether to copy tensors into pinned memory. Defaults to ``False``.
    **kwargs
        Optional ``train_size``, ``val_size``, and ``test_size`` integer
        overrides for the computed split sizes.

    Returns
    -------
    tuple
        A ``(train_loader, val_loader)`` pair, or a
        ``(train_loader, val_loader, test_loader)`` triple when
        ``return_test`` is ``True``.
    """
    total_size = len(dataset)
    if train_ratio is None:
        assert val_ratio + test_ratio < 1
        train_ratio = 1 - val_ratio - test_ratio
        print('[Warning] train_ratio is None, using all training data.')
    else:
        assert train_ratio + val_ratio + test_ratio <= 1
    indices = list(range(total_size))
    train_size = kwargs.get('train_size', int(train_ratio * total_size))
    test_size = kwargs.get('test_size', int(test_ratio * total_size))
    valid_size = kwargs.get('val_size', int(val_ratio * total_size))
    train_sampler = SubsetRandomSampler(indices[:train_size])
    if test_size > 0:
        val_sampler = SubsetRandomSampler(
            indices[-(valid_size + test_size):-test_size])
    else:
        val_sampler = SubsetRandomSampler(
            indices[-valid_size:] if valid_size > 0 else [])
    if return_test:
        test_sampler = SubsetRandomSampler(indices[-test_size:]
                                           if test_size > 0 else [])
    train_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=num_workers,
                              collate_fn=collate_fn, pin_memory=pin_memory)
    val_loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=val_sampler,
                            num_workers=num_workers,
                            collate_fn=collate_fn, pin_memory=pin_memory)
    if return_test:
        test_loader = DataLoader(dataset, batch_size=batch_size,
                                 sampler=test_sampler,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn, pin_memory=pin_memory)
    if return_test:
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader


def collate_pool(dataset_list):
    """Collate a list of samples into a batch for crystal property prediction.

    N = sum(n_i); N0 = sum(i)

    Parameters
    ----------
    dataset_list : list of tuple
        A list of tuples, one per data point, structured as
        ``((atom_fea, nbr_fea, nbr_fea_idx), target, cif_id)`` where:

        atom_fea : torch.Tensor
            Shape ``(n_i, atom_fea_len)``.
        nbr_fea : torch.Tensor
            Shape ``(n_i, M, nbr_fea_len)``.
        nbr_fea_idx : torch.LongTensor
            Shape ``(n_i, M)``.
        target : torch.Tensor
            Shape ``(1,)``.
        cif_id : str or int
            Unique crystal identifier.

    Returns
    -------
    tuple
        A tuple of the form ``((batch_atom_fea, batch_nbr_fea,
        batch_nbr_fea_idx, crystal_atom_idx), target, batch_cif_ids)`` where:

        batch_atom_fea : torch.Tensor
            Shape ``(N, orig_atom_fea_len)``, atom features from atom type.
        batch_nbr_fea : torch.Tensor
            Shape ``(N, M, nbr_fea_len)``, bond features of each atom's M
            neighbors.
        batch_nbr_fea_idx : torch.LongTensor
            Shape ``(N, M)``, indices of the M neighbors of each atom.
        crystal_atom_idx : list of torch.LongTensor
            List of length N0 mapping crystal indices to atom indices.
        target : torch.Tensor
            Shape ``(N, 1)``, target values for prediction.
        batch_cif_ids : list
            List of crystal identifiers.
    """
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
    crystal_atom_idx, batch_target = [], []
    batch_cif_ids = []
    base_idx = 0
    for _, ((atom_fea, nbr_fea, nbr_fea_idx), target, cif_id)\
            in enumerate(dataset_list):
        n_i = atom_fea.shape[0]  # number of atoms for this crystal
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx + base_idx)
        new_idx = torch.LongTensor(np.arange(n_i) + base_idx)
        crystal_atom_idx.append(new_idx)
        batch_target.append(target)
        batch_cif_ids.append(cif_id)
        base_idx += n_i
    return (torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            crystal_atom_idx), \
        torch.stack(batch_target, dim=0), \
        batch_cif_ids


class GaussianDistance(object):
    """Expand interatomic distances using a Gaussian basis.

    Unit: angstrom.
    """

    def __init__(self, dmin, dmax, step, var=None):
        """Initialize the Gaussian distance filter.

        Parameters
        ----------
        dmin : float
            Minimum interatomic distance.
        dmax : float
            Maximum interatomic distance.
        step : float
            Step size for the Gaussian filter.
        var : float, optional
            Variance of the Gaussian. Defaults to ``step`` when ``None``.
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax + step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """Apply the Gaussian distance filter to a distance array.

        Parameters
        ----------
        distances : np.ndarray
            A distance matrix of any shape.

        Returns
        -------
        np.ndarray
            Expanded distance matrix with an added trailing dimension of
            length ``len(self.filter)``.
        """
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 /
                      self.var**2)


class AtomInitializer(object):
    """Base class for initializing the vector representation for atoms.

    !!! Use one AtomInitializer per dataset !!!
    """

    def __init__(self, atom_types):
        """Initialize the atom initializer.

        Parameters
        ----------
        atom_types : iterable
            Collection of valid atom type identifiers.
        """
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        """Return the feature vector for a given atom type.

        Parameters
        ----------
        atom_type
            The atom type identifier.

        Returns
        -------
        The feature vector associated with ``atom_type``.
        """
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        """Load atom embeddings from a state dictionary.

        Parameters
        ----------
        state_dict : dict
            Mapping from atom type to feature vector.
        """
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        """Return the current atom embedding state dictionary.

        Returns
        -------
        dict
            Mapping from atom type to feature vector.
        """
        return self._embedding

    def decode(self, idx):
        """Decode an embedding index back to its atom type.

        Parameters
        ----------
        idx
            The embedding index to decode.

        Returns
        -------
        The atom type corresponding to ``idx``.
        """
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):
    """Initialize atom feature vectors from a JSON file.

    The JSON file is a Python dictionary mapping from element number to a
    list representing the feature vector of the element.
    """ 

    def __init__(self, elem_embedding_file):
        """Initialize atom features from a JSON embedding file.

        Parameters
        ----------
        elem_embedding_file : str
            The path to the ``.json`` file.
        """
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


class CIFData(Dataset):
    """A dataset wrapper for crystal structures stored as CIF files.

    The dataset should have the following directory structure::

        root_dir
        ├── id_prop.csv
        ├── atom_init.json
        ├── id0.cif
        ├── id1.cif
        ├── ...

    - ``id_prop.csv``: A CSV file with two columns. The first column records a
      unique ID for each crystal, and the second column records the value of
      the target property.
    - ``atom_init.json``: A JSON file that stores the initialization vector
      for each element.
    - ``ID.cif``: A CIF file that records the crystal structure, where ``ID``
      is the unique ID for the crystal.
    """ 

    def __init__(self, root_dir, max_num_nbr=12, radius=8, dmin=0, step=0.2,
                 random_seed=123):
        """Initialize the CIFData dataset.

        Parameters
        ----------
        root_dir : str
            The path to the root directory of the dataset.
        max_num_nbr : int, optional
            The maximum number of neighbors used when constructing the crystal
            graph. Defaults to 12.
        radius : float, optional
            The cutoff radius for searching neighbors. Defaults to 8.
        dmin : float, optional
            The minimum distance for constructing ``GaussianDistance``.
            Defaults to 0.
        step : float, optional
            The step size for constructing ``GaussianDistance``. Defaults to
            0.2.
        random_seed : int, optional
            Random seed for shuffling the dataset. Defaults to 123.
        """
        self.root_dir = root_dir
        self.max_num_nbr, self.radius = max_num_nbr, radius
        assert os.path.exists(root_dir), 'root_dir does not exist!'
        id_prop_file = os.path.join(self.root_dir, 'id_prop.csv')
        assert os.path.exists(id_prop_file), 'id_prop.csv does not exist!'
        with open(id_prop_file) as f:
            reader = csv.reader(f)
            self.id_prop_data = [row for row in reader]
        random.seed(random_seed)
        random.shuffle(self.id_prop_data)
        atom_init_file = os.path.join(self.root_dir, 'atom_init.json')
        assert os.path.exists(atom_init_file), 'atom_init.json does not exist!'
        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)
        self._cache = {}
        self._cache_maxsize = 1024

    def __len__(self):
        """Return the number of crystals in the dataset.

        Returns
        -------
        int
            The number of entries in ``id_prop.csv``.
        """
        return len(self.id_prop_data)

    def __getitem__(self, idx):
        """Return the graph representation and target for a crystal.

        Parameters
        ----------
        idx : int
            Index of the crystal to retrieve.

        Returns
        -------
        tuple
            ``((atom_fea, nbr_fea, nbr_fea_idx), target, cif_id)`` where:

            atom_fea : torch.Tensor
                Shape ``(n_i, atom_fea_len)``.
            nbr_fea : torch.Tensor
                Shape ``(n_i, M, nbr_fea_len)``.
            nbr_fea_idx : torch.LongTensor
                Shape ``(n_i, M)``.
            target : torch.Tensor
                Shape ``(1,)``.
            cif_id : str or int
                The unique crystal identifier.
        """
        if idx in self._cache:
            return self._cache[idx]
        result = self._compute_item(idx)
        if len(self._cache) < self._cache_maxsize:
            self._cache[idx] = result
        return result

    def _compute_item(self, idx):
        """Compute the graph representation for a crystal (uncached)."""
        cif_id, target = self.id_prop_data[idx]
        crystal = Structure.from_file(os.path.join(self.root_dir,
                                                   cif_id + '.cif'))
        atom_fea = np.vstack([self.ari.get_atom_fea(crystal[i].specie.number)
                              for i in range(len(crystal))])
        atom_fea = torch.Tensor(atom_fea)
        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        nbr_fea_idx, nbr_fea = [], []
        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:
                warnings.warn(
                    f"{cif_id} not find enough neighbors to build graph. "
                    "If it happens frequently, consider increase radius.",
                    stacklevel=3,
                )
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                                   [0] * (self.max_num_nbr - len(nbr)))
                nbr_fea.append(list(map(lambda x: x[1], nbr)) +
                               [self.radius + 1.] * (self.max_num_nbr -
                                                     len(nbr)))
            else:
                nbr_fea_idx.append(list(map(lambda x: x[2],
                                            nbr[:self.max_num_nbr])))
                nbr_fea.append(list(map(lambda x: x[1],
                                        nbr[:self.max_num_nbr])))
        nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)
        nbr_fea = self.gdf.expand(nbr_fea)
        nbr_fea = torch.Tensor(nbr_fea)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        target = torch.Tensor([float(target)])
        return (atom_fea, nbr_fea, nbr_fea_idx), target, cif_id
