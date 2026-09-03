from __future__ import annotations

import os
import shutil
from typing import Any, Mapping

from parsl import bash_app

import ml_models.cgcnn as cgcnn_pkg
from parsl_configs.parsl_executors_labels import CGCNN_EXECUTOR_LABEL
from tools.config_labels import ConfigKeys as CK


def cmd_cgcnn_prediction(config: Mapping[str, Any], n_chunks: int, id: int) -> str:
    """Prepare the working environment and build the CGCNN prediction command.

    The prediction workload is partitioned into ``n_chunks`` disjoint segments.
    This function handles the segment identified by ``id``.

    Parameters
    ----------
    config : collections.abc.Mapping
        A :class:`~tools.config_manager.ConfigManager` (or dict with the same
        fields). The following keys are read:

        - ``work_dir`` (str): root working directory for inputs/outputs.
        - ``batch_size`` (int): inference batch size.
        - ``num_workers`` (int): data-loading workers for inference.

        See :class:`~tools.config_manager.ConfigManager` for full field
        descriptions.
    n_chunks : int
        Total number of chunks for the workload.
    id : int
        Zero-based index of the partition to execute, where
        ``0 <= id < n_chunks``.

    Returns
    -------
    str
        The shell command that runs CGCNN prediction for this partition.

    Raises
    ------
    ValueError
        If ``n_chunks`` is not positive or ``id`` is out of range.
    Exception
        On directory navigation or file I/O failures.
    """
    if n_chunks <= 0:
        raise ValueError(f"n_chunks must be positive, got {n_chunks}")
    if not (0 <= id < n_chunks):
        raise ValueError(f"id must satisfy 0 <= id < {n_chunks}, got {id}")

    try:
        os.chdir(config[CK.WORK_DIR])

        pkg_dir = os.path.dirname(cgcnn_pkg.__file__)
        model_path = os.path.join(pkg_dir, "form_1st.pth.tar")
        atom_init_json = os.path.join(pkg_dir, "atom_init.json")
        predict_script_path = os.path.join(pkg_dir, "predict.py")

        dir_structures = os.path.join(config[CK.WORK_DIR], "structures", str(id))
        shutil.copy(atom_init_json, dir_structures)
        num_workers = config[CK.NUM_WORKERS]
    except Exception:
        raise
    return (
        f"python {predict_script_path} {model_path} {dir_structures} "
        f"--batch-size {config[CK.BATCH_SIZE]} --workers {num_workers} --chunk_id {id}"
    )


@bash_app(executors=[CGCNN_EXECUTOR_LABEL])
def cgcnn_prediction(config: Mapping[str, Any], n_chunks: int, id: int) -> str:
    """Parsl ``bash_app`` wrapper around :func:`cmd_cgcnn_prediction`.

    Parameters
    ----------
    config : collections.abc.Mapping
        Configuration mapping (see :func:`cmd_cgcnn_prediction`).
    n_chunks : int
        Total number of chunks for the workload.
    id : int
        Zero-based index of the partition to execute.

    Returns
    -------
    str
        The shell command executed by Parsl for this partition.
    """
    return cmd_cgcnn_prediction(config, n_chunks, id)
