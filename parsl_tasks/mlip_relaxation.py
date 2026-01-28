from __future__ import annotations
from parsl import bash_app

from parsl_configs.parsl_executors_labels import MLIP_RELAXATION_EXECUTOR_LABEL
from tools.config_labels import ConfigKeys as CK
import ml_models.mlip as mlip_pkg


def cmd_mlip_relaxation(config, file_paths):
    """
    Prepare the working environment and build the command to run CGCNN predictions.

    The prediction workload is partitioned into ``n_chunks`` disjoint segments.
    This task handles the segment identified by ``id``.

    :param dict config:
        A :class:`~tools.config_manager.ConfigManager` (or dict with the same
        fields). The following keys are read:

        - ``work_dir`` (str): root working directory for inputs/outputs
        - ``batch_size`` (int): inference batch size
        - ``num_workers`` (int): data-loading workers for inference

        See :class:`~tools.config_manager.ConfigManager` for full field descriptions.

    :param int n_chunks:
        Total number of chunks for the workload.

    :param int id:
        Zero-based index of the partition to execute, where ``0 <= id < n_chunks``.

    :returns: Absolute path to this partition’s predictions CSV.
    :rtype: str

    :raises ValueError: if ``n_chunks`` is not positive or ``id`` is out of range
    :raises Exception: on directory navigation or file I/O failures
    """
    import os
    import shlex

    os.chdir(config[CK.WORK_DIR])

    # for sanity
    if isinstance(file_paths, (str, os.PathLike)):
        file_paths = [file_paths]

    pkg_dir = os.path.dirname(mlip_pkg.__file__)
    model_path = os.path.join(pkg_dir, "uma-s-1p1.pt")
    mlip_relax_script = os.path.join(pkg_dir, "mlip_relax.py")

    energy_log_dir = os.path.join(config[CK.WORK_DIR], CK.MLIP_LOG_DIR)
    os.makedirs(energy_log_dir, exist_ok=True)
    files_argv = " ".join(shlex.quote(str(p)) for p in file_paths)

    return (f"python {mlip_relax_script} {model_path} {energy_log_dir} {files_argv}")


@bash_app(executors=[MLIP_RELAXATION_EXECUTOR_LABEL])
def mlip_relaxation(config, file_paths):
    return cmd_mlip_relaxation(config, file_paths)
