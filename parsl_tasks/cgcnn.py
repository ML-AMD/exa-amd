from __future__ import annotations

import os
import shutil
from parsl import bash_app

from parsl_configs.parsl_executors_labels import CGCNN_EXECUTOR_LABEL
from tools.config_labels import ConfigKeys as CK
import ml_models.cgcnn as cgcnn_pkg


def cmd_cgcnn_prediction(config, n_chunks, id):
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
    import shutil
    try:
        os.chdir(config[CK.WORK_DIR])

        pkg_dir = os.path.dirname(cgcnn_pkg.__file__)
        model_path = os.path.join(pkg_dir, "form_1st.pth.tar")
        atom_init_json = os.path.join(pkg_dir, "atom_init.json")
        predict_script_path = os.path.join(pkg_dir, "predict.py")

        dir_structures = os.path.join(config[CK.WORK_DIR], "structures", str(id))
        shutil.copy(atom_init_json, dir_structures)
    except Exception as e:
        raise
    num_workers = config[CK.NUM_WORKERS]
    return (
        f"srun -N 1 -n 1 --exclusive -c {num_workers} --gpus=1 "
        f"python {predict_script_path} {model_path} {dir_structures} "
        f"--batch-size {config[CK.BATCH_SIZE]} --workers {num_workers} --chunk_id {id}"
    )


@bash_app(executors=[CGCNN_EXECUTOR_LABEL])
def cgcnn_prediction(config, n_chunks, id):
    return cmd_cgcnn_prediction(config, n_chunks, id)
