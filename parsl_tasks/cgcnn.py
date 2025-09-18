from __future__ import annotations

import os
import shutil
from parsl import bash_app

from parsl_configs.parsl_executors_labels import CGCNN_EXECUTOR_LABEL
from tools.config_labels import ConfigKeys as CK
import ml_models.cgcnn as cgcnn_pkg


def cmd_cgcnn_prediction(config, n_chunks, id):
    """
    Prepares the working environment and generates the command to execute
    CGCNN predictions.

    Args:
        config (dict): ConfigManager. The following fields are used:
            - work_dir
            - batch_size
            - num_workers

            See :class:`~tools.config_manager.ConfigManager` for full field descriptions.

    Returns:
        str: Absolute path to csv file containing the predictions.
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
