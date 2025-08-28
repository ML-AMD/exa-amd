from parsl import python_app, bash_app

from parsl_configs.parsl_executors_labels import CGCNN_EXECUTOR_LABEL
from tools.config_labels import ConfigKeys as CK


def cmd_cgcnn_prediction(config, n_chunks, id):
    """
    Construct the shell command used to run a CGCNN prediction via Parsl.

    Prepares the working environment and generates the command to execute
    the CGCNN `cms_dir/predict.py` script.

    Args:
        config (dict): ConfigManager. The following fields are used:
            - work_dir
            - cms_dir
            - batch_size
            - num_workers

            See :class:`~tools.config_manager.ConfigManager` for full field descriptions.

    Returns:
        str: A formatted shell command string to execute `predict.py`.

    Raises:
        Exception: If directory navigation or file copying fails.
    """
    import os
    import shutil
    try:
        os.chdir(config[CK.WORK_DIR])

        predict_script_path = os.path.join(config[CK.CMS_DIR], "predict.py")
        model_path = os.path.join(config[CK.CMS_DIR], "form_1st.pth.tar")

        dir_structures = os.path.join(
            config[CK.WORK_DIR], "structures", str(id))
        atom_init_json = os.path.join(config[CK.CMS_DIR], "atom_init.json")

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
